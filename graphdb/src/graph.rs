//! Weighted graph store with a single-file `.graphdb` format.
//!
//! Nodes are external `u64` ids (e.g. note ids); edges are directed and carry a
//! `weight` (semantic similarity or a link constant) and a [`EdgeKind`]. The
//! adjacency is stored CSR-style and each node's out-edges are kept sorted by
//! descending weight, so `related` is just a prefix. Persistence mirrors the
//! `vectordb` `.vecdb` discipline: an 8-byte magic, a 64-byte header, 16-byte
//! aligned sections, little-endian, and byte-level `to_bytes` / `from_bytes`
//! (so it can be mmap'd or shipped over any IO adapter).

use std::collections::HashMap;
use std::io;
use std::path::Path;
use vectordb::Metric;

/// What produced an edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeKind {
    /// From embedding kNN (weight = similarity).
    Semantic,
    /// From an explicit link (e.g. an Obsidian `[[wikilink]]`).
    Link,
    /// Both a semantic neighbor and an explicit link.
    Both,
}

impl EdgeKind {
    pub fn as_u8(self) -> u8 {
        match self {
            EdgeKind::Semantic => 0,
            EdgeKind::Link => 1,
            EdgeKind::Both => 2,
        }
    }
    pub fn from_u8(x: u8) -> EdgeKind {
        match x {
            1 => EdgeKind::Link,
            2 => EdgeKind::Both,
            _ => EdgeKind::Semantic,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            EdgeKind::Semantic => "semantic",
            EdgeKind::Link => "link",
            EdgeKind::Both => "both",
        }
    }
}

/// One out-edge as seen by queries.
#[derive(Clone, Copy, Debug)]
pub struct Neighbor {
    pub id: u64,
    pub weight: f32,
    pub kind: EdgeKind,
}

/// A small extracted subgraph (for the local graph view).
#[derive(Clone, Debug, Default)]
pub struct Subgraph {
    pub nodes: Vec<u64>,
    pub edges: Vec<(u64, u64, f32, EdgeKind)>,
}

/// A weighted directed graph over `u64` node ids, CSR-backed.
pub struct GraphStore {
    metric: Metric,
    directed: bool,
    ids: Vec<u64>,       // node index -> external id
    offsets: Vec<usize>, // CSR, len n+1
    dst: Vec<u32>,       // neighbor node index
    weight: Vec<f32>,
    kind: Vec<u8>,
    idx: HashMap<u64, u32>, // external id -> node index
}

impl GraphStore {
    /// Assemble from a CSR triple. Each node's slice should already be sorted by
    /// descending weight. `ids[i]` is node `i`'s external id.
    pub(crate) fn from_csr(
        metric: Metric,
        directed: bool,
        ids: Vec<u64>,
        offsets: Vec<usize>,
        dst: Vec<u32>,
        weight: Vec<f32>,
        kind: Vec<u8>,
    ) -> GraphStore {
        let idx = ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i as u32))
            .collect();
        GraphStore {
            metric,
            directed,
            ids,
            offsets,
            dst,
            weight,
            kind,
            idx,
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
    pub fn edge_count(&self) -> usize {
        self.dst.len()
    }
    pub fn metric(&self) -> Metric {
        self.metric
    }
    pub fn is_directed(&self) -> bool {
        self.directed
    }
    /// External ids in node-index order.
    pub fn ids(&self) -> &[u64] {
        &self.ids
    }

    fn node_of(&self, id: u64) -> Option<u32> {
        self.idx.get(&id).copied()
    }

    fn range(&self, node: u32) -> std::ops::Range<usize> {
        let n = node as usize;
        self.offsets[n]..self.offsets[n + 1]
    }

    /// Out-degree of `id` (0 if unknown).
    pub fn degree(&self, id: u64) -> usize {
        self.node_of(id).map(|n| self.range(n).len()).unwrap_or(0)
    }

    /// All out-edges of `id`, sorted by descending weight.
    pub fn neighbors(&self, id: u64) -> Vec<Neighbor> {
        let Some(node) = self.node_of(id) else {
            return Vec::new();
        };
        self.range(node)
            .map(|e| Neighbor {
                id: self.ids[self.dst[e] as usize],
                weight: self.weight[e],
                kind: EdgeKind::from_u8(self.kind[e]),
            })
            .collect()
    }

    /// Top-`k` related nodes (highest-weight out-edges).
    pub fn related(&self, id: u64, k: usize) -> Vec<Neighbor> {
        let mut v = self.neighbors(id);
        v.truncate(k);
        v
    }

    /// Extract the neighborhood around `id`: a best-first (highest-weight)
    /// expansion up to `depth` hops, capped at `max_nodes` nodes. This is the
    /// data an Obsidian-style *local* graph view needs.
    pub fn neighborhood(&self, id: u64, depth: usize, max_nodes: usize) -> Subgraph {
        let mut sub = Subgraph::default();
        let Some(start) = self.node_of(id) else {
            return sub;
        };
        let mut seen: HashMap<u32, ()> = HashMap::new();
        seen.insert(start, ());
        sub.nodes.push(id);
        // frontier: (node, hops-remaining), expanded best-first within each ring.
        let mut frontier = vec![(start, depth)];
        while let Some((node, hops)) = frontier.pop() {
            if hops == 0 {
                continue;
            }
            // neighbors are pre-sorted by weight; expand strongest first.
            for e in self.range(node) {
                let dnode = self.dst[e];
                let did = self.ids[dnode as usize];
                // Record the edge if the destination is (or becomes) in-view.
                let newly = !seen.contains_key(&dnode);
                if newly && sub.nodes.len() >= max_nodes {
                    continue;
                }
                if newly {
                    seen.insert(dnode, ());
                    sub.nodes.push(did);
                    frontier.push((dnode, hops - 1));
                }
                sub.edges.push((
                    self.ids[node as usize],
                    did,
                    self.weight[e],
                    EdgeKind::from_u8(self.kind[e]),
                ));
            }
        }
        sub
    }

    /// Export the whole graph as JSON (`{nodes:[...], edges:[...]}`) for a graph
    /// view. `labels` optionally names nodes; `communities` (per node index)
    /// optionally colors them; node `size` is the out-degree.
    pub fn export_json(
        &self,
        labels: Option<&HashMap<u64, String>>,
        communities: Option<&[u32]>,
    ) -> String {
        let mut s = String::from("{\"nodes\":[");
        for (i, &id) in self.ids.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            let deg = self.range(i as u32).len();
            s.push_str(&format!("{{\"id\":{id},\"degree\":{deg}"));
            if let Some(m) = labels {
                if let Some(name) = m.get(&id) {
                    s.push_str(&format!(",\"label\":{}", json_str(name)));
                }
            }
            if let Some(c) = communities {
                s.push_str(&format!(",\"community\":{}", c[i]));
            }
            s.push('}');
        }
        s.push_str("],\"edges\":[");
        let mut first = true;
        for (i, &src) in self.ids.iter().enumerate() {
            for e in self.range(i as u32) {
                if !first {
                    s.push(',');
                }
                first = false;
                let dst = self.ids[self.dst[e] as usize];
                s.push_str(&format!(
                    "{{\"source\":{src},\"target\":{dst},\"weight\":{:.4},\"kind\":{}}}",
                    self.weight[e],
                    json_str(EdgeKind::from_u8(self.kind[e]).label()),
                ));
            }
        }
        s.push_str("]}");
        s
    }

    // --- persistence -------------------------------------------------------

    pub fn to_bytes(&self) -> Vec<u8> {
        let n = self.ids.len();
        let edges = self.dst.len();
        let ids_off = align16(HEADER_LEN);
        let off_off = align16(ids_off + n * 8);
        let dst_off = align16(off_off + (n + 1) * 8);
        let w_off = align16(dst_off + edges * 4);
        let kind_off = align16(w_off + edges * 4);
        let total = align16(kind_off + edges);

        let mut b = vec![0u8; total];
        b[0..8].copy_from_slice(MAGIC);
        b[8..12].copy_from_slice(&VERSION.to_le_bytes());
        b[12..16].copy_from_slice(&(self.metric as u32).to_le_bytes());
        b[16..20].copy_from_slice(&(n as u32).to_le_bytes());
        b[20..24].copy_from_slice(&(edges as u32).to_le_bytes());
        b[24] = self.directed as u8;

        for (i, &id) in self.ids.iter().enumerate() {
            put_u64(&mut b, ids_off + i * 8, id);
        }
        for (i, &o) in self.offsets.iter().enumerate() {
            put_u64(&mut b, off_off + i * 8, o as u64);
        }
        for (i, &d) in self.dst.iter().enumerate() {
            put_u32(&mut b, dst_off + i * 4, d);
        }
        for (i, &w) in self.weight.iter().enumerate() {
            b[w_off + i * 4..w_off + i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        b[kind_off..kind_off + edges].copy_from_slice(&self.kind);
        b
    }

    pub fn from_bytes(b: &[u8]) -> io::Result<GraphStore> {
        let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, format!("graphdb: {m}"));
        if b.len() < HEADER_LEN || &b[0..8] != MAGIC {
            return Err(bad("bad magic"));
        }
        if read_u32(b, 8) != VERSION {
            return Err(bad("unsupported version"));
        }
        let metric = Metric::from_u32(read_u32(b, 12)).ok_or_else(|| bad("bad metric"))?;
        let n = read_u32(b, 16) as usize;
        let edges = read_u32(b, 20) as usize;
        let directed = b[24] != 0;

        let ids_off = align16(HEADER_LEN);
        let off_off = align16(ids_off + n * 8);
        let dst_off = align16(off_off + (n + 1) * 8);
        let w_off = align16(dst_off + edges * 4);
        let kind_off = align16(w_off + edges * 4);
        let total = align16(kind_off + edges);
        if b.len() < total {
            return Err(bad("file truncated"));
        }

        let ids: Vec<u64> = (0..n).map(|i| read_u64(b, ids_off + i * 8)).collect();
        let offsets: Vec<usize> = (0..=n)
            .map(|i| read_u64(b, off_off + i * 8) as usize)
            .collect();
        if offsets[n] != edges {
            return Err(bad("offset/edge mismatch"));
        }
        let dst: Vec<u32> = (0..edges).map(|i| read_u32(b, dst_off + i * 4)).collect();
        let weight: Vec<f32> = (0..edges).map(|i| read_f32(b, w_off + i * 4)).collect();
        let kind: Vec<u8> = b[kind_off..kind_off + edges].to_vec();
        Ok(GraphStore::from_csr(
            metric, directed, ids, offsets, dst, weight, kind,
        ))
    }

    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        std::fs::write(path, self.to_bytes())
    }
    pub fn load(path: impl AsRef<Path>) -> io::Result<GraphStore> {
        GraphStore::from_bytes(&std::fs::read(path)?)
    }
}

const MAGIC: &[u8; 8] = b"GRAPHDB1";
const VERSION: u32 = 1;
const HEADER_LEN: usize = 64;

#[inline]
fn align16(x: usize) -> usize {
    (x + 15) & !15
}
#[inline]
fn put_u32(b: &mut [u8], o: usize, v: u32) {
    b[o..o + 4].copy_from_slice(&v.to_le_bytes());
}
#[inline]
fn put_u64(b: &mut [u8], o: usize, v: u64) {
    b[o..o + 8].copy_from_slice(&v.to_le_bytes());
}
#[inline]
fn read_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
#[inline]
fn read_u64(b: &[u8], o: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(a)
}
#[inline]
fn read_f32(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

/// Minimal JSON string escaping.
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

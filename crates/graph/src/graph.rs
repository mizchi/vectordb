//! Weighted graph store with a single-file `.graphdb` format.
//!
//! Nodes are external `u64` ids (e.g. note ids); edges are directed and carry a
//! `weight` (semantic similarity or a link constant) and a [`EdgeKind`]. Nodes
//! also carry optional metadata: a `title` and a set of `tags` (interned). The
//! adjacency is stored CSR-style and each node's out-edges are kept sorted by
//! descending weight, so `related` is just a prefix. Persistence mirrors the
//! vector-layer `.vecdb` discipline: an 8-byte magic, a 64-byte header, 16-byte
//! aligned sections, little-endian, and byte-level `to_bytes` / `from_bytes`
//! (so it can be mmap'd or shipped over any IO adapter).

use meandb_vector::Metric;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io;
use std::path::Path;

/// What produced an edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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

/// Per-node metadata attached to the graph.
#[derive(Clone, Debug, Default)]
pub struct NodeMeta {
    pub title: String,
    pub tags: Vec<String>,
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

/// A weighted directed graph over `u64` node ids, CSR-backed, with optional
/// per-node metadata (title + interned tags).
pub struct GraphStore {
    metric: Metric,
    directed: bool,
    ids: Vec<u64>,       // node index -> external id
    offsets: Vec<usize>, // CSR, len n+1
    dst: Vec<u32>,       // neighbor node index
    weight: Vec<f32>,
    kind: Vec<u8>,
    idx: HashMap<u64, u32>, // external id -> node index
    // metadata (len n; empty title / no tags allowed)
    titles: Vec<String>,
    node_tags: Vec<Vec<u32>>,      // per node, interned tag ids
    tag_names: Vec<String>,        // tag id -> name
    tag_ids: HashMap<String, u32>, // name -> tag id
    tag_members: Vec<Vec<u32>>,    // tag id -> node indices (built with tags)
}

impl GraphStore {
    /// Assemble from a CSR triple (metadata empty). Each node's slice should be
    /// sorted by descending weight; `ids[i]` is node `i`'s external id.
    pub(crate) fn from_csr(
        metric: Metric,
        directed: bool,
        ids: Vec<u64>,
        offsets: Vec<usize>,
        dst: Vec<u32>,
        weight: Vec<f32>,
        kind: Vec<u8>,
    ) -> GraphStore {
        let n = ids.len();
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
            titles: vec![String::new(); n],
            node_tags: vec![Vec::new(); n],
            tag_names: Vec::new(),
            tag_ids: HashMap::new(),
            tag_members: Vec::new(),
        }
    }

    /// Attach per-node metadata (title + tags), keyed by external id. Tags are
    /// interned; unknown ids are ignored. Replaces any existing metadata.
    pub fn set_metadata(&mut self, meta: impl IntoIterator<Item = (u64, NodeMeta)>) {
        let n = self.ids.len();
        self.titles = vec![String::new(); n];
        self.node_tags = vec![Vec::new(); n];
        self.tag_names.clear();
        self.tag_ids.clear();
        for (id, m) in meta {
            let Some(&node) = self.idx.get(&id) else {
                continue;
            };
            let node = node as usize;
            self.titles[node] = m.title;
            let mut tids: Vec<u32> = m.tags.iter().map(|t| self.intern_tag(t)).collect();
            tids.sort_unstable();
            tids.dedup();
            self.node_tags[node] = tids;
        }
        self.rebuild_tag_members();
    }

    fn intern_tag(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.tag_ids.get(name) {
            return id;
        }
        let id = self.tag_names.len() as u32;
        self.tag_names.push(name.to_string());
        self.tag_ids.insert(name.to_string(), id);
        id
    }

    fn rebuild_tag_members(&mut self) {
        let mut members = vec![Vec::new(); self.tag_names.len()];
        for (node, tags) in self.node_tags.iter().enumerate() {
            for &t in tags {
                members[t as usize].push(node as u32);
            }
        }
        self.tag_members = members;
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
    /// Distinct tag names in the graph.
    pub fn tag_names(&self) -> &[String] {
        &self.tag_names
    }

    fn node_of(&self, id: u64) -> Option<u32> {
        self.idx.get(&id).copied()
    }

    fn range(&self, node: u32) -> std::ops::Range<usize> {
        let n = node as usize;
        self.offsets[n]..self.offsets[n + 1]
    }

    /// Title of `id` (None if unset/empty or unknown).
    pub fn title(&self, id: u64) -> Option<&str> {
        let n = self.node_of(id)? as usize;
        let t = self.titles[n].as_str();
        (!t.is_empty()).then_some(t)
    }

    /// Tags of `id` (empty if none/unknown).
    pub fn tags(&self, id: u64) -> Vec<&str> {
        let Some(n) = self.node_of(id) else {
            return Vec::new();
        };
        self.node_tags[n as usize]
            .iter()
            .map(|&t| self.tag_names[t as usize].as_str())
            .collect()
    }

    /// True if `id` carries `tag`.
    pub fn has_tag(&self, id: u64, tag: &str) -> bool {
        let (Some(n), Some(&t)) = (self.node_of(id), self.tag_ids.get(tag)) else {
            return false;
        };
        self.node_tags[n as usize].contains(&t)
    }

    /// External ids of every node carrying `tag`.
    pub fn nodes_with_tag(&self, tag: &str) -> Vec<u64> {
        match self.tag_ids.get(tag) {
            Some(&t) => self.tag_members[t as usize]
                .iter()
                .map(|&n| self.ids[n as usize])
                .collect(),
            None => Vec::new(),
        }
    }

    /// (tag, count) for every tag, most frequent first.
    pub fn tag_counts(&self) -> Vec<(&str, usize)> {
        let mut v: Vec<(&str, usize)> = self
            .tag_names
            .iter()
            .enumerate()
            .map(|(t, name)| (name.as_str(), self.tag_members[t].len()))
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        v
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

    /// Top-`k` related nodes whose id satisfies `keep` (e.g. a tag filter).
    pub fn related_filter(&self, id: u64, k: usize, keep: impl Fn(u64) -> bool) -> Vec<Neighbor> {
        self.neighbors(id)
            .into_iter()
            .filter(|n| keep(n.id))
            .take(k)
            .collect()
    }

    /// Extract the neighborhood around `id`: a best-first (highest-weight)
    /// expansion up to `depth` hops, capped at `max_nodes` nodes, keeping only
    /// nodes for which `keep` holds (use `|_| true` for no filter). This is the
    /// data an Obsidian-style *local* graph view needs.
    pub fn neighborhood(
        &self,
        id: u64,
        depth: usize,
        max_nodes: usize,
        keep: impl Fn(u64) -> bool,
    ) -> Subgraph {
        let mut sub = Subgraph::default();
        let Some(start) = self.node_of(id) else {
            return sub;
        };
        if !keep(id) {
            return sub;
        }
        let mut seen: HashMap<u32, ()> = HashMap::new();
        seen.insert(start, ());
        sub.nodes.push(id);
        let mut frontier = vec![(start, depth)];
        while let Some((node, hops)) = frontier.pop() {
            if hops == 0 {
                continue;
            }
            for e in self.range(node) {
                let dnode = self.dst[e];
                let did = self.ids[dnode as usize];
                if !keep(did) {
                    continue;
                }
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
    /// view. Each node carries `degree` (size), any `label`/`tags`, and — if
    /// `communities` (per node index) is given — a `community` (color).
    pub fn export_json(&self, communities: Option<&[u32]>) -> String {
        let mut s = String::from("{\"nodes\":[");
        for (i, &id) in self.ids.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            let deg = self.range(i as u32).len();
            s.push_str(&format!("{{\"id\":{id},\"degree\":{deg}"));
            if !self.titles[i].is_empty() {
                s.push_str(&format!(",\"label\":{}", json_str(&self.titles[i])));
            }
            if !self.node_tags[i].is_empty() {
                s.push_str(",\"tags\":[");
                for (j, &t) in self.node_tags[i].iter().enumerate() {
                    if j > 0 {
                        s.push(',');
                    }
                    s.push_str(&json_str(&self.tag_names[t as usize]));
                }
                s.push(']');
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
        let tag_count = self.tag_names.len();

        // Flatten metadata.
        let mut label_blob = Vec::new();
        let mut label_off = Vec::with_capacity(n + 1);
        label_off.push(0u64);
        for t in &self.titles {
            label_blob.extend_from_slice(t.as_bytes());
            label_off.push(label_blob.len() as u64);
        }
        let mut tag_ids: Vec<u32> = Vec::new();
        let mut tag_off = Vec::with_capacity(n + 1);
        tag_off.push(0u64);
        for tags in &self.node_tags {
            tag_ids.extend_from_slice(tags);
            tag_off.push(tag_ids.len() as u64);
        }
        let mut tn_blob = Vec::new();
        let mut tn_off = Vec::with_capacity(tag_count + 1);
        tn_off.push(0u64);
        for name in &self.tag_names {
            tn_blob.extend_from_slice(name.as_bytes());
            tn_off.push(tn_blob.len() as u64);
        }

        // Section offsets.
        let mut pos = HEADER_LEN;
        let ids_off = sec(&mut pos, n * 8);
        let off_off = sec(&mut pos, (n + 1) * 8);
        let dst_off = sec(&mut pos, edges * 4);
        let w_off = sec(&mut pos, edges * 4);
        let kind_off = sec(&mut pos, edges);
        let loff_off = sec(&mut pos, (n + 1) * 8);
        let lblob_off = sec(&mut pos, label_blob.len());
        let toff_off = sec(&mut pos, (n + 1) * 8);
        let tids_off = sec(&mut pos, tag_ids.len() * 4);
        let tnoff_off = sec(&mut pos, (tag_count + 1) * 8);
        let tnblob_off = sec(&mut pos, tn_blob.len());
        let total = align16(pos);

        let mut b = vec![0u8; total];
        b[0..8].copy_from_slice(MAGIC);
        b[8..12].copy_from_slice(&VERSION.to_le_bytes());
        b[12..16].copy_from_slice(&(self.metric as u32).to_le_bytes());
        b[16..20].copy_from_slice(&(n as u32).to_le_bytes());
        b[20..24].copy_from_slice(&(edges as u32).to_le_bytes());
        b[24] = self.directed as u8;
        b[28..32].copy_from_slice(&(tag_count as u32).to_le_bytes());

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
        for (i, &o) in label_off.iter().enumerate() {
            put_u64(&mut b, loff_off + i * 8, o);
        }
        b[lblob_off..lblob_off + label_blob.len()].copy_from_slice(&label_blob);
        for (i, &o) in tag_off.iter().enumerate() {
            put_u64(&mut b, toff_off + i * 8, o);
        }
        for (i, &t) in tag_ids.iter().enumerate() {
            put_u32(&mut b, tids_off + i * 4, t);
        }
        for (i, &o) in tn_off.iter().enumerate() {
            put_u64(&mut b, tnoff_off + i * 8, o);
        }
        b[tnblob_off..tnblob_off + tn_blob.len()].copy_from_slice(&tn_blob);
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
        let tag_count = read_u32(b, 28) as usize;

        // Walk sections (reading each offset array to size the following blob).
        let mut pos = HEADER_LEN;
        let ids_off = adv(&mut pos, n * 8);
        let off_off = adv(&mut pos, (n + 1) * 8);
        let dst_off = adv(&mut pos, edges * 4);
        let w_off = adv(&mut pos, edges * 4);
        let kind_off = adv(&mut pos, edges);
        let loff_off = adv(&mut pos, (n + 1) * 8);
        if b.len() < loff_off + (n + 1) * 8 {
            return Err(bad("truncated (labels)"));
        }
        let label_blob_len = read_u64(b, loff_off + n * 8) as usize;
        let lblob_off = adv(&mut pos, label_blob_len);
        let toff_off = adv(&mut pos, (n + 1) * 8);
        if b.len() < toff_off + (n + 1) * 8 {
            return Err(bad("truncated (tags)"));
        }
        let total_tag_ids = read_u64(b, toff_off + n * 8) as usize;
        let tids_off = adv(&mut pos, total_tag_ids * 4);
        let tnoff_off = adv(&mut pos, (tag_count + 1) * 8);
        if b.len() < tnoff_off + (tag_count + 1) * 8 {
            return Err(bad("truncated (tag names)"));
        }
        let tn_blob_len = read_u64(b, tnoff_off + tag_count * 8) as usize;
        let tnblob_off = adv(&mut pos, tn_blob_len);
        if b.len() < align16(pos) {
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

        let mut g = GraphStore::from_csr(metric, directed, ids, offsets, dst, weight, kind);

        // Tag names.
        g.tag_names = (0..tag_count)
            .map(|t| {
                let s = read_u64(b, tnoff_off + t * 8) as usize;
                let e = read_u64(b, tnoff_off + (t + 1) * 8) as usize;
                String::from_utf8_lossy(&b[tnblob_off + s..tnblob_off + e]).into_owned()
            })
            .collect();
        g.tag_ids = g
            .tag_names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i as u32))
            .collect();
        // Titles + node tags.
        g.titles = (0..n)
            .map(|i| {
                let s = read_u64(b, loff_off + i * 8) as usize;
                let e = read_u64(b, loff_off + (i + 1) * 8) as usize;
                String::from_utf8_lossy(&b[lblob_off + s..lblob_off + e]).into_owned()
            })
            .collect();
        g.node_tags = (0..n)
            .map(|i| {
                let s = read_u64(b, toff_off + i * 8) as usize;
                let e = read_u64(b, toff_off + (i + 1) * 8) as usize;
                (s..e).map(|j| read_u32(b, tids_off + j * 4)).collect()
            })
            .collect();
        g.rebuild_tag_members();
        Ok(g)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        std::fs::write(path, self.to_bytes())
    }
    pub fn load(path: impl AsRef<Path>) -> io::Result<GraphStore> {
        GraphStore::from_bytes(&std::fs::read(path)?)
    }
}

const MAGIC: &[u8; 8] = b"GRAPHDB1";
const VERSION: u32 = 2;
const HEADER_LEN: usize = 64;

#[inline]
fn align16(x: usize) -> usize {
    (x + 15) & !15
}
/// Reserve a 16-aligned section of `len` bytes, returning its start.
#[inline]
fn sec(pos: &mut usize, len: usize) -> usize {
    let start = align16(*pos);
    *pos = start + len;
    start
}
/// Same as [`sec`] but for the read path (identical arithmetic).
#[inline]
fn adv(pos: &mut usize, len: usize) -> usize {
    sec(pos, len)
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

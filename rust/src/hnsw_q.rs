//! Memory-compact HNSW that stores int8-quantized vectors instead of f32.
//!
//! Same multi-layer proximity graph as [`crate::hnsw::HnswIndex`], but each node
//! keeps its vector as `dim` int8 codes plus a per-vector `scale` and `sqnorm`
//! (≈`dim + 8` bytes vs `dim * 4` for f32 — roughly a 4x reduction of the vector
//! payload; the graph edges are unchanged). All graph distances use the int8
//! approximate dot product, so the graph is both built and searched in the
//! quantized space. Keeping the f32 originals (`keep_raw`) additionally enables
//! an exact rerank of the final beam, recovering most of the lost recall while
//! still traversing the compact graph.

use crate::distance::{dot_f32, dot_i8, l2sq_f32};
use crate::index::{Hit, Metric};
use crate::quantize::quantize;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::io;
use std::path::Path;

/// A quantized query (int8 codes + the scalars needed to reconstruct a
/// metric distance), passed to the query-side search helpers.
struct QQuery<'a> {
    codes: &'a [i8],
    scale: f32,
    sqnorm: f32,
}

/// Distance/node pair ordered by distance (smaller = closer).
#[derive(Clone, Copy, PartialEq)]
struct DN {
    dist: f32,
    node: u32,
}
impl Eq for DN {}
impl PartialOrd for DN {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for DN {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist.total_cmp(&other.dist)
    }
}

/// int8-quantized HNSW index.
pub struct HnswQIndex {
    dim: usize,
    metric: Metric,
    m: usize,
    m0: usize,
    ef_construction: usize,
    ml: f64,
    codes: Vec<i8>,    // count * dim
    scales: Vec<f32>,  // count
    sqnorms: Vec<f32>, // count
    raw: Option<Vec<f32>>,
    ids: Vec<u64>,
    links: Vec<Vec<Vec<u32>>>,
    levels: Vec<usize>,
    entry: Option<u32>,
    max_level: usize,
    rng: u64,
}

impl HnswQIndex {
    /// Create an empty index. `keep_raw` retains f32 originals to enable an
    /// exact rerank of the final beam (at the cost of the memory it saves).
    pub fn new(
        dim: usize,
        metric: Metric,
        m: usize,
        ef_construction: usize,
        keep_raw: bool,
    ) -> Self {
        assert!(dim > 0 && m > 0);
        HnswQIndex {
            dim,
            metric,
            m,
            m0: m * 2,
            ef_construction: ef_construction.max(m),
            ml: 1.0 / (m as f64).ln(),
            codes: Vec::new(),
            scales: Vec::new(),
            sqnorms: Vec::new(),
            raw: if keep_raw { Some(Vec::new()) } else { None },
            ids: Vec::new(),
            links: Vec::new(),
            levels: Vec::new(),
            entry: None,
            max_level: 0,
            rng: 0x9E37_79B9_7F4A_7C15,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn metric(&self) -> Metric {
        self.metric
    }
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
    pub fn has_raw(&self) -> bool {
        self.raw.is_some()
    }

    fn next_rand(&mut self) -> f64 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        (x >> 11) as f64 / (1u64 << 53) as f64
    }

    fn random_level(&mut self) -> usize {
        let r = self.next_rand().max(1e-12);
        (-r.ln() * self.ml) as usize
    }

    #[inline]
    fn codes_at(&self, i: u32) -> &[i8] {
        let i = i as usize;
        &self.codes[i * self.dim..(i + 1) * self.dim]
    }
    #[inline]
    fn raw_at(&self, i: u32) -> &[f32] {
        let raw = self.raw.as_ref().expect("raw not kept");
        let i = i as usize;
        &raw[i * self.dim..(i + 1) * self.dim]
    }

    /// Approximate distance between two stored nodes (smaller = closer).
    #[inline]
    fn dn(&self, a: u32, b: u32) -> f32 {
        let approx = dot_i8(self.codes_at(a), self.codes_at(b)) as f32
            * self.scales[a as usize]
            * self.scales[b as usize];
        match self.metric {
            Metric::L2 => self.sqnorms[a as usize] + self.sqnorms[b as usize] - 2.0 * approx,
            Metric::Dot | Metric::Cosine => -approx,
        }
    }

    /// Approximate distance from a quantized query to a stored node.
    #[inline]
    fn dq(&self, q: &QQuery, node: u32) -> f32 {
        let approx =
            dot_i8(q.codes, self.codes_at(node)) as f32 * q.scale * self.scales[node as usize];
        match self.metric {
            Metric::L2 => q.sqnorm + self.sqnorms[node as usize] - 2.0 * approx,
            Metric::Dot | Metric::Cosine => -approx,
        }
    }

    /// Add a vector with an external id.
    pub fn add(&mut self, id: u64, vector: &[f32]) {
        assert_eq!(vector.len(), self.dim, "dimension mismatch");
        let processed = if self.metric == Metric::Cosine {
            normalize(vector)
        } else {
            vector.to_vec()
        };
        let q = quantize(&processed);
        let node = self.ids.len() as u32;
        let level = self.random_level();
        self.codes.extend_from_slice(&q.codes);
        self.scales.push(q.scale);
        self.sqnorms.push(q.sqnorm);
        if let Some(raw) = self.raw.as_mut() {
            raw.extend_from_slice(&processed);
        }
        self.ids.push(id);
        self.levels.push(level);
        self.links.push((0..=level).map(|_| Vec::new()).collect());

        let Some(mut ep) = self.entry else {
            self.entry = Some(node);
            self.max_level = level;
            return;
        };

        // Descend greedily from the top down to level+1 (node-to-node distances,
        // since the inserted vector is now stored as `node`).
        let mut cur = self.dn(node, ep);
        let top = self.max_level;
        for lc in ((level + 1)..=top).rev() {
            (ep, cur) = self.greedy1_node(node, ep, cur, lc);
        }

        let start = level.min(top);
        for lc in (0..=start).rev() {
            let mut w = self.search_layer_node(node, ep, self.ef_construction, lc);
            if let Some(best) = w.iter().min_by(|a, b| a.dist.total_cmp(&b.dist)) {
                ep = best.node;
            }
            let m = if lc == 0 { self.m0 } else { self.m };
            let selected = self.select_neighbors(&mut w, m);
            for &nb in &selected {
                self.links[node as usize][lc].push(nb);
                self.links[nb as usize][lc].push(node);
                let budget = if lc == 0 { self.m0 } else { self.m };
                if self.links[nb as usize][lc].len() > budget {
                    self.prune(nb, lc, budget);
                }
            }
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry = Some(node);
        }
    }

    /// Greedy single-best descent within one layer, distance from a query node.
    fn greedy1_node(&self, q: u32, mut ep: u32, mut ep_dist: f32, lc: usize) -> (u32, f32) {
        let mut improved = true;
        while improved {
            improved = false;
            for &nb in &self.links[ep as usize][lc] {
                let d = self.dn(q, nb);
                if d < ep_dist {
                    ep_dist = d;
                    ep = nb;
                    improved = true;
                }
            }
        }
        (ep, ep_dist)
    }

    /// Beam search within one layer, distances from a stored query node.
    fn search_layer_node(&self, q: u32, ep: u32, ef: usize, lc: usize) -> Vec<DN> {
        let mut visited = vec![false; self.ids.len()];
        let mut cands: BinaryHeap<Reverse<DN>> = BinaryHeap::new();
        let mut w: BinaryHeap<DN> = BinaryHeap::new();
        let d0 = self.dn(q, ep);
        visited[ep as usize] = true;
        cands.push(Reverse(DN { dist: d0, node: ep }));
        w.push(DN { dist: d0, node: ep });
        while let Some(Reverse(c)) = cands.pop() {
            let farthest = w.peek().map(|x| x.dist).unwrap_or(f32::INFINITY);
            if c.dist > farthest && w.len() >= ef {
                break;
            }
            for &nb in &self.links[c.node as usize][lc] {
                if visited[nb as usize] {
                    continue;
                }
                visited[nb as usize] = true;
                let d = self.dn(q, nb);
                let farthest = w.peek().map(|x| x.dist).unwrap_or(f32::INFINITY);
                if d < farthest || w.len() < ef {
                    cands.push(Reverse(DN { dist: d, node: nb }));
                    w.push(DN { dist: d, node: nb });
                    if w.len() > ef {
                        w.pop();
                    }
                }
            }
        }
        w.into_vec()
    }

    /// Beam search within one layer, distances from an external quantized query.
    /// `filter(id)` gates admission into the result beam while still traversing.
    fn search_layer_query<F: Fn(u64) -> bool>(
        &self,
        q: &QQuery,
        ep: u32,
        ef: usize,
        lc: usize,
        filter: &F,
    ) -> Vec<DN> {
        let mut visited = vec![false; self.ids.len()];
        let mut cands: BinaryHeap<Reverse<DN>> = BinaryHeap::new();
        let mut w: BinaryHeap<DN> = BinaryHeap::new();
        let d0 = self.dq(q, ep);
        visited[ep as usize] = true;
        cands.push(Reverse(DN { dist: d0, node: ep }));
        if filter(self.ids[ep as usize]) {
            w.push(DN { dist: d0, node: ep });
        }
        while let Some(Reverse(c)) = cands.pop() {
            let farthest = w.peek().map(|x| x.dist).unwrap_or(f32::INFINITY);
            if c.dist > farthest && w.len() >= ef {
                break;
            }
            for &nb in &self.links[c.node as usize][lc] {
                if visited[nb as usize] {
                    continue;
                }
                visited[nb as usize] = true;
                let d = self.dq(q, nb);
                let farthest = w.peek().map(|x| x.dist).unwrap_or(f32::INFINITY);
                if d < farthest || w.len() < ef {
                    cands.push(Reverse(DN { dist: d, node: nb }));
                    if filter(self.ids[nb as usize]) {
                        w.push(DN { dist: d, node: nb });
                        if w.len() > ef {
                            w.pop();
                        }
                    }
                }
            }
        }
        w.into_vec()
    }

    /// Diversity neighbor-selection heuristic (node-to-node distances).
    fn select_neighbors(&self, w: &mut [DN], m: usize) -> Vec<u32> {
        w.sort_by(|a, b| a.dist.total_cmp(&b.dist));
        let mut result: Vec<u32> = Vec::with_capacity(m);
        for cand in w.iter() {
            if result.len() >= m {
                break;
            }
            let mut keep = true;
            for &r in &result {
                if self.dn(cand.node, r) < cand.dist {
                    keep = false;
                    break;
                }
            }
            if keep {
                result.push(cand.node);
            }
        }
        result
    }

    fn prune(&mut self, node: u32, lc: usize, budget: usize) {
        let mut w: Vec<DN> = self.links[node as usize][lc]
            .iter()
            .map(|&nb| DN {
                dist: self.dn(node, nb),
                node: nb,
            })
            .collect();
        let kept = self.select_neighbors(&mut w, budget);
        self.links[node as usize][lc] = kept;
    }

    /// Search for the `k` nearest neighbors with beam width `ef_search`. When
    /// originals are kept, the beam is reranked with exact f32 distances.
    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<Hit> {
        self.search_filter(query, k, ef_search, |_| true)
    }

    /// Filtered search: only ids satisfying `filter` become results (the graph
    /// is still traversed through non-passing nodes).
    pub fn search_filter<F: Fn(u64) -> bool>(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        filter: F,
    ) -> Vec<Hit> {
        if k == 0 || self.is_empty() {
            return Vec::new();
        }
        assert_eq!(query.len(), self.dim, "dimension mismatch");
        let processed = if self.metric == Metric::Cosine {
            normalize(query)
        } else {
            query.to_vec()
        };
        let quant = quantize(&processed);
        let q = QQuery {
            codes: &quant.codes,
            scale: quant.scale,
            sqnorm: quant.sqnorm,
        };
        let mut ep = self.entry.unwrap();
        let mut cur = self.dq(&q, ep);
        for lc in (1..=self.max_level).rev() {
            (ep, cur) = self.greedy1_query(&q, ep, cur, lc);
        }
        let ef = ef_search.max(k);
        let mut w = self.search_layer_query(&q, ep, ef, 0, &filter);

        let hib = self.metric.higher_is_better();
        if self.raw.is_some() {
            // Rerank the beam with exact f32 distances.
            for dn in w.iter_mut() {
                let v = self.raw_at(dn.node);
                dn.dist = match self.metric {
                    Metric::L2 => l2sq_f32(v, &processed),
                    Metric::Dot | Metric::Cosine => -dot_f32(v, &processed),
                };
            }
        }
        w.sort_by(|a, b| a.dist.total_cmp(&b.dist));
        w.into_iter()
            .take(k)
            .map(|DN { dist, node }| Hit {
                id: self.ids[node as usize],
                score: if hib { -dist } else { dist },
            })
            .collect()
    }

    fn greedy1_query(&self, q: &QQuery, mut ep: u32, mut ep_dist: f32, lc: usize) -> (u32, f32) {
        let mut improved = true;
        while improved {
            improved = false;
            for &nb in &self.links[ep as usize][lc] {
                let d = self.dq(q, nb);
                if d < ep_dist {
                    ep_dist = d;
                    ep = nb;
                    improved = true;
                }
            }
        }
        (ep, ep_dist)
    }
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        v.iter().map(|x| x * inv).collect()
    } else {
        v.to_vec()
    }
}

// ---------------------------------------------------------------------------
// Persistence: a quantized-HNSW `.vecdb` file (magic "VECDBHQ1").
// Header (64B) + codes i8 + scales f32 + sqnorms f32 + ids u64 + levels u32
// + variable-length links + raw f32? (each section 16B-aligned).
// ---------------------------------------------------------------------------

const HQ_MAGIC: &[u8; 8] = b"VECDBHQ1";
const HQ_VERSION: u32 = 1;
const HQ_FLAG_HAS_RAW: u32 = 1;

#[inline]
fn align16(x: usize) -> usize {
    (x + 15) & !15
}

impl HnswQIndex {
    /// Serialize the graph to a quantized-HNSW `.vecdb` file.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let dim = self.dim;
        let count = self.len();
        let has_raw = self.raw.is_some();
        let mut links_bytes = 0usize;
        for i in 0..count {
            for lc in 0..=self.levels[i] {
                links_bytes += 4 + self.links[i][lc].len() * 4;
            }
        }
        let codes_off = align16(64);
        let scales_off = align16(codes_off + count * dim);
        let sqnorms_off = align16(scales_off + count * 4);
        let ids_off = align16(sqnorms_off + count * 4);
        let lvl_off = align16(ids_off + count * 8);
        let links_off = align16(lvl_off + count * 4);
        let raw_off = align16(links_off + links_bytes);
        let total = if has_raw {
            align16(raw_off + count * dim * 4)
        } else {
            align16(links_off + links_bytes)
        };

        let mut b = vec![0u8; total];
        b[0..8].copy_from_slice(HQ_MAGIC);
        b[8..12].copy_from_slice(&HQ_VERSION.to_le_bytes());
        b[12..16].copy_from_slice(&(self.metric as u32).to_le_bytes());
        b[16..20].copy_from_slice(&(dim as u32).to_le_bytes());
        b[20..24].copy_from_slice(&(count as u32).to_le_bytes());
        b[24..28].copy_from_slice(&(if has_raw { HQ_FLAG_HAS_RAW } else { 0 }).to_le_bytes());
        b[28..32].copy_from_slice(&(self.m as u32).to_le_bytes());
        b[32..36].copy_from_slice(&(self.ef_construction as u32).to_le_bytes());
        b[36..40].copy_from_slice(&(self.max_level as u32).to_le_bytes());
        b[40..44].copy_from_slice(&(self.entry.unwrap_or(0)).to_le_bytes());
        b[44..52].copy_from_slice(&self.rng.to_le_bytes());

        let codes_u8: &[u8] = unsafe {
            std::slice::from_raw_parts(self.codes.as_ptr() as *const u8, self.codes.len())
        };
        b[codes_off..codes_off + codes_u8.len()].copy_from_slice(codes_u8);
        for (i, &x) in self.scales.iter().enumerate() {
            b[scales_off + i * 4..scales_off + i * 4 + 4].copy_from_slice(&x.to_le_bytes());
        }
        for (i, &x) in self.sqnorms.iter().enumerate() {
            b[sqnorms_off + i * 4..sqnorms_off + i * 4 + 4].copy_from_slice(&x.to_le_bytes());
        }
        for (i, &id) in self.ids.iter().enumerate() {
            b[ids_off + i * 8..ids_off + i * 8 + 8].copy_from_slice(&id.to_le_bytes());
        }
        for (i, &lv) in self.levels.iter().enumerate() {
            b[lvl_off + i * 4..lvl_off + i * 4 + 4].copy_from_slice(&(lv as u32).to_le_bytes());
        }
        let mut p = links_off;
        for i in 0..count {
            for lc in 0..=self.levels[i] {
                let nbrs = &self.links[i][lc];
                b[p..p + 4].copy_from_slice(&(nbrs.len() as u32).to_le_bytes());
                p += 4;
                for &nb in nbrs {
                    b[p..p + 4].copy_from_slice(&nb.to_le_bytes());
                    p += 4;
                }
            }
        }
        if let Some(raw) = self.raw.as_ref() {
            for (i, &x) in raw.iter().enumerate() {
                b[raw_off + i * 4..raw_off + i * 4 + 4].copy_from_slice(&x.to_le_bytes());
            }
        }
        std::fs::write(path, &b)
    }

    /// Load a quantized-HNSW `.vecdb` file.
    pub fn load(path: impl AsRef<Path>) -> io::Result<HnswQIndex> {
        let b = std::fs::read(path)?;
        let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, format!("hnsw_q: {m}"));
        if b.len() < 64 || &b[0..8] != HQ_MAGIC {
            return Err(bad("bad magic"));
        }
        let u32_at = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        if u32_at(8) != HQ_VERSION {
            return Err(bad("unsupported version"));
        }
        let metric = Metric::from_u32(u32_at(12)).ok_or_else(|| bad("bad metric"))?;
        let dim = u32_at(16) as usize;
        let count = u32_at(20) as usize;
        let has_raw = u32_at(24) & HQ_FLAG_HAS_RAW != 0;
        let m = u32_at(28) as usize;
        let ef_construction = u32_at(32) as usize;
        let max_level = u32_at(36) as usize;
        let entry_raw = u32_at(40);
        let mut rng = [0u8; 8];
        rng.copy_from_slice(&b[44..52]);
        let rng = u64::from_le_bytes(rng);

        let codes_off = align16(64);
        let scales_off = align16(codes_off + count * dim);
        let sqnorms_off = align16(scales_off + count * 4);
        let ids_off = align16(sqnorms_off + count * 4);
        let lvl_off = align16(ids_off + count * 8);
        let links_off = align16(lvl_off + count * 4);
        if b.len() < links_off {
            return Err(bad("file truncated"));
        }

        let codes: Vec<i8> = b[codes_off..codes_off + count * dim]
            .iter()
            .map(|&x| x as i8)
            .collect();
        let scales: Vec<f32> = (0..count)
            .map(|i| {
                let o = scales_off + i * 4;
                f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
            })
            .collect();
        let sqnorms: Vec<f32> = (0..count)
            .map(|i| {
                let o = sqnorms_off + i * 4;
                f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
            })
            .collect();
        let ids: Vec<u64> = (0..count)
            .map(|i| {
                let o = ids_off + i * 8;
                let mut a = [0u8; 8];
                a.copy_from_slice(&b[o..o + 8]);
                u64::from_le_bytes(a)
            })
            .collect();
        let levels: Vec<usize> = (0..count)
            .map(|i| u32_at(lvl_off + i * 4) as usize)
            .collect();

        let mut links: Vec<Vec<Vec<u32>>> = Vec::with_capacity(count);
        let mut p = links_off;
        for &lv in levels.iter() {
            let mut node_links = Vec::with_capacity(lv + 1);
            for _ in 0..=lv {
                if p + 4 > b.len() {
                    return Err(bad("links truncated"));
                }
                let len = u32_at(p) as usize;
                p += 4;
                let mut lst = Vec::with_capacity(len);
                for _ in 0..len {
                    if p + 4 > b.len() {
                        return Err(bad("links truncated"));
                    }
                    lst.push(u32_at(p));
                    p += 4;
                }
                node_links.push(lst);
            }
            links.push(node_links);
        }
        let raw = if has_raw {
            let raw_off = align16(p);
            if b.len() < raw_off + count * dim * 4 {
                return Err(bad("raw truncated"));
            }
            Some(
                (0..count * dim)
                    .map(|i| {
                        let o = raw_off + i * 4;
                        f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
                    })
                    .collect(),
            )
        } else {
            None
        };

        Ok(HnswQIndex {
            dim,
            metric,
            m,
            m0: m * 2,
            ef_construction,
            ml: 1.0 / (m as f64).ln(),
            codes,
            scales,
            sqnorms,
            raw,
            ids,
            links,
            levels,
            entry: if count > 0 { Some(entry_raw) } else { None },
            max_level,
            rng,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(metric: Metric, keep_raw: bool) -> (HnswQIndex, Vec<(u64, Vec<f32>)>) {
        let dim = 32;
        let mut s: u64 = 987;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        };
        let items: Vec<(u64, Vec<f32>)> = (0..2000)
            .map(|i| (i as u64, (0..dim).map(|_| next()).collect()))
            .collect();
        let mut idx = HnswQIndex::new(dim, metric, 16, 200, keep_raw);
        for (id, v) in &items {
            idx.add(*id, v);
        }
        (idx, items)
    }

    #[test]
    fn quantized_hnsw_recall_with_rerank() {
        let (idx, items) = build(Metric::L2, true);
        let mut flat = crate::FlatIndex::new(32, Metric::L2, true);
        for (id, v) in &items {
            flat.add(*id, v);
        }
        let mut hit = 0;
        let mut total = 0;
        for t in 0..50 {
            let q = &items[t * 13 % items.len()].1;
            let truth: std::collections::HashSet<u64> =
                flat.search_exact(q, 10).iter().map(|h| h.id).collect();
            hit += idx
                .search(q, 10, 96)
                .iter()
                .filter(|h| truth.contains(&h.id))
                .count();
            total += truth.len();
        }
        let recall = hit as f64 / total as f64;
        assert!(recall >= 0.90, "recall too low: {recall}");
    }

    #[test]
    fn quantized_hnsw_compact_no_raw_ranks() {
        // Without rerank the int8 graph should still find good neighbors.
        let (idx, items) = build(Metric::L2, false);
        assert!(!idx.has_raw());
        let mut flat = crate::FlatIndex::new(32, Metric::L2, true);
        for (id, v) in &items {
            flat.add(*id, v);
        }
        let mut hit = 0;
        let mut total = 0;
        for t in 0..30 {
            let q = &items[t * 17 % items.len()].1;
            let truth: std::collections::HashSet<u64> =
                flat.search_exact(q, 10).iter().map(|h| h.id).collect();
            hit += idx
                .search(q, 10, 96)
                .iter()
                .filter(|h| truth.contains(&h.id))
                .count();
            total += truth.len();
        }
        let recall = hit as f64 / total as f64;
        assert!(recall >= 0.70, "int8-graph recall too low: {recall}");
    }

    #[test]
    fn quantized_hnsw_filtered() {
        let (idx, items) = build(Metric::L2, true);
        let got = idx.search_filter(&items[0].1, 10, 128, |id| id % 3 == 0);
        assert!(got.iter().all(|h| h.id % 3 == 0));
    }

    #[test]
    fn quantized_hnsw_save_load_roundtrip() {
        let (idx, items) = build(Metric::L2, true);
        let mut path = std::env::temp_dir();
        path.push("vecdb_hnswq_test.vecdb");
        idx.save(&path).unwrap();
        let loaded = HnswQIndex::load(&path).unwrap();
        assert_eq!(loaded.len(), idx.len());
        for t in 0..20 {
            let q = &items[t * 41 % items.len()].1;
            let a: Vec<u64> = idx.search(q, 10, 64).iter().map(|h| h.id).collect();
            let b: Vec<u64> = loaded.search(q, 10, 64).iter().map(|h| h.id).collect();
            assert_eq!(a, b);
        }
        std::fs::remove_file(&path).ok();
    }
}

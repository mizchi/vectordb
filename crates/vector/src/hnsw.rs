//! HNSW (Hierarchical Navigable Small World) graph index.
//!
//! Multi-layer proximity graph (Malkov & Yashunin, 2016): a query greedily
//! descends from the top layer to layer 0, then does a beam search of width
//! `ef` at layer 0. Gives the best recall/latency trade-off of the indexes
//! here, at the cost of storing the graph edges.
//!
//! Vectors are kept as f32 (normalized for cosine) and distances use the SIMD
//! kernels. Parameters: `m` (edges/layer), `ef_construction` (build beam),
//! `ef_search` (query beam).

use crate::distance::{dot_f32, l2sq_f32};
use crate::index::{Hit, Metric};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::io;
use std::path::Path;

/// A distance/node pair ordered by distance (smaller = closer). Used in the
/// max-heap of current best results; wrap in `Reverse` for a nearest-first heap.
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

/// HNSW index over f32 vectors.
pub struct HnswIndex {
    dim: usize,
    metric: Metric,
    m: usize,
    m0: usize,
    ef_construction: usize,
    ml: f64,
    vectors: Vec<f32>, // count * dim (processed)
    ids: Vec<u64>,
    links: Vec<Vec<Vec<u32>>>, // [node][level] -> neighbors
    levels: Vec<usize>,
    entry: Option<u32>,
    max_level: usize,
    rng: u64,
    deleted: Vec<bool>, // per-node tombstones (in-memory)
    deleted_count: usize,
}

impl HnswIndex {
    /// Create an empty index. `m` ≈ 16–32; `ef_construction` ≈ 100–400.
    pub fn new(dim: usize, metric: Metric, m: usize, ef_construction: usize) -> Self {
        assert!(dim > 0 && m > 0);
        HnswIndex {
            dim,
            metric,
            m,
            m0: m * 2,
            ef_construction: ef_construction.max(m),
            ml: 1.0 / (m as f64).ln(),
            vectors: Vec::new(),
            ids: Vec::new(),
            links: Vec::new(),
            levels: Vec::new(),
            entry: None,
            max_level: 0,
            rng: 0x9E37_79B9_7F4A_7C15,
            deleted: Vec::new(),
            deleted_count: 0,
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
    /// Number of live (non-tombstoned) vectors.
    pub fn live_len(&self) -> usize {
        self.ids.len() - self.deleted_count
    }

    /// Tombstone every node with external id `id`; it is excluded from search
    /// results (the graph is still traversed through it, so connectivity is
    /// preserved). Tombstones are in-memory — call [`compact`](Self::compact)
    /// to rebuild the graph without them (and before `save` to persist the
    /// removal). Returns how many nodes were newly deleted.
    pub fn remove(&mut self, id: u64) -> usize {
        let mut removed = 0;
        for i in 0..self.ids.len() {
            if self.ids[i] == id && !self.deleted[i] {
                self.deleted[i] = true;
                self.deleted_count += 1;
                removed += 1;
            }
        }
        removed
    }

    /// Rebuild the graph from the live nodes only, physically dropping
    /// tombstoned ones. This re-inserts every survivor, so it is O(n log n).
    pub fn compact(&mut self) {
        if self.deleted_count == 0 {
            return;
        }
        let mut rebuilt = HnswIndex::new(self.dim, self.metric, self.m, self.ef_construction);
        for i in 0..self.ids.len() {
            if !self.deleted[i] {
                let v = self.vec_at(i as u32).to_vec();
                rebuilt.add(self.ids[i], &v);
            }
        }
        *self = rebuilt;
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

    fn vec_at(&self, i: u32) -> &[f32] {
        let i = i as usize;
        &self.vectors[i * self.dim..(i + 1) * self.dim]
    }

    /// Distance where smaller = closer, for the active metric.
    #[inline]
    fn d(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric {
            Metric::L2 => l2sq_f32(a, b),
            Metric::Dot | Metric::Cosine => -dot_f32(a, b),
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
        let node = self.ids.len() as u32;
        let level = self.random_level();
        self.vectors.extend_from_slice(&processed);
        self.ids.push(id);
        self.levels.push(level);
        self.links.push((0..=level).map(|_| Vec::new()).collect());
        self.deleted.push(false);

        let Some(mut ep) = self.entry else {
            self.entry = Some(node);
            self.max_level = level;
            return;
        };

        // Descend from the top down to level+1 with a greedy ef=1 search.
        let mut cur = self.d(self.vec_at(ep), &processed);
        let top = self.max_level;
        for lc in ((level + 1)..=top).rev() {
            (ep, cur) = self.greedy1(&processed, ep, cur, lc);
        }

        // Insert into layers min(level, top) .. 0.
        let start = level.min(top);
        for lc in (0..=start).rev() {
            let mut w = self.search_layer(&processed, &[ep], self.ef_construction, lc, &|_| true);
            // Entry point for the next layer down = nearest found.
            if let Some(best) = w.iter().min_by(|a, b| a.dist.total_cmp(&b.dist)) {
                ep = best.node;
            }
            let m = if lc == 0 { self.m0 } else { self.m };
            let selected = self.select_neighbors(&processed, &mut w, m);
            for &nb in &selected {
                self.links[node as usize][lc].push(nb);
                self.links[nb as usize][lc].push(node);
                // Prune the neighbor's list if it now exceeds the budget.
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

    /// Greedy single-best descent within one layer.
    fn greedy1(&self, q: &[f32], mut ep: u32, mut ep_dist: f32, lc: usize) -> (u32, f32) {
        let mut improved = true;
        while improved {
            improved = false;
            for &nb in &self.links[ep as usize][lc] {
                let d = self.d(self.vec_at(nb), q);
                if d < ep_dist {
                    ep_dist = d;
                    ep = nb;
                    improved = true;
                }
            }
        }
        (ep, ep_dist)
    }

    /// Beam search within one layer; returns up to `ef` nearest as a vec.
    /// `filter(id)` gates admission into the result beam `w` — non-passing
    /// nodes are still traversed (so the graph stays navigable), they just
    /// never become results. Build passes an always-true filter.
    fn search_layer<F: Fn(u64) -> bool>(
        &self,
        q: &[f32],
        eps: &[u32],
        ef: usize,
        lc: usize,
        filter: &F,
    ) -> Vec<DN> {
        let mut visited = vec![false; self.ids.len()];
        let mut cands: BinaryHeap<Reverse<DN>> = BinaryHeap::new(); // nearest first
        let mut w: BinaryHeap<DN> = BinaryHeap::new(); // farthest on top
        for &e in eps {
            let dist = self.d(self.vec_at(e), q);
            visited[e as usize] = true;
            cands.push(Reverse(DN { dist, node: e }));
            if !self.deleted[e as usize] && filter(self.ids[e as usize]) {
                w.push(DN { dist, node: e });
            }
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
                let d = self.d(self.vec_at(nb), q);
                let farthest = w.peek().map(|x| x.dist).unwrap_or(f32::INFINITY);
                if d < farthest || w.len() < ef {
                    cands.push(Reverse(DN { dist: d, node: nb }));
                    if !self.deleted[nb as usize] && filter(self.ids[nb as usize]) {
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

    /// Neighbor selection heuristic (keeps a diverse set close to `q`).
    fn select_neighbors(&self, q: &[f32], w: &mut [DN], m: usize) -> Vec<u32> {
        let _ = q;
        w.sort_by(|a, b| a.dist.total_cmp(&b.dist));
        let mut result: Vec<u32> = Vec::with_capacity(m);
        for cand in w.iter() {
            if result.len() >= m {
                break;
            }
            // Keep `cand` only if it is closer to q than to any already-kept
            // neighbor (promotes diversity, avoids clustered edges).
            let mut keep = true;
            for &r in &result {
                if self.d(self.vec_at(cand.node), self.vec_at(r)) < cand.dist {
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

    /// Re-select a node's neighbor list down to `budget` via the heuristic.
    fn prune(&mut self, node: u32, lc: usize, budget: usize) {
        let q = self.vec_at(node).to_vec();
        let mut w: Vec<DN> = self.links[node as usize][lc]
            .iter()
            .map(|&nb| DN {
                dist: self.d(self.vec_at(nb), &q),
                node: nb,
            })
            .collect();
        let kept = self.select_neighbors(&q, &mut w, budget);
        self.links[node as usize][lc] = kept;
    }

    /// Search for the `k` nearest neighbors with beam width `ef_search`.
    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<Hit> {
        self.search_filter(query, k, ef_search, |_| true)
    }

    /// Filtered search: only ids satisfying `filter` become results. The graph
    /// is still traversed through non-passing nodes to stay navigable, so use a
    /// larger `ef_search` when the filter is very selective.
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
        let mut ep = self.entry.unwrap();
        let mut cur = self.d(self.vec_at(ep), &processed);
        for lc in (1..=self.max_level).rev() {
            (ep, cur) = self.greedy1(&processed, ep, cur, lc);
        }
        let ef = ef_search.max(k);
        let mut w = self.search_layer(&processed, &[ep], ef, 0, &filter);
        w.sort_by(|a, b| a.dist.total_cmp(&b.dist));
        let hib = self.metric.higher_is_better();
        w.into_iter()
            .take(k)
            .map(|DN { dist, node }| Hit {
                id: self.ids[node as usize],
                score: if hib { -dist } else { dist },
            })
            .collect()
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
// Persistence: an HNSW `.vecdb` file (magic "VECDBHN1").
// Header (64B) + f32 vectors + u64 ids + u32 levels + variable-length links
// (per node, per level: u32 len then len × u32 neighbor).
// ---------------------------------------------------------------------------

const HNSW_MAGIC: &[u8; 8] = b"VECDBHN1";
const HNSW_VERSION: u32 = 1;
const HNSW_FLAG_HAS_DELETED: u32 = 1;

#[inline]
fn align16(x: usize) -> usize {
    (x + 15) & !15
}

impl HnswIndex {
    /// Serialize the graph to an HNSW `.vecdb` file.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        std::fs::write(path, self.to_bytes())
    }

    /// Serialize the graph to the in-memory HNSW `.vecdb` byte image that
    /// [`save`](Self::save) writes (byte-identical), for a filesystem-free
    /// "bytes in / bytes out" round trip. Pair with [`from_bytes`](Self::from_bytes).
    pub fn to_bytes(&self) -> Vec<u8> {
        let dim = self.dim;
        let count = self.len();
        let mut links_bytes = 0usize;
        for i in 0..count {
            for lc in 0..=self.levels[i] {
                links_bytes += 4 + self.links[i][lc].len() * 4;
            }
        }
        let has_deleted = self.deleted_count > 0;
        let vec_off = align16(64);
        let ids_off = align16(vec_off + count * dim * 4);
        let lvl_off = align16(ids_off + count * 8);
        let links_off = align16(lvl_off + count * 4);
        let links_end = links_off + links_bytes;
        // Optional tombstone section (1 byte/node) after the links, gated by a
        // flag in the previously-reserved header word at offset 48. Absent when
        // there are no deletions, so the layout stays identical to before.
        let deleted_off = align16(links_end);
        let total = if has_deleted {
            deleted_off + count
        } else {
            links_end
        };

        let mut b = vec![0u8; total];
        b[0..8].copy_from_slice(HNSW_MAGIC);
        b[8..12].copy_from_slice(&HNSW_VERSION.to_le_bytes());
        b[12..16].copy_from_slice(&(self.metric as u32).to_le_bytes());
        b[16..20].copy_from_slice(&(dim as u32).to_le_bytes());
        b[20..24].copy_from_slice(&(count as u32).to_le_bytes());
        b[24..28].copy_from_slice(&(self.m as u32).to_le_bytes());
        b[28..32].copy_from_slice(&(self.ef_construction as u32).to_le_bytes());
        b[32..36].copy_from_slice(&(self.max_level as u32).to_le_bytes());
        b[36..40].copy_from_slice(&(self.entry.unwrap_or(0)).to_le_bytes());
        b[40..48].copy_from_slice(&self.rng.to_le_bytes());
        b[48..52].copy_from_slice(
            &(if has_deleted {
                HNSW_FLAG_HAS_DELETED
            } else {
                0
            })
            .to_le_bytes(),
        );

        for (j, &x) in self.vectors.iter().enumerate() {
            b[vec_off + j * 4..vec_off + j * 4 + 4].copy_from_slice(&x.to_le_bytes());
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
        if has_deleted {
            for (i, &d) in self.deleted.iter().enumerate() {
                b[deleted_off + i] = d as u8;
            }
        }
        b
    }

    /// Load an HNSW `.vecdb` file.
    pub fn load(path: impl AsRef<Path>) -> io::Result<HnswIndex> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    /// Parse an HNSW index from a `.vecdb` byte image (the "bytes in" counterpart
    /// to [`to_bytes`](Self::to_bytes)). Performs the same validation as
    /// [`load`](Self::load) and returns the same [`io::Error`]s.
    pub fn from_bytes(b: &[u8]) -> io::Result<HnswIndex> {
        let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, format!("hnsw: {m}"));
        if b.len() < 64 || &b[0..8] != HNSW_MAGIC {
            return Err(bad("bad magic"));
        }
        let u32_at = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        if u32_at(8) != HNSW_VERSION {
            return Err(bad("unsupported version"));
        }
        let metric = Metric::from_u32(u32_at(12)).ok_or_else(|| bad("bad metric"))?;
        let dim = u32_at(16) as usize;
        let count = u32_at(20) as usize;
        let m = u32_at(24) as usize;
        let ef_construction = u32_at(28) as usize;
        let max_level = u32_at(32) as usize;
        let entry_raw = u32_at(36);
        let mut rng = [0u8; 8];
        rng.copy_from_slice(&b[40..48]);
        let rng = u64::from_le_bytes(rng);

        let vec_off = align16(64);
        let ids_off = align16(vec_off + count * dim * 4);
        let lvl_off = align16(ids_off + count * 8);
        let links_off = align16(lvl_off + count * 4);
        if b.len() < links_off {
            return Err(bad("file truncated"));
        }

        let vectors: Vec<f32> = (0..count * dim)
            .map(|j| {
                let o = vec_off + j * 4;
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

        // Optional tombstone section after the links (flag at offset 48).
        let (deleted, deleted_count) = if u32_at(48) & HNSW_FLAG_HAS_DELETED != 0 {
            let doff = align16(p);
            if b.len() < doff + count {
                return Err(bad("tombstones truncated"));
            }
            let d: Vec<bool> = (0..count).map(|i| b[doff + i] != 0).collect();
            let dc = d.iter().filter(|&&x| x).count();
            (d, dc)
        } else {
            (vec![false; count], 0)
        };

        Ok(HnswIndex {
            dim,
            metric,
            m,
            m0: m * 2,
            ef_construction,
            ml: 1.0 / (m as f64).ln(),
            vectors,
            ids,
            links,
            levels,
            entry: if count > 0 { Some(entry_raw) } else { None },
            max_level,
            rng,
            deleted,
            deleted_count,
        })
    }
}

#[cfg(feature = "parallel")]
impl HnswIndex {
    /// Run many queries concurrently (one query per rayon task).
    pub fn search_batch(&self, queries: &[Vec<f32>], k: usize, ef_search: usize) -> Vec<Vec<Hit>> {
        use rayon::prelude::*;
        queries
            .par_iter()
            .map(|q| self.search(q, k, ef_search))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(metric: Metric) -> (HnswIndex, Vec<(u64, Vec<f32>)>) {
        let dim = 32;
        let mut s: u64 = 123;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        };
        let items: Vec<(u64, Vec<f32>)> = (0..2000)
            .map(|i| (i as u64, (0..dim).map(|_| next()).collect()))
            .collect();
        let mut idx = HnswIndex::new(dim, metric, 16, 200);
        for (id, v) in &items {
            idx.add(*id, v);
        }
        (idx, items)
    }

    #[test]
    fn hnsw_high_recall_vs_exact() {
        let (idx, items) = build(Metric::L2);
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
                .search(q, 10, 64)
                .iter()
                .filter(|h| truth.contains(&h.id))
                .count();
            total += truth.len();
        }
        let recall = hit as f64 / total as f64;
        assert!(recall >= 0.95, "recall too low: {recall}");
    }

    #[test]
    fn hnsw_filtered_search() {
        let (idx, items) = build(Metric::L2);
        let mut flat = crate::FlatIndex::new(32, Metric::L2, true);
        for (id, v) in &items {
            flat.add(*id, v);
        }
        let filter = |id: u64| id.is_multiple_of(3);
        let mut hit = 0;
        let mut total = 0;
        for t in 0..30 {
            let q = &items[t * 17 % items.len()].1;
            let got = idx.search_filter(q, 10, 128, filter);
            assert!(got.iter().all(|h| h.id % 3 == 0));
            // Ground truth among the filtered subset.
            let truth: std::collections::HashSet<u64> = flat
                .search_filter(q, 10, 8, filter)
                .iter()
                .map(|h| h.id)
                .collect();
            hit += got.iter().filter(|h| truth.contains(&h.id)).count();
            total += truth.len();
        }
        let recall = hit as f64 / total as f64;
        assert!(recall >= 0.90, "filtered recall too low: {recall}");
    }

    #[test]
    fn hnsw_save_load_roundtrip() {
        let (idx, items) = build(Metric::L2);
        let mut path = std::env::temp_dir();
        path.push("vecdb_hnsw_test.vecdb");
        idx.save(&path).unwrap();
        let loaded = HnswIndex::load(&path).unwrap();
        assert_eq!(loaded.len(), idx.len());
        // Same queries return the same results after a save/load round-trip.
        for t in 0..20 {
            let q = &items[t * 41 % items.len()].1;
            let a: Vec<u64> = idx.search(q, 10, 64).iter().map(|h| h.id).collect();
            let b: Vec<u64> = loaded.search(q, 10, 64).iter().map(|h| h.id).collect();
            assert_eq!(a, b);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn hnsw_bytes_roundtrip() {
        let (idx, items) = build(Metric::L2);
        let b = idx.to_bytes();
        let idx2 = HnswIndex::from_bytes(&b).unwrap();
        assert_eq!(idx2.len(), idx.len());
        for t in 0..20 {
            let q = &items[t * 41 % items.len()].1;
            let a: Vec<u64> = idx.search(q, 10, 64).iter().map(|h| h.id).collect();
            let c: Vec<u64> = idx2.search(q, 10, 64).iter().map(|h| h.id).collect();
            assert_eq!(a, c);
        }
        assert_eq!(idx2.to_bytes(), b); // byte-stable round trip
    }

    #[test]
    fn hnsw_soft_delete_and_compact() {
        let (mut idx, items) = build(Metric::L2);
        let n = items.len();
        // Remove the exact nearest neighbor of several queries and confirm it
        // never appears; a live neighbor should take its place.
        let victim = items[7].0;
        assert_eq!(idx.remove(victim), 1);
        assert_eq!(idx.remove(victim), 0);
        assert_eq!(idx.live_len(), n - 1);
        for t in 0..20 {
            let q = &items[t * 13 % n].1;
            assert!(idx.search(q, 10, 64).iter().all(|h| h.id != victim));
        }
        // Tombstones survive a save/load round-trip (before compaction).
        let mut path = std::env::temp_dir();
        path.push("vecdb_hnsw_tombstone.vecdb");
        idx.save(&path).unwrap();
        let loaded = HnswIndex::load(&path).unwrap();
        assert_eq!(loaded.len(), n);
        assert_eq!(loaded.live_len(), n - 1);
        for t in 0..20 {
            let q = &items[t * 13 % n].1;
            assert!(loaded.search(q, 10, 64).iter().all(|h| h.id != victim));
        }
        std::fs::remove_file(&path).ok();

        // Compaction rebuilds without the tombstone; results stay valid.
        idx.compact();
        assert_eq!(idx.len(), n - 1);
        assert_eq!(idx.live_len(), n - 1);
        // Recall of the compacted graph is still high vs exact over live set.
        let mut flat = crate::FlatIndex::new(32, Metric::L2, true);
        for (id, v) in &items {
            if *id != victim {
                flat.add(*id, v);
            }
        }
        let mut hit = 0;
        let mut total = 0;
        for t in 0..30 {
            let q = &items[t * 11 % n].1;
            let truth: std::collections::HashSet<u64> =
                flat.search_exact(q, 10).iter().map(|h| h.id).collect();
            hit += idx
                .search(q, 10, 64)
                .iter()
                .filter(|h| truth.contains(&h.id))
                .count();
            total += truth.len();
        }
        assert!(hit as f64 / total as f64 >= 0.90);
    }

    #[test]
    fn hnsw_cosine_finds_self() {
        let (idx, items) = build(Metric::Cosine);
        for t in 0..20 {
            let (id, v) = &items[t * 31 % items.len()];
            let hits = idx.search(v, 1, 32);
            assert_eq!(hits[0].id, *id);
        }
    }
}

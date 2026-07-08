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
            if filter(self.ids[e as usize]) {
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
            hit += idx.search(q, 10, 64).iter().filter(|h| truth.contains(&h.id)).count();
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
    fn hnsw_cosine_finds_self() {
        let (idx, items) = build(Metric::Cosine);
        for t in 0..20 {
            let (id, v) = &items[t * 31 % items.len()];
            let hits = idx.search(v, 1, 32);
            assert_eq!(hits[0].id, *id);
        }
    }
}

//! Build a [`GraphStore`] from embeddings: run kNN over the vectors (reusing a
//! vector index) to get semantic edges, optionally keep only reciprocal
//! (mutual-kNN) edges, then merge explicit links. This is where the vector
//! layer does the heavy lifting — the graph layer only turns neighbor lists into
//! a weighted graph.

use crate::graph::{EdgeKind, GraphStore};
use meandb_vector::{HnswIndex, Metric};
use std::collections::HashMap;

/// Knobs for [`GraphBuilder::build`].
pub struct GraphBuilder {
    /// Semantic out-edges kept per node (before link merge).
    pub k: usize,
    /// HNSW search beam (higher = better neighbor recall).
    pub ef: usize,
    /// Drop semantic edges below this weight (similarity). `f32::MIN` = keep all.
    pub min_weight: f32,
    /// Keep a semantic edge only if it is reciprocal (mutual-kNN) — a cleaner,
    /// more "undirected" graph for a global view.
    pub mutual: bool,
    /// Weight assigned to an explicit link edge.
    pub link_weight: f32,
    pub metric: Metric,
}

impl Default for GraphBuilder {
    fn default() -> Self {
        GraphBuilder {
            k: 10,
            ef: 64,
            min_weight: f32::MIN,
            mutual: false,
            link_weight: 1.0,
            metric: Metric::Cosine,
        }
    }
}

impl GraphBuilder {
    pub fn new(metric: Metric, k: usize) -> Self {
        GraphBuilder {
            k,
            metric,
            ..Default::default()
        }
    }

    /// Build the graph. `items` are `(id, embedding)`; `links` are explicit
    /// directed `(src_id, dst_id)` edges (e.g. wikilinks) — pass `&[]` for none.
    pub fn build(&self, items: &[(u64, Vec<f32>)], links: &[(u64, u64)]) -> GraphStore {
        assert!(!items.is_empty(), "GraphBuilder::build: empty input");
        let n = items.len();
        let dim = items[0].1.len();
        let id_to_idx: HashMap<u64, u32> = items
            .iter()
            .enumerate()
            .map(|(i, (id, _))| (*id, i as u32))
            .collect();

        // 1) Semantic kNN via the vector layer's HNSW index.
        let mut hnsw = HnswIndex::new(dim, self.metric, 16, 200.max(self.ef));
        for (id, v) in items {
            hnsw.add(*id, v);
        }

        // adj[i] = out-edges of node i, keyed by dst index -> (weight, kind).
        let mut adj: Vec<HashMap<u32, (f32, EdgeKind)>> = vec![HashMap::new(); n];
        for (i, (_, v)) in items.iter().enumerate() {
            for h in hnsw.search(v, self.k + 1, self.ef) {
                let Some(&j) = id_to_idx.get(&h.id) else {
                    continue;
                };
                if j as usize == i || h.score < self.min_weight {
                    continue; // skip self and weak edges
                }
                adj[i].insert(j, (h.score, EdgeKind::Semantic));
            }
        }

        // 2) Optional mutual-kNN: keep i->j only if j->i also exists.
        if self.mutual {
            for i in 0..n {
                let keep: Vec<u32> = adj[i]
                    .keys()
                    .copied()
                    .filter(|&j| adj[j as usize].contains_key(&(i as u32)))
                    .collect();
                adj[i].retain(|k, _| keep.contains(k));
            }
        }

        // 3) Merge explicit links (upgrade to `Both` where a semantic edge exists).
        for &(a, b) in links {
            let (Some(&ai), Some(&bi)) = (id_to_idx.get(&a), id_to_idx.get(&b)) else {
                continue;
            };
            if ai == bi {
                continue;
            }
            let e = adj[ai as usize]
                .entry(bi)
                .or_insert((self.link_weight, EdgeKind::Link));
            e.0 = e.0.max(self.link_weight);
            e.1 = match e.1 {
                EdgeKind::Semantic => EdgeKind::Both,
                other => other,
            };
        }

        // 4) Flatten to CSR, each node's edges sorted by descending weight.
        let ids: Vec<u64> = items.iter().map(|(id, _)| *id).collect();
        let mut offsets = Vec::with_capacity(n + 1);
        let mut dst = Vec::new();
        let mut weight = Vec::new();
        let mut kind = Vec::new();
        offsets.push(0usize);
        for out in &adj {
            let mut es: Vec<(u32, f32, EdgeKind)> =
                out.iter().map(|(&j, &(w, k))| (j, w, k)).collect();
            es.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            for (j, w, k) in es {
                dst.push(j);
                weight.push(w);
                kind.push(k.as_u8());
            }
            offsets.push(dst.len());
        }

        GraphStore::from_csr(self.metric, true, ids, offsets, dst, weight, kind)
    }
}

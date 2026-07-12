//! Incremental / live graph: `GraphIndex` keeps a mutable semantic graph you can
//! `insert` (upsert), `remove`, and `link` one node at a time, then `freeze`
//! into an immutable [`GraphStore`] for querying / persistence. This is the
//! FreshDiskANN-style update path — editing a note re-embeds just that note and
//! touches only its edges, instead of rebuilding the whole graph.
//!
//! Semantic edges are kept **symmetric** by adding a reciprocal edge whenever a
//! kNN edge is created, so `related` reflects new/edited notes from both sides.
//! Explicit links are stored separately and merged at `freeze`.
//!
//! Note: edges reflect each node's kNN *at insertion time*, so — unlike the
//! batch [`GraphBuilder`](crate::GraphBuilder) — a node inserted before its
//! cluster is populated can pick up weak, far edges. Set a `min_weight` cutoff
//! (recommended for a knowledge base — you rarely want low-similarity "related"
//! notes) so those never enter the graph.

use crate::graph::{EdgeKind, GraphStore, NodeMeta};
use std::collections::{HashMap, HashSet};
use vectordb::{HnswIndex, Metric};

/// A mutable, incrementally-updated semantic graph.
pub struct GraphIndex {
    metric: Metric,
    dim: usize, // 0 until the first insert fixes it
    /// Semantic out-edges kept per node on (re)insert.
    pub k: usize,
    /// HNSW search beam.
    pub ef: usize,
    /// Drop semantic edges below this weight.
    pub min_weight: f32,
    /// Cap on a node's out-degree in the frozen graph.
    pub max_degree: usize,
    /// Weight assigned to explicit link edges.
    pub link_weight: f32,
    hnsw: Option<HnswIndex>,
    sem: HashMap<u64, HashMap<u64, f32>>, // id -> (neighbor id -> weight)
    links: HashSet<(u64, u64)>,
    meta: HashMap<u64, NodeMeta>,
}

impl GraphIndex {
    pub fn new(metric: Metric, k: usize) -> Self {
        GraphIndex {
            metric,
            dim: 0,
            k,
            ef: 64,
            min_weight: f32::MIN,
            max_degree: k.max(1) * 4,
            link_weight: 1.0,
            hnsw: None,
            sem: HashMap::new(),
            links: HashSet::new(),
            meta: HashMap::new(),
        }
    }

    /// Number of live nodes.
    pub fn len(&self) -> usize {
        self.sem.len()
    }
    pub fn is_empty(&self) -> bool {
        self.sem.is_empty()
    }
    pub fn contains(&self, id: u64) -> bool {
        self.sem.contains_key(&id)
    }

    /// Insert or replace a node. On an edit (existing id) the old vector is
    /// tombstoned in the ANN index and the node's edges are recomputed; the
    /// reciprocal edges on its (old and new) neighbors are kept consistent.
    pub fn insert(&mut self, id: u64, embedding: &[f32], meta: NodeMeta) {
        if self.dim == 0 {
            self.dim = embedding.len();
            self.hnsw = Some(HnswIndex::new(self.dim, self.metric, 16, 200.max(self.ef)));
        }
        assert_eq!(embedding.len(), self.dim, "dimension mismatch");
        let hnsw = self.hnsw.as_mut().unwrap();
        let existed = self.sem.contains_key(&id);
        if existed {
            hnsw.remove(id); // tombstone the stale vector
        }
        hnsw.add(id, embedding);
        self.meta.insert(id, meta);

        // Drop every stale edge that referenced this node (both directions).
        for nbrs in self.sem.values_mut() {
            nbrs.remove(&id);
        }
        self.sem.entry(id).or_default().clear();

        // Recompute this node's kNN edges and mirror them reciprocally.
        let hits = hnsw.search(embedding, self.k + 1, self.ef);
        for h in hits {
            if h.id == id || h.score < self.min_weight {
                continue;
            }
            self.sem.get_mut(&id).unwrap().insert(h.id, h.score);
            self.sem.entry(h.id).or_default().insert(id, h.score);
        }
    }

    /// Remove a node and every edge touching it.
    pub fn remove(&mut self, id: u64) {
        if !self.sem.contains_key(&id) {
            return;
        }
        if let Some(h) = self.hnsw.as_mut() {
            h.remove(id);
        }
        self.sem.remove(&id);
        for nbrs in self.sem.values_mut() {
            nbrs.remove(&id);
        }
        self.meta.remove(&id);
        self.links.retain(|&(a, b)| a != id && b != id);
    }

    /// Add an explicit directed link (e.g. a wikilink). Endpoints need not exist
    /// yet; the link is applied at `freeze` if both are present.
    pub fn link(&mut self, src: u64, dst: u64) {
        self.links.insert((src, dst));
    }

    /// Suggest tags for a new note's `embedding` from its tagged neighbors in
    /// the current graph (kNN vote). Does not modify the graph.
    pub fn suggest_tags(
        &self,
        embedding: &[f32],
        opts: &crate::classify::SuggestOpts,
    ) -> Vec<crate::classify::TagSuggestion> {
        let Some(h) = &self.hnsw else {
            return Vec::new();
        };
        let hits = h.search(embedding, opts.k, opts.ef);
        let nbrs = hits.iter().filter_map(|hit| {
            self.meta
                .get(&hit.id)
                .map(|m| (m.tags.as_slice(), hit.score))
        });
        crate::classify::score_tags(nbrs, opts)
    }

    /// Auto-tag then insert: suggest tags from the current graph, apply them,
    /// insert the node, and return the applied tags. The classic
    /// "new article arrives → classify → file it" flow.
    pub fn insert_auto_tagged(
        &mut self,
        id: u64,
        embedding: &[f32],
        title: String,
        opts: &crate::classify::SuggestOpts,
    ) -> Vec<String> {
        let tags: Vec<String> = self
            .suggest_tags(embedding, opts)
            .into_iter()
            .map(|s| s.tag)
            .collect();
        self.insert(
            id,
            embedding,
            NodeMeta {
                title,
                tags: tags.clone(),
            },
        );
        tags
    }

    /// Freeze into an immutable [`GraphStore`]: merge semantic + link edges
    /// (overlaps become [`EdgeKind::Both`]), sort each node's edges by weight,
    /// cap at `max_degree`, and attach metadata. Nodes are ordered by ascending
    /// id for determinism.
    pub fn freeze(&self) -> GraphStore {
        let mut ids: Vec<u64> = self.sem.keys().copied().collect();
        ids.sort_unstable();
        let id_to_idx: HashMap<u64, u32> = ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i as u32))
            .collect();

        // Explicit links grouped by source (only where both endpoints exist).
        let mut link_of: HashMap<u64, Vec<u64>> = HashMap::new();
        for &(a, b) in &self.links {
            if id_to_idx.contains_key(&a) && id_to_idx.contains_key(&b) && a != b {
                link_of.entry(a).or_default().push(b);
            }
        }

        let mut offsets = Vec::with_capacity(ids.len() + 1);
        let mut dst = Vec::new();
        let mut weight = Vec::new();
        let mut kind = Vec::new();
        offsets.push(0usize);
        for &id in &ids {
            // Merge semantic edges with link edges for this node.
            let mut merged: HashMap<u64, (f32, EdgeKind)> = HashMap::new();
            for (&nb, &w) in &self.sem[&id] {
                if id_to_idx.contains_key(&nb) {
                    merged.insert(nb, (w, EdgeKind::Semantic));
                }
            }
            if let Some(dsts) = link_of.get(&id) {
                for &nb in dsts {
                    let e = merged
                        .entry(nb)
                        .or_insert((self.link_weight, EdgeKind::Link));
                    e.0 = e.0.max(self.link_weight);
                    e.1 = match e.1 {
                        EdgeKind::Semantic => EdgeKind::Both,
                        other => other,
                    };
                }
            }
            let mut es: Vec<(u32, f32, EdgeKind)> = merged
                .into_iter()
                .map(|(nb, (w, k))| (id_to_idx[&nb], w, k))
                .collect();
            es.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            es.truncate(self.max_degree);
            for (j, w, k) in es {
                dst.push(j);
                weight.push(w);
                kind.push(k.as_u8());
            }
            offsets.push(dst.len());
        }

        let mut g =
            GraphStore::from_csr(self.metric, true, ids.clone(), offsets, dst, weight, kind);
        let metas = ids
            .iter()
            .filter_map(|id| self.meta.get(id).map(|m| (*id, m.clone())));
        g.set_metadata(metas);
        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clustered() -> Vec<(u64, Vec<f32>)> {
        let mut s: u64 = 0x51ED_2701_ABCD_0001;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 * 0.05
        };
        let a = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        (0..8u64)
            .map(|i| {
                let c = if i < 4 { &a } else { &b };
                (i, c.iter().map(|&x| x + rnd()).collect())
            })
            .collect()
    }

    #[test]
    fn incremental_keeps_clusters() {
        let items = clustered();
        let mut gi = GraphIndex::new(Metric::Cosine, 3);
        gi.min_weight = 0.8;
        for (id, v) in &items {
            gi.insert(*id, v, NodeMeta::default());
        }
        gi.link(0, 5); // cross-cluster explicit link
        let g = gi.freeze();
        assert_eq!(g.len(), 8);
        // Semantic neighbors of a cluster-A node stay in cluster A.
        for id in 0..4u64 {
            for nb in g.neighbors(id) {
                if nb.kind == EdgeKind::Semantic {
                    assert!(nb.id < 4, "{id} -> {} crosses clusters", nb.id);
                }
            }
        }
        // The explicit link survived.
        assert!(g
            .neighbors(0)
            .iter()
            .any(|n| n.id == 5 && matches!(n.kind, EdgeKind::Link | EdgeKind::Both)));
    }

    #[test]
    fn edit_moves_node_between_clusters() {
        let items = clustered();
        let mut gi = GraphIndex::new(Metric::Cosine, 3);
        gi.min_weight = 0.8;
        for (id, v) in &items {
            gi.insert(*id, v, NodeMeta::default());
        }
        // Re-embed node 0 into cluster B; its related should now be cluster B.
        let bvec: Vec<f32> = items[5].1.clone();
        gi.insert(
            0,
            &bvec,
            NodeMeta {
                title: "moved".into(),
                tags: vec!["b".into()],
            },
        );
        let g = gi.freeze();
        assert_eq!(g.len(), 8); // upsert, not a new node
        assert_eq!(g.title(0), Some("moved"));
        let rel = g.related(0, 3);
        assert!(
            rel.iter().all(|n| n.id >= 4),
            "node 0 should now sit in cluster B"
        );
    }

    #[test]
    fn auto_tag_new_note() {
        let items = clustered();
        let mut gi = GraphIndex::new(Metric::Cosine, 3);
        gi.min_weight = 0.8;
        // Tag cluster A "a", cluster B "b".
        for (id, v) in &items {
            let tag = if *id < 4 { "a" } else { "b" };
            gi.insert(
                *id,
                v,
                NodeMeta {
                    title: String::new(),
                    tags: vec![tag.into()],
                },
            );
        }
        // A new note near cluster B should be auto-tagged "b".
        let newvec: Vec<f32> = items[6].1.clone();
        let opts = crate::classify::SuggestOpts {
            k: 3,
            min_score: 0.5,
            ..Default::default()
        };
        let sugg = gi.suggest_tags(&newvec, &opts);
        assert_eq!(sugg[0].tag, "b", "expected tag b, got {sugg:?}");

        let applied = gi.insert_auto_tagged(99, &newvec, "new".into(), &opts);
        assert_eq!(applied, vec!["b".to_string()]);
        let g = gi.freeze();
        assert_eq!(g.tags(99), vec!["b"]);
    }

    #[test]
    fn remove_drops_node_and_edges() {
        let items = clustered();
        let mut gi = GraphIndex::new(Metric::Cosine, 3);
        gi.min_weight = 0.8;
        for (id, v) in &items {
            gi.insert(*id, v, NodeMeta::default());
        }
        gi.remove(2);
        let g = gi.freeze();
        assert_eq!(g.len(), 7);
        assert!(g.neighbors(2).is_empty());
        for &id in g.ids() {
            assert!(g.neighbors(id).iter().all(|n| n.id != 2), "stale edge to 2");
        }
    }
}

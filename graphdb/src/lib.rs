//! # graphdb
//!
//! A compact **semantic knowledge graph** built on top of [`vectordb`]. A
//! knowledge-base graph (think Obsidian) is essentially a *kNN graph over
//! embeddings* plus explicit links, so graphdb reuses vectordb for the heavy
//! lifting (ANN search + SIMD similarity) and only adds the graph layer:
//!
//! - [`GraphBuilder`] turns embeddings into weighted edges (semantic kNN,
//!   optional mutual-kNN pruning) and merges explicit links.
//! - [`GraphStore`] is a CSR weighted-graph with a single-file `.graphdb`
//!   format (`to_bytes`/`from_bytes`, mmap-friendly, same discipline as
//!   `.vecdb`). It answers `related`, `neighborhood` (local graph view), and
//!   `export_json` (global graph view).
//! - [`analytics`] provides degree (node size) and label-propagation
//!   communities (node color) for rendering a graph view.
//!
//! ```
//! use graphdb::{GraphBuilder};
//! use vectordb::Metric;
//! let items = vec![
//!     (1u64, vec![1.0, 0.0, 0.0]),
//!     (2, vec![0.9, 0.1, 0.0]),
//!     (3, vec![0.0, 1.0, 0.0]),
//! ];
//! let g = GraphBuilder::new(Metric::Cosine, 2).build(&items, &[]);
//! // note 1's strongest related note is note 2
//! assert_eq!(g.related(1, 1)[0].id, 2);
//! ```

pub mod analytics;
pub mod build;
pub mod graph;

pub use build::GraphBuilder;
pub use graph::{EdgeKind, GraphStore, Neighbor, NodeMeta, Subgraph};

#[cfg(test)]
mod tests {
    use super::*;
    use vectordb::Metric;

    /// Two tight clusters in 8-d; ids 0..3 near center A, 4..7 near center B.
    fn clustered() -> Vec<(u64, Vec<f32>)> {
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
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
    fn related_stays_within_cluster() {
        let items = clustered();
        let g = GraphBuilder::new(Metric::Cosine, 3).build(&items, &[]);
        // Every related note of a cluster-A node is also in cluster A.
        for id in 0..4u64 {
            for nb in g.related(id, 3) {
                assert!(nb.id < 4, "id {id} linked across clusters to {}", nb.id);
            }
        }
    }

    #[test]
    fn roundtrip_bytes() {
        let items = clustered();
        let g = GraphBuilder::new(Metric::Cosine, 3).build(&items, &[(0, 5)]);
        let g2 = GraphStore::from_bytes(&g.to_bytes()).unwrap();
        assert_eq!(g2.len(), g.len());
        assert_eq!(g2.edge_count(), g.edge_count());
        for &id in g.ids() {
            let a: Vec<u64> = g.neighbors(id).iter().map(|n| n.id).collect();
            let b: Vec<u64> = g2.neighbors(id).iter().map(|n| n.id).collect();
            assert_eq!(a, b);
        }
        // The explicit cross-cluster link 0->5 exists and is a Link/Both edge.
        assert!(g2
            .neighbors(0)
            .iter()
            .any(|n| n.id == 5 && matches!(n.kind, EdgeKind::Link | EdgeKind::Both)));
    }

    #[test]
    fn mutual_knn_is_subset() {
        let items = clustered();
        let full = GraphBuilder::new(Metric::Cosine, 4).build(&items, &[]);
        let mut b = GraphBuilder::new(Metric::Cosine, 4);
        b.mutual = true;
        let mutual = b.build(&items, &[]);
        assert!(mutual.edge_count() <= full.edge_count());
    }

    #[test]
    fn neighborhood_and_communities() {
        let items = clustered();
        let g = GraphBuilder::new(Metric::Cosine, 3).build(&items, &[]);
        let sub = g.neighborhood(0, 2, 10, |_| true);
        assert!(sub.nodes.contains(&0));
        assert!(sub.nodes.len() >= 2);

        let comms = analytics::communities_label_propagation(&g, 20);
        assert_eq!(comms.len(), items.len());
        // The two clusters should land in different communities.
        assert_ne!(comms[0], comms[7]);

        let json = g.export_json(Some(&comms));
        assert!(json.starts_with("{\"nodes\":["));
        assert!(json.contains("\"community\":"));
    }

    #[test]
    fn tags_and_metadata() {
        let items = clustered();
        let mut g = GraphBuilder::new(Metric::Cosine, 3).build(&items, &[]);
        // Cluster A -> tag "a", cluster B -> tag "b"; a couple share "shared".
        let meta = (0..8u64).map(|id| {
            let mut tags = vec![if id < 4 { "a" } else { "b" }.to_string()];
            if id % 4 == 0 {
                tags.push("shared".into());
            }
            (
                id,
                NodeMeta {
                    title: format!("note {id}"),
                    tags,
                },
            )
        });
        g.set_metadata(meta);

        assert_eq!(g.title(3), Some("note 3"));
        assert!(g.has_tag(0, "a") && g.has_tag(0, "shared"));
        assert_eq!(g.nodes_with_tag("shared").len(), 2); // ids 0 and 4
        assert_eq!(g.nodes_with_tag("a").len(), 4);
        assert!(g.tag_counts().iter().any(|(t, c)| *t == "a" && *c == 4));

        // related restricted to tag "a" never leaves cluster A.
        let allow: std::collections::HashSet<u64> = g.nodes_with_tag("a").into_iter().collect();
        for nb in g.related_filter(1, 5, |id| allow.contains(&id)) {
            assert!(nb.id < 4);
        }

        // Metadata survives a byte round-trip; export carries tags.
        let g2 = GraphStore::from_bytes(&g.to_bytes()).unwrap();
        assert_eq!(g2.title(3), Some("note 3"));
        assert_eq!(g2.tags(0).len(), 2);
        assert_eq!(g2.nodes_with_tag("b").len(), 4);
        assert!(g2.export_json(None).contains("\"tags\":[\"a\",\"shared\"]"));
    }
}

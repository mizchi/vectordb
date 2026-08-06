//! Incremental / live graph updates: insert notes one at a time, query related,
//! then *edit* a note (re-embed) and *delete* one — touching only the affected
//! edges, no full rebuild. Freeze to a `GraphStore` (and `.graphdb`) whenever
//! you need to query or persist.
//!
//!   cargo run -p meandb-graph --example live_update

use meandb_graph::{GraphIndex, NodeMeta};
use meandb_vector::Metric;

fn topic_vec(topic: usize, jitter: f32, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|d| {
            let base = if d % 5 == topic { 1.0 } else { 0.0 };
            base + jitter * (d as f32).sin()
        })
        .collect()
}

fn main() {
    let dim = 20;
    let mut gi = GraphIndex::new(Metric::Cosine, 4);
    gi.min_weight = 0.5; // only keep meaningfully-similar edges

    // 15 notes across 5 topics, streamed in.
    for i in 0..15u64 {
        let topic = (i % 5) as usize;
        let meta = NodeMeta {
            title: format!("note {i}"),
            tags: vec![["rust", "graph", "search", "wasm", "misc"][topic].into()],
        };
        gi.insert(i, &topic_vec(topic, 0.02, dim), meta);
    }
    println!("inserted {} notes", gi.len());

    let show = |gi: &GraphIndex, id: u64, msg: &str| {
        let g = gi.freeze();
        let rel: Vec<String> = g
            .related(id, 3)
            .iter()
            .map(|n| format!("{}({})", n.id, g.tags(n.id).join("|")))
            .collect();
        println!("{msg}: related({id}) = [{}]", rel.join(", "));
    };

    // note 0 is topic "rust" (ids 0,5,10) — related should be its topic-mates.
    show(&gi, 0, "initial");

    // EDIT: re-embed note 0 into topic "wasm" (ids 3,8,13) + retag.
    gi.insert(
        0,
        &topic_vec(3, 0.02, dim),
        NodeMeta {
            title: "note 0 (moved to wasm)".into(),
            tags: vec!["wasm".into()],
        },
    );
    show(&gi, 0, "after edit");

    // DELETE note 5 (a former topic-mate of 0); it disappears from all edges.
    gi.remove(5);
    println!("after remove(5): {} notes", gi.len());
    let g = gi.freeze();
    let has5 = g.ids().contains(&5)
        || g.ids()
            .iter()
            .any(|&id| g.neighbors(id).iter().any(|n| n.id == 5));
    println!("node 5 still referenced anywhere: {has5}");

    // Persist the current snapshot.
    let path = std::env::temp_dir().join("live_kb.graphdb");
    g.save(&path).unwrap();
    println!(
        "saved {} nodes / {} edges -> {}",
        g.len(),
        g.edge_count(),
        path.display()
    );
    std::fs::remove_file(&path).ok();
}

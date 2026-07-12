//! Build a semantic knowledge graph over synthetic clustered "notes" and print
//! the graph-view JSON (nodes with degree + community, weighted edges) plus a
//! "related notes" example — the two things an Obsidian-like UI needs.
//!
//!   cargo run -p graphdb --example graph_view

use graphdb::{analytics, GraphBuilder};
use vectordb::Metric;

fn main() {
    // 30 notes in 5 topical clusters, 32-d embeddings.
    let (n, dim, clusters) = (30usize, 32usize, 5usize);
    let mut s: u64 = 0xDEAD_BEEF_1234_5678;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32
    };
    let centers: Vec<Vec<f32>> = (0..clusters)
        .map(|_| (0..dim).map(|_| rnd() * 2.0 - 1.0).collect())
        .collect();
    let items: Vec<(u64, Vec<f32>)> = (0..n)
        .map(|i| {
            let c = &centers[i % clusters];
            (
                i as u64,
                c.iter().map(|&x| x + (rnd() - 0.5) * 0.2).collect(),
            )
        })
        .collect();

    // A couple of explicit "wikilinks" across clusters.
    let links = [(0u64, 1u64), (0, 6)];

    let mut b = GraphBuilder::new(Metric::Cosine, 4);
    b.mutual = true; // cleaner global view
    let mut g = b.build(&items, &links);

    // Attach metadata: a title and a per-cluster tag on every note.
    g.set_metadata(items.iter().map(|(id, _)| {
        let topic = ["rust", "graph", "search", "wasm", "misc"][*id as usize % clusters];
        (
            *id,
            graphdb::NodeMeta {
                title: format!("note {id}"),
                tags: vec![topic.to_string()],
            },
        )
    }));

    println!("nodes={} edges={}", g.len(), g.edge_count());
    let comms = analytics::communities_label_propagation(&g, 20);
    let ncomm = comms.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    println!("communities={ncomm}");
    println!("tags: {:?}", g.tag_counts());

    println!("\nrelated to note 0 (same cluster = 0,5,10,15,20,25):");
    for nb in g.related(0, 5) {
        let t = g.tags(nb.id).join(",");
        println!("  {} w={:.3} ({}) [{t}]", nb.id, nb.weight, nb.kind.label());
    }

    println!("\nrelated to note 0, restricted to tag \"rust\":");
    let rust: std::collections::HashSet<u64> = g.nodes_with_tag("rust").into_iter().collect();
    for nb in g.related_filter(0, 5, |id| rust.contains(&id)) {
        println!("  {} w={:.3}", nb.id, nb.weight);
    }

    println!("\ngraph-view JSON:");
    println!("{}", g.export_json(Some(&comms)));
}

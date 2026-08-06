//! Compile-and-run contract for external crate consumers.
//!
//! The packages are namespaced on crates.io, but these are the intentionally
//! stable Rust import names.

use meandb::{
    graph::{EdgeKind, EdgePattern, GraphBuilder, NodePattern, Query},
    vector::Metric,
};

fn main() {
    let items = vec![
        (1_u64, vec![1.0_f32, 0.0, 0.0]),
        (2, vec![0.9, 0.1, 0.0]),
        (3, vec![0.0, 1.0, 0.0]),
    ];

    let graph = GraphBuilder::new(Metric::Cosine, 2).build(&items, &[(1, 2)]);
    assert_eq!(graph.related(1, 1)[0].id, 2);

    let rows = Query::select(["source", "target"])
        .node(NodePattern::new("source").id(1))
        .edge(EdgePattern::new("source", "target").kind(EdgeKind::Link))
        .limit(1)
        .execute(&graph)
        .expect("query should execute");
    assert_eq!(rows[0].get("target"), Some(2));

    let rows = Query::from_dsl(
        r#"
        FIND source, target
        WHERE source.id = 1
          AND source -[link]-> target
        LIMIT 1
        "#,
    )
    .expect("text DSL should parse")
    .execute(&graph)
    .expect("text DSL query should execute");
    assert_eq!(rows[0].get("target"), Some(2));
}

use meandb_graph::{GraphBuilder, NodeMeta, QueryExecution};
use meandb_vector::Metric;
use std::process::Command;

#[test]
fn cli_executes_a_text_dsl_query_file() {
    let items = vec![
        (1_u64, vec![1.0_f32, 0.0]),
        (2, vec![0.9, 0.1]),
        (3, vec![0.0, 1.0]),
    ];
    let mut graph = GraphBuilder::new(Metric::Cosine, 1).build(&items, &[(1, 2), (1, 3)]);
    graph.set_metadata([
        (
            1,
            NodeMeta {
                title: "Purchase".into(),
                tags: vec!["spec".into()],
            },
        ),
        (
            2,
            NodeMeta {
                title: "Stock cannot be negative".into(),
                tags: vec!["invariant".into()],
            },
        ),
        (
            3,
            NodeMeta {
                title: "Guide".into(),
                tags: vec!["guide".into()],
            },
        ),
    ]);
    let temp = std::env::temp_dir().join(format!(
        "meandb-query-cli-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    std::fs::create_dir_all(&temp).unwrap();
    let graph_path = temp.join("knowledge.graphdb");
    let query_path = temp.join("query.gql");
    graph.save(&graph_path).unwrap();
    std::fs::write(
        &query_path,
        r#"
        FIND source, target
        WHERE source.tag = "spec"
          AND source -[link, weight >= 0.8]-> target
          AND target.tag = "invariant"
        LIMIT 20
        "#,
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_meandb"))
        .args([
            "query-dsl",
            graph_path.to_str().unwrap(),
            query_path.to_str().unwrap(),
            "--explain",
        ])
        .output()
        .unwrap();

    std::fs::remove_dir_all(&temp).unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: QueryExecution = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0].row.get("source"), Some(1));
    assert_eq!(result.rows[0].row.get("target"), Some(2));
    assert!(result.rows[0].evidence.iter().any(|entry| entry.is_edge()));
}

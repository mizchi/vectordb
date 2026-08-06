use meandb_graph::{
    EdgeKind, EdgePattern, GraphBuilder, NodeMeta, NodePattern, Query, QueryBudget, QueryError,
    QUERY_VERSION,
};
use meandb_vector::Metric;

fn graph() -> meandb_graph::GraphStore {
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
    graph
}

#[test]
fn joins_node_and_link_patterns() {
    let query = Query::select(["source", "target"])
        .node(NodePattern::new("source").tag("spec"))
        .edge(EdgePattern::new("source", "target").kind(EdgeKind::Link))
        .node(NodePattern::new("target").tag("invariant"))
        .limit(20);

    let rows = query.execute(&graph()).unwrap();

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("source"), Some(1));
    assert_eq!(rows[0].get("target"), Some(2));
}

#[test]
fn preserves_the_query_contract_as_json() {
    let query = Query::select(["source", "target"])
        .node(NodePattern::new("source").tag("spec"))
        .edge(EdgePattern::new("source", "target").min_weight(0.8))
        .limit(3);

    let json = query.to_json().unwrap();
    let decoded = Query::from_json(&json).unwrap();

    assert_eq!(decoded, query);
}

#[test]
fn executes_a_json_query_when_the_edge_source_is_not_bound_yet() {
    let query = Query::from_json(
        r#"{
          "select": ["source", "target"],
          "clauses": [
            {"type": "node", "var": "target", "tags": ["invariant"]},
            {"type": "edge", "from": "source", "to": "target", "kinds": ["link"]}
          ],
          "limit": 10
        }"#,
    )
    .unwrap();

    let rows = query.execute(&graph()).unwrap();

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("source"), Some(1));
    assert_eq!(rows[0].get("target"), Some(2));
}

#[test]
fn zero_limit_returns_no_rows() {
    let query = Query::select(["source"])
        .node(NodePattern::new("source").tag("spec"))
        .limit(0);

    assert!(query.execute(&graph()).unwrap().is_empty());
}

#[test]
fn one_variable_on_both_ends_requires_a_self_edge() {
    let query = Query::select(["node"])
        .edge(EdgePattern::new("node", "node"))
        .limit(10);

    assert!(query.execute(&graph()).unwrap().is_empty());
}

#[test]
fn parses_and_executes_the_text_dsl() {
    let query = Query::from_dsl(
        r#"
        FIND source, target
        WHERE source.tag = "spec"
          AND source -[link, weight >= 0.8]-> target
          AND target.tag = "invariant"
        LIMIT 20
        "#,
    )
    .unwrap();

    let rows = query.execute(&graph()).unwrap();

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("source"), Some(1));
    assert_eq!(rows[0].get("target"), Some(2));
}

#[test]
fn returns_versioned_execution_evidence_with_a_bounded_plan() {
    let query = Query::select(["source", "target"])
        .node(NodePattern::new("source").tag("spec"))
        .edge(EdgePattern::new("source", "target").kind(EdgeKind::Link))
        .node(NodePattern::new("target").tag("invariant"))
        .budget(QueryBudget {
            max_intermediate_rows: 10,
            max_edges_scanned: 10,
        });

    let result = query.execute_with_explain(&graph()).unwrap();

    assert_eq!(result.query_version, QUERY_VERSION);
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0].row.get("source"), Some(1));
    assert!(result.rows[0].evidence.iter().any(|entry| entry.is_edge()));
    assert_eq!(result.plan.clauses.len(), 3);
    assert!(result.plan.edges_scanned > 0);
}

#[test]
fn stops_when_the_edge_scan_budget_is_exhausted() {
    let query = Query::select(["source", "target"])
        .edge(EdgePattern::new("source", "target"))
        .budget(QueryBudget {
            max_intermediate_rows: 10,
            max_edges_scanned: 0,
        });

    assert_eq!(
        query.execute(&graph()).unwrap_err(),
        QueryError::EdgeScanLimitExceeded { limit: 0 }
    );
}

#[test]
fn rejects_an_unknown_serialized_query_version() {
    let query = Query::from_json(
        r#"{
          "version": 2,
          "select": ["node"],
          "clauses": [{"type": "node", "var": "node", "id": 1}],
          "limit": 1
        }"#,
    )
    .unwrap();

    assert_eq!(
        query.execute(&graph()).unwrap_err(),
        QueryError::UnsupportedVersion {
            expected: QUERY_VERSION,
            actual: 2,
        }
    );
}

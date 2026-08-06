//! CLI for meandb's graph layer: build a semantic graph from embeddings (+ optional links
//! and node metadata), query related notes, extract a local neighborhood, list
//! tags, or export the whole graph as JSON.
//!
//! Reuses `meandb_vector::cli` argument/CSV helpers so the two tools feel the same.
//!
//! Usage:
//!   meandb build-graph <vecs.csv> <out.graphdb> [--metric l2|dot|cosine]
//!             [--k N] [--ef N] [--min-weight W] [--mutual]
//!             [--links links.csv] [--meta meta.tsv]
//!   meandb related      <g.graphdb> <id> [-k N] [--tag T]
//!   meandb neighborhood <g.graphdb> <id> [--depth D] [--max N] [--tag T]
//!   meandb tags         <g.graphdb>
//!   meandb by-tag       <g.graphdb> <tag>
//!   meandb query        <g.graphdb> <query.json> [--explain]
//!   meandb query-dsl    <g.graphdb> <query.gql> [--explain]
//!   meandb export       <g.graphdb> [--communities]
//!   meandb info         <g.graphdb>
//!
//! Vectors CSV: `id,v0,v1,...`. Links CSV: `src_id,dst_id`.
//! Meta TSV:    `id<TAB>title<TAB>tag1,tag2,...` (title/tags optional).

use meandb_graph::{
    analytics, GraphBuilder, GraphStore, NodeMeta, Query, SuggestOpts, TagClassifier,
};
use meandb_vector::cli::{flag_value, has_flag, parse_csv, parse_flag, parse_metric};
use meandb_vector::Metric;
use std::collections::HashSet;
use std::process::ExitCode;

pub fn entrypoint() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str);
    let rest = &args[args.len().min(1)..];
    let r = match cmd {
        Some("build-graph") => build_graph(rest),
        Some("related") => related(rest),
        Some("neighborhood") => neighborhood(rest),
        Some("tags") => tags(rest),
        Some("by-tag") => by_tag(rest),
        Some("query") => query(rest),
        Some("query-dsl") => query_dsl(rest),
        Some("suggest-tags") => suggest_tags(rest),
        Some("export") => export(rest),
        Some("info") => info(rest),
        _ => Err(
            "usage: meandb <build-graph|related|neighborhood|tags|by-tag|query|query-dsl|suggest-tags|export|info> ... (compatibility alias: graphdb)"
                .into(),
        ),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn read_records(path: &str) -> Result<Vec<(u64, Vec<f32>)>, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    parse_csv(&bytes)
}

fn read_links(path: &str) -> Result<Vec<(u64, u64)>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        let mut it = l.split(',');
        let a = it.next().and_then(|s| s.trim().parse().ok());
        let b = it.next().and_then(|s| s.trim().parse().ok());
        match (a, b) {
            (Some(a), Some(b)) => out.push((a, b)),
            _ => return Err(format!("links line {}: expected `src,dst`", i + 1)),
        }
    }
    Ok(out)
}

/// Meta TSV: `id<TAB>title<TAB>tag1,tag2,...` (title and tags optional).
fn read_meta(path: &str) -> Result<Vec<(u64, NodeMeta)>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let l = line.trim_end_matches(['\r', '\n']);
        if l.trim().is_empty() || l.starts_with('#') {
            continue;
        }
        let mut cols = l.split('\t');
        let id: u64 = cols
            .next()
            .and_then(|s| s.trim().parse().ok())
            .ok_or_else(|| format!("meta line {}: bad id", i + 1))?;
        let title = cols.next().unwrap_or("").trim().to_string();
        let tags = cols
            .next()
            .map(|s| {
                s.split(',')
                    .map(str::trim)
                    .filter(|t| !t.is_empty())
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_default();
        out.push((id, NodeMeta { title, tags }));
    }
    Ok(out)
}

fn build_graph(args: &[String]) -> Result<(), String> {
    if args.len() < 2 {
        return Err("build-graph <vecs.csv> <out.graphdb> [...]".into());
    }
    let (input, output) = (&args[0], &args[1]);
    let metric: Metric = flag_value(args, "--metric").map_or(Ok(Metric::Cosine), parse_metric)?;
    let items = read_records(input)?;
    if items.is_empty() {
        return Err("no records".into());
    }
    let links = match flag_value(args, "--links") {
        Some(p) => read_links(p)?,
        None => Vec::new(),
    };
    let mut b = GraphBuilder::new(metric, parse_flag(args, "--k", 10)?);
    b.ef = parse_flag(args, "--ef", 64)?;
    b.min_weight = parse_flag(args, "--min-weight", f32::MIN)?;
    b.mutual = has_flag(args, "--mutual");
    let mut g = b.build(&items, &links);
    let mut tag_note = String::new();
    if let Some(p) = flag_value(args, "--meta") {
        let meta = read_meta(p)?;
        g.set_metadata(meta);
        tag_note = format!(", {} tags", g.tag_names().len());
    }
    g.save(output).map_err(|e| e.to_string())?;
    println!(
        "built graph: {} nodes, {} edges (k {}, {metric:?}{}{tag_note}) -> {output}",
        g.len(),
        g.edge_count(),
        b.k,
        if b.mutual { ", mutual" } else { "" },
    );
    Ok(())
}

/// Build a keep-predicate from an optional `--tag` filter.
fn tag_filter(g: &GraphStore, args: &[String]) -> Option<HashSet<u64>> {
    flag_value(args, "--tag").map(|t| g.nodes_with_tag(t).into_iter().collect())
}

fn related(args: &[String]) -> Result<(), String> {
    if args.len() < 2 {
        return Err("related <g.graphdb> <id> [-k N] [--tag T]".into());
    }
    let g = GraphStore::load(&args[0]).map_err(|e| e.to_string())?;
    let id: u64 = args[1].parse().map_err(|_| "bad id")?;
    let k: usize = parse_flag(args, "-k", 10)?;
    let allow = tag_filter(&g, args);
    let keep = |x: u64| allow.as_ref().is_none_or(|s| s.contains(&x));
    for (rank, nb) in g.related_filter(id, k, keep).iter().enumerate() {
        let title = g.title(nb.id).unwrap_or("");
        println!(
            "{rank}\t{}\t{:.4}\t{}\t{}",
            nb.id,
            nb.weight,
            nb.kind.label(),
            title
        );
    }
    Ok(())
}

fn neighborhood(args: &[String]) -> Result<(), String> {
    if args.len() < 2 {
        return Err("neighborhood <g.graphdb> <id> [--depth D] [--max N] [--tag T]".into());
    }
    let g = GraphStore::load(&args[0]).map_err(|e| e.to_string())?;
    let id: u64 = args[1].parse().map_err(|_| "bad id")?;
    let depth: usize = parse_flag(args, "--depth", 2)?;
    let max: usize = parse_flag(args, "--max", 50)?;
    let allow = tag_filter(&g, args);
    let keep = |x: u64| allow.as_ref().is_none_or(|s| s.contains(&x));
    let sub = g.neighborhood(id, depth, max, keep);
    println!("# {} nodes, {} edges", sub.nodes.len(), sub.edges.len());
    for (s, d, w, kind) in &sub.edges {
        println!("{s}\t{d}\t{w:.4}\t{}", kind.label());
    }
    Ok(())
}

fn tags(args: &[String]) -> Result<(), String> {
    let path = args.first().ok_or("tags <g.graphdb>")?;
    let g = GraphStore::load(path).map_err(|e| e.to_string())?;
    for (tag, count) in g.tag_counts() {
        println!("{count}\t{tag}");
    }
    Ok(())
}

fn by_tag(args: &[String]) -> Result<(), String> {
    if args.len() < 2 {
        return Err("by-tag <g.graphdb> <tag>".into());
    }
    let g = GraphStore::load(&args[0]).map_err(|e| e.to_string())?;
    for id in g.nodes_with_tag(&args[1]) {
        println!("{id}\t{}", g.title(id).unwrap_or(""));
    }
    Ok(())
}

/// Execute a serialized graphdb query and print its projected bindings as JSON.
fn query(args: &[String]) -> Result<(), String> {
    if args.len() != 2 && !(args.len() == 3 && args[2] == "--explain") {
        return Err("query <g.graphdb> <query.json> [--explain]".into());
    }
    let graph = GraphStore::load(&args[0]).map_err(|err| err.to_string())?;
    let json = std::fs::read_to_string(&args[1]).map_err(|err| err.to_string())?;
    let query = Query::from_json(&json).map_err(|err| err.to_string())?;
    if args.get(2).is_some_and(|arg| arg == "--explain") {
        let result = query
            .execute_with_explain(&graph)
            .map_err(|err| err.to_string())?;
        println!(
            "{}",
            serde_json::to_string(&result).map_err(|err| err.to_string())?
        );
    } else {
        let rows = query.execute(&graph).map_err(|err| err.to_string())?;
        println!(
            "{}",
            serde_json::to_string(&rows).map_err(|err| err.to_string())?
        );
    }
    Ok(())
}

/// Parse a text DSL query, execute it, and print its projected bindings as JSON.
fn query_dsl(args: &[String]) -> Result<(), String> {
    if args.len() != 2 && !(args.len() == 3 && args[2] == "--explain") {
        return Err("query-dsl <g.graphdb> <query.gql> [--explain]".into());
    }
    let graph = GraphStore::load(&args[0]).map_err(|err| err.to_string())?;
    let text = std::fs::read_to_string(&args[1]).map_err(|err| err.to_string())?;
    let query = Query::from_dsl(&text).map_err(|err| err.to_string())?;
    if args.get(2).is_some_and(|arg| arg == "--explain") {
        let result = query
            .execute_with_explain(&graph)
            .map_err(|err| err.to_string())?;
        println!(
            "{}",
            serde_json::to_string(&result).map_err(|err| err.to_string())?
        );
    } else {
        let rows = query.execute(&graph).map_err(|err| err.to_string())?;
        println!(
            "{}",
            serde_json::to_string(&rows).map_err(|err| err.to_string())?
        );
    }
    Ok(())
}

/// Auto-classify new notes: build a classifier from labeled embeddings
/// (`vecs.csv` + `meta.tsv`) and suggest tags for each query embedding.
fn suggest_tags(args: &[String]) -> Result<(), String> {
    if args.len() < 3 {
        return Err(
            "suggest-tags <vecs.csv> <meta.tsv> <query.csv> [--k N] [--min-score S] [--max-tags N]"
                .into(),
        );
    }
    let metric: Metric = flag_value(args, "--metric").map_or(Ok(Metric::Cosine), parse_metric)?;
    let items = read_records(&args[0])?;
    let meta = read_meta(&args[1])?;
    let tag_of: std::collections::HashMap<u64, Vec<String>> =
        meta.into_iter().map(|(id, m)| (id, m.tags)).collect();
    let labeled: Vec<(u64, Vec<f32>, Vec<String>)> = items
        .into_iter()
        .map(|(id, v)| (id, v, tag_of.get(&id).cloned().unwrap_or_default()))
        .collect();
    let clf = TagClassifier::build(&labeled, metric);

    let opts = SuggestOpts {
        k: parse_flag(args, "--k", 10)?,
        max_tags: parse_flag(args, "--max-tags", 5)?,
        min_score: parse_flag(args, "--min-score", 0.15)?,
        ..Default::default()
    };
    let queries = read_records(&args[2])?;
    for (qid, v) in &queries {
        for s in clf.suggest(v, &opts) {
            println!("{qid}\t{}\t{:.3}\t{}", s.tag, s.score, s.votes);
        }
    }
    Ok(())
}

fn export(args: &[String]) -> Result<(), String> {
    let path = args.first().ok_or("export <g.graphdb> [--communities]")?;
    let g = GraphStore::load(path).map_err(|e| e.to_string())?;
    let comms = if has_flag(args, "--communities") {
        Some(analytics::communities_label_propagation(&g, 20))
    } else {
        None
    };
    println!("{}", g.export_json(comms.as_deref()));
    Ok(())
}

fn info(args: &[String]) -> Result<(), String> {
    let path = args.first().ok_or("info <g.graphdb>")?;
    let g = GraphStore::load(path).map_err(|e| e.to_string())?;
    let degs = analytics::degrees(&g);
    let avg = if g.is_empty() {
        0.0
    } else {
        degs.iter().sum::<usize>() as f32 / g.len() as f32
    };
    println!("path:     {path}");
    println!("nodes:    {}", g.len());
    println!("edges:    {}", g.edge_count());
    println!("directed: {}", g.is_directed());
    println!("metric:   {:?}", g.metric());
    println!("tags:     {}", g.tag_names().len());
    println!("avg-deg:  {avg:.1}");
    Ok(())
}

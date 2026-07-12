//! CLI for graphdb: build a semantic graph from embeddings, query related
//! notes, extract a local neighborhood, or export the whole graph as JSON.
//!
//! Reuses `vectordb::cli` argument/CSV helpers so the two tools feel the same.
//!
//! Usage:
//!   graphdb build-graph <vecs.csv> <out.graphdb> [--metric l2|dot|cosine]
//!                       [--k N] [--ef N] [--min-weight W] [--mutual] [--links links.csv]
//!   graphdb related      <g.graphdb> <id> [-k N]
//!   graphdb neighborhood <g.graphdb> <id> [--depth D] [--max N]
//!   graphdb export       <g.graphdb> [--communities]
//!   graphdb info         <g.graphdb>
//!
//! Vectors CSV: `id,v0,v1,...`. Links CSV: `src_id,dst_id` per line.

use graphdb::{analytics, GraphBuilder, GraphStore};
use std::process::ExitCode;
use vectordb::cli::{flag_value, has_flag, parse_csv, parse_flag, parse_metric};
use vectordb::Metric;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str);
    let rest = &args[args.len().min(1)..];
    let r = match cmd {
        Some("build-graph") => build_graph(rest),
        Some("related") => related(rest),
        Some("neighborhood") => neighborhood(rest),
        Some("export") => export(rest),
        Some("info") => info(rest),
        _ => Err("usage: graphdb <build-graph|related|neighborhood|export|info> ...".into()),
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

fn build_graph(args: &[String]) -> Result<(), String> {
    if args.len() < 2 {
        return Err("build-graph <vecs.csv> <out.graphdb> [...]".into());
    }
    let input = &args[0];
    let output = &args[1];
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
    let g = b.build(&items, &links);
    g.save(output).map_err(|e| e.to_string())?;
    println!(
        "built graph: {} nodes, {} edges (k {}, {metric:?}{}) -> {output}",
        g.len(),
        g.edge_count(),
        b.k,
        if b.mutual { ", mutual" } else { "" },
    );
    Ok(())
}

fn related(args: &[String]) -> Result<(), String> {
    if args.len() < 2 {
        return Err("related <g.graphdb> <id> [-k N]".into());
    }
    let g = GraphStore::load(&args[0]).map_err(|e| e.to_string())?;
    let id: u64 = args[1].parse().map_err(|_| "bad id")?;
    let k: usize = parse_flag(args, "-k", 10)?;
    for (rank, nb) in g.related(id, k).iter().enumerate() {
        println!("{rank}\t{}\t{:.4}\t{}", nb.id, nb.weight, nb.kind.label());
    }
    Ok(())
}

fn neighborhood(args: &[String]) -> Result<(), String> {
    if args.len() < 2 {
        return Err("neighborhood <g.graphdb> <id> [--depth D] [--max N]".into());
    }
    let g = GraphStore::load(&args[0]).map_err(|e| e.to_string())?;
    let id: u64 = args[1].parse().map_err(|_| "bad id")?;
    let depth: usize = parse_flag(args, "--depth", 2)?;
    let max: usize = parse_flag(args, "--max", 50)?;
    let sub = g.neighborhood(id, depth, max);
    println!("# {} nodes, {} edges", sub.nodes.len(), sub.edges.len());
    for (s, d, w, kind) in &sub.edges {
        println!("{s}\t{d}\t{w:.4}\t{}", kind.label());
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
    println!("{}", g.export_json(None, comms.as_deref()));
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
    println!("avg-deg:  {avg:.1}");
    Ok(())
}

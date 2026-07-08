//! Minimal CLI for the vectordb `.vecdb` format.
//!
//! Vectors are read/written as CSV-ish text: one record per line,
//! `id,v0,v1,...,vD-1`.
//!
//! Usage:
//!   vecdb build <in.csv> <out.vecdb> [--metric l2|dot|cosine] [--compact]
//!   vecdb search <index.vecdb> <query.csv> [-k N] [--oversample M]
//!   vecdb info <index.vecdb>

use std::process::ExitCode;
use vectordb::{save, FlatIndex, Metric};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str);
    let rest = &args[args.len().min(1)..];
    let result = match cmd {
        Some("build") => cmd_build(rest),
        Some("search") => cmd_search(rest),
        Some("info") => cmd_info(rest),
        _ => {
            eprintln!("usage: vecdb <build|search|info> ...");
            return ExitCode::FAILURE;
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn parse_metric(s: &str) -> Result<Metric, String> {
    match s {
        "l2" | "L2" => Ok(Metric::L2),
        "dot" | "ip" => Ok(Metric::Dot),
        "cosine" | "cos" => Ok(Metric::Cosine),
        _ => Err(format!("unknown metric: {s}")),
    }
}

fn read_records(path: &str) -> Result<Vec<(u64, Vec<f32>)>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split(',');
        let id: u64 = it
            .next()
            .ok_or("missing id")?
            .trim()
            .parse()
            .map_err(|_| format!("line {}: bad id", lineno + 1))?;
        let vec: Result<Vec<f32>, _> = it.map(|f| f.trim().parse::<f32>()).collect();
        let vec = vec.map_err(|_| format!("line {}: bad float", lineno + 1))?;
        out.push((id, vec));
    }
    Ok(out)
}

fn flag_value<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn cmd_build(args: &[String]) -> Result<(), String> {
    if args.len() < 2 {
        return Err("build <in.csv> <out.vecdb> [--metric m] [--compact]".into());
    }
    let input = &args[0];
    let output = &args[1];
    let metric = flag_value(args, "--metric").map_or(Ok(Metric::Cosine), parse_metric)?;
    let compact = args.iter().any(|a| a == "--compact");

    let records = read_records(input)?;
    if records.is_empty() {
        return Err("no records".into());
    }
    let dim = records[0].1.len();
    let mut idx = FlatIndex::new(dim, metric, !compact);
    for (id, v) in &records {
        if v.len() != dim {
            return Err(format!("id {id}: dim {} != {dim}", v.len()));
        }
        idx.add(*id, v);
    }
    save(&idx, output).map_err(|e| e.to_string())?;
    println!(
        "built {} vectors (dim {dim}, {metric:?}, {}) -> {output}",
        idx.len(),
        if compact { "compact/int8-only" } else { "int8+raw" }
    );
    Ok(())
}

fn cmd_search(args: &[String]) -> Result<(), String> {
    if args.len() < 2 {
        return Err("search <index.vecdb> <query.csv> [-k N] [--oversample M]".into());
    }
    let index_path = &args[0];
    let query_path = &args[1];
    let k: usize = flag_value(args, "-k").unwrap_or("10").parse().map_err(|_| "bad -k")?;
    let over: usize = flag_value(args, "--oversample")
        .unwrap_or("4")
        .parse()
        .map_err(|_| "bad --oversample")?;

    let m = vectordb::open(index_path).map_err(|e| e.to_string())?;
    let queries = read_records(query_path)?;
    for (qid, qv) in &queries {
        if qv.len() != m.dim() {
            return Err(format!("query {qid}: dim {} != {}", qv.len(), m.dim()));
        }
    }
    let qvs: Vec<Vec<f32>> = queries.iter().map(|(_, v)| v.clone()).collect();

    // Search all queries at once; the parallel build spreads them across cores.
    #[cfg(feature = "parallel")]
    let results = m.search_batch(&qvs, k, over);
    #[cfg(not(feature = "parallel"))]
    let results: Vec<_> = qvs.iter().map(|v| m.search(v, k, over)).collect();

    for ((qid, _), hits) in queries.iter().zip(results.iter()) {
        for (rank, h) in hits.iter().enumerate() {
            println!("{qid}\t{rank}\t{}\t{:.6}", h.id, h.score);
        }
    }
    Ok(())
}

fn cmd_info(args: &[String]) -> Result<(), String> {
    let path = args.first().ok_or("info <index.vecdb>")?;
    let m = vectordb::open(path).map_err(|e| e.to_string())?;
    println!("path:    {path}");
    println!("count:   {}", m.len());
    println!("dim:     {}", m.dim());
    println!("metric:  {:?}", m.metric());
    println!("raw f32: {}", m.has_raw());
    Ok(())
}

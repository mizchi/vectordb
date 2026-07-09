//! Minimal CLI for the vectordb `.vecdb` formats (Flat / IVF / HNSW).
//!
//! Vectors are read/written as CSV-ish text: one record per line,
//! `id,v0,v1,...,vD-1`.
//!
//! Usage:
//!   vecdb build      <in.csv> <out.vecdb> [--metric l2|dot|cosine] [--compact]
//!   vecdb build-ivf  <in.csv> <out.vecdb> [--metric m] [--nlist N] [--iters N]
//!   vecdb build-hnsw <in.csv> <out.vecdb> [--metric m] [-m M] [--ef-construction N]
//!   vecdb search     <index.vecdb> <query.csv> [-k N] [--oversample M]
//!                    [--nprobe N] [--ef N]   (index type auto-detected)
//!   vecdb info       <index.vecdb>

use std::io::Read;
use std::process::ExitCode;
use vectordb::{save, FlatIndex, Hit, HnswIndex, IvfIndex, Metric};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str);
    let rest = &args[args.len().min(1)..];
    let result = match cmd {
        Some("build") => cmd_build(rest),
        Some("build-ivf") => cmd_build_ivf(rest),
        Some("build-hnsw") => cmd_build_hnsw(rest),
        Some("search") => cmd_search(rest),
        Some("info") => cmd_info(rest),
        _ => {
            eprintln!("usage: vecdb <build|build-ivf|build-hnsw|search|info> ...");
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

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Flat,
    Ivf,
    Hnsw,
}

fn detect(path: &str) -> Result<Kind, String> {
    let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic).map_err(|e| e.to_string())?;
    match &magic {
        b"VECDB1\0\0" => Ok(Kind::Flat),
        b"VECDBIV1" => Ok(Kind::Ivf),
        b"VECDBHN1" => Ok(Kind::Hnsw),
        _ => Err("unrecognized index file (bad magic)".into()),
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

fn parse_flag<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> Result<T, String> {
    match flag_value(args, name) {
        Some(s) => s.parse().map_err(|_| format!("bad {name}")),
        None => Ok(default),
    }
}

type Record = (u64, Vec<f32>);

/// Read `<in.csv> <out>` plus the records, validating dimensions.
fn load_build_input(args: &[String]) -> Result<(String, String, usize, Vec<Record>), String> {
    if args.len() < 2 {
        return Err("expected <in.csv> <out.vecdb>".into());
    }
    let input = args[0].clone();
    let output = args[1].clone();
    let records = read_records(&input)?;
    if records.is_empty() {
        return Err("no records".into());
    }
    let dim = records[0].1.len();
    for (id, v) in &records {
        if v.len() != dim {
            return Err(format!("id {id}: dim {} != {dim}", v.len()));
        }
    }
    Ok((input, output, dim, records))
}

fn cmd_build(args: &[String]) -> Result<(), String> {
    let (_, output, dim, records) = load_build_input(args)?;
    let metric = flag_value(args, "--metric").map_or(Ok(Metric::Cosine), parse_metric)?;
    let compact = args.iter().any(|a| a == "--compact");
    let mut idx = FlatIndex::new(dim, metric, !compact);
    for (id, v) in &records {
        idx.add(*id, v);
    }
    save(&idx, &output).map_err(|e| e.to_string())?;
    println!(
        "built flat: {} vectors (dim {dim}, {metric:?}, {}) -> {output}",
        idx.len(),
        if compact { "int8-only" } else { "int8+raw" }
    );
    Ok(())
}

fn cmd_build_ivf(args: &[String]) -> Result<(), String> {
    let (_, output, dim, records) = load_build_input(args)?;
    let metric = flag_value(args, "--metric").map_or(Ok(Metric::Cosine), parse_metric)?;
    let nlist: usize = parse_flag(args, "--nlist", 0)?; // 0 => ~sqrt(n)
    let iters: usize = parse_flag(args, "--iters", 12)?;
    let idx = IvfIndex::build(dim, metric, nlist, &records, true, iters);
    idx.save(&output).map_err(|e| e.to_string())?;
    println!(
        "built ivf: {} vectors (dim {dim}, {metric:?}, nlist {}) -> {output}",
        idx.len(),
        idx.nlist()
    );
    Ok(())
}

fn cmd_build_hnsw(args: &[String]) -> Result<(), String> {
    let (_, output, dim, records) = load_build_input(args)?;
    let metric = flag_value(args, "--metric").map_or(Ok(Metric::Cosine), parse_metric)?;
    let m: usize = parse_flag(args, "-m", 16)?;
    let efc: usize = parse_flag(args, "--ef-construction", 200)?;
    let mut idx = HnswIndex::new(dim, metric, m, efc);
    for (id, v) in &records {
        idx.add(*id, v);
    }
    idx.save(&output).map_err(|e| e.to_string())?;
    println!("built hnsw: {} vectors (dim {dim}, {metric:?}, M {m}) -> {output}", idx.len());
    Ok(())
}

fn cmd_search(args: &[String]) -> Result<(), String> {
    if args.len() < 2 {
        return Err("search <index.vecdb> <query.csv> [-k N] [--oversample M] [--nprobe N] [--ef N]".into());
    }
    let index_path = &args[0];
    let query_path = &args[1];
    let k: usize = parse_flag(args, "-k", 10)?;
    let over: usize = parse_flag(args, "--oversample", 4)?;
    let nprobe: usize = parse_flag(args, "--nprobe", 16)?;
    let ef: usize = parse_flag(args, "--ef", 64)?;

    let queries = read_records(query_path)?;
    let qvs: Vec<Vec<f32>> = queries.iter().map(|(_, v)| v.clone()).collect();

    let results: Vec<Vec<Hit>> = match detect(index_path)? {
        Kind::Flat => {
            let m = vectordb::open(index_path).map_err(|e| e.to_string())?;
            check_dims(&qvs, m.dim())?;
            #[cfg(feature = "parallel")]
            {
                m.search_batch(&qvs, k, over)
            }
            #[cfg(not(feature = "parallel"))]
            {
                qvs.iter().map(|v| m.search(v, k, over)).collect()
            }
        }
        Kind::Ivf => {
            let idx = IvfIndex::load(index_path).map_err(|e| e.to_string())?;
            check_dims(&qvs, idx.dim())?;
            qvs.iter().map(|v| idx.search(v, k, nprobe, over)).collect()
        }
        Kind::Hnsw => {
            let idx = HnswIndex::load(index_path).map_err(|e| e.to_string())?;
            check_dims(&qvs, idx.dim())?;
            qvs.iter().map(|v| idx.search(v, k, ef)).collect()
        }
    };

    for ((qid, _), hits) in queries.iter().zip(results.iter()) {
        for (rank, h) in hits.iter().enumerate() {
            println!("{qid}\t{rank}\t{}\t{:.6}", h.id, h.score);
        }
    }
    Ok(())
}

fn check_dims(qvs: &[Vec<f32>], dim: usize) -> Result<(), String> {
    for v in qvs {
        if v.len() != dim {
            return Err(format!("query dim {} != index dim {dim}", v.len()));
        }
    }
    Ok(())
}

fn cmd_info(args: &[String]) -> Result<(), String> {
    let path = args.first().ok_or("info <index.vecdb>")?;
    match detect(path)? {
        Kind::Flat => {
            let m = vectordb::open(path).map_err(|e| e.to_string())?;
            println!("path:    {path}");
            println!("type:    flat");
            println!("count:   {}", m.len());
            println!("dim:     {}", m.dim());
            println!("metric:  {:?}", m.metric());
            println!("raw f32: {}", m.has_raw());
        }
        Kind::Ivf => {
            let idx = IvfIndex::load(path).map_err(|e| e.to_string())?;
            println!("path:    {path}");
            println!("type:    ivf");
            println!("count:   {}", idx.len());
            println!("dim:     {}", idx.dim());
            println!("metric:  {:?}", idx.metric());
            println!("nlist:   {}", idx.nlist());
            println!("raw f32: {}", idx.has_raw());
        }
        Kind::Hnsw => {
            let idx = HnswIndex::load(path).map_err(|e| e.to_string())?;
            println!("path:    {path}");
            println!("type:    hnsw");
            println!("count:   {}", idx.len());
            println!("dim:     {}", idx.dim());
            println!("metric:  {:?}", idx.metric());
        }
    }
    Ok(())
}

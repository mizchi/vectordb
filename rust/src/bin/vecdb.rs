//! Minimal CLI for the vectordb `.vecdb` formats
//! (Flat / IVF / HNSW / int8-HNSW / PQ).
//!
//! Vectors are read/written as CSV-ish text: one record per line,
//! `id,v0,v1,...,vD-1`.
//!
//! Usage:
//!   vecdb build        <in.csv> <out.vecdb> [--metric l2|dot|cosine] [--compact]
//!   vecdb build-ivf    <in.csv> <out.vecdb> [--metric m] [--nlist N] [--iters N]
//!   vecdb build-hnsw   <in.csv> <out.vecdb> [--metric m] [-m M] [--ef-construction N]
//!   vecdb build-hnsw-q <in.csv> <out.vecdb> [--metric m] [-m M] [--ef-construction N] [--compact]
//!   vecdb build-pq     <in.csv> <out.vecdb> [--metric m] [--pq-m M] [--ksub K] [--iters N] [--compact]
//!   vecdb build-ivfpq  <in.csv> <out.vecdb> [--metric m] [--nlist N] [--pq-m M] [--ksub K] [--iters N] [--compact]
//!   vecdb build-opq    <in.csv> <out.vecdb> [--metric m] [--pq-m M] [--ksub K] [--iters N] [--opq-iters N] [--compact]
//!   vecdb build-diskann <in.csv> <out.vecdb> [--metric m] [-r R] [--l-build N] [--alpha A] [--pq-m M] [--ksub K] [--streaming [--sample N]]
//!   vecdb search       <index.vecdb> <query.csv> [-k N] [--oversample M]
//!                      [--nprobe N] [--ef N]   (index type auto-detected)
//!   vecdb info         <index.vecdb>

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::process::ExitCode;
use vectordb::{
    save, DiskAnnIndex, FlatIndex, Hit, HnswIndex, HnswQIndex, IvfIndex, IvfPqIndex, Metric,
    OpqIndex, PqIndex,
};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str);
    let rest = &args[args.len().min(1)..];
    let result = match cmd {
        Some("build") => cmd_build(rest),
        Some("build-ivf") => cmd_build_ivf(rest),
        Some("build-hnsw") => cmd_build_hnsw(rest),
        Some("build-hnsw-q") => cmd_build_hnsw_q(rest),
        Some("build-pq") => cmd_build_pq(rest),
        Some("build-ivfpq") => cmd_build_ivfpq(rest),
        Some("build-opq") => cmd_build_opq(rest),
        Some("build-diskann") => cmd_build_diskann(rest),
        Some("search") => cmd_search(rest),
        Some("info") => cmd_info(rest),
        _ => {
            eprintln!(
                "usage: vecdb <build|build-ivf|build-hnsw|build-hnsw-q|build-pq|build-ivfpq|build-opq|build-diskann|search|info> ..."
            );
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
    HnswQ,
    Pq,
    IvfPq,
    Opq,
    DiskAnn,
}

fn detect(path: &str) -> Result<Kind, String> {
    let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic).map_err(|e| e.to_string())?;
    match &magic {
        b"VECDB1\0\0" => Ok(Kind::Flat),
        b"VECDBIV1" => Ok(Kind::Ivf),
        b"VECDBHN1" => Ok(Kind::Hnsw),
        b"VECDBHQ1" => Ok(Kind::HnswQ),
        b"VECDBPQ1" => Ok(Kind::Pq),
        b"VECDBIP1" => Ok(Kind::IvfPq),
        b"VECDBOP1" => Ok(Kind::Opq),
        b"VECDBDA1" => Ok(Kind::DiskAnn),
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

/// Dimension of the first data record, without reading the whole file — used by
/// the streaming DiskANN build so it can size the index before consuming rows.
fn peek_csv_dim(path: &str) -> Result<usize, String> {
    let f = File::open(path).map_err(|e| e.to_string())?;
    for line in BufReader::new(f).lines() {
        let line = line.map_err(|e| e.to_string())?;
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let cols = t.split(',').count();
        if cols < 2 {
            return Err("first record has no vector".into());
        }
        return Ok(cols - 1);
    }
    Err("no records".into())
}

/// A lazy CSV record reader: yields `(id, Vec<f32>)` one line at a time so the
/// caller never holds the whole file in memory. On a malformed line it prints
/// the error and exits (an infallible iterator can't surface a `Result`).
struct CsvStream {
    lines: std::io::Lines<BufReader<File>>,
    lineno: usize,
}

impl CsvStream {
    fn open(path: &str) -> Result<Self, String> {
        let f = File::open(path).map_err(|e| e.to_string())?;
        Ok(Self {
            lines: BufReader::new(f).lines(),
            lineno: 0,
        })
    }
}

impl Iterator for CsvStream {
    type Item = (u64, Vec<f32>);
    fn next(&mut self) -> Option<(u64, Vec<f32>)> {
        loop {
            let line = self.lines.next()?;
            self.lineno += 1;
            let line = line.unwrap_or_else(|e| {
                eprintln!("error: read failed: {e}");
                std::process::exit(1);
            });
            let t = line.trim();
            if t.is_empty() || t.starts_with('#') {
                continue;
            }
            let mut it = t.split(',');
            let fail = |what: &str, lineno: usize| -> ! {
                eprintln!("error: line {lineno}: {what}");
                std::process::exit(1);
            };
            let id = it
                .next()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or_else(|| fail("bad id", self.lineno));
            let v: Vec<f32> = it
                .map(|f| f.trim().parse::<f32>())
                .collect::<Result<_, _>>()
                .unwrap_or_else(|_| fail("bad float", self.lineno));
            return Some((id, v));
        }
    }
}

fn flag_value<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
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
    println!(
        "built hnsw: {} vectors (dim {dim}, {metric:?}, M {m}) -> {output}",
        idx.len()
    );
    Ok(())
}

fn cmd_build_hnsw_q(args: &[String]) -> Result<(), String> {
    let (_, output, dim, records) = load_build_input(args)?;
    let metric = flag_value(args, "--metric").map_or(Ok(Metric::Cosine), parse_metric)?;
    let m: usize = parse_flag(args, "-m", 16)?;
    let efc: usize = parse_flag(args, "--ef-construction", 200)?;
    let compact = args.iter().any(|a| a == "--compact");
    let mut idx = HnswQIndex::new(dim, metric, m, efc, !compact);
    for (id, v) in &records {
        idx.add(*id, v);
    }
    idx.save(&output).map_err(|e| e.to_string())?;
    println!(
        "built hnsw-q (int8 graph): {} vectors (dim {dim}, {metric:?}, M {m}, {}) -> {output}",
        idx.len(),
        if compact {
            "int8-only"
        } else {
            "int8+raw rerank"
        }
    );
    Ok(())
}

fn cmd_build_pq(args: &[String]) -> Result<(), String> {
    let (_, output, dim, records) = load_build_input(args)?;
    let metric = flag_value(args, "--metric").map_or(Ok(Metric::Cosine), parse_metric)?;
    let pq_m: usize = parse_flag(args, "--pq-m", 16)?;
    let ksub: usize = parse_flag(args, "--ksub", 256)?;
    let iters: usize = parse_flag(args, "--iters", 20)?;
    let compact = args.iter().any(|a| a == "--compact");
    if !dim.is_multiple_of(pq_m) {
        return Err(format!("--pq-m {pq_m} must divide dim {dim}"));
    }
    let idx = PqIndex::build(&records, metric, pq_m, ksub, iters, !compact);
    idx.save(&output).map_err(|e| e.to_string())?;
    println!(
        "built pq: {} vectors (dim {dim}, {metric:?}, m {pq_m}, ksub {ksub}, {} bytes/vec) -> {output}",
        idx.len(),
        idx.code_bytes() / idx.len().max(1)
    );
    Ok(())
}

fn cmd_build_ivfpq(args: &[String]) -> Result<(), String> {
    let (_, output, dim, records) = load_build_input(args)?;
    let metric = flag_value(args, "--metric").map_or(Ok(Metric::Cosine), parse_metric)?;
    let nlist: usize = parse_flag(args, "--nlist", 0)?; // 0 => ~sqrt(n)
    let pq_m: usize = parse_flag(args, "--pq-m", 16)?;
    let ksub: usize = parse_flag(args, "--ksub", 256)?;
    let iters: usize = parse_flag(args, "--iters", 15)?;
    let compact = args.iter().any(|a| a == "--compact");
    if !dim.is_multiple_of(pq_m) {
        return Err(format!("--pq-m {pq_m} must divide dim {dim}"));
    }
    let idx = IvfPqIndex::build(&records, metric, nlist, pq_m, ksub, iters, !compact);
    idx.save(&output).map_err(|e| e.to_string())?;
    println!(
        "built ivfpq: {} vectors (dim {dim}, {metric:?}, nlist {}, m {pq_m}, {} bytes/vec) -> {output}",
        idx.len(),
        idx.nlist(),
        idx.code_bytes() / idx.len().max(1)
    );
    Ok(())
}

fn cmd_build_opq(args: &[String]) -> Result<(), String> {
    let (_, output, dim, records) = load_build_input(args)?;
    let metric = flag_value(args, "--metric").map_or(Ok(Metric::Cosine), parse_metric)?;
    let pq_m: usize = parse_flag(args, "--pq-m", 16)?;
    let ksub: usize = parse_flag(args, "--ksub", 256)?;
    let iters: usize = parse_flag(args, "--iters", 20)?;
    let opq_iters: usize = parse_flag(args, "--opq-iters", 4)?;
    let compact = args.iter().any(|a| a == "--compact");
    if !dim.is_multiple_of(pq_m) {
        return Err(format!("--pq-m {pq_m} must divide dim {dim}"));
    }
    let idx = OpqIndex::build(&records, metric, pq_m, ksub, iters, opq_iters, !compact);
    idx.save(&output).map_err(|e| e.to_string())?;
    println!(
        "built opq: {} vectors (dim {dim}, {metric:?}, m {pq_m}, ksub {ksub}, opq-iters {opq_iters}, {} bytes/vec) -> {output}",
        idx.len(),
        idx.code_bytes() / idx.len().max(1)
    );
    Ok(())
}

fn cmd_build_diskann(args: &[String]) -> Result<(), String> {
    let metric = flag_value(args, "--metric").map_or(Ok(Metric::Cosine), parse_metric)?;
    let r: usize = parse_flag(args, "-r", 32)?;
    let l_build: usize = parse_flag(args, "--l-build", 96)?;
    let alpha: f32 = parse_flag(args, "--alpha", 1.2)?;
    let pq_m: usize = parse_flag(args, "--pq-m", 16)?;
    let ksub: usize = parse_flag(args, "--ksub", 256)?;

    if has_flag(args, "--streaming") {
        // Low-memory build: the CSV is consumed lazily (one record at a time)
        // and the raw vectors are never all resident. `--sample` bounds the
        // PQ-training set.
        if args.len() < 2 {
            return Err("expected <in.csv> <out.vecdb>".into());
        }
        let input = &args[0];
        let output = &args[1];
        let dim = peek_csv_dim(input)?;
        if !dim.is_multiple_of(pq_m) {
            return Err(format!("--pq-m {pq_m} must divide dim {dim}"));
        }
        let sample: usize = parse_flag(args, "--sample", 50_000)?;
        DiskAnnIndex::build_streaming(
            output,
            dim,
            metric,
            r,
            l_build,
            alpha,
            pq_m,
            ksub,
            sample,
            CsvStream::open(input)?,
        )
        .map_err(|e| e.to_string())?;
        // `open` reads only the small resident tier (ids/codebooks/codes).
        let n = DiskAnnIndex::open(output).map_err(|e| e.to_string())?.len();
        println!(
            "built diskann (streaming): {n} vectors (dim {dim}, {metric:?}, R {r}, sample {sample}) -> {output}"
        );
        return Ok(());
    }

    let (_, output, dim, records) = load_build_input(args)?;
    if !dim.is_multiple_of(pq_m) {
        return Err(format!("--pq-m {pq_m} must divide dim {dim}"));
    }
    let idx = DiskAnnIndex::build(&records, metric, r, l_build, alpha, pq_m, ksub);
    idx.save(&output).map_err(|e| e.to_string())?;
    println!(
        "built diskann: {} vectors (dim {dim}, {metric:?}, R {r}, avg-deg {:.1}) -> {output}",
        idx.len(),
        idx.avg_degree()
    );
    Ok(())
}

fn cmd_search(args: &[String]) -> Result<(), String> {
    if args.len() < 2 {
        return Err(
            "search <index.vecdb> <query.csv> [-k N] [--oversample M] [--nprobe N] [--ef N]".into(),
        );
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
        Kind::HnswQ => {
            let idx = HnswQIndex::load(index_path).map_err(|e| e.to_string())?;
            check_dims(&qvs, idx.dim())?;
            qvs.iter().map(|v| idx.search(v, k, ef)).collect()
        }
        Kind::Pq => {
            let idx = PqIndex::load(index_path).map_err(|e| e.to_string())?;
            check_dims(&qvs, idx.dim())?;
            qvs.iter().map(|v| idx.search(v, k, over)).collect()
        }
        Kind::IvfPq => {
            let idx = IvfPqIndex::load(index_path).map_err(|e| e.to_string())?;
            check_dims(&qvs, idx.dim())?;
            qvs.iter().map(|v| idx.search(v, k, nprobe, over)).collect()
        }
        Kind::Opq => {
            let idx = OpqIndex::load(index_path).map_err(|e| e.to_string())?;
            check_dims(&qvs, idx.dim())?;
            qvs.iter().map(|v| idx.search(v, k, over)).collect()
        }
        Kind::DiskAnn => {
            // Disk-resident: graph + raw stay mmap'd, only PQ codes are in RAM.
            let idx = DiskAnnIndex::open(index_path).map_err(|e| e.to_string())?;
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
        Kind::HnswQ => {
            let idx = HnswQIndex::load(path).map_err(|e| e.to_string())?;
            println!("path:    {path}");
            println!("type:    hnsw-q (int8 graph)");
            println!("count:   {}", idx.len());
            println!("dim:     {}", idx.dim());
            println!("metric:  {:?}", idx.metric());
            println!("raw f32: {}", idx.has_raw());
        }
        Kind::Pq => {
            let idx = PqIndex::load(path).map_err(|e| e.to_string())?;
            println!("path:    {path}");
            println!("type:    pq");
            println!("count:   {}", idx.len());
            println!("dim:     {}", idx.dim());
            println!("metric:  {:?}", idx.metric());
            println!("raw f32: {}", idx.has_raw());
        }
        Kind::IvfPq => {
            let idx = IvfPqIndex::load(path).map_err(|e| e.to_string())?;
            println!("path:    {path}");
            println!("type:    ivfpq");
            println!("count:   {}", idx.len());
            println!("dim:     {}", idx.dim());
            println!("metric:  {:?}", idx.metric());
            println!("nlist:   {}", idx.nlist());
            println!("raw f32: {}", idx.has_raw());
        }
        Kind::Opq => {
            let idx = OpqIndex::load(path).map_err(|e| e.to_string())?;
            println!("path:    {path}");
            println!("type:    opq");
            println!("count:   {}", idx.len());
            println!("dim:     {}", idx.dim());
            println!("metric:  {:?}", idx.metric());
            println!("raw f32: {}", idx.has_raw());
        }
        Kind::DiskAnn => {
            let idx = DiskAnnIndex::load(path).map_err(|e| e.to_string())?;
            println!("path:    {path}");
            println!("type:    diskann (vamana)");
            println!("count:   {}", idx.len());
            println!("dim:     {}", idx.dim());
            println!("metric:  {:?}", idx.metric());
            println!("avg-deg: {:.1}", idx.avg_degree());
        }
    }
    Ok(())
}

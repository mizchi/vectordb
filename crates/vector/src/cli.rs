//! Environment-agnostic CLI core.
//!
//! All IO goes through the [`Env`] trait, so the exact same command logic runs
//! against a real filesystem ([`StdEnv`]), fully in memory ([`MemEnv`], for
//! tests / sandboxes / hosts without a filesystem), or any host adapter you
//! implement. Indexes are (de)serialized with the byte APIs (`to_bytes` /
//! `from_bytes`), never touching paths directly — the `Env` supplies bytes.
//!
//! ```
//! use meandb_vector::cli::{run, MemEnv};
//! let mut env = MemEnv::new();
//! env.put_str("vecs.csv", "0,1,0,0\n1,0,1,0\n2,0.9,0.1,0\n");
//! run(&mut env, &args(["build", "vecs.csv", "idx.vecdb", "--metric", "cosine"])).unwrap();
//! env.put_str("q.csv", "0,1,0.05,0\n");
//! run(&mut env, &args(["search", "idx.vecdb", "q.csv", "-k", "2"])).unwrap();
//! assert!(env.output().contains('\t'));
//! fn args<const N: usize>(a: [&str; N]) -> Vec<String> { a.iter().map(|s| s.to_string()).collect() }
//! ```

use crate::{
    DiskAnnIndex, FlatIndex, HnswIndex, HnswQIndex, IvfIndex, IvfPqIndex, Metric, OpqIndex, PqIndex,
};
use std::collections::HashMap;
use std::io;

/// Abstract IO for the CLI: read a file, write a file, print a line, print an
/// error line. Implement this to run the CLI in any environment.
pub trait Env {
    fn read(&self, path: &str) -> io::Result<Vec<u8>>;
    fn write(&mut self, path: &str, data: &[u8]) -> io::Result<()>;
    fn print(&mut self, line: &str);
    fn eprint(&mut self, line: &str);
}

/// Real-filesystem + stdio adapter (the default for the `vecdb` binary).
#[derive(Default)]
pub struct StdEnv;

impl StdEnv {
    pub fn new() -> Self {
        StdEnv
    }
}

impl Env for StdEnv {
    fn read(&self, path: &str) -> io::Result<Vec<u8>> {
        std::fs::read(path)
    }
    fn write(&mut self, path: &str, data: &[u8]) -> io::Result<()> {
        std::fs::write(path, data)
    }
    fn print(&mut self, line: &str) {
        println!("{line}");
    }
    fn eprint(&mut self, line: &str) {
        eprintln!("{line}");
    }
}

/// In-memory adapter: no real IO, so it works in any environment (including
/// wasm and sandboxes). Seed inputs with [`put`](Self::put) / [`put_str`], run
/// the CLI, then read written files with [`get`](Self::get) and captured stdout
/// with [`output`](Self::output).
#[derive(Default)]
pub struct MemEnv {
    files: HashMap<String, Vec<u8>>,
    out: String,
    err: String,
}

impl MemEnv {
    pub fn new() -> Self {
        MemEnv::default()
    }
    /// Seed a file from raw bytes.
    pub fn put(&mut self, path: &str, data: Vec<u8>) {
        self.files.insert(path.to_string(), data);
    }
    /// Seed a file from text (e.g. a CSV literal).
    pub fn put_str(&mut self, path: &str, text: &str) {
        self.files
            .insert(path.to_string(), text.as_bytes().to_vec());
    }
    /// Bytes of a written (or seeded) file.
    pub fn get(&self, path: &str) -> Option<&[u8]> {
        self.files.get(path).map(|v| v.as_slice())
    }
    /// Everything printed so far (newline-separated).
    pub fn output(&self) -> &str {
        &self.out
    }
    /// Everything printed to stderr so far.
    pub fn errors(&self) -> &str {
        &self.err
    }
}

impl Env for MemEnv {
    fn read(&self, path: &str) -> io::Result<Vec<u8>> {
        match self.files.get(path) {
            Some(v) => Ok(v.clone()),
            None => Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("no such file: {path}"),
            )),
        }
    }
    fn write(&mut self, path: &str, data: &[u8]) -> io::Result<()> {
        self.files.insert(path.to_string(), data.to_vec());
        Ok(())
    }
    fn print(&mut self, line: &str) {
        self.out.push_str(line);
        self.out.push('\n');
    }
    fn eprint(&mut self, line: &str) {
        self.err.push_str(line);
        self.err.push('\n');
    }
}

// --- argument helpers (public so the binary can share them) ---------------

pub fn flag_value<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

pub fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

pub fn parse_flag<T: std::str::FromStr>(
    args: &[String],
    name: &str,
    default: T,
) -> Result<T, String> {
    match flag_value(args, name) {
        Some(s) => s.parse().map_err(|_| format!("bad {name}")),
        None => Ok(default),
    }
}

pub fn parse_metric(s: &str) -> Result<Metric, String> {
    match s {
        "l2" | "L2" => Ok(Metric::L2),
        "dot" | "ip" => Ok(Metric::Dot),
        "cosine" | "cos" => Ok(Metric::Cosine),
        _ => Err(format!("unknown metric: {s}")),
    }
}

fn metric_of(args: &[String]) -> Result<Metric, String> {
    flag_value(args, "--metric").map_or(Ok(Metric::Cosine), parse_metric)
}

// --- CSV parsing over bytes -----------------------------------------------

type Record = (u64, Vec<f32>);

/// Parse `id,v0,v1,...` rows from CSV bytes (`#` lines and blanks skipped).
pub fn parse_csv(bytes: &[u8]) -> Result<Vec<Record>, String> {
    let text = std::str::from_utf8(bytes).map_err(|_| "input is not valid UTF-8".to_string())?;
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
        let v: Result<Vec<f32>, _> = it.map(|f| f.trim().parse::<f32>()).collect();
        let v = v.map_err(|_| format!("line {}: bad float", lineno + 1))?;
        out.push((id, v));
    }
    Ok(out)
}

fn read_records(env: &impl Env, path: &str) -> Result<Vec<Record>, String> {
    let bytes = env.read(path).map_err(|e| e.to_string())?;
    parse_csv(&bytes)
}

/// `<in.csv> <out.vecdb>` + validated records with a uniform dimension.
fn build_input(env: &impl Env, args: &[String]) -> Result<(String, usize, Vec<Record>), String> {
    if args.len() < 2 {
        return Err("expected <in.csv> <out.vecdb>".into());
    }
    let output = args[1].clone();
    let records = read_records(env, &args[0])?;
    if records.is_empty() {
        return Err("no records".into());
    }
    let dim = records[0].1.len();
    for (id, v) in &records {
        if v.len() != dim {
            return Err(format!("id {id}: dim {} != {dim}", v.len()));
        }
    }
    Ok((output, dim, records))
}

// --- index-kind detection --------------------------------------------------

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

fn detect(b: &[u8]) -> Result<Kind, String> {
    if b.len() < 8 {
        return Err("unrecognized index file (too short)".into());
    }
    match &b[0..8] {
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

// --- command dispatch ------------------------------------------------------

/// Run one CLI invocation (`args[0]` is the subcommand). All IO via `env`.
pub fn run<E: Env>(env: &mut E, args: &[String]) -> Result<(), String> {
    let cmd = args.first().map(String::as_str);
    let rest = &args[args.len().min(1)..];
    match cmd {
        Some("build") => cmd_build(env, rest),
        Some("build-ivf") => cmd_build_ivf(env, rest),
        Some("build-hnsw") => cmd_build_hnsw(env, rest),
        Some("build-hnsw-q") => cmd_build_hnsw_q(env, rest),
        Some("build-pq") => cmd_build_pq(env, rest),
        Some("build-ivfpq") => cmd_build_ivfpq(env, rest),
        Some("build-opq") => cmd_build_opq(env, rest),
        Some("build-diskann") => cmd_build_diskann(env, rest),
        Some("search") => cmd_search(env, rest),
        Some("info") => cmd_info(env, rest),
        _ => Err("usage: meandb-vector <build|build-ivf|build-hnsw|build-hnsw-q|build-pq|build-ivfpq|build-opq|build-diskann|search|info> ... (compatibility alias: vecdb)".into()),
    }
}

fn require_dim_multiple(dim: usize, pq_m: usize) -> Result<(), String> {
    if !dim.is_multiple_of(pq_m) {
        return Err(format!("--pq-m {pq_m} must divide dim {dim}"));
    }
    Ok(())
}

fn cmd_build<E: Env>(env: &mut E, args: &[String]) -> Result<(), String> {
    let (output, dim, records) = build_input(env, args)?;
    let metric = metric_of(args)?;
    let compact = has_flag(args, "--compact");
    let mut idx = FlatIndex::new(dim, metric, !compact);
    for (id, v) in &records {
        idx.add(*id, v);
    }
    env.write(&output, &crate::to_bytes(&idx))
        .map_err(|e| e.to_string())?;
    env.print(&format!(
        "built flat: {} vectors (dim {dim}, {metric:?}, {}) -> {output}",
        idx.len(),
        if compact { "int8-only" } else { "int8+raw" }
    ));
    Ok(())
}

fn cmd_build_ivf<E: Env>(env: &mut E, args: &[String]) -> Result<(), String> {
    let (output, dim, records) = build_input(env, args)?;
    let metric = metric_of(args)?;
    let nlist: usize = parse_flag(args, "--nlist", 0)?;
    let iters: usize = parse_flag(args, "--iters", 12)?;
    let idx = IvfIndex::build(dim, metric, nlist, &records, true, iters);
    env.write(&output, &idx.to_bytes())
        .map_err(|e| e.to_string())?;
    env.print(&format!(
        "built ivf: {} vectors (dim {dim}, {metric:?}, nlist {}) -> {output}",
        idx.len(),
        idx.nlist()
    ));
    Ok(())
}

fn cmd_build_hnsw<E: Env>(env: &mut E, args: &[String]) -> Result<(), String> {
    let (output, dim, records) = build_input(env, args)?;
    let metric = metric_of(args)?;
    let m: usize = parse_flag(args, "-m", 16)?;
    let efc: usize = parse_flag(args, "--ef-construction", 200)?;
    let mut idx = HnswIndex::new(dim, metric, m, efc);
    for (id, v) in &records {
        idx.add(*id, v);
    }
    env.write(&output, &idx.to_bytes())
        .map_err(|e| e.to_string())?;
    env.print(&format!(
        "built hnsw: {} vectors (dim {dim}, {metric:?}, M {m}) -> {output}",
        idx.len()
    ));
    Ok(())
}

fn cmd_build_hnsw_q<E: Env>(env: &mut E, args: &[String]) -> Result<(), String> {
    let (output, dim, records) = build_input(env, args)?;
    let metric = metric_of(args)?;
    let m: usize = parse_flag(args, "-m", 16)?;
    let efc: usize = parse_flag(args, "--ef-construction", 200)?;
    let compact = has_flag(args, "--compact");
    let mut idx = HnswQIndex::new(dim, metric, m, efc, !compact);
    for (id, v) in &records {
        idx.add(*id, v);
    }
    env.write(&output, &idx.to_bytes())
        .map_err(|e| e.to_string())?;
    env.print(&format!(
        "built hnsw-q (int8 graph): {} vectors (dim {dim}, {metric:?}, M {m}, {}) -> {output}",
        idx.len(),
        if compact {
            "int8-only"
        } else {
            "int8+raw rerank"
        }
    ));
    Ok(())
}

fn cmd_build_pq<E: Env>(env: &mut E, args: &[String]) -> Result<(), String> {
    let (output, dim, records) = build_input(env, args)?;
    let metric = metric_of(args)?;
    let pq_m: usize = parse_flag(args, "--pq-m", 16)?;
    let ksub: usize = parse_flag(args, "--ksub", 256)?;
    let iters: usize = parse_flag(args, "--iters", 20)?;
    let compact = has_flag(args, "--compact");
    require_dim_multiple(dim, pq_m)?;
    let idx = PqIndex::build(&records, metric, pq_m, ksub, iters, !compact);
    env.write(&output, &idx.to_bytes())
        .map_err(|e| e.to_string())?;
    env.print(&format!(
        "built pq: {} vectors (dim {dim}, {metric:?}, m {pq_m}, ksub {ksub}, {} bytes/vec) -> {output}",
        idx.len(),
        idx.code_bytes() / idx.len().max(1)
    ));
    Ok(())
}

fn cmd_build_ivfpq<E: Env>(env: &mut E, args: &[String]) -> Result<(), String> {
    let (output, dim, records) = build_input(env, args)?;
    let metric = metric_of(args)?;
    let nlist: usize = parse_flag(args, "--nlist", 0)?;
    let pq_m: usize = parse_flag(args, "--pq-m", 16)?;
    let ksub: usize = parse_flag(args, "--ksub", 256)?;
    let iters: usize = parse_flag(args, "--iters", 15)?;
    let compact = has_flag(args, "--compact");
    require_dim_multiple(dim, pq_m)?;
    let idx = IvfPqIndex::build(&records, metric, nlist, pq_m, ksub, iters, !compact);
    env.write(&output, &idx.to_bytes())
        .map_err(|e| e.to_string())?;
    env.print(&format!(
        "built ivfpq: {} vectors (dim {dim}, {metric:?}, nlist {}, m {pq_m}, {} bytes/vec) -> {output}",
        idx.len(),
        idx.nlist(),
        idx.code_bytes() / idx.len().max(1)
    ));
    Ok(())
}

fn cmd_build_opq<E: Env>(env: &mut E, args: &[String]) -> Result<(), String> {
    let (output, dim, records) = build_input(env, args)?;
    let metric = metric_of(args)?;
    let pq_m: usize = parse_flag(args, "--pq-m", 16)?;
    let ksub: usize = parse_flag(args, "--ksub", 256)?;
    let iters: usize = parse_flag(args, "--iters", 20)?;
    let opq_iters: usize = parse_flag(args, "--opq-iters", 4)?;
    let compact = has_flag(args, "--compact");
    require_dim_multiple(dim, pq_m)?;
    let idx = OpqIndex::build(&records, metric, pq_m, ksub, iters, opq_iters, !compact);
    env.write(&output, &idx.to_bytes())
        .map_err(|e| e.to_string())?;
    env.print(&format!(
        "built opq: {} vectors (dim {dim}, {metric:?}, m {pq_m}, ksub {ksub}, opq-iters {opq_iters}, {} bytes/vec) -> {output}",
        idx.len(),
        idx.code_bytes() / idx.len().max(1)
    ));
    Ok(())
}

/// In-memory DiskANN build (portable). The low-memory *streaming* build needs a
/// real filesystem and lives in the `vecdb` binary (`--streaming`).
fn cmd_build_diskann<E: Env>(env: &mut E, args: &[String]) -> Result<(), String> {
    let (output, dim, records) = build_input(env, args)?;
    let metric = metric_of(args)?;
    let r: usize = parse_flag(args, "-r", 32)?;
    let l_build: usize = parse_flag(args, "--l-build", 96)?;
    let alpha: f32 = parse_flag(args, "--alpha", 1.2)?;
    let pq_m: usize = parse_flag(args, "--pq-m", 16)?;
    let ksub: usize = parse_flag(args, "--ksub", 256)?;
    require_dim_multiple(dim, pq_m)?;
    let idx = DiskAnnIndex::build(&records, metric, r, l_build, alpha, pq_m, ksub);
    env.write(&output, &idx.to_bytes())
        .map_err(|e| e.to_string())?;
    env.print(&format!(
        "built diskann: {} vectors (dim {dim}, {metric:?}, R {r}, avg-deg {:.1}) -> {output}",
        idx.len(),
        idx.avg_degree()
    ));
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

fn cmd_search<E: Env>(env: &mut E, args: &[String]) -> Result<(), String> {
    if args.len() < 2 {
        return Err(
            "search <index.vecdb> <query.csv> [-k N] [--oversample M] [--nprobe N] [--ef N]".into(),
        );
    }
    let idx_bytes = env.read(&args[0]).map_err(|e| e.to_string())?;
    let queries = read_records(env, &args[1])?;
    let qvs: Vec<Vec<f32>> = queries.iter().map(|(_, v)| v.clone()).collect();
    let k: usize = parse_flag(args, "-k", 10)?;
    let over: usize = parse_flag(args, "--oversample", 4)?;
    let nprobe: usize = parse_flag(args, "--nprobe", 16)?;
    let ef: usize = parse_flag(args, "--ef", 64)?;

    let e = |s: String| s;
    let results: Vec<Vec<crate::Hit>> = match detect(&idx_bytes)? {
        Kind::Flat => {
            let idx = crate::from_bytes(&idx_bytes).map_err(|x| e(x.to_string()))?;
            check_dims(&qvs, idx.dim())?;
            qvs.iter().map(|v| idx.search(v, k, over)).collect()
        }
        Kind::Ivf => {
            let idx = IvfIndex::from_bytes(&idx_bytes).map_err(|x| e(x.to_string()))?;
            check_dims(&qvs, idx.dim())?;
            qvs.iter().map(|v| idx.search(v, k, nprobe, over)).collect()
        }
        Kind::Hnsw => {
            let idx = HnswIndex::from_bytes(&idx_bytes).map_err(|x| e(x.to_string()))?;
            check_dims(&qvs, idx.dim())?;
            qvs.iter().map(|v| idx.search(v, k, ef)).collect()
        }
        Kind::HnswQ => {
            let idx = HnswQIndex::from_bytes(&idx_bytes).map_err(|x| e(x.to_string()))?;
            check_dims(&qvs, idx.dim())?;
            qvs.iter().map(|v| idx.search(v, k, ef)).collect()
        }
        Kind::Pq => {
            let idx = PqIndex::from_bytes(&idx_bytes).map_err(|x| e(x.to_string()))?;
            check_dims(&qvs, idx.dim())?;
            qvs.iter().map(|v| idx.search(v, k, over)).collect()
        }
        Kind::IvfPq => {
            let idx = IvfPqIndex::from_bytes(&idx_bytes).map_err(|x| e(x.to_string()))?;
            check_dims(&qvs, idx.dim())?;
            qvs.iter().map(|v| idx.search(v, k, nprobe, over)).collect()
        }
        Kind::Opq => {
            let idx = OpqIndex::from_bytes(&idx_bytes).map_err(|x| e(x.to_string()))?;
            check_dims(&qvs, idx.dim())?;
            qvs.iter().map(|v| idx.search(v, k, over)).collect()
        }
        Kind::DiskAnn => {
            let idx = DiskAnnIndex::from_bytes(&idx_bytes).map_err(|x| e(x.to_string()))?;
            check_dims(&qvs, idx.dim())?;
            qvs.iter().map(|v| idx.search(v, k, ef)).collect()
        }
    };

    for ((qid, _), hits) in queries.iter().zip(results.iter()) {
        for (rank, h) in hits.iter().enumerate() {
            env.print(&format!("{qid}\t{rank}\t{}\t{:.6}", h.id, h.score));
        }
    }
    Ok(())
}

fn cmd_info<E: Env>(env: &mut E, args: &[String]) -> Result<(), String> {
    let path = args.first().ok_or("info <index.vecdb>")?.clone();
    let b = env.read(&path).map_err(|e| e.to_string())?;
    let mut lines: Vec<String> = vec![format!("path:    {path}")];
    match detect(&b)? {
        Kind::Flat => {
            let idx = crate::from_bytes(&b).map_err(|x| x.to_string())?;
            lines.push("type:    flat".into());
            lines.push(format!("count:   {}", idx.len()));
            lines.push(format!("dim:     {}", idx.dim()));
            lines.push(format!("metric:  {:?}", idx.metric()));
            lines.push(format!("raw f32: {}", idx.has_raw()));
        }
        Kind::Ivf => {
            let idx = IvfIndex::from_bytes(&b).map_err(|x| x.to_string())?;
            lines.push("type:    ivf".into());
            lines.push(format!("count:   {}", idx.len()));
            lines.push(format!("dim:     {}", idx.dim()));
            lines.push(format!("metric:  {:?}", idx.metric()));
            lines.push(format!("nlist:   {}", idx.nlist()));
            lines.push(format!("raw f32: {}", idx.has_raw()));
        }
        Kind::Hnsw => {
            let idx = HnswIndex::from_bytes(&b).map_err(|x| x.to_string())?;
            lines.push("type:    hnsw".into());
            lines.push(format!("count:   {}", idx.len()));
            lines.push(format!("dim:     {}", idx.dim()));
            lines.push(format!("metric:  {:?}", idx.metric()));
        }
        Kind::HnswQ => {
            let idx = HnswQIndex::from_bytes(&b).map_err(|x| x.to_string())?;
            lines.push("type:    hnsw-q (int8 graph)".into());
            lines.push(format!("count:   {}", idx.len()));
            lines.push(format!("dim:     {}", idx.dim()));
            lines.push(format!("metric:  {:?}", idx.metric()));
            lines.push(format!("raw f32: {}", idx.has_raw()));
        }
        Kind::Pq => {
            let idx = PqIndex::from_bytes(&b).map_err(|x| x.to_string())?;
            lines.push("type:    pq".into());
            lines.push(format!("count:   {}", idx.len()));
            lines.push(format!("dim:     {}", idx.dim()));
            lines.push(format!("metric:  {:?}", idx.metric()));
            lines.push(format!("raw f32: {}", idx.has_raw()));
        }
        Kind::IvfPq => {
            let idx = IvfPqIndex::from_bytes(&b).map_err(|x| x.to_string())?;
            lines.push("type:    ivfpq".into());
            lines.push(format!("count:   {}", idx.len()));
            lines.push(format!("dim:     {}", idx.dim()));
            lines.push(format!("metric:  {:?}", idx.metric()));
            lines.push(format!("nlist:   {}", idx.nlist()));
            lines.push(format!("raw f32: {}", idx.has_raw()));
        }
        Kind::Opq => {
            let idx = OpqIndex::from_bytes(&b).map_err(|x| x.to_string())?;
            lines.push("type:    opq".into());
            lines.push(format!("count:   {}", idx.len()));
            lines.push(format!("dim:     {}", idx.dim()));
            lines.push(format!("metric:  {:?}", idx.metric()));
            lines.push(format!("raw f32: {}", idx.has_raw()));
        }
        Kind::DiskAnn => {
            let idx = DiskAnnIndex::from_bytes(&b).map_err(|x| x.to_string())?;
            lines.push("type:    diskann (vamana)".into());
            lines.push(format!("count:   {}", idx.len()));
            lines.push(format!("dim:     {}", idx.dim()));
            lines.push(format!("metric:  {:?}", idx.metric()));
            lines.push(format!("avg-deg: {:.1}", idx.avg_degree()));
        }
    }
    for l in &lines {
        env.print(l);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args<const N: usize>(a: [&str; N]) -> Vec<String> {
        a.iter().map(|s| s.to_string()).collect()
    }

    fn sample_csv() -> &'static str {
        "0,1,0,0,0\n1,0,1,0,0\n2,0.9,0.1,0,0\n3,0,0,1,0\n# comment\n4,0,0,0,1\n"
    }

    #[test]
    fn mem_env_build_search_info_all_kinds() {
        let builds: &[(&str, Vec<String>)] = &[
            (
                "build",
                args(["build", "v.csv", "i.vecdb", "--metric", "cosine"]),
            ),
            (
                "build-ivf",
                args(["build-ivf", "v.csv", "i.vecdb", "--nlist", "2"]),
            ),
            ("build-hnsw", args(["build-hnsw", "v.csv", "i.vecdb"])),
            ("build-hnsw-q", args(["build-hnsw-q", "v.csv", "i.vecdb"])),
            (
                "build-pq",
                args(["build-pq", "v.csv", "i.vecdb", "--pq-m", "1", "--ksub", "4"]),
            ),
            (
                "build-ivfpq",
                args([
                    "build-ivfpq",
                    "v.csv",
                    "i.vecdb",
                    "--nlist",
                    "2",
                    "--pq-m",
                    "1",
                    "--ksub",
                    "4",
                ]),
            ),
            (
                "build-opq",
                args([
                    "build-opq",
                    "v.csv",
                    "i.vecdb",
                    "--pq-m",
                    "1",
                    "--ksub",
                    "4",
                ]),
            ),
            (
                "build-diskann",
                args([
                    "build-diskann",
                    "v.csv",
                    "i.vecdb",
                    "--pq-m",
                    "1",
                    "--ksub",
                    "4",
                    "--metric",
                    "l2",
                ]),
            ),
        ];
        for (name, build_args) in builds {
            let mut env = MemEnv::new();
            env.put_str("v.csv", sample_csv());
            run(&mut env, build_args).unwrap_or_else(|e| panic!("{name} build failed: {e}"));
            assert!(env.get("i.vecdb").is_some(), "{name}: no index written");

            run(&mut env, &args(["info", "i.vecdb"])).unwrap();
            assert!(env.output().contains("count:   5"), "{name}: bad info");

            env.put_str("q.csv", "0,1,0.05,0,0\n");
            let before = env.output().len();
            run(
                &mut env,
                &args([
                    "search", "i.vecdb", "q.csv", "-k", "3", "--nprobe", "2", "--ef", "16",
                ]),
            )
            .unwrap_or_else(|e| panic!("{name} search failed: {e}"));
            let printed = &env.output()[before..];
            // top hit for a query near row 0 should be id 0.
            assert!(
                printed.starts_with("0\t0\t0\t"),
                "{name}: top hit not id 0: {printed}"
            );
        }
    }

    #[test]
    fn mem_env_errors() {
        let mut env = MemEnv::new();
        assert!(run(&mut env, &args(["bogus"])).is_err());
        env.put_str("x.vecdb", "not an index file");
        assert!(run(&mut env, &args(["info", "x.vecdb"])).is_err());
        assert!(run(&mut env, &args(["search", "missing.vecdb", "q.csv"])).is_err());
    }

    #[test]
    fn round_trips_through_bytes_only() {
        // The whole flow never touches the filesystem.
        let mut env = MemEnv::new();
        env.put_str("v.csv", sample_csv());
        run(&mut env, &args(["build", "v.csv", "i.vecdb"])).unwrap();
        let bytes = env.get("i.vecdb").unwrap();
        assert_eq!(&bytes[0..6], b"VECDB1");
    }
}

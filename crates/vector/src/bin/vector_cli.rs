//! Minimal CLI for the meandb vector layer's `.vecdb` formats.
//!
//! Vectors are read/written as CSV-ish text: one record per line,
//! `id,v0,v1,...,vD-1`. The command logic lives in [`meandb_vector::cli`] over an
//! `Env` abstraction; this binary just wires it to the real filesystem
//! ([`meandb_vector::cli::StdEnv`]). The one exception is the low-memory DiskANN
//! *streaming* build (`build-diskann --streaming`), which genuinely needs files
//! (a temp spill + lazy CSV read) and so is handled here.
//!
//! Usage:
//!   meandb-vector build        <in.csv> <out.vecdb> [--metric l2|dot|cosine] [--compact]
//!   meandb-vector build-ivf    <in.csv> <out.vecdb> [--metric m] [--nlist N] [--iters N]
//!   meandb-vector build-hnsw   <in.csv> <out.vecdb> [--metric m] [-m M] [--ef-construction N]
//!   meandb-vector build-hnsw-q <in.csv> <out.vecdb> [--metric m] [-m M] [--ef-construction N] [--compact]
//!   meandb-vector build-pq     <in.csv> <out.vecdb> [--metric m] [--pq-m M] [--ksub K] [--iters N] [--compact]
//!   meandb-vector build-ivfpq  <in.csv> <out.vecdb> [--metric m] [--nlist N] [--pq-m M] [--ksub K] [--iters N] [--compact]
//!   meandb-vector build-opq    <in.csv> <out.vecdb> [--metric m] [--pq-m M] [--ksub K] [--iters N] [--opq-iters N] [--compact]
//!   meandb-vector build-diskann <in.csv> <out.vecdb> [--metric m] [-r R] [--l-build N] [--alpha A] [--pq-m M] [--ksub K] [--streaming [--sample N]]
//!   meandb-vector search       <index.vecdb> <query.csv> [-k N] [--oversample M] [--nprobe N] [--ef N]
//!   meandb-vector info         <index.vecdb>

use meandb_vector::cli::{self, flag_value, has_flag, parse_flag, parse_metric, StdEnv};
use meandb_vector::{DiskAnnIndex, Metric};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::process::ExitCode;

pub fn entrypoint() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str);
    let rest = &args[args.len().min(1)..];

    // The streaming DiskANN build is filesystem-specific (temp spill + lazy CSV
    // read); everything else is environment-agnostic and goes through cli::run.
    let result = if cmd == Some("build-diskann") && has_flag(rest, "--streaming") {
        build_diskann_streaming(rest)
    } else {
        cli::run(&mut StdEnv::new(), &args)
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Low-memory DiskANN build: consume the CSV lazily (one record at a time) and
/// never hold all raw vectors resident. Not part of the portable `cli::run`
/// because it depends on real files.
fn build_diskann_streaming(args: &[String]) -> Result<(), String> {
    if args.len() < 2 {
        return Err("expected <in.csv> <out.vecdb>".into());
    }
    let input = &args[0];
    let output = &args[1];
    let metric = flag_value(args, "--metric").map_or(Ok(Metric::Cosine), parse_metric)?;
    let r: usize = parse_flag(args, "-r", 32)?;
    let l_build: usize = parse_flag(args, "--l-build", 96)?;
    let alpha: f32 = parse_flag(args, "--alpha", 1.2)?;
    let pq_m: usize = parse_flag(args, "--pq-m", 16)?;
    let ksub: usize = parse_flag(args, "--ksub", 256)?;
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
    let n = DiskAnnIndex::open(output).map_err(|e| e.to_string())?.len();
    println!(
        "built diskann (streaming): {n} vectors (dim {dim}, {metric:?}, R {r}, sample {sample}) -> {output}"
    );
    Ok(())
}

/// Dimension of the first data record, without reading the whole file.
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

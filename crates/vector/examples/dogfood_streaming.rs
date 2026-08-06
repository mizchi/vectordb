//! Dogfooding the low-memory DiskANN build (`DiskAnnIndex::build_streaming`) on
//! real ANN_SIFT10K data, exactly as an integrator would: the base vectors are
//! **streamed off disk one at a time** through a lazy iterator, so the full
//! `n * dim * f32` matrix is never resident. We then `open()` the result and
//! measure recall@10 against the exact ground truth, cross-checked against the
//! in-memory `build`.
//!
//!   cargo run --release --example dogfood_streaming -- siftsmall
//!
//! The point isn't a new benchmark — it's to use the shipped API end-to-end on
//! real data and confirm the streaming path (a) produces a valid index, (b)
//! keeps recall comparable to the exact-L2 build, and (c) works when the caller
//! genuinely cannot hold all vectors in RAM.

use meandb_vector::{DiskAnnIndex, Metric};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::time::Instant;

/// A lazy `.fvecs` reader: yields `(id, Vec<f32>)` one vector at a time from a
/// buffered file handle — it never holds more than a single vector in memory.
struct FvecsIter {
    rdr: BufReader<File>,
    next_id: u64,
}

impl FvecsIter {
    fn open(path: &Path) -> std::io::Result<Self> {
        Ok(Self {
            rdr: BufReader::new(File::open(path)?),
            next_id: 0,
        })
    }
}

impl Iterator for FvecsIter {
    type Item = (u64, Vec<f32>);
    fn next(&mut self) -> Option<(u64, Vec<f32>)> {
        let mut dbuf = [0u8; 4];
        if self.rdr.read_exact(&mut dbuf).is_err() {
            return None; // clean EOF
        }
        let d = i32::from_le_bytes(dbuf) as usize;
        let mut raw = vec![0u8; d * 4];
        self.rdr.read_exact(&mut raw).ok()?;
        let v: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let id = self.next_id;
        self.next_id += 1;
        Some((id, v))
    }
}

fn read_fvecs_all(path: &Path) -> Vec<Vec<f32>> {
    FvecsIter::open(path).unwrap().map(|(_, v)| v).collect()
}

fn read_ivecs(path: &Path) -> Vec<Vec<u32>> {
    let b = std::fs::read(path).unwrap();
    let mut out = Vec::new();
    let mut i = 0;
    while i + 4 <= b.len() {
        let d = i32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as usize;
        i += 4;
        let mut v = Vec::with_capacity(d);
        for _ in 0..d {
            v.push(u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]));
            i += 4;
        }
        out.push(v);
    }
    out
}

fn recall_at_k(got: &[meandb_vector::Hit], truth: &[u32], k: usize) -> usize {
    let t: std::collections::HashSet<u64> = truth.iter().take(k).map(|&x| x as u64).collect();
    got.iter().take(k).filter(|h| t.contains(&h.id)).count()
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "siftsmall".into());
    let dir = Path::new(&dir);
    let base_path = dir.join("siftsmall_base.fvecs");
    let queries = read_fvecs_all(&dir.join("siftsmall_query.fvecs"));
    let gt = read_ivecs(&dir.join("siftsmall_groundtruth.ivecs"));
    let dim = queries[0].len();
    let (nq, k) = (queries.len(), 10usize);

    // Count n without holding the vectors (a peek at the stream).
    let n = FvecsIter::open(&base_path).unwrap().count();
    let (r, l_build, alpha, pq_m, ksub, sample) = (32, 96, 1.2f32, 16, 256, 5000);
    println!("SIFT10K dogfood: {n} base x {dim}d, {nq} queries, metric=L2");
    // The exact-L2 build must hold the whole raw matrix; the streaming build's
    // resident set is PQ codes (n·m, grows with n) + the SDC centroid-pair table
    // (m·ksub², constant in n) — never the raw vectors.
    let raw_kib = n * dim * 4 / 1024;
    let codes_kib = n * pq_m / 1024;
    let sdc_kib = pq_m * ksub * ksub * 4 / 1024;
    println!(
        "RAM: raw f32 matrix {raw_kib} KiB  vs  streaming-resident ≈ codes {codes_kib} KiB + SDC table {sdc_kib} KiB"
    );
    println!("     (raw scales as n·dim·4; codes as n·m = {pq_m} B/vec; SDC is constant in n)\n");
    let tmp = std::env::temp_dir().join("sift_dogfood_stream.vecdb");

    // --- Streaming build: feed the lazy fvecs iterator straight in. ---
    let t = Instant::now();
    DiskAnnIndex::build_streaming(
        &tmp,
        dim,
        Metric::L2,
        r,
        l_build,
        alpha,
        pq_m,
        ksub,
        sample,
        FvecsIter::open(&base_path).unwrap(),
    )
    .expect("streaming build failed");
    let build_ms = t.elapsed().as_secs_f64() * 1e3;
    let disk = DiskAnnIndex::open(&tmp).expect("open failed");
    assert_eq!(disk.len(), n, "streaming index lost vectors");

    for &l_search in &[32usize, 64, 128] {
        let t = Instant::now();
        let mut hit = 0;
        for (qi, q) in queries.iter().enumerate() {
            hit += recall_at_k(&disk.search(q, k, l_search), &gt[qi], k);
        }
        let dur = t.elapsed();
        println!(
            "streaming(SDC) + mmap  L={l_search:<3} recall@{k}={:.4}  {:.3} ms/query",
            hit as f64 / (nq * k) as f64,
            dur.as_secs_f64() * 1e3 / nq as f64,
        );
    }
    println!("  (streaming build: {build_ms:.0} ms)\n");

    // --- Cross-check against the exact-L2 in-memory build. ---
    let items: Vec<(u64, Vec<f32>)> = FvecsIter::open(&base_path).unwrap().collect();
    let mem = DiskAnnIndex::build(&items, Metric::L2, r, l_build, alpha, pq_m, ksub);
    for &l_search in &[32usize, 64, 128] {
        let mut hit = 0;
        for (qi, q) in queries.iter().enumerate() {
            hit += recall_at_k(&mem.search(q, k, l_search), &gt[qi], k);
        }
        println!(
            "in-memory(exact-L2)    L={l_search:<3} recall@{k}={:.4}",
            hit as f64 / (nq * k) as f64,
        );
    }

    std::fs::remove_file(&tmp).ok();
}

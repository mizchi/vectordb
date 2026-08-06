//! Rough benchmark + recall check against exact brute force.
//!
//! Run with: `cargo run --release --example bench`

use meandb_vector::{
    BinaryIndex, FlatIndex, HnswIndex, IvfIndex, IvfRabitqIndex, Metric, RabitqIndex,
};
use std::time::Instant;

fn main() {
    let n = 50_000usize;
    let dim = 128usize;
    let queries = 200usize;
    let k = 10usize;
    let oversample = 8usize;

    // Clustered data (like real embeddings): points are drawn around a set of
    // random centers. IVF relies on this structure; uniform-random data is its
    // worst case and would show near-zero recall.
    let n_centers = 250usize;
    let noise = 0.15f32;
    println!("generating {n} x {dim} vectors in {n_centers} clusters...");
    let mut rng = Rng::new(0xC0FFEE);
    let centers: Vec<Vec<f32>> = (0..n_centers)
        .map(|_| (0..dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect())
        .collect();
    let jitter = |rng: &mut Rng, c: &[f32]| -> Vec<f32> {
        c.iter()
            .map(|x| x + (rng.next_f32() * 2.0 - 1.0) * noise)
            .collect()
    };
    let items: Vec<(u64, Vec<f32>)> = (0..n)
        .map(|i| {
            let c = &centers[(rng.next_u64() as usize) % n_centers];
            (i as u64, jitter(&mut rng, c))
        })
        .collect();
    let mut idx = FlatIndex::new(dim, Metric::Cosine, true);
    for (id, v) in &items {
        idx.add(*id, v);
    }
    // Queries drawn from the same cluster distribution.
    let qs: Vec<Vec<f32>> = (0..queries)
        .map(|_| {
            let c = &centers[(rng.next_u64() as usize) % n_centers];
            jitter(&mut rng, c)
        })
        .collect();

    // Warm up + measure quantized+rerank search.
    let t = Instant::now();
    let mut approx_results = Vec::with_capacity(queries);
    for q in &qs {
        approx_results.push(idx.search(q, k, oversample));
    }
    let approx_dur = t.elapsed();

    // int8-only scan (oversample = 1 skips the rerank stage).
    let t = Instant::now();
    for q in &qs {
        std::hint::black_box(idx.search(q, k, 1));
    }
    let int8_dur = t.elapsed();

    // Exact ground truth.
    let t = Instant::now();
    let mut exact_results = Vec::with_capacity(queries);
    for q in &qs {
        exact_results.push(idx.search_exact(q, k));
    }
    let exact_dur = t.elapsed();

    // Recall@k of the quantized+rerank path vs exact.
    let mut hits = 0usize;
    let mut total = 0usize;
    for (a, e) in approx_results.iter().zip(exact_results.iter()) {
        let truth: std::collections::HashSet<u64> = e.iter().map(|h| h.id).collect();
        for h in a {
            if truth.contains(&h.id) {
                hits += 1;
            }
        }
        total += e.len();
    }
    let recall = hits as f64 / total as f64;

    let per_query = |d: std::time::Duration| d.as_secs_f64() * 1e3 / queries as f64;
    println!(
        "int8+rerank(o={oversample}): {:.3} ms/query",
        per_query(approx_dur)
    );
    println!(
        "int8 only:                  {:.3} ms/query",
        per_query(int8_dur)
    );
    println!(
        "exact f32:                  {:.3} ms/query",
        per_query(exact_dur)
    );
    println!("recall@{k}: {:.4}", recall);
    println!(
        "footprint: int8 codes {} MiB vs f32 {} MiB (raw kept for rerank)",
        n * dim / (1024 * 1024),
        n * dim * 4 / (1024 * 1024)
    );

    #[cfg(feature = "parallel")]
    {
        let threads = rayon::current_num_threads();

        // Throughput: whole batch across threads (one query per task).
        let t = Instant::now();
        std::hint::black_box(idx.search_batch(&qs, k, oversample));
        let batch_dur = t.elapsed();

        // Latency: each query's scan split across threads.
        let t = Instant::now();
        for q in &qs {
            std::hint::black_box(idx.search_parallel(q, k, oversample));
        }
        let single_dur = t.elapsed();

        println!("--- parallel ({threads} threads) ---");
        println!(
            "int8+rerank batch:          {:.3} ms/query  ({:.1}x vs serial)",
            per_query(batch_dur),
            approx_dur.as_secs_f64() / batch_dur.as_secs_f64()
        );
        println!(
            "int8+rerank single(par):    {:.3} ms/query  ({:.1}x vs serial)",
            per_query(single_dur),
            approx_dur.as_secs_f64() / single_dur.as_secs_f64()
        );
    }

    // ---- IVF: coarse quantization, scan only nprobe cells ----
    let nlist = 256usize;
    let t = Instant::now();
    let ivf = IvfIndex::build(dim, Metric::Cosine, nlist, &items, true, 12);
    let ivf_build = t.elapsed();
    println!(
        "--- IVF (nlist={nlist}, built in {:.2}s) vs flat exact ---",
        ivf_build.as_secs_f64()
    );

    let recall_of = |results: &[Vec<meandb_vector::Hit>]| -> f64 {
        let mut h = 0usize;
        let mut tot = 0usize;
        for (a, e) in results.iter().zip(exact_results.iter()) {
            let truth: std::collections::HashSet<u64> = e.iter().map(|x| x.id).collect();
            h += a.iter().filter(|x| truth.contains(&x.id)).count();
            tot += e.len();
        }
        h as f64 / tot as f64
    };

    for &nprobe in &[1usize, 4, 8, 16, 32] {
        let t = Instant::now();
        let results: Vec<Vec<meandb_vector::Hit>> = qs
            .iter()
            .map(|q| ivf.search(q, k, nprobe, oversample))
            .collect();
        let dur = t.elapsed();
        println!(
            "nprobe={nprobe:<3} {:.3} ms/query  recall@{k}={:.4}  ({:.1}x vs flat int8+rerank)",
            per_query(dur),
            recall_of(&results),
            approx_dur.as_secs_f64() / dur.as_secs_f64()
        );
    }

    // ---- Binary (1-bit) quantization + rerank ----
    let bin = BinaryIndex::build(dim, Metric::Cosine, &items, true);
    let t = Instant::now();
    let bin_results: Vec<Vec<meandb_vector::Hit>> =
        qs.iter().map(|q| bin.search(q, k, 16)).collect();
    let bin_dur = t.elapsed();
    println!(
        "--- binary 1-bit (codes {} KiB = 1/32 of f32) ---",
        bin.code_bytes() / 1024
    );
    println!(
        "binary+rerank(o=16): {:.3} ms/query  recall@{k}={:.4}",
        per_query(bin_dur),
        recall_of(&bin_results)
    );
    println!(
        "  code sizes: binary {} KiB vs int8 {} KiB vs f32 {} KiB",
        bin.code_bytes() / 1024,
        n * dim / 1024,
        n * dim * 4 / 1024
    );

    // ---- RaBitQ (1-bit + unbiased estimator) vs plain binary ----
    let rq = RabitqIndex::build(dim, Metric::Cosine, &items, true, 0x5EED);
    println!("--- RaBitQ vs binary (both 1-bit codes, recall@{k}) ---");
    for &o in &[1usize, 4, 16] {
        let t = Instant::now();
        let rq_res: Vec<Vec<meandb_vector::Hit>> = qs.iter().map(|q| rq.search(q, k, o)).collect();
        let rq_dur = t.elapsed();
        let bin_res: Vec<Vec<meandb_vector::Hit>> =
            qs.iter().map(|q| bin.search(q, k, o)).collect();
        println!(
            "oversample={o:<3} RaBitQ recall={:.4} ({:.3} ms)   binary recall={:.4}",
            recall_of(&rq_res),
            per_query(rq_dur),
            recall_of(&bin_res)
        );
    }

    // ---- IVF + RaBitQ: per-cell centroids, 1-bit codes ----
    let ivfrq = IvfRabitqIndex::build(dim, Metric::Cosine, nlist, &items, true, 12, 0xF00D);
    println!(
        "--- IVF+RaBitQ (nlist={nlist}, 1-bit codes {} KiB) ---",
        ivfrq.code_bytes() / 1024
    );
    // recall is bounded by the rerank candidate count (oversample), not nprobe
    // here, since 1-bit estimates are coarser than int8 — sweep oversample.
    for &o in &[8usize, 16, 32, 64] {
        let t = Instant::now();
        let res: Vec<Vec<meandb_vector::Hit>> =
            qs.iter().map(|q| ivfrq.search(q, k, 16, o)).collect();
        let dur = t.elapsed();
        println!(
            "nprobe=16 oversample={o:<3} {:.3} ms/query  recall@{k}={:.4}",
            per_query(dur),
            recall_of(&res)
        );
    }

    // ---- HNSW: graph index ----
    let t = Instant::now();
    let mut hnsw = HnswIndex::new(dim, Metric::Cosine, 16, 200);
    for (id, v) in &items {
        hnsw.add(*id, v);
    }
    let hnsw_build = t.elapsed();
    println!(
        "--- HNSW (M=16, efC=200, built in {:.2}s) ---",
        hnsw_build.as_secs_f64()
    );
    for &ef in &[16usize, 32, 64, 128] {
        let t = Instant::now();
        let res: Vec<Vec<meandb_vector::Hit>> = qs.iter().map(|q| hnsw.search(q, k, ef)).collect();
        let dur = t.elapsed();
        println!(
            "efSearch={ef:<3} {:.3} ms/query  recall@{k}={:.4}",
            per_query(dur),
            recall_of(&res)
        );
    }
}

/// Tiny deterministic xorshift RNG (no external deps).
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

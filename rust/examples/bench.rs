//! Rough benchmark + recall check against exact brute force.
//!
//! Run with: `cargo run --release --example bench`

use std::time::Instant;
use vectordb::{FlatIndex, Metric};

fn main() {
    let n = 50_000usize;
    let dim = 128usize;
    let queries = 200usize;
    let k = 10usize;
    let oversample = 8usize;

    println!("generating {n} x {dim} vectors...");
    let mut idx = FlatIndex::new(dim, Metric::Cosine, true);
    let mut rng = Rng::new(0xC0FFEE);
    for i in 0..n {
        let v: Vec<f32> = (0..dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        idx.add(i as u64, &v);
    }
    let qs: Vec<Vec<f32>> = (0..queries)
        .map(|_| (0..dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect())
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
    println!("int8+rerank(o={oversample}): {:.3} ms/query", per_query(approx_dur));
    println!("int8 only:                  {:.3} ms/query", per_query(int8_dur));
    println!("exact f32:                  {:.3} ms/query", per_query(exact_dur));
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

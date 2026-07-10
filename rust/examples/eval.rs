//! Real-dataset evaluation on ANN_SIFT10K (siftsmall): 10k base vectors,
//! 128-dim SIFT descriptors, 100 queries, exact 100-NN ground truth.
//!
//! Download (Hugging Face mirror of the IRISA corpus):
//!   curl -fsSL https://huggingface.co/datasets/vecdata/siftsmall/resolve/main/siftsmall.tar.gz \
//!     | tar xz
//! Then run:
//!   cargo run --release --example eval -- siftsmall
//!
//! Reports recall@10 vs the ground truth and throughput for each index type.

use std::path::Path;
use std::time::Instant;
use vectordb::{
    BinaryIndex, DiskAnnIndex, FlatIndex, HnswIndex, HnswQIndex, IvfIndex, IvfPqIndex,
    IvfRabitqIndex, Metric, OpqIndex, PqIndex, RabitqIndex,
};

type SearchFn<'a> = Box<dyn FnMut(&[f32]) -> Vec<vectordb::Hit> + 'a>;

fn read_fvecs(path: &Path) -> (usize, Vec<Vec<f32>>) {
    let b = std::fs::read(path).unwrap_or_else(|_| panic!("cannot read {}", path.display()));
    let mut out = Vec::new();
    let mut i = 0;
    let mut dim = 0;
    while i + 4 <= b.len() {
        let d = i32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as usize;
        i += 4;
        dim = d;
        let mut v = Vec::with_capacity(d);
        for _ in 0..d {
            v.push(f32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]));
            i += 4;
        }
        out.push(v);
    }
    (dim, out)
}

fn read_ivecs(path: &Path) -> Vec<Vec<u32>> {
    let b = std::fs::read(path).unwrap_or_else(|_| panic!("cannot read {}", path.display()));
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

fn recall_at_k(got: &[vectordb::Hit], truth: &[u32], k: usize) -> usize {
    let t: std::collections::HashSet<u64> = truth.iter().take(k).map(|&x| x as u64).collect();
    got.iter().take(k).filter(|h| t.contains(&h.id)).count()
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "siftsmall".to_string());
    let dir = Path::new(&dir);
    let (dim, base) = read_fvecs(&dir.join("siftsmall_base.fvecs"));
    let (_, queries) = read_fvecs(&dir.join("siftsmall_query.fvecs"));
    let gt = read_ivecs(&dir.join("siftsmall_groundtruth.ivecs"));
    let n = base.len();
    let nq = queries.len();
    let k = 10usize;
    println!("SIFT10K: {n} base x {dim}d, {nq} queries, metric=L2\n");

    let items: Vec<(u64, Vec<f32>)> = base
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, v)| (i as u64, v))
        .collect();

    // Aggregate recall@k over all queries for a search closure.
    let eval = |name: &str, mut f: SearchFn| {
        let t = Instant::now();
        let mut hit = 0usize;
        for (qi, q) in queries.iter().enumerate() {
            let got = f(q);
            hit += recall_at_k(&got, &gt[qi], k);
        }
        let dur = t.elapsed();
        println!(
            "{name:<34} recall@{k}={:.4}  {:.3} ms/query  {:.0} qps",
            hit as f64 / (nq * k) as f64,
            dur.as_secs_f64() * 1e3 / nq as f64,
            nq as f64 / dur.as_secs_f64(),
        );
    };

    // Flat exact (reference).
    let mut flat = FlatIndex::new(dim, Metric::L2, true);
    for (id, v) in &items {
        flat.add(*id, v);
    }
    {
        let f = &flat;
        eval("flat exact f32", Box::new(move |q| f.search_exact(q, k)));
    }
    {
        let f = &flat;
        eval(
            "flat int8+rerank (o=8)",
            Box::new(move |q| f.search(q, k, 8)),
        );
    }

    // Binary + rerank.
    let bin = BinaryIndex::build(dim, Metric::L2, &items, true);
    {
        let b = &bin;
        eval(
            "binary 1-bit+rerank (o=32)",
            Box::new(move |q| b.search(q, k, 32)),
        );
    }

    // RaBitQ flat.
    let rq = RabitqIndex::build(dim, Metric::L2, &items, true, 1);
    {
        let r = &rq;
        eval(
            "RaBitQ flat+rerank (o=32)",
            Box::new(move |q| r.search(q, k, 32)),
        );
    }

    // IVF.
    let ivf = IvfIndex::build(dim, Metric::L2, 100, &items, true, 15);
    for &np in &[4usize, 8, 16] {
        let i = &ivf;
        eval(
            &format!("IVF nprobe={np} (o=8)"),
            Box::new(move |q| i.search(q, k, np, 8)),
        );
    }

    // IVF + RaBitQ.
    let ivfrq = IvfRabitqIndex::build(dim, Metric::L2, 100, &items, true, 15, 1);
    for &np in &[8usize, 16] {
        let i = &ivfrq;
        eval(
            &format!("IVF+RaBitQ nprobe={np} (o=32)"),
            Box::new(move |q| i.search(q, k, np, 32)),
        );
    }

    // PQ (SIFT is 128-D; m=16 -> 16 bytes/vector = 32x over raw f32).
    let pq = PqIndex::build(&items, Metric::L2, 16, 256, 20, true);
    for &over in &[8usize, 32] {
        let p = &pq;
        eval(
            &format!("PQ m=16 (o={over})"),
            Box::new(move |q| p.search(q, k, over)),
        );
    }

    // OPQ (learned rotation + PQ, same 16 bytes/vector).
    let opq = OpqIndex::build(&items, Metric::L2, 16, 256, 20, 4, true);
    for &over in &[8usize, 32] {
        let o = &opq;
        eval(
            &format!("OPQ m=16 (o={over})"),
            Box::new(move |q| o.search(q, k, over)),
        );
    }

    // IVF+PQ (coarse cells + residual PQ codes).
    let ivfpq = IvfPqIndex::build(&items, Metric::L2, 100, 16, 256, 15, true);
    for &np in &[8usize, 16] {
        let i = &ivfpq;
        eval(
            &format!("IVF+PQ nprobe={np} (o=16)"),
            Box::new(move |q| i.search(q, k, np, 16)),
        );
    }

    // HNSW.
    let mut hnsw = HnswIndex::new(dim, Metric::L2, 16, 200);
    for (id, v) in &items {
        hnsw.add(*id, v);
    }
    for &ef in &[16usize, 32, 64, 128] {
        let h = &hnsw;
        eval(
            &format!("HNSW efSearch={ef}"),
            Box::new(move |q| h.search(q, k, ef)),
        );
    }

    // int8-quantized HNSW (≈1/4 the vector memory; f32 rerank of the beam).
    let mut hnswq = HnswQIndex::new(dim, Metric::L2, 16, 200, true);
    for (id, v) in &items {
        hnswq.add(*id, v);
    }
    for &ef in &[32usize, 64, 128] {
        let h = &hnswq;
        eval(
            &format!("HNSW-q(int8) efSearch={ef}"),
            Box::new(move |q| h.search(q, k, ef)),
        );
    }

    // DiskANN / Vamana (single-layer graph; PQ-steered traversal + f32 rerank).
    let diskann = DiskAnnIndex::build(&items, Metric::L2, 32, 96, 1.2, 16, 256);
    for &l in &[32usize, 64, 128] {
        let d = &diskann;
        eval(
            &format!("DiskANN L={l}"),
            Box::new(move |q| d.search(q, k, l)),
        );
    }
}

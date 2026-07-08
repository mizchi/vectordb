//! Binary (1-bit) quantization with Hamming-distance scan + optional f32 rerank.
//!
//! Each dimension is reduced to its sign bit, packed into `u64` words — 32x
//! smaller than f32. The coarse scan ranks by Hamming distance (`popcount` of
//! XOR), which tracks angular distance for sign-based codes; a widened
//! candidate set is then reranked with exact f32 distances (when the originals
//! are kept). Intended to be used *with* rerank.
//!
//! This is the practical 1-bit path. RaBitQ (SIGMOD'24/'25) is the accuracy
//! upgrade over plain sign bits — a random rotation plus an unbiased,
//! error-bounded distance estimator — and would slot in here as an alternative
//! `encode`/`estimate` pair over the same packed-bit storage.

use crate::distance::{dot_f32, l2sq_f32};
use crate::index::{push_bounded, Hit, Metric, Ranked};
use std::collections::BinaryHeap;

/// A flat index over 1-bit (sign) codes.
pub struct BinaryIndex {
    dim: usize,
    metric: Metric,
    words: usize, // u64 words per vector = ceil(dim / 64)
    ids: Vec<u64>,
    bits: Vec<u64>,        // len = count * words
    raw: Option<Vec<f32>>, // len = count * dim when kept (for rerank)
}

impl BinaryIndex {
    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn metric(&self) -> Metric {
        self.metric
    }
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
    pub fn has_raw(&self) -> bool {
        self.raw.is_some()
    }

    /// Bytes used by the packed codes (excludes ids / kept raw vectors).
    pub fn code_bytes(&self) -> usize {
        self.bits.len() * 8
    }

    /// Build from `(id, vector)` pairs. `keep_raw` keeps f32 originals so the
    /// Hamming candidates can be reranked exactly.
    pub fn build(dim: usize, metric: Metric, items: &[(u64, Vec<f32>)], keep_raw: bool) -> Self {
        assert!(dim > 0);
        let words = dim.div_ceil(64);
        let n = items.len();
        let mut ids = Vec::with_capacity(n);
        let mut bits = vec![0u64; n * words];
        let mut raw = if keep_raw {
            Some(Vec::with_capacity(n * dim))
        } else {
            None
        };
        let cosine = metric == Metric::Cosine;
        for (row, (id, v)) in items.iter().enumerate() {
            assert_eq!(v.len(), dim, "dimension mismatch");
            let processed = if cosine { normalize(v) } else { v.clone() };
            encode_into(&processed, &mut bits[row * words..(row + 1) * words]);
            ids.push(*id);
            if let Some(r) = raw.as_mut() {
                r.extend_from_slice(&processed);
            }
        }
        BinaryIndex {
            dim,
            metric,
            words,
            ids,
            bits,
            raw,
        }
    }

    /// Search for the `k` nearest neighbors. Ranks candidates by Hamming
    /// distance, then reranks `k * oversample` of them with exact f32 distances
    /// when the originals are kept.
    pub fn search(&self, query: &[f32], k: usize, oversample: usize) -> Vec<Hit> {
        if k == 0 || self.is_empty() {
            return Vec::new();
        }
        assert_eq!(query.len(), self.dim, "dimension mismatch");
        let processed = if self.metric == Metric::Cosine {
            normalize(query)
        } else {
            query.to_vec()
        };
        let mut qbits = vec![0u64; self.words];
        encode_into(&processed, &mut qbits);

        let cand_n = if self.raw.is_some() {
            (k * oversample.max(1)).min(self.len())
        } else {
            k.min(self.len())
        };
        let mut heap: BinaryHeap<Ranked> = BinaryHeap::with_capacity(cand_n + 1);
        for i in 0..self.len() {
            let h = hamming(&self.bits[i * self.words..(i + 1) * self.words], &qbits);
            push_bounded(&mut heap, Ranked { key: h as f32, idx: i }, cand_n);
        }

        if self.raw.is_some() && oversample > 1 {
            let mut rr: BinaryHeap<Ranked> = BinaryHeap::with_capacity(k + 1);
            for Ranked { idx, .. } in heap.into_iter() {
                let key = self.exact_key(idx, &processed);
                push_bounded(&mut rr, Ranked { key, idx }, k);
            }
            self.finish(rr, true)
        } else {
            self.finish(heap, false)
        }
    }

    #[inline]
    fn exact_key(&self, i: usize, processed_query: &[f32]) -> f32 {
        let v = &self.raw.as_ref().expect("raw not kept")[i * self.dim..(i + 1) * self.dim];
        match self.metric {
            Metric::L2 => l2sq_f32(v, processed_query),
            Metric::Dot | Metric::Cosine => -dot_f32(v, processed_query),
        }
    }

    /// `reranked`: hits carry exact metric scores; otherwise the raw Hamming
    /// distance (lower = closer).
    fn finish(&self, heap: BinaryHeap<Ranked>, reranked: bool) -> Vec<Hit> {
        let mut items: Vec<Ranked> = heap.into_vec();
        items.sort_by(|a, b| a.key.total_cmp(&b.key));
        let hib = self.metric.higher_is_better();
        items
            .into_iter()
            .map(|Ranked { key, idx }| Hit {
                id: self.ids[idx],
                score: if reranked && hib { -key } else { key },
            })
            .collect()
    }
}

/// Pack sign bits of `v` into `out` (`out.len()` u64 words). Bit set iff x >= 0.
fn encode_into(v: &[f32], out: &mut [u64]) {
    for w in out.iter_mut() {
        *w = 0;
    }
    for (j, &x) in v.iter().enumerate() {
        if x >= 0.0 {
            out[j / 64] |= 1u64 << (j % 64);
        }
    }
}

#[inline]
fn hamming(a: &[u64], b: &[u64]) -> u32 {
    let mut d = 0u32;
    for (x, y) in a.iter().zip(b.iter()) {
        d += (x ^ y).count_ones();
    }
    d
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        v.iter().map(|x| x * inv).collect()
    } else {
        v.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn items() -> Vec<(u64, Vec<f32>)> {
        vec![
            (10, vec![1.0, 0.1, 0.0, 0.0]),
            (20, vec![0.0, 1.0, 0.0, 0.0]),
            (30, vec![0.9, 0.2, 0.0, 0.0]),
            (40, vec![-1.0, -1.0, 0.0, 0.0]),
        ]
    }

    #[test]
    fn compression_is_32x_shape() {
        let idx = BinaryIndex::build(128, Metric::Cosine, &[(0, vec![0.5; 128])], false);
        // 128 dims -> 2 u64 words -> 16 bytes vs 512 bytes of f32.
        assert_eq!(idx.code_bytes(), 16);
    }

    #[test]
    fn cosine_with_rerank_finds_nearest() {
        let idx = BinaryIndex::build(4, Metric::Cosine, &items(), true);
        let hits = idx.search(&[1.0, 0.15, 0.0, 0.0], 2, 4);
        assert_eq!(hits[0].id, 10);
        assert_eq!(hits[1].id, 30);
    }

    #[test]
    fn recall_vs_exact_on_clustered_data() {
        // Binary + rerank should recover exact neighbors on angular data.
        let dim = 64usize;
        let mut s: u64 = 42;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32
        };
        let ncenters = 30usize;
        let centers: Vec<Vec<f32>> = (0..ncenters)
            .map(|_| (0..dim).map(|_| next() * 2.0 - 1.0).collect())
            .collect();
        let items: Vec<(u64, Vec<f32>)> = (0..3000)
            .map(|i| {
                let c = &centers[i % ncenters];
                (i as u64, c.iter().map(|x| x + (next() - 0.5) * 0.2).collect())
            })
            .collect();

        let bin = BinaryIndex::build(dim, Metric::Cosine, &items, true);
        let mut flat = crate::FlatIndex::new(dim, Metric::Cosine, true);
        for (id, v) in &items {
            flat.add(*id, v);
        }

        let mut hit = 0usize;
        let mut total = 0usize;
        for t in 0..40 {
            let c = &centers[t % ncenters];
            let q: Vec<f32> = c.iter().map(|x| x + 0.01).collect();
            let truth: std::collections::HashSet<u64> =
                flat.search_exact(&q, 10).iter().map(|h| h.id).collect();
            let got = bin.search(&q, 10, 16);
            hit += got.iter().filter(|h| truth.contains(&h.id)).count();
            total += truth.len();
        }
        let recall = hit as f64 / total as f64;
        assert!(recall >= 0.90, "recall too low: {recall}");
    }
}

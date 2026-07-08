//! Flat (brute-force) index: int8 quantized scan + optional f32 rerank.
//!
//! Search logic lives on [`View`], a borrowed window over the raw sections, so
//! the exact same code path serves both the owned [`FlatIndex`] and a
//! zero-copy mmap-backed view (see [`crate::storage`]).

use crate::distance::{dot_f32, dot_i8, l2sq_f32};
use crate::quantize::{dequantize, quantize, Quantized};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Similarity / distance metric.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Metric {
    /// Squared Euclidean distance (smaller is better).
    L2 = 0,
    /// Inner product (larger is better).
    Dot = 1,
    /// Cosine similarity (larger is better); vectors are stored L2-normalized.
    Cosine = 2,
}

impl Metric {
    pub fn from_u32(v: u32) -> Option<Metric> {
        match v {
            0 => Some(Metric::L2),
            1 => Some(Metric::Dot),
            2 => Some(Metric::Cosine),
            _ => None,
        }
    }
    #[inline]
    fn higher_is_better(self) -> bool {
        matches!(self, Metric::Dot | Metric::Cosine)
    }
}

/// A single search hit. `score` is the metric value: squared distance for
/// `L2` (smaller better), similarity for `Dot`/`Cosine` (larger better).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub id: u64,
    pub score: f32,
}

/// A borrowed window over an index's sections. All query logic lives here.
pub struct View<'a> {
    pub dim: usize,
    pub metric: Metric,
    pub ids: &'a [u64],
    pub codes: &'a [i8],   // len = count * dim
    pub scales: &'a [f32], // len = count
    pub sqnorms: &'a [f32],
    pub raw: Option<&'a [f32]>, // len = count * dim when present
}

impl<'a> View<'a> {
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
    pub fn has_raw(&self) -> bool {
        self.raw.is_some()
    }

    #[inline]
    fn codes_at(&self, i: usize) -> &[i8] {
        &self.codes[i * self.dim..(i + 1) * self.dim]
    }
    #[inline]
    fn raw_at(&self, i: usize) -> &[f32] {
        let raw = self.raw.expect("raw vectors not present");
        &raw[i * self.dim..(i + 1) * self.dim]
    }

    /// Prepare a query for scanning: normalize for cosine, else copy.
    fn process(&self, v: &[f32]) -> Vec<f32> {
        assert_eq!(v.len(), self.dim, "dimension mismatch");
        if self.metric == Metric::Cosine {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                let inv = 1.0 / norm;
                return v.iter().map(|x| x * inv).collect();
            }
        }
        v.to_vec()
    }

    /// Search for the `k` nearest neighbors. When originals are present and
    /// `oversample > 1`, the int8 scan feeds `k * oversample` candidates into
    /// an exact f32 rerank stage.
    pub fn search(&self, query: &[f32], k: usize, oversample: usize) -> Vec<Hit> {
        if k == 0 || self.is_empty() {
            return Vec::new();
        }
        let processed = self.process(query);
        let q = quantize(&processed);
        let count = self.len();

        let cand_n = if self.raw.is_some() {
            (k * oversample.max(1)).min(count)
        } else {
            k.min(count)
        };
        let mut heap: BinaryHeap<Ranked> = BinaryHeap::with_capacity(cand_n + 1);
        for i in 0..count {
            let dp = dot_i8(self.codes_at(i), &q.codes) as f32 * self.scales[i] * q.scale;
            let key = self.approx_key(i, &q, dp);
            push_bounded(&mut heap, Ranked { key, idx: i }, cand_n);
        }

        if self.raw.is_some() && oversample > 1 {
            let mut rr: BinaryHeap<Ranked> = BinaryHeap::with_capacity(k + 1);
            for Ranked { idx, .. } in heap.into_iter() {
                let key = self.exact_key(idx, &processed);
                push_bounded(&mut rr, Ranked { key, idx }, k);
            }
            self.finish(rr)
        } else {
            self.finish(heap)
        }
    }

    /// Exact full-precision search (requires stored originals).
    pub fn search_exact(&self, query: &[f32], k: usize) -> Vec<Hit> {
        assert!(self.raw.is_some(), "search_exact requires stored originals");
        if k == 0 || self.is_empty() {
            return Vec::new();
        }
        let processed = self.process(query);
        let mut heap: BinaryHeap<Ranked> = BinaryHeap::with_capacity(k + 1);
        for i in 0..self.len() {
            let key = self.exact_key(i, &processed);
            push_bounded(&mut heap, Ranked { key, idx: i }, k);
        }
        self.finish(heap)
    }

    /// Ranking key from the approximate int8 dot product. Lower key = better.
    #[inline]
    fn approx_key(&self, i: usize, q: &Quantized, approx_dot: f32) -> f32 {
        match self.metric {
            Metric::L2 => self.sqnorms[i] + q.sqnorm - 2.0 * approx_dot,
            Metric::Dot | Metric::Cosine => -approx_dot,
        }
    }

    /// Exact ranking key from stored f32 originals. Lower key = better.
    #[inline]
    fn exact_key(&self, i: usize, processed_query: &[f32]) -> f32 {
        let v = self.raw_at(i);
        match self.metric {
            Metric::L2 => l2sq_f32(v, processed_query),
            Metric::Dot | Metric::Cosine => -dot_f32(v, processed_query),
        }
    }

    fn finish(&self, heap: BinaryHeap<Ranked>) -> Vec<Hit> {
        let mut items: Vec<Ranked> = heap.into_vec();
        items.sort_by(|a, b| a.key.total_cmp(&b.key));
        items
            .into_iter()
            .map(|Ranked { key, idx }| Hit {
                id: self.ids[idx],
                score: if self.metric.higher_is_better() {
                    -key
                } else {
                    key
                },
            })
            .collect()
    }

    /// Reconstruct the approximate stored vector `i` from its int8 codes.
    pub fn reconstruct(&self, i: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.dim];
        dequantize(self.codes_at(i), self.scales[i], &mut out);
        out
    }
}

/// Flat index owning int8-quantized vectors, with the originals optionally
/// kept for exact reranking.
pub struct FlatIndex {
    pub(crate) dim: usize,
    pub(crate) metric: Metric,
    pub(crate) ids: Vec<u64>,
    pub(crate) codes: Vec<i8>,
    pub(crate) scales: Vec<f32>,
    pub(crate) sqnorms: Vec<f32>,
    pub(crate) raw: Option<Vec<f32>>,
}

impl FlatIndex {
    /// Create an empty index. `keep_raw` retains the f32 originals to enable
    /// exact reranking; dropping them yields the most compact index but
    /// disables rerank.
    pub fn new(dim: usize, metric: Metric, keep_raw: bool) -> Self {
        assert!(dim > 0, "dim must be positive");
        FlatIndex {
            dim,
            metric,
            ids: Vec::new(),
            codes: Vec::new(),
            scales: Vec::new(),
            sqnorms: Vec::new(),
            raw: if keep_raw { Some(Vec::new()) } else { None },
        }
    }

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

    /// Borrow the index as a [`View`] for querying.
    pub fn view(&self) -> View<'_> {
        View {
            dim: self.dim,
            metric: self.metric,
            ids: &self.ids,
            codes: &self.codes,
            scales: &self.scales,
            sqnorms: &self.sqnorms,
            raw: self.raw.as_deref(),
        }
    }

    /// Add a vector with an external id.
    pub fn add(&mut self, id: u64, vector: &[f32]) {
        assert_eq!(vector.len(), self.dim, "dimension mismatch");
        let processed = if self.metric == Metric::Cosine {
            normalize(vector)
        } else {
            vector.to_vec()
        };
        let q = quantize(&processed);
        self.ids.push(id);
        self.codes.extend_from_slice(&q.codes);
        self.scales.push(q.scale);
        self.sqnorms.push(q.sqnorm);
        if let Some(raw) = self.raw.as_mut() {
            raw.extend_from_slice(&processed);
        }
    }

    /// Search for the `k` nearest neighbors (see [`View::search`]).
    pub fn search(&self, query: &[f32], k: usize, oversample: usize) -> Vec<Hit> {
        self.view().search(query, k, oversample)
    }

    /// Exact full-precision search (requires `keep_raw = true`).
    pub fn search_exact(&self, query: &[f32], k: usize) -> Vec<Hit> {
        self.view().search_exact(query, k)
    }

    /// Reconstruct the approximate stored vector `i` from its int8 codes.
    pub fn reconstruct(&self, i: usize) -> Vec<f32> {
        self.view().reconstruct(i)
    }
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

/// Heap element ordered by `key` (larger key = "worse" = popped first).
struct Ranked {
    key: f32,
    idx: usize,
}
impl PartialEq for Ranked {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}
impl Eq for Ranked {}
impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Ranked {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.total_cmp(&other.key)
    }
}

/// Push into a max-heap capped at `cap`, evicting the current worst.
#[inline]
fn push_bounded(heap: &mut BinaryHeap<Ranked>, item: Ranked, cap: usize) {
    if cap == 0 {
        return;
    }
    if heap.len() < cap {
        heap.push(item);
    } else if let Some(top) = heap.peek() {
        if item.key < top.key {
            heap.pop();
            heap.push(item);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(metric: Metric, keep_raw: bool) -> FlatIndex {
        let mut idx = FlatIndex::new(4, metric, keep_raw);
        idx.add(10, &[1.0, 0.0, 0.0, 0.0]);
        idx.add(20, &[0.0, 1.0, 0.0, 0.0]);
        idx.add(30, &[0.9, 0.1, 0.0, 0.0]);
        idx.add(40, &[0.0, 0.0, 1.0, 0.0]);
        idx
    }

    #[test]
    fn cosine_finds_nearest() {
        let idx = build(Metric::Cosine, true);
        let hits = idx.search(&[1.0, 0.0, 0.0, 0.0], 2, 4);
        assert_eq!(hits[0].id, 10);
        assert_eq!(hits[1].id, 30);
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn l2_finds_nearest() {
        let idx = build(Metric::L2, true);
        let hits = idx.search(&[0.9, 0.1, 0.0, 0.0], 1, 4);
        assert_eq!(hits[0].id, 30);
    }

    #[test]
    fn compact_mode_without_rerank_works() {
        let idx = build(Metric::Dot, false);
        assert!(!idx.has_raw());
        let hits = idx.search(&[1.0, 0.0, 0.0, 0.0], 2, 4);
        assert_eq!(hits[0].id, 10);
    }

    #[test]
    fn rerank_matches_exact_on_small_set() {
        let idx = build(Metric::L2, true);
        let approx = idx.search(&[0.5, 0.5, 0.0, 0.0], 4, 4);
        let exact = idx.search_exact(&[0.5, 0.5, 0.0, 0.0], 4);
        let a: Vec<u64> = approx.iter().map(|h| h.id).collect();
        let e: Vec<u64> = exact.iter().map(|h| h.id).collect();
        assert_eq!(a, e);
    }
}

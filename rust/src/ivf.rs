//! IVF (inverted file) index: k-means coarse quantization + `nprobe` search.
//!
//! Vectors are partitioned into `nlist` cells by k-means. A query scans only
//! the `nprobe` cells whose centroids are nearest to it, instead of the whole
//! dataset. Within the scanned cells it uses the same int8 approximate scan and
//! optional f32 rerank as the flat index.
//!
//! Vectors are stored grouped by cell (CSR layout) so each cell scan walks a
//! contiguous slice — cache-friendly and easy to parallelize later.

use crate::distance::{dot_f32, dot_i8, l2sq_f32};
use crate::index::{push_bounded, Hit, Metric, Ranked};
use crate::quantize::quantize;
use std::collections::BinaryHeap;

/// An IVF index over int8-quantized vectors.
pub struct IvfIndex {
    dim: usize,
    metric: Metric,
    nlist: usize,
    centroids: Vec<f32>, // nlist * dim, in processed (normalized-for-cosine) space
    offsets: Vec<usize>, // nlist + 1, CSR row offsets into the arrays below
    ids: Vec<u64>,
    codes: Vec<i8>,   // count * dim
    scales: Vec<f32>, // count
    sqnorms: Vec<f32>,
    raw: Option<Vec<f32>>, // count * dim when kept
}

impl IvfIndex {
    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn metric(&self) -> Metric {
        self.metric
    }
    pub fn nlist(&self) -> usize {
        self.nlist
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

    /// Build an IVF index from a batch of `(id, vector)` pairs.
    ///
    /// - `nlist`: number of cells (0 → a sensible default of ~`sqrt(n)`).
    /// - `keep_raw`: keep f32 originals for exact reranking.
    /// - `kmeans_iters`: Lloyd iterations for the coarse quantizer.
    pub fn build(
        dim: usize,
        metric: Metric,
        nlist: usize,
        items: &[(u64, Vec<f32>)],
        keep_raw: bool,
        kmeans_iters: usize,
    ) -> IvfIndex {
        assert!(dim > 0, "dim must be positive");
        let n = items.len();
        assert!(n > 0, "cannot build an empty index");

        let nlist = if nlist == 0 {
            ((n as f64).sqrt() as usize).clamp(1, n)
        } else {
            nlist.min(n)
        };

        // Processed vectors (normalized for cosine).
        let cosine = metric == Metric::Cosine;
        let mut proc: Vec<f32> = Vec::with_capacity(n * dim);
        for (_, v) in items {
            assert_eq!(v.len(), dim, "dimension mismatch");
            if cosine {
                proc.extend(normalize(v));
            } else {
                proc.extend_from_slice(v);
            }
        }

        // Coarse quantizer: k-means over the processed vectors.
        let (centroids, assign) = kmeans(&proc, n, dim, nlist, kmeans_iters, cosine);

        // Group vectors by cell (counting sort into CSR layout).
        let mut counts = vec![0usize; nlist];
        for &c in &assign {
            counts[c] += 1;
        }
        let mut offsets = vec![0usize; nlist + 1];
        for c in 0..nlist {
            offsets[c + 1] = offsets[c] + counts[c];
        }
        let mut cursor = offsets.clone();

        let mut ids = vec![0u64; n];
        let mut codes = vec![0i8; n * dim];
        let mut scales = vec![0f32; n];
        let mut sqnorms = vec![0f32; n];
        let mut raw = if keep_raw {
            Some(vec![0f32; n * dim])
        } else {
            None
        };

        for (i, (id, _)) in items.iter().enumerate() {
            let cell = assign[i];
            let dst = cursor[cell];
            cursor[cell] += 1;
            let v = &proc[i * dim..(i + 1) * dim];
            let q = quantize(v);
            ids[dst] = *id;
            codes[dst * dim..(dst + 1) * dim].copy_from_slice(&q.codes);
            scales[dst] = q.scale;
            sqnorms[dst] = q.sqnorm;
            if let Some(r) = raw.as_mut() {
                r[dst * dim..(dst + 1) * dim].copy_from_slice(v);
            }
        }

        IvfIndex {
            dim,
            metric,
            nlist,
            centroids,
            offsets,
            ids,
            codes,
            scales,
            sqnorms,
            raw,
        }
    }

    /// Search the `nprobe` nearest cells for the `k` nearest neighbors.
    /// `oversample` widens the int8 candidate set fed into the f32 rerank.
    pub fn search(&self, query: &[f32], k: usize, nprobe: usize, oversample: usize) -> Vec<Hit> {
        if k == 0 || self.is_empty() {
            return Vec::new();
        }
        assert_eq!(query.len(), self.dim, "dimension mismatch");
        let processed = if self.metric == Metric::Cosine {
            normalize(query)
        } else {
            query.to_vec()
        };
        let q = quantize(&processed);
        let nprobe = nprobe.clamp(1, self.nlist);

        // Stage 1: pick the nprobe nearest centroids.
        let cells = self.nearest_cells(&processed, nprobe);

        // Stage 2: int8 approximate scan over the chosen cells.
        let cand_n = if self.raw.is_some() {
            k * oversample.max(1)
        } else {
            k
        };
        let mut heap: BinaryHeap<Ranked> = BinaryHeap::with_capacity(cand_n + 1);
        for &cell in &cells {
            for i in self.offsets[cell]..self.offsets[cell + 1] {
                let dp =
                    dot_i8(self.codes_at(i), &q.codes) as f32 * self.scales[i] * q.scale;
                let key = match self.metric {
                    Metric::L2 => self.sqnorms[i] + q.sqnorm - 2.0 * dp,
                    Metric::Dot | Metric::Cosine => -dp,
                };
                push_bounded(&mut heap, Ranked { key, idx: i }, cand_n);
            }
        }

        // Stage 3: exact f32 rerank when originals are kept.
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

    fn nearest_cells(&self, processed_query: &[f32], nprobe: usize) -> Vec<usize> {
        let mut heap: BinaryHeap<Ranked> = BinaryHeap::with_capacity(nprobe + 1);
        for c in 0..self.nlist {
            let cen = &self.centroids[c * self.dim..(c + 1) * self.dim];
            let key = match self.metric {
                Metric::L2 => l2sq_f32(cen, processed_query),
                Metric::Dot | Metric::Cosine => -dot_f32(cen, processed_query),
            };
            push_bounded(&mut heap, Ranked { key, idx: c }, nprobe);
        }
        heap.into_iter().map(|r| r.idx).collect()
    }

    #[inline]
    fn codes_at(&self, i: usize) -> &[i8] {
        &self.codes[i * self.dim..(i + 1) * self.dim]
    }

    #[inline]
    fn exact_key(&self, i: usize, processed_query: &[f32]) -> f32 {
        let v = &self.raw.as_ref().expect("raw not kept")[i * self.dim..(i + 1) * self.dim];
        match self.metric {
            Metric::L2 => l2sq_f32(v, processed_query),
            Metric::Dot | Metric::Cosine => -dot_f32(v, processed_query),
        }
    }

    fn finish(&self, heap: BinaryHeap<Ranked>) -> Vec<Hit> {
        let mut items: Vec<Ranked> = heap.into_vec();
        items.sort_by(|a, b| a.key.total_cmp(&b.key));
        let hib = self.metric.higher_is_better();
        items
            .into_iter()
            .map(|Ranked { key, idx }| Hit {
                id: self.ids[idx],
                score: if hib { -key } else { key },
            })
            .collect()
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

/// Lloyd's k-means over row-major `data` (`n * dim`). Returns
/// `(centroids [nlist*dim], assignment [n])`.
fn kmeans(
    data: &[f32],
    n: usize,
    dim: usize,
    nlist: usize,
    iters: usize,
    renorm: bool,
) -> (Vec<f32>, Vec<usize>) {
    // Init: pick nlist points spread across the dataset (deterministic).
    let mut centroids = vec![0f32; nlist * dim];
    let stride = (n / nlist).max(1);
    for c in 0..nlist {
        let src = (c * stride) % n;
        centroids[c * dim..(c + 1) * dim].copy_from_slice(&data[src * dim..(src + 1) * dim]);
    }

    let mut assign = vec![0usize; n];
    for _ in 0..iters.max(1) {
        // Assignment step (nearest centroid by L2 in processed space).
        assign_points(data, n, dim, &centroids, nlist, &mut assign);

        // Update step: mean of assigned points.
        let mut sums = vec![0f32; nlist * dim];
        let mut counts = vec![0usize; nlist];
        for i in 0..n {
            let c = assign[i];
            counts[c] += 1;
            let row = &data[i * dim..(i + 1) * dim];
            let dst = &mut sums[c * dim..(c + 1) * dim];
            for (s, x) in dst.iter_mut().zip(row.iter()) {
                *s += x;
            }
        }
        for c in 0..nlist {
            if counts[c] == 0 {
                // Empty cluster: reseed to a pseudo-random point.
                let src = (c * 2654435761) % n;
                centroids[c * dim..(c + 1) * dim]
                    .copy_from_slice(&data[src * dim..(src + 1) * dim]);
                continue;
            }
            let inv = 1.0 / counts[c] as f32;
            let cen = &mut centroids[c * dim..(c + 1) * dim];
            let sum = &sums[c * dim..(c + 1) * dim];
            for (d, s) in cen.iter_mut().zip(sum.iter()) {
                *d = s * inv;
            }
            if renorm {
                let norm = cen.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    let ninv = 1.0 / norm;
                    for d in cen.iter_mut() {
                        *d *= ninv;
                    }
                }
            }
        }
    }
    (centroids, assign)
}

#[cfg(feature = "parallel")]
fn assign_points(
    data: &[f32],
    n: usize,
    dim: usize,
    centroids: &[f32],
    nlist: usize,
    assign: &mut [usize],
) {
    use rayon::prelude::*;
    assign
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, a)| {
            *a = nearest_centroid(&data[i * dim..(i + 1) * dim], centroids, nlist, dim);
        });
    let _ = n;
}

#[cfg(not(feature = "parallel"))]
fn assign_points(
    data: &[f32],
    n: usize,
    dim: usize,
    centroids: &[f32],
    nlist: usize,
    assign: &mut [usize],
) {
    for i in 0..n {
        assign[i] = nearest_centroid(&data[i * dim..(i + 1) * dim], centroids, nlist, dim);
    }
}

#[inline]
fn nearest_centroid(row: &[f32], centroids: &[f32], nlist: usize, dim: usize) -> usize {
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for c in 0..nlist {
        let d = l2sq_f32(&centroids[c * dim..(c + 1) * dim], row);
        if d < best_d {
            best_d = d;
            best = c;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_items() -> Vec<(u64, Vec<f32>)> {
        // Four tight clusters in 2D.
        let mut v = Vec::new();
        let centers = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)];
        let mut id = 0u64;
        for (cx, cy) in centers {
            for j in 0..25 {
                let jitter = (j as f32) * 0.01;
                v.push((id, vec![cx + jitter, cy - jitter]));
                id += 1;
            }
        }
        v
    }

    #[test]
    fn ivf_l2_finds_local_cluster() {
        let items = make_items();
        let idx = IvfIndex::build(2, Metric::L2, 4, &items, true, 15);
        // Query near the (10,10) cluster.
        let hits = idx.search(&[10.0, 10.0], 3, 2, 4);
        assert_eq!(hits.len(), 3);
        // All returned ids should belong to the 4th cluster (ids 75..100).
        for h in &hits {
            assert!(h.id >= 75, "unexpected id {}", h.id);
        }
    }

    #[test]
    fn ivf_matches_flat_with_full_nprobe() {
        use crate::FlatIndex;
        let items = make_items();
        let ivf = IvfIndex::build(2, Metric::L2, 4, &items, true, 20);
        let mut flat = FlatIndex::new(2, Metric::L2, true);
        for (id, v) in &items {
            flat.add(*id, v);
        }
        let q = [7.0, 9.0];
        // With nprobe = nlist, IVF scans everything → top-1 must match flat.
        let a = ivf.search(&q, 1, 4, 8);
        let b = flat.search(&q, 1, 8);
        assert_eq!(a[0].id, b[0].id);
    }

    #[test]
    fn ivf_cosine_runs() {
        let items = make_items();
        let idx = IvfIndex::build(2, Metric::Cosine, 4, &items, true, 10);
        let hits = idx.search(&[10.0, 10.0], 5, 4, 4);
        assert_eq!(hits.len(), 5);
    }
}

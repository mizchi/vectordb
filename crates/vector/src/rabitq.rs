//! RaBitQ (Gao & Long, SIGMOD 2024): 1-bit quantization with a theoretical
//! error bound — much better accuracy than plain sign bits at the same storage.
//!
//! Idea: subtract a centroid `c`, apply a fixed random rotation `P`, then keep
//! only the sign of each rotated residual (1 bit/dim). The rotation spreads
//! information so that an *unbiased* estimator of the inner product can be
//! recovered from the sign code plus one per-vector correction scalar.
//!
//! For residual `r = P(o - c)` with sign code `b = sign(r)` and query residual
//! `qr = P(q - c)`, this implementation uses the estimator
//!
//! ```text
//!   <o - c, q - c>  ≈  coef * S ,   S = <b, qr>,   coef = ||r||^2 / sum|r_i|
//! ```
//!
//! which is exact when the query lies along the data direction. The database
//! keeps 1 bit/dim (32x vs f32) plus three f32 scalars/vector; the query stays
//! full precision, so recall is far higher than Hamming-only binary codes.
//!
//! Note: the coarse scan here is O(dim) per vector (sign bit × f32 query). The
//! bit-packed low-bit query + popcount trick from the paper (which makes the
//! scan faster than int8) is a further optimization not done here — the win
//! demonstrated is accuracy at 1-bit storage.

use crate::distance::{dot_f32, l2sq_f32};
use crate::index::{push_bounded, Hit, Metric, Ranked};
use std::collections::BinaryHeap;

/// A RaBitQ index (1-bit codes + per-vector correction scalars).
pub struct RabitqIndex {
    dim: usize,
    metric: Metric,
    words: usize,  // u64 words per vector
    rot: Vec<f32>, // dim * dim row-major random rotation
    centroid: Vec<f32>,
    cc: f32, // <c, c>
    ids: Vec<u64>,
    signs: Vec<u64>,       // count * words: sign bits of rotated residuals
    popcnt: Vec<u32>,      // number of set sign bits per vector
    res_sq: Vec<f32>,      // ||o - c||^2
    coef: Vec<f32>,        // ||r||^2 / sum|r_i|
    oc: Vec<f32>,          // <o, c> (for dot/cosine reconstruction)
    raw: Option<Vec<f32>>, // originals for exact rerank
}

/// Bits used to quantize the query for the popcount estimator.
const QUERY_BITS: usize = 4;

impl RabitqIndex {
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
    /// Bytes used by the packed 1-bit codes (excludes scalars / raw).
    pub fn code_bytes(&self) -> usize {
        self.signs.len() * 8
    }

    /// Build from `(id, vector)` pairs. `seed` fixes the random rotation.
    pub fn build(
        dim: usize,
        metric: Metric,
        items: &[(u64, Vec<f32>)],
        keep_raw: bool,
        seed: u64,
    ) -> Self {
        assert!(dim > 0);
        let n = items.len();
        assert!(n > 0, "cannot build an empty index");
        let words = dim.div_ceil(64);
        let cosine = metric == Metric::Cosine;

        // Processed vectors (normalized for cosine).
        let mut proc: Vec<Vec<f32>> = Vec::with_capacity(n);
        for (_, v) in items {
            assert_eq!(v.len(), dim, "dimension mismatch");
            proc.push(if cosine { normalize(v) } else { v.clone() });
        }

        // Centroid.
        let mut centroid = vec![0f32; dim];
        for v in &proc {
            for (c, x) in centroid.iter_mut().zip(v.iter()) {
                *c += x;
            }
        }
        let inv_n = 1.0 / n as f32;
        for c in centroid.iter_mut() {
            *c *= inv_n;
        }
        let cc = dot(&centroid, &centroid);
        let rot = random_rotation(dim, seed);

        let mut ids = Vec::with_capacity(n);
        let mut signs = vec![0u64; n * words];
        let mut popcnt = vec![0u32; n];
        let mut res_sq = vec![0f32; n];
        let mut coef = vec![0f32; n];
        let mut oc = vec![0f32; n];
        let mut raw = if keep_raw {
            Some(Vec::with_capacity(n * dim))
        } else {
            None
        };

        let mut d = vec![0f32; dim];
        let mut r = vec![0f32; dim];
        for (row, ((id, _), o)) in items.iter().zip(proc.iter()).enumerate() {
            for j in 0..dim {
                d[j] = o[j] - centroid[j];
            }
            matvec(&rot, &d, &mut r, dim);
            let rsq = dot(&r, &r);
            let mut sum_abs = 0f32;
            let mut pc = 0u32;
            for j in 0..dim {
                sum_abs += r[j].abs();
                if r[j] >= 0.0 {
                    signs[row * words + j / 64] |= 1u64 << (j % 64);
                    pc += 1;
                }
            }
            popcnt[row] = pc;
            res_sq[row] = rsq;
            coef[row] = if sum_abs > 0.0 { rsq / sum_abs } else { 0.0 };
            oc[row] = dot(o, &centroid);
            ids.push(*id);
            if let Some(rw) = raw.as_mut() {
                rw.extend_from_slice(o);
            }
        }

        RabitqIndex {
            dim,
            metric,
            words,
            rot,
            centroid,
            cc,
            ids,
            signs,
            popcnt,
            res_sq,
            coef,
            oc,
            raw,
        }
    }

    /// Search for the `k` nearest neighbors using the RaBitQ estimator, then
    /// rerank `k * oversample` candidates with exact f32 distances (if kept).
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
        // Query residual, rotated.
        let mut qd = vec![0f32; self.dim];
        for j in 0..self.dim {
            qd[j] = processed[j] - self.centroid[j];
        }
        let mut qr = vec![0f32; self.dim];
        matvec(&self.rot, &qd, &mut qr, self.dim);
        let qr_sq = dot(&qd, &qd);
        let qc = dot(&processed, &self.centroid);
        let l2 = self.metric == Metric::L2;

        // Quantize the rotated query into QUERY_BITS bitplanes so <b, qr> can be
        // estimated with a few popcounts per vector (the RaBitQ fast scan).
        let qq = QuantizedQuery::new(&qr, self.words, self.dim);

        let cand_n = if self.raw.is_some() {
            (k * oversample.max(1)).min(self.len())
        } else {
            k.min(self.len())
        };
        let mut heap: BinaryHeap<Ranked> = BinaryHeap::with_capacity(cand_n + 1);
        for i in 0..self.len() {
            let s = qq.estimate_sign_dot(
                &self.signs[i * self.words..(i + 1) * self.words],
                self.popcnt[i],
            );
            let ip_c = self.coef[i] * s; // estimate of <o-c, q-c>
            let key = if l2 {
                self.res_sq[i] + qr_sq - 2.0 * ip_c
            } else {
                // dot / cosine: <o,q> = <o-c,q-c> + <o,c> + <c,q> - <c,c>
                -(ip_c + self.oc[i] + qc - self.cc)
            };
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

/// A query quantized into `QUERY_BITS` bitplanes for the popcount estimator.
///
/// `S = <sign(r), qr>` is recovered from the database sign bits `b` via
/// `qr_j ≈ vl + delta * q_u[j]`:
/// `S ≈ delta * (2·<b, q_u> - Σq_u) + vl · (2·popcount(b) - D)`,
/// where `<b, q_u> = Σ_t 2^t · Σ_words popcount(b & plane_t)`.
struct QuantizedQuery {
    planes: Vec<u64>, // QUERY_BITS * words
    words: usize,
    dim: usize,
    vl: f32,
    delta: f32,
    total_qu: f32,
}

impl QuantizedQuery {
    fn new(qr: &[f32], words: usize, dim: usize) -> Self {
        let mut vl = f32::INFINITY;
        let mut vmax = f32::NEG_INFINITY;
        for &x in qr {
            vl = vl.min(x);
            vmax = vmax.max(x);
        }
        let levels = ((1usize << QUERY_BITS) - 1) as f32;
        let delta = ((vmax - vl) / levels).max(1e-9);
        let mut planes = vec![0u64; QUERY_BITS * words];
        let mut total_qu = 0f32;
        for (j, &x) in qr.iter().enumerate() {
            let qu = (((x - vl) / delta).round() as i32).clamp(0, levels as i32) as u32;
            total_qu += qu as f32;
            for (t, plane) in planes.chunks_mut(words).enumerate() {
                if (qu >> t) & 1 == 1 {
                    plane[j / 64] |= 1u64 << (j % 64);
                }
            }
        }
        QuantizedQuery {
            planes,
            words,
            dim,
            vl,
            delta,
            total_qu,
        }
    }

    #[inline]
    fn estimate_sign_dot(&self, signs: &[u64], popcnt: u32) -> f32 {
        let mut inner = 0u64; // <b, q_u>
        for (t, plane) in self.planes.chunks(self.words).enumerate() {
            let mut pc = 0u32;
            for w in 0..self.words {
                pc += (signs[w] & plane[w]).count_ones();
            }
            inner += (pc as u64) << t;
        }
        self.delta * (2.0 * inner as f32 - self.total_qu)
            + self.vl * (2.0 * popcnt as f32 - self.dim as f32)
    }
}

/// IVF + RaBitQ: per-cell centroids make the residuals small and well spread,
/// which is where RaBitQ's estimator becomes accurate. This is the canonical
/// high-recall / low-memory configuration.
pub struct IvfRabitqIndex {
    dim: usize,
    metric: Metric,
    nlist: usize,
    words: usize,
    rot: Vec<f32>,       // dim * dim rotation
    centroids: Vec<f32>, // nlist * dim
    cc: Vec<f32>,        // <c, c> per cell
    offsets: Vec<usize>, // nlist + 1
    ids: Vec<u64>,
    signs: Vec<u64>, // count * words
    popcnt: Vec<u32>,
    res_sq: Vec<f32>, // ||v - c_cell||^2
    coef: Vec<f32>,   // ||r||^2 / sum|r_i|
    oc: Vec<f32>,     // <v, c_cell>
    raw: Option<Vec<f32>>,
}

impl IvfRabitqIndex {
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
    pub fn code_bytes(&self) -> usize {
        self.signs.len() * 8
    }

    /// Build an IVF+RaBitQ index. `nlist == 0` → ~sqrt(n) cells.
    pub fn build(
        dim: usize,
        metric: Metric,
        nlist: usize,
        items: &[(u64, Vec<f32>)],
        keep_raw: bool,
        kmeans_iters: usize,
        seed: u64,
    ) -> Self {
        assert!(dim > 0);
        let n = items.len();
        assert!(n > 0, "cannot build an empty index");
        let nlist = if nlist == 0 {
            ((n as f64).sqrt() as usize).clamp(1, n)
        } else {
            nlist.min(n)
        };
        let cosine = metric == Metric::Cosine;
        let words = dim.div_ceil(64);

        // Processed vectors, flattened.
        let mut proc = vec![0f32; n * dim];
        for (i, (_, v)) in items.iter().enumerate() {
            assert_eq!(v.len(), dim, "dimension mismatch");
            if cosine {
                proc[i * dim..(i + 1) * dim].copy_from_slice(&normalize(v));
            } else {
                proc[i * dim..(i + 1) * dim].copy_from_slice(v);
            }
        }

        let (centroids, assign) = crate::ivf::kmeans(&proc, n, dim, nlist, kmeans_iters, cosine);
        let mut cc = vec![0f32; nlist];
        for c in 0..nlist {
            let cen = &centroids[c * dim..(c + 1) * dim];
            cc[c] = dot(cen, cen);
        }

        // CSR grouping by cell.
        let mut counts = vec![0usize; nlist];
        for &c in &assign {
            counts[c] += 1;
        }
        let mut offsets = vec![0usize; nlist + 1];
        for c in 0..nlist {
            offsets[c + 1] = offsets[c] + counts[c];
        }
        let mut cursor = offsets.clone();

        let rot = random_rotation(dim, seed);
        let mut ids = vec![0u64; n];
        let mut signs = vec![0u64; n * words];
        let mut popcnt = vec![0u32; n];
        let mut res_sq = vec![0f32; n];
        let mut coef = vec![0f32; n];
        let mut oc = vec![0f32; n];
        let mut raw = if keep_raw {
            Some(vec![0f32; n * dim])
        } else {
            None
        };

        let mut d = vec![0f32; dim];
        let mut r = vec![0f32; dim];
        for (i, (id, _)) in items.iter().enumerate() {
            let cell = assign[i];
            let dst = cursor[cell];
            cursor[cell] += 1;
            let v = &proc[i * dim..(i + 1) * dim];
            let cen = &centroids[cell * dim..(cell + 1) * dim];
            for j in 0..dim {
                d[j] = v[j] - cen[j];
            }
            matvec(&rot, &d, &mut r, dim);
            let rsq = dot(&r, &r);
            let mut sum_abs = 0f32;
            let mut pc = 0u32;
            for (j, &rj) in r.iter().enumerate() {
                sum_abs += rj.abs();
                if rj >= 0.0 {
                    signs[dst * words + j / 64] |= 1u64 << (j % 64);
                    pc += 1;
                }
            }
            popcnt[dst] = pc;
            res_sq[dst] = rsq;
            coef[dst] = if sum_abs > 0.0 { rsq / sum_abs } else { 0.0 };
            oc[dst] = dot(v, cen);
            ids[dst] = *id;
            if let Some(rw) = raw.as_mut() {
                rw[dst * dim..(dst + 1) * dim].copy_from_slice(v);
            }
        }

        IvfRabitqIndex {
            dim,
            metric,
            nlist,
            words,
            rot,
            centroids,
            cc,
            offsets,
            ids,
            signs,
            popcnt,
            res_sq,
            coef,
            oc,
            raw,
        }
    }

    /// Search the `nprobe` nearest cells with the RaBitQ estimator, then rerank.
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
        let nprobe = nprobe.clamp(1, self.nlist);
        let l2 = self.metric == Metric::L2;

        // Pick the nprobe nearest cells.
        let mut cell_heap: BinaryHeap<Ranked> = BinaryHeap::with_capacity(nprobe + 1);
        for c in 0..self.nlist {
            let cen = &self.centroids[c * self.dim..(c + 1) * self.dim];
            let key = if l2 {
                l2sq_f32(cen, &processed)
            } else {
                -dot_f32(cen, &processed)
            };
            push_bounded(&mut cell_heap, Ranked { key, idx: c }, nprobe);
        }
        let cells: Vec<usize> = cell_heap.into_iter().map(|r| r.idx).collect();

        let cand_n = if self.raw.is_some() {
            (k * oversample.max(1)).min(self.len())
        } else {
            k.min(self.len())
        };
        let mut heap: BinaryHeap<Ranked> = BinaryHeap::with_capacity(cand_n + 1);
        let mut qd = vec![0f32; self.dim];
        let mut qr = vec![0f32; self.dim];
        for &c in &cells {
            let cen = &self.centroids[c * self.dim..(c + 1) * self.dim];
            for j in 0..self.dim {
                qd[j] = processed[j] - cen[j];
            }
            matvec(&self.rot, &qd, &mut qr, self.dim);
            let qr_sq = dot(&qd, &qd);
            let qc = dot(&processed, cen);
            let qq = QuantizedQuery::new(&qr, self.words, self.dim);
            for i in self.offsets[c]..self.offsets[c + 1] {
                let s = qq.estimate_sign_dot(
                    &self.signs[i * self.words..(i + 1) * self.words],
                    self.popcnt[i],
                );
                let ip_c = self.coef[i] * s;
                let key = if l2 {
                    self.res_sq[i] + qr_sq - 2.0 * ip_c
                } else {
                    -(ip_c + self.oc[i] + qc - self.cc[c])
                };
                push_bounded(&mut heap, Ranked { key, idx: i }, cand_n);
            }
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

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn matvec(m: &[f32], v: &[f32], out: &mut [f32], d: usize) {
    for i in 0..d {
        let row = &m[i * d..(i + 1) * d];
        out[i] = dot(row, v);
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

/// A random orthonormal `d x d` matrix (row-major) via Gram-Schmidt on
/// seeded Gaussian rows.
fn random_rotation(d: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed | 1);
    let mut m = vec![0f32; d * d];
    for x in m.iter_mut() {
        *x = rng.next_gauss();
    }
    for i in 0..d {
        // Orthogonalize row i against previous rows.
        for p in 0..i {
            let mut proj = 0f32;
            for j in 0..d {
                proj += m[i * d + j] * m[p * d + j];
            }
            for j in 0..d {
                m[i * d + j] -= proj * m[p * d + j];
            }
        }
        let mut norm = 0f32;
        for j in 0..d {
            norm += m[i * d + j] * m[i * d + j];
        }
        let inv = 1.0 / norm.sqrt().max(1e-12);
        for j in 0..d {
            m[i * d + j] *= inv;
        }
    }
    m
}

/// Small xorshift RNG with a Box-Muller Gaussian.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32).clamp(1e-7, 1.0 - 1e-7)
    }
    fn next_gauss(&mut self) -> f32 {
        let u1 = self.next_f32();
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

#[cfg(feature = "parallel")]
impl RabitqIndex {
    /// Run many queries concurrently (one query per rayon task).
    pub fn search_batch(&self, queries: &[Vec<f32>], k: usize, oversample: usize) -> Vec<Vec<Hit>> {
        use rayon::prelude::*;
        queries
            .par_iter()
            .map(|q| self.search(q, k, oversample))
            .collect()
    }
}

#[cfg(feature = "parallel")]
impl IvfRabitqIndex {
    /// Run many queries concurrently (one query per rayon task).
    pub fn search_batch(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        nprobe: usize,
        oversample: usize,
    ) -> Vec<Vec<Hit>> {
        use rayon::prelude::*;
        queries
            .par_iter()
            .map(|q| self.search(q, k, nprobe, oversample))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::type_complexity)]
    fn clustered(dim: usize, n: usize, ncenters: usize) -> (Vec<(u64, Vec<f32>)>, Vec<Vec<f32>>) {
        let mut s: u64 = 7;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32
        };
        let centers: Vec<Vec<f32>> = (0..ncenters)
            .map(|_| (0..dim).map(|_| next() * 2.0 - 1.0).collect())
            .collect();
        let items = (0..n)
            .map(|i| {
                let c = &centers[i % ncenters];
                (
                    i as u64,
                    c.iter().map(|x| x + (next() - 0.5) * 0.2).collect(),
                )
            })
            .collect();
        (items, centers)
    }

    #[test]
    fn rabitq_rerank_high_recall() {
        let dim = 64;
        let (items, centers) = clustered(dim, 3000, 30);
        let idx = RabitqIndex::build(dim, Metric::Cosine, &items, true, 0xBEEF);
        let mut flat = crate::FlatIndex::new(dim, Metric::Cosine, true);
        for (id, v) in &items {
            flat.add(*id, v);
        }
        let mut hit = 0;
        let mut total = 0;
        for t in 0..40 {
            let c = &centers[t % centers.len()];
            let q: Vec<f32> = c.iter().map(|x| x + 0.01).collect();
            let truth: std::collections::HashSet<u64> =
                flat.search_exact(&q, 10).iter().map(|h| h.id).collect();
            let got = idx.search(&q, 10, 16);
            hit += got.iter().filter(|h| truth.contains(&h.id)).count();
            total += truth.len();
        }
        let recall = hit as f64 / total as f64;
        assert!(recall >= 0.95, "recall too low: {recall}");
    }

    #[test]
    fn rabitq_estimator_recovers_near_identity() {
        // The estimator (no rerank) should rank a lightly-perturbed copy of a
        // stored vector near the top — a direct check of estimator fidelity on
        // well-separated points.
        let dim = 128;
        let mut s: u64 = 99;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        };
        let items: Vec<(u64, Vec<f32>)> = (0..1000)
            .map(|i| (i as u64, (0..dim).map(|_| next()).collect()))
            .collect();
        let rq = RabitqIndex::build(dim, Metric::Cosine, &items, false, 5);

        let mut hit = 0;
        let probes = 50;
        for t in 0..probes {
            let (id, base) = &items[t * 7 % items.len()];
            let q: Vec<f32> = base.iter().map(|x| x + next() * 0.02).collect();
            // The perturbed original should appear in the estimator's top-10.
            if rq.search(&q, 10, 1).iter().any(|h| h.id == *id) {
                hit += 1;
            }
        }
        let recall = hit as f64 / probes as f64;
        assert!(
            recall >= 0.8,
            "estimator near-identity recall too low: {recall}"
        );
    }

    #[test]
    fn ivf_rabitq_beats_flat_rabitq_estimator() {
        // Per-cell centroids should give the estimator a clear edge over a
        // single global centroid on clustered data (no rerank, oversample=1).
        let dim = 64;
        let (items, centers) = clustered(dim, 4000, 40);
        let flat = RabitqIndex::build(dim, Metric::Cosine, &items, false, 7);
        let ivf = IvfRabitqIndex::build(dim, Metric::Cosine, 40, &items, false, 12, 7);
        let mut truth_idx = crate::FlatIndex::new(dim, Metric::Cosine, true);
        for (id, v) in &items {
            truth_idx.add(*id, v);
        }
        let (mut flat_hit, mut ivf_hit, mut total) = (0, 0, 0);
        for t in 0..40 {
            let c = &centers[t % centers.len()];
            let q: Vec<f32> = c.iter().map(|x| x + 0.03).collect();
            let truth: std::collections::HashSet<u64> = truth_idx
                .search_exact(&q, 10)
                .iter()
                .map(|h| h.id)
                .collect();
            flat_hit += flat
                .search(&q, 10, 1)
                .iter()
                .filter(|h| truth.contains(&h.id))
                .count();
            ivf_hit += ivf
                .search(&q, 10, 8, 1)
                .iter()
                .filter(|h| truth.contains(&h.id))
                .count();
            total += truth.len();
        }
        let (fr, ir) = (
            flat_hit as f64 / total as f64,
            ivf_hit as f64 / total as f64,
        );
        assert!(ir >= fr, "IVF+RaBitQ {ir} should beat flat RaBitQ {fr}");
    }

    #[test]
    fn ivf_rabitq_high_recall_with_rerank() {
        let dim = 64;
        let (items, centers) = clustered(dim, 4000, 40);
        let ivf = IvfRabitqIndex::build(dim, Metric::Cosine, 40, &items, true, 12, 1);
        let mut truth_idx = crate::FlatIndex::new(dim, Metric::Cosine, true);
        for (id, v) in &items {
            truth_idx.add(*id, v);
        }
        let (mut hit, mut total) = (0, 0);
        for t in 0..40 {
            let c = &centers[t % centers.len()];
            let q: Vec<f32> = c.iter().map(|x| x + 0.01).collect();
            let truth: std::collections::HashSet<u64> = truth_idx
                .search_exact(&q, 10)
                .iter()
                .map(|h| h.id)
                .collect();
            hit += ivf
                .search(&q, 10, 8, 16)
                .iter()
                .filter(|h| truth.contains(&h.id))
                .count();
            total += truth.len();
        }
        let recall = hit as f64 / total as f64;
        assert!(recall >= 0.95, "recall too low: {recall}");
    }

    #[test]
    fn rabitq_l2_runs() {
        let dim = 16;
        let (items, _) = clustered(dim, 500, 10);
        let idx = RabitqIndex::build(dim, Metric::L2, &items, true, 3);
        let hits = idx.search(&items[0].1, 5, 8);
        assert_eq!(hits.len(), 5);
        assert_eq!(hits[0].id, 0); // nearest to itself
    }
}

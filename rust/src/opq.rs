//! Optimized Product Quantization (OPQ; Ge et al., CVPR 2013).
//!
//! Plain PQ quantizes fixed slices of the vector, so its error depends on how
//! energy happens to be distributed across dimensions. OPQ first learns an
//! orthonormal rotation `R` that redistributes that energy to make the subspaces
//! easier to quantize, then runs PQ on `R x`. Because `R` is orthonormal it
//! preserves L2 distances and inner products, so search is just "rotate the
//! query, then ADC" — identical machinery to [`crate::pq::PqIndex`] on rotated
//! data.
//!
//! `R` is learned by the non-parametric alternating optimization from the
//! paper: repeatedly (1) train PQ on the currently-rotated data and reconstruct
//! it, then (2) update `R` as the orthogonal Procrustes solution aligning the
//! raw data to those reconstructions. Step 2 needs an SVD of a `dim × dim`
//! matrix, computed here via a self-contained Jacobi eigensolver (no external
//! linear-algebra dependency).

use crate::distance::{dot_f32, l2sq_f32};
use crate::index::{push_bounded, Hit, Metric, Ranked};
use crate::pq::{adc_sum, build_lut, encode_vector, train_codebooks};
use std::collections::BinaryHeap;
use std::io;
use std::path::Path;

/// An OPQ index: a learned rotation `R` followed by PQ on the rotated vectors.
pub struct OpqIndex {
    dim: usize,
    metric: Metric,
    m: usize,
    dsub: usize,
    ksub: usize,
    count: usize,
    rot: Vec<f32>,          // dim * dim orthonormal (row-major); y = R x
    pq_codebooks: Vec<f32>, // m * ksub * dsub, trained on rotated data
    codes: Vec<u8>,         // count * m
    ids: Vec<u64>,
    raw: Option<Vec<f32>>, // count * dim (processed, un-rotated) for rerank
}

impl OpqIndex {
    /// Build an OPQ index. `opq_iters` alternating rotation/PQ refinement passes
    /// (2–5 is typical; 0 falls back to an identity rotation = plain PQ).
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        vectors: &[(u64, Vec<f32>)],
        metric: Metric,
        m: usize,
        ksub: usize,
        kmeans_iters: usize,
        opq_iters: usize,
        keep_raw: bool,
    ) -> OpqIndex {
        assert!(!vectors.is_empty(), "OpqIndex::build: empty input");
        let dim = vectors[0].1.len();
        assert!(m > 0 && dim.is_multiple_of(m), "m must divide dim");
        assert!(ksub > 0 && ksub <= 256, "ksub must be in 1..=256");
        let dsub = dim / m;
        let n = vectors.len();
        let ksub = ksub.min(n);
        let cosine = metric == Metric::Cosine;

        // Processed (normalized for cosine) contiguous matrix.
        let mut data = vec![0f32; n * dim];
        let mut ids = Vec::with_capacity(n);
        for (i, (id, v)) in vectors.iter().enumerate() {
            assert_eq!(v.len(), dim, "inconsistent dimension");
            ids.push(*id);
            if cosine {
                let p = normalize(v);
                data[i * dim..(i + 1) * dim].copy_from_slice(&p);
            } else {
                data[i * dim..(i + 1) * dim].copy_from_slice(v);
            }
        }

        // Learn the rotation, then train the final codebooks on rotated data.
        let rot = learn_rotation(&data, n, dim, m, ksub, kmeans_iters, opq_iters);
        let mut rotated = vec![0f32; n * dim];
        for i in 0..n {
            rotate(
                &rot,
                &data[i * dim..(i + 1) * dim],
                &mut rotated[i * dim..(i + 1) * dim],
                dim,
            );
        }
        let pq_codebooks = train_codebooks(&rotated, n, dim, m, ksub, kmeans_iters);
        let mut codes = vec![0u8; n * m];
        for i in 0..n {
            encode_vector(
                &pq_codebooks,
                dim,
                m,
                ksub,
                &rotated[i * dim..(i + 1) * dim],
                &mut codes[i * m..(i + 1) * m],
            );
        }

        let raw = if keep_raw { Some(data) } else { None };
        OpqIndex {
            dim,
            metric,
            m,
            dsub,
            ksub,
            count: n,
            rot,
            pq_codebooks,
            codes,
            ids,
            raw,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn metric(&self) -> Metric {
        self.metric
    }
    pub fn len(&self) -> usize {
        self.count
    }
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    pub fn has_raw(&self) -> bool {
        self.raw.is_some()
    }
    pub fn code_bytes(&self) -> usize {
        self.codes.len()
    }

    /// Search for the `k` nearest neighbors (rotate the query, then ADC + rerank).
    pub fn search(&self, query: &[f32], k: usize, oversample: usize) -> Vec<Hit> {
        self.search_filter(query, k, oversample, |_| true)
    }

    /// Filtered search: only ids satisfying `filter` enter the candidate set.
    pub fn search_filter<F: Fn(u64) -> bool>(
        &self,
        query: &[f32],
        k: usize,
        oversample: usize,
        filter: F,
    ) -> Vec<Hit> {
        if k == 0 || self.count == 0 {
            return Vec::new();
        }
        assert_eq!(query.len(), self.dim, "dimension mismatch");
        let processed = if self.metric == Metric::Cosine {
            normalize(query)
        } else {
            query.to_vec()
        };
        let mut rq = vec![0f32; self.dim];
        rotate(&self.rot, &processed, &mut rq, self.dim);
        let l2 = self.metric == Metric::L2;
        let lut = build_lut(&self.pq_codebooks, self.dim, self.m, self.ksub, &rq, l2);

        let cand_n = if self.raw.is_some() {
            (k * oversample.max(1)).min(self.count)
        } else {
            k.min(self.count)
        };
        let mut heap: BinaryHeap<Ranked> = BinaryHeap::with_capacity(cand_n + 1);
        for i in 0..self.count {
            if !filter(self.ids[i]) {
                continue;
            }
            let acc = adc_sum(&lut, self.ksub, &self.codes[i * self.m..(i + 1) * self.m]);
            let key = if l2 { acc } else { -acc };
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
}

/// Apply the row-major `dim × dim` rotation `r` to `x`: `out = R x`.
fn rotate(r: &[f32], x: &[f32], out: &mut [f32], dim: usize) {
    for (i, o) in out.iter_mut().enumerate() {
        let row = &r[i * dim..(i + 1) * dim];
        let mut s = 0f32;
        for (a, b) in row.iter().zip(x.iter()) {
            s += a * b;
        }
        *o = s;
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

/// Learn the OPQ rotation by non-parametric alternating optimization.
fn learn_rotation(
    data: &[f32],
    n: usize,
    dim: usize,
    m: usize,
    ksub: usize,
    kmeans_iters: usize,
    opq_iters: usize,
) -> Vec<f32> {
    // R starts at identity.
    let mut rot = vec![0f32; dim * dim];
    for i in 0..dim {
        rot[i * dim + i] = 1.0;
    }
    let mut rotated = vec![0f32; n * dim];
    let mut recon = vec![0f32; n * dim];
    for _ in 0..opq_iters {
        // (1) Rotate data, train PQ, reconstruct in the rotated space.
        for i in 0..n {
            rotate(
                &rot,
                &data[i * dim..(i + 1) * dim],
                &mut rotated[i * dim..(i + 1) * dim],
                dim,
            );
        }
        let cb = train_codebooks(&rotated, n, dim, m, ksub, kmeans_iters);
        let dsub = dim / m;
        let mut code = vec![0u8; m];
        for i in 0..n {
            encode_vector(
                &cb,
                dim,
                m,
                ksub,
                &rotated[i * dim..(i + 1) * dim],
                &mut code,
            );
            for j in 0..m {
                let base = (j * ksub + code[j] as usize) * dsub;
                recon[i * dim + j * dsub..i * dim + (j + 1) * dsub]
                    .copy_from_slice(&cb[base..base + dsub]);
            }
        }
        // (2) Orthogonal Procrustes: minimize ||R data_i - recon_i|| over
        // orthonormal R. With C = sum_i data_i recon_i^T (= dataᵀ·recon), the
        // solution is R = V Uᵀ where C = U S Vᵀ.
        let c = gram_cross(data, &recon, n, dim);
        rot = procrustes_rotation(&c, dim);
    }
    rot
}

/// `C = dataᵀ · recon` (a `dim × dim` matrix), both inputs row-major `n × dim`.
fn gram_cross(data: &[f32], recon: &[f32], n: usize, dim: usize) -> Vec<f32> {
    let mut c = vec![0f32; dim * dim];
    for i in 0..n {
        let a = &data[i * dim..(i + 1) * dim];
        let b = &recon[i * dim..(i + 1) * dim];
        for r in 0..dim {
            let ar = a[r];
            let dst = &mut c[r * dim..(r + 1) * dim];
            for (d, &bc) in dst.iter_mut().zip(b.iter()) {
                *d += ar * bc;
            }
        }
    }
    c
}

/// Orthogonal Procrustes rotation for `C` (`dim × dim`): returns `R = V Uᵀ`
/// where `C = U S Vᵀ`. `V` is the eigenvector matrix of `CᵀC`; the paired left
/// factor `U` is formed by Gram-Schmidt on the columns `C·vₖ` (processed in
/// descending-eigenvalue order). For well-conditioned directions this equals
/// `C·vₖ / σₖ`; for degenerate (`σₖ ≈ 0`) directions it yields an arbitrary
/// orthonormal completion, so the returned `R` is always exactly orthonormal.
fn procrustes_rotation(c: &[f32], dim: usize) -> Vec<f32> {
    // A = Cᵀ C (symmetric, dim × dim).
    let mut a = vec![0f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut s = 0f32;
            for k in 0..dim {
                s += c[k * dim + i] * c[k * dim + j];
            }
            a[i * dim + j] = s;
        }
    }
    let (eval, v) = jacobi_eigen(&a, dim);

    // Process columns from largest eigenvalue to smallest so that the
    // well-conditioned directions fix the basis before degenerate ones.
    let mut order: Vec<usize> = (0..dim).collect();
    order.sort_by(|&i, &j| eval[j].total_cmp(&eval[i]));

    let mut u = vec![0f32; dim * dim]; // u[row * dim + k] = k-th left vector
    let mut basis: Vec<Vec<f32>> = Vec::with_capacity(dim);
    for &k in &order {
        // w = C v_k.
        let mut w = vec![0f32; dim];
        for (row, wr) in w.iter_mut().enumerate() {
            let mut s = 0f32;
            for t in 0..dim {
                s += c[row * dim + t] * v[t * dim + k];
            }
            *wr = s;
        }
        gram_schmidt(&mut w, &basis);
        let norm = w.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 {
            for x in w.iter_mut() {
                *x /= norm;
            }
        } else {
            // Degenerate direction: any unit vector orthogonal to the basis.
            w = orthonormal_complement(&basis, dim);
        }
        for row in 0..dim {
            u[row * dim + k] = w[row];
        }
        basis.push(w);
    }

    // R = V Uᵀ  =>  R[i][j] = sum_k V[i][k] * U[j][k].
    let mut r = vec![0f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut s = 0f32;
            for k in 0..dim {
                s += v[i * dim + k] * u[j * dim + k];
            }
            r[i * dim + j] = s;
        }
    }
    r
}

/// Modified Gram-Schmidt: subtract the projection of `w` onto each basis vector.
fn gram_schmidt(w: &mut [f32], basis: &[Vec<f32>]) {
    for b in basis {
        let dp: f32 = w.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        for (x, y) in w.iter_mut().zip(b.iter()) {
            *x -= dp * y;
        }
    }
}

/// A unit vector orthogonal to every vector in `basis` (uses the first standard
/// basis direction that survives orthogonalization).
fn orthonormal_complement(basis: &[Vec<f32>], dim: usize) -> Vec<f32> {
    for i in 0..dim {
        let mut e = vec![0f32; dim];
        e[i] = 1.0;
        gram_schmidt(&mut e, basis);
        let norm = e.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 {
            for x in e.iter_mut() {
                *x /= norm;
            }
            return e;
        }
    }
    // Should be unreachable while basis.len() < dim.
    let mut e = vec![0f32; dim];
    e[0] = 1.0;
    e
}

/// Classic cyclic Jacobi eigensolver for a symmetric `dim × dim` matrix `a`
/// (row-major). Returns `(eigenvalues, eigenvectors)` where eigenvector `k` is
/// column `k` of the returned matrix (`v[i * dim + k]`).
fn jacobi_eigen(a: &[f32], dim: usize) -> (Vec<f32>, Vec<f32>) {
    // Work in f64 for numerical stability during the sweeps.
    let mut m: Vec<f64> = a.iter().map(|&x| x as f64).collect();
    let mut v = vec![0f64; dim * dim];
    for i in 0..dim {
        v[i * dim + i] = 1.0;
    }
    for _sweep in 0..100 {
        // Sum of squared off-diagonal entries; stop when negligible.
        let mut off = 0f64;
        for p in 0..dim {
            for q in (p + 1)..dim {
                off += m[p * dim + q] * m[p * dim + q];
            }
        }
        if off < 1e-18 {
            break;
        }
        for p in 0..dim {
            for q in (p + 1)..dim {
                let apq = m[p * dim + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = m[p * dim + p];
                let aqq = m[q * dim + q];
                // Rotation angle that zeroes m[p][q].
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let cph = 1.0 / (t * t + 1.0).sqrt();
                let sph = t * cph;
                // Apply the rotation to rows/cols p, q of m.
                for i in 0..dim {
                    let mip = m[i * dim + p];
                    let miq = m[i * dim + q];
                    m[i * dim + p] = cph * mip - sph * miq;
                    m[i * dim + q] = sph * mip + cph * miq;
                }
                for i in 0..dim {
                    let mpi = m[p * dim + i];
                    let mqi = m[q * dim + i];
                    m[p * dim + i] = cph * mpi - sph * mqi;
                    m[q * dim + i] = sph * mpi + cph * mqi;
                }
                // Accumulate the rotation into V.
                for i in 0..dim {
                    let vip = v[i * dim + p];
                    let viq = v[i * dim + q];
                    v[i * dim + p] = cph * vip - sph * viq;
                    v[i * dim + q] = sph * vip + cph * viq;
                }
            }
        }
    }
    let eval: Vec<f32> = (0..dim).map(|i| m[i * dim + i] as f32).collect();
    let vf: Vec<f32> = v.iter().map(|&x| x as f32).collect();
    (eval, vf)
}

// ---------------------------------------------------------------------------
// Persistence: an OPQ `.vecdb` file (magic "VECDBOP1").
// Header (64B) + rot f32 + pq_codebooks f32 + ids u64 + codes u8 + raw f32?.
// ---------------------------------------------------------------------------

const OP_MAGIC: &[u8; 8] = b"VECDBOP1";
const OP_VERSION: u32 = 1;
const OP_FLAG_HAS_RAW: u32 = 1;

#[inline]
fn align16(x: usize) -> usize {
    (x + 15) & !15
}

struct OpLayout {
    rot: usize,
    codebooks: usize,
    ids: usize,
    codes: usize,
    raw: usize,
    total: usize,
}

impl OpqIndex {
    fn layout(&self) -> OpLayout {
        let dim = self.dim;
        let count = self.count;
        let rot = align16(64);
        let codebooks = align16(rot + dim * dim * 4);
        let ids = align16(codebooks + self.m * self.ksub * self.dsub * 4);
        let codes = align16(ids + count * 8);
        let raw = align16(codes + count * self.m);
        let total = if self.raw.is_some() {
            align16(raw + count * dim * 4)
        } else {
            raw
        };
        OpLayout {
            rot,
            codebooks,
            ids,
            codes,
            raw,
            total,
        }
    }

    /// Serialize the index to an OPQ `.vecdb` file.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let l = self.layout();
        let has_raw = self.raw.is_some();
        let mut b = vec![0u8; l.total];
        b[0..8].copy_from_slice(OP_MAGIC);
        b[8..12].copy_from_slice(&OP_VERSION.to_le_bytes());
        b[12..16].copy_from_slice(&(self.metric as u32).to_le_bytes());
        b[16..20].copy_from_slice(&(self.dim as u32).to_le_bytes());
        b[20..24].copy_from_slice(&(self.count as u32).to_le_bytes());
        b[24..28].copy_from_slice(&(if has_raw { OP_FLAG_HAS_RAW } else { 0 }).to_le_bytes());
        b[28..32].copy_from_slice(&(self.m as u32).to_le_bytes());
        b[32..36].copy_from_slice(&(self.ksub as u32).to_le_bytes());

        write_f32s(&mut b, l.rot, &self.rot);
        write_f32s(&mut b, l.codebooks, &self.pq_codebooks);
        for (i, &id) in self.ids.iter().enumerate() {
            b[l.ids + i * 8..l.ids + i * 8 + 8].copy_from_slice(&id.to_le_bytes());
        }
        b[l.codes..l.codes + self.codes.len()].copy_from_slice(&self.codes);
        if let Some(rb) = self.raw.as_ref() {
            write_f32s(&mut b, l.raw, rb);
        }
        std::fs::write(path, &b)
    }

    /// Load an OPQ `.vecdb` file.
    pub fn load(path: impl AsRef<Path>) -> io::Result<OpqIndex> {
        let b = std::fs::read(path)?;
        let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, format!("opq: {m}"));
        if b.len() < 64 || &b[0..8] != OP_MAGIC {
            return Err(bad("bad magic"));
        }
        let u32_at = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        if u32_at(8) != OP_VERSION {
            return Err(bad("unsupported version"));
        }
        let metric = Metric::from_u32(u32_at(12)).ok_or_else(|| bad("bad metric"))?;
        let dim = u32_at(16) as usize;
        let count = u32_at(20) as usize;
        let has_raw = u32_at(24) & OP_FLAG_HAS_RAW != 0;
        let m = u32_at(28) as usize;
        let ksub = u32_at(32) as usize;
        if dim == 0 || m == 0 || !dim.is_multiple_of(m) {
            return Err(bad("bad dim/m"));
        }
        let dsub = dim / m;
        let mut idx = OpqIndex {
            dim,
            metric,
            m,
            dsub,
            ksub,
            count,
            rot: Vec::new(),
            pq_codebooks: Vec::new(),
            codes: Vec::new(),
            ids: Vec::new(),
            raw: if has_raw { Some(Vec::new()) } else { None },
        };
        let l = idx.layout();
        if b.len() < l.total {
            return Err(bad("file truncated"));
        }
        idx.rot = read_f32s(&b, l.rot, dim * dim);
        idx.pq_codebooks = read_f32s(&b, l.codebooks, m * ksub * dsub);
        idx.ids = (0..count)
            .map(|i| {
                let o = l.ids + i * 8;
                let mut a = [0u8; 8];
                a.copy_from_slice(&b[o..o + 8]);
                u64::from_le_bytes(a)
            })
            .collect();
        idx.codes = b[l.codes..l.codes + count * m].to_vec();
        if has_raw {
            idx.raw = Some(read_f32s(&b, l.raw, count * dim));
        }
        Ok(idx)
    }
}

fn write_f32s(b: &mut [u8], off: usize, data: &[f32]) {
    for (i, &x) in data.iter().enumerate() {
        b[off + i * 4..off + i * 4 + 4].copy_from_slice(&x.to_le_bytes());
    }
}

fn read_f32s(b: &[u8], off: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let o = off + i * 4;
            f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Data with strongly correlated / unbalanced dimensions, where a rotation
    /// helps PQ the most.
    fn skewed(n: usize, dim: usize) -> Vec<(u64, Vec<f32>)> {
        let mut s: u64 = 0x0DDF_00D5_1234;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        };
        (0..n)
            .map(|i| {
                // A few latent factors spread across all dims with per-dim scale
                // that varies a lot -> unbalanced subspace energy.
                let f0 = next();
                let f1 = next();
                let f2 = next();
                let v: Vec<f32> = (0..dim)
                    .map(|d| {
                        let scale = ((d % 8) as f32 + 1.0) * 0.3;
                        (f0 * ((d as f32) * 0.1).sin()
                            + f1 * ((d as f32) * 0.05).cos()
                            + f2 * 0.5
                            + next() * 0.05)
                            * scale
                    })
                    .collect();
                (i as u64, v)
            })
            .collect()
    }

    #[test]
    fn jacobi_eigen_reconstructs_symmetric() {
        // A = V diag(λ) Vᵀ should hold for the returned decomposition.
        let dim = 5;
        let items = skewed(200, dim);
        // Build a symmetric PSD matrix (a Gram matrix) to decompose.
        let mut a = vec![0f32; dim * dim];
        for (_, v) in &items {
            for i in 0..dim {
                for j in 0..dim {
                    a[i * dim + j] += v[i] * v[j];
                }
            }
        }
        let (eval, vec_) = jacobi_eigen(&a, dim);
        // Reconstruct and compare.
        let mut recon = vec![0f32; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                let mut s = 0f32;
                for k in 0..dim {
                    s += vec_[i * dim + k] * eval[k] * vec_[j * dim + k];
                }
                recon[i * dim + j] = s;
            }
        }
        for (x, y) in a.iter().zip(recon.iter()) {
            assert!((x - y).abs() < 1e-2, "eigendecomp mismatch: {x} vs {y}");
        }
    }

    #[test]
    fn opq_rotation_is_orthonormal() {
        let dim = 16;
        let items = skewed(1000, dim);
        let idx = OpqIndex::build(&items, Metric::L2, 8, 64, 12, 3, false);
        // R Rᵀ ≈ I.
        for i in 0..dim {
            for j in 0..dim {
                let mut s = 0f32;
                for k in 0..dim {
                    s += idx.rot[i * dim + k] * idx.rot[j * dim + k];
                }
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((s - expect).abs() < 1e-2, "not orthonormal at {i},{j}: {s}");
            }
        }
    }

    #[test]
    fn opq_recall_beats_or_matches_plain_pq() {
        let dim = 32;
        let items = skewed(3000, dim);
        let flat = {
            let mut f = crate::FlatIndex::new(dim, Metric::L2, true);
            for (id, v) in &items {
                f.add(*id, v);
            }
            f
        };
        let recall = |search: &dyn Fn(&[f32]) -> Vec<Hit>| {
            let mut hit = 0;
            let mut total = 0;
            for t in 0..40 {
                let q = &items[t * 19 % items.len()].1;
                let truth: std::collections::HashSet<u64> =
                    flat.search_exact(q, 10).iter().map(|h| h.id).collect();
                hit += search(q).iter().filter(|h| truth.contains(&h.id)).count();
                total += truth.len();
            }
            hit as f64 / total as f64
        };
        // Compare coarse (no rerank) recall so the rotation's effect is visible.
        let pq = crate::PqIndex::build(&items, Metric::L2, 8, 256, 15, false);
        let opq = OpqIndex::build(&items, Metric::L2, 8, 256, 15, 4, false);
        let pq_recall = recall(&|q| pq.search(q, 10, 1));
        let opq_recall = recall(&|q| opq.search(q, 10, 1));
        // OPQ should not be worse than plain PQ on skewed data (usually better).
        assert!(
            opq_recall + 0.02 >= pq_recall,
            "OPQ ({opq_recall}) unexpectedly worse than PQ ({pq_recall})"
        );
    }

    #[test]
    fn opq_filtered_search_restricts_ids() {
        let dim = 16;
        let items = skewed(1000, dim);
        let opq = OpqIndex::build(&items, Metric::L2, 8, 64, 12, 3, true);
        let got = opq.search_filter(&items[0].1, 10, 8, |id| id % 4 == 0);
        assert!(!got.is_empty());
        assert!(got.iter().all(|h| h.id % 4 == 0));
    }

    #[test]
    fn opq_rerank_high_recall_and_roundtrip() {
        let dim = 32;
        let items = skewed(2000, dim);
        let idx = OpqIndex::build(&items, Metric::L2, 8, 256, 15, 4, true);
        let flat = {
            let mut f = crate::FlatIndex::new(dim, Metric::L2, true);
            for (id, v) in &items {
                f.add(*id, v);
            }
            f
        };
        let mut hit = 0;
        let mut total = 0;
        for t in 0..30 {
            let q = &items[t * 17 % items.len()].1;
            let truth: std::collections::HashSet<u64> =
                flat.search_exact(q, 10).iter().map(|h| h.id).collect();
            hit += idx
                .search(q, 10, 16)
                .iter()
                .filter(|h| truth.contains(&h.id))
                .count();
            total += truth.len();
        }
        assert!(hit as f64 / total as f64 >= 0.90);

        // Save/load round-trip returns identical results.
        let mut path = std::env::temp_dir();
        path.push("vecdb_opq_test.vecdb");
        idx.save(&path).unwrap();
        let loaded = OpqIndex::load(&path).unwrap();
        for t in 0..10 {
            let q = &items[t * 41 % items.len()].1;
            let a: Vec<u64> = idx.search(q, 10, 16).iter().map(|h| h.id).collect();
            let b: Vec<u64> = loaded.search(q, 10, 16).iter().map(|h| h.id).collect();
            assert_eq!(a, b);
        }
        std::fs::remove_file(&path).ok();
    }
}

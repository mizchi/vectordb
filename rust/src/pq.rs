//! Product Quantization (PQ; Jégou et al., 2011).
//!
//! Split each `dim`-vector into `m` contiguous subvectors of length
//! `dsub = dim / m`, and quantize every subvector independently against a small
//! per-subspace codebook (`ksub` centroids learned by k-means). A vector is
//! then stored as `m` codes — one byte per subspace when `ksub <= 256` — for a
//! `dim * 4 / m` compression over raw f32 (e.g. 128-D f32 → 16 bytes at
//! `m = 16`, a 32x reduction).
//!
//! Search uses **ADC** (asymmetric distance computation): the query stays in
//! full precision. We precompute a lookup table of query-subvector ↔ centroid
//! distances (`m * ksub` entries), then each database vector's approximate
//! distance is just `m` table lookups summed — no per-dimension math in the
//! inner loop. As with the other indexes, keeping the f32 originals enables an
//! exact rerank of a widened candidate set.

use crate::distance::{dot_f32, l2sq_f32};
use crate::index::{push_bounded, Hit, Metric, Ranked};
use std::collections::BinaryHeap;
use std::io::{self, Write};
use std::path::Path;

// ---------------------------------------------------------------------------
// Shared PQ primitives, reused by `PqIndex`, `OpqIndex`, and `IvfPqIndex`.
// ---------------------------------------------------------------------------

/// Train `m` per-subspace codebooks (each `ksub` centroids of `dim/m`) over
/// row-major `data` (`n * dim`). Returns the `m * ksub * dsub` codebook block,
/// laid out `[subspace][centroid][component]`. `ksub` must already be `<= n`.
pub(crate) fn train_codebooks(
    data: &[f32],
    n: usize,
    dim: usize,
    m: usize,
    ksub: usize,
    iters: usize,
) -> Vec<f32> {
    let dsub = dim / m;
    let mut codebooks = vec![0f32; m * ksub * dsub];
    let mut sub = vec![0f32; n * dsub];
    for j in 0..m {
        for i in 0..n {
            let src = &data[i * dim + j * dsub..i * dim + j * dsub + dsub];
            sub[i * dsub..(i + 1) * dsub].copy_from_slice(src);
        }
        // renorm=false: per-subspace renormalization would distort the subvector
        // geometry (cosine is handled by whole-vector normalization upstream).
        let (centroids, _) = crate::ivf::kmeans(&sub, n, dsub, ksub, iters, false);
        codebooks[j * ksub * dsub..(j + 1) * ksub * dsub].copy_from_slice(&centroids);
    }
    codebooks
}

/// Encode one vector into `m` PQ codes (nearest centroid per subspace by L2).
pub(crate) fn encode_vector(
    codebooks: &[f32],
    dim: usize,
    m: usize,
    ksub: usize,
    v: &[f32],
    out: &mut [u8],
) {
    let dsub = dim / m;
    for j in 0..m {
        let vs = &v[j * dsub..(j + 1) * dsub];
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for c in 0..ksub {
            let base = (j * ksub + c) * dsub;
            let d = l2sq_f32(vs, &codebooks[base..base + dsub]);
            if d < best_d {
                best_d = d;
                best = c;
            }
        }
        out[j] = best as u8;
    }
}

/// Build an ADC lookup table for `query` against `codebooks`:
/// `lut[j * ksub + c]` is the subspace-`j`, centroid-`c` contribution
/// (squared-L2 when `l2`, else dot product).
pub(crate) fn build_lut(
    codebooks: &[f32],
    dim: usize,
    m: usize,
    ksub: usize,
    query: &[f32],
    l2: bool,
) -> Vec<f32> {
    let dsub = dim / m;
    let mut lut = vec![0f32; m * ksub];
    for j in 0..m {
        let qs = &query[j * dsub..(j + 1) * dsub];
        for c in 0..ksub {
            let base = (j * ksub + c) * dsub;
            let cen = &codebooks[base..base + dsub];
            lut[j * ksub + c] = if l2 {
                l2sq_f32(qs, cen)
            } else {
                dot_f32(qs, cen)
            };
        }
    }
    lut
}

/// Sum an ADC table over one vector's `m` codes.
#[inline]
pub(crate) fn adc_sum(lut: &[f32], ksub: usize, code: &[u8]) -> f32 {
    let mut acc = 0f32;
    for (j, &c) in code.iter().enumerate() {
        acc += lut[j * ksub + c as usize];
    }
    acc
}

/// A Product Quantization index.
pub struct PqIndex {
    dim: usize,
    metric: Metric,
    m: usize,    // number of subspaces
    dsub: usize, // dim / m
    ksub: usize, // centroids per subspace (<= 256)
    count: usize,
    /// Codebooks, laid out `[subspace][centroid][component]`
    /// (`m * ksub * dsub` f32).
    codebooks: Vec<f32>,
    /// PQ codes, `count * m` bytes (row `i` at `i * m`).
    codes: Vec<u8>,
    ids: Vec<u64>,
    /// f32 originals (processed) for exact rerank, when kept.
    raw: Option<Vec<f32>>,
}

impl PqIndex {
    /// Build a PQ index. `m` must divide `dim`; `ksub` is the codebook size per
    /// subspace (must be `<= 256` so a code fits in one byte). `kmeans_iters`
    /// bounds the codebook training. `keep_raw` retains f32 originals for exact
    /// reranking.
    pub fn build(
        vectors: &[(u64, Vec<f32>)],
        metric: Metric,
        m: usize,
        ksub: usize,
        kmeans_iters: usize,
        keep_raw: bool,
    ) -> PqIndex {
        assert!(!vectors.is_empty(), "PqIndex::build: empty input");
        let dim = vectors[0].1.len();
        assert!(m > 0 && dim.is_multiple_of(m), "m must divide dim");
        assert!(ksub > 0 && ksub <= 256, "ksub must be in 1..=256");
        let dsub = dim / m;
        let n = vectors.len();
        let ksub = ksub.min(n); // can't learn more centroids than points

        // Processed (normalized for cosine) contiguous matrix.
        let cos = metric == Metric::Cosine;
        let mut data = vec![0f32; n * dim];
        let mut ids = Vec::with_capacity(n);
        for (i, (id, v)) in vectors.iter().enumerate() {
            assert_eq!(v.len(), dim, "inconsistent dimension");
            ids.push(*id);
            if cos {
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                let inv = if norm > 0.0 { 1.0 / norm } else { 0.0 };
                for (d, &x) in data[i * dim..(i + 1) * dim].iter_mut().zip(v.iter()) {
                    *d = x * inv;
                }
            } else {
                data[i * dim..(i + 1) * dim].copy_from_slice(v);
            }
        }

        let codebooks = train_codebooks(&data, n, dim, m, ksub, kmeans_iters);
        let mut codes = vec![0u8; n * m];
        for i in 0..n {
            encode_vector(
                &codebooks,
                dim,
                m,
                ksub,
                &data[i * dim..(i + 1) * dim],
                &mut codes[i * m..(i + 1) * m],
            );
        }

        let raw = if keep_raw { Some(data) } else { None };
        PqIndex {
            dim,
            metric,
            m,
            dsub,
            ksub,
            count: n,
            codebooks,
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
    /// Bytes used by the PQ codes (excludes codebooks and any kept originals).
    pub fn code_bytes(&self) -> usize {
        self.codes.len()
    }

    /// Approximate ranking key for row `i` via the ADC table (lower = better).
    #[inline]
    fn adc_key(&self, i: usize, lut: &[f32]) -> f32 {
        let acc = adc_sum(lut, self.ksub, &self.codes[i * self.m..(i + 1) * self.m]);
        // L2 accumulates squared distance (smaller better); dot accumulates
        // similarity, so negate for a "smaller = better" key.
        match self.metric {
            Metric::L2 => acc,
            Metric::Dot | Metric::Cosine => -acc,
        }
    }

    #[inline]
    fn exact_key(&self, i: usize, processed_query: &[f32]) -> f32 {
        let raw = self.raw.as_ref().expect("raw not kept");
        let v = &raw[i * self.dim..(i + 1) * self.dim];
        match self.metric {
            Metric::L2 => l2sq_f32(v, processed_query),
            Metric::Dot | Metric::Cosine => -dot_f32(v, processed_query),
        }
    }

    fn process(&self, query: &[f32]) -> Vec<f32> {
        assert_eq!(query.len(), self.dim, "dimension mismatch");
        if self.metric == Metric::Cosine {
            let norm = query.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                let inv = 1.0 / norm;
                return query.iter().map(|x| x * inv).collect();
            }
        }
        query.to_vec()
    }

    /// Search for the `k` nearest neighbors with ADC. When originals are kept
    /// and `oversample > 1`, `k * oversample` ADC candidates are reranked with
    /// exact f32 distances.
    pub fn search(&self, query: &[f32], k: usize, oversample: usize) -> Vec<Hit> {
        self.search_filter(query, k, oversample, |_| true)
    }

    /// Filtered search: only ids satisfying `filter` enter the candidate set
    /// (applied during the ADC scan, so filtered-out vectors never compete).
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
        let processed = self.process(query);
        let lut = build_lut(
            &self.codebooks,
            self.dim,
            self.m,
            self.ksub,
            &processed,
            self.metric == Metric::L2,
        );
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
            push_bounded(
                &mut heap,
                Ranked {
                    key: self.adc_key(i, &lut),
                    idx: i,
                },
                cand_n,
            );
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

// ---------------------------------------------------------------------------
// Persistence: a PQ `.vecdb` file (magic "VECDBPQ1").
// Header (64B) + codebooks f32 + ids u64 + codes u8 + raw f32? (16B-aligned).
// ---------------------------------------------------------------------------

const PQ_MAGIC: &[u8; 8] = b"VECDBPQ1";
const PQ_VERSION: u32 = 1;
const PQ_HEADER_LEN: usize = 64;
const PQ_FLAG_HAS_RAW: u32 = 1;

#[inline]
fn align16(x: usize) -> usize {
    (x + 15) & !15
}

struct PqLayout {
    codebooks: usize,
    ids: usize,
    codes: usize,
    raw: usize,
    total: usize,
}

fn pq_layout(
    dim: usize,
    count: usize,
    m: usize,
    ksub: usize,
    dsub: usize,
    has_raw: bool,
) -> PqLayout {
    let codebooks = align16(PQ_HEADER_LEN);
    let ids = align16(codebooks + m * ksub * dsub * 4);
    let codes = align16(ids + count * 8);
    let raw = align16(codes + count * m);
    let total = if has_raw {
        align16(raw + count * dim * 4)
    } else {
        raw
    };
    PqLayout {
        codebooks,
        ids,
        codes,
        raw,
        total,
    }
}

impl PqIndex {
    /// Serialize the index to a PQ `.vecdb` file.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let has_raw = self.raw.is_some();
        let l = pq_layout(self.dim, self.count, self.m, self.ksub, self.dsub, has_raw);
        let mut b = vec![0u8; l.total];
        b[0..8].copy_from_slice(PQ_MAGIC);
        b[8..12].copy_from_slice(&PQ_VERSION.to_le_bytes());
        b[12..16].copy_from_slice(&(self.metric as u32).to_le_bytes());
        b[16..20].copy_from_slice(&(self.dim as u32).to_le_bytes());
        b[20..24].copy_from_slice(&(self.count as u32).to_le_bytes());
        b[24..28].copy_from_slice(&(if has_raw { PQ_FLAG_HAS_RAW } else { 0 }).to_le_bytes());
        b[28..32].copy_from_slice(&(self.m as u32).to_le_bytes());
        b[32..36].copy_from_slice(&(self.ksub as u32).to_le_bytes());

        for (i, &x) in self.codebooks.iter().enumerate() {
            let o = l.codebooks + i * 4;
            b[o..o + 4].copy_from_slice(&x.to_le_bytes());
        }
        for (i, &id) in self.ids.iter().enumerate() {
            let o = l.ids + i * 8;
            b[o..o + 8].copy_from_slice(&id.to_le_bytes());
        }
        b[l.codes..l.codes + self.codes.len()].copy_from_slice(&self.codes);
        if let Some(raw) = self.raw.as_ref() {
            for (i, &x) in raw.iter().enumerate() {
                let o = l.raw + i * 4;
                b[o..o + 4].copy_from_slice(&x.to_le_bytes());
            }
        }
        let mut f = std::fs::File::create(path)?;
        f.write_all(&b)?;
        f.flush()?;
        Ok(())
    }

    /// Load a PQ `.vecdb` file.
    pub fn load(path: impl AsRef<Path>) -> io::Result<PqIndex> {
        let b = std::fs::read(path)?;
        let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, format!("pq: {m}"));
        if b.len() < PQ_HEADER_LEN || &b[0..8] != PQ_MAGIC {
            return Err(bad("bad magic"));
        }
        let u32_at = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        if u32_at(8) != PQ_VERSION {
            return Err(bad("unsupported version"));
        }
        let metric = Metric::from_u32(u32_at(12)).ok_or_else(|| bad("bad metric"))?;
        let dim = u32_at(16) as usize;
        let count = u32_at(20) as usize;
        let has_raw = u32_at(24) & PQ_FLAG_HAS_RAW != 0;
        let m = u32_at(28) as usize;
        let ksub = u32_at(32) as usize;
        if dim == 0 || m == 0 || !dim.is_multiple_of(m) {
            return Err(bad("bad dim/m"));
        }
        let dsub = dim / m;
        let l = pq_layout(dim, count, m, ksub, dsub, has_raw);
        if b.len() < l.total {
            return Err(bad("file truncated"));
        }

        let codebooks: Vec<f32> = (0..m * ksub * dsub)
            .map(|i| {
                let o = l.codebooks + i * 4;
                f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
            })
            .collect();
        let ids: Vec<u64> = (0..count)
            .map(|i| {
                let o = l.ids + i * 8;
                let mut a = [0u8; 8];
                a.copy_from_slice(&b[o..o + 8]);
                u64::from_le_bytes(a)
            })
            .collect();
        let codes = b[l.codes..l.codes + count * m].to_vec();
        let raw = if has_raw {
            Some(
                (0..count * dim)
                    .map(|i| {
                        let o = l.raw + i * 4;
                        f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
                    })
                    .collect(),
            )
        } else {
            None
        };
        Ok(PqIndex {
            dim,
            metric,
            m,
            dsub,
            ksub,
            count,
            codebooks,
            codes,
            ids,
            raw,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Clustered data: PQ should recover the exact neighbors well after rerank.
    fn clustered(n: usize, dim: usize, ncenters: usize) -> Vec<(u64, Vec<f32>)> {
        let mut s: u64 = 0xDEAD_BEEF_1234;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        };
        let centers: Vec<Vec<f32>> = (0..ncenters)
            .map(|_| (0..dim).map(|_| next()).collect())
            .collect();
        (0..n)
            .map(|i| {
                let c = &centers[i % ncenters];
                let v: Vec<f32> = c.iter().map(|&x| x + next() * 0.1).collect();
                (i as u64, v)
            })
            .collect()
    }

    #[test]
    fn pq_rerank_high_recall_vs_exact() {
        let dim = 64;
        let items = clustered(2000, dim, 20);
        let pq = PqIndex::build(&items, Metric::L2, 16, 256, 15, true);
        assert_eq!(pq.code_bytes(), 2000 * 16); // 16 bytes/vector
        let mut flat = crate::FlatIndex::new(dim, Metric::L2, true);
        for (id, v) in &items {
            flat.add(*id, v);
        }
        let mut hit = 0;
        let mut total = 0;
        for t in 0..30 {
            let q = &items[t * 17 % items.len()].1;
            let got = pq.search(q, 10, 16);
            let truth: std::collections::HashSet<u64> =
                flat.search_exact(q, 10).iter().map(|h| h.id).collect();
            hit += got.iter().filter(|h| truth.contains(&h.id)).count();
            total += truth.len();
        }
        let recall = hit as f64 / total as f64;
        assert!(recall >= 0.90, "PQ recall too low: {recall}");
    }

    #[test]
    fn pq_cosine_finds_self() {
        let dim = 32;
        let items = clustered(300, dim, 30);
        let pq = PqIndex::build(&items, Metric::Cosine, 8, 256, 20, true);
        for t in 0..20 {
            let (id, v) = &items[t * 13 % items.len()];
            let hits = pq.search(v, 1, 8);
            assert_eq!(hits[0].id, *id);
        }
    }

    #[test]
    fn pq_save_load_roundtrip() {
        let dim = 48;
        let items = clustered(500, dim, 16);
        let pq = PqIndex::build(&items, Metric::L2, 12, 128, 12, true);
        let mut path = std::env::temp_dir();
        path.push("vecdb_pq_test.vecdb");
        pq.save(&path).unwrap();
        let loaded = PqIndex::load(&path).unwrap();
        assert_eq!(loaded.len(), pq.len());
        assert_eq!(loaded.dim(), pq.dim());
        for t in 0..15 {
            let q = &items[t * 29 % items.len()].1;
            let a: Vec<u64> = pq.search(q, 10, 8).iter().map(|h| h.id).collect();
            let b: Vec<u64> = loaded.search(q, 10, 8).iter().map(|h| h.id).collect();
            assert_eq!(a, b);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn pq_filtered_search_restricts_ids() {
        let dim = 32;
        let items = clustered(1000, dim, 10);
        let pq = PqIndex::build(&items, Metric::L2, 8, 64, 12, true);
        let got = pq.search_filter(&items[0].1, 10, 8, |id| id % 3 == 0);
        assert!(!got.is_empty());
        assert!(got.iter().all(|h| h.id % 3 == 0));
    }

    #[test]
    fn pq_compact_no_raw_still_ranks() {
        let dim = 32;
        let items = clustered(400, dim, 10);
        let pq = PqIndex::build(&items, Metric::L2, 8, 64, 12, false);
        assert!(!pq.has_raw());
        // Without rerank, the nearest center's members should still dominate.
        let q = &items[0].1;
        let hits = pq.search(q, 5, 1);
        assert_eq!(hits.len(), 5);
    }
}

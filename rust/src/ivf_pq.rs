//! IVF + Product Quantization (the classic Faiss `IVFPQ`).
//!
//! A coarse k-means quantizer assigns each vector to one of `nlist` cells; the
//! **residual** `x - centroid` is then encoded with a shared PQ codebook
//! (trained on residuals across all cells). Residuals are small and centered,
//! so a given PQ budget captures them far more accurately than quantizing the
//! raw vectors — this is why IVFPQ is the standard billion-scale index.
//!
//! Search probes the `nprobe` nearest cells. For each cell we form the residual
//! query `q - centroid` and evaluate the PQ codes with an ADC lookup table
//! (per-cell for L2, since the residual query differs; a single table plus a
//! per-cell `<q, centroid>` term for dot/cosine). A widened candidate set is
//! reranked with exact f32 distances when the originals are kept.

use crate::distance::{dot_f32, l2sq_f32};
use crate::index::{push_bounded, Hit, Metric, Ranked};
use crate::pq::{adc_sum, build_lut, encode_vector, train_codebooks};
use std::collections::BinaryHeap;
use std::io;
use std::path::Path;

/// An IVF+PQ index.
pub struct IvfPqIndex {
    dim: usize,
    metric: Metric,
    nlist: usize,
    m: usize,
    dsub: usize,
    ksub: usize,
    count: usize,
    centroids: Vec<f32>,    // nlist * dim (coarse quantizer, processed space)
    pq_codebooks: Vec<f32>, // m * ksub * dsub (trained on residuals)
    offsets: Vec<usize>,    // nlist + 1, CSR row offsets
    ids: Vec<u64>,          // count, grouped by cell
    codes: Vec<u8>,         // count * m, grouped by cell
    raw: Option<Vec<f32>>,  // count * dim (processed) for rerank, grouped by cell
}

impl IvfPqIndex {
    /// Build an IVF+PQ index. `nlist` coarse cells (0 => ~sqrt(n)); PQ with `m`
    /// subspaces of `ksub` centroids each (`m` must divide `dim`, `ksub <= 256`).
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        vectors: &[(u64, Vec<f32>)],
        metric: Metric,
        nlist: usize,
        m: usize,
        ksub: usize,
        kmeans_iters: usize,
        keep_raw: bool,
    ) -> IvfPqIndex {
        assert!(!vectors.is_empty(), "IvfPqIndex::build: empty input");
        let dim = vectors[0].1.len();
        assert!(m > 0 && dim.is_multiple_of(m), "m must divide dim");
        assert!(ksub > 0 && ksub <= 256, "ksub must be in 1..=256");
        let dsub = dim / m;
        let n = vectors.len();
        let nlist = if nlist == 0 {
            ((n as f64).sqrt() as usize).clamp(1, n)
        } else {
            nlist.min(n)
        };
        let ksub = ksub.min(n);
        let cosine = metric == Metric::Cosine;

        // Processed (normalized for cosine) contiguous matrix.
        let mut proc = vec![0f32; n * dim];
        for (i, (_, v)) in vectors.iter().enumerate() {
            assert_eq!(v.len(), dim, "inconsistent dimension");
            if cosine {
                let p = normalize(v);
                proc[i * dim..(i + 1) * dim].copy_from_slice(&p);
            } else {
                proc[i * dim..(i + 1) * dim].copy_from_slice(v);
            }
        }

        // Coarse quantizer + residuals.
        let (centroids, assign) = crate::ivf::kmeans(&proc, n, dim, nlist, kmeans_iters, cosine);
        let mut residuals = vec![0f32; n * dim];
        for i in 0..n {
            let c = assign[i];
            for j in 0..dim {
                residuals[i * dim + j] = proc[i * dim + j] - centroids[c * dim + j];
            }
        }
        let pq_codebooks = train_codebooks(&residuals, n, dim, m, ksub, kmeans_iters);

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

        let mut ids = vec![0u64; n];
        let mut codes = vec![0u8; n * m];
        let mut raw = if keep_raw {
            Some(vec![0f32; n * dim])
        } else {
            None
        };
        for i in 0..n {
            let cell = assign[i];
            let dst = cursor[cell];
            cursor[cell] = dst + 1;
            ids[dst] = vectors[i].0;
            encode_vector(
                &pq_codebooks,
                dim,
                m,
                ksub,
                &residuals[i * dim..(i + 1) * dim],
                &mut codes[dst * m..(dst + 1) * m],
            );
            if let Some(rb) = raw.as_mut() {
                rb[dst * dim..(dst + 1) * dim].copy_from_slice(&proc[i * dim..(i + 1) * dim]);
            }
        }

        IvfPqIndex {
            dim,
            metric,
            nlist,
            m,
            dsub,
            ksub,
            count: n,
            centroids,
            pq_codebooks,
            offsets,
            ids,
            codes,
            raw,
        }
    }

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

    /// Search the `nprobe` nearest cells with residual ADC, then rerank.
    pub fn search(&self, query: &[f32], k: usize, nprobe: usize, oversample: usize) -> Vec<Hit> {
        self.search_filter(query, k, nprobe, oversample, |_| true)
    }

    /// Filtered search: only ids satisfying `filter` enter the candidate set
    /// (applied within each probed cell).
    pub fn search_filter<F: Fn(u64) -> bool>(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
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
        let l2 = self.metric == Metric::L2;
        let nprobe = nprobe.clamp(1, self.nlist);

        // Nearest cells.
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

        // For dot/cosine the ADC table is over the (cell-independent) full query;
        // only the `<q, centroid>` offset changes per cell. For L2 the table is
        // over the residual query, so it is rebuilt per cell.
        let dot_lut = if l2 {
            Vec::new()
        } else {
            build_lut(
                &self.pq_codebooks,
                self.dim,
                self.m,
                self.ksub,
                &processed,
                false,
            )
        };

        let cand_n = if self.raw.is_some() {
            (k * oversample.max(1)).min(self.count)
        } else {
            k.min(self.count)
        };
        let mut heap: BinaryHeap<Ranked> = BinaryHeap::with_capacity(cand_n + 1);
        let mut qr = vec![0f32; self.dim];
        for &c in &cells {
            let cen = &self.centroids[c * self.dim..(c + 1) * self.dim];
            let (lut, qc) = if l2 {
                for j in 0..self.dim {
                    qr[j] = processed[j] - cen[j];
                }
                (
                    build_lut(&self.pq_codebooks, self.dim, self.m, self.ksub, &qr, true),
                    0.0,
                )
            } else {
                (dot_lut.clone(), dot_f32(&processed, cen))
            };
            for i in self.offsets[c]..self.offsets[c + 1] {
                if !filter(self.ids[i]) {
                    continue;
                }
                let acc = adc_sum(&lut, self.ksub, &self.codes[i * self.m..(i + 1) * self.m]);
                let key = if l2 { acc } else { -(qc + acc) };
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

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        v.iter().map(|x| x * inv).collect()
    } else {
        v.to_vec()
    }
}

// ---------------------------------------------------------------------------
// Persistence: an IVF+PQ `.vecdb` file (magic "VECDBIP1").
// Header (64B) + centroids f32 + pq_codebooks f32 + offsets u64 + ids u64
// + codes u8 + raw f32? (each section 16B-aligned).
// ---------------------------------------------------------------------------

const IP_MAGIC: &[u8; 8] = b"VECDBIP1";
const IP_VERSION: u32 = 1;
const IP_FLAG_HAS_RAW: u32 = 1;

#[inline]
fn align16(x: usize) -> usize {
    (x + 15) & !15
}

struct IpLayout {
    centroids: usize,
    codebooks: usize,
    offsets: usize,
    ids: usize,
    codes: usize,
    raw: usize,
    total: usize,
}

impl IvfPqIndex {
    fn layout(&self) -> IpLayout {
        let dim = self.dim;
        let count = self.count;
        let centroids = align16(64);
        let codebooks = align16(centroids + self.nlist * dim * 4);
        let offsets = align16(codebooks + self.m * self.ksub * self.dsub * 4);
        let ids = align16(offsets + (self.nlist + 1) * 8);
        let codes = align16(ids + count * 8);
        let raw = align16(codes + count * self.m);
        let total = if self.raw.is_some() {
            align16(raw + count * dim * 4)
        } else {
            raw
        };
        IpLayout {
            centroids,
            codebooks,
            offsets,
            ids,
            codes,
            raw,
            total,
        }
    }

    /// Serialize the index to an IVF+PQ `.vecdb` file.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        std::fs::write(path, self.to_bytes())
    }

    /// Serialize the index to the in-memory IVF+PQ `.vecdb` byte image that
    /// [`save`](Self::save) writes (byte-identical), for a filesystem-free
    /// "bytes in / bytes out" round trip. Pair with [`from_bytes`](Self::from_bytes).
    pub fn to_bytes(&self) -> Vec<u8> {
        let l = self.layout();
        let has_raw = self.raw.is_some();
        let mut b = vec![0u8; l.total];
        b[0..8].copy_from_slice(IP_MAGIC);
        b[8..12].copy_from_slice(&IP_VERSION.to_le_bytes());
        b[12..16].copy_from_slice(&(self.metric as u32).to_le_bytes());
        b[16..20].copy_from_slice(&(self.dim as u32).to_le_bytes());
        b[20..24].copy_from_slice(&(self.count as u32).to_le_bytes());
        b[24..28].copy_from_slice(&(if has_raw { IP_FLAG_HAS_RAW } else { 0 }).to_le_bytes());
        b[28..32].copy_from_slice(&(self.nlist as u32).to_le_bytes());
        b[32..36].copy_from_slice(&(self.m as u32).to_le_bytes());
        b[36..40].copy_from_slice(&(self.ksub as u32).to_le_bytes());

        write_f32s(&mut b, l.centroids, &self.centroids);
        write_f32s(&mut b, l.codebooks, &self.pq_codebooks);
        for (i, &o) in self.offsets.iter().enumerate() {
            b[l.offsets + i * 8..l.offsets + i * 8 + 8].copy_from_slice(&(o as u64).to_le_bytes());
        }
        for (i, &id) in self.ids.iter().enumerate() {
            b[l.ids + i * 8..l.ids + i * 8 + 8].copy_from_slice(&id.to_le_bytes());
        }
        b[l.codes..l.codes + self.codes.len()].copy_from_slice(&self.codes);
        if let Some(rb) = self.raw.as_ref() {
            write_f32s(&mut b, l.raw, rb);
        }
        b
    }

    /// Load an IVF+PQ `.vecdb` file.
    pub fn load(path: impl AsRef<Path>) -> io::Result<IvfPqIndex> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    /// Parse an IVF+PQ index from a `.vecdb` byte image (the "bytes in"
    /// counterpart to [`to_bytes`](Self::to_bytes)). Performs the same validation
    /// as [`load`](Self::load) and returns the same [`io::Error`]s.
    pub fn from_bytes(b: &[u8]) -> io::Result<IvfPqIndex> {
        let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, format!("ivfpq: {m}"));
        if b.len() < 64 || &b[0..8] != IP_MAGIC {
            return Err(bad("bad magic"));
        }
        let u32_at = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        if u32_at(8) != IP_VERSION {
            return Err(bad("unsupported version"));
        }
        let metric = Metric::from_u32(u32_at(12)).ok_or_else(|| bad("bad metric"))?;
        let dim = u32_at(16) as usize;
        let count = u32_at(20) as usize;
        let has_raw = u32_at(24) & IP_FLAG_HAS_RAW != 0;
        let nlist = u32_at(28) as usize;
        let m = u32_at(32) as usize;
        let ksub = u32_at(36) as usize;
        if dim == 0 || m == 0 || !dim.is_multiple_of(m) {
            return Err(bad("bad dim/m"));
        }
        let dsub = dim / m;
        let mut idx = IvfPqIndex {
            dim,
            metric,
            nlist,
            m,
            dsub,
            ksub,
            count,
            centroids: Vec::new(),
            pq_codebooks: Vec::new(),
            offsets: Vec::new(),
            ids: Vec::new(),
            codes: Vec::new(),
            raw: if has_raw { Some(Vec::new()) } else { None },
        };
        let l = idx.layout();
        if b.len() < l.total {
            return Err(bad("file truncated"));
        }
        idx.centroids = read_f32s(b, l.centroids, nlist * dim);
        idx.pq_codebooks = read_f32s(b, l.codebooks, m * ksub * dsub);
        idx.offsets = (0..=nlist)
            .map(|i| {
                let o = l.offsets + i * 8;
                let mut a = [0u8; 8];
                a.copy_from_slice(&b[o..o + 8]);
                u64::from_le_bytes(a) as usize
            })
            .collect();
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
            idx.raw = Some(read_f32s(b, l.raw, count * dim));
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

#[cfg(feature = "parallel")]
impl IvfPqIndex {
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

    fn clustered(n: usize, dim: usize, ncenters: usize) -> Vec<(u64, Vec<f32>)> {
        let mut s: u64 = 0x51F7_0DEF_2244;
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
                (i as u64, c.iter().map(|&x| x + next() * 0.1).collect())
            })
            .collect()
    }

    #[test]
    fn ivfpq_recall_l2() {
        let dim = 64;
        let items = clustered(3000, dim, 25);
        let idx = IvfPqIndex::build(&items, Metric::L2, 40, 16, 256, 15, true);
        assert_eq!(idx.code_bytes(), 3000 * 16);
        let mut flat = crate::FlatIndex::new(dim, Metric::L2, true);
        for (id, v) in &items {
            flat.add(*id, v);
        }
        let mut hit = 0;
        let mut total = 0;
        for t in 0..30 {
            let q = &items[t * 17 % items.len()].1;
            let truth: std::collections::HashSet<u64> =
                flat.search_exact(q, 10).iter().map(|h| h.id).collect();
            hit += idx
                .search(q, 10, 16, 16)
                .iter()
                .filter(|h| truth.contains(&h.id))
                .count();
            total += truth.len();
        }
        let recall = hit as f64 / total as f64;
        assert!(recall >= 0.90, "IVFPQ L2 recall too low: {recall}");
    }

    #[test]
    fn ivfpq_cosine_finds_self() {
        let dim = 48;
        let items = clustered(1000, dim, 20);
        let idx = IvfPqIndex::build(&items, Metric::Cosine, 30, 12, 256, 15, true);
        for t in 0..20 {
            let (id, v) = &items[t * 13 % items.len()];
            let hits = idx.search(v, 1, 8, 16);
            assert_eq!(hits[0].id, *id);
        }
    }

    #[test]
    fn ivfpq_filtered_search_restricts_ids() {
        let dim = 48;
        let items = clustered(1500, dim, 20);
        let idx = IvfPqIndex::build(&items, Metric::L2, 30, 12, 128, 12, true);
        let got = idx.search_filter(&items[0].1, 10, 8, 16, |id| id % 2 == 0);
        assert!(!got.is_empty());
        assert!(got.iter().all(|h| h.id % 2 == 0));
    }

    #[test]
    fn ivfpq_bytes_roundtrip() {
        let dim = 48;
        let items = clustered(800, dim, 16);
        let idx = IvfPqIndex::build(&items, Metric::L2, 24, 12, 128, 12, true);
        let b = idx.to_bytes();
        let loaded = IvfPqIndex::from_bytes(&b).unwrap();
        assert_eq!(loaded.len(), idx.len());
        assert_eq!(loaded.nlist(), idx.nlist());
        for t in 0..15 {
            let q = &items[t * 29 % items.len()].1;
            let a: Vec<u64> = idx.search(q, 10, 8, 8).iter().map(|h| h.id).collect();
            let c: Vec<u64> = loaded.search(q, 10, 8, 8).iter().map(|h| h.id).collect();
            assert_eq!(a, c);
        }
        assert_eq!(loaded.to_bytes(), b); // byte-stable round trip
    }

    #[test]
    fn ivfpq_save_load_roundtrip() {
        let dim = 48;
        let items = clustered(800, dim, 16);
        let idx = IvfPqIndex::build(&items, Metric::L2, 24, 12, 128, 12, true);
        let mut path = std::env::temp_dir();
        path.push("vecdb_ivfpq_test.vecdb");
        idx.save(&path).unwrap();
        let loaded = IvfPqIndex::load(&path).unwrap();
        assert_eq!(loaded.len(), idx.len());
        assert_eq!(loaded.nlist(), idx.nlist());
        for t in 0..15 {
            let q = &items[t * 29 % items.len()].1;
            let a: Vec<u64> = idx.search(q, 10, 8, 8).iter().map(|h| h.id).collect();
            let b: Vec<u64> = loaded.search(q, 10, 8, 8).iter().map(|h| h.id).collect();
            assert_eq!(a, b);
        }
        std::fs::remove_file(&path).ok();
    }
}

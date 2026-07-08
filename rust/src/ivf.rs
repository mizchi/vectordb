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
use std::io;
use std::path::Path;

/// Below this many scanned candidates, IVF's parallel search stays serial.
#[cfg(feature = "parallel")]
const IVF_PAR_THRESHOLD: usize = 8192;

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
        let (processed, q, cells) = self.probe(query, nprobe);
        let cand_n = self.cand_n(k, oversample);
        let mut heap: BinaryHeap<Ranked> = BinaryHeap::with_capacity(cand_n + 1);
        for &cell in &cells {
            for i in self.offsets[cell]..self.offsets[cell + 1] {
                push_bounded(&mut heap, Ranked { key: self.approx_key_at(i, &q), idx: i }, cand_n);
            }
        }
        self.finish_from(heap, &processed, k, oversample)
    }

    /// Parallel variant of [`search`](Self::search): the int8 scan over the
    /// probed cells is split across the rayon pool. Falls back to serial when
    /// the number of scanned candidates is small.
    #[cfg(feature = "parallel")]
    pub fn search_parallel(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
        oversample: usize,
    ) -> Vec<Hit> {
        use rayon::prelude::*;
        if k == 0 || self.is_empty() {
            return Vec::new();
        }
        let (processed, q, cells) = self.probe(query, nprobe);
        let cand_n = self.cand_n(k, oversample);
        let cand: Vec<usize> = cells
            .iter()
            .flat_map(|&c| self.offsets[c]..self.offsets[c + 1])
            .collect();

        let heap = if cand.len() < IVF_PAR_THRESHOLD {
            let mut h: BinaryHeap<Ranked> = BinaryHeap::with_capacity(cand_n + 1);
            for &i in &cand {
                push_bounded(&mut h, Ranked { key: self.approx_key_at(i, &q), idx: i }, cand_n);
            }
            h
        } else {
            cand.par_iter()
                .fold(
                    || BinaryHeap::with_capacity(cand_n + 1),
                    |mut h, &i| {
                        push_bounded(&mut h, Ranked { key: self.approx_key_at(i, &q), idx: i }, cand_n);
                        h
                    },
                )
                .reduce(
                    || BinaryHeap::with_capacity(cand_n + 1),
                    |mut a, b| {
                        for r in b.into_iter() {
                            push_bounded(&mut a, r, cand_n);
                        }
                        a
                    },
                )
        };
        self.finish_from(heap, &processed, k, oversample)
    }

    /// Run many queries concurrently (one query per rayon task).
    #[cfg(feature = "parallel")]
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

    /// Shared query prep: process, quantize, pick the nprobe nearest cells.
    fn probe(&self, query: &[f32], nprobe: usize) -> (Vec<f32>, crate::quantize::Quantized, Vec<usize>) {
        assert_eq!(query.len(), self.dim, "dimension mismatch");
        let processed = if self.metric == Metric::Cosine {
            normalize(query)
        } else {
            query.to_vec()
        };
        let q = quantize(&processed);
        let nprobe = nprobe.clamp(1, self.nlist);
        let cells = self.nearest_cells(&processed, nprobe);
        (processed, q, cells)
    }

    #[inline]
    fn cand_n(&self, k: usize, oversample: usize) -> usize {
        if self.raw.is_some() {
            k * oversample.max(1)
        } else {
            k
        }
    }

    #[inline]
    fn approx_key_at(&self, i: usize, q: &crate::quantize::Quantized) -> f32 {
        let dp = dot_i8(self.codes_at(i), &q.codes) as f32 * self.scales[i] * q.scale;
        match self.metric {
            Metric::L2 => self.sqnorms[i] + q.sqnorm - 2.0 * dp,
            Metric::Dot | Metric::Cosine => -dp,
        }
    }

    /// Exact f32 rerank (when originals kept) then produce sorted hits.
    fn finish_from(
        &self,
        heap: BinaryHeap<Ranked>,
        processed: &[f32],
        k: usize,
        oversample: usize,
    ) -> Vec<Hit> {
        if self.raw.is_some() && oversample > 1 {
            let mut rr: BinaryHeap<Ranked> = BinaryHeap::with_capacity(k + 1);
            for Ranked { idx, .. } in heap.into_iter() {
                let key = self.exact_key(idx, processed);
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

// ---------------------------------------------------------------------------
// Persistence: an IVF-flavored `.vecdb` file (magic "VECDBIV1").
// Layout (little-endian): 64-byte header + 16-byte-aligned sections:
//   [header][ centroids f32 ][ offsets u64 ][ ids u64 ]
//           [ scales f32 ][ sqnorms f32 ][ codes i8 ][ raw f32? ]
// ---------------------------------------------------------------------------

const IVF_MAGIC: &[u8; 8] = b"VECDBIV1";
const IVF_VERSION: u32 = 1;
const IVF_HEADER_LEN: usize = 64;
const IVF_FLAG_HAS_RAW: u32 = 1;

#[inline]
fn align16(x: usize) -> usize {
    (x + 15) & !15
}

struct IvfLayout {
    centroids: usize,
    offsets: usize,
    ids: usize,
    scales: usize,
    sqnorms: usize,
    codes: usize,
    raw: usize,
    total: usize,
}

fn ivf_layout(dim: usize, count: usize, nlist: usize, has_raw: bool) -> IvfLayout {
    let centroids = align16(IVF_HEADER_LEN);
    let offsets = align16(centroids + nlist * dim * 4);
    let ids = align16(offsets + (nlist + 1) * 8);
    let scales = align16(ids + count * 8);
    let sqnorms = align16(scales + count * 4);
    let codes = align16(sqnorms + count * 4);
    let raw = align16(codes + count * dim);
    let total = if has_raw {
        align16(raw + count * dim * 4)
    } else {
        raw
    };
    IvfLayout {
        centroids,
        offsets,
        ids,
        scales,
        sqnorms,
        codes,
        raw,
        total,
    }
}

fn put_f32s(buf: &mut [u8], off: usize, data: &[f32]) {
    for (i, &x) in data.iter().enumerate() {
        buf[off + i * 4..off + i * 4 + 4].copy_from_slice(&x.to_le_bytes());
    }
}
fn put_u64s(buf: &mut [u8], off: usize, data: impl Iterator<Item = u64>) {
    for (i, x) in data.enumerate() {
        buf[off + i * 8..off + i * 8 + 8].copy_from_slice(&x.to_le_bytes());
    }
}
fn get_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}
fn get_u64(b: &[u8], off: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[off..off + 8]);
    u64::from_le_bytes(a)
}
fn get_f32s(b: &[u8], off: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| f32::from_le_bytes([b[off + i * 4], b[off + i * 4 + 1], b[off + i * 4 + 2], b[off + i * 4 + 3]]))
        .collect()
}

impl IvfIndex {
    /// Serialize to an IVF `.vecdb` file.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let dim = self.dim;
        let count = self.len();
        let has_raw = self.raw.is_some();
        let l = ivf_layout(dim, count, self.nlist, has_raw);
        let mut buf = vec![0u8; l.total];

        buf[0..8].copy_from_slice(IVF_MAGIC);
        buf[8..12].copy_from_slice(&IVF_VERSION.to_le_bytes());
        buf[12..16].copy_from_slice(&(self.metric as u32).to_le_bytes());
        buf[16..20].copy_from_slice(&(dim as u32).to_le_bytes());
        buf[20..24].copy_from_slice(&(count as u32).to_le_bytes());
        buf[24..28].copy_from_slice(&(if has_raw { IVF_FLAG_HAS_RAW } else { 0 }).to_le_bytes());
        buf[28..32].copy_from_slice(&(self.nlist as u32).to_le_bytes());

        put_f32s(&mut buf, l.centroids, &self.centroids);
        put_u64s(&mut buf, l.offsets, self.offsets.iter().map(|&o| o as u64));
        put_u64s(&mut buf, l.ids, self.ids.iter().copied());
        put_f32s(&mut buf, l.scales, &self.scales);
        put_f32s(&mut buf, l.sqnorms, &self.sqnorms);
        for (i, &c) in self.codes.iter().enumerate() {
            buf[l.codes + i] = c as u8;
        }
        if let Some(raw) = &self.raw {
            put_f32s(&mut buf, l.raw, raw);
        }

        std::fs::write(path, &buf)
    }

    /// Load an IVF `.vecdb` file (parsed via mmap, then copied into owned Vecs).
    pub fn load(path: impl AsRef<Path>) -> io::Result<IvfIndex> {
        let file = std::fs::File::open(path)?;
        // SAFETY: read-only mmap of a regular file held open for the call.
        let m = unsafe { memmap2::Mmap::map(&file)? };
        let b: &[u8] = &m;
        let bad = |msg: &str| io::Error::new(io::ErrorKind::InvalidData, format!("ivf: {msg}"));
        if b.len() < IVF_HEADER_LEN || &b[0..8] != IVF_MAGIC {
            return Err(bad("bad magic"));
        }
        if get_u32(b, 8) != IVF_VERSION {
            return Err(bad("unsupported version"));
        }
        let metric = Metric::from_u32(get_u32(b, 12)).ok_or_else(|| bad("bad metric"))?;
        let dim = get_u32(b, 16) as usize;
        let count = get_u32(b, 20) as usize;
        let has_raw = get_u32(b, 24) & IVF_FLAG_HAS_RAW != 0;
        let nlist = get_u32(b, 28) as usize;
        if dim == 0 || nlist == 0 {
            return Err(bad("zero dim/nlist"));
        }
        let l = ivf_layout(dim, count, nlist, has_raw);
        if b.len() < l.total {
            return Err(bad("file truncated"));
        }

        let centroids = get_f32s(b, l.centroids, nlist * dim);
        let offsets: Vec<usize> = (0..=nlist)
            .map(|i| get_u64(b, l.offsets + i * 8) as usize)
            .collect();
        let ids: Vec<u64> = (0..count).map(|i| get_u64(b, l.ids + i * 8)).collect();
        let scales = get_f32s(b, l.scales, count);
        let sqnorms = get_f32s(b, l.sqnorms, count);
        let codes: Vec<i8> = (0..count * dim).map(|i| b[l.codes + i] as i8).collect();
        let raw = if has_raw {
            Some(get_f32s(b, l.raw, count * dim))
        } else {
            None
        };

        Ok(IvfIndex {
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
        })
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
    fn ivf_save_load_roundtrip() {
        let items = make_items();
        let idx = IvfIndex::build(2, Metric::L2, 4, &items, true, 15);
        let mut path = std::env::temp_dir();
        path.push("vecdb_ivf_test.vecdb");
        idx.save(&path).unwrap();

        let loaded = IvfIndex::load(&path).unwrap();
        assert_eq!(loaded.len(), idx.len());
        assert_eq!(loaded.nlist(), idx.nlist());
        assert!(loaded.has_raw());
        let q = [10.0, 10.0];
        let a = idx.search(&q, 3, 4, 8);
        let b = loaded.search(&q, 3, 4, 8);
        assert_eq!(
            a.iter().map(|h| h.id).collect::<Vec<_>>(),
            b.iter().map(|h| h.id).collect::<Vec<_>>()
        );
        std::fs::remove_file(&path).ok();
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn ivf_parallel_matches_serial() {
        // Enough vectors that the probed cells exceed IVF_PAR_THRESHOLD.
        let dim = 8usize;
        let n = 40_000usize;
        let mut s: u64 = 0xABCD_1234;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32
        };
        let ncenters = 20usize;
        let centers: Vec<Vec<f32>> = (0..ncenters)
            .map(|_| (0..dim).map(|_| next() * 2.0 - 1.0).collect())
            .collect();
        let items: Vec<(u64, Vec<f32>)> = (0..n)
            .map(|i| {
                let c = &centers[i % ncenters];
                (i as u64, c.iter().map(|x| x + (next() - 0.5) * 0.1).collect())
            })
            .collect();
        let idx = IvfIndex::build(dim, Metric::L2, 64, &items, true, 8);
        let q: Vec<f32> = centers[3].iter().map(|x| x + 0.01).collect();
        let a: Vec<u64> = idx.search(&q, 10, 32, 8).iter().map(|h| h.id).collect();
        let b: Vec<u64> = idx.search_parallel(&q, 10, 32, 8).iter().map(|h| h.id).collect();
        assert_eq!(a, b);
    }

    #[test]
    fn ivf_cosine_runs() {
        let items = make_items();
        let idx = IvfIndex::build(2, Metric::Cosine, 4, &items, true, 10);
        let hits = idx.search(&[10.0, 10.0], 5, 4, 4);
        assert_eq!(hits.len(), 5);
    }
}

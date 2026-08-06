//! `.vecdb` single-file format: save to disk and zero-copy mmap load.
//!
//! Layout (little-endian), matching `DESIGN.md`:
//!
//! ```text
//! [ 64-byte header ][ ids u64 ][ scales f32 ][ sqnorms f32 ][ codes i8 ]
//!   [ raw f32? ][ deleted u8? ][ payload_offsets u64? ][ payload_blob u8? ]
//! ```
//!
//! Each section starts on a 16-byte boundary. The header records `dim`,
//! `count`, `metric` and flag bits for which optional sections are present.
//! The last three sections are written only when the index actually carries
//! tombstones / payloads, so an index with neither produces byte-identical
//! output to the original v1 format (and stays interoperable with MoonBit,
//! which reads the leading sections and ignores any trailing bytes).

use crate::index::{FlatIndex, Metric, View};
use memmap2::Mmap;
use std::fs::File;
use std::io;
use std::mem::{align_of, size_of};
use std::path::Path;

const MAGIC: &[u8; 8] = b"VECDB1\0\0";
const VERSION: u32 = 1;
const HEADER_LEN: usize = 64;
const FLAG_HAS_RAW: u32 = 1;
const FLAG_HAS_DELETED: u32 = 2;
const FLAG_HAS_PAYLOADS: u32 = 4;

#[inline]
fn align16(x: usize) -> usize {
    (x + 15) & !15
}

/// Byte offsets of each section within the file. Optional sections are `None`
/// when absent.
struct Layout {
    ids: usize,
    scales: usize,
    sqnorms: usize,
    codes: usize,
    raw: Option<usize>,
    deleted: Option<usize>,
    payload_offsets: Option<usize>,
    payload_blob: Option<usize>,
    total: usize,
}

fn layout(
    dim: usize,
    count: usize,
    has_raw: bool,
    has_deleted: bool,
    payload_blob_len: Option<usize>,
) -> Layout {
    let ids = align16(HEADER_LEN);
    let scales = align16(ids + count * size_of::<u64>());
    let sqnorms = align16(scales + count * size_of::<f32>());
    let codes = align16(sqnorms + count * size_of::<f32>());
    let mut cur = align16(codes + count * dim * size_of::<i8>());
    let raw = if has_raw {
        let o = cur;
        cur = align16(o + count * dim * size_of::<f32>());
        Some(o)
    } else {
        None
    };
    let deleted = if has_deleted {
        let o = cur;
        cur = align16(o + count * size_of::<u8>());
        Some(o)
    } else {
        None
    };
    let (payload_offsets, payload_blob) = if let Some(blen) = payload_blob_len {
        let po = cur;
        cur = align16(po + (count + 1) * size_of::<u64>());
        let pb = cur;
        cur = align16(pb + blen);
        (Some(po), Some(pb))
    } else {
        (None, None)
    };
    Layout {
        ids,
        scales,
        sqnorms,
        codes,
        raw,
        deleted,
        payload_offsets,
        payload_blob,
        total: cur,
    }
}

/// Serialize an index to a `.vecdb` file.
///
/// Soft-delete tombstones and metadata payloads are persisted when present, so
/// a saved index reloads with its deletions and payloads intact. Call
/// [`FlatIndex::compact`] first if you would rather drop tombstoned rows
/// entirely instead of carrying them across the round-trip.
pub fn save(index: &FlatIndex, path: impl AsRef<Path>) -> io::Result<()> {
    std::fs::write(path, to_bytes(index))
}

/// Serialize an index to the in-memory `.vecdb` byte image that [`save`] writes.
///
/// The bytes are identical to what [`save`] produces on disk; use this for a
/// "bytes in / bytes out" round trip that never touches the filesystem (pair
/// with [`from_bytes`]). Soft-delete tombstones and metadata payloads are
/// included when present.
pub fn to_bytes(index: &FlatIndex) -> Vec<u8> {
    let dim = index.dim();
    let count = index.len();
    let has_raw = index.has_raw();
    let has_deleted = index.deleted_count > 0;
    // Build the payload blob + CSR offsets only if any row carries one.
    let has_payloads = index.payloads.iter().any(|p| !p.is_empty());
    let (payload_offsets, payload_blob) = if has_payloads {
        let mut offs: Vec<u64> = Vec::with_capacity(count + 1);
        let mut blob: Vec<u8> = Vec::new();
        offs.push(0);
        for p in &index.payloads {
            blob.extend_from_slice(p);
            offs.push(blob.len() as u64);
        }
        (Some(offs), Some(blob))
    } else {
        (None, None)
    };
    let l = layout(
        dim,
        count,
        has_raw,
        has_deleted,
        payload_blob.as_ref().map(|b| b.len()),
    );

    let mut buf = vec![0u8; l.total];
    // Header.
    buf[0..8].copy_from_slice(MAGIC);
    buf[8..12].copy_from_slice(&VERSION.to_le_bytes());
    buf[12..16].copy_from_slice(&(index.metric() as u32).to_le_bytes());
    buf[16..20].copy_from_slice(&(dim as u32).to_le_bytes());
    buf[20..24].copy_from_slice(&(count as u32).to_le_bytes());
    let mut flags = 0;
    if has_raw {
        flags |= FLAG_HAS_RAW;
    }
    if has_deleted {
        flags |= FLAG_HAS_DELETED;
    }
    if has_payloads {
        flags |= FLAG_HAS_PAYLOADS;
    }
    buf[24..28].copy_from_slice(&flags.to_le_bytes());

    // Sections. Copy each field's bytes into place.
    write_slice(&mut buf, l.ids, &index.ids);
    write_slice(&mut buf, l.scales, &index.scales);
    write_slice(&mut buf, l.sqnorms, &index.sqnorms);
    // codes are i8; reinterpret as u8 for the byte copy.
    let codes_u8: &[u8] =
        unsafe { std::slice::from_raw_parts(index.codes.as_ptr() as *const u8, index.codes.len()) };
    buf[l.codes..l.codes + codes_u8.len()].copy_from_slice(codes_u8);
    if let Some(off) = l.raw {
        write_slice(&mut buf, off, index.raw.as_ref().unwrap());
    }
    if let Some(off) = l.deleted {
        // Tombstones as one byte per row (0 = live, 1 = deleted).
        for (i, &d) in index.deleted.iter().enumerate() {
            buf[off + i] = d as u8;
        }
    }
    if let (Some(off), Some(offs)) = (l.payload_offsets, payload_offsets.as_ref()) {
        write_slice(&mut buf, off, offs);
    }
    if let (Some(off), Some(blob)) = (l.payload_blob, payload_blob.as_ref()) {
        buf[off..off + blob.len()].copy_from_slice(blob);
    }

    buf
}

fn write_slice<T: Copy>(buf: &mut [u8], off: usize, data: &[T]) {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    buf[off..off + bytes.len()].copy_from_slice(bytes);
}

/// A zero-copy, read-only index backed by an mmap of a `.vecdb` file.
///
/// The large sections (codes, raw) are read straight out of the mapping. The
/// small tombstone vector is copied into an owned `Vec<bool>` at open time
/// (both to avoid reinterpreting arbitrary bytes as `bool` and because it is
/// negligible in size); payload offsets are likewise parsed into an owned
/// vector while the payload blob itself stays in the mapping.
pub struct MmapIndex {
    _mmap: Mmap,
    dim: usize,
    metric: Metric,
    count: usize,
    ids: *const u64,
    scales: *const f32,
    sqnorms: *const f32,
    codes: *const i8,
    raw: Option<*const f32>,
    deleted: Option<Vec<bool>>,
    payload_offsets: Option<Vec<usize>>,
    payload_blob: Option<*const u8>,
}

// SAFETY: the pointers only ever alias the owned `_mmap` region, which lives as
// long as `self`; nothing mutates through them.
unsafe impl Send for MmapIndex {}
unsafe impl Sync for MmapIndex {}

impl MmapIndex {
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
    /// Number of live (non-tombstoned) vectors.
    pub fn live_len(&self) -> usize {
        match &self.deleted {
            Some(d) => self.count - d.iter().filter(|&&x| x).count(),
            None => self.count,
        }
    }

    /// Borrow the mmap'd sections as a [`View`] for querying. Persisted
    /// tombstones are honored, so deleted rows are excluded from results.
    pub fn view(&self) -> View<'_> {
        // SAFETY: offsets/lengths were validated at open() time and the mmap
        // outlives the returned view (tied to &self).
        unsafe {
            View {
                dim: self.dim,
                metric: self.metric,
                ids: std::slice::from_raw_parts(self.ids, self.count),
                codes: std::slice::from_raw_parts(self.codes, self.count * self.dim),
                scales: std::slice::from_raw_parts(self.scales, self.count),
                sqnorms: std::slice::from_raw_parts(self.sqnorms, self.count),
                raw: self
                    .raw
                    .map(|p| std::slice::from_raw_parts(p, self.count * self.dim)),
                deleted: self.deleted.as_deref(),
            }
        }
    }

    /// The metadata payload of the first live row with `id`, if any (empty
    /// payloads and unknown ids return `None`). Backed by the mmap'd blob.
    pub fn payload(&self, id: u64) -> Option<&[u8]> {
        let offs = self.payload_offsets.as_ref()?;
        let blob = self.payload_blob?;
        // SAFETY: `ids` covers `count` entries in the mapping; offsets were
        // validated monotonic within the blob at open() time.
        let ids = unsafe { std::slice::from_raw_parts(self.ids, self.count) };
        for i in 0..self.count {
            let live = self.deleted.as_ref().is_none_or(|d| !d[i]);
            if ids[i] == id && live {
                let (s, e) = (offs[i], offs[i + 1]);
                if e > s {
                    // SAFETY: [s, e) lies within the blob section (checked at open).
                    return Some(unsafe { std::slice::from_raw_parts(blob.add(s), e - s) });
                }
            }
        }
        None
    }

    /// Convenience wrapper for [`View::search`].
    pub fn search(&self, query: &[f32], k: usize, oversample: usize) -> Vec<crate::index::Hit> {
        self.view().search(query, k, oversample)
    }

    /// Parallel single-query search (see [`View::search_parallel`]).
    #[cfg(feature = "parallel")]
    pub fn search_parallel(
        &self,
        query: &[f32],
        k: usize,
        oversample: usize,
    ) -> Vec<crate::index::Hit> {
        self.view().search_parallel(query, k, oversample)
    }

    /// Concurrent batch search (see [`View::search_batch`]).
    #[cfg(feature = "parallel")]
    pub fn search_batch(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        oversample: usize,
    ) -> Vec<Vec<crate::index::Hit>> {
        self.view().search_batch(queries, k, oversample)
    }
}

/// Open a `.vecdb` file as a zero-copy mmap index.
pub fn open(path: impl AsRef<Path>) -> io::Result<MmapIndex> {
    let file = File::open(path)?;
    // SAFETY: standard mmap of a regular file we hold open read-only.
    let mmap = unsafe { Mmap::map(&file)? };
    if mmap.len() < HEADER_LEN {
        return Err(bad("file smaller than header"));
    }
    if &mmap[0..8] != MAGIC {
        return Err(bad("bad magic"));
    }
    let version = read_u32(&mmap, 8);
    if version != VERSION {
        return Err(bad("unsupported version"));
    }
    let metric = Metric::from_u32(read_u32(&mmap, 12)).ok_or_else(|| bad("bad metric"))?;
    let dim = read_u32(&mmap, 16) as usize;
    let count = read_u32(&mmap, 20) as usize;
    let flags = read_u32(&mmap, 24);
    let has_raw = flags & FLAG_HAS_RAW != 0;
    let has_deleted = flags & FLAG_HAS_DELETED != 0;
    let has_payloads = flags & FLAG_HAS_PAYLOADS != 0;
    if dim == 0 {
        return Err(bad("zero dim"));
    }

    // First pass with a zero-length blob to locate the payload_offsets section,
    // read the true blob length from offs[count], then recompute the layout.
    let probe = layout(dim, count, has_raw, has_deleted, has_payloads.then_some(0));
    if mmap.len() < probe.total {
        return Err(bad("file truncated"));
    }
    let blob_len = if has_payloads {
        let po = probe.payload_offsets.unwrap();
        if mmap.len() < po + (count + 1) * size_of::<u64>() {
            return Err(bad("file truncated"));
        }
        Some(read_u64(&mmap, po + count * size_of::<u64>()) as usize)
    } else {
        None
    };
    let l = layout(dim, count, has_raw, has_deleted, blob_len);
    if mmap.len() < l.total {
        return Err(bad("file truncated"));
    }

    let base = mmap.as_ptr();
    // The base pointer of an mmap is page-aligned and every section is
    // 16-aligned, so typed reads below are properly aligned.
    let ids = typed_ptr::<u64>(base, l.ids)?;
    let scales = typed_ptr::<f32>(base, l.scales)?;
    let sqnorms = typed_ptr::<f32>(base, l.sqnorms)?;
    let codes = unsafe { base.add(l.codes) } as *const i8;
    let raw = match l.raw {
        Some(off) => Some(typed_ptr::<f32>(base, off)?),
        None => None,
    };
    let deleted = l.deleted.map(|off| {
        (0..count)
            .map(|i| mmap[off + i] != 0)
            .collect::<Vec<bool>>()
    });
    let payload_offsets = match l.payload_offsets {
        Some(off) => {
            let offs: Vec<usize> = (0..=count)
                .map(|i| read_u64(&mmap, off + i * size_of::<u64>()) as usize)
                .collect();
            // Validate monotonicity and bounds against the blob section.
            let blen = blob_len.unwrap_or(0);
            if offs[0] != 0 || offs[count] != blen || offs.windows(2).any(|w| w[1] < w[0]) {
                return Err(bad("bad payload offsets"));
            }
            Some(offs)
        }
        None => None,
    };
    let payload_blob = l.payload_blob.map(|off| unsafe { base.add(off) });

    Ok(MmapIndex {
        _mmap: mmap,
        dim,
        metric,
        count,
        ids,
        scales,
        sqnorms,
        codes,
        raw,
        deleted,
        payload_offsets,
        payload_blob,
    })
}

/// Load a `.vecdb` file into an owned, mutable [`FlatIndex`] (copies data).
/// Persisted tombstones and payloads are restored.
pub fn load(path: impl AsRef<Path>) -> io::Result<FlatIndex> {
    from_bytes(&std::fs::read(path)?)
}

/// Parse an owned, mutable [`FlatIndex`] from a `.vecdb` byte image (copies data
/// out of the slice). This is the "bytes in" counterpart to [`to_bytes`]: it
/// reconstructs exactly what [`load`] returns, restoring persisted tombstones
/// and payloads, and performs the same validation (magic, version, truncation,
/// payload-offset) as [`open`]/[`load`], returning the same [`io::Error`]s.
pub fn from_bytes(b: &[u8]) -> io::Result<FlatIndex> {
    if b.len() < HEADER_LEN {
        return Err(bad("file smaller than header"));
    }
    if &b[0..8] != MAGIC {
        return Err(bad("bad magic"));
    }
    let version = read_u32(b, 8);
    if version != VERSION {
        return Err(bad("unsupported version"));
    }
    let metric = Metric::from_u32(read_u32(b, 12)).ok_or_else(|| bad("bad metric"))?;
    let dim = read_u32(b, 16) as usize;
    let count = read_u32(b, 20) as usize;
    let flags = read_u32(b, 24);
    let has_raw = flags & FLAG_HAS_RAW != 0;
    let has_deleted = flags & FLAG_HAS_DELETED != 0;
    let has_payloads = flags & FLAG_HAS_PAYLOADS != 0;
    if dim == 0 {
        return Err(bad("zero dim"));
    }

    // Same two-pass layout resolution as `open`: probe to find the payload
    // offsets section, read the real blob length, then recompute the layout.
    let probe = layout(dim, count, has_raw, has_deleted, has_payloads.then_some(0));
    if b.len() < probe.total {
        return Err(bad("file truncated"));
    }
    let blob_len = if has_payloads {
        let po = probe.payload_offsets.unwrap();
        if b.len() < po + (count + 1) * size_of::<u64>() {
            return Err(bad("file truncated"));
        }
        Some(read_u64(b, po + count * size_of::<u64>()) as usize)
    } else {
        None
    };
    let l = layout(dim, count, has_raw, has_deleted, blob_len);
    if b.len() < l.total {
        return Err(bad("file truncated"));
    }

    let mut idx = FlatIndex::new(dim, metric, has_raw);
    idx.ids = (0..count)
        .map(|i| read_u64(b, l.ids + i * size_of::<u64>()))
        .collect();
    idx.scales = read_f32s(b, l.scales, count);
    idx.sqnorms = read_f32s(b, l.sqnorms, count);
    idx.codes = (0..count * dim).map(|i| b[l.codes + i] as i8).collect();
    if let Some(off) = l.raw {
        idx.raw = Some(read_f32s(b, off, count * dim));
    }
    let deleted: Vec<bool> = match l.deleted {
        Some(off) => (0..count).map(|i| b[off + i] != 0).collect(),
        None => vec![false; count],
    };
    idx.deleted_count = deleted.iter().filter(|&&x| x).count();
    idx.deleted = deleted;
    idx.payloads = match (l.payload_offsets, l.payload_blob) {
        (Some(po), Some(pb)) => {
            let offs: Vec<usize> = (0..=count)
                .map(|i| read_u64(b, po + i * size_of::<u64>()) as usize)
                .collect();
            // Validate monotonicity and bounds against the blob section.
            let blen = blob_len.unwrap_or(0);
            if offs[0] != 0 || offs[count] != blen || offs.windows(2).any(|w| w[1] < w[0]) {
                return Err(bad("bad payload offsets"));
            }
            (0..count)
                .map(|i| b[pb + offs[i]..pb + offs[i + 1]].to_vec())
                .collect()
        }
        _ => vec![Vec::new(); count],
    };
    Ok(idx)
}

fn typed_ptr<T>(base: *const u8, off: usize) -> io::Result<*const T> {
    let p = unsafe { base.add(off) };
    if !(p as usize).is_multiple_of(align_of::<T>()) {
        return Err(bad("misaligned section"));
    }
    Ok(p as *const T)
}

fn read_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

fn read_u64(b: &[u8], off: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[off..off + 8]);
    u64::from_le_bytes(a)
}

fn read_f32s(b: &[u8], off: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let o = off + i * 4;
            f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
        })
        .collect()
}

fn bad(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("vecdb: {msg}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("vecdb_test_{name}.vecdb"));
        p
    }

    #[test]
    fn roundtrip_with_raw() {
        let mut idx = FlatIndex::new(3, Metric::L2, true);
        idx.add(1, &[1.0, 2.0, 3.0]);
        idx.add(2, &[-1.0, 0.5, 4.0]);
        idx.add(3, &[0.0, 0.0, 1.0]);
        let path = tmp("raw");
        save(&idx, &path).unwrap();

        let m = open(&path).unwrap();
        assert_eq!(m.len(), 3);
        assert_eq!(m.dim(), 3);
        assert!(m.has_raw());
        let a = m.search(&[1.0, 2.0, 3.0], 1, 4);
        assert_eq!(a[0].id, 1);

        let loaded = load(&path).unwrap();
        assert_eq!(loaded.len(), 3);
        let b = loaded.search(&[1.0, 2.0, 3.0], 1, 4);
        assert_eq!(b[0].id, 1);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn roundtrip_persists_tombstones() {
        let mut idx = FlatIndex::new(4, Metric::Cosine, true);
        idx.add(10, &[1.0, 0.0, 0.0, 0.0]);
        idx.add(20, &[0.0, 1.0, 0.0, 0.0]);
        idx.add(30, &[0.9, 0.1, 0.0, 0.0]);
        idx.remove(10); // tombstone the nearest to the x axis
        let path = tmp("tombstone");
        save(&idx, &path).unwrap();

        // mmap view honors the tombstone.
        let m = open(&path).unwrap();
        assert_eq!(m.len(), 3);
        assert_eq!(m.live_len(), 2);
        let hits = m.search(&[1.0, 0.0, 0.0, 0.0], 3, 4);
        assert!(hits.iter().all(|h| h.id != 10));
        assert_eq!(hits[0].id, 30);

        // owned load restores the tombstone too.
        let loaded = load(&path).unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.live_len(), 2);
        assert!(loaded
            .search(&[1.0, 0.0, 0.0, 0.0], 3, 4)
            .iter()
            .all(|h| h.id != 10));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn roundtrip_persists_payloads() {
        let mut idx = FlatIndex::new(4, Metric::L2, true);
        idx.add_with_payload(1, &[1.0, 0.0, 0.0, 0.0], b"first");
        idx.add(2, &[0.0, 1.0, 0.0, 0.0]); // no payload
        idx.add_with_payload(3, &[0.0, 0.0, 1.0, 0.0], b"third-doc");
        let path = tmp("payloads");
        save(&idx, &path).unwrap();

        let m = open(&path).unwrap();
        assert_eq!(m.payload(1), Some(&b"first"[..]));
        assert_eq!(m.payload(2), None);
        assert_eq!(m.payload(3), Some(&b"third-doc"[..]));
        assert_eq!(m.payload(999), None);

        let loaded = load(&path).unwrap();
        assert_eq!(loaded.payload(1), Some(&b"first"[..]));
        assert_eq!(loaded.payload(2), None);
        assert_eq!(loaded.payload(3), Some(&b"third-doc"[..]));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn no_extra_sections_is_byte_identical() {
        // An index with neither tombstones nor payloads must serialize exactly
        // as the original v1 format (preserving MoonBit byte-compatibility).
        let mut idx = FlatIndex::new(3, Metric::L2, true);
        idx.add(1, &[1.0, 2.0, 3.0]);
        idx.add(2, &[-1.0, 0.5, 4.0]);
        let path = tmp("noextra");
        save(&idx, &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        // flags == FLAG_HAS_RAW only.
        assert_eq!(read_u32(&bytes, 24), FLAG_HAS_RAW);
        let l = layout(3, 2, true, false, None);
        assert_eq!(bytes.len(), l.total);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn bytes_roundtrip_payloads_and_tombstone() {
        // to_bytes / from_bytes exercise the payload and tombstone sections
        // without touching the filesystem.
        let mut idx = FlatIndex::new(4, Metric::L2, true);
        idx.add_with_payload(1, &[1.0, 0.0, 0.0, 0.0], b"first");
        idx.add(2, &[0.0, 1.0, 0.0, 0.0]); // no payload
        idx.add_with_payload(3, &[0.0, 0.0, 1.0, 0.0], b"third-doc");
        idx.remove(2); // tombstone the middle row

        let bytes = to_bytes(&idx);
        let idx2 = from_bytes(&bytes).unwrap();

        // Search ids agree between original and reconstructed index.
        for q in [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
        ] {
            let a: Vec<u64> = idx.search(&q, 3, 4).iter().map(|h| h.id).collect();
            let c: Vec<u64> = idx2.search(&q, 3, 4).iter().map(|h| h.id).collect();
            assert_eq!(a, c);
            assert!(a.iter().all(|&id| id != 2)); // tombstoned row excluded
        }
        // Payloads and tombstone survived.
        assert_eq!(idx2.payload(1), Some(&b"first"[..]));
        assert_eq!(idx2.payload(3), Some(&b"third-doc"[..]));
        assert_eq!(idx2.live_len(), 2);
        // Byte-stable round trip.
        assert_eq!(to_bytes(&idx2), bytes);
    }

    #[test]
    fn roundtrip_compact_no_raw() {
        let mut idx = FlatIndex::new(4, Metric::Cosine, false);
        idx.add(7, &[1.0, 0.0, 0.0, 0.0]);
        idx.add(8, &[0.0, 1.0, 0.0, 0.0]);
        let path = tmp("compact");
        save(&idx, &path).unwrap();
        let m = open(&path).unwrap();
        assert!(!m.has_raw());
        let hits = m.search(&[0.9, 0.1, 0.0, 0.0], 1, 1);
        assert_eq!(hits[0].id, 7);
        std::fs::remove_file(&path).ok();
    }
}

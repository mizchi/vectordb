//! `.vecdb` single-file format: save to disk and zero-copy mmap load.
//!
//! Layout (little-endian), matching `DESIGN.md`:
//!
//! ```text
//! [ 64-byte header ][ ids u64 ][ scales f32 ][ sqnorms f32 ][ codes i8 ][ raw f32? ]
//! ```
//!
//! Each section starts on a 16-byte boundary. The header records `dim`,
//! `count`, `metric` and a flag for whether the raw f32 section is present.

use crate::index::{FlatIndex, Metric, View};
use memmap2::Mmap;
use std::fs::File;
use std::io::{self, Write};
use std::mem::{align_of, size_of};
use std::path::Path;

const MAGIC: &[u8; 8] = b"VECDB1\0\0";
const VERSION: u32 = 1;
const HEADER_LEN: usize = 64;
const FLAG_HAS_RAW: u32 = 1;

#[inline]
fn align16(x: usize) -> usize {
    (x + 15) & !15
}

/// Byte offsets of each section within the file.
struct Layout {
    ids: usize,
    scales: usize,
    sqnorms: usize,
    codes: usize,
    raw: usize,
    total: usize,
}

fn layout(dim: usize, count: usize, has_raw: bool) -> Layout {
    let ids = align16(HEADER_LEN);
    let scales = align16(ids + count * size_of::<u64>());
    let sqnorms = align16(scales + count * size_of::<f32>());
    let codes = align16(sqnorms + count * size_of::<f32>());
    let raw = align16(codes + count * dim * size_of::<i8>());
    let total = if has_raw {
        align16(raw + count * dim * size_of::<f32>())
    } else {
        raw
    };
    Layout {
        ids,
        scales,
        sqnorms,
        codes,
        raw,
        total,
    }
}

/// Serialize an index to a `.vecdb` file.
///
/// Tombstoned rows are in-memory only; call [`FlatIndex::compact`] before
/// saving to persist deletions (otherwise all rows, including tombstoned ones,
/// are written and would come back live on load).
pub fn save(index: &FlatIndex, path: impl AsRef<Path>) -> io::Result<()> {
    let dim = index.dim();
    let count = index.len();
    let has_raw = index.has_raw();
    let l = layout(dim, count, has_raw);

    let mut buf = vec![0u8; l.total];
    // Header.
    buf[0..8].copy_from_slice(MAGIC);
    buf[8..12].copy_from_slice(&VERSION.to_le_bytes());
    buf[12..16].copy_from_slice(&(index.metric() as u32).to_le_bytes());
    buf[16..20].copy_from_slice(&(dim as u32).to_le_bytes());
    buf[20..24].copy_from_slice(&(count as u32).to_le_bytes());
    let flags = if has_raw { FLAG_HAS_RAW } else { 0 };
    buf[24..28].copy_from_slice(&flags.to_le_bytes());

    // Sections. Copy each field's bytes into place.
    write_slice(&mut buf, l.ids, &index.ids);
    write_slice(&mut buf, l.scales, &index.scales);
    write_slice(&mut buf, l.sqnorms, &index.sqnorms);
    // codes are i8; reinterpret as u8 for the byte copy.
    let codes_u8: &[u8] =
        unsafe { std::slice::from_raw_parts(index.codes.as_ptr() as *const u8, index.codes.len()) };
    buf[l.codes..l.codes + codes_u8.len()].copy_from_slice(codes_u8);
    if has_raw {
        let raw = index.raw.as_ref().unwrap();
        write_slice(&mut buf, l.raw, raw);
    }

    let mut f = File::create(path)?;
    f.write_all(&buf)?;
    f.flush()?;
    Ok(())
}

fn write_slice<T: Copy>(buf: &mut [u8], off: usize, data: &[T]) {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) };
    buf[off..off + bytes.len()].copy_from_slice(bytes);
}

/// A zero-copy, read-only index backed by an mmap of a `.vecdb` file.
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

    /// Borrow the mmap'd sections as a [`View`] for querying.
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
                deleted: None,
            }
        }
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
    if dim == 0 {
        return Err(bad("zero dim"));
    }

    let l = layout(dim, count, has_raw);
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
    let raw = if has_raw {
        Some(typed_ptr::<f32>(base, l.raw)?)
    } else {
        None
    };

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
    })
}

/// Load a `.vecdb` file into an owned, mutable [`FlatIndex`] (copies data).
pub fn load(path: impl AsRef<Path>) -> io::Result<FlatIndex> {
    let m = open(path)?;
    let v = m.view();
    let mut idx = FlatIndex::new(v.dim, v.metric, v.has_raw());
    idx.ids.extend_from_slice(v.ids);
    idx.scales.extend_from_slice(v.scales);
    idx.sqnorms.extend_from_slice(v.sqnorms);
    idx.codes.extend_from_slice(v.codes);
    if let Some(raw) = v.raw {
        idx.raw.as_mut().unwrap().extend_from_slice(raw);
    }
    idx.deleted = vec![false; idx.ids.len()];
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

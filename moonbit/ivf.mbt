// IVF (inverted file) index for MoonBit: k-means coarse quantization + nprobe
// search. Mirrors the Rust `ivf.rs`. Reuses the SIMD distance kernels, the int8
// quantizer, and the bounded TopK from this package.

///|
pub struct IvfIndex {
  dim : Int
  metric : Metric
  nlist : Int
  count : Int
  centroids : Array[Array[Float]] // nlist rows of dim (processed space)
  offsets : Array[Int] // nlist + 1, CSR row offsets
  ids : Array[Int64]
  codes : FixedArray[Byte] // count * dim int8
  scales : Array[Float]
  sqnorms : Array[Float]
  raw : FixedArray[Byte]? // count * dim * 4 (f32 bytes) when kept
}

///|
fn l2_arr(a : Array[Float], b : Array[Float]) -> Float {
  let mut s : Float = 0.0
  for i in 0..<a.length() {
    let d = a[i] - b[i]
    s = s + d * d
  }
  s
}

///|
fn dot_arr(a : Array[Float], b : Array[Float]) -> Float {
  let mut s : Float = 0.0
  for i in 0..<a.length() {
    s = s + a[i] * b[i]
  }
  s
}

///|
fn copy_row(src : Array[Float]) -> Array[Float] {
  let out = Array::make(src.length(), (0.0 : Float))
  for i in 0..<src.length() {
    out[i] = src[i]
  }
  out
}

///|
fn nearest_centroid(
  row : Array[Float],
  centroids : Array[Array[Float]],
  nlist : Int,
) -> Int {
  let mut best = 0
  let mut best_d : Float = 3.0e38
  for c in 0..<nlist {
    let d = l2_arr(centroids[c], row)
    if d < best_d {
      best_d = d
      best = c
    }
  }
  best
}

///|
/// k-means++ (D^2) initialization, deterministic via a fixed-seed xorshift.
fn kmeanspp_init(
  data : Array[Array[Float]],
  nlist : Int,
) -> Array[Array[Float]] {
  let n = data.length()
  let mut rng : UInt64 = 0x9E3779B97F4A7C15UL ^
    n.to_int64().reinterpret_as_uint64()
  let next = fn() -> UInt64 {
    let mut x = rng
    x = x ^ (x << 13)
    x = x ^ (x >> 7)
    x = x ^ (x << 17)
    rng = x
    x
  }
  let unit = fn() -> Double {
    (next() >> 11).reinterpret_as_int64().to_double() / 9007199254740992.0
  }
  let centroids : Array[Array[Float]] = []
  let first = (next() % n.to_int64().reinterpret_as_uint64()).to_int()
  centroids.push(copy_row(data[first]))
  let d2 : Array[Float] = Array::make(n, (3.0e38 : Float))
  for c in 1..<nlist {
    let prev = centroids[c - 1]
    let mut sum = 0.0
    for i in 0..<n {
      let dd = l2_arr(data[i], prev)
      if dd < d2[i] {
        d2[i] = dd
      }
      sum = sum + d2[i].to_double()
    }
    let mut pick = n - 1
    if sum > 0.0 {
      let mut target = unit() * sum
      for i in 0..<n {
        target = target - d2[i].to_double()
        if target <= 0.0 {
          pick = i
          break
        }
      }
    } else {
      pick = (next() % n.to_int64().reinterpret_as_uint64()).to_int()
    }
    centroids.push(copy_row(data[pick]))
  }
  centroids
}

///|
/// Lloyd's k-means over `data` (n rows of dim). Returns (centroids, assign).
fn kmeans(
  data : Array[Array[Float]],
  nlist : Int,
  iters : Int,
  renorm : Bool,
) -> (Array[Array[Float]], Array[Int]) {
  let n = data.length()
  let dim = data[0].length()
  let centroids = kmeanspp_init(data, nlist)
  let assign = Array::make(n, 0)
  let it_n = if iters < 1 { 1 } else { iters }
  for _it in 0..<it_n {
    for i in 0..<n {
      assign[i] = nearest_centroid(data[i], centroids, nlist)
    }
    let sums : Array[Array[Float]] = []
    for _c in 0..<nlist {
      sums.push(Array::make(dim, (0.0 : Float)))
    }
    let counts = Array::make(nlist, 0)
    for i in 0..<n {
      let c = assign[i]
      counts[c] = counts[c] + 1
      let row = data[i]
      let s = sums[c]
      for d in 0..<dim {
        s[d] = s[d] + row[d]
      }
    }
    for c in 0..<nlist {
      if counts[c] == 0 {
        // Empty cluster: reseed to a spread-out point.
        centroids[c] = copy_row(data[(c * 97 + 13) % n])
        continue
      }
      let inv = (1.0 : Float) / Float::from_double(counts[c].to_double())
      let cen = centroids[c]
      let s = sums[c]
      for d in 0..<dim {
        cen[d] = s[d] * inv
      }
      if renorm {
        let mut nrm : Float = 0.0
        for d in 0..<dim {
          nrm = nrm + cen[d] * cen[d]
        }
        let norm = Float::from_double(nrm.to_double().sqrt())
        if norm > 0.0 {
          let ninv = (1.0 : Float) / norm
          for d in 0..<dim {
            cen[d] = cen[d] * ninv
          }
        }
      }
    }
  }
  (centroids, assign)
}

///|
/// Build an IVF index. `nlist <= 0` picks ~sqrt(n) cells.
pub fn IvfIndex::build(
  vectors : Array[Array[Float]],
  ids : Array[Int64],
  metric : Metric,
  nlist : Int,
  keep_raw : Bool,
  kmeans_iters : Int,
) -> IvfIndex {
  let n = vectors.length()
  if n == 0 {
    abort("IvfIndex::build: empty input")
  }
  if ids.length() != n {
    abort("IvfIndex::build: ids/vectors length mismatch")
  }
  let dim = vectors[0].length()
  let nl = if nlist <= 0 {
    let s = n.to_double().sqrt().to_int()
    if s < 1 {
      1
    } else if s > n {
      n
    } else {
      s
    }
  } else if nlist > n {
    n
  } else {
    nlist
  }
  let cos = is_cosine(metric)

  // Processed vectors (normalized for cosine).
  let proc : Array[Array[Float]] = []
  for i in 0..<n {
    if vectors[i].length() != dim {
      abort("IvfIndex::build: inconsistent dimension")
    }
    proc.push(if cos { normalize(vectors[i]) } else { copy_row(vectors[i]) })
  }

  // Coarse quantizer.
  let (centroids, assign) = kmeans(proc, nl, kmeans_iters, cos)

  // CSR grouping by cell.
  let counts = Array::make(nl, 0)
  for i in 0..<n {
    counts[assign[i]] = counts[assign[i]] + 1
  }
  let offsets = Array::make(nl + 1, 0)
  for c in 0..<nl {
    offsets[c + 1] = offsets[c] + counts[c]
  }
  let cursor = Array::make(nl, 0)
  for c in 0..<nl {
    cursor[c] = offsets[c]
  }
  let ids_out = Array::make(n, (0 : Int64))
  let codes : FixedArray[Byte] = FixedArray::make(n * dim, 0)
  let scales = Array::make(n, (0.0 : Float))
  let sqnorms = Array::make(n, (0.0 : Float))
  let raw : FixedArray[Byte]? = if keep_raw {
    Some(FixedArray::make(n * dim * 4, 0))
  } else {
    None
  }
  for i in 0..<n {
    let cell = assign[i]
    let dst = cursor[cell]
    cursor[cell] = dst + 1
    let v = proc[i]
    let q = quantize(v)
    for j in 0..<dim {
      codes[dst * dim + j] = q.codes[j]
    }
    scales[dst] = q.scale
    sqnorms[dst] = q.sqnorm
    ids_out[dst] = ids[i]
    match raw {
      Some(rb) =>
        for j in 0..<dim {
          write_f32(rb, (dst * dim + j) * 4, v[j])
        }
      None => ()
    }
  }
  {
    dim,
    metric,
    nlist: nl,
    count: n,
    centroids,
    offsets,
    ids: ids_out,
    codes,
    scales,
    sqnorms,
    raw,
  }
}

///|
pub fn IvfIndex::length(self : IvfIndex) -> Int {
  self.count
}

///|
pub fn IvfIndex::nlist(self : IvfIndex) -> Int {
  self.nlist
}

///|
pub fn IvfIndex::has_raw(self : IvfIndex) -> Bool {
  match self.raw {
    Some(_) => true
    None => false
  }
}

///|
fn IvfIndex::nearest_cells(
  self : IvfIndex,
  query : Array[Float],
  nprobe : Int,
) -> Array[Int] {
  let l2 = is_l2(self.metric)
  let top = TopK::new(nprobe)
  for c in 0..<self.nlist {
    let cen = self.centroids[c]
    let key = if l2 { l2_arr(cen, query) } else { -dot_arr(cen, query) }
    top.push(key, c)
  }
  let sorted = top.drain_sorted()
  let cells : Array[Int] = []
  for i in 0..<sorted.length() {
    cells.push(sorted[i].idx)
  }
  cells
}

///|
/// Search the nprobe nearest cells for the k nearest neighbors.
pub fn IvfIndex::search(
  self : IvfIndex,
  query : Array[Float],
  k : Int,
  nprobe : Int,
  oversample : Int,
) -> Array[Hit] {
  self.search_filter(query, k, nprobe, oversample, fn(_id) { true })
}

///|
/// Like `search`, but only rows whose id satisfies `filter` are eligible. The
/// probed cells are chosen as usual; within each cell, non-matching ids are
/// skipped before entering the candidate heap (mirror of the Rust
/// `search_filter`).
pub fn IvfIndex::search_filter(
  self : IvfIndex,
  query : Array[Float],
  k : Int,
  nprobe : Int,
  oversample : Int,
  filter : (Int64) -> Bool,
) -> Array[Hit] {
  if k <= 0 || self.count == 0 {
    return []
  }
  if query.length() != self.dim {
    abort("IvfIndex::search_filter: dimension mismatch")
  }
  let dim = self.dim
  let l2 = is_l2(self.metric)
  let processed = if is_cosine(self.metric) { normalize(query) } else { query }
  let q = quantize(processed)
  let np = if nprobe < 1 {
    1
  } else if nprobe > self.nlist {
    self.nlist
  } else {
    nprobe
  }
  let cells = self.nearest_cells(processed, np)

  let cand_n = match self.raw {
    Some(_) => if oversample > 1 { k * oversample } else { k }
    None => k
  }
  let top = TopK::new(cand_n)
  for ci in 0..<cells.length() {
    let cell = cells[ci]
    for i in self.offsets[cell]..<self.offsets[cell + 1] {
      if !filter(self.ids[i]) {
        continue
      }
      let dp = dot_i8(self.codes, i * dim, q.codes, 0, dim)
      let approx = Float::from_double(dp.to_double()) * self.scales[i] * q.scale
      let key = if l2 {
        self.sqnorms[i] + q.sqnorm - 2.0 * approx
      } else {
        -approx
      }
      top.push(key, i)
    }
  }

  match self.raw {
    Some(rb) =>
      if oversample > 1 {
        let qbytes = f32_array_to_bytes(processed)
        let rr = TopK::new(k)
        let cand = top.drain_sorted()
        for c in 0..<cand.length() {
          let i = cand[c].idx
          let off = i * dim * 4
          let key = if l2 {
            l2sq_f32(rb, off, qbytes, 0, dim)
          } else {
            -dot_f32(rb, off, qbytes, 0, dim)
          }
          rr.push(key, i)
        }
        ivf_finish(rr.drain_sorted(), k, self.ids, self.metric)
      } else {
        ivf_finish(top.drain_sorted(), k, self.ids, self.metric)
      }
    None => ivf_finish(top.drain_sorted(), k, self.ids, self.metric)
  }
}

///|
fn ivf_finish(
  sorted : Array[Cand],
  k : Int,
  ids : Array[Int64],
  metric : Metric,
) -> Array[Hit] {
  let hib = higher_is_better(metric)
  let n = min(k, sorted.length())
  let out : Array[Hit] = []
  for c in 0..<n {
    let cand = sorted[c]
    let score = if hib { -cand.key } else { cand.key }
    out.push({ id: ids[cand.idx], score })
  }
  out
}

// ---------------------------------------------------------------------------
// Persistence: IVF `.vecdb` (magic "VECDBIV1"), byte-compatible with Rust.
// Layout: 64B header + 16B-aligned sections:
//   [header][ centroids f32 ][ offsets u64 ][ ids u64 ]
//           [ scales f32 ][ sqnorms f32 ][ codes i8 ][ raw f32? ]
// (k-means is not identical across languages, so byte-for-byte equality with
//  Rust is not expected; the *format* is compatible — each can read the other.)
// ---------------------------------------------------------------------------

///|
priv struct IvfLayout {
  centroids : Int
  offsets : Int
  ids : Int
  scales : Int
  sqnorms : Int
  codes : Int
  raw : Int
  total : Int
}

///|
fn ivf_layout(dim : Int, count : Int, nlist : Int, has_raw : Bool) -> IvfLayout {
  let centroids = align16(64)
  let offsets = align16(centroids + nlist * dim * 4)
  let ids = align16(offsets + (nlist + 1) * 8)
  let scales = align16(ids + count * 8)
  let sqnorms = align16(scales + count * 4)
  let codes = align16(sqnorms + count * 4)
  let raw = align16(codes + count * dim)
  let total = if has_raw { align16(raw + count * dim * 4) } else { raw }
  { centroids, offsets, ids, scales, sqnorms, codes, raw, total }
}

///|
/// Serialize the IVF index to an IVF `.vecdb` byte buffer.
pub fn IvfIndex::to_bytes(self : IvfIndex) -> FixedArray[Byte] {
  let dim = self.dim
  let count = self.count
  let nlist = self.nlist
  let has_raw = self.has_raw()
  let l = ivf_layout(dim, count, nlist, has_raw)
  let b : FixedArray[Byte] = FixedArray::make(l.total, 0)

  let magic : FixedArray[Byte] = [
    0x56, 0x45, 0x43, 0x44, 0x42, 0x49, 0x56, 0x31,
  ]
  for i in 0..<8 {
    b[i] = magic[i]
  }
  write_u32_le(b, 8, 1)
  write_u32_le(b, 12, metric_to_u32(self.metric))
  write_u32_le(b, 16, dim.reinterpret_as_uint())
  write_u32_le(b, 20, count.reinterpret_as_uint())
  write_u32_le(b, 24, if has_raw { 1 } else { 0 })
  write_u32_le(b, 28, nlist.reinterpret_as_uint())

  for c in 0..<nlist {
    let row = self.centroids[c]
    for d in 0..<dim {
      write_f32(b, l.centroids + (c * dim + d) * 4, row[d])
    }
  }
  for i in 0..<(nlist + 1) {
    write_u64_le(
      b,
      l.offsets + i * 8,
      self.offsets[i].to_int64().reinterpret_as_uint64(),
    )
  }
  for i in 0..<count {
    write_u64_le(b, l.ids + i * 8, self.ids[i].reinterpret_as_uint64())
    write_f32(b, l.scales + i * 4, self.scales[i])
    write_f32(b, l.sqnorms + i * 4, self.sqnorms[i])
  }
  for j in 0..<(count * dim) {
    b[l.codes + j] = self.codes[j]
  }
  match self.raw {
    Some(rb) =>
      for j in 0..<(count * dim * 4) {
        b[l.raw + j] = rb[j]
      }
    None => ()
  }
  b
}

///|
/// Parse an IVF `.vecdb` byte buffer into an [`IvfIndex`].
pub fn IvfIndex::from_bytes(b : FixedArray[Byte]) -> IvfIndex {
  if b.length() < 64 {
    abort("IvfIndex::from_bytes: buffer smaller than header")
  }
  // magic "VECDBIV1"
  if b[0] != 0x56 || b[5] != 0x49 || b[6] != 0x56 || b[7] != 0x31 {
    abort("IvfIndex::from_bytes: bad magic")
  }
  if read_u32_le(b, 8) != 1 {
    abort("IvfIndex::from_bytes: unsupported version")
  }
  let metric = metric_from_u32(read_u32_le(b, 12))
  let dim = read_u32_le(b, 16).reinterpret_as_int()
  let count = read_u32_le(b, 20).reinterpret_as_int()
  let has_raw = read_u32_le(b, 24) != 0
  let nlist = read_u32_le(b, 28).reinterpret_as_int()
  let l = ivf_layout(dim, count, nlist, has_raw)
  if b.length() < l.total {
    abort("IvfIndex::from_bytes: buffer truncated")
  }

  let centroids : Array[Array[Float]] = []
  for c in 0..<nlist {
    let row = Array::make(dim, (0.0 : Float))
    for d in 0..<dim {
      row[d] = read_f32(b, l.centroids + (c * dim + d) * 4)
    }
    centroids.push(row)
  }
  let offsets = Array::make(nlist + 1, 0)
  for i in 0..<(nlist + 1) {
    offsets[i] = read_u64_le(b, l.offsets + i * 8)
      .reinterpret_as_int64()
      .to_int()
  }
  let ids = Array::make(count, (0 : Int64))
  let scales = Array::make(count, (0.0 : Float))
  let sqnorms = Array::make(count, (0.0 : Float))
  for i in 0..<count {
    ids[i] = read_u64_le(b, l.ids + i * 8).reinterpret_as_int64()
    scales[i] = read_f32(b, l.scales + i * 4)
    sqnorms[i] = read_f32(b, l.sqnorms + i * 4)
  }
  let codes : FixedArray[Byte] = FixedArray::make(count * dim, 0)
  for j in 0..<(count * dim) {
    codes[j] = b[l.codes + j]
  }
  let raw : FixedArray[Byte]? = if has_raw {
    let rb : FixedArray[Byte] = FixedArray::make(count * dim * 4, 0)
    for j in 0..<(count * dim * 4) {
      rb[j] = b[l.raw + j]
    }
    Some(rb)
  } else {
    None
  }
  {
    dim,
    metric,
    nlist,
    count,
    centroids,
    offsets,
    ids,
    codes,
    scales,
    sqnorms,
    raw,
  }
}

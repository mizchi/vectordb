// Product Quantization for MoonBit, mirroring the Rust `pq.rs`.
//
// Split each vector into `m` subvectors of `dim/m` dims and quantize each
// against a per-subspace k-means codebook (`ksub` centroids, 1 byte/subspace
// when ksub<=256). Search uses ADC: precompute an `m*ksub` query↔centroid
// table, then each vector's approximate distance is `m` table lookups summed.
// Reuses kmeans (ivf.mbt), TopK/bin_finish, the f32 SIMD kernels, and storage
// helpers. Compatible `.vecdb` layout (magic "VECDBPQ1") with the Rust crate.

///|
pub struct PqIndex {
  dim : Int
  metric : Metric
  m : Int
  dsub : Int
  ksub : Int
  count : Int
  codebooks : Array[Array[Float]] // m*ksub rows of dsub ([j*ksub + c])
  codes : FixedArray[Byte] // count * m
  ids : Array[Int64]
  raw : FixedArray[Byte]? // count * dim * 4 when kept
}

///|
/// Slice a subvector `[off, off+len)` out of `v` into a fresh array.
fn sub_slice(v : Array[Float], off : Int, len : Int) -> Array[Float] {
  let out = Array::make(len, (0.0 : Float))
  for d in 0..<len {
    out[d] = v[off + d]
  }
  out
}

// ---------------------------------------------------------------------------
// Shared PQ primitives, reused by PqIndex, IvfPqIndex, and OpqIndex.
// ---------------------------------------------------------------------------

///|
/// Train `m` per-subspace codebooks (each `ksub` centroids of `dsub`) over
/// `rows` (n vectors of dim = m*dsub). Returns `m*ksub` rows (`[j*ksub + c]`).
fn pq_train(
  rows : Array[Array[Float]],
  m : Int,
  ksub : Int,
  dsub : Int,
  iters : Int,
) -> Array[Array[Float]] {
  let cb : Array[Array[Float]] = Array::make(m * ksub, [])
  let n = rows.length()
  for j in 0..<m {
    let sub : Array[Array[Float]] = []
    for i in 0..<n {
      sub.push(sub_slice(rows[i], j * dsub, dsub))
    }
    // renorm=false: per-subspace renorm would distort geometry.
    let (cen, _) = kmeans(sub, ksub, iters, false)
    for c in 0..<ksub {
      cb[j * ksub + c] = cen[c]
    }
  }
  cb
}

///|
/// Encode one vector into `m` PQ codes (nearest centroid per subspace by L2).
fn pq_encode(
  cb : Array[Array[Float]],
  m : Int,
  ksub : Int,
  dsub : Int,
  v : Array[Float],
) -> FixedArray[Byte] {
  let out : FixedArray[Byte] = FixedArray::make(m, 0)
  for j in 0..<m {
    let vs = sub_slice(v, j * dsub, dsub)
    let mut best = 0
    let mut best_d : Float = 3.0e38
    for c in 0..<ksub {
      let d = l2_arr(vs, cb[j * ksub + c])
      if d < best_d {
        best_d = d
        best = c
      }
    }
    out[j] = best.to_byte()
  }
  out
}

///|
/// ADC lookup table for `query`: `lut[j*ksub + c]` is the subspace-j,
/// centroid-c contribution (squared-L2 when l2, else dot product).
fn pq_lut(
  cb : Array[Array[Float]],
  m : Int,
  ksub : Int,
  dsub : Int,
  query : Array[Float],
  l2 : Bool,
) -> Array[Float] {
  let lut = Array::make(m * ksub, (0.0 : Float))
  for j in 0..<m {
    let qs = sub_slice(query, j * dsub, dsub)
    for c in 0..<ksub {
      let cen = cb[j * ksub + c]
      lut[j * ksub + c] = if l2 { l2_arr(qs, cen) } else { dot_arr(qs, cen) }
    }
  }
  lut
}

///|
/// Sum an ADC table over `m` codes starting at `codes[off]`.
fn pq_adc_sum(
  lut : Array[Float],
  ksub : Int,
  codes : FixedArray[Byte],
  off : Int,
  m : Int,
) -> Float {
  let mut acc : Float = 0.0
  for j in 0..<m {
    acc = acc + lut[j * ksub + codes[off + j].to_int()]
  }
  acc
}

///|
/// Build a PQ index. `m` must divide `dim`; `ksub <= 256`.
pub fn PqIndex::build(
  vectors : Array[Array[Float]],
  ids : Array[Int64],
  metric : Metric,
  m : Int,
  ksub : Int,
  kmeans_iters : Int,
  keep_raw : Bool,
) -> PqIndex {
  let n = vectors.length()
  if n == 0 {
    abort("PqIndex::build: empty input")
  }
  if ids.length() != n {
    abort("PqIndex::build: ids/vectors length mismatch")
  }
  let dim = vectors[0].length()
  if m <= 0 || dim % m != 0 {
    abort("PqIndex::build: m must divide dim")
  }
  if ksub <= 0 || ksub > 256 {
    abort("PqIndex::build: ksub must be in 1..=256")
  }
  let dsub = dim / m
  let ks = if ksub > n { n } else { ksub }
  let cos = is_cosine(metric)
  let proc : Array[Array[Float]] = []
  for i in 0..<n {
    if vectors[i].length() != dim {
      abort("PqIndex::build: inconsistent dimension")
    }
    proc.push(if cos { normalize(vectors[i]) } else { copy_row(vectors[i]) })
  }
  let codebooks = pq_train(proc, m, ks, dsub, kmeans_iters)
  let codes : FixedArray[Byte] = FixedArray::make(n * m, 0)
  for i in 0..<n {
    let c = pq_encode(codebooks, m, ks, dsub, proc[i])
    for j in 0..<m {
      codes[i * m + j] = c[j]
    }
  }
  let raw : FixedArray[Byte]? = if keep_raw {
    Some(FixedArray::make(n * dim * 4, 0))
  } else {
    None
  }
  match raw {
    Some(rb) =>
      for i in 0..<n {
        for d in 0..<dim {
          write_f32(rb, (i * dim + d) * 4, proc[i][d])
        }
      }
    None => ()
  }
  { dim, metric, m, dsub, ksub: ks, count: n, codebooks, codes, ids, raw }
}

///|
pub fn PqIndex::length(self : PqIndex) -> Int {
  self.count
}

///|
pub fn PqIndex::has_raw(self : PqIndex) -> Bool {
  match self.raw {
    Some(_) => true
    None => false
  }
}

///|
pub fn PqIndex::code_bytes(self : PqIndex) -> Int {
  self.count * self.m
}

///|
/// Search for the k nearest neighbors with ADC, then rerank `k*oversample`
/// candidates with exact f32 distances (if originals kept).
pub fn PqIndex::search(
  self : PqIndex,
  query : Array[Float],
  k : Int,
  oversample : Int,
) -> Array[Hit] {
  if k <= 0 || self.count == 0 {
    return []
  }
  if query.length() != self.dim {
    abort("PqIndex::search: dimension mismatch")
  }
  let processed = if is_cosine(self.metric) { normalize(query) } else { query }
  let l2 = is_l2(self.metric)
  let lut = pq_lut(self.codebooks, self.m, self.ksub, self.dsub, processed, l2)
  let cand_n = match self.raw {
    Some(_) => if oversample > 1 { k * oversample } else { k }
    None => k
  }
  let top = TopK::new(cand_n)
  for i in 0..<self.count {
    let acc = pq_adc_sum(lut, self.ksub, self.codes, i * self.m, self.m)
    let key = if l2 { acc } else { -acc }
    top.push(key, i)
  }
  match self.raw {
    Some(rb) =>
      if oversample > 1 {
        let qbytes = f32_array_to_bytes(processed)
        let rr = TopK::new(k)
        let cand = top.drain_sorted()
        for c in 0..<cand.length() {
          let i = cand[c].idx
          let off = i * self.dim * 4
          let key = if l2 {
            l2sq_f32(rb, off, qbytes, 0, self.dim)
          } else {
            -dot_f32(rb, off, qbytes, 0, self.dim)
          }
          rr.push(key, i)
        }
        bin_finish(rr.drain_sorted(), k, self.ids, self.metric, true)
      } else {
        bin_finish(top.drain_sorted(), k, self.ids, self.metric, false)
      }
    None => bin_finish(top.drain_sorted(), k, self.ids, self.metric, false)
  }
}

// ---------------------------------------------------------------------------
// Persistence: PQ `.vecdb` (magic "VECDBPQ1"), same layout as the Rust crate.
// Header (64B) + codebooks f32 + ids u64 + codes u8 + raw f32?.
// (Codebooks differ from Rust because k-means++ RNG differs across languages,
//  so bytes are not identical; the *format* is compatible — each reads the
//  other's data sections.)
// ---------------------------------------------------------------------------

///|
priv struct PqLayout {
  codebooks : Int
  ids : Int
  codes : Int
  raw : Int
  total : Int
}

///|
fn pq_layout(
  dim : Int,
  count : Int,
  m : Int,
  ksub : Int,
  dsub : Int,
  has_raw : Bool,
) -> PqLayout {
  let codebooks = align16(64)
  let ids = align16(codebooks + m * ksub * dsub * 4)
  let codes = align16(ids + count * 8)
  let raw = align16(codes + count * m)
  let total = if has_raw { align16(raw + count * dim * 4) } else { raw }
  { codebooks, ids, codes, raw, total }
}

///|
/// Serialize the index to a PQ `.vecdb` byte buffer.
pub fn PqIndex::to_bytes(self : PqIndex) -> FixedArray[Byte] {
  let dim = self.dim
  let count = self.count
  let has_raw = self.has_raw()
  let l = pq_layout(dim, count, self.m, self.ksub, self.dsub, has_raw)
  let b : FixedArray[Byte] = FixedArray::make(l.total, 0)
  let magic : FixedArray[Byte] = [
    0x56, 0x45, 0x43, 0x44, 0x42, 0x50, 0x51, 0x31,
  ]
  for i in 0..<8 {
    b[i] = magic[i]
  }
  write_u32_le(b, 8, 1)
  write_u32_le(b, 12, metric_to_u32(self.metric))
  write_u32_le(b, 16, dim.reinterpret_as_uint())
  write_u32_le(b, 20, count.reinterpret_as_uint())
  write_u32_le(b, 24, if has_raw { 1 } else { 0 })
  write_u32_le(b, 28, self.m.reinterpret_as_uint())
  write_u32_le(b, 32, self.ksub.reinterpret_as_uint())
  for jc in 0..<(self.m * self.ksub) {
    let row = self.codebooks[jc]
    for d in 0..<self.dsub {
      write_f32(b, l.codebooks + (jc * self.dsub + d) * 4, row[d])
    }
  }
  for i in 0..<count {
    write_u64_le(b, l.ids + i * 8, self.ids[i].reinterpret_as_uint64())
  }
  for j in 0..<(count * self.m) {
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
/// Parse a PQ `.vecdb` byte buffer into a [`PqIndex`].
pub fn PqIndex::from_bytes(b : FixedArray[Byte]) -> PqIndex {
  if b.length() < 64 {
    abort("PqIndex::from_bytes: buffer smaller than header")
  }
  // magic "VECDBPQ1"
  if b[0] != 0x56 || b[5] != 0x50 || b[6] != 0x51 || b[7] != 0x31 {
    abort("PqIndex::from_bytes: bad magic")
  }
  if read_u32_le(b, 8) != 1 {
    abort("PqIndex::from_bytes: unsupported version")
  }
  let metric = metric_from_u32(read_u32_le(b, 12))
  let dim = read_u32_le(b, 16).reinterpret_as_int()
  let count = read_u32_le(b, 20).reinterpret_as_int()
  let has_raw = (read_u32_le(b, 24) & 1) != 0
  let m = read_u32_le(b, 28).reinterpret_as_int()
  let ksub = read_u32_le(b, 32).reinterpret_as_int()
  if dim == 0 || m == 0 || dim % m != 0 {
    abort("PqIndex::from_bytes: bad dim/m")
  }
  let dsub = dim / m
  let l = pq_layout(dim, count, m, ksub, dsub, has_raw)
  if b.length() < l.total {
    abort("PqIndex::from_bytes: buffer truncated")
  }
  let codebooks : Array[Array[Float]] = Array::make(m * ksub, [])
  for jc in 0..<(m * ksub) {
    let row = Array::make(dsub, (0.0 : Float))
    for d in 0..<dsub {
      row[d] = read_f32(b, l.codebooks + (jc * dsub + d) * 4)
    }
    codebooks[jc] = row
  }
  let ids = Array::make(count, (0 : Int64))
  for i in 0..<count {
    ids[i] = read_u64_le(b, l.ids + i * 8).reinterpret_as_int64()
  }
  let codes : FixedArray[Byte] = FixedArray::make(count * m, 0)
  for j in 0..<(count * m) {
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
  { dim, metric, m, dsub, ksub, count, codebooks, codes, ids, raw }
}

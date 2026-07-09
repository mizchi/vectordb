// IVF + Product Quantization for MoonBit, mirroring the Rust `ivf_pq.rs`
// (Faiss IVFPQ). A coarse k-means quantizer assigns each vector to a cell; the
// residual `x - centroid` is encoded with a shared PQ codebook trained on
// residuals. Search probes `nprobe` cells, evaluates residual ADC per cell,
// then reranks. Reuses kmeans (ivf.mbt), the shared PQ helpers (pq.mbt),
// TopK/bin_finish, and the storage helpers. `.vecdb` layout "VECDBIP1".

///|
pub struct IvfPqIndex {
  dim : Int
  metric : Metric
  nlist : Int
  m : Int
  dsub : Int
  ksub : Int
  count : Int
  centroids : Array[Array[Float]] // nlist rows of dim
  pq_codebooks : Array[Array[Float]] // m*ksub rows of dsub (trained on residuals)
  offsets : Array[Int] // nlist + 1
  ids : Array[Int64] // grouped by cell
  codes : FixedArray[Byte] // count * m, grouped by cell
  raw : FixedArray[Byte]? // count * dim * 4 (processed) when kept
}

///|
/// Build an IVF+PQ index. `nlist <= 0` picks ~sqrt(n); `m` must divide `dim`.
pub fn IvfPqIndex::build(
  vectors : Array[Array[Float]],
  ids : Array[Int64],
  metric : Metric,
  nlist : Int,
  m : Int,
  ksub : Int,
  kmeans_iters : Int,
  keep_raw : Bool,
) -> IvfPqIndex {
  let n = vectors.length()
  if n == 0 {
    abort("IvfPqIndex::build: empty input")
  }
  if ids.length() != n {
    abort("IvfPqIndex::build: ids/vectors length mismatch")
  }
  let dim = vectors[0].length()
  if m <= 0 || dim % m != 0 {
    abort("IvfPqIndex::build: m must divide dim")
  }
  if ksub <= 0 || ksub > 256 {
    abort("IvfPqIndex::build: ksub must be in 1..=256")
  }
  let dsub = dim / m
  let ks = if ksub > n { n } else { ksub }
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
  let proc : Array[Array[Float]] = []
  for i in 0..<n {
    if vectors[i].length() != dim {
      abort("IvfPqIndex::build: inconsistent dimension")
    }
    proc.push(if cos { normalize(vectors[i]) } else { copy_row(vectors[i]) })
  }
  let (centroids, assign) = kmeans(proc, nl, kmeans_iters, cos)

  // Residuals x - centroid.
  let residuals : Array[Array[Float]] = Array::make(n, [])
  for i in 0..<n {
    let cen = centroids[assign[i]]
    let r = Array::make(dim, (0.0 : Float))
    for d in 0..<dim {
      r[d] = proc[i][d] - cen[d]
    }
    residuals[i] = r
  }
  let pq_codebooks = pq_train(residuals, m, ks, dsub, kmeans_iters)

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
  let codes : FixedArray[Byte] = FixedArray::make(n * m, 0)
  let raw : FixedArray[Byte]? = if keep_raw {
    Some(FixedArray::make(n * dim * 4, 0))
  } else {
    None
  }
  for i in 0..<n {
    let cell = assign[i]
    let dst = cursor[cell]
    cursor[cell] = dst + 1
    ids_out[dst] = ids[i]
    let c = pq_encode(pq_codebooks, m, ks, dsub, residuals[i])
    for j in 0..<m {
      codes[dst * m + j] = c[j]
    }
    match raw {
      Some(rb) =>
        for d in 0..<dim {
          write_f32(rb, (dst * dim + d) * 4, proc[i][d])
        }
      None => ()
    }
  }
  {
    dim,
    metric,
    nlist: nl,
    m,
    dsub,
    ksub: ks,
    count: n,
    centroids,
    pq_codebooks,
    offsets,
    ids: ids_out,
    codes,
    raw,
  }
}

///|
pub fn IvfPqIndex::length(self : IvfPqIndex) -> Int {
  self.count
}

///|
pub fn IvfPqIndex::nlist(self : IvfPqIndex) -> Int {
  self.nlist
}

///|
pub fn IvfPqIndex::has_raw(self : IvfPqIndex) -> Bool {
  match self.raw {
    Some(_) => true
    None => false
  }
}

///|
pub fn IvfPqIndex::code_bytes(self : IvfPqIndex) -> Int {
  self.count * self.m
}

///|
/// Search the `nprobe` nearest cells with residual ADC, then rerank.
pub fn IvfPqIndex::search(
  self : IvfPqIndex,
  query : Array[Float],
  k : Int,
  nprobe : Int,
  oversample : Int,
) -> Array[Hit] {
  if k <= 0 || self.count == 0 {
    return []
  }
  if query.length() != self.dim {
    abort("IvfPqIndex::search: dimension mismatch")
  }
  let dim = self.dim
  let l2 = is_l2(self.metric)
  let processed = if is_cosine(self.metric) { normalize(query) } else { query }
  let np = if nprobe < 1 {
    1
  } else if nprobe > self.nlist {
    self.nlist
  } else {
    nprobe
  }

  // Pick the nprobe nearest cells.
  let cell_top = TopK::new(np)
  for c in 0..<self.nlist {
    let cen = self.centroids[c]
    let key = if l2 { l2_arr(cen, processed) } else { -dot_arr(cen, processed) }
    cell_top.push(key, c)
  }
  let cells = cell_top.drain_sorted()

  // For dot/cosine the ADC table is cell-independent (full query); only the
  // <q, centroid> offset varies. For L2 the table uses the per-cell residual.
  let dot_lut = if l2 {
    []
  } else {
    pq_lut(self.pq_codebooks, self.m, self.ksub, self.dsub, processed, false)
  }
  let cand_n = match self.raw {
    Some(_) => if oversample > 1 { k * oversample } else { k }
    None => k
  }
  let top = TopK::new(cand_n)
  let qr = Array::make(dim, (0.0 : Float))
  for ci in 0..<cells.length() {
    let c = cells[ci].idx
    let cen = self.centroids[c]
    let lut = if l2 {
      for d in 0..<dim {
        qr[d] = processed[d] - cen[d]
      }
      pq_lut(self.pq_codebooks, self.m, self.ksub, self.dsub, qr, true)
    } else {
      dot_lut
    }
    let qc : Float = if l2 { 0.0 } else { dot_arr(processed, cen) }
    for i in self.offsets[c]..<self.offsets[c + 1] {
      let acc = pq_adc_sum(lut, self.ksub, self.codes, i * self.m, self.m)
      let key = if l2 { acc } else { -(qc + acc) }
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
        bin_finish(rr.drain_sorted(), k, self.ids, self.metric, true)
      } else {
        bin_finish(top.drain_sorted(), k, self.ids, self.metric, false)
      }
    None => bin_finish(top.drain_sorted(), k, self.ids, self.metric, false)
  }
}

// ---------------------------------------------------------------------------
// Persistence: IVF+PQ `.vecdb` (magic "VECDBIP1"), same layout as the Rust
// crate. Header (64B) + centroids f32 + pq_codebooks f32 + offsets u64
// + ids u64 + codes u8 + raw f32?. (k-means RNG differs across languages, so
// bytes are not identical with Rust; the format is cross-readable.)
// ---------------------------------------------------------------------------

///|
priv struct IpLayout {
  centroids : Int
  codebooks : Int
  offsets : Int
  ids : Int
  codes : Int
  raw : Int
  total : Int
}

///|
fn ip_layout(
  dim : Int,
  count : Int,
  nlist : Int,
  m : Int,
  ksub : Int,
  dsub : Int,
  has_raw : Bool,
) -> IpLayout {
  let centroids = align16(64)
  let codebooks = align16(centroids + nlist * dim * 4)
  let offsets = align16(codebooks + m * ksub * dsub * 4)
  let ids = align16(offsets + (nlist + 1) * 8)
  let codes = align16(ids + count * 8)
  let raw = align16(codes + count * m)
  let total = if has_raw { align16(raw + count * dim * 4) } else { raw }
  { centroids, codebooks, offsets, ids, codes, raw, total }
}

///|
/// Serialize the index to an IVF+PQ `.vecdb` byte buffer.
pub fn IvfPqIndex::to_bytes(self : IvfPqIndex) -> FixedArray[Byte] {
  let dim = self.dim
  let count = self.count
  let has_raw = self.has_raw()
  let l = ip_layout(
    dim,
    count,
    self.nlist,
    self.m,
    self.ksub,
    self.dsub,
    has_raw,
  )
  let b : FixedArray[Byte] = FixedArray::make(l.total, 0)
  let magic : FixedArray[Byte] = [
    0x56, 0x45, 0x43, 0x44, 0x42, 0x49, 0x50, 0x31,
  ]
  for i in 0..<8 {
    b[i] = magic[i]
  }
  write_u32_le(b, 8, 1)
  write_u32_le(b, 12, metric_to_u32(self.metric))
  write_u32_le(b, 16, dim.reinterpret_as_uint())
  write_u32_le(b, 20, count.reinterpret_as_uint())
  write_u32_le(b, 24, if has_raw { 1 } else { 0 })
  write_u32_le(b, 28, self.nlist.reinterpret_as_uint())
  write_u32_le(b, 32, self.m.reinterpret_as_uint())
  write_u32_le(b, 36, self.ksub.reinterpret_as_uint())
  for c in 0..<self.nlist {
    let row = self.centroids[c]
    for d in 0..<dim {
      write_f32(b, l.centroids + (c * dim + d) * 4, row[d])
    }
  }
  for jc in 0..<(self.m * self.ksub) {
    let row = self.pq_codebooks[jc]
    for d in 0..<self.dsub {
      write_f32(b, l.codebooks + (jc * self.dsub + d) * 4, row[d])
    }
  }
  for i in 0..<(self.nlist + 1) {
    write_u64_le(
      b,
      l.offsets + i * 8,
      self.offsets[i].to_int64().reinterpret_as_uint64(),
    )
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
/// Parse an IVF+PQ `.vecdb` byte buffer into an [`IvfPqIndex`].
pub fn IvfPqIndex::from_bytes(b : FixedArray[Byte]) -> IvfPqIndex {
  if b.length() < 64 {
    abort("IvfPqIndex::from_bytes: buffer smaller than header")
  }
  // magic "VECDBIP1"
  if b[0] != 0x56 || b[5] != 0x49 || b[6] != 0x50 || b[7] != 0x31 {
    abort("IvfPqIndex::from_bytes: bad magic")
  }
  if read_u32_le(b, 8) != 1 {
    abort("IvfPqIndex::from_bytes: unsupported version")
  }
  let metric = metric_from_u32(read_u32_le(b, 12))
  let dim = read_u32_le(b, 16).reinterpret_as_int()
  let count = read_u32_le(b, 20).reinterpret_as_int()
  let has_raw = (read_u32_le(b, 24) & 1) != 0
  let nlist = read_u32_le(b, 28).reinterpret_as_int()
  let m = read_u32_le(b, 32).reinterpret_as_int()
  let ksub = read_u32_le(b, 36).reinterpret_as_int()
  if dim == 0 || m == 0 || dim % m != 0 {
    abort("IvfPqIndex::from_bytes: bad dim/m")
  }
  let dsub = dim / m
  let l = ip_layout(dim, count, nlist, m, ksub, dsub, has_raw)
  if b.length() < l.total {
    abort("IvfPqIndex::from_bytes: buffer truncated")
  }
  let centroids : Array[Array[Float]] = Array::make(nlist, [])
  for c in 0..<nlist {
    let row = Array::make(dim, (0.0 : Float))
    for d in 0..<dim {
      row[d] = read_f32(b, l.centroids + (c * dim + d) * 4)
    }
    centroids[c] = row
  }
  let pq_codebooks : Array[Array[Float]] = Array::make(m * ksub, [])
  for jc in 0..<(m * ksub) {
    let row = Array::make(dsub, (0.0 : Float))
    for d in 0..<dsub {
      row[d] = read_f32(b, l.codebooks + (jc * dsub + d) * 4)
    }
    pq_codebooks[jc] = row
  }
  let offsets = Array::make(nlist + 1, 0)
  for i in 0..<(nlist + 1) {
    offsets[i] = read_u64_le(b, l.offsets + i * 8)
      .reinterpret_as_int64()
      .to_int()
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
  {
    dim,
    metric,
    nlist,
    m,
    dsub,
    ksub,
    count,
    centroids,
    pq_codebooks,
    offsets,
    ids,
    codes,
    raw,
  }
}

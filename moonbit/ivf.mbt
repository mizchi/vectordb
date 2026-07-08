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
/// Lloyd's k-means over `data` (n rows of dim). Returns (centroids, assign).
fn kmeans(
  data : Array[Array[Float]],
  nlist : Int,
  iters : Int,
  renorm : Bool,
) -> (Array[Array[Float]], Array[Int]) {
  let n = data.length()
  let dim = data[0].length()
  let stride = if n / nlist > 1 { n / nlist } else { 1 }
  let centroids : Array[Array[Float]] = []
  for c in 0..<nlist {
    centroids.push(copy_row(data[(c * stride) % n]))
  }
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
  if k <= 0 || self.count == 0 {
    return []
  }
  if query.length() != self.dim {
    abort("IvfIndex::search: dimension mismatch")
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
      let dp = dot_i8(self.codes, i * dim, q.codes, 0, dim)
      let approx = Float::from_double(dp.to_double()) * self.scales[i] * q.scale
      let key = if l2 { self.sqnorms[i] + q.sqnorm - 2.0 * approx } else { -approx }
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

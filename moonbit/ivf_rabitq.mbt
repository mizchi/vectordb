// IVF + RaBitQ for MoonBit, mirroring the Rust `IvfRabitqIndex`. Per-cell
// centroids keep residuals small and well spread, which is where RaBitQ's
// estimator becomes accurate. Reuses kmeans (ivf.mbt), the rotation +
// bitplane query (rabitq.mbt), TopK, and the SIMD f32 kernels.

///|
pub struct IvfRabitqIndex {
  dim : Int
  metric : Metric
  nlist : Int
  words : Int
  count : Int
  rot : Array[Float]
  centroids : Array[Array[Float]] // nlist rows
  cc : Array[Float] // <c, c> per cell
  offsets : Array[Int] // nlist + 1
  ids : Array[Int64]
  signs : Array[UInt64] // count * words
  popcnt : Array[Int]
  res_sq : Array[Float]
  coef : Array[Float]
  oc : Array[Float]
  raw : FixedArray[Byte]? // count * dim * 4 when kept
}

///|
/// Build an IVF+RaBitQ index. `nlist <= 0` picks ~sqrt(n) cells; `seed` fixes
/// the random rotation.
pub fn IvfRabitqIndex::build(
  vectors : Array[Array[Float]],
  ids : Array[Int64],
  metric : Metric,
  nlist : Int,
  keep_raw : Bool,
  kmeans_iters : Int,
  seed : UInt64,
) -> IvfRabitqIndex {
  let n = vectors.length()
  if n == 0 {
    abort("IvfRabitqIndex::build: empty input")
  }
  if ids.length() != n {
    abort("IvfRabitqIndex::build: ids/vectors length mismatch")
  }
  let dim = vectors[0].length()
  let words = (dim + 63) / 64
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
      abort("IvfRabitqIndex::build: inconsistent dimension")
    }
    proc.push(if cos { normalize(vectors[i]) } else { copy_row(vectors[i]) })
  }
  let (centroids, assign) = kmeans(proc, nl, kmeans_iters, cos)
  let cc = Array::make(nl, (0.0 : Float))
  for c in 0..<nl {
    cc[c] = dot_arr(centroids[c], centroids[c])
  }

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
  let rot = random_rotation(dim, seed)
  let signs = Array::make(n * words, 0UL)
  let popcnt = Array::make(n, 0)
  let res_sq = Array::make(n, (0.0 : Float))
  let coef = Array::make(n, (0.0 : Float))
  let oc = Array::make(n, (0.0 : Float))
  let ids_out = Array::make(n, (0 : Int64))
  let raw : FixedArray[Byte]? = if keep_raw {
    Some(FixedArray::make(n * dim * 4, 0))
  } else {
    None
  }
  let d_res = Array::make(dim, (0.0 : Float))
  let r = Array::make(dim, (0.0 : Float))
  for i in 0..<n {
    let cell = assign[i]
    let dst = cursor[cell]
    cursor[cell] = dst + 1
    let v = proc[i]
    let cen = centroids[cell]
    for j in 0..<dim {
      d_res[j] = v[j] - cen[j]
    }
    rq_matvec(rot, d_res, r, dim)
    let rsq = dot_arr(r, r)
    let mut sum_abs : Float = 0.0
    let mut pc = 0
    for j in 0..<dim {
      sum_abs = sum_abs + r[j].abs()
      if r[j] >= 0.0 {
        let w = dst * words + j / 64
        signs[w] = signs[w] | (1UL << (j % 64))
        pc = pc + 1
      }
    }
    popcnt[dst] = pc
    res_sq[dst] = rsq
    coef[dst] = if sum_abs > 0.0 { rsq / sum_abs } else { 0.0 }
    oc[dst] = dot_arr(v, cen)
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
    words,
    count: n,
    rot,
    centroids,
    cc,
    offsets,
    ids: ids_out,
    signs,
    popcnt,
    res_sq,
    coef,
    oc,
    raw,
  }
}

///|
pub fn IvfRabitqIndex::length(self : IvfRabitqIndex) -> Int {
  self.count
}

///|
pub fn IvfRabitqIndex::nlist(self : IvfRabitqIndex) -> Int {
  self.nlist
}

///|
pub fn IvfRabitqIndex::has_raw(self : IvfRabitqIndex) -> Bool {
  match self.raw {
    Some(_) => true
    None => false
  }
}

///|
/// Search the nprobe nearest cells with the RaBitQ estimator, then rerank.
pub fn IvfRabitqIndex::search(
  self : IvfRabitqIndex,
  query : Array[Float],
  k : Int,
  nprobe : Int,
  oversample : Int,
) -> Array[Hit] {
  if k <= 0 || self.count == 0 {
    return []
  }
  if query.length() != self.dim {
    abort("IvfRabitqIndex::search: dimension mismatch")
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

  let cand_n = match self.raw {
    Some(_) => if oversample > 1 { k * oversample } else { k }
    None => k
  }
  let top = TopK::new(cand_n)
  let qd = Array::make(dim, (0.0 : Float))
  let qr = Array::make(dim, (0.0 : Float))
  for ci in 0..<cells.length() {
    let c = cells[ci].idx
    let cen = self.centroids[c]
    for j in 0..<dim {
      qd[j] = processed[j] - cen[j]
    }
    rq_matvec(self.rot, qd, qr, dim)
    let qr_sq = dot_arr(qd, qd)
    let qc = dot_arr(processed, cen)
    let qq = QQuery::new(qr, self.words, dim)
    for i in self.offsets[c]..<self.offsets[c + 1] {
      let s = qq.estimate(self.signs, i * self.words, self.popcnt[i])
      let ip_c = self.coef[i] * s
      let key = if l2 {
        self.res_sq[i] + qr_sq - 2.0 * ip_c
      } else {
        -(ip_c + self.oc[i] + qc - self.cc[c])
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
        bin_finish(rr.drain_sorted(), k, self.ids, self.metric, true)
      } else {
        bin_finish(top.drain_sorted(), k, self.ids, self.metric, false)
      }
    None => bin_finish(top.drain_sorted(), k, self.ids, self.metric, false)
  }
}

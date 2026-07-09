// RaBitQ (Gao & Long, SIGMOD 2024) for MoonBit, mirroring the Rust `rabitq.rs`.
// Subtract a centroid, apply a fixed random rotation, keep 1 sign bit/dim, and
// recover an unbiased inner-product estimate from the sign code + a per-vector
// correction scalar. The query is quantized into QUERY_BITS bitplanes so the
// estimate is a few popcounts per vector.

///|
let query_bits : Int = 4

///|
pub struct RaBitqIndex {
  dim : Int
  metric : Metric
  words : Int
  count : Int
  rot : Array[Float] // dim * dim rotation (row-major)
  centroid : Array[Float]
  cc : Float // <c, c>
  ids : Array[Int64]
  signs : Array[UInt64] // count * words
  popcnt : Array[Int]
  res_sq : Array[Float] // ||o - c||^2
  coef : Array[Float] // ||r||^2 / sum|r_i|
  oc : Array[Float] // <o, c>
  raw : FixedArray[Byte]? // count * dim * 4 (f32 bytes) when kept
}

///|
fn rq_matvec(
  m : Array[Float],
  v : Array[Float],
  out : Array[Float],
  d : Int,
) -> Unit {
  for i in 0..<d {
    let base = i * d
    let mut s : Float = 0.0
    for j in 0..<d {
      s = s + m[base + j] * v[j]
    }
    out[i] = s
  }
}

///|
/// Random orthonormal `d x d` matrix (row-major) via Gram-Schmidt on seeded
/// Gaussian rows.
fn random_rotation(d : Int, seed : UInt64) -> Array[Float] {
  let mut rng = seed | 1UL
  let next = fn() -> UInt64 {
    let mut x = rng
    x = x ^ (x << 13)
    x = x ^ (x >> 7)
    x = x ^ (x << 17)
    rng = x
    x
  }
  let unit = fn() -> Double {
    let u = (next() >> 11).reinterpret_as_int64().to_double() /
      9007199254740992.0
    if u < 1.0e-7 {
      1.0e-7
    } else if u > 1.0 - 1.0e-7 {
      1.0 - 1.0e-7
    } else {
      u
    }
  }
  let gauss = fn() -> Float {
    let u1 = unit()
    let u2 = unit()
    Float::from_double(
      (-2.0 * @math.ln(u1)).sqrt() * @math.cos(6.283185307179586 * u2),
    )
  }
  let m = Array::make(d * d, (0.0 : Float))
  for i in 0..<(d * d) {
    m[i] = gauss()
  }
  for i in 0..<d {
    for p in 0..<i {
      let mut proj : Float = 0.0
      for j in 0..<d {
        proj = proj + m[i * d + j] * m[p * d + j]
      }
      for j in 0..<d {
        m[i * d + j] = m[i * d + j] - proj * m[p * d + j]
      }
    }
    let mut nrm : Float = 0.0
    for j in 0..<d {
      nrm = nrm + m[i * d + j] * m[i * d + j]
    }
    let norm = nrm.sqrt()
    let inv = if norm > 1.0e-12 { (1.0 : Float) / norm } else { 0.0 }
    for j in 0..<d {
      m[i * d + j] = m[i * d + j] * inv
    }
  }
  m
}

///|
/// Build a RaBitQ index. `seed` fixes the random rotation.
pub fn RaBitqIndex::build(
  vectors : Array[Array[Float]],
  ids : Array[Int64],
  metric : Metric,
  keep_raw : Bool,
  seed : UInt64,
) -> RaBitqIndex {
  let n = vectors.length()
  if n == 0 {
    abort("RaBitqIndex::build: empty input")
  }
  if ids.length() != n {
    abort("RaBitqIndex::build: ids/vectors length mismatch")
  }
  let dim = vectors[0].length()
  let words = (dim + 63) / 64
  let cos = is_cosine(metric)
  let proc : Array[Array[Float]] = []
  for i in 0..<n {
    if vectors[i].length() != dim {
      abort("RaBitqIndex::build: inconsistent dimension")
    }
    proc.push(if cos { normalize(vectors[i]) } else { copy_row(vectors[i]) })
  }
  let centroid = Array::make(dim, (0.0 : Float))
  for i in 0..<n {
    for d in 0..<dim {
      centroid[d] = centroid[d] + proc[i][d]
    }
  }
  let inv_n = (1.0 : Float) / Float::from_double(n.to_double())
  for d in 0..<dim {
    centroid[d] = centroid[d] * inv_n
  }
  let cc = dot_arr(centroid, centroid)
  let rot = random_rotation(dim, seed)
  let signs = Array::make(n * words, 0UL)
  let popcnt = Array::make(n, 0)
  let res_sq = Array::make(n, (0.0 : Float))
  let coef = Array::make(n, (0.0 : Float))
  let oc = Array::make(n, (0.0 : Float))
  let raw : FixedArray[Byte]? = if keep_raw {
    Some(FixedArray::make(n * dim * 4, 0))
  } else {
    None
  }
  let d_res = Array::make(dim, (0.0 : Float))
  let r = Array::make(dim, (0.0 : Float))
  for i in 0..<n {
    for j in 0..<dim {
      d_res[j] = proc[i][j] - centroid[j]
    }
    rq_matvec(rot, d_res, r, dim)
    let rsq = dot_arr(r, r)
    let mut sum_abs : Float = 0.0
    let mut pc = 0
    for j in 0..<dim {
      sum_abs = sum_abs + r[j].abs()
      if r[j] >= 0.0 {
        let w = i * words + j / 64
        signs[w] = signs[w] | (1UL << (j % 64))
        pc = pc + 1
      }
    }
    popcnt[i] = pc
    res_sq[i] = rsq
    coef[i] = if sum_abs > 0.0 { rsq / sum_abs } else { 0.0 }
    oc[i] = dot_arr(proc[i], centroid)
    match raw {
      Some(rb) =>
        for j in 0..<dim {
          write_f32(rb, (i * dim + j) * 4, proc[i][j])
        }
      None => ()
    }
  }
  {
    dim,
    metric,
    words,
    count: n,
    rot,
    centroid,
    cc,
    ids,
    signs,
    popcnt,
    res_sq,
    coef,
    oc,
    raw,
  }
}

///|
pub fn RaBitqIndex::length(self : RaBitqIndex) -> Int {
  self.count
}

///|
pub fn RaBitqIndex::has_raw(self : RaBitqIndex) -> Bool {
  match self.raw {
    Some(_) => true
    None => false
  }
}

///|
/// A query quantized into `query_bits` bitplanes for the popcount estimator.
priv struct QQuery {
  planes : Array[UInt64] // query_bits * words
  words : Int
  dim : Int
  vl : Float
  delta : Float
  total_qu : Float
}

///|
fn QQuery::new(qr : Array[Float], words : Int, dim : Int) -> QQuery {
  let mut vl : Float = 3.0e38
  let mut vmax : Float = -3.0e38
  for j in 0..<dim {
    if qr[j] < vl {
      vl = qr[j]
    }
    if qr[j] > vmax {
      vmax = qr[j]
    }
  }
  let levels : Float = 15.0 // 2^4 - 1
  let span = (vmax - vl) / levels
  let delta = if span > 1.0e-9 { span } else { 1.0e-9 }
  let planes = Array::make(query_bits * words, 0UL)
  let mut total_qu : Float = 0.0
  for j in 0..<dim {
    let q0 = round_f((qr[j] - vl) / delta)
    let qu = if q0 < 0 { 0 } else if q0 > 15 { 15 } else { q0 }
    total_qu = total_qu + Float::from_double(qu.to_double())
    for t in 0..<query_bits {
      if ((qu >> t) & 1) == 1 {
        let idx = t * words + j / 64
        planes[idx] = planes[idx] | (1UL << (j % 64))
      }
    }
  }
  { planes, words, dim, vl, delta, total_qu }
}

///|
fn QQuery::estimate(
  self : QQuery,
  signs : Array[UInt64],
  base : Int,
  popcnt : Int,
) -> Float {
  let mut inner = 0
  for t in 0..<query_bits {
    let mut pc = 0
    for w in 0..<self.words {
      pc = pc + (signs[base + w] & self.planes[t * self.words + w]).popcnt()
    }
    inner = inner + (pc << t)
  }
  self.delta * (2.0 * Float::from_double(inner.to_double()) - self.total_qu) +
  self.vl *
  (
    2.0 * Float::from_double(popcnt.to_double()) -
    Float::from_double(self.dim.to_double())
  )
}

///|
/// Search for the k nearest neighbors using the RaBitQ estimator, then rerank
/// `k * oversample` candidates with exact f32 distances (if originals kept).
pub fn RaBitqIndex::search(
  self : RaBitqIndex,
  query : Array[Float],
  k : Int,
  oversample : Int,
) -> Array[Hit] {
  if k <= 0 || self.count == 0 {
    return []
  }
  if query.length() != self.dim {
    abort("RaBitqIndex::search: dimension mismatch")
  }
  let processed = if is_cosine(self.metric) { normalize(query) } else { query }
  let qd = Array::make(self.dim, (0.0 : Float))
  for j in 0..<self.dim {
    qd[j] = processed[j] - self.centroid[j]
  }
  let qr = Array::make(self.dim, (0.0 : Float))
  rq_matvec(self.rot, qd, qr, self.dim)
  let qr_sq = dot_arr(qd, qd)
  let qc = dot_arr(processed, self.centroid)
  let l2 = is_l2(self.metric)
  let qq = QQuery::new(qr, self.words, self.dim)
  let cand_n = match self.raw {
    Some(_) => if oversample > 1 { k * oversample } else { k }
    None => k
  }
  let top = TopK::new(cand_n)
  for i in 0..<self.count {
    let s = qq.estimate(self.signs, i * self.words, self.popcnt[i])
    let ip_c = self.coef[i] * s
    let key = if l2 {
      self.res_sq[i] + qr_sq - 2.0 * ip_c
    } else {
      -(ip_c + self.oc[i] + qc - self.cc)
    }
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

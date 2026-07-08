// Flat (brute-force) index: int8 quantized scan + optional f32 rerank.
//
// Prototype counterpart of the Rust crate. Vectors are quantized to int8 for a
// fast SIMD scan; the f32 originals are optionally kept (as raw bytes) so a
// widened candidate set can be reranked with exact distances.

///|
pub(all) enum Metric {
  L2
  Dot
  Cosine
}

///|
pub struct Hit {
  id : Int64
  score : Float
}

///|
pub struct FlatIndex {
  dim : Int
  metric : Metric
  count : Int
  ids : Array[Int64]
  codes : FixedArray[Byte] // count * dim
  scales : Array[Float]
  sqnorms : Array[Float]
  raw : FixedArray[Byte]? // count * dim * 4 (f32 bytes), when kept
}

///|
priv struct Cand {
  key : Float
  idx : Int
}

// ---------------------------------------------------------------------------
// Bounded top-k: a fixed-capacity max-heap keyed on `key` (smaller = better).
// Keeping only the best `cap` candidates avoids sorting the whole dataset on
// every query, which dominated the earlier implementation.
// ---------------------------------------------------------------------------

///|
priv struct TopK {
  keys : FixedArray[Float]
  idxs : FixedArray[Int]
  mut size : Int
  cap : Int
}

///|
fn TopK::new(cap : Int) -> TopK {
  let c = if cap < 1 { 1 } else { cap }
  {
    keys: FixedArray::make(c, (0.0 : Float)),
    idxs: FixedArray::make(c, 0),
    size: 0,
    cap: c,
  }
}

///|
fn TopK::swap(self : TopK, a : Int, b : Int) -> Unit {
  let tk = self.keys[a]
  self.keys[a] = self.keys[b]
  self.keys[b] = tk
  let ti = self.idxs[a]
  self.idxs[a] = self.idxs[b]
  self.idxs[b] = ti
}

///|
fn TopK::sift_down(self : TopK, start : Int) -> Unit {
  let mut i = start
  let n = self.size
  while true {
    let l = 2 * i + 1
    let r = 2 * i + 2
    let mut largest = i
    if l < n && self.keys[l] > self.keys[largest] {
      largest = l
    }
    if r < n && self.keys[r] > self.keys[largest] {
      largest = r
    }
    if largest == i {
      break
    }
    self.swap(i, largest)
    i = largest
  }
}

///|
fn TopK::push(self : TopK, key : Float, idx : Int) -> Unit {
  if self.size < self.cap {
    let mut i = self.size
    self.keys[i] = key
    self.idxs[i] = idx
    self.size = self.size + 1
    while i > 0 {
      let p = (i - 1) / 2
      if self.keys[p] < self.keys[i] {
        self.swap(p, i)
        i = p
      } else {
        break
      }
    }
  } else if key < self.keys[0] {
    self.keys[0] = key
    self.idxs[0] = idx
    self.sift_down(0)
  }
}

///|
fn TopK::drain_sorted(self : TopK) -> Array[Cand] {
  let out : Array[Cand] = []
  for i in 0..<self.size {
    out.push({ key: self.keys[i], idx: self.idxs[i] })
  }
  out.sort_by(fn(a, b) { cmp_float(a.key, b.key) })
  out
}

// ---------------------------------------------------------------------------

///|
fn is_cosine(m : Metric) -> Bool {
  match m {
    Cosine => true
    _ => false
  }
}

///|
fn is_l2(m : Metric) -> Bool {
  match m {
    L2 => true
    _ => false
  }
}

///|
fn higher_is_better(m : Metric) -> Bool {
  match m {
    L2 => false
    _ => true
  }
}

///|
fn write_f32(b : FixedArray[Byte], off : Int, v : Float) -> Unit {
  let bits = v.reinterpret_as_uint()
  b[off] = (bits & 0xFF).to_byte()
  b[off + 1] = ((bits >> 8) & 0xFF).to_byte()
  b[off + 2] = ((bits >> 16) & 0xFF).to_byte()
  b[off + 3] = ((bits >> 24) & 0xFF).to_byte()
}

///|
fn f32_array_to_bytes(x : Array[Float]) -> FixedArray[Byte] {
  let b : FixedArray[Byte] = FixedArray::make(x.length() * 4, 0)
  for i in 0..<x.length() {
    write_f32(b, i * 4, x[i])
  }
  b
}

///|
fn normalize(v : Array[Float]) -> Array[Float] {
  let mut s : Float = 0.0
  for i in 0..<v.length() {
    s = s + v[i] * v[i]
  }
  let norm = Float::from_double(s.to_double().sqrt())
  if norm > 0.0 {
    let inv = (1.0 : Float) / norm
    let out = Array::make(v.length(), (0.0 : Float))
    for i in 0..<v.length() {
      out[i] = v[i] * inv
    }
    out
  } else {
    v
  }
}

///|
/// Build a flat index from a batch of vectors and their ids.
///
/// `keep_raw` retains f32 originals to enable exact reranking; dropping them
/// yields the most compact index (int8 only) but disables rerank.
pub fn FlatIndex::build(
  vectors : Array[Array[Float]],
  ids : Array[Int64],
  metric : Metric,
  keep_raw : Bool,
) -> FlatIndex {
  let count = vectors.length()
  if count == 0 {
    abort("FlatIndex::build: empty input")
  }
  if ids.length() != count {
    abort("FlatIndex::build: ids/vectors length mismatch")
  }
  let dim = vectors[0].length()
  let codes : FixedArray[Byte] = FixedArray::make(count * dim, 0)
  let scales = Array::make(count, (0.0 : Float))
  let sqnorms = Array::make(count, (0.0 : Float))
  let raw : FixedArray[Byte]? = if keep_raw {
    Some(FixedArray::make(count * dim * 4, 0))
  } else {
    None
  }
  let cos = is_cosine(metric)
  for i in 0..<count {
    if vectors[i].length() != dim {
      abort("FlatIndex::build: inconsistent dimension")
    }
    let processed = if cos { normalize(vectors[i]) } else { vectors[i] }
    let q = quantize(processed)
    for j in 0..<dim {
      codes[i * dim + j] = q.codes[j]
    }
    scales[i] = q.scale
    sqnorms[i] = q.sqnorm
    match raw {
      Some(rb) =>
        for j in 0..<dim {
          write_f32(rb, (i * dim + j) * 4, processed[j])
        }
      None => ()
    }
  }
  { dim, metric, count, ids, codes, scales, sqnorms, raw }
}

///|
pub fn FlatIndex::length(self : FlatIndex) -> Int {
  self.count
}

///|
pub fn FlatIndex::has_raw(self : FlatIndex) -> Bool {
  match self.raw {
    Some(_) => true
    None => false
  }
}

///|
/// Search for the `k` nearest neighbors. When originals are kept and
/// `oversample > 1`, the int8 scan feeds `k * oversample` candidates into an
/// exact f32 rerank stage.
pub fn FlatIndex::search(
  self : FlatIndex,
  query : Array[Float],
  k : Int,
  oversample : Int,
) -> Array[Hit] {
  if k <= 0 || self.count == 0 {
    return []
  }
  if query.length() != self.dim {
    abort("FlatIndex::search: dimension mismatch")
  }
  let dim = self.dim
  let count = self.count
  let l2 = is_l2(self.metric)
  let processed = if is_cosine(self.metric) { normalize(query) } else { query }
  let q = quantize(processed)
  let qcodes = q.codes
  let qscale = q.scale
  let qsq = q.sqnorm

  // Phase 1: approximate int8 scan into a bounded top-k.
  let cand_n = match self.raw {
    Some(_) =>
      if oversample > 1 {
        min(k * oversample, count)
      } else {
        min(k, count)
      }
    None => min(k, count)
  }
  let top = TopK::new(cand_n)
  let codes = self.codes
  let scales = self.scales
  let sqnorms = self.sqnorms
  for i in 0..<count {
    let dp = dot_i8(codes, i * dim, qcodes, 0, dim)
    let approx_dot = Float::from_double(dp.to_double()) * scales[i] * qscale
    let key = if l2 { sqnorms[i] + qsq - 2.0 * approx_dot } else { -approx_dot }
    top.push(key, i)
  }

  // Phase 2: rerank a widened candidate set with exact f32 distances.
  match self.raw {
    Some(rb) =>
      if oversample > 1 {
        let qbytes = f32_array_to_bytes(processed)
        let rr = TopK::new(k)
        for c in 0..<top.size {
          let i = top.idxs[c]
          rr.push(exact_key(self.metric, rb, i, dim, qbytes), i)
        }
        self.emit(rr.drain_sorted(), k)
      } else {
        self.emit(top.drain_sorted(), k)
      }
    None => self.emit(top.drain_sorted(), k)
  }
}

///|
/// Exact full-precision search (requires kept originals).
pub fn FlatIndex::search_exact(
  self : FlatIndex,
  query : Array[Float],
  k : Int,
) -> Array[Hit] {
  let rb = match self.raw {
    Some(rb) => rb
    None => abort("search_exact requires keep_raw = true")
  }
  if k <= 0 || self.count == 0 {
    return []
  }
  let processed = if is_cosine(self.metric) { normalize(query) } else { query }
  let qbytes = f32_array_to_bytes(processed)
  let top = TopK::new(min(k, self.count))
  for i in 0..<self.count {
    top.push(exact_key(self.metric, rb, i, self.dim, qbytes), i)
  }
  self.emit(top.drain_sorted(), k)
}

///|
/// Exact ranking key from stored f32 originals (lower = better).
fn exact_key(
  metric : Metric,
  raw : FixedArray[Byte],
  i : Int,
  dim : Int,
  qbytes : FixedArray[Byte],
) -> Float {
  let off = i * dim * 4
  if is_l2(metric) {
    l2sq_f32(raw, off, qbytes, 0, dim)
  } else {
    -dot_f32(raw, off, qbytes, 0, dim)
  }
}

///|
/// Take the first `k` sorted candidates as hits, converting the internal
/// ranking key back into the reported metric score.
fn FlatIndex::emit(self : FlatIndex, sorted : Array[Cand], k : Int) -> Array[Hit] {
  let hib = higher_is_better(self.metric)
  let n = min(k, sorted.length())
  let out : Array[Hit] = []
  for c in 0..<n {
    let cand = sorted[c]
    let score = if hib { -cand.key } else { cand.key }
    out.push({ id: self.ids[cand.idx], score })
  }
  out
}

///|
fn cmp_float(a : Float, b : Float) -> Int {
  if a < b {
    -1
  } else if a > b {
    1
  } else {
    0
  }
}

///|
fn min(a : Int, b : Int) -> Int {
  if a < b {
    a
  } else {
    b
  }
}

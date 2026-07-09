// Binary (1-bit) quantization for MoonBit, mirroring the Rust `bin_quant.rs`.
// Each dimension becomes its sign bit, packed into UInt64 words (32x smaller
// than f32). Candidates are ranked by Hamming distance (popcount of XOR) and
// reranked with exact f32 when the originals are kept.

///|
pub struct BinaryIndex {
  dim : Int
  metric : Metric
  words : Int
  count : Int
  ids : Array[Int64]
  bits : Array[UInt64] // count * words
  raw : FixedArray[Byte]? // count * dim * 4 (f32 bytes) when kept
}

///|
fn encode_bits(
  v : Array[Float],
  out : Array[UInt64],
  base : Int,
  words : Int,
) -> Unit {
  for w in 0..<words {
    out[base + w] = 0UL
  }
  for j in 0..<v.length() {
    if v[j] >= 0.0 {
      let w = base + j / 64
      out[w] = out[w] | (1UL << (j % 64))
    }
  }
}

///|
/// Build a binary index. `keep_raw` keeps f32 originals for exact rerank.
pub fn BinaryIndex::build(
  vectors : Array[Array[Float]],
  ids : Array[Int64],
  metric : Metric,
  keep_raw : Bool,
) -> BinaryIndex {
  let n = vectors.length()
  if n == 0 {
    abort("BinaryIndex::build: empty input")
  }
  if ids.length() != n {
    abort("BinaryIndex::build: ids/vectors length mismatch")
  }
  let dim = vectors[0].length()
  let words = (dim + 63) / 64
  let cos = is_cosine(metric)
  let bits = Array::make(n * words, 0UL)
  let raw : FixedArray[Byte]? = if keep_raw {
    Some(FixedArray::make(n * dim * 4, 0))
  } else {
    None
  }
  for i in 0..<n {
    if vectors[i].length() != dim {
      abort("BinaryIndex::build: inconsistent dimension")
    }
    let processed = if cos { normalize(vectors[i]) } else { vectors[i] }
    encode_bits(processed, bits, i * words, words)
    match raw {
      Some(rb) =>
        for j in 0..<dim {
          write_f32(rb, (i * dim + j) * 4, processed[j])
        }
      None => ()
    }
  }
  { dim, metric, words, count: n, ids, bits, raw }
}

///|
pub fn BinaryIndex::length(self : BinaryIndex) -> Int {
  self.count
}

///|
pub fn BinaryIndex::has_raw(self : BinaryIndex) -> Bool {
  match self.raw {
    Some(_) => true
    None => false
  }
}

///|
/// Bytes used by the packed codes (excludes ids / kept raw vectors).
pub fn BinaryIndex::code_bytes(self : BinaryIndex) -> Int {
  self.bits.length() * 8
}

///|
fn hamming(a : Array[UInt64], a_off : Int, b : Array[UInt64], words : Int) -> Int {
  let mut d = 0
  for w in 0..<words {
    d = d + (a[a_off + w] ^ b[w]).popcnt()
  }
  d
}

///|
/// Search for the k nearest neighbors by Hamming distance, then rerank
/// `k * oversample` candidates with exact f32 distances when originals kept.
pub fn BinaryIndex::search(
  self : BinaryIndex,
  query : Array[Float],
  k : Int,
  oversample : Int,
) -> Array[Hit] {
  if k <= 0 || self.count == 0 {
    return []
  }
  if query.length() != self.dim {
    abort("BinaryIndex::search: dimension mismatch")
  }
  let processed = if is_cosine(self.metric) { normalize(query) } else { query }
  let qbits = Array::make(self.words, 0UL)
  encode_bits(processed, qbits, 0, self.words)
  let cand_n = match self.raw {
    Some(_) => if oversample > 1 { k * oversample } else { k }
    None => k
  }
  let top = TopK::new(cand_n)
  for i in 0..<self.count {
    let h = hamming(self.bits, i * self.words, qbits, self.words)
    top.push(Float::from_double(h.to_double()), i)
  }
  match self.raw {
    Some(rb) =>
      if oversample > 1 {
        let qbytes = f32_array_to_bytes(processed)
        let rr = TopK::new(k)
        let cand = top.drain_sorted()
        let l2 = is_l2(self.metric)
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

///|
fn bin_finish(
  sorted : Array[Cand],
  k : Int,
  ids : Array[Int64],
  metric : Metric,
  reranked : Bool,
) -> Array[Hit] {
  let hib = higher_is_better(metric)
  let n = min(k, sorted.length())
  let out : Array[Hit] = []
  for c in 0..<n {
    let cand = sorted[c]
    let score = if reranked && hib { -cand.key } else { cand.key }
    out.push({ id: ids[cand.idx], score })
  }
  out
}

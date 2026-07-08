// Whitebox tests (same package) for the SIMD kernels, quantizer, and index.

///|
fn approx_eq(a : Float, b : Float, tol : Float) -> Bool {
  let d = a - b
  (if d < 0.0 { -d } else { d }) <= tol
}

///|
test "dot_i8 matches scalar with negatives" {
  let n = 20
  let a : FixedArray[Byte] = FixedArray::make(n, 0)
  let b : FixedArray[Byte] = FixedArray::make(n, 0)
  for i in 0..<n {
    a[i] = to_i8_byte(i - 10) // -10..=9
    b[i] = to_i8_byte(3)
  }
  let mut want = 0
  for i in 0..<n {
    want = want + (i - 10) * 3
  }
  assert_eq(dot_i8(a, 0, b, 0, n), want)
}

///|
test "f32 dot and l2 via simd bytes" {
  let xa : Array[Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
  let xb : Array[Float] = [2.0, 0.5, 1.0, 2.0, -1.0, 0.0, 3.0]
  let a = f32_array_to_bytes(xa)
  let b = f32_array_to_bytes(xb)
  let mut wdot : Float = 0.0
  let mut wl2 : Float = 0.0
  for i in 0..<xa.length() {
    wdot = wdot + xa[i] * xb[i]
    let d = xa[i] - xb[i]
    wl2 = wl2 + d * d
  }
  assert_true(approx_eq(dot_f32(a, 0, b, 0, xa.length()), wdot, 1.0e-3))
  assert_true(approx_eq(l2sq_f32(a, 0, b, 0, xa.length()), wl2, 1.0e-3))
}

///|
test "quantize roundtrip is close" {
  let x : Array[Float] = Array::make(64, (0.0 : Float))
  for i in 0..<64 {
    x[i] = (Float::from_double(i.to_double()) - 32.0) * 0.1
  }
  let q = quantize(x)
  let back = dequantize(q.codes, q.scale)
  for i in 0..<64 {
    assert_true(approx_eq(x[i], back[i], q.scale + 1.0e-6))
  }
}

///|
test "cosine search finds nearest" {
  let vectors : Array[Array[Float]] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.9, 0.1, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
  ]
  let ids : Array[Int64] = [10L, 20L, 30L, 40L]
  let idx = FlatIndex::build(vectors, ids, Cosine, true)
  let hits = idx.search([1.0, 0.0, 0.0, 0.0], 2, 4)
  assert_eq(hits[0].id, 10L)
  assert_eq(hits[1].id, 30L)
  assert_true(hits[0].score > hits[1].score)
}

///|
test "l2 search finds nearest" {
  let vectors : Array[Array[Float]] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.9, 0.1, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
  ]
  let ids : Array[Int64] = [10L, 20L, 30L, 40L]
  let idx = FlatIndex::build(vectors, ids, L2, true)
  let hits = idx.search([0.9, 0.1, 0.0, 0.0], 1, 4)
  assert_eq(hits[0].id, 30L)
}

///|
test "compact mode without rerank" {
  let vectors : Array[Array[Float]] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.9, 0.1, 0.0, 0.0],
  ]
  let ids : Array[Int64] = [1L, 2L, 3L]
  let idx = FlatIndex::build(vectors, ids, Dot, false)
  assert_false(idx.has_raw())
  let hits = idx.search([1.0, 0.0, 0.0, 0.0], 2, 4)
  assert_eq(hits[0].id, 1L)
}

///|
test "rerank matches exact on small set" {
  let vectors : Array[Array[Float]] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.9, 0.1, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
  ]
  let ids : Array[Int64] = [10L, 20L, 30L, 40L]
  let idx = FlatIndex::build(vectors, ids, L2, true)
  let approx = idx.search([0.5, 0.5, 0.0, 0.0], 4, 4)
  let exact = idx.search_exact([0.5, 0.5, 0.0, 0.0], 4)
  assert_eq(approx.length(), exact.length())
  for i in 0..<approx.length() {
    assert_eq(approx[i].id, exact[i].id)
  }
}

///|
/// Tiny deterministic xorshift RNG (no deps) for the recall test.
priv struct Rng {
  mut state : UInt64
}

///|
fn Rng::next(self : Rng) -> UInt64 {
  let mut x = self.state
  x = x ^ (x << 13)
  x = x ^ (x >> 7)
  x = x ^ (x << 17)
  self.state = x
  x
}

///|
fn Rng::unit(self : Rng) -> Float {
  // 24 random bits mapped to [0,1)
  let r = (self.next() >> 40).to_int()
  Float::from_double(r.to_double() / 16777216.0)
}

///|
test "int8+rerank recall vs exact on random data" {
  let n = 2000
  let dim = 32
  let rng = Rng::{ state: 0x9E3779B97F4A7C15UL }
  let vectors : Array[Array[Float]] = []
  let ids : Array[Int64] = []
  for i in 0..<n {
    let v = Array::make(dim, (0.0 : Float))
    for j in 0..<dim {
      v[j] = rng.unit() * 2.0 - 1.0
    }
    vectors.push(v)
    ids.push(i.to_int64())
  }
  let idx = FlatIndex::build(vectors, ids, Cosine, true)
  let k = 10
  let mut hit = 0
  let mut total = 0
  for _q in 0..<30 {
    let query = Array::make(dim, (0.0 : Float))
    for j in 0..<dim {
      query[j] = rng.unit() * 2.0 - 1.0
    }
    let approx = idx.search(query, k, 8)
    let exact = idx.search_exact(query, k)
    for a in approx {
      for e in exact {
        if a.id == e.id {
          hit = hit + 1
          break
        }
      }
    }
    total = total + exact.length()
  }
  // int8 + rerank should recover essentially all exact neighbors.
  let recall = hit.to_double() / total.to_double()
  assert_true(recall >= 0.95)
}

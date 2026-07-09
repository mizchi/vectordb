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

///|
test "vecdb bytes round-trip (with raw)" {
  let vectors : Array[Array[Float]] = [
    [1.0, 2.0, 3.0],
    [-1.0, 0.5, 4.0],
    [0.0, 0.0, 1.0],
  ]
  let ids : Array[Int64] = [1L, 2L, 3L]
  let idx = FlatIndex::build(vectors, ids, L2, true)
  let bytes = idx.to_bytes()
  let back = FlatIndex::from_bytes(bytes)
  assert_eq(back.length(), 3)
  assert_true(back.has_raw())
  let a = idx.search([1.0, 2.0, 3.0], 1, 4)
  let b = back.search([1.0, 2.0, 3.0], 1, 4)
  assert_eq(a[0].id, b[0].id)
  assert_eq(b[0].id, 1L)
}

///|
test "vecdb bytes round-trip (compact, no raw)" {
  let vectors : Array[Array[Float]] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
  ]
  let ids : Array[Int64] = [7L, 8L]
  let idx = FlatIndex::build(vectors, ids, Cosine, false)
  let back = FlatIndex::from_bytes(idx.to_bytes())
  assert_false(back.has_raw())
  let hits = back.search([0.9, 0.1, 0.0, 0.0], 1, 1)
  assert_eq(hits[0].id, 7L)
}

///|
fn make_clusters() -> (Array[Array[Float]], Array[Int64]) {
  let vectors : Array[Array[Float]] = []
  let ids : Array[Int64] = []
  let centers : Array[Array[Float]] = [
    [0.0, 0.0],
    [10.0, 0.0],
    [0.0, 10.0],
    [10.0, 10.0],
  ]
  let mut id = 0L
  for ci in 0..<4 {
    for j in 0..<25 {
      let jit = Float::from_double(j.to_double()) * 0.01
      vectors.push([centers[ci][0] + jit, centers[ci][1] - jit])
      ids.push(id)
      id = id + 1L
    }
  }
  (vectors, ids)
}

///|
test "ivf finds local cluster (l2)" {
  let (vectors, ids) = make_clusters()
  let idx = IvfIndex::build(vectors, ids, L2, 4, true, 15)
  assert_eq(idx.length(), 100)
  let hits = idx.search([10.0, 10.0], 3, 2, 4)
  assert_eq(hits.length(), 3)
  for h in hits {
    assert_true(h.id >= 75L) // 4th cluster is ids 75..99
  }
}

///|
test "ivf matches flat with full nprobe" {
  let (vectors, ids) = make_clusters()
  let ivf = IvfIndex::build(vectors, ids, L2, 4, true, 20)
  let flat = FlatIndex::build(vectors, ids, L2, true)
  let q : Array[Float] = [7.0, 9.0]
  let a = ivf.search(q, 1, 4, 8) // nprobe = nlist => scans all
  let b = flat.search(q, 1, 8)
  assert_eq(a[0].id, b[0].id)
}

///|
test "ivf bytes round-trip" {
  let (vectors, ids) = make_clusters()
  let idx = IvfIndex::build(vectors, ids, L2, 4, true, 15)
  let bytes = idx.to_bytes()
  let back = IvfIndex::from_bytes(bytes)
  assert_eq(back.length(), idx.length())
  assert_eq(back.nlist(), idx.nlist())
  assert_true(back.has_raw())
  let q : Array[Float] = [10.0, 10.0]
  let a = idx.search(q, 3, 2, 4)
  let b = back.search(q, 3, 2, 4)
  assert_eq(a.length(), b.length())
  for i in 0..<a.length() {
    assert_eq(a[i].id, b[i].id)
  }
}

///|
test "hnsw high recall vs exact (l2)" {
  let n = 1500
  let dim = 24
  let rng = Rng::{ state: 0x1234_5678_9ABC_DEF0UL }
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
  let hnsw = HnswIndex::new(dim, L2, 16, 200)
  for i in 0..<n {
    hnsw.add(ids[i], vectors[i])
  }
  let flat = FlatIndex::build(vectors, ids, L2, true)
  let k = 10
  let mut hit = 0
  let mut total = 0
  for t in 0..<40 {
    let q = vectors[t * 13 % n]
    let got = hnsw.search(q, k, 64)
    let truth = flat.search_exact(q, k)
    for g in got {
      for e in truth {
        if g.id == e.id {
          hit = hit + 1
          break
        }
      }
    }
    total = total + truth.length()
  }
  let recall = hit.to_double() / total.to_double()
  assert_true(recall >= 0.95)
}

///|
test "hnsw cosine finds self" {
  // Distinct random directions so each vector's nearest (by cosine) is itself.
  let n = 500
  let dim = 16
  let rng = Rng::{ state: 0xF00DBABE_12345678UL }
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
  let hnsw = HnswIndex::new(dim, Cosine, 16, 100)
  for i in 0..<n {
    hnsw.add(ids[i], vectors[i])
  }
  for t in 0..<20 {
    let i = t * 23 % n
    let hits = hnsw.search(vectors[i], 1, 32)
    assert_eq(hits[0].id, ids[i])
  }
}

///|
test "hnsw bytes round-trip" {
  let n = 400
  let dim = 12
  let rng = Rng::{ state: 0xCAFED00D_00C0FFEEUL }
  let hnsw = HnswIndex::new(dim, L2, 16, 100)
  let queries : Array[Array[Float]] = []
  for i in 0..<n {
    let v = Array::make(dim, (0.0 : Float))
    for j in 0..<dim {
      v[j] = rng.unit() * 2.0 - 1.0
    }
    hnsw.add(i.to_int64(), v)
    if i < 15 {
      queries.push(v)
    }
  }
  let back = HnswIndex::from_bytes(hnsw.to_bytes())
  assert_eq(back.length(), hnsw.length())
  for q in queries {
    let a = hnsw.search(q, 10, 64)
    let b = back.search(q, 10, 64)
    assert_eq(a.length(), b.length())
    for i in 0..<a.length() {
      assert_eq(a[i].id, b[i].id)
    }
  }
}

///|
test "binary 1-bit + rerank recall vs exact" {
  let dim = 64
  let ncenters = 20
  let rng = Rng::{ state: 0x00BADC0DE_0BADF00DUL }
  let centers : Array[Array[Float]] = []
  for _c in 0..<ncenters {
    let v = Array::make(dim, (0.0 : Float))
    for j in 0..<dim {
      v[j] = rng.unit() * 2.0 - 1.0
    }
    centers.push(v)
  }
  let vectors : Array[Array[Float]] = []
  let ids : Array[Int64] = []
  for i in 0..<2000 {
    let c = centers[i % ncenters]
    let v = Array::make(dim, (0.0 : Float))
    for j in 0..<dim {
      v[j] = c[j] + (rng.unit() - 0.5) * 0.2
    }
    vectors.push(v)
    ids.push(i.to_int64())
  }
  let bin = BinaryIndex::build(vectors, ids, Cosine, true)
  assert_true(bin.code_bytes() == 2000 * 1 * 8) // 64 dims -> 1 word (8 bytes)
  let flat = FlatIndex::build(vectors, ids, Cosine, true)
  let mut hit = 0
  let mut total = 0
  for t in 0..<30 {
    let c = centers[t % ncenters]
    let q = Array::make(dim, (0.0 : Float))
    for j in 0..<dim {
      q[j] = c[j] + 0.01
    }
    let got = bin.search(q, 10, 16)
    let truth = flat.search_exact(q, 10)
    for g in got {
      for e in truth {
        if g.id == e.id {
          hit = hit + 1
          break
        }
      }
    }
    total = total + truth.length()
  }
  let recall = hit.to_double() / total.to_double()
  assert_true(recall >= 0.90)
}

///|
test "rabitq rerank high recall + fast scan works" {
  let dim = 64
  let ncenters = 20
  let rng = Rng::{ state: 0xBEEF_1234_5678_9AB0UL }
  let centers : Array[Array[Float]] = []
  for _c in 0..<ncenters {
    let v = Array::make(dim, (0.0 : Float))
    for j in 0..<dim {
      v[j] = rng.unit() * 2.0 - 1.0
    }
    centers.push(v)
  }
  let vectors : Array[Array[Float]] = []
  let ids : Array[Int64] = []
  for i in 0..<2000 {
    let c = centers[i % ncenters]
    let v = Array::make(dim, (0.0 : Float))
    for j in 0..<dim {
      v[j] = c[j] + (rng.unit() - 0.5) * 0.2
    }
    vectors.push(v)
    ids.push(i.to_int64())
  }
  let rq = RaBitqIndex::build(vectors, ids, Cosine, true, 1UL)
  let flat = FlatIndex::build(vectors, ids, Cosine, true)
  let mut hit = 0
  let mut total = 0
  for t in 0..<30 {
    let c = centers[t % ncenters]
    let q = Array::make(dim, (0.0 : Float))
    for j in 0..<dim {
      q[j] = c[j] + 0.01
    }
    let got = rq.search(q, 10, 32)
    let truth = flat.search_exact(q, 10)
    for g in got {
      for e in truth {
        if g.id == e.id {
          hit = hit + 1
          break
        }
      }
    }
    total = total + truth.length()
  }
  let recall = hit.to_double() / total.to_double()
  assert_true(recall >= 0.90)
}

///|
test "rabitq l2 finds self" {
  let dim = 32
  let rng = Rng::{ state: 0x0102030405060708UL }
  let vectors : Array[Array[Float]] = []
  let ids : Array[Int64] = []
  for i in 0..<300 {
    let v = Array::make(dim, (0.0 : Float))
    for j in 0..<dim {
      v[j] = rng.unit() * 2.0 - 1.0
    }
    vectors.push(v)
    ids.push(i.to_int64())
  }
  let rq = RaBitqIndex::build(vectors, ids, L2, true, 3UL)
  let hits = rq.search(vectors[5], 1, 16)
  assert_eq(hits[0].id, 5L)
}

///|
test "flat filtered search restricts to matching ids" {
  let (vectors, ids) = make_clusters()
  let idx = FlatIndex::build(vectors, ids, L2, true)
  // Only even ids are eligible.
  let filter = fn(id : Int64) -> Bool { id % 2L == 0L }
  let hits = idx.search_filter([10.0, 10.0], 5, 4, filter)
  assert_eq(hits.length(), 5)
  for h in hits {
    assert_true(h.id % 2L == 0L)
    assert_true(h.id >= 75L) // still the nearest (4th) cluster
  }
}

///|
test "ivf filtered search restricts to matching ids" {
  let (vectors, ids) = make_clusters()
  let idx = IvfIndex::build(vectors, ids, L2, 4, true, 20)
  let filter = fn(id : Int64) -> Bool { id % 3L == 0L }
  // nprobe = nlist so every cell is scanned; result must respect the filter.
  let hits = idx.search_filter([10.0, 10.0], 4, 4, 8, filter)
  assert_eq(hits.length(), 4)
  for h in hits {
    assert_true(h.id % 3L == 0L)
  }
}

///|
test "hnsw filtered search matches flat ground truth" {
  let n = 1500
  let dim = 24
  let rng = Rng::{ state: 0x0F1E2D3C4B5A6978UL }
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
  let hnsw = HnswIndex::new(dim, L2, 16, 200)
  for i in 0..<n {
    hnsw.add(ids[i], vectors[i])
  }
  let flat = FlatIndex::build(vectors, ids, L2, true)
  let filter = fn(id : Int64) -> Bool { id % 3L == 0L }
  let k = 10
  let mut hit = 0
  let mut total = 0
  for t in 0..<30 {
    let q = vectors[t * 17 % n]
    let got = hnsw.search_filter(q, k, 128, filter)
    // Every result respects the predicate.
    for g in got {
      assert_true(g.id % 3L == 0L)
    }
    let truth = flat.search_filter(q, k, 8, filter)
    for g in got {
      for e in truth {
        if g.id == e.id {
          hit = hit + 1
          break
        }
      }
    }
    total = total + truth.length()
  }
  let recall = hit.to_double() / total.to_double()
  assert_true(recall >= 0.90)
}

///|
test "ivf+rabitq high recall with rerank" {
  let dim = 64
  let ncenters = 20
  let rng = Rng::{ state: 0xABCDEF01_23456789UL }
  let centers : Array[Array[Float]] = []
  for _c in 0..<ncenters {
    let v = Array::make(dim, (0.0 : Float))
    for j in 0..<dim {
      v[j] = rng.unit() * 2.0 - 1.0
    }
    centers.push(v)
  }
  let vectors : Array[Array[Float]] = []
  let ids : Array[Int64] = []
  for i in 0..<3000 {
    let c = centers[i % ncenters]
    let v = Array::make(dim, (0.0 : Float))
    for j in 0..<dim {
      v[j] = c[j] + (rng.unit() - 0.5) * 0.2
    }
    vectors.push(v)
    ids.push(i.to_int64())
  }
  let idx = IvfRabitqIndex::build(vectors, ids, Cosine, 40, true, 12, 1UL)
  assert_eq(idx.length(), 3000)
  let flat = FlatIndex::build(vectors, ids, Cosine, true)
  let mut hit = 0
  let mut total = 0
  for t in 0..<30 {
    let c = centers[t % ncenters]
    let q = Array::make(dim, (0.0 : Float))
    for j in 0..<dim {
      q[j] = c[j] + 0.01
    }
    let got = idx.search(q, 10, 8, 32)
    let truth = flat.search_exact(q, 10)
    for g in got {
      for e in truth {
        if g.id == e.id {
          hit = hit + 1
          break
        }
      }
    }
    total = total + truth.length()
  }
  let recall = hit.to_double() / total.to_double()
  assert_true(recall >= 0.90)
}

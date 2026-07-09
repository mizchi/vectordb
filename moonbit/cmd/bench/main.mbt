///|
/// Integrated benchmark across every MoonBit index/quantizer.
/// Clustered data (recall is meaningful), n x 128 cosine, k=10.
/// Run with `moon run cmd/bench --target native --release` (or --target wasm).
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
  let r = (self.next() >> 40).to_int()
  Float::from_double(r.to_double() / 16777216.0)
}

///|
fn rand_vec(rng : Rng, dim : Int) -> Array[Float] {
  let v = Array::make(dim, (0.0 : Float))
  for j in 0..<dim {
    v[j] = rng.unit() * 2.0 - 1.0
  }
  v
}

///|
/// recall@k of `got` vs the exact-truth hit list.
fn recall_of(
  got : Array[Array[@vectordb.Hit]],
  truth : Array[Array[@vectordb.Hit]],
  nq : Int,
) -> Double {
  let mut hit = 0
  let mut total = 0
  for qi in 0..<nq {
    for a in got[qi] {
      for e in truth[qi] {
        if a.id == e.id {
          hit = hit + 1
          break
        }
      }
    }
    total = total + truth[qi].length()
  }
  hit.to_double() / total.to_double()
}

///|
fn main {
  let n = 20000
  let dim = 128
  let ncenters = 200
  let nq = 100
  let k = 10
  let rng = Rng::{ state: 0x00C0FFEE00C0FFEEUL }

  println("generating \{n} x \{dim} clustered vectors (\{ncenters} centers)...")
  let centers : Array[Array[Float]] = []
  for _c in 0..<ncenters {
    centers.push(rand_vec(rng, dim))
  }
  let vectors : Array[Array[Float]] = []
  let ids : Array[Int64] = []
  for i in 0..<n {
    let c = centers[i % ncenters]
    let v = Array::make(dim, (0.0 : Float))
    for j in 0..<dim {
      v[j] = c[j] + (rng.unit() - 0.5) * 0.3
    }
    vectors.push(v)
    ids.push(i.to_int64())
  }
  let queries : Array[Array[Float]] = []
  for i in 0..<nq {
    let c = centers[i % ncenters]
    let v = Array::make(dim, (0.0 : Float))
    for j in 0..<dim {
      v[j] = c[j] + (rng.unit() - 0.5) * 0.3
    }
    queries.push(v)
  }
  let per = fn(us : Double) -> Double { us / 1000.0 / nq.to_double() }

  // Flat exact f32 = ground truth.
  let flat = @vectordb.FlatIndex::build(vectors, ids, @vectordb.Cosine, true)
  let exact : Array[Array[@vectordb.Hit]] = []
  let t_ex = @perf.monotonic_clock_start()
  for q in queries {
    exact.push(flat.search_exact(q, k))
  }
  let us_exact = @perf.monotonic_clock_end(t_ex)
  println("=== \{n}x\{dim} cosine, k=\{k}, \{nq} queries ===")
  println("flat exact f32        : \{per(us_exact)} ms/q  recall=1.0000")

  // Flat int8 + rerank.
  let got : Array[Array[@vectordb.Hit]] = []
  let t0 = @perf.monotonic_clock_start()
  for q in queries {
    got.push(flat.search(q, k, 8))
  }
  let us0 = @perf.monotonic_clock_end(t0)
  println(
    "flat int8+rerank(o=8) : \{per(us0)} ms/q  recall=\{recall_of(got, exact, nq)}",
  )

  // IVF.
  let ivf = @vectordb.IvfIndex::build(
    vectors,
    ids,
    @vectordb.Cosine,
    0,
    true,
    12,
  )
  let g_ivf : Array[Array[@vectordb.Hit]] = []
  let t_ivf = @perf.monotonic_clock_start()
  for q in queries {
    g_ivf.push(ivf.search(q, k, 16, 8))
  }
  let us_ivf = @perf.monotonic_clock_end(t_ivf)
  println(
    "IVF nprobe=16 (o=8)   : \{per(us_ivf)} ms/q  recall=\{recall_of(g_ivf, exact, nq)}",
  )

  // PQ (m=16 -> 16 bytes/vector = 32x).
  let pq = @vectordb.PqIndex::build(
    vectors,
    ids,
    @vectordb.Cosine,
    16,
    256,
    15,
    true,
  )
  let g_pq : Array[Array[@vectordb.Hit]] = []
  let t_pq = @perf.monotonic_clock_start()
  for q in queries {
    g_pq.push(pq.search(q, k, 16))
  }
  let us_pq = @perf.monotonic_clock_end(t_pq)
  println(
    "PQ m=16 (o=16)        : \{per(us_pq)} ms/q  recall=\{recall_of(g_pq, exact, nq)}",
  )

  // OPQ.
  let opq = @vectordb.OpqIndex::build(
    vectors,
    ids,
    @vectordb.Cosine,
    16,
    256,
    15,
    3,
    true,
  )
  let g_opq : Array[Array[@vectordb.Hit]] = []
  let t_opq = @perf.monotonic_clock_start()
  for q in queries {
    g_opq.push(opq.search(q, k, 16))
  }
  let us_opq = @perf.monotonic_clock_end(t_opq)
  println(
    "OPQ m=16 (o=16)       : \{per(us_opq)} ms/q  recall=\{recall_of(g_opq, exact, nq)}",
  )

  // IVF+PQ.
  let ipq = @vectordb.IvfPqIndex::build(
    vectors,
    ids,
    @vectordb.Cosine,
    0,
    16,
    256,
    12,
    true,
  )
  let g_ipq : Array[Array[@vectordb.Hit]] = []
  let t_ipq = @perf.monotonic_clock_start()
  for q in queries {
    g_ipq.push(ipq.search(q, k, 16, 16))
  }
  let us_ipq = @perf.monotonic_clock_end(t_ipq)
  println(
    "IVF+PQ np=16 (o=16)   : \{per(us_ipq)} ms/q  recall=\{recall_of(g_ipq, exact, nq)}",
  )

  // HNSW.
  let hnsw = @vectordb.HnswIndex::new(dim, @vectordb.Cosine, 16, 200)
  for i in 0..<n {
    hnsw.add(ids[i], vectors[i])
  }
  let g_h : Array[Array[@vectordb.Hit]] = []
  let t_h = @perf.monotonic_clock_start()
  for q in queries {
    g_h.push(hnsw.search(q, k, 64))
  }
  let us_h = @perf.monotonic_clock_end(t_h)
  println(
    "HNSW ef=64            : \{per(us_h)} ms/q  recall=\{recall_of(g_h, exact, nq)}",
  )

  // int8-graph HNSW.
  let hq = @vectordb.HnswQIndex::new(dim, @vectordb.Cosine, 16, 200, true)
  for i in 0..<n {
    hq.add(ids[i], vectors[i])
  }
  let g_hq : Array[Array[@vectordb.Hit]] = []
  let t_hq = @perf.monotonic_clock_start()
  for q in queries {
    g_hq.push(hq.search(q, k, 64))
  }
  let us_hq = @perf.monotonic_clock_end(t_hq)
  println(
    "HNSW-q(int8) ef=64    : \{per(us_hq)} ms/q  recall=\{recall_of(g_hq, exact, nq)}",
  )
}

///|
/// Rough benchmark mirroring the Rust `examples/bench.rs`:
/// 50k x 128 cosine vectors, 200 queries, k=10.
/// Run with `moon run cmd/bench --target native --release`.
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
fn main {
  let n = 50000
  let dim = 128
  let nq = 200
  let k = 10
  let oversample = 8
  let rng = Rng::{ state: 0x00C0FFEE00C0FFEEUL }

  println("generating \{n} x \{dim} vectors...")
  let vectors : Array[Array[Float]] = []
  let ids : Array[Int64] = []
  for i in 0..<n {
    vectors.push(rand_vec(rng, dim))
    ids.push(i.to_int64())
  }
  let idx = @vectordb.FlatIndex::build(vectors, ids, @vectordb.Cosine, true)
  let queries : Array[Array[Float]] = []
  for _i in 0..<nq {
    queries.push(rand_vec(rng, dim))
  }

  // int8 + rerank
  let approx : Array[Array[@vectordb.Hit]] = []
  let t0 = @perf.monotonic_clock_start()
  for q in queries {
    approx.push(idx.search(q, k, oversample))
  }
  let us_rerank = @perf.monotonic_clock_end(t0)

  // int8 only (oversample = 1 skips the rerank stage)
  let t1 = @perf.monotonic_clock_start()
  for q in queries {
    idx.search(q, k, 1) |> ignore
  }
  let us_int8 = @perf.monotonic_clock_end(t1)

  // exact f32
  let exact : Array[Array[@vectordb.Hit]] = []
  let t2 = @perf.monotonic_clock_start()
  for q in queries {
    exact.push(idx.search_exact(q, k))
  }
  let us_exact = @perf.monotonic_clock_end(t2)

  // recall@k of int8+rerank vs exact
  let mut hit = 0
  let mut total = 0
  for qi in 0..<nq {
    for a in approx[qi] {
      for e in exact[qi] {
        if a.id == e.id {
          hit = hit + 1
          break
        }
      }
    }
    total = total + exact[qi].length()
  }
  let recall = hit.to_double() / total.to_double()
  let per = fn(us : Double) -> Double { us / 1000.0 / nq.to_double() }

  println("int8+rerank(o=\{oversample}): \{per(us_rerank)} ms/query")
  println("int8 only:                  \{per(us_int8)} ms/query")
  println("exact f32:                  \{per(us_exact)} ms/query")
  println("recall@\{k}: \{recall}")
}

// HNSW (Hierarchical Navigable Small World) graph index for MoonBit.
// Mirrors the Rust `hnsw.rs`: multi-layer graph, greedy descent, ef-width beam
// search at layer 0, diversity neighbor-selection heuristic. Reuses the
// package's scalar distance helpers, normalize, and bounded TopK (as the
// result beam `w`).

///|
/// Binary min-heap over (distance, node), used for the candidate frontier.
priv struct MinHeap {
  keys : Array[Float]
  nodes : Array[Int]
}

///|
fn MinHeap::new() -> MinHeap {
  { keys: [], nodes: [] }
}

///|
fn MinHeap::len(self : MinHeap) -> Int {
  self.keys.length()
}

///|
fn MinHeap::swap(self : MinHeap, a : Int, b : Int) -> Unit {
  let tk = self.keys[a]
  self.keys[a] = self.keys[b]
  self.keys[b] = tk
  let tn = self.nodes[a]
  self.nodes[a] = self.nodes[b]
  self.nodes[b] = tn
}

///|
fn MinHeap::push(self : MinHeap, key : Float, node : Int) -> Unit {
  self.keys.push(key)
  self.nodes.push(node)
  let mut i = self.keys.length() - 1
  while i > 0 {
    let p = (i - 1) / 2
    if self.keys[i] < self.keys[p] {
      self.swap(i, p)
      i = p
    } else {
      break
    }
  }
}

///|
fn MinHeap::pop_min(self : MinHeap) -> (Float, Int) {
  let n = self.keys.length()
  let rk = self.keys[0]
  let rn = self.nodes[0]
  let lk = self.keys[n - 1]
  let ln = self.nodes[n - 1]
  self.keys.pop() |> ignore
  self.nodes.pop() |> ignore
  if self.keys.length() > 0 {
    self.keys[0] = lk
    self.nodes[0] = ln
    let mut i = 0
    let m = self.keys.length()
    while true {
      let l = 2 * i + 1
      let r = 2 * i + 2
      let mut smallest = i
      if l < m && self.keys[l] < self.keys[smallest] {
        smallest = l
      }
      if r < m && self.keys[r] < self.keys[smallest] {
        smallest = r
      }
      if smallest == i {
        break
      }
      self.swap(i, smallest)
      i = smallest
    }
  }
  (rk, rn)
}

///|
pub struct HnswIndex {
  dim : Int
  metric : Metric
  m : Int
  m0 : Int
  ef_construction : Int
  ml : Double
  mut count : Int
  vectors : Array[Array[Float]]
  ids : Array[Int64]
  links : Array[Array[Array[Int]]] // [node][level] -> neighbors
  levels : Array[Int]
  mut entry : Int // -1 when empty
  mut max_level : Int
  mut rng : UInt64
}

///|
/// Create an empty index. `m` ~ 16, `ef_construction` ~ 200.
pub fn HnswIndex::new(
  dim : Int,
  metric : Metric,
  m : Int,
  ef_construction : Int,
) -> HnswIndex {
  {
    dim,
    metric,
    m,
    m0: m * 2,
    ef_construction: if ef_construction < m { m } else { ef_construction },
    ml: 1.0 / @math.ln(m.to_double()),
    count: 0,
    vectors: [],
    ids: [],
    links: [],
    levels: [],
    entry: -1,
    max_level: 0,
    rng: 0x9E3779B97F4A7C15UL,
  }
}

///|
pub fn HnswIndex::length(self : HnswIndex) -> Int {
  self.count
}

///|
fn HnswIndex::d(self : HnswIndex, a : Array[Float], b : Array[Float]) -> Float {
  if is_l2(self.metric) {
    l2_arr(a, b)
  } else {
    -dot_arr(a, b)
  }
}

///|
fn HnswIndex::next_rand(self : HnswIndex) -> Double {
  let mut x = self.rng
  x = x ^ (x << 13)
  x = x ^ (x >> 7)
  x = x ^ (x << 17)
  self.rng = x
  (x >> 11).reinterpret_as_int64().to_double() / 9007199254740992.0
}

///|
fn HnswIndex::random_level(self : HnswIndex) -> Int {
  let mut r = self.next_rand()
  if r < 1.0e-12 {
    r = 1.0e-12
  }
  (-@math.ln(r) * self.ml).to_int()
}

///|
fn HnswIndex::greedy1(
  self : HnswIndex,
  q : Array[Float],
  ep0 : Int,
  ep_dist0 : Float,
  lc : Int,
) -> (Int, Float) {
  let mut ep = ep0
  let mut ep_dist = ep_dist0
  let mut improved = true
  while improved {
    improved = false
    for nb in self.links[ep][lc] {
      let dd = self.d(self.vectors[nb], q)
      if dd < ep_dist {
        ep_dist = dd
        ep = nb
        improved = true
      }
    }
  }
  (ep, ep_dist)
}

///|
/// Beam search within one layer; returns the ef nearest as (dist, node) pairs.
fn HnswIndex::search_layer(
  self : HnswIndex,
  q : Array[Float],
  ep : Int,
  ef : Int,
  lc : Int,
) -> Array[(Float, Int)] {
  let visited = Array::make(self.count, false)
  let cands = MinHeap::new()
  let w = TopK::new(ef)
  let d0 = self.d(self.vectors[ep], q)
  visited[ep] = true
  cands.push(d0, ep)
  w.push(d0, ep)
  while cands.len() > 0 {
    let (cd, cn) = cands.pop_min()
    let farthest = if w.size >= ef { w.keys[0] } else { 3.0e38 }
    if cd > farthest && w.size >= ef {
      break
    }
    for nb in self.links[cn][lc] {
      if visited[nb] {
        continue
      }
      visited[nb] = true
      let dd = self.d(self.vectors[nb], q)
      let far = if w.size >= ef { w.keys[0] } else { 3.0e38 }
      if dd < far || w.size < ef {
        cands.push(dd, nb)
        w.push(dd, nb) // TopK keeps the ef smallest automatically
      }
    }
  }
  let out : Array[(Float, Int)] = []
  for i in 0..<w.size {
    out.push((w.keys[i], w.idxs[i]))
  }
  out
}

///|
/// Diversity neighbor-selection heuristic: keep candidates closer to `q` than
/// to any already-kept neighbor.
fn HnswIndex::select_neighbors(
  self : HnswIndex,
  w : Array[(Float, Int)],
  m : Int,
) -> Array[Int] {
  w.sort_by(fn(a, b) { cmp_float(a.0, b.0) })
  let result : Array[Int] = []
  for i in 0..<w.length() {
    if result.length() >= m {
      break
    }
    let (dq, e) = w[i]
    let mut keep = true
    for r in result {
      if self.d(self.vectors[e], self.vectors[r]) < dq {
        keep = false
        break
      }
    }
    if keep {
      result.push(e)
    }
  }
  result
}

///|
fn HnswIndex::prune(self : HnswIndex, node : Int, lc : Int, budget : Int) -> Unit {
  let q = self.vectors[node]
  let w : Array[(Float, Int)] = []
  for nb in self.links[node][lc] {
    w.push((self.d(self.vectors[nb], q), nb))
  }
  self.links[node][lc] = self.select_neighbors(w, budget)
}

///|
/// Insert a vector with an external id.
pub fn HnswIndex::add(self : HnswIndex, id : Int64, vector : Array[Float]) -> Unit {
  if vector.length() != self.dim {
    abort("HnswIndex::add: dimension mismatch")
  }
  let processed = if is_cosine(self.metric) {
    normalize(vector)
  } else {
    copy_row(vector)
  }
  let node = self.count
  let level = self.random_level()
  self.vectors.push(processed)
  self.ids.push(id)
  self.levels.push(level)
  let nl : Array[Array[Int]] = []
  for _l in 0..<(level + 1) {
    nl.push([])
  }
  self.links.push(nl)
  self.count = self.count + 1
  if self.entry < 0 {
    self.entry = node
    self.max_level = level
    return
  }
  let mut ep = self.entry
  let mut cur = self.d(self.vectors[ep], processed)
  let top = self.max_level
  let mut lc = top
  while lc > level {
    let (nep, ncur) = self.greedy1(processed, ep, cur, lc)
    ep = nep
    cur = ncur
    lc = lc - 1
  }
  let start = if level < top { level } else { top }
  let mut l2c = start
  while l2c >= 0 {
    let w = self.search_layer(processed, ep, self.ef_construction, l2c)
    // nearest candidate becomes the entry point for the next layer down
    let mut best = ep
    let mut best_d : Float = 3.0e38
    for pair in w {
      if pair.0 < best_d {
        best_d = pair.0
        best = pair.1
      }
    }
    ep = best
    let m = if l2c == 0 { self.m0 } else { self.m }
    let selected = self.select_neighbors(w, m)
    for nb in selected {
      self.links[node][l2c].push(nb)
      self.links[nb][l2c].push(node)
      let budget = if l2c == 0 { self.m0 } else { self.m }
      if self.links[nb][l2c].length() > budget {
        self.prune(nb, l2c, budget)
      }
    }
    l2c = l2c - 1
  }
  if level > self.max_level {
    self.max_level = level
    self.entry = node
  }
}

///|
/// Search for the k nearest neighbors with beam width `ef_search`.
pub fn HnswIndex::search(
  self : HnswIndex,
  query : Array[Float],
  k : Int,
  ef_search : Int,
) -> Array[Hit] {
  if k <= 0 || self.count == 0 {
    return []
  }
  if query.length() != self.dim {
    abort("HnswIndex::search: dimension mismatch")
  }
  let processed = if is_cosine(self.metric) { normalize(query) } else { query }
  let mut ep = self.entry
  let mut cur = self.d(self.vectors[ep], processed)
  let mut lc = self.max_level
  while lc >= 1 {
    let (nep, ncur) = self.greedy1(processed, ep, cur, lc)
    ep = nep
    cur = ncur
    lc = lc - 1
  }
  let ef = if ef_search > k { ef_search } else { k }
  let w = self.search_layer(processed, ep, ef, 0)
  w.sort_by(fn(a, b) { cmp_float(a.0, b.0) })
  let hib = higher_is_better(self.metric)
  let out : Array[Hit] = []
  let n = if k < w.length() { k } else { w.length() }
  for i in 0..<n {
    let (dist, node) = w[i]
    let score = if hib { -dist } else { dist }
    out.push({ id: self.ids[node], score })
  }
  out
}

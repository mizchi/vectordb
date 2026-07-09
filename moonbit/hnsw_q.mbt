// Memory-compact int8-quantized HNSW for MoonBit, mirroring the Rust
// `hnsw_q.rs`. Same multi-layer graph as `HnswIndex`, but each node stores int8
// codes + a per-vector scale/sqnorm (~1/4 the vector memory). Graph build and
// search both run in the quantized space via the int8 approximate dot product;
// keeping the f32 originals enables an exact rerank of the final beam. Reuses
// MinHeap (hnsw.mbt), TopK, quantize, dot_i8, and the storage helpers.

///|
pub struct HnswQIndex {
  dim : Int
  metric : Metric
  m : Int
  m0 : Int
  ef_construction : Int
  ml : Double
  mut count : Int
  codes : Array[FixedArray[Byte]] // per node, length dim
  scales : Array[Float]
  sqnorms : Array[Float]
  raw : Array[Array[Float]]? // per node processed f32, when kept
  ids : Array[Int64]
  links : Array[Array[Array[Int]]] // [node][level] -> neighbors
  levels : Array[Int]
  mut entry : Int // -1 when empty
  mut max_level : Int
  mut rng : UInt64
}

///|
/// Create an empty index. `keep_raw` retains f32 originals for exact rerank.
pub fn HnswQIndex::new(
  dim : Int,
  metric : Metric,
  m : Int,
  ef_construction : Int,
  keep_raw : Bool,
) -> HnswQIndex {
  {
    dim,
    metric,
    m,
    m0: m * 2,
    ef_construction: if ef_construction < m {
      m
    } else {
      ef_construction
    },
    ml: 1.0 / @math.ln(m.to_double()),
    count: 0,
    codes: [],
    scales: [],
    sqnorms: [],
    raw: if keep_raw {
      Some([])
    } else {
      None
    },
    ids: [],
    links: [],
    levels: [],
    entry: -1,
    max_level: 0,
    rng: 0x9E3779B97F4A7C15UL,
  }
}

///|
pub fn HnswQIndex::length(self : HnswQIndex) -> Int {
  self.count
}

///|
pub fn HnswQIndex::has_raw(self : HnswQIndex) -> Bool {
  match self.raw {
    Some(_) => true
    None => false
  }
}

///|
fn HnswQIndex::next_rand(self : HnswQIndex) -> Double {
  let mut x = self.rng
  x = x ^ (x << 13)
  x = x ^ (x >> 7)
  x = x ^ (x << 17)
  self.rng = x
  (x >> 11).reinterpret_as_int64().to_double() / 9007199254740992.0
}

///|
fn HnswQIndex::random_level(self : HnswQIndex) -> Int {
  let mut r = self.next_rand()
  if r < 1.0e-12 {
    r = 1.0e-12
  }
  (-@math.ln(r) * self.ml).to_int()
}

///|
/// Approximate distance between two stored nodes (smaller = closer).
fn HnswQIndex::dn(self : HnswQIndex, a : Int, b : Int) -> Float {
  let dp = dot_i8(self.codes[a], 0, self.codes[b], 0, self.dim)
  let approx = Float::from_double(dp.to_double()) *
    self.scales[a] *
    self.scales[b]
  if is_l2(self.metric) {
    self.sqnorms[a] + self.sqnorms[b] - 2.0 * approx
  } else {
    -approx
  }
}

///|
/// Approximate distance from a quantized query to a stored node.
fn HnswQIndex::dq(
  self : HnswQIndex,
  qc : FixedArray[Byte],
  qs : Float,
  qsq : Float,
  node : Int,
) -> Float {
  let dp = dot_i8(qc, 0, self.codes[node], 0, self.dim)
  let approx = Float::from_double(dp.to_double()) * qs * self.scales[node]
  if is_l2(self.metric) {
    qsq + self.sqnorms[node] - 2.0 * approx
  } else {
    -approx
  }
}

///|
/// Insert a vector with an external id.
pub fn HnswQIndex::add(
  self : HnswQIndex,
  id : Int64,
  vector : Array[Float],
) -> Unit {
  if vector.length() != self.dim {
    abort("HnswQIndex::add: dimension mismatch")
  }
  let processed = if is_cosine(self.metric) {
    normalize(vector)
  } else {
    copy_row(vector)
  }
  let q = quantize(processed)
  let node = self.count
  let level = self.random_level()
  self.codes.push(q.codes)
  self.scales.push(q.scale)
  self.sqnorms.push(q.sqnorm)
  match self.raw {
    Some(rb) => rb.push(processed)
    None => ()
  }
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
  let mut cur = self.dn(node, ep)
  let top = self.max_level
  let mut lc = top
  while lc > level {
    let (nep, ncur) = self.greedy1_node(node, ep, cur, lc)
    ep = nep
    cur = ncur
    lc = lc - 1
  }
  let start = if level < top { level } else { top }
  let mut l2c = start
  while l2c >= 0 {
    let w = self.search_layer_node(node, ep, self.ef_construction, l2c)
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
fn HnswQIndex::greedy1_node(
  self : HnswQIndex,
  qnode : Int,
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
      let dd = self.dn(qnode, nb)
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
fn HnswQIndex::greedy1_query(
  self : HnswQIndex,
  qc : FixedArray[Byte],
  qs : Float,
  qsq : Float,
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
      let dd = self.dq(qc, qs, qsq, nb)
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
/// Beam search within one layer from a stored query node (used during build).
fn HnswQIndex::search_layer_node(
  self : HnswQIndex,
  qnode : Int,
  ep : Int,
  ef : Int,
  lc : Int,
) -> Array[(Float, Int)] {
  let visited = Array::make(self.count, false)
  let cands = MinHeap::new()
  let w = TopK::new(ef)
  let d0 = self.dn(qnode, ep)
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
      let dd = self.dn(qnode, nb)
      let far = if w.size >= ef { w.keys[0] } else { 3.0e38 }
      if dd < far || w.size < ef {
        cands.push(dd, nb)
        w.push(dd, nb)
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
/// Beam search within one layer from an external quantized query. `filter`
/// gates admission into the result beam; non-passing nodes are still traversed.
fn HnswQIndex::search_layer_query(
  self : HnswQIndex,
  qc : FixedArray[Byte],
  qs : Float,
  qsq : Float,
  ep : Int,
  ef : Int,
  lc : Int,
  filter : (Int64) -> Bool,
) -> Array[(Float, Int)] {
  let visited = Array::make(self.count, false)
  let cands = MinHeap::new()
  let w = TopK::new(ef)
  let d0 = self.dq(qc, qs, qsq, ep)
  visited[ep] = true
  cands.push(d0, ep)
  if filter(self.ids[ep]) {
    w.push(d0, ep)
  }
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
      let dd = self.dq(qc, qs, qsq, nb)
      let far = if w.size >= ef { w.keys[0] } else { 3.0e38 }
      if dd < far || w.size < ef {
        cands.push(dd, nb)
        if filter(self.ids[nb]) {
          w.push(dd, nb)
        }
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
fn HnswQIndex::select_neighbors(
  self : HnswQIndex,
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
      if self.dn(e, r) < dq {
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
fn HnswQIndex::prune(
  self : HnswQIndex,
  node : Int,
  lc : Int,
  budget : Int,
) -> Unit {
  let w : Array[(Float, Int)] = []
  for nb in self.links[node][lc] {
    w.push((self.dn(node, nb), nb))
  }
  self.links[node][lc] = self.select_neighbors(w, budget)
}

///|
/// Search for the k nearest neighbors with beam width `ef_search`; the beam is
/// reranked with exact f32 distances when originals are kept.
pub fn HnswQIndex::search(
  self : HnswQIndex,
  query : Array[Float],
  k : Int,
  ef_search : Int,
) -> Array[Hit] {
  self.search_filter(query, k, ef_search, fn(_id) { true })
}

///|
/// Filtered search: only ids satisfying `filter` become results (the graph is
/// still traversed through non-passing nodes).
pub fn HnswQIndex::search_filter(
  self : HnswQIndex,
  query : Array[Float],
  k : Int,
  ef_search : Int,
  filter : (Int64) -> Bool,
) -> Array[Hit] {
  if k <= 0 || self.count == 0 {
    return []
  }
  if query.length() != self.dim {
    abort("HnswQIndex::search_filter: dimension mismatch")
  }
  let processed = if is_cosine(self.metric) { normalize(query) } else { query }
  let q = quantize(processed)
  let mut ep = self.entry
  let mut cur = self.dq(q.codes, q.scale, q.sqnorm, ep)
  let mut lc = self.max_level
  while lc >= 1 {
    let (nep, ncur) = self.greedy1_query(
      q.codes,
      q.scale,
      q.sqnorm,
      ep,
      cur,
      lc,
    )
    ep = nep
    cur = ncur
    lc = lc - 1
  }
  let ef = if ef_search > k { ef_search } else { k }
  let w = self.search_layer_query(q.codes, q.scale, q.sqnorm, ep, ef, 0, filter)
  let hib = higher_is_better(self.metric)
  // Rerank the beam with exact f32 distances when originals are kept.
  let scored : Array[(Float, Int)] = match self.raw {
    Some(rb) => {
      let out : Array[(Float, Int)] = []
      for pair in w {
        let node = pair.1
        let d = if is_l2(self.metric) {
          l2_arr(rb[node], processed)
        } else {
          -dot_arr(rb[node], processed)
        }
        out.push((d, node))
      }
      out
    }
    None => w
  }
  scored.sort_by(fn(a, b) { cmp_float(a.0, b.0) })
  let out : Array[Hit] = []
  let n = if k < scored.length() { k } else { scored.length() }
  for i in 0..<n {
    let (dist, node) = scored[i]
    let score = if hib { -dist } else { dist }
    out.push({ id: self.ids[node], score })
  }
  out
}

// ---------------------------------------------------------------------------
// Persistence: int8-HNSW `.vecdb` (magic "VECDBHQ1"), same layout as Rust.
// Header (64B) + codes i8 + scales f32 + sqnorms f32 + ids u64 + levels u32
// + variable-length links + raw f32? (each section 16B-aligned).
// ---------------------------------------------------------------------------

///|
/// Serialize the graph to an int8-HNSW `.vecdb` byte buffer.
pub fn HnswQIndex::to_bytes(self : HnswQIndex) -> FixedArray[Byte] {
  let dim = self.dim
  let count = self.count
  let has_raw = self.has_raw()
  let mut links_bytes = 0
  for i in 0..<count {
    for lc in 0..<(self.levels[i] + 1) {
      links_bytes = links_bytes + 4 + self.links[i][lc].length() * 4
    }
  }
  let codes_off = align16(64)
  let scales_off = align16(codes_off + count * dim)
  let sqnorms_off = align16(scales_off + count * 4)
  let ids_off = align16(sqnorms_off + count * 4)
  let lvl_off = align16(ids_off + count * 8)
  let links_off = align16(lvl_off + count * 4)
  let raw_off = align16(links_off + links_bytes)
  let total = if has_raw {
    align16(raw_off + count * dim * 4)
  } else {
    align16(links_off + links_bytes)
  }
  let b : FixedArray[Byte] = FixedArray::make(total, 0)
  let magic : FixedArray[Byte] = [
    0x56, 0x45, 0x43, 0x44, 0x42, 0x48, 0x51, 0x31,
  ]
  for i in 0..<8 {
    b[i] = magic[i]
  }
  write_u32_le(b, 8, 1)
  write_u32_le(b, 12, metric_to_u32(self.metric))
  write_u32_le(b, 16, dim.reinterpret_as_uint())
  write_u32_le(b, 20, count.reinterpret_as_uint())
  write_u32_le(b, 24, if has_raw { 1 } else { 0 })
  write_u32_le(b, 28, self.m.reinterpret_as_uint())
  write_u32_le(b, 32, self.ef_construction.reinterpret_as_uint())
  write_u32_le(b, 36, self.max_level.reinterpret_as_uint())
  let entry_store = if self.entry < 0 { 0 } else { self.entry }
  write_u32_le(b, 40, entry_store.reinterpret_as_uint())
  write_u64_le(b, 44, self.rng)
  for i in 0..<count {
    for d in 0..<dim {
      b[codes_off + i * dim + d] = self.codes[i][d]
    }
    write_f32(b, scales_off + i * 4, self.scales[i])
    write_f32(b, sqnorms_off + i * 4, self.sqnorms[i])
    write_u64_le(b, ids_off + i * 8, self.ids[i].reinterpret_as_uint64())
    write_u32_le(b, lvl_off + i * 4, self.levels[i].reinterpret_as_uint())
  }
  let mut p = links_off
  for i in 0..<count {
    for lc in 0..<(self.levels[i] + 1) {
      let nbrs = self.links[i][lc]
      write_u32_le(b, p, nbrs.length().reinterpret_as_uint())
      p = p + 4
      for nb in nbrs {
        write_u32_le(b, p, nb.reinterpret_as_uint())
        p = p + 4
      }
    }
  }
  match self.raw {
    Some(rb) =>
      for i in 0..<count {
        for d in 0..<dim {
          write_f32(b, raw_off + (i * dim + d) * 4, rb[i][d])
        }
      }
    None => ()
  }
  b
}

///|
/// Parse an int8-HNSW `.vecdb` byte buffer into an [`HnswQIndex`].
pub fn HnswQIndex::from_bytes(b : FixedArray[Byte]) -> HnswQIndex {
  if b.length() < 64 {
    abort("HnswQIndex::from_bytes: buffer smaller than header")
  }
  // magic "VECDBHQ1"
  if b[0] != 0x56 || b[5] != 0x48 || b[6] != 0x51 || b[7] != 0x31 {
    abort("HnswQIndex::from_bytes: bad magic")
  }
  if read_u32_le(b, 8) != 1 {
    abort("HnswQIndex::from_bytes: unsupported version")
  }
  let metric = metric_from_u32(read_u32_le(b, 12))
  let dim = read_u32_le(b, 16).reinterpret_as_int()
  let count = read_u32_le(b, 20).reinterpret_as_int()
  let has_raw = read_u32_le(b, 24) != 0
  let m = read_u32_le(b, 28).reinterpret_as_int()
  let ef_construction = read_u32_le(b, 32).reinterpret_as_int()
  let max_level = read_u32_le(b, 36).reinterpret_as_int()
  let entry_raw = read_u32_le(b, 40).reinterpret_as_int()
  let rng = read_u64_le(b, 44)
  let codes_off = align16(64)
  let scales_off = align16(codes_off + count * dim)
  let sqnorms_off = align16(scales_off + count * 4)
  let ids_off = align16(sqnorms_off + count * 4)
  let lvl_off = align16(ids_off + count * 8)
  let links_off = align16(lvl_off + count * 4)
  let codes : Array[FixedArray[Byte]] = Array::make(
    count,
    FixedArray::make(0, 0),
  )
  let scales = Array::make(count, (0.0 : Float))
  let sqnorms = Array::make(count, (0.0 : Float))
  let ids = Array::make(count, (0 : Int64))
  let levels = Array::make(count, 0)
  for i in 0..<count {
    let row : FixedArray[Byte] = FixedArray::make(dim, 0)
    for d in 0..<dim {
      row[d] = b[codes_off + i * dim + d]
    }
    codes[i] = row
    scales[i] = read_f32(b, scales_off + i * 4)
    sqnorms[i] = read_f32(b, sqnorms_off + i * 4)
    ids[i] = read_u64_le(b, ids_off + i * 8).reinterpret_as_int64()
    levels[i] = read_u32_le(b, lvl_off + i * 4).reinterpret_as_int()
  }
  let links : Array[Array[Array[Int]]] = []
  let mut p = links_off
  for i in 0..<count {
    let node_links : Array[Array[Int]] = []
    for _lc in 0..<(levels[i] + 1) {
      let len = read_u32_le(b, p).reinterpret_as_int()
      p = p + 4
      let lst : Array[Int] = []
      for _j in 0..<len {
        lst.push(read_u32_le(b, p).reinterpret_as_int())
        p = p + 4
      }
      node_links.push(lst)
    }
    links.push(node_links)
  }
  let raw : Array[Array[Float]]? = if has_raw {
    let raw_off = align16(p)
    let rows : Array[Array[Float]] = Array::make(count, [])
    for i in 0..<count {
      let row = Array::make(dim, (0.0 : Float))
      for d in 0..<dim {
        row[d] = read_f32(b, raw_off + (i * dim + d) * 4)
      }
      rows[i] = row
    }
    Some(rows)
  } else {
    None
  }
  {
    dim,
    metric,
    m,
    m0: m * 2,
    ef_construction,
    ml: 1.0 / @math.ln(m.to_double()),
    count,
    codes,
    scales,
    sqnorms,
    raw,
    ids,
    links,
    levels,
    entry: if count > 0 {
      entry_raw
    } else {
      -1
    },
    max_level,
    rng,
  }
}

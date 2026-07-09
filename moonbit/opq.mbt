// Optimized Product Quantization for MoonBit, mirroring the Rust `opq.rs`
// (Ge et al., CVPR 2013). Learn an orthonormal rotation R that balances the
// subspaces, then PQ on R x. R is trained by the non-parametric alternating
// optimization (PQ-reconstruct, then orthogonal-Procrustes update), with the
// required SVD computed via a self-contained cyclic Jacobi eigensolver. Reuses
// the shared PQ helpers (pq.mbt), kmeans, TopK/bin_finish, storage helpers.
// `.vecdb` layout "VECDBOP1".

///|
pub struct OpqIndex {
  dim : Int
  metric : Metric
  m : Int
  dsub : Int
  ksub : Int
  count : Int
  rot : Array[Float] // dim*dim orthonormal (row-major); y = R x
  pq_codebooks : Array[Array[Float]] // m*ksub rows of dsub (rotated space)
  codes : FixedArray[Byte] // count * m
  ids : Array[Int64]
  raw : FixedArray[Byte]? // count * dim * 4 (processed, un-rotated) when kept
}

///|
/// Apply the row-major `dim*dim` rotation `r` to `x`: `out = R x`.
fn opq_rotate(r : Array[Float], x : Array[Float], dim : Int) -> Array[Float] {
  let out = Array::make(dim, (0.0 : Float))
  for i in 0..<dim {
    let base = i * dim
    let mut s : Float = 0.0
    for j in 0..<dim {
      s = s + r[base + j] * x[j]
    }
    out[i] = s
  }
  out
}

///|
/// Build an OPQ index. `opq_iters` alternating rotation/PQ refinement passes.
pub fn OpqIndex::build(
  vectors : Array[Array[Float]],
  ids : Array[Int64],
  metric : Metric,
  m : Int,
  ksub : Int,
  kmeans_iters : Int,
  opq_iters : Int,
  keep_raw : Bool,
) -> OpqIndex {
  let n = vectors.length()
  if n == 0 {
    abort("OpqIndex::build: empty input")
  }
  if ids.length() != n {
    abort("OpqIndex::build: ids/vectors length mismatch")
  }
  let dim = vectors[0].length()
  if m <= 0 || dim % m != 0 {
    abort("OpqIndex::build: m must divide dim")
  }
  if ksub <= 0 || ksub > 256 {
    abort("OpqIndex::build: ksub must be in 1..=256")
  }
  let dsub = dim / m
  let ks = if ksub > n { n } else { ksub }
  let cos = is_cosine(metric)
  let data : Array[Array[Float]] = []
  for i in 0..<n {
    if vectors[i].length() != dim {
      abort("OpqIndex::build: inconsistent dimension")
    }
    data.push(if cos { normalize(vectors[i]) } else { copy_row(vectors[i]) })
  }

  // Learn the rotation, then train the final codebooks on rotated data.
  let rot = learn_rotation(data, n, dim, m, ks, dsub, kmeans_iters, opq_iters)
  let rotated : Array[Array[Float]] = Array::make(n, [])
  for i in 0..<n {
    rotated[i] = opq_rotate(rot, data[i], dim)
  }
  let pq_codebooks = pq_train(rotated, m, ks, dsub, kmeans_iters)
  let codes : FixedArray[Byte] = FixedArray::make(n * m, 0)
  for i in 0..<n {
    let c = pq_encode(pq_codebooks, m, ks, dsub, rotated[i])
    for j in 0..<m {
      codes[i * m + j] = c[j]
    }
  }
  let raw : FixedArray[Byte]? = if keep_raw {
    Some(FixedArray::make(n * dim * 4, 0))
  } else {
    None
  }
  match raw {
    Some(rb) =>
      for i in 0..<n {
        for d in 0..<dim {
          write_f32(rb, (i * dim + d) * 4, data[i][d])
        }
      }
    None => ()
  }
  {
    dim,
    metric,
    m,
    dsub,
    ksub: ks,
    count: n,
    rot,
    pq_codebooks,
    codes,
    ids,
    raw,
  }
}

///|
pub fn OpqIndex::length(self : OpqIndex) -> Int {
  self.count
}

///|
pub fn OpqIndex::has_raw(self : OpqIndex) -> Bool {
  match self.raw {
    Some(_) => true
    None => false
  }
}

///|
pub fn OpqIndex::code_bytes(self : OpqIndex) -> Int {
  self.count * self.m
}

///|
/// Search for the k nearest neighbors (rotate the query, then ADC + rerank).
pub fn OpqIndex::search(
  self : OpqIndex,
  query : Array[Float],
  k : Int,
  oversample : Int,
) -> Array[Hit] {
  if k <= 0 || self.count == 0 {
    return []
  }
  if query.length() != self.dim {
    abort("OpqIndex::search: dimension mismatch")
  }
  let processed = if is_cosine(self.metric) { normalize(query) } else { query }
  let rq = opq_rotate(self.rot, processed, self.dim)
  let l2 = is_l2(self.metric)
  let lut = pq_lut(self.pq_codebooks, self.m, self.ksub, self.dsub, rq, l2)
  let cand_n = match self.raw {
    Some(_) => if oversample > 1 { k * oversample } else { k }
    None => k
  }
  let top = TopK::new(cand_n)
  for i in 0..<self.count {
    let acc = pq_adc_sum(lut, self.ksub, self.codes, i * self.m, self.m)
    let key = if l2 { acc } else { -acc }
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

// ---------------------------------------------------------------------------
// Rotation learning: non-parametric alternating optimization.
// ---------------------------------------------------------------------------

///|
fn learn_rotation(
  data : Array[Array[Float]],
  n : Int,
  dim : Int,
  m : Int,
  ksub : Int,
  dsub : Int,
  kmeans_iters : Int,
  opq_iters : Int,
) -> Array[Float] {
  // R starts at identity.
  let rot = Array::make(dim * dim, (0.0 : Float))
  for i in 0..<dim {
    rot[i * dim + i] = 1.0
  }
  let mut cur = rot
  for _it in 0..<opq_iters {
    // (1) rotate data, train PQ, reconstruct in the rotated space.
    let rotated : Array[Array[Float]] = Array::make(n, [])
    for i in 0..<n {
      rotated[i] = opq_rotate(cur, data[i], dim)
    }
    let cb = pq_train(rotated, m, ksub, dsub, kmeans_iters)
    let recon : Array[Array[Float]] = Array::make(n, [])
    for i in 0..<n {
      let code = pq_encode(cb, m, ksub, dsub, rotated[i])
      let rr = Array::make(dim, (0.0 : Float))
      for j in 0..<m {
        let cen = cb[j * ksub + code[j].to_int()]
        for d in 0..<dsub {
          rr[j * dsub + d] = cen[d]
        }
      }
      recon[i] = rr
    }
    // (2) orthogonal Procrustes: C = dataᵀ·recon, R = V Uᵀ (C = U S Vᵀ).
    let c = gram_cross(data, recon, n, dim)
    cur = procrustes_rotation(c, dim)
  }
  cur
}

///|
/// `C = dataᵀ · recon` (dim*dim), inputs are n rows of dim.
fn gram_cross(
  data : Array[Array[Float]],
  recon : Array[Array[Float]],
  n : Int,
  dim : Int,
) -> Array[Float] {
  let c = Array::make(dim * dim, (0.0 : Float))
  for i in 0..<n {
    let a = data[i]
    let b = recon[i]
    for r in 0..<dim {
      let ar = a[r]
      let base = r * dim
      for d in 0..<dim {
        c[base + d] = c[base + d] + ar * b[d]
      }
    }
  }
  c
}

///|
/// Orthogonal Procrustes rotation for `C`: `R = V Uᵀ` where `C = U S Vᵀ`. `V`
/// is the eigenvector matrix of `CᵀC`; `U` is built by Gram-Schmidt on `C·vₖ`
/// (descending eigenvalue order), so `R` is always exactly orthonormal.
fn procrustes_rotation(c : Array[Float], dim : Int) -> Array[Float] {
  // A = Cᵀ C (symmetric).
  let a = Array::make(dim * dim, (0.0 : Float))
  for i in 0..<dim {
    for j in 0..<dim {
      let mut s : Float = 0.0
      for k in 0..<dim {
        s = s + c[k * dim + i] * c[k * dim + j]
      }
      a[i * dim + j] = s
    }
  }
  let (eval, v) = jacobi_eigen(a, dim)

  // Column order by descending eigenvalue.
  let order = Array::make(dim, 0)
  for i in 0..<dim {
    order[i] = i
  }
  order.sort_by(fn(i, j) {
    if eval[j] < eval[i] {
      -1
    } else if eval[j] > eval[i] {
      1
    } else {
      0
    }
  })

  // Build U columns via Gram-Schmidt on w_k = C v_k.
  let u = Array::make(dim * dim, (0.0 : Float)) // u[row*dim + k]
  let basis : Array[Array[Float]] = []
  for oi in 0..<dim {
    let k = order[oi]
    let w = Array::make(dim, (0.0 : Float))
    for row in 0..<dim {
      let mut s : Float = 0.0
      for t in 0..<dim {
        s = s + c[row * dim + t] * v[t * dim + k]
      }
      w[row] = s
    }
    gram_schmidt(w, basis, dim)
    let mut nrm : Float = 0.0
    for row in 0..<dim {
      nrm = nrm + w[row] * w[row]
    }
    let norm = nrm.sqrt()
    let col = if norm > 1.0e-6 {
      let inv = (1.0 : Float) / norm
      for row in 0..<dim {
        w[row] = w[row] * inv
      }
      w
    } else {
      orthonormal_complement(basis, dim)
    }
    for row in 0..<dim {
      u[row * dim + k] = col[row]
    }
    basis.push(col)
  }

  // R = V Uᵀ.
  let r = Array::make(dim * dim, (0.0 : Float))
  for i in 0..<dim {
    for j in 0..<dim {
      let mut s : Float = 0.0
      for k in 0..<dim {
        s = s + v[i * dim + k] * u[j * dim + k]
      }
      r[i * dim + j] = s
    }
  }
  r
}

///|
/// Modified Gram-Schmidt: subtract the projection of `w` onto each basis vector.
fn gram_schmidt(
  w : Array[Float],
  basis : Array[Array[Float]],
  dim : Int,
) -> Unit {
  for b in basis {
    let mut dp : Float = 0.0
    for i in 0..<dim {
      dp = dp + w[i] * b[i]
    }
    for i in 0..<dim {
      w[i] = w[i] - dp * b[i]
    }
  }
}

///|
/// A unit vector orthogonal to every vector in `basis`.
fn orthonormal_complement(
  basis : Array[Array[Float]],
  dim : Int,
) -> Array[Float] {
  for i in 0..<dim {
    let e = Array::make(dim, (0.0 : Float))
    e[i] = 1.0
    gram_schmidt(e, basis, dim)
    let mut nrm : Float = 0.0
    for r in 0..<dim {
      nrm = nrm + e[r] * e[r]
    }
    let norm = nrm.sqrt()
    if norm > 1.0e-6 {
      let inv = (1.0 : Float) / norm
      for r in 0..<dim {
        e[r] = e[r] * inv
      }
      return e
    }
  }
  let e = Array::make(dim, (0.0 : Float))
  e[0] = 1.0
  e
}

///|
/// Cyclic Jacobi eigensolver for a symmetric `dim*dim` matrix (row-major).
/// Returns `(eigenvalues, eigenvectors)` where eigenvector `k` is column `k`
/// (`v[i*dim + k]`). Works in Double for stability.
fn jacobi_eigen(a : Array[Float], dim : Int) -> (Array[Float], Array[Float]) {
  let m = Array::make(dim * dim, 0.0)
  for i in 0..<(dim * dim) {
    m[i] = a[i].to_double()
  }
  let v = Array::make(dim * dim, 0.0)
  for i in 0..<dim {
    v[i * dim + i] = 1.0
  }
  for _sweep in 0..<100 {
    let mut off = 0.0
    for p in 0..<dim {
      for q in (p + 1)..<dim {
        off = off + m[p * dim + q] * m[p * dim + q]
      }
    }
    if off < 1.0e-18 {
      break
    }
    for p in 0..<dim {
      for q in (p + 1)..<dim {
        let apq = m[p * dim + q]
        if apq.abs() < 1.0e-300 {
          continue
        }
        let app = m[p * dim + p]
        let aqq = m[q * dim + q]
        let theta = (aqq - app) / (2.0 * apq)
        let sign = if theta < 0.0 { -1.0 } else { 1.0 }
        let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt())
        let cph = 1.0 / (t * t + 1.0).sqrt()
        let sph = t * cph
        for i in 0..<dim {
          let mip = m[i * dim + p]
          let miq = m[i * dim + q]
          m[i * dim + p] = cph * mip - sph * miq
          m[i * dim + q] = sph * mip + cph * miq
        }
        for i in 0..<dim {
          let mpi = m[p * dim + i]
          let mqi = m[q * dim + i]
          m[p * dim + i] = cph * mpi - sph * mqi
          m[q * dim + i] = sph * mpi + cph * mqi
        }
        for i in 0..<dim {
          let vip = v[i * dim + p]
          let viq = v[i * dim + q]
          v[i * dim + p] = cph * vip - sph * viq
          v[i * dim + q] = sph * vip + cph * viq
        }
      }
    }
  }
  let eval = Array::make(dim, (0.0 : Float))
  for i in 0..<dim {
    eval[i] = Float::from_double(m[i * dim + i])
  }
  let vf = Array::make(dim * dim, (0.0 : Float))
  for i in 0..<(dim * dim) {
    vf[i] = Float::from_double(v[i])
  }
  (eval, vf)
}

// ---------------------------------------------------------------------------
// Persistence: OPQ `.vecdb` (magic "VECDBOP1"), same layout as Rust.
// Header (64B) + rot f32 + pq_codebooks f32 + ids u64 + codes u8 + raw f32?.
// ---------------------------------------------------------------------------

///|
priv struct OpLayout {
  rot : Int
  codebooks : Int
  ids : Int
  codes : Int
  raw : Int
  total : Int
}

///|
fn op_layout(
  dim : Int,
  count : Int,
  m : Int,
  ksub : Int,
  dsub : Int,
  has_raw : Bool,
) -> OpLayout {
  let rot = align16(64)
  let codebooks = align16(rot + dim * dim * 4)
  let ids = align16(codebooks + m * ksub * dsub * 4)
  let codes = align16(ids + count * 8)
  let raw = align16(codes + count * m)
  let total = if has_raw { align16(raw + count * dim * 4) } else { raw }
  { rot, codebooks, ids, codes, raw, total }
}

///|
/// Serialize the index to an OPQ `.vecdb` byte buffer.
pub fn OpqIndex::to_bytes(self : OpqIndex) -> FixedArray[Byte] {
  let dim = self.dim
  let count = self.count
  let has_raw = self.has_raw()
  let l = op_layout(dim, count, self.m, self.ksub, self.dsub, has_raw)
  let b : FixedArray[Byte] = FixedArray::make(l.total, 0)
  let magic : FixedArray[Byte] = [
    0x56, 0x45, 0x43, 0x44, 0x42, 0x4F, 0x50, 0x31,
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
  write_u32_le(b, 32, self.ksub.reinterpret_as_uint())
  for i in 0..<(dim * dim) {
    write_f32(b, l.rot + i * 4, self.rot[i])
  }
  for jc in 0..<(self.m * self.ksub) {
    let row = self.pq_codebooks[jc]
    for d in 0..<self.dsub {
      write_f32(b, l.codebooks + (jc * self.dsub + d) * 4, row[d])
    }
  }
  for i in 0..<count {
    write_u64_le(b, l.ids + i * 8, self.ids[i].reinterpret_as_uint64())
  }
  for j in 0..<(count * self.m) {
    b[l.codes + j] = self.codes[j]
  }
  match self.raw {
    Some(rb) =>
      for j in 0..<(count * dim * 4) {
        b[l.raw + j] = rb[j]
      }
    None => ()
  }
  b
}

///|
/// Parse an OPQ `.vecdb` byte buffer into an [`OpqIndex`].
pub fn OpqIndex::from_bytes(b : FixedArray[Byte]) -> OpqIndex {
  if b.length() < 64 {
    abort("OpqIndex::from_bytes: buffer smaller than header")
  }
  // magic "VECDBOP1"
  if b[0] != 0x56 || b[5] != 0x4F || b[6] != 0x50 || b[7] != 0x31 {
    abort("OpqIndex::from_bytes: bad magic")
  }
  if read_u32_le(b, 8) != 1 {
    abort("OpqIndex::from_bytes: unsupported version")
  }
  let metric = metric_from_u32(read_u32_le(b, 12))
  let dim = read_u32_le(b, 16).reinterpret_as_int()
  let count = read_u32_le(b, 20).reinterpret_as_int()
  let has_raw = (read_u32_le(b, 24) & 1) != 0
  let m = read_u32_le(b, 28).reinterpret_as_int()
  let ksub = read_u32_le(b, 32).reinterpret_as_int()
  if dim == 0 || m == 0 || dim % m != 0 {
    abort("OpqIndex::from_bytes: bad dim/m")
  }
  let dsub = dim / m
  let l = op_layout(dim, count, m, ksub, dsub, has_raw)
  if b.length() < l.total {
    abort("OpqIndex::from_bytes: buffer truncated")
  }
  let rot = Array::make(dim * dim, (0.0 : Float))
  for i in 0..<(dim * dim) {
    rot[i] = read_f32(b, l.rot + i * 4)
  }
  let pq_codebooks : Array[Array[Float]] = Array::make(m * ksub, [])
  for jc in 0..<(m * ksub) {
    let row = Array::make(dsub, (0.0 : Float))
    for d in 0..<dsub {
      row[d] = read_f32(b, l.codebooks + (jc * dsub + d) * 4)
    }
    pq_codebooks[jc] = row
  }
  let ids = Array::make(count, (0 : Int64))
  for i in 0..<count {
    ids[i] = read_u64_le(b, l.ids + i * 8).reinterpret_as_int64()
  }
  let codes : FixedArray[Byte] = FixedArray::make(count * m, 0)
  for j in 0..<(count * m) {
    codes[j] = b[l.codes + j]
  }
  let raw : FixedArray[Byte]? = if has_raw {
    let rb : FixedArray[Byte] = FixedArray::make(count * dim * 4, 0)
    for j in 0..<(count * dim * 4) {
      rb[j] = b[l.raw + j]
    }
    Some(rb)
  } else {
    None
  }
  { dim, metric, m, dsub, ksub, count, rot, pq_codebooks, codes, ids, raw }
}

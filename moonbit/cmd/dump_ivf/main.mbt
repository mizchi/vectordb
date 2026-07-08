///|
/// Dump IVF `.vecdb` bytes of a fixed tiny index (deterministic k-means:
/// nlist == count, 1 iteration) to diff against the Rust `dump_ivf` output.
fn main {
  let vectors : Array[Array[Float]] = [
    [1.0, 2.0, 3.0],
    [-1.0, 0.5, 4.0],
    [0.0, 0.0, 1.0],
  ]
  let ids : Array[Int64] = [1L, 2L, 3L]
  let idx = @vectordb.IvfIndex::build(vectors, ids, @vectordb.L2, 3, true, 1)
  let bytes = idx.to_bytes()
  let parts : Array[String] = []
  for i in 0..<bytes.length() {
    parts.push(bytes[i].to_int().to_string())
  }
  println(parts.join(" "))
}

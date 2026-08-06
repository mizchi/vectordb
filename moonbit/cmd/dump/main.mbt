///|
/// Dump the `.vecdb` bytes of a fixed tiny index as space-separated decimals,
/// so they can be diffed against the Rust `save` output (byte-compat check).
fn main {
  let vectors : Array[Array[Float]] = [
    [1.0, 2.0, 3.0],
    [-1.0, 0.5, 4.0],
    [0.0, 0.0, 1.0],
  ]
  let ids : Array[Int64] = [1L, 2L, 3L]
  let idx = @meandb.FlatIndex::build(vectors, ids, @meandb.L2, true)
  let bytes = idx.to_bytes()
  let parts : Array[String] = []
  for i in 0..<bytes.length() {
    parts.push(bytes[i].to_int().to_string())
  }
  println(parts.join(" "))
}

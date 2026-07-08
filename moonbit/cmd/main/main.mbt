///|
/// Demo: build a tiny cosine index and run a query.
/// Run with `moon run cmd/main --target native` from the moonbit/ directory.
fn main {
  let vectors : Array[Array[Float]] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.9, 0.1, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
  ]
  let ids : Array[Int64] = [10L, 20L, 30L, 40L]
  let idx = @vectordb.FlatIndex::build(vectors, ids, @vectordb.Cosine, true)
  println("built \{idx.length()} vectors, raw kept = \{idx.has_raw()}")
  let hits = idx.search([1.0, 0.05, 0.0, 0.0], 3, 4)
  println("query ~ [1, 0.05, 0, 0], top-3 (cosine, int8 scan + f32 rerank):")
  for h in hits {
    println("  id=\{h.id}  score=\{h.score}")
  }
}

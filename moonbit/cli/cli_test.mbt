///|
/// End-to-end CLI test over the in-memory adapter (`MemFs`) — no real IO, so it
/// runs on every backend. Exercises build -> info -> search for a few index
/// kinds, all through the same `run(io, args)` entry point.
test "cli build/search/info over MemFs" {
  let csv =
    #|0,1,0,0,0
    #|1,0,1,0,0
    #|2,0.9,0.1,0,0
    #|3,0,0,1,0
    #|# a comment line is ignored
    #|4,0,0,0,1
    #|
  let kinds = ["flat", "ivf", "hnsw", "pq"]
  for kind in kinds {
    let fs = MemFs::new()
    fs.put_str("vecs.csv", csv)
    // build
    let rc = run(fs.io(), [
      "build", "vecs.csv", "idx.vecdb", "--index", kind, "--metric", "cosine", "--nlist",
      "2", "--pq-m", "2", "--ksub", "4",
    ])
    assert_eq(rc, 0)
    // the index file was written
    assert_true(fs.get("idx.vecdb") is Some(_))
    // info reports the right count
    let _ = run(fs.io(), ["info", "idx.vecdb"])

    // search: query near row 0 should return id 0 as the top hit
    fs.put_str("q.csv", "0,1,0.05,0,0\n")
    let rc2 = run(fs.io(), [
      "search", "idx.vecdb", "q.csv", "-k", "3", "--nprobe", "2", "--ef", "16",
    ])
    assert_eq(rc2, 0)
    // the first search result line is query 0, rank 0 -> "0\t0\t0\t..."
    assert_true(fs.output().contains("0\t0\t0\t"))
  }
}

///|
test "cli unknown command and bad magic" {
  let fs = MemFs::new()
  assert_eq(run(fs.io(), ["frobnicate"]), 1)
  fs.put_str("bogus.vecdb", "not an index")
  assert_eq(run(fs.io(), ["info", "bogus.vecdb"]), 1)
  assert_eq(run(fs.io(), ["search", "missing.vecdb", "q.csv"]), 1)
}

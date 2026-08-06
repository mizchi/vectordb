///|
/// Runnable demo of the environment-agnostic meandb vector CLI.
///
/// MoonBit's argv/fs access is not portable across backends, so instead of
/// reading real files this wires the CLI core (`@cli.run`) to a `MemFs` adapter
/// for file IO and real `println` for output — the exact pattern a host embeds.
/// Any environment can supply its own three closures via `@cli.Io::new`.
///
///   moon run cmd/cli --target native
fn main {
  let fs = @cli.MemFs::new()
  fs.put_str(
    "vecs.csv",
    (
      #|0,1,0,0,0
      #|1,0,1,0,0
      #|2,0.9,0.1,0,0
      #|3,0,0,1,0
      #|4,0,0,0,1
    ),
  )
  // Files live in the in-memory store; output goes to the real console.
  let io = @cli.Io::new(fn(p) { fs.get(p) }, fn(p, b) { fs.put(p, b) }, fn(s) {
    println(s)
  })
  println("$ build vecs.csv idx.vecdb --metric cosine")
  @cli.run(io, ["build", "vecs.csv", "idx.vecdb", "--metric", "cosine"])
  |> ignore
  println("$ info idx.vecdb")
  @cli.run(io, ["info", "idx.vecdb"]) |> ignore
  fs.put_str("q.csv", "0,1,0.05,0,0\n")
  println("$ search idx.vecdb q.csv -k 3   (query near row 0)")
  @cli.run(io, ["search", "idx.vecdb", "q.csv", "-k", "3"]) |> ignore
}

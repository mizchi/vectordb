///|
/// A small, environment-agnostic CLI for the meandb vector layer. All IO goes
/// through the injected `Io` adapter, so the exact same command logic runs on
/// native, wasm, js, or inside a host with no filesystem (see `MemFs`).
///
/// Commands (args are provided by the host, e.g. from argv or a config):
///   build  <in.csv> <out.vecdb> [--index flat|ivf|hnsw|hnsw-q|pq|ivfpq|opq]
///                               [--metric l2|dot|cosine] [--compact]
///                               [--nlist N] [-m M] [--ef-construction N]
///                               [--pq-m M] [--ksub K]
///   search <index.vecdb> <query.csv> [-k N] [--oversample M] [--nprobe N] [--ef N]
///   info   <index.vecdb>
///
/// CSV: one row = `id,v0,v1,...` (`#` lines and blanks ignored). Index type is
/// auto-detected from the file magic on search / info.

///|
fn parse_float(s : String) -> Float {
  let d = @string.parse_double(s) catch { _ => 0.0 }
  Float::from_double(d)
}

///|
fn parse_i64(s : String) -> Int64 {
  @string.parse_int64(s) catch {
    _ => 0L
  }
}

///|
fn parse_int(s : String, default : Int) -> Int {
  let v = @string.parse_int64(s) catch { _ => default.to_int64() }
  v.to_int()
}

///|
/// Parse `id,v0,v1,...` rows from CSV bytes. Whitespace is ignored; `#` lines
/// and blank lines are skipped.
fn parse_csv(b : FixedArray[Byte]) -> (Array[Int64], Array[Array[Float]]) {
  let ids : Array[Int64] = []
  let vecs : Array[Array[Float]] = []
  let n = b.length()
  let mut i = 0
  while i < n {
    let fields : Array[String] = []
    let mut sb = StringBuilder::new()
    let mut any = false
    let mut firstc = 0
    while i < n && b[i] != 10 { // until '\n'
      let c = b[i]
      i = i + 1
      if c == 13 || c == 32 || c == 9 {
        continue // skip '\r', space, tab
      }
      if !any {
        firstc = c.to_int()
      }
      any = true
      if c == 44 { // ','
        fields.push(sb.to_string())
        sb = StringBuilder::new()
      } else {
        sb.write_char(c.to_int().to_char().unwrap())
      }
    }
    if i < n {
      i = i + 1 // skip '\n'
    }
    if any {
      fields.push(sb.to_string())
      let is_comment = firstc == 35 // '#'
      if !is_comment && fields.length() >= 2 {
        ids.push(parse_i64(fields[0]))
        let v : Array[Float] = []
        for j in 1..<fields.length() {
          v.push(parse_float(fields[j]))
        }
        vecs.push(v)
      }
    }
  }
  (ids, vecs)
}

///|
fn flag_value(args : Array[String], name : String) -> String? {
  for i in 0..<(args.length() - 1) {
    if args[i] == name {
      return Some(args[i + 1])
    }
  }
  None
}

///|
fn has_flag(args : Array[String], name : String) -> Bool {
  for a in args {
    if a == name {
      return true
    }
  }
  false
}

///|
fn int_flag(args : Array[String], name : String, default : Int) -> Int {
  match flag_value(args, name) {
    Some(s) => parse_int(s, default)
    None => default
  }
}

///|
fn parse_metric(args : Array[String]) -> @meandb.Metric {
  match flag_value(args, "--metric") {
    Some("l2") | Some("L2") => @meandb.L2
    Some("dot") | Some("ip") => @meandb.Dot
    _ => @meandb.Cosine
  }
}

///|
fn magic_is(b : FixedArray[Byte], tag : String) -> Bool {
  let t = str_to_bytes(tag)
  if b.length() < t.length() {
    return false
  }
  for i in 0..<t.length() {
    if b[i] != t[i] {
      return false
    }
  }
  true
}

///|
/// Detect the index kind from the 8-byte file magic (returns "" if unknown).
fn detect(b : FixedArray[Byte]) -> String {
  if magic_is(b, "VECDBIV1") {
    "ivf"
  } else if magic_is(b, "VECDBHN1") {
    "hnsw"
  } else if magic_is(b, "VECDBHQ1") {
    "hnsw-q"
  } else if magic_is(b, "VECDBPQ1") {
    "pq"
  } else if magic_is(b, "VECDBIP1") {
    "ivfpq"
  } else if magic_is(b, "VECDBOP1") {
    "opq"
  } else if b.length() >= 6 &&
    b[0] == 86 &&
    b[1] == 69 &&
    b[2] == 67 &&
    b[3] == 68 &&
    b[4] == 66 &&
    b[5] == 49 {
    "flat" // "VECDB1\0\0"
  } else {
    ""
  }
}

///|
/// Run one CLI invocation. Returns 0 on success, 1 on error. All IO is via `io`.
pub fn run(io : Io, args : Array[String]) -> Int {
  if args.length() == 0 {
    (io.print)("usage: <build|search|info> ...")
    return 1
  }
  match args[0] {
    "build" => cmd_build(io, args)
    "search" => cmd_search(io, args)
    "info" => cmd_info(io, args)
    other => {
      (io.print)("unknown command: " + other)
      1
    }
  }
}

///|
fn read_or_err(io : Io, path : String) -> FixedArray[Byte]? {
  match (io.read)(path) {
    Some(b) => Some(b)
    None => {
      (io.print)("error: cannot read " + path)
      None
    }
  }
}

///|
fn cmd_build(io : Io, args : Array[String]) -> Int {
  if args.length() < 3 {
    (io.print)("usage: build <in.csv> <out.vecdb> [--index K] [--metric M] ...")
    return 1
  }
  let input = args[1]
  let output = args[2]
  let bytes = match read_or_err(io, input) {
    Some(b) => b
    None => return 1
  }
  let (ids, vecs) = parse_csv(bytes)
  if vecs.length() == 0 {
    (io.print)("error: no records in " + input)
    return 1
  }
  let metric = parse_metric(args)
  let keep_raw = !has_flag(args, "--compact")
  let kind = flag_value(args, "--index").unwrap_or("flat")
  let out : FixedArray[Byte] = match kind {
    "flat" => @meandb.FlatIndex::build(vecs, ids, metric, keep_raw).to_bytes()
    "ivf" => {
      let nlist = int_flag(args, "--nlist", 256)
      @meandb.IvfIndex::build(vecs, ids, metric, nlist, keep_raw, 12).to_bytes()
    }
    "hnsw" => {
      let m = int_flag(args, "-m", 16)
      let efc = int_flag(args, "--ef-construction", 200)
      let idx = @meandb.HnswIndex::new(vecs[0].length(), metric, m, efc)
      for i in 0..<vecs.length() {
        idx.add(ids[i], vecs[i])
      }
      idx.to_bytes()
    }
    "hnsw-q" => {
      let m = int_flag(args, "-m", 16)
      let efc = int_flag(args, "--ef-construction", 200)
      let idx = @meandb.HnswQIndex::new(
        vecs[0].length(),
        metric,
        m,
        efc,
        keep_raw,
      )
      for i in 0..<vecs.length() {
        idx.add(ids[i], vecs[i])
      }
      idx.to_bytes()
    }
    "pq" => {
      let m = int_flag(args, "--pq-m", 16)
      let ksub = int_flag(args, "--ksub", 256)
      @meandb.PqIndex::build(vecs, ids, metric, m, ksub, 20, keep_raw).to_bytes()
    }
    "ivfpq" => {
      let nlist = int_flag(args, "--nlist", 256)
      let m = int_flag(args, "--pq-m", 16)
      let ksub = int_flag(args, "--ksub", 256)
      @meandb.IvfPqIndex::build(
        vecs, ids, metric, nlist, m, ksub, 15, keep_raw,
      ).to_bytes()
    }
    "opq" => {
      let m = int_flag(args, "--pq-m", 16)
      let ksub = int_flag(args, "--ksub", 256)
      @meandb.OpqIndex::build(vecs, ids, metric, m, ksub, 20, 4, keep_raw).to_bytes()
    }
    other => {
      (io.print)("error: unknown --index " + other)
      return 1
    }
  }
  (io.write)(output, out)
  (io.print)("built " + kind + ": \{vecs.length()} vectors -> " + output)
  0
}

///|
fn cmd_search(io : Io, args : Array[String]) -> Int {
  if args.length() < 3 {
    (io.print)("usage: search <index.vecdb> <query.csv> [-k N] ...")
    return 1
  }
  let idx_bytes = match read_or_err(io, args[1]) {
    Some(b) => b
    None => return 1
  }
  let q_bytes = match read_or_err(io, args[2]) {
    Some(b) => b
    None => return 1
  }
  let (_, queries) = parse_csv(q_bytes)
  let k = int_flag(args, "-k", 10)
  let over = int_flag(args, "--oversample", 4)
  let nprobe = int_flag(args, "--nprobe", 16)
  let ef = int_flag(args, "--ef", 64)
  let kind = detect(idx_bytes)
  if kind == "" {
    (io.print)("error: unrecognized index file (bad magic)")
    return 1
  }
  let search_one : (Array[Float]) -> Array[@meandb.Hit] = match kind {
    "flat" => {
      let idx = @meandb.FlatIndex::from_bytes(idx_bytes)
      fn(q) { idx.search(q, k, over) }
    }
    "ivf" => {
      let idx = @meandb.IvfIndex::from_bytes(idx_bytes)
      fn(q) { idx.search(q, k, nprobe, over) }
    }
    "hnsw" => {
      let idx = @meandb.HnswIndex::from_bytes(idx_bytes)
      fn(q) { idx.search(q, k, ef) }
    }
    "hnsw-q" => {
      let idx = @meandb.HnswQIndex::from_bytes(idx_bytes)
      fn(q) { idx.search(q, k, ef) }
    }
    "pq" => {
      let idx = @meandb.PqIndex::from_bytes(idx_bytes)
      fn(q) { idx.search(q, k, over) }
    }
    "ivfpq" => {
      let idx = @meandb.IvfPqIndex::from_bytes(idx_bytes)
      fn(q) { idx.search(q, k, nprobe, over) }
    }
    _ => {
      let idx = @meandb.OpqIndex::from_bytes(idx_bytes)
      fn(q) { idx.search(q, k, over) }
    }
  }
  for qi in 0..<queries.length() {
    let hits = search_one(queries[qi])
    for rank in 0..<hits.length() {
      let h = hits[rank]
      (io.print)("\{qi}\t\{rank}\t\{h.id}\t\{h.score}")
    }
  }
  0
}

///|
fn cmd_info(io : Io, args : Array[String]) -> Int {
  if args.length() < 2 {
    (io.print)("usage: info <index.vecdb>")
    return 1
  }
  let b = match read_or_err(io, args[1]) {
    Some(x) => x
    None => return 1
  }
  let kind = detect(b)
  if kind == "" {
    (io.print)("error: unrecognized index file (bad magic)")
    return 1
  }
  let count = match kind {
    "flat" => @meandb.FlatIndex::from_bytes(b).length()
    "ivf" => @meandb.IvfIndex::from_bytes(b).length()
    "hnsw" => @meandb.HnswIndex::from_bytes(b).length()
    "hnsw-q" => @meandb.HnswQIndex::from_bytes(b).length()
    "pq" => @meandb.PqIndex::from_bytes(b).length()
    "ivfpq" => @meandb.IvfPqIndex::from_bytes(b).length()
    _ => @meandb.OpqIndex::from_bytes(b).length()
  }
  (io.print)("type:  " + kind)
  (io.print)("count: \{count}")
  0
}

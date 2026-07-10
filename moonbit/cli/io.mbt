///|
/// IO abstraction for the CLI so the same command logic runs in **any
/// environment**. The host supplies three closures — read a file, write a
/// file, print a line — and the CLI core (`run`) touches nothing else. Backends
/// without a real filesystem (e.g. the wasm target) back these with in-memory
/// buffers or host calls; a native host wires them to its own file IO.
pub struct Io {
  read : (String) -> FixedArray[Byte]?
  write : (String, FixedArray[Byte]) -> Unit
  print : (String) -> Unit
}

///|
/// Generic adapter: build an `Io` from explicit closures. Any host can plug in
/// its own environment this way (real fs, object storage, a JS `fetch`, ...).
pub fn Io::new(
  read : (String) -> FixedArray[Byte]?,
  write : (String, FixedArray[Byte]) -> Unit,
  print : (String) -> Unit,
) -> Io {
  { read, write, print }
}

///|
/// In-memory filesystem adapter. Works on every backend (native / wasm / js)
/// because it performs no real IO — ideal for tests, sandboxes, and wasm hosts
/// that hand vectors in as byte buffers. Seed inputs with `put` / `put_str`,
/// run the CLI over `io()`, then read written files with `get` and captured
/// stdout with `output`.
pub struct MemFs {
  store : Map[String, FixedArray[Byte]]
  out : StringBuilder
}

///|
pub fn MemFs::new() -> MemFs {
  { store: {}, out: StringBuilder::new() }
}

///|
/// Seed a file from raw bytes.
pub fn MemFs::put(self : MemFs, path : String, data : FixedArray[Byte]) -> Unit {
  self.store[path] = data
}

///|
/// Seed a file from ASCII/UTF-8 text (e.g. a CSV literal).
pub fn MemFs::put_str(self : MemFs, path : String, text : String) -> Unit {
  self.store[path] = str_to_bytes(text)
}

///|
/// Read a written file's bytes (None if absent).
pub fn MemFs::get(self : MemFs, path : String) -> FixedArray[Byte]? {
  self.store.get(path)
}

///|
/// Everything printed by the CLI so far.
pub fn MemFs::output(self : MemFs) -> String {
  self.out.to_string()
}

///|
/// The `Io` view over this in-memory store.
pub fn MemFs::io(self : MemFs) -> Io {
  Io::new(fn(p) { self.store.get(p) }, fn(p, b) { self.store[p] = b }, fn(s) {
    self.out.write_string(s)
    self.out.write_char('\n')
  })
}

///|
/// UTF-8 encode text into bytes (used to seed CSV inputs / compare magics).
pub fn str_to_bytes(s : String) -> FixedArray[Byte] {
  let u = @utf8.encode(s)
  let fa = FixedArray::make(u.length(), (0).to_byte())
  for i in 0..<u.length() {
    fa[i] = u[i]
  }
  fa
}

///|
/// bytes (ASCII) -> String.
pub fn bytes_to_str(b : FixedArray[Byte]) -> String {
  let sb = StringBuilder::new()
  for i in 0..<b.length() {
    sb.write_char(b[i].to_int().to_char().unwrap())
  }
  sb.to_string()
}

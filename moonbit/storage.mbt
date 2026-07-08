// `.vecdb` serialization, byte-compatible with the Rust crate's format.
//
// Layout (little-endian): 64-byte header + sections aligned to 16 bytes:
//   [header][ ids u64 ][ scales f32 ][ sqnorms f32 ][ codes i8 ][ raw f32? ]
// See DESIGN.md for the header fields.
//
// MoonBit's wasm runtime has no direct filesystem, so these work on byte
// buffers (`FixedArray[Byte]`); the host provides file I/O. The produced bytes
// are identical to what the Rust `save` writes, so indexes are interoperable.

///|
fn align16(x : Int) -> Int {
  (x + 15) & (0xFFFFFFF0)
}

///|
fn metric_to_u32(m : Metric) -> UInt {
  match m {
    L2 => 0
    Dot => 1
    Cosine => 2
  }
}

///|
fn metric_from_u32(v : UInt) -> Metric {
  if v == 1 {
    Dot
  } else if v == 2 {
    Cosine
  } else {
    L2
  }
}

///|
fn read_u32_le(b : FixedArray[Byte], off : Int) -> UInt {
  b[off].to_uint() |
  (b[off + 1].to_uint() << 8) |
  (b[off + 2].to_uint() << 16) |
  (b[off + 3].to_uint() << 24)
}

///|
fn write_u32_le(b : FixedArray[Byte], off : Int, v : UInt) -> Unit {
  b[off] = (v & 0xFF).to_byte()
  b[off + 1] = ((v >> 8) & 0xFF).to_byte()
  b[off + 2] = ((v >> 16) & 0xFF).to_byte()
  b[off + 3] = ((v >> 24) & 0xFF).to_byte()
}

///|
fn read_u64_le(b : FixedArray[Byte], off : Int) -> UInt64 {
  let lo = read_u32_le(b, off).to_uint64()
  let hi = read_u32_le(b, off + 4).to_uint64()
  lo | (hi << 32)
}

///|
fn write_u64_le(b : FixedArray[Byte], off : Int, v : UInt64) -> Unit {
  write_u32_le(b, off, (v & 0xFFFFFFFF).to_uint())
  write_u32_le(b, off + 4, (v >> 32).to_uint())
}

///|
/// Section byte offsets for a given shape.
priv struct Layout {
  ids : Int
  scales : Int
  sqnorms : Int
  codes : Int
  raw : Int
  total : Int
}

///|
fn layout(dim : Int, count : Int, has_raw : Bool) -> Layout {
  let ids = align16(64)
  let scales = align16(ids + count * 8)
  let sqnorms = align16(scales + count * 4)
  let codes = align16(sqnorms + count * 4)
  let raw = align16(codes + count * dim)
  let total = if has_raw { align16(raw + count * dim * 4) } else { raw }
  { ids, scales, sqnorms, codes, raw, total }
}

///|
/// Serialize the index to a `.vecdb` byte buffer.
pub fn FlatIndex::to_bytes(self : FlatIndex) -> FixedArray[Byte] {
  let dim = self.dim
  let count = self.count
  let has_raw = self.has_raw()
  let l = layout(dim, count, has_raw)
  let b : FixedArray[Byte] = FixedArray::make(l.total, 0)

  // Header. magic = "VECDB1\0\0"
  let magic : FixedArray[Byte] = [0x56, 0x45, 0x43, 0x44, 0x42, 0x31, 0x00, 0x00]
  for i in 0..<8 {
    b[i] = magic[i]
  }
  write_u32_le(b, 8, 1) // version
  write_u32_le(b, 12, metric_to_u32(self.metric))
  write_u32_le(b, 16, dim.reinterpret_as_uint())
  write_u32_le(b, 20, count.reinterpret_as_uint())
  write_u32_le(b, 24, if has_raw { 1 } else { 0 })

  for i in 0..<count {
    write_u64_le(b, l.ids + i * 8, self.ids[i].reinterpret_as_uint64())
    write_f32(b, l.scales + i * 4, self.scales[i])
    write_f32(b, l.sqnorms + i * 4, self.sqnorms[i])
  }
  for j in 0..<(count * dim) {
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
/// Parse a `.vecdb` byte buffer into a [`FlatIndex`].
pub fn FlatIndex::from_bytes(b : FixedArray[Byte]) -> FlatIndex {
  if b.length() < 64 {
    abort("from_bytes: buffer smaller than header")
  }
  if b[0] != 0x56 || b[1] != 0x45 || b[2] != 0x43 || b[3] != 0x44 {
    abort("from_bytes: bad magic")
  }
  if read_u32_le(b, 8) != 1 {
    abort("from_bytes: unsupported version")
  }
  let metric = metric_from_u32(read_u32_le(b, 12))
  let dim = read_u32_le(b, 16).reinterpret_as_int()
  let count = read_u32_le(b, 20).reinterpret_as_int()
  let has_raw = read_u32_le(b, 24) != 0
  let l = layout(dim, count, has_raw)
  if b.length() < l.total {
    abort("from_bytes: buffer truncated")
  }

  let ids = Array::make(count, (0 : Int64))
  let scales = Array::make(count, (0.0 : Float))
  let sqnorms = Array::make(count, (0.0 : Float))
  for i in 0..<count {
    ids[i] = read_u64_le(b, l.ids + i * 8).reinterpret_as_int64()
    scales[i] = read_f32(b, l.scales + i * 4)
    sqnorms[i] = read_f32(b, l.sqnorms + i * 4)
  }
  let codes : FixedArray[Byte] = FixedArray::make(count * dim, 0)
  for j in 0..<(count * dim) {
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
  { dim, metric, count, ids, codes, scales, sqnorms, raw }
}

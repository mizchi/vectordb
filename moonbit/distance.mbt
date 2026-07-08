// Distance kernels for int8 codes and f32 vectors, accelerated with the
// `moonbitlang/core/v128` SIMD intrinsics.
//
// Data is stored as raw little-endian bytes in `FixedArray[Byte]` so the
// `v128_load*` intrinsics can read it directly:
//   - int8 codes: one byte per lane, loaded 8-at-a-time via `v128_load8x8_s`
//     (sign-extends to i16x8) and reduced with `i32x4_dot_i16x8_s`.
//   - f32 values: 4 bytes per lane, loaded 16 bytes (4 floats) at a time.
//
// The intrinsics fall back to scalar code on backends without SIMD, so these
// functions are portable across all MoonBit targets.

///|
/// Read a two's-complement int8 stored in a byte as a signed `Int`.
fn read_i8(b : FixedArray[Byte], off : Int) -> Int {
  let v = b[off].to_int()
  if v >= 128 {
    v - 256
  } else {
    v
  }
}

///|
/// Read a little-endian f32 from a byte offset.
fn read_f32(b : FixedArray[Byte], off : Int) -> Float {
  let bits = b[off].to_uint() |
    (b[off + 1].to_uint() << 8) |
    (b[off + 2].to_uint() << 16) |
    (b[off + 3].to_uint() << 24)
  Float::reinterpret_from_uint(bits)
}

///|
/// Integer inner product of two int8 code ranges (8 lanes / iteration).
///
/// The caller multiplies the result by the two per-vector scales to recover
/// the approximate floating-point inner product.
pub fn dot_i8(
  a : FixedArray[Byte],
  a_off : Int,
  b : FixedArray[Byte],
  b_off : Int,
  dim : Int,
) -> Int {
  let mut acc = @v128.i32x4_splat(0U)
  let mut i = 0
  while i + 8 <= dim {
    let va = @v128.v128_load8x8_s(a, a_off + i)
    let vb = @v128.v128_load8x8_s(b, b_off + i)
    acc = @v128.i32x4_add(acc, @v128.i32x4_dot_i16x8_s(va, vb))
    i = i + 8
  }
  let mut sum = 0
  for lane in 0..<4 {
    sum = sum + @v128.i32x4_extract_lane(acc, lane).reinterpret_as_int()
  }
  while i < dim {
    sum = sum + read_i8(a, a_off + i) * read_i8(b, b_off + i)
    i = i + 1
  }
  sum
}

///|
/// Inner product of two f32 ranges (4 lanes / iteration). Offsets are in bytes.
pub fn dot_f32(
  a : FixedArray[Byte],
  a_off : Int,
  b : FixedArray[Byte],
  b_off : Int,
  dim : Int,
) -> Float {
  let mut acc = @v128.f32x4_splat((0.0 : Float))
  let mut i = 0
  while i + 4 <= dim {
    let va = @v128.v128_load(a, a_off + i * 4)
    let vb = @v128.v128_load(b, b_off + i * 4)
    acc = @v128.f32x4_add(acc, @v128.f32x4_mul(va, vb))
    i = i + 4
  }
  let mut sum : Float = 0.0
  for lane in 0..<4 {
    sum = sum + @v128.f32x4_extract_lane(acc, lane)
  }
  while i < dim {
    sum = sum + read_f32(a, a_off + i * 4) * read_f32(b, b_off + i * 4)
    i = i + 1
  }
  sum
}

///|
/// Squared Euclidean distance of two f32 ranges. Offsets are in bytes.
pub fn l2sq_f32(
  a : FixedArray[Byte],
  a_off : Int,
  b : FixedArray[Byte],
  b_off : Int,
  dim : Int,
) -> Float {
  let mut acc = @v128.f32x4_splat((0.0 : Float))
  let mut i = 0
  while i + 4 <= dim {
    let va = @v128.v128_load(a, a_off + i * 4)
    let vb = @v128.v128_load(b, b_off + i * 4)
    let d = @v128.f32x4_sub(va, vb)
    acc = @v128.f32x4_add(acc, @v128.f32x4_mul(d, d))
    i = i + 4
  }
  let mut sum : Float = 0.0
  for lane in 0..<4 {
    sum = sum + @v128.f32x4_extract_lane(acc, lane)
  }
  while i < dim {
    let d = read_f32(a, a_off + i * 4) - read_f32(b, b_off + i * 4)
    sum = sum + d * d
    i = i + 1
  }
  sum
}

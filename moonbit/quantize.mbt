// Per-vector symmetric int8 scalar quantization.
//
// For a vector `x`: `scale = max(|x_i|) / 127`, `q_i = round(x_i / scale)`
// clamped to [-127, 127], stored as a two's-complement byte.

///|
/// A quantized vector: int8 codes (as bytes) plus reconstruction metadata.
pub struct Quantized {
  codes : FixedArray[Byte]
  scale : Float
  sqnorm : Float
}

///|
/// Round a `Float` to the nearest integer (ties away from zero).
fn round_f(x : Float) -> Int {
  let d = x.to_double()
  if d >= 0.0 {
    (d + 0.5).to_int()
  } else {
    (d - 0.5).to_int()
  }
}

///|
/// Store a signed int8 value (`-127..=127`) as a two's-complement byte.
fn to_i8_byte(q : Int) -> Byte {
  if q < 0 {
    (q + 256).to_byte()
  } else {
    q.to_byte()
  }
}

///|
/// Quantize an f32 vector.
pub fn quantize(x : Array[Float]) -> Quantized {
  let dim = x.length()
  let mut amax : Float = 0.0
  let mut sqnorm : Float = 0.0
  for i in 0..<dim {
    let v = x[i]
    let a = if v < 0.0 { -v } else { v }
    if a > amax {
      amax = a
    }
    sqnorm = sqnorm + v * v
  }
  let scale = if amax > 0.0 { amax / 127.0 } else { (1.0 : Float) }
  let inv = (1.0 : Float) / scale
  let codes : FixedArray[Byte] = FixedArray::make(dim, 0)
  for i in 0..<dim {
    let q = round_f(x[i] * inv)
    let qc = if q > 127 { 127 } else if q < -127 { -127 } else { q }
    codes[i] = to_i8_byte(qc)
  }
  { codes, scale, sqnorm }
}

///|
/// Reconstruct an approximate f32 vector from int8 codes and a scale.
pub fn dequantize(codes : FixedArray[Byte], scale : Float) -> Array[Float] {
  let out = Array::make(codes.length(), (0.0 : Float))
  for i in 0..<codes.length() {
    out[i] = Float::from_double(read_i8(codes, i).to_double()) * scale
  }
  out
}

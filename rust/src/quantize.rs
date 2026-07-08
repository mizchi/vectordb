//! Per-vector symmetric int8 scalar quantization.
//!
//! For a vector `x`, `scale = max(|x_i|) / 127` and `q_i = round(x_i / scale)`,
//! clamped to `[-127, 127]`. The approximate inner product of two quantized
//! vectors is `scale_a * scale_b * dot_i8(q_a, q_b)`.

/// A quantized vector: int8 codes plus the metadata needed to reconstruct
/// approximate distances.
#[derive(Clone, Debug)]
pub struct Quantized {
    pub codes: Vec<i8>,
    /// Per-vector scale: `original ≈ code * scale`.
    pub scale: f32,
    /// Squared L2 norm of the *original* vector (used for L2 reconstruction).
    pub sqnorm: f32,
}

/// Quantize a single f32 vector.
pub fn quantize(x: &[f32]) -> Quantized {
    let mut amax = 0.0f32;
    let mut sqnorm = 0.0f32;
    for &v in x {
        let a = v.abs();
        if a > amax {
            amax = a;
        }
        sqnorm += v * v;
    }
    let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
    let inv = 1.0 / scale;
    let codes = x
        .iter()
        .map(|&v| {
            let q = (v * inv).round();
            q.clamp(-127.0, 127.0) as i8
        })
        .collect();
    Quantized {
        codes,
        scale,
        sqnorm,
    }
}

/// Reconstruct an approximate f32 vector from its codes and scale.
pub fn dequantize(codes: &[i8], scale: f32, out: &mut [f32]) {
    debug_assert_eq!(codes.len(), out.len());
    for (o, &c) in out.iter_mut().zip(codes.iter()) {
        *o = c as f32 * scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_is_close() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let q = quantize(&x);
        let mut back = vec![0.0f32; x.len()];
        dequantize(&q.codes, q.scale, &mut back);
        // Max abs error is bounded by half a quantization step.
        let step = q.scale;
        for (a, b) in x.iter().zip(back.iter()) {
            assert!((a - b).abs() <= step, "{a} vs {b} (step {step})");
        }
    }

    #[test]
    fn zero_vector_is_safe() {
        let q = quantize(&[0.0, 0.0, 0.0]);
        assert!(q.codes.iter().all(|&c| c == 0));
        assert_eq!(q.sqnorm, 0.0);
    }
}

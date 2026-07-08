//! Distance kernels for f32 and int8 vectors.
//!
//! Each public function dispatches to an AVX2 implementation at runtime on
//! x86_64 when available, and falls back to a scalar loop written so the
//! autovectorizer can still make good use of it elsewhere.

/// Inner product of two equal-length f32 slices.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: guarded by runtime feature detection.
            return unsafe { dot_f32_avx2(a, b) };
        }
    }
    dot_f32_scalar(a, b)
}

/// Squared Euclidean distance of two equal-length f32 slices.
#[inline]
pub fn l2sq_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: guarded by runtime feature detection.
            return unsafe { l2sq_f32_avx2(a, b) };
        }
    }
    l2sq_f32_scalar(a, b)
}

/// Integer inner product of two equal-length int8 slices.
///
/// Returns an `i32`; the caller multiplies by the two per-vector scales to
/// recover the approximate floating-point inner product.
#[inline]
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime feature detection.
            return unsafe { dot_i8_avx2(a, b) };
        }
    }
    dot_i8_scalar(a, b)
}

// ----------------------------------------------------------------------------
// Scalar implementations
// ----------------------------------------------------------------------------

pub fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        acc += x * y;
    }
    acc
}

pub fn l2sq_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        acc += d * d;
    }
    acc
}

pub fn dot_i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    let mut acc: i32 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        acc += (*x as i32) * (*y as i32);
    }
    acc
}

// ----------------------------------------------------------------------------
// AVX2 implementations (x86_64)
// ----------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let av = _mm256_loadu_ps(a.as_ptr().add(i));
        let bv = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_fmadd_ps(av, bv, acc);
        i += 8;
    }
    let mut sum = hsum256_ps(acc);
    while i < n {
        sum += a.get_unchecked(i) * b.get_unchecked(i);
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2sq_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let av = _mm256_loadu_ps(a.as_ptr().add(i));
        let bv = _mm256_loadu_ps(b.as_ptr().add(i));
        let d = _mm256_sub_ps(av, bv);
        acc = _mm256_fmadd_ps(d, d, acc);
        i += 8;
    }
    let mut sum = hsum256_ps(acc);
    while i < n {
        let d = a.get_unchecked(i) - b.get_unchecked(i);
        sum += d * d;
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_avx2(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    // Two independent accumulators over 32 int8 / iteration to hide the latency
    // of cvtepi8_epi16 + madd_epi16 (better instruction-level parallelism).
    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let mut i = 0;
    while i + 32 <= n {
        let a0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(ap.add(i) as *const __m128i));
        let b0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(bp.add(i) as *const __m128i));
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(a0, b0));
        let a1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(ap.add(i + 16) as *const __m128i));
        let b1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(bp.add(i + 16) as *const __m128i));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(a1, b1));
        i += 32;
    }
    // 16-wide tail.
    while i + 16 <= n {
        let av = _mm256_cvtepi8_epi16(_mm_loadu_si128(ap.add(i) as *const __m128i));
        let bv = _mm256_cvtepi8_epi16(_mm_loadu_si128(bp.add(i) as *const __m128i));
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(av, bv));
        i += 16;
    }
    let mut sum = hsum256_epi32(_mm256_add_epi32(acc0, acc1));
    while i < n {
        sum += (*a.get_unchecked(i) as i32) * (*b.get_unchecked(i) as i32);
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum256_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let sums2 = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(sums2)
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum256_epi32(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    let sum128 = _mm_add_epi32(lo, hi);
    let hi64 = _mm_unpackhi_epi64(sum128, sum128);
    let sum64 = _mm_add_epi32(sum128, hi64);
    let hi32 = _mm_shuffle_epi32(sum64, 0b_00_00_00_01);
    let sum32 = _mm_add_epi32(sum64, hi32);
    _mm_cvtsi128_si32(sum32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_scalar() {
        let a: Vec<f32> = (0..37).map(|i| (i as f32) * 0.5 - 3.0).collect();
        let b: Vec<f32> = (0..37).map(|i| (i as f32) * -0.25 + 1.0).collect();
        let got = dot_f32(&a, &b);
        let want = dot_f32_scalar(&a, &b);
        assert!((got - want).abs() < 1e-3, "{got} vs {want}");
    }

    #[test]
    fn l2sq_matches_scalar() {
        let a: Vec<f32> = (0..40).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..40).map(|i| (i as f32).cos()).collect();
        let got = l2sq_f32(&a, &b);
        let want = l2sq_f32_scalar(&a, &b);
        assert!((got - want).abs() < 1e-3, "{got} vs {want}");
    }

    #[test]
    fn dot_i8_matches_scalar() {
        let a: Vec<i8> = (0..50).map(|i| (i as i8) - 25).collect();
        let b: Vec<i8> = (0..50).map(|i| 40 - (i as i8)).collect();
        assert_eq!(dot_i8(&a, &b), dot_i8_scalar(&a, &b));
    }
}

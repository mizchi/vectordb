name = "mizchi/vectordb"

version = "0.1.0"

license = "MIT OR Apache-2.0"

description = "Compact vector search: flat + int8 scalar quantization + rerank, with v128 SIMD distance kernels"

// Build/run/test on the native backend by default so the v128 SIMD intrinsics
// are used. They also compile and run on every other backend via the scalar
// fallbacks in moonbitlang/core/v128.
preferred_target = "native"

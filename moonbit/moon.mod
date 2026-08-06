name = "mizchi/meandb"

version = "0.1.0"

license = "MIT OR Apache-2.0"

description = "Compact vector search: flat + int8 scalar quantization + rerank, with v128 SIMD distance kernels"

// Default to the wasm backend: it is the only target where the core v128
// intrinsics are lowered to real hardware SIMD (WASM SIMD). The code also
// compiles and runs on native/wasm-gc/js via scalar fallbacks.

preferred_target = "wasm"

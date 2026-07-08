//! # vectordb
//!
//! A compact, embeddable vector search database:
//!
//! - **Flat** (brute-force) index with SIMD distance kernels.
//! - **int8 scalar quantization** for a 4x smaller footprint.
//! - **rerank**: a widened int8 candidate set re-scored with exact f32.
//! - **mmap single-file** persistence (`.vecdb`), zero-copy on load.
//!
//! ```
//! use vectordb::{FlatIndex, Metric};
//!
//! let mut idx = FlatIndex::new(3, Metric::Cosine, true);
//! idx.add(1, &[1.0, 0.0, 0.0]);
//! idx.add(2, &[0.0, 1.0, 0.0]);
//! let hits = idx.search(&[0.9, 0.1, 0.0], 1, 4);
//! assert_eq!(hits[0].id, 1);
//! ```

pub mod distance;
pub mod index;
pub mod quantize;
pub mod storage;

pub use index::{FlatIndex, Hit, Metric, View};
pub use quantize::{quantize, Quantized};
pub use storage::{load, open, save, MmapIndex};

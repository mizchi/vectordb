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

pub mod bin_quant;
pub mod cli;
pub mod diskann;
pub mod distance;
pub mod hnsw;
pub mod hnsw_q;
pub mod index;
pub mod ivf;
pub mod ivf_pq;
pub mod opq;
pub mod pq;
pub mod quantize;
pub mod rabitq;
pub mod storage;

pub use bin_quant::BinaryIndex;
pub use diskann::{DiskAnnIndex, MmapDiskAnn};
pub use hnsw::HnswIndex;
pub use hnsw_q::HnswQIndex;
pub use index::{FlatIndex, Hit, Metric, View};
pub use ivf::IvfIndex;
pub use ivf_pq::IvfPqIndex;
pub use opq::OpqIndex;
pub use pq::PqIndex;
pub use quantize::{quantize, Quantized};
pub use rabitq::{IvfRabitqIndex, RabitqIndex};
pub use storage::{from_bytes, load, open, save, to_bytes, MmapIndex};

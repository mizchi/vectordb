//! # meandb
//!
//! A meaning-aware database composed from two explicit layers:
//!
//! - [`vector`]: vector indexes, metrics, persistence, and ANN search;
//! - [`graph`]: semantic links, metadata, traversal, provenance-aware queries,
//!   and graph persistence.
//!
//! The umbrella crate keeps the two layers discoverable under one stable public
//! namespace while each layer remains independently usable and releasable.

/// Vector-search primitives and index implementations.
pub mod vector {
    pub use meandb_vector::*;
}

/// Semantic graph primitives, traversal, and structured query support.
pub mod graph {
    pub use meandb_graph::*;
}

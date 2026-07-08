//! Dump the IVF `.vecdb` bytes of a fixed tiny index (deterministic k-means:
//! nlist == count, 1 iteration → identity assignment, centroids == inputs) as
//! space-separated decimals, to diff against MoonBit's IVF `to_bytes`.
//!
//! Run: `cargo run --example dump_ivf`

use vectordb::{IvfIndex, Metric};

fn main() {
    let items = vec![
        (1u64, vec![1.0f32, 2.0, 3.0]),
        (2, vec![-1.0, 0.5, 4.0]),
        (3, vec![0.0, 0.0, 1.0]),
    ];
    let idx = IvfIndex::build(3, Metric::L2, /*nlist=*/ 3, &items, /*keep_raw=*/ true, 1);
    let path = std::env::temp_dir().join("interop_ivf.vecdb");
    idx.save(&path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let parts: Vec<String> = bytes.iter().map(|b| b.to_string()).collect();
    println!("{}", parts.join(" "));
}

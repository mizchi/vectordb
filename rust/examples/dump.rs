//! Dump the `.vecdb` bytes of a fixed tiny index as space-separated decimals,
//! to diff against the MoonBit `to_bytes` output (byte-compatibility check).
//!
//! Run: `cargo run --example dump`

use vectordb::{save, FlatIndex, Metric};

fn main() {
    let mut idx = FlatIndex::new(3, Metric::L2, true);
    idx.add(1, &[1.0, 2.0, 3.0]);
    idx.add(2, &[-1.0, 0.5, 4.0]);
    idx.add(3, &[0.0, 0.0, 1.0]);

    let path = std::env::temp_dir().join("interop_dump.vecdb");
    save(&idx, &path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let parts: Vec<String> = bytes.iter().map(|b| b.to_string()).collect();
    println!("{}", parts.join(" "));
}

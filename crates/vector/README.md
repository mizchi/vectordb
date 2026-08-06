# meandb-vector

`meandb-vector` is the vector-search engine in
[`mizchi/meandb`](https://github.com/mizchi/meandb). It owns vector
storage, metrics, ANN indexes, and `.vecdb` persistence. It intentionally does
not own graph links, tags, traversal, or graph analytics; those belong to the
separate [`meandb-graph`](https://crates.io/crates/meandb-graph) package.

Most consumers should use the `meandb` facade and import this layer as
`meandb::vector`. Depend on `meandb-vector` directly only for a vector-only
deployment.

## Install

```toml
[dependencies]
meandb-vector = "0.1"
```

```rust
use meandb_vector::{FlatIndex, Metric};

let mut index = FlatIndex::new(3, Metric::Cosine, true);
index.add(1, &[1.0, 0.0, 0.0]);
let hits = index.search(&[1.0, 0.0, 0.0], 1, 1);
assert_eq!(hits[0].id, 1);
```

The repository README documents the index choices, CLI, file formats, and
MoonBit implementation.

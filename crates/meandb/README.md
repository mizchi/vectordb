# meandb

`meandb` is a meaning-aware database with two composable layers:

- `meandb::vector` for vector indexes and nearest-neighbor search;
- `meandb::graph` for explicit semantic links, metadata, traversal, and
  provenance-aware structured queries.

```rust
use meandb::{
    graph::{GraphBuilder, GraphStore},
    vector::Metric,
};

let graph: GraphStore = GraphBuilder::new(Metric::Cosine, 8).build(&[], &[]);
```

Use `meandb-vector` or `meandb-graph` directly only when a consumer needs one
layer without the umbrella facade.

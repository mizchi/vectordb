# Releasing the Rust crates

This repository is the `meandb` Cargo workspace. Its public facade contains two
independently usable layers:

| Responsibility | crates.io package | Rust import |
| --- | --- | --- |
| Vector metrics, ANN indexes, `.vecdb` persistence | `meandb-vector` | `meandb::vector` |
| Graph links, traversal, tags, analytics, `.graphdb` persistence | `meandb-graph` | `meandb::graph` |
| Unified facade | `meandb` | `meandb` |

`meandb-graph` depends only on the public `meandb-vector` crate API. It must not
move graph traversal, tags, link semantics, or graph file-format concerns into
`meandb-vector`.

## Release contract

- The packages use independent semantic versions.
- The supported minimum Rust version is 1.87, declared both in Cargo metadata
  and in `rust-toolchain.toml`.
- A `meandb-vector` change does not require a graph release unless the graph
  layer's
  supported dependency range must change.
- Before a graph release, update its `meandb-vector` version requirement to
  the oldest compatible released vector-engine version.
- The `meandb` facade re-exports both layers under `meandb::{vector, graph}`.
  Consumers can select an individual package only when they deliberately need
  one layer.

## Checklist

1. Bump only the crate version(s) being released and update the dependency
   requirement when applicable.
2. Run `just check`. It formats, tests, lints, verifies every publishable
   package, and compiles a consumer through the `meandb` facade.
3. Inspect staged crate contents with `cargo package --list -p meandb-vector`,
   `meandb-graph`, or `meandb`.
4. Publish the vector layer first when multiple packages change; then publish
   the graph layer and facade:

   ```sh
   cargo publish -p meandb-vector
   cargo publish -p meandb-graph
   just package-meandb-registry
   cargo publish -p meandb
   ```

Publishing changes crates.io state and is intentionally not part of `just`.
The CI workflow runs `just check` on pull requests and on `main`, but never
publishes a crate.

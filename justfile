default: check

fmt:
    cargo fmt --all -- --check

test:
    cargo test --workspace

clippy:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

# Verify the vector crate as a complete registry package. The graph layer and
# facade are checked for their exact publishable file sets because they may
# intentionally depend on not-yet-published workspace releases.
package:
    cargo package --allow-dirty -p meandb-vector
    cargo package --allow-dirty --list -p meandb-graph > /dev/null
    cargo package --allow-dirty --list -p meandb > /dev/null

# Verify package metadata, the public facade, and layer dependencies without
# needing publication.
release-contract:
    cargo metadata --format-version 1 --no-deps | jq -e '([.packages[] | select(.name == "meandb-vector" or .name == "meandb-graph" or .name == "meandb")] | length == 3) and ([.packages[] | select(.name == "meandb-vector" or .name == "meandb-graph" or .name == "meandb") | select(.repository == "https://github.com/mizchi/meandb" and .homepage == "https://github.com/mizchi/meandb" and .rust_version == "1.87")] | length == 3) and ([.packages[] | select(.name == "meandb-vector") | .targets[] | select((.kind | index("lib")) and (.name == "meandb_vector"))] | length == 1) and ([.packages[] | select(.name == "meandb-graph") | .targets[] | select((.kind | index("lib")) and (.name == "meandb_graph"))] | length == 1) and ([.packages[] | select(.name == "meandb") | .targets[] | select((.kind | index("lib")) and (.name == "meandb"))] | length == 1) and ([.packages[] | select(.name == "meandb-graph") | .dependencies[] | select(.name == "meandb-vector" and .req == "^0.1.0")] | length == 1) and ([.packages[] | select(.name == "meandb") | .dependencies[] | select(.name == "meandb-vector" or .name == "meandb-graph")] | length == 2)'

# Run after the selected layer versions are available on crates.io. This
# verifies the facade exactly as a registry consumer receives it.
package-meandb-registry:
    cargo package --allow-dirty -p meandb

# Exercises the dependency declarations and imports that an external Rust
# consumer uses, without making this fixture a publishable third library.
consumer-contract:
    cargo run -p release-consumer-contract

check: fmt test clippy release-contract consumer-contract package

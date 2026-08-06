//! Compatibility binary for the former `graphdb` command.

#[path = "graph_cli.rs"]
mod implementation;

fn main() -> std::process::ExitCode {
    implementation::entrypoint()
}

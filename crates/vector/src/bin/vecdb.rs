//! Compatibility binary for the former `vecdb` command.

#[path = "vector_cli.rs"]
mod implementation;

fn main() -> std::process::ExitCode {
    implementation::entrypoint()
}

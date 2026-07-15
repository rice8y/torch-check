//! `torch-check` executable entry point.

#[tokio::main]
async fn main() -> std::process::ExitCode {
    torch_check::cli::run().await
}

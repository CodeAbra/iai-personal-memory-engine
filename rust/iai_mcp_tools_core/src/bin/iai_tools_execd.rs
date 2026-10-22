//! Resident executor daemon entrypoint: binds its own Unix socket, serves
//! `tool_run`/`result_fetch`, and self-terminates after an idle window.
//! Builds pyo3-free (no libpython link), same posture as `iai_tools_run`.

use std::process::ExitCode;

use iai_mcp_tools_core::daemon;

fn main() -> ExitCode {
    match daemon::run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("iai_tools_execd failed: {e}");
            ExitCode::FAILURE
        }
    }
}

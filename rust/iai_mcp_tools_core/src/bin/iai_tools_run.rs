//! Standalone one-shot runner: `iai_tools_run <program> [args...]` spawns and
//! captures a command; `iai_tools_run --fetch <run_id>` re-serves a prior
//! run's result without re-running it. Builds pyo3-free (no libpython link):
//! the default feature set never pulls the `pyo3` dependency in.

use std::process::ExitCode;

use iai_mcp_tools_core::exec;

fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().skip(1).collect();

    if args.first().map(String::as_str) == Some("--fetch") {
        let run_id = match args.get(1) {
            Some(id) => id.clone(),
            None => {
                eprintln!("usage: iai_tools_run --fetch <run_id>");
                return ExitCode::FAILURE;
            }
        };
        return match exec::fetch_result(&run_id) {
            Ok(result) => {
                print_result(&run_id, &result);
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("fetch failed: {e}");
                ExitCode::FAILURE
            }
        };
    }

    if args.is_empty() {
        eprintln!("usage: iai_tools_run <program> [args...]");
        return ExitCode::FAILURE;
    }

    let program = args.remove(0);
    let spec = exec::SpawnSpec { program, args, timeout: None };
    match exec::spawn_run(spec) {
        Ok(handle) => {
            print_result(&handle.run_id, &handle.result);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("spawn failed: {e}");
            ExitCode::FAILURE
        }
    }
}

fn print_result(run_id: &str, result: &exec::BoundedResult) {
    print!("{}", String::from_utf8_lossy(&result.stdout));
    eprint!("{}", String::from_utf8_lossy(&result.stderr));
    println!("run_id: {run_id}");
}

//! One-shot session-event CLI. `capture` reads hook-event JSON from stdin,
//! classifies it, and appends any resulting event to the session's own
//! store; `search` prints matching events as JSONL lines. Reuses the same
//! classify + append_event + query_events core as the pyo3 bindings -- no
//! forked capture path. Builds pyo3-free (no libpython link), same posture
//! as `iai_tools_run`. Scriptable entry point only -- not registered into
//! any live hook.

use std::io::Read;
use std::process::ExitCode;

use iai_mcp_tools_core::session::{extract, store};

fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: iai_tools_session <capture|search> [args...]");
        return ExitCode::FAILURE;
    }
    let subcommand = args.remove(0);
    match subcommand.as_str() {
        "capture" => run_capture(),
        "search" => run_search(&args),
        other => {
            eprintln!("unknown subcommand: {other}");
            ExitCode::FAILURE
        }
    }
}

/// Malformed/truncated/unreadable stdin is treated the same as a `None`
/// classification: a no-op, exit 0, never a panic that could disrupt the
/// calling hook.
fn run_capture() -> ExitCode {
    let mut payload = String::new();
    if std::io::stdin().read_to_string(&mut payload).is_err() {
        return ExitCode::SUCCESS;
    }
    let input: extract::HookInput = match serde_json::from_str(&payload) {
        Ok(v) => v,
        Err(_) => return ExitCode::SUCCESS,
    };
    let Some(event) = extract::classify(&input) else {
        return ExitCode::SUCCESS;
    };
    let root = match store::default_root() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("iai_tools_session: {e}");
            return ExitCode::FAILURE;
        }
    };
    let session_id = event.session_id.clone();
    match store::append_event(&root, &session_id, &event) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("iai_tools_session: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_search(args: &[String]) -> ExitCode {
    let mut session_id: Option<String> = None;
    let mut category: Option<String> = None;
    let mut substring: Option<String> = None;
    let mut since: Option<String> = None;
    let mut until: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--session" => {
                session_id = args.get(i + 1).cloned();
                i += 2;
            }
            "--category" => {
                category = args.get(i + 1).cloned();
                i += 2;
            }
            "--substring" => {
                substring = args.get(i + 1).cloned();
                i += 2;
            }
            "--since" => {
                since = args.get(i + 1).cloned();
                i += 2;
            }
            "--until" => {
                until = args.get(i + 1).cloned();
                i += 2;
            }
            _ => i += 1,
        }
    }

    let Some(session_id) = session_id else {
        eprintln!(
            "usage: iai_tools_session search --session <id> [--category <cat>] [--substring <text>] [--since <ts>] [--until <ts>]"
        );
        return ExitCode::FAILURE;
    };

    let root = match store::default_root() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("iai_tools_session: {e}");
            return ExitCode::FAILURE;
        }
    };

    let filter = store::QueryFilter {
        substring: substring.as_deref(),
        category: category.as_deref(),
        since: since.as_deref(),
        until: until.as_deref(),
    };
    let events = match store::query_events(&root, &session_id, filter) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("iai_tools_session: {e}");
            return ExitCode::FAILURE;
        }
    };

    for event in &events {
        match serde_json::to_string(event) {
            Ok(line) => println!("{line}"),
            Err(e) => eprintln!("iai_tools_session: {e}"),
        }
    }
    ExitCode::SUCCESS
}

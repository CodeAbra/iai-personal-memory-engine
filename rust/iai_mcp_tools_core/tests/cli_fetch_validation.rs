//! Proves the standalone `iai_tools_run --fetch <run_id>` CLI surface
//! rejects a malicious run_id rather than reading/writing outside the
//! artifact root -- the fourth surface (core fetch, daemon fetch, this CLI,
//! the Python client) `artifact::validate_run_id` must protect by
//! construction.

use std::path::Path;
use std::process::Command;

fn run_fetch(root: &Path, run_id: &str) -> std::process::Output {
    let bin = env!("CARGO_BIN_EXE_iai_tools_run");
    Command::new(bin)
        .env("IAI_TOOLS_ROOT", root)
        .args(["--fetch", run_id])
        .output()
        .expect("failed to run iai_tools_run")
}

#[test]
fn fetch_rejects_relative_traversal() {
    let tmp = tempfile::tempdir().unwrap();
    let output = run_fetch(tmp.path(), "../../../../etc");
    assert!(!output.status.success(), "traversal run_id must be refused");
}

#[test]
fn fetch_rejects_absolute_path_substitution() {
    let tmp = tempfile::tempdir().unwrap();
    let output = run_fetch(tmp.path(), "/etc");
    assert!(!output.status.success(), "absolute-path run_id must be refused");
}

#[test]
fn fetch_rejects_dotdot() {
    let tmp = tempfile::tempdir().unwrap();
    let output = run_fetch(tmp.path(), "..");
    assert!(!output.status.success());
}

#[test]
fn fetch_rejects_empty_run_id() {
    let tmp = tempfile::tempdir().unwrap();
    let output = run_fetch(tmp.path(), "");
    assert!(!output.status.success());
}

#[test]
fn fetch_rejects_embedded_separator() {
    let tmp = tempfile::tempdir().unwrap();
    let output = run_fetch(tmp.path(), "a/b");
    assert!(!output.status.success());
}

#[test]
fn fetch_rejects_unicode() {
    let tmp = tempfile::tempdir().unwrap();
    let output = run_fetch(tmp.path(), "1a-2b-\u{0441}03");
    assert!(!output.status.success());
}

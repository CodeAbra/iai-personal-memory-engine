//! Fail-closed sandbox seam. No OS-level sandboxing ships: a spawned
//! command inherits the caller's full environment and permissions, same
//! as a local shell. `ensure_sandbox_ready` is the seam a future sandbox
//! mode would refuse through on setup failure -- fail-closed, never
//! fail-open.
//!
//! Sits inside `spawn::spawn_bounded`, the only call site of
//! `Command::new` for a user-requested command in this crate (enforced by
//! `command_new_is_confined_to_spawn_rs` below) -- every command-running
//! surface reaches this check by construction.

use super::ExecError;

#[cfg(test)]
thread_local! {
    static FORCE_MISCONFIGURED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Test-only hook: forces `ensure_sandbox_ready` to take the refuse-to-run
/// branch on the CALLING THREAD only (each `#[test]` fn runs on its own
/// thread, so this never leaks across tests).
#[cfg(test)]
pub(crate) fn force_misconfigured_for_test(on: bool) {
    FORCE_MISCONFIGURED.with(|f| f.set(on));
}

pub fn ensure_sandbox_ready() -> Result<(), ExecError> {
    #[cfg(test)]
    {
        if FORCE_MISCONFIGURED.with(|f| f.get()) {
            return Err(ExecError::SandboxRefused(
                "sandbox mode requested but unavailable".to_string(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const REFUSAL_TEXT: &str = "sandbox requested but unavailable";

    #[test]
    fn default_posture_is_ok() {
        assert!(ensure_sandbox_ready().is_ok());
    }

    #[test]
    fn misconfigured_hook_refuses() {
        force_misconfigured_for_test(true);
        let result = ensure_sandbox_ready();
        force_misconfigured_for_test(false);
        assert!(result.is_err());
    }

    fn with_tools_root<T>(f: impl FnOnce() -> T) -> T {
        let _guard = crate::exec::artifact::ENV_LOCK.lock().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        // SAFETY: ENV_LOCK serializes every test that touches IAI_TOOLS_ROOT.
        unsafe {
            std::env::set_var("IAI_TOOLS_ROOT", tmp.path());
        }
        let result = f();
        unsafe {
            std::env::remove_var("IAI_TOOLS_ROOT");
        }
        result
    }

    /// Surface: the one-shot `iai_tools_run` binary, and the resident
    /// daemon's foreground `tool_run` path -- both call `exec::spawn_run`
    /// directly.
    #[test]
    fn exec_spawn_run_refuses_when_seam_misconfigured() {
        with_tools_root(|| {
            force_misconfigured_for_test(true);
            let result = crate::exec::spawn_run(crate::exec::SpawnSpec {
                program: "echo".into(),
                args: vec!["hi".into()],
                timeout: None,
            });
            force_misconfigured_for_test(false);

            match result {
                Ok(_) => panic!("spawn_run must refuse when the seam is misconfigured"),
                Err(err) => assert!(
                    err.to_string().contains(REFUSAL_TEXT),
                    "unexpected error: {err}"
                ),
            }
        });
    }

    /// Surface: the resident daemon's `background: true` path --
    /// `Daemon::spawn_background`.
    #[test]
    fn daemon_spawn_background_refuses_when_seam_misconfigured() {
        with_tools_root(|| {
            let daemon = crate::daemon::Daemon::new();
            force_misconfigured_for_test(true);
            let result = crate::daemon::Daemon::spawn_background(
                &daemon,
                crate::exec::SpawnSpec {
                    program: "echo".into(),
                    args: vec!["hi".into()],
                    timeout: None,
                },
            );
            force_misconfigured_for_test(false);

            let err =
                result.expect_err("spawn_background must refuse when the seam is misconfigured");
            assert!(
                err.to_string().contains(REFUSAL_TEXT),
                "unexpected error: {err}"
            );
        });
    }

    /// Surface: the resident daemon's `tool_batch_run` path (both
    /// concurrency modes share `run_one_with_timeout`, exercised here via
    /// the same wire-level entry point the socket dispatches to).
    #[test]
    fn daemon_batch_run_refuses_when_seam_misconfigured() {
        with_tools_root(|| {
            force_misconfigured_for_test(true);
            let result = crate::daemon::batch::dispatch_tool_batch_run(&serde_json::json!({
                "commands": [{"command": "echo", "args": ["hi"]}],
            }));
            force_misconfigured_for_test(false);

            let arr = result
                .expect("dispatch_tool_batch_run only errors on malformed params")
                .as_array()
                .unwrap()
                .clone();
            assert_eq!(arr.len(), 1);
            let state = arr[0]["state"].as_str().unwrap();
            assert!(
                state.starts_with("failed:") && state.contains(REFUSAL_TEXT),
                "expected a failed state carrying the refusal text, got: {state}"
            );
        });
    }

    fn list_rs_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                list_rs_files(&path, out);
            } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }

    /// Static half of the fail-closed proof: the process-creation call this
    /// seam guards must appear in exactly one PRODUCTION file. Scans only
    /// the part of each file before its own `mod tests` block (this
    /// repo's uniform convention: the test module is always the trailing
    /// item) -- a test-only subprocess spawn (e.g. `state.rs`'s exit-code
    /// fixture) never touches this seam's registry/artifact machinery and
    /// is not part of the surface this test is proving. If a future
    /// change spawns a child anywhere else in PRODUCTION code, it
    /// bypasses the seam and this test catches it, rather than relying on
    /// every author remembering to route through `spawn_bounded`.
    #[test]
    fn command_new_is_confined_to_spawn_rs() {
        // Built via concat! so this literal does not match itself in a
        // naive scan; excluding each file's `mod tests` block (below)
        // additionally keeps this very panic message out of its own scan.
        let needle = concat!("Command", "::new(");
        let src_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");

        let mut files = Vec::new();
        list_rs_files(&src_root, &mut files);
        assert!(
            !files.is_empty(),
            "guard test found no .rs files -- src_root resolution is broken: {src_root:?}"
        );

        let mut offenders = Vec::new();
        for path in &files {
            let text = std::fs::read_to_string(path).unwrap();
            let production_text = match text.find("mod tests") {
                Some(idx) => &text[..idx],
                None => &text[..],
            };
            if production_text.contains(needle) {
                let rel = path.strip_prefix(&src_root).unwrap().to_path_buf();
                if rel != std::path::Path::new("exec/spawn.rs") {
                    offenders.push(rel);
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "the process-spawn call must appear only in exec/spawn.rs, also found in: {offenders:?}"
        );
    }
}

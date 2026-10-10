//! Command executor core: spawns a command, captures stdout/stderr under an
//! artifact root (owner-only `0600` on unix, unrestricted elsewhere -- see
//! `artifact.rs`), and re-serves a run's result from disk without
//! re-running it. `spawn_run` calls `sandbox::ensure_sandbox_ready` first --
//! the fail-closed seam sits on the one code path every run takes.

pub mod artifact;
#[cfg(test)]
mod guard;
pub mod rlimit;
pub mod sandbox;
pub mod spawn;
pub mod state;
pub mod tail;
pub mod watcher;

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;

#[derive(Debug, thiserror::Error)]
pub enum ExecError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("sandbox requested but unavailable: {0}")]
    SandboxRefused(String),
}

pub struct SpawnSpec {
    pub program: String,
    pub args: Vec<String>,
    /// Per-run override of `WatchConfig::timeout`; `None` keeps the
    /// watcher's own default.
    pub timeout: Option<Duration>,
}

pub struct RunHandle {
    pub run_id: String,
    pub run_state: state::RunState,
    pub result: BoundedResult,
}

#[derive(Debug, Clone, Default)]
pub struct BoundedResult {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub stdout_offset: u64,
    pub stderr_offset: u64,
    pub elided_bytes: u64,
    pub truncated: bool,
}

/// Spawns `spec`, waits for exit, and returns the run's id plus its first
/// bounded output window. The sandbox check runs FIRST -- a misconfigured
/// seam refuses the spawn rather than falling through unsandboxed.
pub fn spawn_run(spec: SpawnSpec) -> Result<RunHandle, ExecError> {
    sandbox::ensure_sandbox_ready()?;

    let run_id = artifact::new_run_id();
    let dir = artifact::run_dir(&run_id)?;
    let (stdout_file, _stdout_path) = artifact::create_artifact_file(&dir, "stdout.log")?;
    let (stderr_file, _stderr_path) = artifact::create_artifact_file(&dir, "stderr.log")?;

    let started_at = unix_nanos_now();
    let limits = rlimit::effective_limits();
    let mut spawned = spawn::spawn_bounded(
        &spec.program,
        &spec.args,
        stdout_file,
        stderr_file,
        &limits,
    )?;
    let group = spawned.group;
    let watch_config = watch_config_for(spec.timeout);
    let run_state = watcher::watch(&mut spawned.child, group, &watch_config)?;
    let exit_code = match run_state {
        state::RunState::Exited(code) => Some(code),
        _ => None,
    };

    let meta = artifact::RunMeta {
        run_id: run_id.clone(),
        cmd: spec.program,
        args: spec.args,
        pid: spawned.child.id(),
        pgid: group.meta_pgid(),
        started_at,
        exit_code,
        stdout_offset: 0,
        stderr_offset: 0,
        rlimits_applied: cfg!(unix),
    };
    artifact::write_meta(&dir, &meta)?;

    let result = fetch_from_dir(&dir, &meta, &tail::TailBound::default_bound())?;
    Ok(RunHandle {
        run_id,
        run_state,
        result,
    })
}

/// Re-reads a prior run's artifact from the offset its own last fetch left
/// off, WITHOUT re-running the command -- a read against files on disk,
/// never a second spawn.
pub fn fetch_result(run_id: &str) -> Result<BoundedResult, ExecError> {
    fetch_result_bounded(run_id, tail::TailBound::default_bound())
}

/// Same as `fetch_result` with a caller-supplied bound instead of the
/// default. The read-modify-write of the persisted offset is serialized per
/// `run_id` via `fetch_lock_for` -- without this, two concurrent fetches on
/// the same run_id (the daemon's own thread-per-connection design invites
/// exactly this) can both read the same stale offset and duplicate-serve or
/// regress bytes.
pub fn fetch_result_bounded(run_id: &str, bound: tail::TailBound) -> Result<BoundedResult, ExecError> {
    let lock = fetch_lock_for(run_id);
    let _guard = lock.lock();
    let dir = artifact::run_dir_path(run_id)?;
    let meta = artifact::read_meta(&dir)?;
    fetch_from_dir(&dir, &meta, &bound)
}

/// Per-`run_id` mutex table, process-global and never shrunk -- one entry
/// per run_id ever fetched for the life of the process, cheap at this
/// crate's scale (a `Mutex<()>` per run, not per byte). Never acquired by
/// `spawn_run`'s own first fetch: the run_id is not yet observable to any
/// other caller at that point, so there is nothing to race against.
fn fetch_lock_for(run_id: &str) -> Arc<Mutex<()>> {
    static LOCKS: OnceLock<Mutex<HashMap<String, Arc<Mutex<()>>>>> = OnceLock::new();
    let table = LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut table = table.lock();
    table
        .entry(run_id.to_string())
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

/// Applies a per-run `SpawnSpec::timeout` override on top of the watcher's
/// own default grace window.
fn watch_config_for(timeout: Option<Duration>) -> watcher::WatchConfig {
    let mut config = watcher::WatchConfig::default();
    if let Some(t) = timeout {
        config.timeout = t;
    }
    config
}

fn fetch_from_dir(
    dir: &Path,
    meta: &artifact::RunMeta,
    bound: &tail::TailBound,
) -> Result<BoundedResult, ExecError> {
    let stdout_window = tail::read_bounded(&dir.join("stdout.log"), meta.stdout_offset, bound)?;
    let stderr_window = tail::read_bounded(&dir.join("stderr.log"), meta.stderr_offset, bound)?;

    // Persist the advanced offset so the NEXT fetch continues from here --
    // this is what makes result_fetch a re-filter, never a re-run.
    let mut updated = meta.clone();
    updated.stdout_offset = stdout_window.new_offset;
    updated.stderr_offset = stderr_window.new_offset;
    artifact::write_meta(dir, &updated)?;

    Ok(BoundedResult {
        stdout: stdout_window.data,
        stderr: stderr_window.data,
        stdout_offset: stdout_window.new_offset,
        stderr_offset: stderr_window.new_offset,
        elided_bytes: stdout_window.elided_bytes + stderr_window.elided_bytes,
        truncated: stdout_window.elided_bytes > 0 || stderr_window.elided_bytes > 0,
    })
}

fn unix_nanos_now() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::artifact::ENV_LOCK;

    fn with_tools_root<T>(f: impl FnOnce() -> T) -> T {
        let _guard = ENV_LOCK.lock().unwrap();
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

    #[test]
    fn spawn_run_creates_artifacts_and_returns_output() {
        with_tools_root(|| {
            let handle = spawn_run(SpawnSpec {
                program: "echo".into(),
                args: vec!["hello".into()],
                timeout: None,
            })
            .unwrap();
            assert_eq!(String::from_utf8_lossy(&handle.result.stdout).trim(), "hello");
        });
    }

    #[test]
    fn fetch_result_does_not_rerun_the_command() {
        with_tools_root(|| {
            // date +%s%N prints a nanosecond timestamp -- a re-run would
            // print a DIFFERENT value, so the on-disk artifact staying at
            // exactly one such value proves the child ran exactly once.
            let handle = spawn_run(SpawnSpec {
                program: "date".into(),
                args: vec!["+%s%N".into()],
                timeout: None,
            })
            .unwrap();
            let nonce = String::from_utf8_lossy(&handle.result.stdout)
                .trim()
                .to_string();
            assert!(!nonce.is_empty());

            // spawn_run's own initial fetch already advanced the offset past
            // the single line the child printed -- both explicit fetches
            // below must see nothing NEW, proving no second execution
            // appended (or overwrote with) a second value.
            let first_fetch = fetch_result(&handle.run_id).unwrap();
            assert!(first_fetch.stdout.is_empty());
            let second_fetch = fetch_result(&handle.run_id).unwrap();
            assert!(second_fetch.stdout.is_empty());

            let dir = artifact::run_dir_path(&handle.run_id).unwrap();
            let on_disk = std::fs::read_to_string(dir.join("stdout.log")).unwrap();
            assert_eq!(on_disk.trim(), nonce);
        });
    }

    #[test]
    fn big_output_is_tail_bounded_with_elision_marker() {
        with_tools_root(|| {
            let handle = spawn_run(SpawnSpec {
                program: "seq".into(),
                args: vec!["1".into(), "100000".into()],
                timeout: None,
            })
            .unwrap();
            assert!(
                handle.result.truncated,
                "100000-line output must be truncated by the tail bound"
            );
            assert!(handle.result.elided_bytes > 0);

            let dir = artifact::run_dir_path(&handle.run_id).unwrap();
            let full = std::fs::read_to_string(dir.join("stdout.log")).unwrap();
            assert_eq!(full.lines().count(), 100_000, "full content stays on disk");
        });
    }

    #[test]
    fn fetch_result_rejects_relative_traversal_run_id() {
        with_tools_root(|| {
            let result = fetch_result("../../../../etc");
            assert!(result.is_err(), "fetch_result must reject a traversal run_id");
        });
    }

    #[test]
    fn fetch_result_rejects_absolute_path_run_id() {
        with_tools_root(|| {
            let result = fetch_result("/etc");
            assert!(result.is_err(), "fetch_result must reject an absolute-path run_id");
        });
    }

    #[test]
    fn concurrent_fetches_on_the_same_run_id_never_duplicate_or_drop_bytes() {
        with_tools_root(|| {
            // Manually built artifact (no real child) so the exact byte
            // content and length are known up front.
            let run_id = "aa-bb-01";
            let dir = artifact::run_dir(run_id).unwrap();
            let content: Vec<u8> = (0..200u32).map(|i| (i % 10) as u8 + b'0').collect();
            std::fs::write(dir.join("stdout.log"), &content).unwrap();
            std::fs::write(dir.join("stderr.log"), b"").unwrap();
            artifact::write_meta(
                &dir,
                &artifact::RunMeta {
                    run_id: run_id.to_string(),
                    cmd: "true".into(),
                    args: vec![],
                    pid: 1,
                    pgid: 1,
                    started_at: 0,
                    exit_code: Some(0),
                    stdout_offset: 0,
                    stderr_offset: 0,
                    rlimits_applied: true,
                },
            )
            .unwrap();

            // Each of N threads fetches exactly once with a small, fixed
            // bound, all racing against the SAME run_id -- the per-run_id
            // lock in `fetch_lock_for` must serialize the read-modify-write
            // of the persisted offset so every byte is served to exactly
            // one caller.
            let bound = tail::TailBound { byte_cap: 20, line_cap: 0 };
            let n = (content.len() / bound.byte_cap as usize) + 2;
            let barrier = std::sync::Arc::new(std::sync::Barrier::new(n));
            let handles: Vec<_> = (0..n)
                .map(|_| {
                    let barrier = std::sync::Arc::clone(&barrier);
                    let bound = bound.clone();
                    std::thread::spawn(move || {
                        barrier.wait();
                        fetch_result_bounded(run_id, bound).unwrap()
                    })
                })
                .collect();
            let mut windows: Vec<(u64, Vec<u8>)> = handles
                .into_iter()
                .map(|h| {
                    let result = h.join().unwrap();
                    (result.stdout_offset, result.stdout)
                })
                .filter(|(_, data)| !data.is_empty())
                .collect();
            windows.sort_by_key(|(end_offset, _)| *end_offset);

            let mut reconstructed = Vec::new();
            let mut expected_start = 0u64;
            for (end_offset, data) in &windows {
                let start = end_offset - data.len() as u64;
                assert_eq!(
                    start, expected_start,
                    "windows must tile [0, total) with no gap or overlap"
                );
                reconstructed.extend_from_slice(data);
                expected_start = *end_offset;
            }
            assert_eq!(
                reconstructed, content,
                "the union of every concurrent fetch's window must equal the full content exactly once"
            );

            let final_meta = artifact::read_meta(&dir).unwrap();
            assert_eq!(final_meta.stdout_offset, content.len() as u64);
        });
    }

    #[test]
    fn spawn_run_refuses_when_sandbox_seam_misconfigured() {
        with_tools_root(|| {
            sandbox::force_misconfigured_for_test(true);
            let result = spawn_run(SpawnSpec {
                program: "echo".into(),
                args: vec!["hi".into()],
                timeout: None,
            });
            sandbox::force_misconfigured_for_test(false);
            assert!(
                result.is_err(),
                "spawn_run must refuse when the sandbox seam is misconfigured"
            );
        });
    }
}

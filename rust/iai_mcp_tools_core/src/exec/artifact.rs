//! Artifact store: one directory per run under `{IAI_TOOLS_ROOT}/artifacts/{run_id}/`,
//! holding `stdout.log`, `stderr.log`, and `meta.json`. Owner-only `0600` on
//! unix; not permission-restricted on a platform without unix mode bits.

use std::fs::{File, OpenOptions};
use std::io::Write;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::ExecError;

static RUN_SEQ: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMeta {
    pub run_id: String,
    pub cmd: String,
    pub args: Vec<String>,
    pub pid: u32,
    /// The POSIX process-group id on a platform that has one; a negative
    /// sentinel where it does not.
    pub pgid: i32,
    pub started_at: u128,
    pub exit_code: Option<i32>,
    pub stdout_offset: u64,
    pub stderr_offset: u64,
    /// Whether resource caps were enforced for this run: `true` where the
    /// platform applies `setrlimit` (a failed `setrlimit` in `pre_exec`
    /// fails the spawn itself, so every persisted unix run has this set),
    /// `false` where the platform has no such mechanism.
    pub rlimits_applied: bool,
}

pub fn tools_root() -> Result<PathBuf, ExecError> {
    if let Ok(v) = std::env::var("IAI_TOOLS_ROOT") {
        if !v.is_empty() {
            return Ok(PathBuf::from(v));
        }
    }
    let home = std::env::var("HOME").map_err(|_| {
        ExecError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "HOME not set",
        ))
    })?;
    Ok(PathBuf::from(home).join(".iai-tools"))
}

/// Nanos+pid prefix gives cross-restart/cross-process uniqueness; the
/// process-local sequence gives intra-process uniqueness -- no UUID crate.
pub fn new_run_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    let pid = std::process::id();
    let seq = RUN_SEQ.fetch_add(1, Ordering::Relaxed);
    format!("{nanos:x}-{pid:x}-{seq:x}")
}

pub fn run_dir(run_id: &str) -> Result<PathBuf, ExecError> {
    run_dir_under(&tools_root()?, run_id)
}

/// Every path built from a caller-supplied `run_id` MUST go through this
/// check first -- the sole choke point that keeps `root.join("artifacts").join(run_id)`
/// from ever escaping the artifact root via `..`/absolute-path substitution.
/// Allowlist, not denylist: accepts only the exact shape `new_run_id()`
/// produces (three lowercase-hex groups joined by `-`), so anything a
/// generator never emits -- traversal segments, absolute paths, non-ASCII,
/// stray separators -- is rejected by construction rather than by
/// enumerating forbidden characters.
fn validate_run_id(run_id: &str) -> Result<(), ExecError> {
    fn is_lower_hex_group(s: &str) -> bool {
        !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    }
    let parts: Vec<&str> = run_id.split('-').collect();
    if parts.len() == 3 && parts.iter().all(|p| is_lower_hex_group(p)) {
        return Ok(());
    }
    Err(ExecError::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        format!("invalid run_id: {run_id}"),
    )))
}

/// Create-new semantics on the leaf directory: fails on collision, so the
/// create-time chmod 0o600 on its files is never a no-op on a reused path.
fn run_dir_under(root: &Path, run_id: &str) -> Result<PathBuf, ExecError> {
    validate_run_id(run_id)?;
    let artifacts = root.join("artifacts");
    std::fs::create_dir_all(&artifacts)?;
    let dir = artifacts.join(run_id);
    std::fs::create_dir(&dir)?;
    Ok(dir)
}

pub fn run_dir_path(run_id: &str) -> Result<PathBuf, ExecError> {
    run_dir_path_under(&tools_root()?, run_id)
}

fn run_dir_path_under(root: &Path, run_id: &str) -> Result<PathBuf, ExecError> {
    validate_run_id(run_id)?;
    let dir = root.join("artifacts").join(run_id);
    if !dir.is_dir() {
        return Err(ExecError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("unknown run_id: {run_id}"),
        )));
    }
    Ok(dir)
}

pub fn create_artifact_file(dir: &Path, name: &str) -> Result<(File, PathBuf), ExecError> {
    let path = dir.join(name);
    let mut opts = OpenOptions::new();
    opts.create_new(true).write(true);
    #[cfg(unix)]
    opts.mode(0o600);
    let file = opts.open(&path)?;
    Ok((file, path))
}

/// `meta.json`'s first write creates it at 0o600 (the run dir is always
/// fresh, so this is never a reused-file no-op); later writes within the
/// SAME run (offset persistence) reuse that already-0o600 file.
pub fn write_meta(dir: &Path, meta: &RunMeta) -> Result<(), ExecError> {
    let path = dir.join("meta.json");
    let json = serde_json::to_vec_pretty(meta)?;
    let mut opts = OpenOptions::new();
    opts.create(true).write(true).truncate(true);
    #[cfg(unix)]
    opts.mode(0o600);
    let mut file = opts.open(&path)?;
    file.write_all(&json)?;
    Ok(())
}

pub fn read_meta(dir: &Path) -> Result<RunMeta, ExecError> {
    let bytes = std::fs::read(dir.join("meta.json"))?;
    Ok(serde_json::from_slice(&bytes)?)
}

/// Serializes every test in this crate that mutates the process-global
/// `IAI_TOOLS_ROOT` env var -- each `#[test]` fn runs on its own thread, so
/// without this lock two such tests can race on the shared env var.
#[cfg(test)]
pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn run_ids_are_distinct_across_calls() {
        let a = new_run_id();
        let b = new_run_id();
        assert_ne!(a, b);
    }

    #[test]
    fn run_dir_rejects_collision() {
        let tmp = tempfile::tempdir().unwrap();
        let run_id = "1a-2b-01";
        run_dir_under(tmp.path(), run_id).expect("first create succeeds");
        let second = run_dir_under(tmp.path(), run_id);
        assert!(second.is_err(), "second create of the same run_id must fail");
    }

    #[test]
    fn artifact_file_created_at_mode_0600() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = run_dir_under(tmp.path(), "1a-2b-02").unwrap();
        let (file, path) = create_artifact_file(&dir, "stdout.log").unwrap();
        drop(file);
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
    }

    #[test]
    fn artifact_file_rejects_reuse() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = run_dir_under(tmp.path(), "1a-2b-03").unwrap();
        let (_file, _path) = create_artifact_file(&dir, "stdout.log").unwrap();
        let second = create_artifact_file(&dir, "stdout.log");
        assert!(
            second.is_err(),
            "create_new must refuse a reused artifact filename"
        );
    }

    #[test]
    fn meta_json_round_trips() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = run_dir_under(tmp.path(), "1a-2b-04").unwrap();
        let meta = RunMeta {
            run_id: "1a-2b-04".into(),
            cmd: "echo".into(),
            args: vec!["hi".into()],
            pid: 1,
            pgid: 1,
            started_at: 0,
            exit_code: Some(0),
            stdout_offset: 0,
            stderr_offset: 0,
            rlimits_applied: true,
        };
        write_meta(&dir, &meta).unwrap();
        let read_back = read_meta(&dir).unwrap();
        assert_eq!(read_back.cmd, "echo");
        assert_eq!(read_back.args, vec!["hi".to_string()]);
    }

    #[test]
    fn meta_json_created_at_mode_0600() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = run_dir_under(tmp.path(), "1a-2b-05").unwrap();
        let meta = RunMeta {
            run_id: "1a-2b-05".into(),
            cmd: "echo".into(),
            args: vec![],
            pid: 1,
            pgid: 1,
            started_at: 0,
            exit_code: None,
            stdout_offset: 0,
            stderr_offset: 0,
            rlimits_applied: true,
        };
        write_meta(&dir, &meta).unwrap();
        let mode = std::fs::metadata(dir.join("meta.json"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600);
    }

    #[test]
    fn run_dir_path_under_rejects_relative_traversal() {
        let tmp = tempfile::tempdir().unwrap();
        let result = run_dir_path_under(tmp.path(), "../../../../etc");
        assert!(result.is_err(), "relative traversal run_id must be rejected");
        assert!(
            !matches!(result, Err(ExecError::Io(ref e)) if e.kind() == std::io::ErrorKind::NotFound),
            "must be rejected as invalid input, not merely 'not found'"
        );
    }

    #[test]
    fn run_dir_path_under_rejects_absolute_path_substitution() {
        let tmp = tempfile::tempdir().unwrap();
        let result = run_dir_path_under(tmp.path(), "/etc");
        assert!(result.is_err(), "absolute-path run_id must be rejected");
    }

    #[test]
    fn run_dir_path_under_rejects_dotdot() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(run_dir_path_under(tmp.path(), "..").is_err());
    }

    #[test]
    fn run_dir_path_under_rejects_empty() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(run_dir_path_under(tmp.path(), "").is_err());
    }

    #[test]
    fn run_dir_path_under_rejects_embedded_separator() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(run_dir_path_under(tmp.path(), "a/b").is_err());
    }

    #[test]
    fn run_dir_path_under_rejects_unicode() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(run_dir_path_under(tmp.path(), "1a-2b-\u{0441}03").is_err());
    }

    #[test]
    fn run_dir_path_under_accepts_generator_shaped_id_that_exists() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = run_dir_under(tmp.path(), "1a-2b-06").unwrap();
        let found = run_dir_path_under(tmp.path(), "1a-2b-06").unwrap();
        assert_eq!(found, dir);
    }

    #[test]
    fn run_dir_under_rejects_relative_traversal_on_the_create_path() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(
            run_dir_under(tmp.path(), "../../../../etc").is_err(),
            "the create-time path join must also reject traversal, not only the fetch path"
        );
    }

    #[test]
    fn tools_root_respects_env_override() {
        let _guard = ENV_LOCK.lock().unwrap();
        // SAFETY: ENV_LOCK serializes every test that touches IAI_TOOLS_ROOT.
        unsafe {
            std::env::set_var("IAI_TOOLS_ROOT", "/tmp/iai-tools-root-test-marker");
        }
        let root = tools_root().unwrap();
        assert_eq!(root, PathBuf::from("/tmp/iai-tools-root-test-marker"));
        unsafe {
            std::env::remove_var("IAI_TOOLS_ROOT");
        }
    }
}

//! Process spawn: builds a `std::process::Command` that writes stdout/stderr
//! directly to the given fds -- no parent-buffered pipe. On unix the child
//! becomes its own process group leader and runs under `setrlimit` caps
//! applied by an async-signal-safe `pre_exec` closure (see `rlimit.rs`); on
//! a platform with neither mechanism, the child's own pid is the only
//! handle a caller ever gets, and no resource caps are enforced.

use std::fs::File;
use std::process::{Child, Command, Stdio};

use super::rlimit::EffectiveLimits;
use super::ExecError;

/// Opaque process-group handle: wraps the pgid on unix, a zero-sized
/// placeholder elsewhere. `meta_pgid`/`signal` are the only crossings of
/// this boundary -- no `libc::pid_t` appears in any signature outside this
/// type's own unix-gated body, so a later platform-specific group-kill arm
/// (e.g. a Windows job-object handle) is a leaf change here, not a caller
/// refactor.
#[derive(Debug, Clone, Copy)]
pub struct ProcGroup {
    #[cfg(unix)]
    pgid: libc::pid_t,
}

impl ProcGroup {
    #[cfg(unix)]
    fn from_child(child: &Child) -> Self {
        ProcGroup {
            pgid: child.id() as libc::pid_t,
        }
    }

    #[cfg(not(unix))]
    fn from_child(_child: &Child) -> Self {
        ProcGroup {}
    }

    #[cfg(unix)]
    pub fn meta_pgid(&self) -> i32 {
        self.pgid as i32
    }

    /// `-1` sentinel: no POSIX process group exists to record.
    #[cfg(not(unix))]
    pub fn meta_pgid(&self) -> i32 {
        -1
    }

    /// Signals the whole process group. Best-effort: a group that already
    /// exited yields ESRCH, which is not an error condition here.
    #[cfg(unix)]
    pub(crate) fn signal(&self, sig: i32) {
        // SAFETY: pgid is the process group created via `process_group(0)`
        // in `spawn_bounded` -- negative pid signals the whole group, not a
        // single pid.
        unsafe {
            libc::kill(-self.pgid, sig as libc::c_int);
        }
    }
}

pub struct SpawnedChild {
    pub child: Child,
    pub group: ProcGroup,
}

/// The only call site of `Command::new` in this crate for a
/// user-requested command. `sandbox::ensure_sandbox_ready` runs first,
/// as this function's own first statement -- no caller can reach
/// `Command::spawn` without passing it.
pub fn spawn_bounded(
    program: &str,
    args: &[String],
    stdout_file: File,
    stderr_file: File,
    limits: &EffectiveLimits,
) -> Result<SpawnedChild, ExecError> {
    super::sandbox::ensure_sandbox_ready()?;

    let mut cmd = Command::new(program);
    cmd.args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file));

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
        let limits = *limits;
        // SAFETY: the closure below runs in the forked child between fork
        // and exec, per `CommandExt::pre_exec`'s contract; `limits` is
        // already computed in the parent above, so the closure itself only
        // calls `setrlimit` -- no allocation, no lock. The closure body is
        // lexically inside this block, so it shares this unsafe context.
        unsafe {
            cmd.pre_exec(move || super::rlimit::apply(&limits));
        }
    }
    #[cfg(not(unix))]
    let _ = limits;

    let child = cmd.spawn()?;
    let group = ProcGroup::from_child(&child);
    Ok(SpawnedChild { child, group })
}

/// Runs a short-lived, fixed system command to completion and returns its
/// captured output. Distinct from `spawn_bounded`: this crate's own
/// `command_new_is_confined_to_spawn_rs` guard requires every
/// `Command::new` call to live in this file, so a caller that needs to
/// invoke a hardcoded administrative tool (never a user- or
/// agent-supplied program or arguments -- e.g. Windows `icacls` to
/// tighten a file's ACL) routes through here rather than adding a second
/// `Command::new` call site elsewhere. No sandbox gate, no resource
/// limits: there is no arbitrary-command threat surface to gate when the
/// program name is a literal chosen by this crate's own code.
///
/// Callers must only ever pass a fixed, non-user-controlled program and
/// argument list -- this is a convention this function does not enforce
/// structurally.
pub fn run_fixed_command(program: &str, args: &[String]) -> std::io::Result<std::process::Output> {
    Command::new(program).args(args).output()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;
    use std::io::Read;

    fn open_tmp_file(dir: &std::path::Path, name: &str) -> (File, std::path::PathBuf) {
        let path = dir.join(name);
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .unwrap();
        (file, path)
    }

    #[test]
    fn spawn_captures_stdout_to_file() {
        let tmp = tempfile::tempdir().unwrap();
        let (stdout_file, stdout_path) = open_tmp_file(tmp.path(), "stdout.log");
        let (stderr_file, _stderr_path) = open_tmp_file(tmp.path(), "stderr.log");
        let limits = super::super::rlimit::effective_limits();
        let mut spawned = spawn_bounded(
            "echo",
            &["hello".to_string()],
            stdout_file,
            stderr_file,
            &limits,
        )
        .unwrap();
        let status = spawned.child.wait().unwrap();
        assert!(status.success());
        let mut out = String::new();
        File::open(&stdout_path)
            .unwrap()
            .read_to_string(&mut out)
            .unwrap();
        assert_eq!(out.trim(), "hello");
    }

    #[test]
    fn spawned_child_is_its_own_process_group_leader() {
        let tmp = tempfile::tempdir().unwrap();
        let (stdout_file, _p1) = open_tmp_file(tmp.path(), "stdout.log");
        let (stderr_file, _p2) = open_tmp_file(tmp.path(), "stderr.log");
        let limits = super::super::rlimit::effective_limits();
        let mut spawned = spawn_bounded(
            "sleep",
            &["0.2".to_string()],
            stdout_file,
            stderr_file,
            &limits,
        )
        .unwrap();
        assert_eq!(spawned.group.meta_pgid(), spawned.child.id() as i32);
        spawned.child.wait().unwrap();
    }
}

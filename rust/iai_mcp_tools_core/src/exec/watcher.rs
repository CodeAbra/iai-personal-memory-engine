//! Process-group timeout watcher: escalates SIGTERM -> grace -> SIGKILL to
//! the child's whole process group so a forking child cannot orphan a
//! grandchild past the deadline. Reaps the child on every terminal path --
//! `std::process::Child` never reaps on Drop. On a platform with no process
//! group, only the direct child is killed at the deadline; a forking
//! grandchild can outlive it.

use std::process::Child;
use std::time::{Duration, Instant};

use super::spawn::ProcGroup;
use super::state::RunState;

/// Wall-clock backstop for a run before escalation begins.
pub const DEFAULT_TIMEOUT_SECS: u64 = 300;
/// Window between SIGTERM and SIGKILL to the process group.
pub const GRACE_SECS: u64 = 5;

/// Terminal-state signal number for a deadline-driven kill.
#[cfg(unix)]
pub const KILL_SIGNAL: i32 = libc::SIGKILL;
/// No POSIX signal concept on this platform; sentinel value only.
#[cfg(not(unix))]
pub const KILL_SIGNAL: i32 = -1;

const POLL_INTERVAL: Duration = Duration::from_millis(5);

#[derive(Debug, Clone, Copy)]
pub struct WatchConfig {
    pub timeout: Duration,
    pub grace: Duration,
}

impl Default for WatchConfig {
    fn default() -> Self {
        WatchConfig {
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            grace: Duration::from_secs(GRACE_SECS),
        }
    }
}

/// Polls `child` for exit until `config.timeout`. On deadline: on unix,
/// signals the whole process group with SIGTERM, then SIGKILL if it has not
/// exited by `config.grace`; on a platform with no process group, kills the
/// direct child only, with no grace phase. Every branch reaps exactly once
/// before returning.
pub fn watch(
    child: &mut Child,
    group: ProcGroup,
    config: &WatchConfig,
) -> std::io::Result<RunState> {
    if let Some(state) = poll_until(child, config.timeout)? {
        return Ok(state);
    }

    #[cfg(unix)]
    {
        group.signal(libc::SIGTERM);
        if poll_until(child, config.grace)?.is_some() {
            return Ok(RunState::TimedOut);
        }
        group.signal(libc::SIGKILL);
        child.wait()?;
        Ok(RunState::Killed(KILL_SIGNAL))
    }
    #[cfg(not(unix))]
    {
        let _ = group;
        child.kill()?;
        child.wait()?;
        Ok(RunState::TimedOut)
    }
}

/// Polls `try_wait` until the child exits or `budget` elapses. A `Some`
/// return has already been reaped by `try_wait`; `None` means the budget
/// ran out with the child still running.
fn poll_until(child: &mut Child, budget: Duration) -> std::io::Result<Option<RunState>> {
    let deadline = Instant::now() + budget;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(Some(RunState::from_exit_status(status)));
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Ok(None);
        }
        std::thread::sleep(POLL_INTERVAL.min(remaining));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::spawn::{self, SpawnedChild};
    use std::fs::{File, OpenOptions};
    use std::path::Path;

    fn open_tmp_file(dir: &Path, name: &str) -> File {
        let path = dir.join(name);
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .unwrap()
    }

    /// Single choke point for spawning a test fixture -- isolates every
    /// test in this module from `spawn_bounded`'s exact parameter list.
    fn spawn_fixture(dir: &Path, program: &str, args: &[String]) -> SpawnedChild {
        let stdout_file = open_tmp_file(dir, "stdout.log");
        let stderr_file = open_tmp_file(dir, "stderr.log");
        let limits = crate::exec::rlimit::effective_limits();
        spawn::spawn_bounded(program, args, stdout_file, stderr_file, &limits).unwrap()
    }

    fn pid_is_alive(pid: libc::pid_t) -> bool {
        // SAFETY: signal 0 is the standard existence probe; no signal sent.
        unsafe { libc::kill(pid, 0) == 0 }
    }

    fn poll_for(budget: Duration, mut cond: impl FnMut() -> bool) -> bool {
        let deadline = Instant::now() + budget;
        loop {
            if cond() {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn wait_for_pidfile(path: &Path) -> libc::pid_t {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if let Ok(contents) = std::fs::read_to_string(path) {
                if let Ok(pid) = contents.trim().parse::<libc::pid_t>() {
                    if pid > 0 {
                        return pid;
                    }
                }
            }
            assert!(Instant::now() < deadline, "grandchild pidfile never appeared");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn timeout_escalates_sigterm_then_sigkill_over_the_group() {
        let tmp = tempfile::tempdir().unwrap();
        // Ignoring SIGTERM forces the watcher past the grace window into
        // the SIGKILL branch -- SIG_IGN survives exec, so this covers the
        // child whether or not the shell tail-call-execs into `sleep`.
        let mut spawned = spawn_fixture(
            tmp.path(),
            "sh",
            &["-c".into(), "trap '' TERM; sleep 5".into()],
        );
        let group = spawned.group;
        let pid = spawned.child.id() as libc::pid_t;

        let config = WatchConfig {
            timeout: Duration::from_millis(150),
            grace: Duration::from_millis(150),
        };
        let state = watch(&mut spawned.child, group, &config).unwrap();

        assert_eq!(state, RunState::Killed(libc::SIGKILL));
        assert!(!pid_is_alive(pid), "child must be gone after SIGKILL to the group");
    }

    #[test]
    fn exit_within_grace_window_classifies_as_timed_out() {
        let tmp = tempfile::tempdir().unwrap();
        let mut spawned = spawn_fixture(tmp.path(), "sleep", &["5".into()]);
        let group = spawned.group;

        let config = WatchConfig {
            timeout: Duration::from_millis(150),
            grace: Duration::from_millis(800),
        };
        let state = watch(&mut spawned.child, group, &config).unwrap();

        assert_eq!(state, RunState::TimedOut);
    }

    #[test]
    fn normal_exit_before_the_deadline_yields_exited() {
        let tmp = tempfile::tempdir().unwrap();
        let mut spawned = spawn_fixture(tmp.path(), "sh", &["-c".into(), "exit 3".into()]);
        let group = spawned.group;

        let state = watch(&mut spawned.child, group, &WatchConfig::default()).unwrap();

        assert_eq!(state, RunState::Exited(3));
    }

    #[test]
    #[cfg(unix)]
    fn forked_grandchild_is_gone_after_the_group_kill() {
        let tmp = tempfile::tempdir().unwrap();
        let pidfile = tmp.path().join("grandchild.pid");
        let script = format!("sleep 5 & echo $! > {}; wait", pidfile.display());
        let mut spawned = spawn_fixture(tmp.path(), "sh", &["-c".into(), script]);
        let group = spawned.group;

        let grandchild_pid = wait_for_pidfile(&pidfile);
        assert!(
            pid_is_alive(grandchild_pid),
            "test setup: grandchild must be running before the kill"
        );

        let config = WatchConfig {
            timeout: Duration::from_millis(150),
            grace: Duration::from_millis(150),
        };
        let state = watch(&mut spawned.child, group, &config).unwrap();
        assert!(matches!(state, RunState::TimedOut | RunState::Killed(_)));

        // The kernel may take a moment to reparent/reap the orphaned
        // grandchild after the group signal; poll briefly rather than
        // asserting instantaneously.
        let gone = poll_for(Duration::from_secs(2), || !pid_is_alive(grandchild_pid));
        assert!(gone, "grandchild must not be running after the process-group kill");
    }

    #[test]
    fn no_zombie_leaks_across_a_run_sequence() {
        let tmp = tempfile::tempdir().unwrap();
        for _ in 0..20 {
            let mut spawned = spawn_fixture(tmp.path(), "true", &[]);
            let pid = spawned.child.id() as libc::pid_t;
            let group = spawned.group;

            let state = watch(&mut spawned.child, group, &WatchConfig::default()).unwrap();
            assert_eq!(state, RunState::Exited(0));

            let gone = poll_for(Duration::from_millis(500), || !pid_is_alive(pid));
            assert!(gone, "a reaped child must leave no zombie entry behind");
        }
    }
}

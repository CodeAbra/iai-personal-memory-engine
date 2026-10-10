//! Terminal run-state classification for a watched command.

#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;
use std::process::ExitStatus;

/// A run's terminal-or-ownership state. `Running`/`Exited`/`Killed`/`TimedOut`
/// come from `watcher::watch`'s own reap of the child; `Detached` is set only
/// by the daemon's registry, never by the watcher, when the requesting
/// connection stops watching a still-running background run -- ownership was
/// already with the daemon's worker thread, this only records that nobody is
/// watching anymore. Liveness is always this explicit field, never inferred
/// from an artifact file's mtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunState {
    Running,
    Exited(i32),
    Killed(i32),
    TimedOut,
    Detached,
}

impl RunState {
    /// Classifies an already-reaped `ExitStatus`: a normal exit yields
    /// `Exited(code)`; a signal death yields `Killed(signal)` where the
    /// platform has POSIX signals. A code-less exit with no signal
    /// concept falls through to `Exited(-1)`.
    pub fn from_exit_status(status: ExitStatus) -> RunState {
        if let Some(code) = status.code() {
            return RunState::Exited(code);
        }
        #[cfg(unix)]
        if let Some(signal) = status.signal() {
            return RunState::Killed(signal);
        }
        RunState::Exited(-1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    #[test]
    fn normal_exit_classifies_as_exited() {
        let status = Command::new("sh").args(["-c", "exit 7"]).status().unwrap();
        assert_eq!(RunState::from_exit_status(status), RunState::Exited(7));
    }

    #[test]
    fn signal_death_classifies_as_killed() {
        let status = Command::new("sh")
            .args(["-c", "kill -KILL $$"])
            .status()
            .unwrap();
        assert_eq!(
            RunState::from_exit_status(status),
            RunState::Killed(libc::SIGKILL)
        );
    }

    #[test]
    fn detached_is_distinct_from_every_other_state() {
        let detached = RunState::Detached;
        assert_ne!(detached, RunState::Running);
        assert_ne!(detached, RunState::Exited(0));
        assert_ne!(detached, RunState::Killed(libc::SIGKILL));
        assert_ne!(detached, RunState::TimedOut);
        assert_eq!(detached, RunState::Detached);
    }
}

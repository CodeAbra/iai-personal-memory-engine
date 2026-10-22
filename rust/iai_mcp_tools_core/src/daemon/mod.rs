//! Resident executor daemon: lazy-started by the first client connect (the
//! caller execs this binary on demand -- nothing runs it ahead of time),
//! serves `tool_run`/`result_fetch` over its own Unix socket, and self-
//! terminates after an idle window so nothing lingers unbounded. Kept
//! deliberately simpler than the brain daemon: no FSM, no WAL-recoverable
//! pipeline -- a crash just means the next call lazy-restarts a fresh one.

pub mod batch;
pub mod socket;

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;

use crate::exec::state::RunState;
use crate::exec::{self, BoundedResult, ExecError, SpawnSpec};

/// Matches the brain daemon's `IDLE_SECS_DEFAULT` shape (seconds of no
/// activity before self-shutdown); `IAI_TOOLS_IDLE_SECS` overrides it.
pub const IDLE_SECS_DEFAULT: f64 = 1800.0;
const IDLE_POLL_INTERVAL: Duration = Duration::from_millis(20);

pub fn idle_secs_from_env() -> f64 {
    std::env::var("IAI_TOOLS_IDLE_SECS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(IDLE_SECS_DEFAULT)
}

/// Run registry: state persisted here survives across separate client
/// calls on the same daemon process -- one connection's `tool_run` result
/// is visible to a later, independent connection's `result_fetch` lookup.
pub struct Daemon {
    registry: Mutex<HashMap<String, RunState>>,
    last_activity: Mutex<Instant>,
    active_connections: AtomicUsize,
    /// Windows-only auth secret, written once by the process that wins the
    /// TCP bind (see `socket::start_tcp_transport`). Unix never sets this;
    /// `verify_auth_token` fails closed while it is unset.
    auth_token: std::sync::OnceLock<String>,
}

impl Daemon {
    pub fn new() -> Arc<Daemon> {
        Arc::new(Daemon {
            registry: Mutex::new(HashMap::new()),
            last_activity: Mutex::new(Instant::now()),
            active_connections: AtomicUsize::new(0),
            auth_token: std::sync::OnceLock::new(),
        })
    }

    /// Populates the auth-token cell once, at daemon startup. Platform-
    /// neutral -- not `#[cfg(windows)]` -- so `socket::start_tcp_transport`
    /// and its native tests can call it on any target; the `OnceLock`
    /// itself makes a second call a no-op rather than an error.
    pub fn set_auth_token(&self, token: String) {
        let _ = self.auth_token.set(token);
    }

    /// Constant-time comparison against the stored token; false when the
    /// cell is unset (the unix arm never sets it, so it fails closed).
    pub fn verify_auth_token(&self, candidate: &str) -> bool {
        match self.auth_token.get() {
            Some(expected) => socket::constant_time_eq(expected.as_bytes(), candidate.as_bytes()),
            None => false,
        }
    }

    pub fn record_activity(&self) {
        *self.last_activity.lock() = Instant::now();
    }

    pub fn connection_opened(&self) {
        self.active_connections.fetch_add(1, Ordering::SeqCst);
        self.record_activity();
    }

    pub fn connection_closed(&self) {
        self.active_connections.fetch_sub(1, Ordering::SeqCst);
        self.record_activity();
    }

    pub fn record_run(&self, run_id: String, state: RunState) {
        self.registry.lock().insert(run_id, state);
    }

    pub fn lookup_run(&self, run_id: &str) -> Option<RunState> {
        self.registry.lock().get(run_id).copied()
    }

    /// Transitions a still-`Running` entry to `Detached`. Ownership of the
    /// underlying child was already with the daemon's worker thread (it was
    /// spawned there, not on the caller's connection) -- this only records
    /// that the requesting connection stopped watching. An entry already in
    /// a terminal state is left untouched: the worker thread's own terminal
    /// write and this call race on the same registry lock, so whichever
    /// runs last simply wins, and a terminal write can never be downgraded
    /// back to `Detached` by a late-arriving disconnect.
    pub fn mark_detached_if_running(&self, run_id: &str) {
        let mut registry = self.registry.lock();
        if registry.get(run_id) == Some(&RunState::Running) {
            registry.insert(run_id.to_string(), RunState::Detached);
        }
    }

    /// Spawns `spec` on a daemon-owned worker thread and returns as soon as
    /// the child exists and its initial artifact meta is on disk -- never
    /// blocks on exit. Takes an explicit `&Arc<Daemon>` (rather than `&self`)
    /// because the worker thread outlives this call: it holds its own owning
    /// reference to the registry it will later write the terminal state
    /// into, so the run keeps making progress and stays recorded even if
    /// every client connection that ever knew about it has since closed.
    pub fn spawn_background(
        daemon: &Arc<Daemon>,
        spec: SpawnSpec,
    ) -> Result<(String, BoundedResult), ExecError> {
        exec::sandbox::ensure_sandbox_ready()?;

        let run_id = exec::artifact::new_run_id();
        let dir = exec::artifact::run_dir(&run_id)?;
        let (stdout_file, _stdout_path) = exec::artifact::create_artifact_file(&dir, "stdout.log")?;
        let (stderr_file, _stderr_path) = exec::artifact::create_artifact_file(&dir, "stderr.log")?;

        let started_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        let limits = exec::rlimit::effective_limits();
        let mut spawned =
            exec::spawn::spawn_bounded(&spec.program, &spec.args, stdout_file, stderr_file, &limits)?;
        let group = spawned.group;

        let meta = exec::artifact::RunMeta {
            run_id: run_id.clone(),
            cmd: spec.program.clone(),
            args: spec.args.clone(),
            pid: spawned.child.id(),
            pgid: group.meta_pgid(),
            started_at,
            exit_code: None,
            stdout_offset: 0,
            stderr_offset: 0,
            rlimits_applied: cfg!(unix),
        };
        exec::artifact::write_meta(&dir, &meta)?;
        daemon.record_run(run_id.clone(), RunState::Running);

        let mut watch_config = exec::watcher::WatchConfig::default();
        if let Some(t) = spec.timeout {
            watch_config.timeout = t;
        }
        let worker_daemon = Arc::clone(daemon);
        let worker_run_id = run_id.clone();
        std::thread::spawn(move || {
            let state = exec::watcher::watch(&mut spawned.child, group, &watch_config)
                // A watch-loop I/O error is vanishingly rare (waitpid
                // failing outright) -- fail closed to `Killed` rather than
                // leaving the registry stuck at `Running` forever with no
                // process actually alive to reap.
                .unwrap_or(RunState::Killed(exec::watcher::KILL_SIGNAL));
            worker_daemon.record_run(worker_run_id, state);
        });

        // The run has just started -- serve its bounded tail exactly like
        // any other fetch, immediately, without waiting for exit. A
        // still-empty artifact at this instant is a correct, ordinary
        // result, not a special case.
        let initial_tail = exec::fetch_result(&run_id)?;
        Ok((run_id, initial_tail))
    }

    /// True only when nothing is connected AND the idle window has
    /// elapsed since the last request/connection activity -- mirrors the
    /// brain daemon's `(now - last_activity > idle_secs) AND connections
    /// == 0` condition by shape.
    fn is_idle(&self, idle_secs: f64) -> bool {
        if self.active_connections.load(Ordering::SeqCst) != 0 {
            return false;
        }
        self.last_activity.lock().elapsed() >= Duration::from_secs_f64(idle_secs.max(0.0))
    }
}

/// Spawns a background thread that checks `Daemon::is_idle` on a fixed
/// poll timer (never a per-request cancellable task) and calls `on_idle`
/// exactly once, the first time the window elapses, then returns.
/// Production callers pass a process-exit callback; tests inject a
/// flag/channel so the real watcher loop runs without killing the test
/// process.
pub fn spawn_idle_watcher(
    daemon: Arc<Daemon>,
    idle_secs: f64,
    on_idle: impl Fn() + Send + 'static,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || loop {
        std::thread::sleep(IDLE_POLL_INTERVAL);
        if daemon.is_idle(idle_secs) {
            on_idle();
            return;
        }
    })
}

/// Unlinks `path` only if its directory-entry inode still matches
/// `bound_inode`: another process may have rebound the same path, and
/// removing the file would then take a live listener offline. The
/// inode snapshot means a sibling rebind that already happened before
/// this check is never removed. A residual stat/unlink race remains:
/// a rebind landing inside the check-to-remove window can still be
/// deleted -- accepted as negligible (microseconds vs. the idle-exit
/// window this guard closes).
fn unlink_if_still_ours(path: &Path, bound_inode: Option<u64>) {
    if bound_inode.is_some() && bound_inode == socket::socket_inode(path) {
        let _ = std::fs::remove_file(path);
    }
}

/// Binds the socket, spawns the idle watcher, and accepts connections
/// until the process exits. Idle self-termination is the only exit path
/// in this stage -- there is no explicit shutdown RPC. `socket::accept_loop`
/// owns both the open-bookkeeping-before-spawn ordering and transient
/// accept-error tolerance; see its own doc comment.
pub fn run() -> std::io::Result<()> {
    let daemon = Daemon::new();
    let path = socket::resolve_socket_path()
        .map_err(|e| std::io::Error::other(e.to_string()))?;

    #[cfg(windows)]
    {
        let listener = socket::bind_listener(&daemon, &path)?;
        let idle_secs = idle_secs_from_env();
        let idle_exit_base = path.clone();
        spawn_idle_watcher(Arc::clone(&daemon), idle_secs, move || {
            // std::process::exit skips destructors -- cleanup must happen here, not in a Drop impl.
            socket::remove_transport_files(&idle_exit_base);
            std::process::exit(0)
        });
        socket::accept_loop(&daemon, listener.incoming());
        return Ok(());
    }

    #[cfg(not(windows))]
    {
        let listener = socket::bind_listener(&path)?;
        let bound_inode = socket::socket_inode(&path);

        let idle_secs = idle_secs_from_env();
        let idle_exit_path = path.clone();
        spawn_idle_watcher(Arc::clone(&daemon), idle_secs, move || {
            // std::process::exit skips destructors -- unlink must happen here, not in a Drop impl.
            unlink_if_still_ours(&idle_exit_path, bound_inode);
            std::process::exit(0)
        });

        socket::accept_loop(&daemon, listener.incoming());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn verify_auth_token_is_false_when_unset() {
        let daemon = Daemon::new();
        assert!(!daemon.verify_auth_token("anything"));
        assert!(!daemon.verify_auth_token(""));
    }

    #[test]
    fn verify_auth_token_matches_only_the_exact_stored_token() {
        let daemon = Daemon::new();
        daemon.set_auth_token("secret-value".to_string());
        assert!(daemon.verify_auth_token("secret-value"));
        assert!(!daemon.verify_auth_token("wrong-value"));
        assert!(!daemon.verify_auth_token(""));
    }

    #[test]
    fn set_auth_token_is_a_no_op_the_second_time() {
        let daemon = Daemon::new();
        daemon.set_auth_token("first".to_string());
        daemon.set_auth_token("second".to_string());
        assert!(daemon.verify_auth_token("first"));
        assert!(!daemon.verify_auth_token("second"));
    }

    #[test]
    fn registry_survives_across_separate_lookups() {
        let daemon = Daemon::new();
        daemon.record_run("run-1".into(), RunState::Exited(0));
        assert_eq!(daemon.lookup_run("run-1"), Some(RunState::Exited(0)));
        // A second, independent lookup call sees the SAME entry -- proves
        // the registry is not per-call/connection-scoped state.
        assert_eq!(daemon.lookup_run("run-1"), Some(RunState::Exited(0)));
        assert_eq!(daemon.lookup_run("unknown"), None);
    }

    #[test]
    fn idle_watcher_fires_after_a_short_injected_window_with_no_connections() {
        let daemon = Daemon::new();
        let fired = Arc::new(AtomicBool::new(false));
        let fired_writer = Arc::clone(&fired);
        let handle = spawn_idle_watcher(Arc::clone(&daemon), 0.15, move || {
            fired_writer.store(true, Ordering::SeqCst);
        });

        std::thread::sleep(Duration::from_millis(30));
        assert!(
            !fired.load(Ordering::SeqCst),
            "idle watcher fired before its injected window elapsed"
        );

        let deadline = Instant::now() + Duration::from_secs(2);
        while !fired.load(Ordering::SeqCst) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(fired.load(Ordering::SeqCst), "idle watcher never fired");
        handle.join().unwrap();
    }

    #[test]
    fn idle_watcher_does_not_fire_while_a_connection_is_open() {
        let daemon = Daemon::new();
        daemon.connection_opened();
        let fired = Arc::new(AtomicBool::new(false));
        let fired_writer = Arc::clone(&fired);
        let handle = spawn_idle_watcher(Arc::clone(&daemon), 0.15, move || {
            fired_writer.store(true, Ordering::SeqCst);
        });

        std::thread::sleep(Duration::from_millis(300));
        assert!(
            !fired.load(Ordering::SeqCst),
            "idle watcher fired with an open connection"
        );

        daemon.connection_closed();
        let deadline = Instant::now() + Duration::from_secs(2);
        while !fired.load(Ordering::SeqCst) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(
            fired.load(Ordering::SeqCst),
            "idle watcher never fired after the connection closed"
        );
        handle.join().unwrap();
    }

    #[test]
    fn mark_detached_if_running_transitions_only_a_running_entry() {
        let daemon = Daemon::new();
        daemon.record_run("still-running".into(), RunState::Running);
        daemon.record_run("already-done".into(), RunState::Exited(0));

        daemon.mark_detached_if_running("still-running");
        daemon.mark_detached_if_running("already-done");
        daemon.mark_detached_if_running("never-seen");

        assert_eq!(daemon.lookup_run("still-running"), Some(RunState::Detached));
        // A terminal state must never be downgraded back to `Detached` by a
        // late-arriving disconnect notice.
        assert_eq!(daemon.lookup_run("already-done"), Some(RunState::Exited(0)));
        assert_eq!(daemon.lookup_run("never-seen"), None);
    }

    #[test]
    fn idle_exit_unlink_skips_when_the_snapshot_inode_no_longer_matches() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("execd.sock");
        let _listener = socket::bind_listener(&path).unwrap();
        let real_inode = socket::socket_inode(&path).unwrap();

        unlink_if_still_ours(&path, Some(real_inode.wrapping_add(1)));
        assert!(path.exists(), "must not unlink when the snapshot inode no longer matches");
    }

    #[test]
    fn idle_exit_unlink_skips_after_a_sibling_daemon_rebinds_the_same_path() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("execd.sock");
        let other_path = tmp.path().join("execd-sibling.sock");
        let _listener1 = socket::bind_listener(&path).unwrap();
        let bound_inode = socket::socket_inode(&path);

        // Simulate a racing sibling daemon winning the bind race: bind at a
        // different path, then atomically replace `path` -- guarantees a
        // different inode, unlike calling bind_listener(&path) a second
        // time (which can hand the just-freed inode straight back on some
        // filesystems).
        let _listener2 = socket::bind_listener(&other_path).unwrap();
        std::fs::rename(&other_path, &path).unwrap();

        unlink_if_still_ours(&path, bound_inode);
        assert!(path.exists(), "must not unlink a path a sibling daemon has since rebound");
    }

    #[test]
    fn idle_exit_unlink_removes_a_path_that_still_matches_the_bind_time_snapshot() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("execd.sock");
        let _listener = socket::bind_listener(&path).unwrap();
        let bound_inode = socket::socket_inode(&path);

        unlink_if_still_ours(&path, bound_inode);
        assert!(!path.exists(), "must unlink a path this process still owns");
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

    #[test]
    fn spawn_background_returns_before_the_child_exits_and_the_worker_later_records_the_terminal_state() {
        with_tools_root(|| {
            let daemon = Daemon::new();
            let (run_id, initial_tail) = Daemon::spawn_background(
                &daemon,
                SpawnSpec {
                    program: "sh".into(),
                    args: vec!["-c".into(), "sleep 0.2; echo spawned-in-background".into()],
                    timeout: None,
                },
            )
            .unwrap();

            // The call returned already -- the registry must show the run
            // as still in progress (or, in the rare case the polling below
            // is slow to schedule, already past it), never absent.
            assert!(daemon.lookup_run(&run_id).is_some());
            assert!(initial_tail.stdout.is_empty(), "nothing written yet at spawn time");

            let deadline = Instant::now() + Duration::from_secs(3);
            let terminal = loop {
                match daemon.lookup_run(&run_id) {
                    Some(RunState::Exited(code)) => break RunState::Exited(code),
                    Some(_) => {
                        assert!(Instant::now() < deadline, "worker never recorded a terminal state");
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    None => panic!("run_id disappeared from the registry"),
                }
            };
            assert_eq!(terminal, RunState::Exited(0));

            let full = exec::fetch_result(&run_id).unwrap();
            assert!(String::from_utf8_lossy(&full.stdout).contains("spawned-in-background"));
        });
    }
}

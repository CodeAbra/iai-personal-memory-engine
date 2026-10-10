//! Unix socket transport for the executor daemon: one JSON-RPC request per
//! line in (envelope: `jsonrpc`, `id`, `method`, `params`), one response
//! per line out, framing copied by shape from the brain daemon's socket
//! server (never by import). Method dispatch (`tool_run`, `result_fetch`)
//! is wired in separately -- this module owns only the envelope/framing
//! contract and the accept loop.

#[cfg(any(unix, windows))]
use std::io::{BufRead, BufReader, Write};
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, PermissionsExt};
#[cfg(unix)]
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;
use serde_json::Value;

use super::Daemon;
use crate::exec::state::RunState;
use crate::exec::{self, artifact, ExecError, SpawnSpec};

pub const ERR_PARSE_ERROR: i32 = -32700;
pub const ERR_INVALID_REQUEST: i32 = -32600;
pub const ERR_METHOD_NOT_FOUND: i32 = -32601;
pub const ERR_INVALID_PARAMS: i32 = -32602;
pub const ERR_INTERNAL: i32 = -32001;

/// Platform-neutral transport aliases so every public signature below names
/// one type regardless of target. Unix: the real Unix-domain-socket types.
/// Windows: real TCP-loopback types, gated behind the mandatory auth-token
/// handshake in `handle_authed_connection`. Any other target: a placeholder
/// this module's own functions refuse to serve from.
#[cfg(unix)]
pub type Listener = UnixListener;
#[cfg(unix)]
pub type Conn = UnixStream;

#[cfg(windows)]
pub type Listener = std::net::TcpListener;
#[cfg(windows)]
pub type Conn = std::net::TcpStream;

#[cfg(not(any(unix, windows)))]
pub struct Listener;
#[cfg(not(any(unix, windows)))]
pub struct Conn;

#[cfg(not(any(unix, windows)))]
impl Listener {
    pub fn incoming(&self) -> std::iter::Empty<std::io::Result<Conn>> {
        std::iter::empty()
    }
}

/// Resolves to `IAI_TOOLS_SOCKET_PATH` when set, else
/// `{IAI_TOOLS_ROOT}/.execd.sock` (`tools_root` itself already honors
/// `IAI_TOOLS_ROOT`, falling back to `~/.iai-tools`).
pub fn resolve_socket_path() -> Result<PathBuf, ExecError> {
    if let Ok(v) = std::env::var("IAI_TOOLS_SOCKET_PATH") {
        if !v.is_empty() {
            return Ok(PathBuf::from(v));
        }
    }
    Ok(artifact::tools_root()?.join(".execd.sock"))
}

/// `<base>.token` -- derived off `resolve_socket_path()`'s base path. This
/// exact naming is the wire contract the Windows client (a later plan)
/// matches.
pub fn token_file_path(base: &Path) -> PathBuf {
    let mut s = base.as_os_str().to_owned();
    s.push(".token");
    PathBuf::from(s)
}

/// `<base>.port` -- see `token_file_path`.
pub fn port_file_path(base: &Path) -> PathBuf {
    let mut s = base.as_os_str().to_owned();
    s.push(".port");
    PathBuf::from(s)
}

/// Constant-time byte comparison mirroring `secrets.compare_digest`: once
/// lengths match, every byte pair is inspected regardless of an earlier
/// mismatch, so comparison time never leaks which byte first differed.
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
thread_local! {
    static FORCE_ACL_FAILURE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Test-only hook: forces `tighten_token_file_acl` to fail on the CALLING
/// THREAD only (mirrors `sandbox::force_misconfigured_for_test`), so the
/// fail-closed contract in `start_tcp_transport` is testable on every
/// platform, not only where `icacls` itself can fail.
#[cfg(test)]
pub(crate) fn force_acl_failure_on_this_thread(force: bool) {
    FORCE_ACL_FAILURE.with(|c| c.set(force));
}

/// Tightens `path`'s ACL to the current user only. No-op off Windows --
/// POSIX callers rely on filesystem mode bits instead. Fails closed: an
/// empty username, a non-zero `icacls` exit, or a spawn failure all return
/// `Err` rather than leaving the file under its default inherited ACL --
/// the caller must never write the token secret after an `Err` here.
pub fn tighten_token_file_acl(path: &Path) -> std::io::Result<()> {
    #[cfg(test)]
    {
        if FORCE_ACL_FAILURE.with(|c| c.get()) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "forced ACL failure (test-only)",
            ));
        }
    }
    #[cfg(windows)]
    {
        let username = std::env::var("USERNAME")
            .or_else(|_| std::env::var("USER"))
            .unwrap_or_default();
        if username.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "token ACL not tightened: no username in environment",
            ));
        }
        let args = vec![
            path.display().to_string(),
            "/inheritance:r".to_string(),
            "/grant:r".to_string(),
            format!("{username}:F"),
        ];
        let output = exec::spawn::run_fixed_command("icacls", &args).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("token ACL not tightened: icacls unavailable: {e}"),
            )
        })?;
        if !output.status.success() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!(
                    "token ACL not tightened: icacls exit status {:?}, stderr: {}",
                    output.status.code(),
                    String::from_utf8_lossy(&output.stderr)
                ),
            ));
        }
    }
    #[cfg(not(windows))]
    {
        let _ = path;
    }
    Ok(())
}

/// Binds a TCP-loopback listener on an ephemeral port, stores the
/// mandatory auth token (client-supplied via `IAI_TOOLS_TOKEN`) on
/// `daemon`, and persists it to `<base>.token` (create-empty ->
/// ACL-tighten -> write-secret, in that exact order) BEFORE writing
/// `<base>.port` -- a client that observes the port file existing may
/// assume the token file is already complete, removing the read race by
/// construction. Platform-neutral so it is natively testable on every
/// target; the `cfg(windows)` `bind_listener` below is production code's
/// only call site.
pub fn start_tcp_transport(daemon: &Daemon, base: &Path) -> std::io::Result<std::net::TcpListener> {
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0))?;
    let port = listener.local_addr()?.port();

    let token = std::env::var("IAI_TOOLS_TOKEN").map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "IAI_TOOLS_TOKEN environment variable not set",
        )
    })?;
    daemon.set_auth_token(token.clone());

    if let Some(parent) = base.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let token_path = token_file_path(base);
    std::fs::write(&token_path, "")?;
    if let Err(e) = tighten_token_file_acl(&token_path) {
        let _ = std::fs::remove_file(&token_path);
        return Err(e);
    }
    std::fs::write(&token_path, &token)?;

    std::fs::write(port_file_path(base), port.to_string())?;

    Ok(listener)
}

/// Removes `<base>.token` and `<base>.port`, idempotent when either is
/// already absent. Windows idle-exit calls this before `process::exit` --
/// TCP loopback has no filesystem-inode identity for
/// `unlink_if_still_ours` to key off, so this is the Windows-specific
/// equivalent cleanup path, mirroring `_ipc.py.shutdown_ipc`'s Windows
/// branch.
pub fn remove_transport_files(base: &Path) {
    let _ = std::fs::remove_file(token_file_path(base));
    let _ = std::fs::remove_file(port_file_path(base));
}

/// Binds `path`, clearing a stale socket file from a prior process that
/// exited without cleanup first, and chmods the result to 0o600 -- the
/// same posture as the brain daemon's socket permissioning.
#[cfg(unix)]
pub fn bind_listener(path: &Path) -> std::io::Result<Listener> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let _ = std::fs::remove_file(path);
    let listener = UnixListener::bind(path)?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    Ok(listener)
}

/// Real TCP-loopback bind for the Windows transport: delegates to
/// `start_tcp_transport` for the bind plus the token/port file writes.
#[cfg(windows)]
pub fn bind_listener(daemon: &Daemon, path: &Path) -> std::io::Result<Listener> {
    start_tcp_transport(daemon, path)
}

/// No socket transport available on this platform.
#[cfg(not(any(unix, windows)))]
pub fn bind_listener(_path: &Path) -> std::io::Result<Listener> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "socket transport unavailable on this platform",
    ))
}

/// Directory-entry inode currently at `path`, or `None` if nothing is
/// there. `lstat`-based, not `fstat` on a listener fd: for `AF_UNIX` the
/// listening socket's own fd inode lives in the socket filesystem, not
/// the directory entry at `path`, so identity must be compared at the
/// path on both ends.
#[cfg(unix)]
pub fn socket_inode(path: &Path) -> Option<u64> {
    std::fs::symlink_metadata(path).ok().map(|m| m.ino())
}

/// TCP loopback has no filesystem-inode identity to compare -- Windows
/// idle-exit uses `remove_transport_files` instead of inode-guarded
/// unlink.
#[cfg(windows)]
pub fn socket_inode(_path: &Path) -> Option<u64> {
    None
}

#[cfg(not(any(unix, windows)))]
pub fn socket_inode(_path: &Path) -> Option<u64> {
    None
}

#[derive(Debug, Default, Deserialize)]
struct Envelope {
    jsonrpc: Option<String>,
    #[serde(default)]
    id: Value,
    method: Option<String>,
    #[serde(default)]
    params: Value,
}

/// Runs the request/response loop for ONE connection: reads one JSON line
/// per request, dispatches, writes one JSON line per response, until EOF
/// or a write error. `connection_closed()` brackets `accept_loop`'s own
/// synchronous `connection_opened()` call (see there) so the idle watcher
/// never fires mid-connection. Any `background: true` run this connection
/// started is transitioned to `Detached` (if still running) once the loop
/// ends -- the run itself was already spawned on a daemon-owned worker
/// thread, so this only updates the bookkeeping that nobody is watching it
/// anymore.
///
/// Callers MUST have already called `daemon.connection_opened()` before
/// invoking this (production: `accept_loop`, below). This function calls
/// `connection_closed()` exactly once, unconditionally, on every return
/// path -- a caller that skips the matching `connection_opened()` corrupts
/// the open/close pairing.
#[cfg(unix)]
pub fn handle_connection(daemon: &Arc<Daemon>, stream: Conn) {
    let mut writer = match stream.try_clone() {
        Ok(s) => s,
        Err(_) => {
            daemon.connection_closed();
            return;
        }
    };
    let reader = BufReader::new(stream);
    let mut background_runs: Vec<String> = Vec::new();
    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        daemon.record_activity();
        let response = handle_line(daemon, &line, &mut background_runs);
        if writer.write_all(response.as_bytes()).is_err() {
            break;
        }
        if writer.write_all(b"\n").is_err() {
            break;
        }
        if writer.flush().is_err() {
            break;
        }
    }
    for run_id in &background_runs {
        daemon.mark_detached_if_running(run_id);
    }
    daemon.connection_closed();
}

/// Reads the mandatory first line and verifies it against `daemon`'s
/// stored auth token; on mismatch (including no line at all, e.g. an
/// immediate EOF) drops the connection with zero response bytes. On
/// match, runs the same JSON-RPC line loop as the unix arm above. Calls
/// `connection_closed()` exactly once on EVERY return path -- callers
/// (production: `accept_loop`) must have already called
/// `connection_opened()`, matching `handle_connection`'s contract on
/// unix. Platform-neutral (not `#[cfg(windows)]`) so the native test
/// drives it directly; the `cfg(windows)` `handle_connection` below
/// delegates to it.
#[cfg(any(unix, windows))]
pub fn handle_authed_connection(daemon: &Arc<Daemon>, stream: std::net::TcpStream) {
    let mut writer = match stream.try_clone() {
        Ok(s) => s,
        Err(_) => {
            daemon.connection_closed();
            return;
        }
    };
    let mut reader = BufReader::new(stream);
    let mut first_line = String::new();
    let authed = matches!(reader.read_line(&mut first_line), Ok(n) if n > 0)
        && daemon.verify_auth_token(first_line.trim_end_matches(['\r', '\n']));
    if !authed {
        daemon.connection_closed();
        return;
    }

    let mut background_runs: Vec<String> = Vec::new();
    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        daemon.record_activity();
        let response = handle_line(daemon, &line, &mut background_runs);
        if writer.write_all(response.as_bytes()).is_err() {
            break;
        }
        if writer.write_all(b"\n").is_err() {
            break;
        }
        if writer.flush().is_err() {
            break;
        }
    }
    for run_id in &background_runs {
        daemon.mark_detached_if_running(run_id);
    }
    daemon.connection_closed();
}

/// Delegates to the platform-neutral authed handler above so the token
/// gate and the `connection_closed()` pairing are the exact code the
/// native test already exercises.
#[cfg(windows)]
pub fn handle_connection(daemon: &Arc<Daemon>, stream: Conn) {
    handle_authed_connection(daemon, stream);
}

/// No-op: must not call `connection_closed()` -- this fallback platform's
/// accept loop opens no connection (an empty iterator), so decrementing
/// here would underflow the active-connection counter.
#[cfg(not(any(unix, windows)))]
pub fn handle_connection(_daemon: &Arc<Daemon>, _stream: Conn) {}

/// Drives the accept loop over `incoming` (production: `listener.incoming()`,
/// injectable in tests). Two invariants live here together because they are
/// two sides of the same loop:
///
/// - `connection_opened()` is called synchronously, right here, for EVERY
///   accepted stream -- BEFORE spawning the per-connection thread, not
///   inside it. This closes the TOCTOU window where the idle watcher could
///   observe zero active connections between `accept()` returning and a
///   spawned thread reaching its own open-bookkeeping call. Call it here
///   exactly once per accepted stream; do not also call it inside
///   `handle_connection` (see that function's contract) -- doing both would
///   double-count opens against a single close and leak `active_connections`
///   upward, permanently disabling idle self-termination.
/// - A transient per-connection `Err` from the iterator is logged and
///   skipped via `continue`, never propagated -- one bad `accept()` must
///   never end the whole loop and take the resident daemon down with it.
///   Only the iterator itself ending (production: never, for a bound
///   listener) ends this loop.
pub fn accept_loop(daemon: &Arc<Daemon>, incoming: impl Iterator<Item = std::io::Result<Conn>>) {
    for stream in incoming {
        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("iai_tools_execd: accept error, continuing: {e}");
                continue;
            }
        };
        daemon.connection_opened();
        let daemon = Arc::clone(daemon);
        std::thread::spawn(move || {
            handle_connection(&daemon, stream);
        });
    }
}

fn handle_line(daemon: &Arc<Daemon>, line: &str, background_runs: &mut Vec<String>) -> String {
    let envelope: Envelope = match serde_json::from_str(line) {
        Ok(e) => e,
        Err(e) => return error_response(Value::Null, ERR_PARSE_ERROR, &e.to_string()),
    };
    if envelope.jsonrpc.as_deref() != Some("2.0") {
        return error_response(envelope.id, ERR_INVALID_REQUEST, "jsonrpc must be '2.0'");
    }
    let method = match envelope.method {
        Some(m) => m,
        None => {
            return error_response(envelope.id, ERR_INVALID_REQUEST, "method must be a string")
        }
    };
    match dispatch(daemon, &method, &envelope.params, background_runs) {
        Ok(result) => ok_response(envelope.id, result),
        Err((code, message)) => error_response(envelope.id, code, &message),
    }
}

/// Method table: `tool_run` spawns and captures a command (or, with
/// `background: true`, hands it to a daemon-owned worker and returns
/// immediately), `result_fetch` re-serves a prior run's bounded output
/// window without re-running it, `tool_batch_run` runs a list of commands
/// under one of two concurrency semantics.
fn dispatch(
    daemon: &Arc<Daemon>,
    method: &str,
    params: &Value,
    background_runs: &mut Vec<String>,
) -> Result<Value, (i32, String)> {
    match method {
        "tool_run" => dispatch_tool_run(daemon, params, background_runs),
        "result_fetch" => dispatch_result_fetch(daemon, params),
        "tool_batch_run" => super::batch::dispatch_tool_batch_run(params),
        other => Err((ERR_METHOD_NOT_FOUND, format!("unknown method '{other}'"))),
    }
}

fn dispatch_tool_run(
    daemon: &Arc<Daemon>,
    params: &Value,
    background_runs: &mut Vec<String>,
) -> Result<Value, (i32, String)> {
    let command = params
        .get("command")
        .and_then(Value::as_str)
        .ok_or_else(|| (ERR_INVALID_PARAMS, "missing required param: 'command'".to_string()))?
        .to_string();
    let args: Vec<String> = params
        .get("args")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let timeout = match params.get("timeout") {
        None | Some(Value::Null) => None,
        Some(v) => Some(parse_timeout_secs(v)?),
    };
    let background = params
        .get("background")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    if background {
        let (run_id, initial_tail) = Daemon::spawn_background(
            daemon,
            SpawnSpec {
                program: command,
                args,
                timeout,
            },
        )
        .map_err(exec_error_to_rpc)?;
        background_runs.push(run_id.clone());
        return Ok(serde_json::json!({
            "run_id": run_id,
            "state": "running",
            "bounded_tail": bounded_result_to_json(&initial_tail),
        }));
    }

    let handle = exec::spawn_run(SpawnSpec {
        program: command,
        args,
        timeout,
    })
    .map_err(exec_error_to_rpc)?;
    daemon.record_run(handle.run_id.clone(), handle.run_state);
    Ok(serde_json::json!({
        "run_id": handle.run_id,
        "state": state_to_str(handle.run_state),
        "bounded_tail": bounded_result_to_json(&handle.result),
    }))
}

/// Parses a wire `timeout` value (seconds) into a `Duration`. Rejects
/// non-finite, negative, or zero values as `ERR_INVALID_PARAMS` rather than
/// letting `Duration::from_secs_f64` panic on a NaN/negative/overflowing
/// value straight off untrusted JSON.
fn parse_timeout_secs(v: &Value) -> Result<std::time::Duration, (i32, String)> {
    let secs = v
        .as_f64()
        .ok_or_else(|| (ERR_INVALID_PARAMS, "'timeout' must be a number".to_string()))?;
    std::time::Duration::try_from_secs_f64(secs)
        .map_err(|_| (ERR_INVALID_PARAMS, "'timeout' must be a finite, positive number of seconds".to_string()))
        .and_then(|d| {
            if d.is_zero() {
                Err((ERR_INVALID_PARAMS, "'timeout' must be greater than zero".to_string()))
            } else {
                Ok(d)
            }
        })
}

/// A caller-supplied `run_id` that fails `artifact::validate_run_id` (bad
/// shape, traversal, absolute path) is a client mistake, not a server
/// fault -- surfaced as `ERR_INVALID_PARAMS`, not `ERR_INTERNAL`.
fn exec_error_to_rpc(err: ExecError) -> (i32, String) {
    if let ExecError::Io(io_err) = &err {
        if io_err.kind() == std::io::ErrorKind::InvalidInput {
            return (ERR_INVALID_PARAMS, err.to_string());
        }
    }
    (ERR_INTERNAL, err.to_string())
}

fn dispatch_result_fetch(daemon: &Daemon, params: &Value) -> Result<Value, (i32, String)> {
    let run_id = params
        .get("run_id")
        .and_then(Value::as_str)
        .ok_or_else(|| (ERR_INVALID_PARAMS, "missing required param: 'run_id'".to_string()))?
        .to_string();
    let bound = match params.get("bound") {
        None | Some(Value::Null) => None,
        Some(v) => Some(
            v.as_u64()
                .filter(|n| *n > 0)
                .ok_or_else(|| (ERR_INVALID_PARAMS, "'bound' must be a positive integer".to_string()))?,
        ),
    };

    let result = match bound {
        Some(byte_cap) => exec::fetch_result_bounded(
            &run_id,
            exec::tail::TailBound {
                byte_cap,
                line_cap: exec::tail::TAIL_LINE_CAP,
            },
        ),
        None => exec::fetch_result(&run_id),
    }
    .map_err(exec_error_to_rpc)?;
    let state = daemon.lookup_run(&run_id).map(state_to_str);
    Ok(serde_json::json!({
        "run_id": run_id,
        "state": state,
        "bounded_tail": bounded_result_to_json(&result),
    }))
}

fn state_to_str(state: RunState) -> String {
    match state {
        RunState::Running => "running".to_string(),
        RunState::Exited(code) => format!("exited:{code}"),
        RunState::Killed(sig) => format!("killed:{sig}"),
        RunState::TimedOut => "timed_out".to_string(),
        RunState::Detached => "detached".to_string(),
    }
}

pub(super) fn bounded_result_to_json(result: &exec::BoundedResult) -> Value {
    serde_json::json!({
        "stdout": String::from_utf8_lossy(&result.stdout),
        "stderr": String::from_utf8_lossy(&result.stderr),
        "stdout_offset": result.stdout_offset,
        "stderr_offset": result.stderr_offset,
        "elided_bytes": result.elided_bytes,
        "truncated": result.truncated,
    })
}

fn ok_response(id: Value, result: Value) -> String {
    serde_json::json!({"jsonrpc": "2.0", "id": id, "result": result}).to_string()
}

fn error_response(id: Value, code: i32, message: &str) -> String {
    serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {"code": code, "message": message},
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::artifact::ENV_LOCK;
    use std::sync::Arc;

    fn with_tools_root<T>(f: impl FnOnce(&Path) -> T) -> T {
        let _guard = ENV_LOCK.lock().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        // SAFETY: ENV_LOCK serializes every test that touches IAI_TOOLS_ROOT.
        unsafe {
            std::env::set_var("IAI_TOOLS_ROOT", tmp.path());
        }
        let result = f(tmp.path());
        unsafe {
            std::env::remove_var("IAI_TOOLS_ROOT");
        }
        result
    }

    fn request_response(client: &mut UnixStream, req: &Value) -> Value {
        client
            .write_all((req.to_string() + "\n").as_bytes())
            .unwrap();
        let mut reader = BufReader::new(client.try_clone().unwrap());
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        serde_json::from_str(&line).unwrap()
    }

    #[test]
    fn constant_time_eq_matches_identical_byte_slices() {
        assert!(constant_time_eq(b"same-value", b"same-value"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn constant_time_eq_rejects_different_length_or_content() {
        assert!(!constant_time_eq(b"abc", b"abcd"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"", b"a"));
    }

    #[test]
    fn start_tcp_transport_fails_closed_when_token_env_var_is_unset() {
        with_tools_root(|root| {
            let base = root.join("no-token.sock");
            let daemon = Daemon::new();
            // SAFETY: ENV_LOCK (held by with_tools_root) serializes every
            // test that touches process env vars.
            unsafe {
                std::env::remove_var("IAI_TOOLS_TOKEN");
            }
            let result = start_tcp_transport(&daemon, &base);
            assert!(
                result.is_err(),
                "must fail closed when no token value is available, never fall back to bare TCP"
            );
        });
    }

    #[test]
    fn start_tcp_transport_removes_token_file_and_never_writes_port_file_when_acl_tightening_fails() {
        with_tools_root(|root| {
            let base = root.join("acl-fail.sock");
            let daemon = Daemon::new();
            // SAFETY: ENV_LOCK (held by with_tools_root) serializes every
            // test that touches process env vars.
            unsafe {
                std::env::set_var("IAI_TOOLS_TOKEN", "some-token");
            }
            force_acl_failure_on_this_thread(true);
            let result = start_tcp_transport(&daemon, &base);
            force_acl_failure_on_this_thread(false);
            unsafe {
                std::env::remove_var("IAI_TOOLS_TOKEN");
            }

            assert!(
                result.is_err(),
                "must fail closed when ACL tightening fails"
            );
            assert!(
                !token_file_path(&base).exists(),
                "the un-tightened token file must be removed, never left holding the secret"
            );
            assert!(
                !port_file_path(&base).exists(),
                "the port file must never be written before the token is fully secured"
            );
        });
    }

    #[test]
    fn native_tcp_transport_correct_token_reaches_json_rpc_dispatch() {
        with_tools_root(|root| {
            let base = root.join("win-execd.sock");
            let daemon = Daemon::new();
            // SAFETY: ENV_LOCK (held by with_tools_root) serializes every
            // test that touches process env vars.
            unsafe {
                std::env::set_var("IAI_TOOLS_TOKEN", "native-test-token");
            }
            let listener = start_tcp_transport(&daemon, &base).unwrap();
            unsafe {
                std::env::remove_var("IAI_TOOLS_TOKEN");
            }
            let port = listener.local_addr().unwrap().port();

            assert_eq!(
                std::fs::read_to_string(port_file_path(&base)).unwrap().trim(),
                port.to_string(),
                "the port file must hold the bound port"
            );
            assert_eq!(
                std::fs::read_to_string(token_file_path(&base)).unwrap(),
                "native-test-token",
                "the token file must hold the client-supplied token"
            );

            let accept_daemon = Arc::clone(&daemon);
            let accept_handle = std::thread::spawn(move || {
                let (stream, _) = listener.accept().unwrap();
                // Mirrors accept_loop's contract: connection_opened() before
                // handle_authed_connection, not inside it.
                accept_daemon.connection_opened();
                handle_authed_connection(&accept_daemon, stream);
            });

            let mut client = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
            client.write_all(b"native-test-token\n").unwrap();
            let req = serde_json::json!({
                "jsonrpc": "2.0", "id": 1, "method": "tool_run",
                "params": {"command": "echo", "args": ["hi"]},
            });
            client.write_all((req.to_string() + "\n").as_bytes()).unwrap();

            let resp: Value = {
                // Scoped so the cloned read fd is dropped BEFORE `client`
                // below -- both dup'd fds share one connection, so the
                // peer only sees EOF once every fd referencing it closes.
                let mut reader = BufReader::new(client.try_clone().unwrap());
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                serde_json::from_str(&line).unwrap()
            };
            assert!(
                resp["result"]["run_id"].as_str().is_some(),
                "a correct-token client must reach JSON-RPC dispatch, got: {resp}"
            );

            drop(client);
            accept_handle.join().unwrap();
        });
    }

    #[test]
    fn native_tcp_transport_wrong_token_closes_connection_with_zero_bytes() {
        with_tools_root(|root| {
            let base = root.join("win-execd-badtoken.sock");
            let daemon = Daemon::new();
            // SAFETY: ENV_LOCK (held by with_tools_root) serializes every
            // test that touches process env vars.
            unsafe {
                std::env::set_var("IAI_TOOLS_TOKEN", "correct-token");
            }
            let listener = start_tcp_transport(&daemon, &base).unwrap();
            unsafe {
                std::env::remove_var("IAI_TOOLS_TOKEN");
            }
            let port = listener.local_addr().unwrap().port();

            let accept_daemon = Arc::clone(&daemon);
            let accept_handle = std::thread::spawn(move || {
                let (stream, _) = listener.accept().unwrap();
                accept_daemon.connection_opened();
                handle_authed_connection(&accept_daemon, stream);
            });

            let mut client = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
            client.write_all(b"wrong-token\n").unwrap();

            use std::io::Read;
            let mut buf = Vec::new();
            let n = client.read_to_end(&mut buf).unwrap();
            assert_eq!(n, 0, "a rejected handshake must return zero response bytes");

            accept_handle.join().unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
            while !daemon.is_idle(0.0) {
                assert!(
                    std::time::Instant::now() < deadline,
                    "active_connections never returned to zero after a rejected handshake"
                );
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        });
    }

    #[test]
    fn native_tcp_transport_immediate_eof_closes_connection_with_zero_bytes() {
        with_tools_root(|root| {
            let base = root.join("win-execd-eof.sock");
            let daemon = Daemon::new();
            // SAFETY: ENV_LOCK (held by with_tools_root) serializes every
            // test that touches process env vars.
            unsafe {
                std::env::set_var("IAI_TOOLS_TOKEN", "correct-token");
            }
            let listener = start_tcp_transport(&daemon, &base).unwrap();
            unsafe {
                std::env::remove_var("IAI_TOOLS_TOKEN");
            }
            let port = listener.local_addr().unwrap().port();

            let accept_daemon = Arc::clone(&daemon);
            let accept_handle = std::thread::spawn(move || {
                let (stream, _) = listener.accept().unwrap();
                accept_daemon.connection_opened();
                handle_authed_connection(&accept_daemon, stream);
            });

            let mut client = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
            client.shutdown(std::net::Shutdown::Write).unwrap();

            use std::io::Read;
            let mut buf = Vec::new();
            let n = client.read_to_end(&mut buf).unwrap();
            assert_eq!(
                n, 0,
                "a client that sends zero bytes (immediate EOF) must be rejected with zero response bytes"
            );

            accept_handle.join().unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
            while !daemon.is_idle(0.0) {
                assert!(
                    std::time::Instant::now() < deadline,
                    "active_connections never returned to zero after an immediate-EOF handshake"
                );
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        });
    }

    #[test]
    fn remove_transport_files_is_idempotent_and_removes_both_files() {
        with_tools_root(|root| {
            let base = root.join("cleanup-test.sock");
            std::fs::write(token_file_path(&base), "tok").unwrap();
            std::fs::write(port_file_path(&base), "12345").unwrap();

            remove_transport_files(&base);
            assert!(!token_file_path(&base).exists());
            assert!(!port_file_path(&base).exists());

            // Second call, both already absent, must not panic.
            remove_transport_files(&base);
        });
    }

    #[test]
    fn bind_listener_creates_socket_at_mode_0600() {
        with_tools_root(|root| {
            let sock_path = root.join("mode-test.sock");
            let _listener = bind_listener(&sock_path).unwrap();
            let mode = std::fs::metadata(&sock_path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        });
    }

    #[test]
    fn bind_listener_clears_a_stale_socket_file() {
        with_tools_root(|root| {
            let sock_path = root.join("stale.sock");
            std::fs::write(&sock_path, b"not a socket").unwrap();
            let listener = bind_listener(&sock_path);
            assert!(listener.is_ok(), "bind must clear a stale non-socket file");
        });
    }

    #[test]
    fn unknown_method_returns_method_not_found_error() {
        with_tools_root(|root| {
            let sock_path = root.join("err.sock");
            let listener = bind_listener(&sock_path).unwrap();
            let daemon = Daemon::new();
            let accept_handle = std::thread::spawn(move || {
                let (stream, _) = listener.accept().unwrap();
                // Mirrors accept_loop's contract: connection_opened() before
                // handle_connection, not inside it.
                daemon.connection_opened();
                handle_connection(&daemon, stream);
            });

            let mut client = UnixStream::connect(&sock_path).unwrap();
            let resp = request_response(
                &mut client,
                &serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": "no_such_method"}),
            );
            assert_eq!(resp["error"]["code"], ERR_METHOD_NOT_FOUND);

            drop(client);
            accept_handle.join().unwrap();
        });
    }

    #[test]
    fn missing_jsonrpc_field_returns_invalid_request_error() {
        with_tools_root(|root| {
            let sock_path = root.join("invalid.sock");
            let listener = bind_listener(&sock_path).unwrap();
            let daemon = Daemon::new();
            let accept_handle = std::thread::spawn(move || {
                let (stream, _) = listener.accept().unwrap();
                // Mirrors accept_loop's contract: connection_opened() before
                // handle_connection, not inside it.
                daemon.connection_opened();
                handle_connection(&daemon, stream);
            });

            let mut client = UnixStream::connect(&sock_path).unwrap();
            let resp = request_response(
                &mut client,
                &serde_json::json!({"id": 1, "method": "tool_run"}),
            );
            assert_eq!(resp["error"]["code"], ERR_INVALID_REQUEST);

            drop(client);
            accept_handle.join().unwrap();
        });
    }

    #[test]
    fn malformed_json_returns_parse_error() {
        with_tools_root(|root| {
            let sock_path = root.join("parse-err.sock");
            let listener = bind_listener(&sock_path).unwrap();
            let daemon = Daemon::new();
            let accept_handle = std::thread::spawn(move || {
                let (stream, _) = listener.accept().unwrap();
                // Mirrors accept_loop's contract: connection_opened() before
                // handle_connection, not inside it.
                daemon.connection_opened();
                handle_connection(&daemon, stream);
            });

            let mut client = UnixStream::connect(&sock_path).unwrap();
            client.write_all(b"{not json\n").unwrap();
            let resp: Value = {
                // Scoped so the cloned read fd is dropped BEFORE `client`
                // below -- both dup'd fds share one connection, so the
                // peer only sees EOF once every fd referencing it closes.
                let mut reader = BufReader::new(client.try_clone().unwrap());
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                serde_json::from_str(&line).unwrap()
            };
            assert_eq!(resp["error"]["code"], ERR_PARSE_ERROR);

            drop(client);
            accept_handle.join().unwrap();
        });
    }

    #[test]
    fn connection_open_and_close_bracket_daemon_activity() {
        with_tools_root(|root| {
            let sock_path = root.join("activity.sock");
            let listener = bind_listener(&sock_path).unwrap();
            let daemon = Daemon::new();
            let daemon_bg = Arc::clone(&daemon);
            let accept_handle = std::thread::spawn(move || {
                let (stream, _) = listener.accept().unwrap();
                // Mirrors accept_loop's contract: connection_opened() before
                // handle_connection, not inside it.
                daemon_bg.connection_opened();
                handle_connection(&daemon_bg, stream);
            });

            let client = UnixStream::connect(&sock_path).unwrap();
            // A live connection with no traffic yet still counts as
            // "connected" for the idle watcher's active_connections gate.
            std::thread::sleep(std::time::Duration::from_millis(20));
            assert!(!daemon.is_idle(0.0), "an open connection must not read as idle");

            drop(client);
            accept_handle.join().unwrap();
        });
    }

    #[test]
    fn tool_run_then_result_fetch_round_trip_over_the_socket() {
        with_tools_root(|root| {
            let sock_path = root.join("rt.sock");
            let listener = bind_listener(&sock_path).unwrap();
            let daemon = Daemon::new();
            let daemon_bg = Arc::clone(&daemon);
            let accept_handle = std::thread::spawn(move || {
                let (stream, _) = listener.accept().unwrap();
                // Mirrors accept_loop's contract: connection_opened() before
                // handle_connection, not inside it.
                daemon_bg.connection_opened();
                handle_connection(&daemon_bg, stream);
            });

            let mut client = UnixStream::connect(&sock_path).unwrap();
            let run_resp = request_response(
                &mut client,
                &serde_json::json!({
                    "jsonrpc": "2.0", "id": 1, "method": "tool_run",
                    "params": {"command": "echo", "args": ["hi"]},
                }),
            );
            let run_id = run_resp["result"]["run_id"].as_str().unwrap().to_string();
            assert!(run_resp["result"]["bounded_tail"]["stdout"]
                .as_str()
                .unwrap()
                .contains("hi"));
            assert_eq!(run_resp["result"]["state"].as_str().unwrap(), "exited:0");

            let fetch_resp = request_response(
                &mut client,
                &serde_json::json!({
                    "jsonrpc": "2.0", "id": 2, "method": "result_fetch",
                    "params": {"run_id": run_id},
                }),
            );
            // The artifact was already fully served by tool_run's own
            // initial fetch -- a re-serve returns no NEW bytes, proving
            // result_fetch does not re-run the command.
            assert_eq!(
                fetch_resp["result"]["bounded_tail"]["stdout"]
                    .as_str()
                    .unwrap(),
                ""
            );

            drop(client);
            accept_handle.join().unwrap();
        });
    }

    #[test]
    fn tool_run_with_missing_command_returns_invalid_params_error() {
        with_tools_root(|root| {
            let sock_path = root.join("missing-cmd.sock");
            let listener = bind_listener(&sock_path).unwrap();
            let daemon = Daemon::new();
            let accept_handle = std::thread::spawn(move || {
                let (stream, _) = listener.accept().unwrap();
                // Mirrors accept_loop's contract: connection_opened() before
                // handle_connection, not inside it.
                daemon.connection_opened();
                handle_connection(&daemon, stream);
            });

            let mut client = UnixStream::connect(&sock_path).unwrap();
            let resp = request_response(
                &mut client,
                &serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": "tool_run", "params": {}}),
            );
            assert_eq!(resp["error"]["code"], ERR_INVALID_PARAMS);

            drop(client);
            accept_handle.join().unwrap();
        });
    }

    #[test]
    fn result_fetch_rejects_malicious_run_ids_over_the_wire() {
        with_tools_root(|root| {
            let sock_path = root.join("traversal.sock");
            let listener = bind_listener(&sock_path).unwrap();
            let daemon = Daemon::new();
            let accept_handle = std::thread::spawn(move || {
                let (stream, _) = listener.accept().unwrap();
                // Mirrors accept_loop's contract: connection_opened() before
                // handle_connection, not inside it.
                daemon.connection_opened();
                handle_connection(&daemon, stream);
            });

            let mut client = UnixStream::connect(&sock_path).unwrap();
            for (i, run_id) in [
                "../../etc/passwd",
                "/etc/passwd",
                "..",
                "",
                "a/b",
                "1a-2b-\u{0441}03",
            ]
            .into_iter()
            .enumerate()
            {
                let resp = request_response(
                    &mut client,
                    &serde_json::json!({
                        "jsonrpc": "2.0", "id": i as i64, "method": "result_fetch",
                        "params": {"run_id": run_id},
                    }),
                );
                assert_eq!(
                    resp["error"]["code"], ERR_INVALID_PARAMS,
                    "run_id {run_id:?} must be rejected as invalid params, got: {resp}"
                );
            }

            drop(client);
            accept_handle.join().unwrap();
        });
    }

    #[test]
    fn tool_run_timeout_kills_a_command_before_its_own_sleep_elapses() {
        with_tools_root(|root| {
            let sock_path = root.join("timeout.sock");
            let listener = bind_listener(&sock_path).unwrap();
            let daemon = Daemon::new();
            let accept_handle = std::thread::spawn(move || {
                let (stream, _) = listener.accept().unwrap();
                // Mirrors accept_loop's contract: connection_opened() before
                // handle_connection, not inside it.
                daemon.connection_opened();
                handle_connection(&daemon, stream);
            });

            let mut client = UnixStream::connect(&sock_path).unwrap();
            let started = std::time::Instant::now();
            let resp = request_response(
                &mut client,
                &serde_json::json!({
                    "jsonrpc": "2.0", "id": 1, "method": "tool_run",
                    "params": {"command": "sleep", "args": ["5"], "timeout": 0.2},
                }),
            );
            let elapsed = started.elapsed();
            let state = resp["result"]["state"].as_str().unwrap();
            assert!(
                state == "timed_out" || state.starts_with("killed:"),
                "a 5s sleep with timeout:0.2 must not exit normally, got state: {state}"
            );
            assert!(
                elapsed < std::time::Duration::from_secs(4),
                "an honored timeout must not wait out the full 5s sleep, took {elapsed:?}"
            );

            drop(client);
            accept_handle.join().unwrap();
        });
    }

    #[test]
    fn tool_run_rejects_a_non_finite_timeout() {
        with_tools_root(|root| {
            let sock_path = root.join("bad-timeout.sock");
            let listener = bind_listener(&sock_path).unwrap();
            let daemon = Daemon::new();
            let accept_handle = std::thread::spawn(move || {
                let (stream, _) = listener.accept().unwrap();
                // Mirrors accept_loop's contract: connection_opened() before
                // handle_connection, not inside it.
                daemon.connection_opened();
                handle_connection(&daemon, stream);
            });

            let mut client = UnixStream::connect(&sock_path).unwrap();
            let resp = request_response(
                &mut client,
                &serde_json::json!({
                    "jsonrpc": "2.0", "id": 1, "method": "tool_run",
                    "params": {"command": "echo", "args": ["hi"], "timeout": -1.0},
                }),
            );
            assert_eq!(resp["error"]["code"], ERR_INVALID_PARAMS);

            drop(client);
            accept_handle.join().unwrap();
        });
    }

    #[test]
    fn result_fetch_bound_caps_the_served_window_over_a_still_growing_background_run() {
        with_tools_root(|root| {
            let sock_path = root.join("bound.sock");
            let listener = bind_listener(&sock_path).unwrap();
            let daemon = Daemon::new();
            let accept_handle = std::thread::spawn(move || {
                let (stream, _) = listener.accept().unwrap();
                // Mirrors accept_loop's contract: connection_opened() before
                // handle_connection, not inside it.
                daemon.connection_opened();
                handle_connection(&daemon, stream);
            });

            let mut client = UnixStream::connect(&sock_path).unwrap();
            // A background run so the artifact keeps growing after the
            // initial fetch -- a synchronous run's own initial fetch
            // already drains everything before any explicit fetch could
            // observe a bound's effect.
            let run_resp = request_response(
                &mut client,
                &serde_json::json!({
                    "jsonrpc": "2.0", "id": 1, "method": "tool_run",
                    "params": {
                        "command": "sh", "args": ["-c", "sleep 0.3; printf '0123456789'"],
                        "background": true,
                    },
                }),
            );
            let run_id = run_resp["result"]["run_id"].as_str().unwrap().to_string();

            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
            loop {
                std::thread::sleep(std::time::Duration::from_millis(20));
                if std::fs::metadata(root.join("artifacts").join(&run_id).join("stdout.log"))
                    .map(|m| m.len())
                    .unwrap_or(0)
                    >= 10
                {
                    break;
                }
                assert!(std::time::Instant::now() < deadline, "child never wrote its output");
            }

            let fetch_resp = request_response(
                &mut client,
                &serde_json::json!({
                    "jsonrpc": "2.0", "id": 2, "method": "result_fetch",
                    "params": {"run_id": run_id, "bound": 4},
                }),
            );
            let first_chunk = fetch_resp["result"]["bounded_tail"]["stdout"].as_str().unwrap();
            assert_eq!(first_chunk, "0123", "bound:4 must cap the window at 4 bytes");

            let second_fetch = request_response(
                &mut client,
                &serde_json::json!({
                    "jsonrpc": "2.0", "id": 3, "method": "result_fetch",
                    "params": {"run_id": run_id, "bound": 100},
                }),
            );
            let second_chunk = second_fetch["result"]["bounded_tail"]["stdout"].as_str().unwrap();
            assert_eq!(
                second_chunk, "456789",
                "a second fetch must continue from the offset the first (bounded) fetch left off"
            );

            drop(client);
            accept_handle.join().unwrap();
        });
    }

    #[test]
    fn result_fetch_rejects_a_non_positive_bound() {
        with_tools_root(|root| {
            let sock_path = root.join("bad-bound.sock");
            let listener = bind_listener(&sock_path).unwrap();
            let daemon = Daemon::new();
            let accept_handle = std::thread::spawn(move || {
                let (stream, _) = listener.accept().unwrap();
                // Mirrors accept_loop's contract: connection_opened() before
                // handle_connection, not inside it.
                daemon.connection_opened();
                handle_connection(&daemon, stream);
            });

            let mut client = UnixStream::connect(&sock_path).unwrap();
            let run_resp = request_response(
                &mut client,
                &serde_json::json!({
                    "jsonrpc": "2.0", "id": 1, "method": "tool_run",
                    "params": {"command": "echo", "args": ["hi"]},
                }),
            );
            let run_id = run_resp["result"]["run_id"].as_str().unwrap().to_string();

            let resp = request_response(
                &mut client,
                &serde_json::json!({
                    "jsonrpc": "2.0", "id": 2, "method": "result_fetch",
                    "params": {"run_id": run_id, "bound": 0},
                }),
            );
            assert_eq!(resp["error"]["code"], ERR_INVALID_PARAMS);

            drop(client);
            accept_handle.join().unwrap();
        });
    }

    #[test]
    fn accept_loop_counts_a_connection_before_returning_no_toctou_window() {
        with_tools_root(|root| {
            let sock_path = root.join("idle-race.sock");
            let listener = bind_listener(&sock_path).unwrap();
            let daemon = Daemon::new();
            let daemon_for_loop = Arc::clone(&daemon);
            // Bounded to exactly one connection so the spawned thread
            // returns once that connection has been accepted, rather than
            // looping forever over a live listener.
            let accept_thread = std::thread::spawn(move || {
                accept_loop(&daemon_for_loop, listener.incoming().take(1));
            });

            let client = UnixStream::connect(&sock_path).unwrap();
            // `connection_opened()` runs synchronously inside accept_loop,
            // BEFORE it spawns the per-connection worker thread -- so by
            // the time this join() unblocks, the daemon MUST already count
            // the connection. A sleep-then-check assertion could still pass
            // under the old racy ordering (connection_opened() inside the
            // spawned thread) if that thread happened to get scheduled in
            // time; this join-based ordering cannot pass under the old
            // ordering, because join() only returns after accept_loop's own
            // synchronous call already ran.
            accept_thread.join().unwrap();
            assert!(
                !daemon.is_idle(0.0),
                "a connection accepted via the real accept path must be counted before accept_loop returns"
            );

            drop(client);
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
            loop {
                if daemon.is_idle(0.0) {
                    break;
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "daemon never became idle after the connection closed"
                );
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        });
    }

    #[test]
    fn accept_loop_logs_and_continues_past_a_transient_accept_error() {
        let daemon = Daemon::new();
        let (fake_accepted_stream, fake_peer) = UnixStream::pair().unwrap();

        let incoming: Vec<std::io::Result<UnixStream>> = vec![
            Err(std::io::Error::new(
                std::io::ErrorKind::ConnectionAborted,
                "induced test error",
            )),
            Ok(fake_accepted_stream),
        ];

        accept_loop(&daemon, incoming.into_iter());

        // accept_loop only calls connection_opened() for the Ok item, and
        // does so synchronously -- if the preceding induced Err had wrongly
        // ended the whole loop (the pre-fix behavior propagated any single
        // accept() error out of the loop via `?`), this would never observe
        // a non-idle daemon.
        assert!(
            !daemon.is_idle(0.0),
            "accept_loop must still process the Ok stream after a preceding induced accept error"
        );

        // Unblock handle_connection's reader loop (running on its own
        // spawned thread) so the connection closes cleanly before the test
        // ends.
        drop(fake_peer);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while !daemon.is_idle(0.0) {
            assert!(
                std::time::Instant::now() < deadline,
                "connection never closed after the peer disconnected"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }
}

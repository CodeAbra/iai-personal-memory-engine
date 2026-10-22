//! Spawns the real `iai_tools_execd` binary as an independent OS process and
//! proves `background: true` semantics end-to-end: a background run returns
//! before its child exits, a client that disconnects mid-run leaves the run
//! owned by the daemon (state `detached`) until it completes, and
//! `result_fetch` against a still-growing artifact serves the current tail
//! correctly. Every fixture uses a short injected delay (hundreds of
//! milliseconds), never a real long sleep.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

struct DaemonProcess {
    child: Child,
}

impl Drop for DaemonProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn wait_for_socket(path: &std::path::Path, budget: Duration) -> bool {
    let deadline = Instant::now() + budget;
    while Instant::now() < deadline {
        if path.exists() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    false
}

/// Connects with a short retry loop: the socket FILE can exist a moment
/// before the daemon's `listen()` backlog is actually accepting, so a
/// bare `connect()` right after `wait_for_socket` can occasionally see
/// `ConnectionRefused` -- a client-side timing artifact, not a daemon
/// defect, and harmless to retry past.
fn connect_retrying(path: &std::path::Path) -> UnixStream {
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        match UnixStream::connect(path) {
            Ok(stream) => return stream,
            Err(e) if Instant::now() < deadline => {
                let _ = e;
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(e) => panic!("could not connect to {}: {e}", path.display()),
        }
    }
}

fn spawn_daemon(root: &std::path::Path) -> (DaemonProcess, PathBuf) {
    let sock_path = root.join(".execd.sock");
    let bin = env!("CARGO_BIN_EXE_iai_tools_execd");
    let child = Command::new(bin)
        .env("IAI_TOOLS_ROOT", root)
        .env("IAI_TOOLS_SOCKET_PATH", &sock_path)
        .env("IAI_TOOLS_IDLE_SECS", "1800")
        .spawn()
        .expect("failed to spawn iai_tools_execd");
    assert!(
        wait_for_socket(&sock_path, Duration::from_secs(5)),
        "socket never appeared -- lazy-start failed"
    );
    (DaemonProcess { child }, sock_path)
}

fn send_line(stream: &mut UnixStream, value: &serde_json::Value) -> serde_json::Value {
    stream
        .write_all((value.to_string() + "\n").as_bytes())
        .unwrap();
    // Scoped so the cloned read fd is dropped with this call, not held for
    // the life of the connection.
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    serde_json::from_str(&line).unwrap()
}

fn tool_run_background(
    client: &mut UnixStream,
    id: i64,
    command: &str,
    args: &[&str],
) -> serde_json::Value {
    send_line(
        client,
        &serde_json::json!({
            "jsonrpc": "2.0", "id": id, "method": "tool_run",
            "params": {"command": command, "args": args, "background": true},
        }),
    )
}

fn result_fetch(client: &mut UnixStream, id: i64, run_id: &str) -> serde_json::Value {
    send_line(
        client,
        &serde_json::json!({
            "jsonrpc": "2.0", "id": id, "method": "result_fetch",
            "params": {"run_id": run_id},
        }),
    )
}

#[test]
fn background_run_returns_before_the_child_exits_and_a_watching_client_sees_running() {
    let tmp = tempfile::tempdir().unwrap();
    let (_daemon, sock_path) = spawn_daemon(tmp.path());
    let mut client = connect_retrying(&sock_path);

    let started = Instant::now();
    let resp = tool_run_background(
        &mut client,
        1,
        "sh",
        &["-c", "sleep 0.5; echo background-done"],
    );
    let elapsed = started.elapsed();

    assert_eq!(resp["result"]["state"].as_str().unwrap(), "running");
    let run_id = resp["result"]["run_id"].as_str().unwrap().to_string();
    assert!(!run_id.is_empty());
    assert!(
        elapsed < Duration::from_millis(300),
        "tool_run with background:true must return well before the child's own delay elapses, took {elapsed:?}"
    );

    // Poll on the SAME connection (still "watching") until the run
    // completes -- a mid-run fetch on this connection must report
    // `running`, never `detached`, since nobody disconnected. Each fetch
    // only returns bytes NEW since the last one, so the output is
    // accumulated across polls rather than assumed to all arrive on
    // whichever single fetch first observes the terminal state (the
    // child's own write can land on an earlier "running" fetch, racing
    // ahead of the worker thread's terminal-state write).
    let mut saw_running = false;
    let mut collected_stdout = resp["result"]["bounded_tail"]["stdout"]
        .as_str()
        .unwrap()
        .to_string();
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let fetch = result_fetch(&mut client, 2, &run_id);
        let state = fetch["result"]["state"].as_str().unwrap().to_string();
        assert_ne!(
            state, "detached",
            "a still-connected client's own run must never read as detached"
        );
        if state == "running" {
            saw_running = true;
        }
        collected_stdout.push_str(fetch["result"]["bounded_tail"]["stdout"].as_str().unwrap());
        if state.starts_with("exited") {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "background run never reached a terminal state"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(saw_running, "must observe the running state at least once before exit");
    assert!(
        collected_stdout.contains("background-done"),
        "the full output must be observed across the accumulated polls, got: {collected_stdout:?}"
    );
}

#[test]
fn detached_run_outlives_a_disconnected_client_and_a_new_connection_fetches_the_full_output() {
    let tmp = tempfile::tempdir().unwrap();
    let (_daemon, sock_path) = spawn_daemon(tmp.path());

    let run_id = {
        let mut client_a = connect_retrying(&sock_path);
        let resp = tool_run_background(
            &mut client_a,
            1,
            "sh",
            &["-c", "sleep 0.5; printf detached-output"],
        );
        let run_id = resp["result"]["run_id"].as_str().unwrap().to_string();
        // Simulate the requesting session disconnecting mid-run: drop
        // closes the socket without waiting for the child to finish.
        drop(client_a);
        run_id
    };

    // A LATER, independent connection observes the run's ownership having
    // moved to the daemon while it is still running, then its eventual
    // full output once it completes.
    let mut client_b = connect_retrying(&sock_path);
    let mut saw_detached = false;
    let mut collected_stdout = String::new();
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let fetch = result_fetch(&mut client_b, 1, &run_id);
        let state = fetch["result"]["state"].as_str().unwrap().to_string();
        if state == "detached" {
            saw_detached = true;
        }
        collected_stdout.push_str(fetch["result"]["bounded_tail"]["stdout"].as_str().unwrap());
        if state.starts_with("exited") {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "detached run never reached a terminal state"
        );
        std::thread::sleep(Duration::from_millis(20));
    }

    assert!(
        saw_detached,
        "run must be observably `detached` while still running after the client disconnects"
    );
    assert!(
        collected_stdout.contains("detached-output"),
        "the full output must be fetchable on a new connection once a detached run completes, got: {collected_stdout:?}"
    );
}

#[test]
fn result_fetch_against_a_still_growing_artifact_returns_incremental_tail_windows() {
    let tmp = tempfile::tempdir().unwrap();
    let (_daemon, sock_path) = spawn_daemon(tmp.path());
    let mut client = connect_retrying(&sock_path);

    // Three SEPARATE `/bin/echo` processes (not the shell's own buffered
    // builtin) so each write is flushed to the artifact file the instant
    // that process exits, making the mid-run growth deterministic.
    let resp = tool_run_background(
        &mut client,
        1,
        "sh",
        &[
            "-c",
            "/bin/echo line1; sleep 0.25; /bin/echo line2; sleep 0.25; /bin/echo line3",
        ],
    );
    let run_id = resp["result"]["run_id"].as_str().unwrap().to_string();

    let mut collected = resp["result"]["bounded_tail"]["stdout"]
        .as_str()
        .unwrap()
        .to_string();
    let mut fetch_count_with_new_bytes = 0;
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        std::thread::sleep(Duration::from_millis(60));
        let fetch = result_fetch(&mut client, 2, &run_id);
        let chunk = fetch["result"]["bounded_tail"]["stdout"].as_str().unwrap();
        if !chunk.is_empty() {
            fetch_count_with_new_bytes += 1;
            collected.push_str(chunk);
        }
        if fetch["result"]["state"].as_str().unwrap().starts_with("exited") {
            break;
        }
        assert!(Instant::now() < deadline, "still-growing run never completed");
    }

    assert!(
        fetch_count_with_new_bytes >= 2,
        "at least two separate fetches must observe newly-appended bytes mid-run, saw {fetch_count_with_new_bytes}"
    );
    assert!(collected.contains("line1"));
    assert!(collected.contains("line2"));
    assert!(collected.contains("line3"));
}

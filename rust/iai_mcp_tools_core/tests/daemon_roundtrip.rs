//! Spawns the real `iai_tools_execd` binary as an independent OS process
//! and proves the socket contract end-to-end: cold start (nothing is
//! running until this test execs the binary), a `tool_run`/`result_fetch`
//! round trip, and idle self-termination on a short injected window.

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

fn spawn_daemon(root: &std::path::Path, idle_secs: &str) -> (DaemonProcess, PathBuf) {
    let sock_path = root.join(".execd.sock");
    let bin = env!("CARGO_BIN_EXE_iai_tools_execd");
    let child = Command::new(bin)
        .env("IAI_TOOLS_ROOT", root)
        .env("IAI_TOOLS_SOCKET_PATH", &sock_path)
        .env("IAI_TOOLS_IDLE_SECS", idle_secs)
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
    // Scoped so the cloned read fd is dropped with this call, not held
    // for the life of the connection.
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    serde_json::from_str(&line).unwrap()
}

#[test]
fn lazy_start_tool_run_then_result_fetch_over_the_socket() {
    let tmp = tempfile::tempdir().unwrap();
    let (_daemon, sock_path) = spawn_daemon(tmp.path(), "1800");

    let mut client = UnixStream::connect(&sock_path).unwrap();
    let run_resp = send_line(
        &mut client,
        &serde_json::json!({
            "jsonrpc": "2.0", "id": 1, "method": "tool_run",
            "params": {"command": "echo", "args": ["hello-roundtrip"]},
        }),
    );
    let run_id = run_resp["result"]["run_id"].as_str().unwrap().to_string();
    assert!(run_resp["result"]["bounded_tail"]["stdout"]
        .as_str()
        .unwrap()
        .contains("hello-roundtrip"));

    let fetch_resp = send_line(
        &mut client,
        &serde_json::json!({
            "jsonrpc": "2.0", "id": 2, "method": "result_fetch",
            "params": {"run_id": run_id},
        }),
    );
    // The artifact was already fully served by tool_run's own initial
    // fetch -- a re-serve returns no NEW bytes, proving no re-run.
    assert_eq!(
        fetch_resp["result"]["bounded_tail"]["stdout"]
            .as_str()
            .unwrap(),
        ""
    );
}

#[test]
fn daemon_self_terminates_after_a_short_idle_window_with_no_traffic() {
    let tmp = tempfile::tempdir().unwrap();
    let (mut daemon, _sock_path) = spawn_daemon(tmp.path(), "0.2");

    let deadline = Instant::now() + Duration::from_secs(5);
    let mut exited = false;
    while Instant::now() < deadline {
        if let Ok(Some(_status)) = daemon.child.try_wait() {
            exited = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(
        exited,
        "daemon did not self-terminate within the idle-window bound"
    );
}

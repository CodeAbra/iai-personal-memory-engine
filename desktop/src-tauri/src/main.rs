#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tauri::{AppHandle, Manager, RunEvent, WebviewUrl, WebviewWindow, WebviewWindowBuilder};

const DEFAULT_PORT: u16 = 4477;
const MAIN_WINDOW: &str = "main";
// The server itself waits out a nightly consolidation window before it can
// attach to the store; the window is bounded by the daemon's own backoff.
const BOOT_TIMEOUT: Duration = Duration::from_secs(300);

struct AppState {
    child: Mutex<Option<Child>>,
    // False whenever the live window still shows the local splash and must be
    // pointed at the server; set true once navigated so the supervisor does
    // not reload the page (and wipe UI state) on every poll. Reset on server
    // loss and on window re-creation so the fresh page gets navigated again.
    navigated: AtomicBool,
}

fn port() -> u16 {
    std::env::var("IAI_BRAIN_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_PORT)
}

fn dashboard_reachable(port: u16) -> bool {
    TcpStream::connect_timeout(
        &std::net::SocketAddr::from(([127, 0, 0, 1], port)),
        Duration::from_millis(400),
    )
    .is_ok()
}

/// The dashboard page marker checked before the webview is ever pointed at
/// the port — identity, not just liveness. Any process squatting the port
/// must never be rendered inside the app window.
const DASHBOARD_MARKER: &str = "cybernetic brain";

fn response_is_dashboard(text: &str) -> bool {
    // The 200 must be on the STATUS LINE — a foreign page whose BODY happens
    // to contain " 200 " and the marker words must not be accepted.
    let status_line = text.lines().next().unwrap_or("");
    status_line.starts_with("HTTP/1.")
        && status_line.contains(" 200 ")
        && text.contains(DASHBOARD_MARKER)
}

/// GET / is served statically by the dashboard (no store read), so this stays
/// truthful even while the daemon grinds through consolidation.
fn dashboard_identity_ok(port: u16) -> bool {
    use std::io::{Read, Write};
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let Ok(mut s) = TcpStream::connect_timeout(&addr, Duration::from_millis(400)) else {
        return false;
    };
    let _ = s.set_read_timeout(Some(Duration::from_millis(1500)));
    let _ = s.set_write_timeout(Some(Duration::from_millis(400)));
    let req = format!("GET / HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    if s.write_all(req.as_bytes()).is_err() {
        return false;
    }
    let mut buf = Vec::new();
    let _ = s.take(262_144).read_to_end(&mut buf);
    response_is_dashboard(&String::from_utf8_lossy(&buf))
}

/// Poison-tolerant lock: a panicking holder must not freeze the supervisor —
/// the exact failure it exists to prevent. The guarded state (a child handle)
/// stays meaningful even after a holder panicked.
fn lock_child(m: &Mutex<Option<Child>>) -> std::sync::MutexGuard<'_, Option<Child>> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The respawn decision: clear the slot when its child has already exited so
/// the supervisor spawns a fresh one. A still-running child (or an empty
/// slot) is left alone. Errors from try_wait keep the child — killing or
/// double-spawning on a transient wait error is worse than one late respawn.
fn clear_exited_child(guard: &mut Option<Child>) -> bool {
    if let Some(child) = guard.as_mut() {
        if matches!(child.try_wait(), Ok(Some(_))) {
            *guard = None;
            return true;
        }
    }
    false
}

/// The exit sequence: kill + reap whatever child the slot holds.
fn kill_child(guard: &mut Option<Child>) {
    if let Some(mut child) = guard.take() {
        let _ = child.kill();
        let _ = child.wait();
    }
}

fn find_iai() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("IAI_BIN") {
        let p = PathBuf::from(explicit);
        if p.is_file() {
            return Some(p);
        }
    }
    let home = std::env::var("HOME").map(PathBuf::from).ok();
    // GUI apps get a minimal PATH; the CLI-path cache written at install time
    // is the reliable pointer to the venv that actually has iai.
    if let Some(h) = &home {
        if let Ok(cached) = std::fs::read_to_string(h.join(".iai-mcp/.cli-path")) {
            let cached = PathBuf::from(cached.trim());
            if let Some(dir) = cached.parent() {
                let sibling = dir.join("iai");
                if sibling.is_file() {
                    return Some(sibling);
                }
            }
            if cached.is_file() {
                return Some(cached);
            }
        }
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(h) = &home {
        candidates.push(h.join(".local/bin/iai"));
    }
    candidates.push(PathBuf::from("/usr/local/bin/iai"));
    candidates.push(PathBuf::from("/opt/homebrew/bin/iai"));
    if let Ok(path_var) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path_var) {
            candidates.push(dir.join("iai"));
        }
    }
    candidates.into_iter().find(|p| p.is_file())
}

fn spawn_dashboard(iai: &PathBuf, port: u16) -> std::io::Result<Child> {
    Command::new(iai)
        .args(["brain", "--no-open", "--port", &port.to_string()])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
}

fn build_main_window(app: &AppHandle) -> tauri::Result<WebviewWindow> {
    WebviewWindowBuilder::new(app, MAIN_WINDOW, WebviewUrl::App("index.html".into()))
        .title("IAI Brain")
        .inner_size(1360.0, 860.0)
        .min_inner_size(900.0, 600.0)
        .build()
}

fn splash_error(app: &AppHandle, message: &str) {
    if let Some(w) = app.get_webview_window(MAIN_WINDOW) {
        let _ = w.eval(&format!(
            "window.showError && window.showError({});",
            json_escape(message)
        ));
    }
}

fn splash_note(app: &AppHandle, message: &str) {
    if let Some(w) = app.get_webview_window(MAIN_WINDOW) {
        let _ = w.eval(&format!(
            "var m = document.getElementById('msg'); if (m) m.textContent = {};",
            json_escape(message)
        ));
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// Point the live window at the running server, but only once per page load —
// re-navigating an already-attached window would reload it and wipe UI state.
fn navigate_if_needed(app: &AppHandle, state: &AppState, port: u16) {
    if state.navigated.load(Ordering::SeqCst) {
        return;
    }
    if !dashboard_identity_ok(port) {
        // Reachable but foreign: never render an unknown responder in the
        // memory window. The splash stays up with an actionable line.
        splash_error(
            app,
            "another program answers the memory port — close it, or set \
             IAI_BRAIN_PORT to a free port",
        );
        return;
    }
    if let Some(w) = app.get_webview_window(MAIN_WINDOW) {
        if let Ok(url) = format!("http://127.0.0.1:{port}/").parse() {
            let _ = w.navigate(url);
            state.navigated.store(true, Ordering::SeqCst);
        }
    }
}

fn supervise(app: AppHandle) {
    // Supervisor, not a one-shot boot: the window must never freeze on a dead
    // page — a crashed/killed server is respawned and the (possibly
    // re-created) webview re-navigated, forever.
    let port = port();
    loop {
        if dashboard_reachable(port) {
            let state = app.state::<AppState>();
            navigate_if_needed(&app, &state, port);
            std::thread::sleep(Duration::from_secs(2));
            continue;
        }
        // Server is gone: whatever the window shows is dead, so the next
        // successful boot must re-point it.
        {
            let state = app.state::<AppState>();
            state.navigated.store(false, Ordering::SeqCst);
            let mut guard = lock_child(&state.child);
            clear_exited_child(&mut guard);
            if guard.is_none() {
                match find_iai() {
                    Some(iai) => match spawn_dashboard(&iai, port) {
                        Ok(child) => {
                            *guard = Some(child);
                        }
                        Err(e) => {
                            drop(guard);
                            splash_error(&app, &format!("could not launch the memory server: {e}"));
                            std::thread::sleep(Duration::from_secs(5));
                            continue;
                        }
                    },
                    None => {
                        drop(guard);
                        splash_error(
                            &app,
                            "iai CLI not found — install iai-mcp, or set IAI_BIN \
                             to the iai executable path",
                        );
                        std::thread::sleep(Duration::from_secs(15));
                        continue;
                    }
                }
            }
        }
        let deadline = Instant::now() + BOOT_TIMEOUT;
        let note_after = Instant::now() + Duration::from_secs(20);
        let mut noted = false;
        while Instant::now() < deadline {
            if dashboard_reachable(port) {
                let state = app.state::<AppState>();
                navigate_if_needed(&app, &state, port);
                break;
            }
            // A child that died while we wait must not blind the supervisor
            // for the whole boot window — bail out so the outer loop respawns.
            {
                let state = app.state::<AppState>();
                let mut guard = lock_child(&state.child);
                if clear_exited_child(&mut guard) {
                    break;
                }
            }
            if !noted && Instant::now() > note_after {
                noted = true;
                splash_note(
                    &app,
                    "still waking — if macOS asks to allow access to your \
                     folders, allow it: the memory lives there",
                );
            }
            std::thread::sleep(Duration::from_millis(300));
        }
        std::thread::sleep(Duration::from_secs(3));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    fn serve_once(body: &'static str) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            if let Ok((mut s, _)) = listener.accept() {
                let mut buf = [0u8; 1024];
                let _ = s.read(&mut buf);
                let _ = s.write_all(body.as_bytes());
            }
        });
        port
    }

    #[test]
    fn identity_accepts_the_dashboard_page() {
        let port = serve_once(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nConnection: close\r\n\r\n\
             <title>IAI — cybernetic brain</title>",
        );
        assert!(dashboard_identity_ok(port));
    }

    #[test]
    fn identity_rejects_a_foreign_responder() {
        let port = serve_once(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nConnection: close\r\n\r\n\
             <title>some other local app</title>",
        );
        assert!(!dashboard_identity_ok(port));
    }

    #[test]
    fn identity_rejects_non_200() {
        let port =
            serve_once("HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\ncybernetic brain");
        assert!(!dashboard_identity_ok(port));
    }

    #[test]
    fn identity_false_when_nothing_listens() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        drop(l);
        assert!(!dashboard_identity_ok(port));
    }

    #[test]
    fn response_matcher_requires_status_and_marker() {
        assert!(response_is_dashboard(
            "HTTP/1.0 200 OK\r\n\r\n...cybernetic brain..."
        ));
        assert!(!response_is_dashboard("cybernetic brain, no http status"));
        assert!(!response_is_dashboard("HTTP/1.1 500 err\r\n\r\ncybernetic brain"));
        assert!(!response_is_dashboard("HTTP/1.1 200 OK\r\n\r\nsomething else"));
        // A 200 in the BODY of a non-200 response must not be accepted.
        assert!(!response_is_dashboard(
            "HTTP/1.1 403 Forbidden\r\n\r\nsaw code 200 once; cybernetic brain"
        ));
    }

    #[test]
    fn exited_child_is_cleared_for_respawn() {
        let child = Command::new("/bin/sh").args(["-c", "exit 0"]).spawn().unwrap();
        let mut slot = Some(child);
        // Reap window: the child needs a moment to exit.
        for _ in 0..50 {
            if clear_exited_child(&mut slot) {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(slot.is_none(), "an exited child must clear the slot");
    }

    #[test]
    fn running_child_is_kept_then_killed() {
        let child = Command::new("/bin/sleep").arg("30").spawn().unwrap();
        let mut slot = Some(child);
        assert!(
            !clear_exited_child(&mut slot),
            "a live child must never be cleared"
        );
        assert!(slot.is_some());
        kill_child(&mut slot);
        assert!(slot.is_none(), "kill_child must take and reap the child");
    }

    #[test]
    fn empty_slot_is_a_noop_for_both_helpers() {
        let mut slot: Option<Child> = None;
        assert!(!clear_exited_child(&mut slot));
        kill_child(&mut slot);
        assert!(slot.is_none());
    }

    #[test]
    fn spawn_kill_cycle_with_a_real_process() {
        // The supervisor's authority end-to-end on a stand-in binary: spawn,
        // observe alive, kill, observe cleared — no zombie left behind.
        let mut slot = Some(
            Command::new("/bin/sleep")
                .arg("30")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap(),
        );
        let pid = slot.as_ref().unwrap().id();
        assert!(!clear_exited_child(&mut slot));
        kill_child(&mut slot);
        // After kill+wait the pid must be reaped: signalling it fails.
        let alive = unsafe { libc_kill_probe(pid as i32) };
        assert!(!alive, "killed child must be fully reaped");
    }

    unsafe fn libc_kill_probe(pid: i32) -> bool {
        extern "C" {
            fn kill(pid: i32, sig: i32) -> i32;
        }
        kill(pid, 0) == 0
    }

    #[test]
    fn json_escape_neutralizes_js_string_breakouts() {
        assert_eq!(json_escape("a\"b\\c\nd"), "\"a\\\"b\\\\c\\nd\"");
        assert_eq!(json_escape("\u{1}"), "\"\\u0001\"");
    }

    #[test]
    fn poisoned_child_lock_recovers() {
        let m = Mutex::new(None::<Child>);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _g = m.lock().unwrap();
            panic!("poison the mutex");
        }));
        assert!(m.is_poisoned());
        assert!(lock_child(&m).is_none());
    }
}

fn main() {
    tauri::Builder::default()
        .manage(AppState {
            child: Mutex::new(None),
            navigated: AtomicBool::new(false),
        })
        .setup(|app| {
            build_main_window(&app.handle().clone())?;
            let handle = app.handle().clone();
            std::thread::spawn(move || supervise(handle));
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building IAI Brain")
        .run(|app, event| match event {
            // macOS keeps the process alive after the last window closes; a
            // Dock/Finder re-activation must bring a window back, or the app
            // looks broken ("won't open"). Re-create it and let the supervisor
            // re-point it at the server on its next tick.
            #[cfg(target_os = "macos")]
            RunEvent::Reopen {
                has_visible_windows,
                ..
            } => {
                if !has_visible_windows && app.get_webview_window(MAIN_WINDOW).is_none() {
                    if build_main_window(app).is_ok() {
                        app.state::<AppState>()
                            .navigated
                            .store(false, Ordering::SeqCst);
                    }
                }
            }
            RunEvent::Exit => {
                let state = app.state::<AppState>();
                kill_child(&mut lock_child(&state.child));
            }
            _ => {}
        });
}

from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil
import pytest


REPO = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX subprocess + AF_UNIX socket; HOME isolation pattern",
)


@pytest.fixture
def iai_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-drain-passphrase")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp" / "lancedb"))

    import keyring.core

    keyring.core._keyring_backend = None
    yield tmp_path
    keyring.core._keyring_backend = None


def _write_deferred_jsonl(
    deferred_dir: Path,
    session_id: str,
    events: list[dict],
    *,
    version: int = 1,
    ts_suffix: int | None = None,
) -> Path:
    deferred_dir.mkdir(parents=True, exist_ok=True)
    suffix = ts_suffix if ts_suffix is not None else int(time.time())
    out = deferred_dir / f"{session_id}-{suffix}.jsonl"
    header = {
        "version": version,
        "deferred_at": "2026-04-26T00:00:00Z",
        "session_id": session_id,
        "cwd": "/tmp",
    }
    lines = [json.dumps(header)] + [json.dumps(e) for e in events]
    out.write_text("\n".join(lines) + "\n")
    return out


def _make_event(text: str, role: str = "user") -> dict:
    return {
        "text": text,
        "cue": f"test cue: {text[:24]}",
        "tier": "episodic",
        "role": role,
        "ts": "2026-04-26T00:00:00Z",
    }


def _open_isolated_store():
    from iai_mcp.store import MemoryStore

    return MemoryStore()


def test_drain_consumes_jsonl_and_deletes_file(iai_home):
    from iai_mcp.capture import drain_deferred_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    events = [
        _make_event("Areg said: drain test event one — must be at least 12 chars"),
        _make_event("assistant reply with sufficient length to pass MIN_CAPTURE", role="assistant"),
        _make_event("third event for the round-trip drain count assertion"),
    ]
    fpath = _write_deferred_jsonl(deferred_dir, "session-A", events)
    assert fpath.exists()

    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    assert counts["files_drained"] == 1, counts
    assert counts["files_failed"] == 0, counts
    assert counts["events_inserted"] == 3, counts
    assert counts["events_skipped_insert_failed"] == 0, counts
    assert not fpath.exists(), "deferred file must be unlinked after drain"

    n_rows = store.db.open_table("records").count_rows()
    assert n_rows >= 3, f"expected ≥3 records inserted, got {n_rows}"


def test_drain_handles_malformed_event_line(iai_home):
    from iai_mcp.capture import drain_deferred_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)

    fpath = deferred_dir / "session-B-12345.jsonl"
    fpath.write_text(
        json.dumps({
            "version": 1,
            "deferred_at": "2026-04-26T00:00:00Z",
            "session_id": "session-B",
            "cwd": "/tmp",
        }) + "\n"
        + json.dumps(_make_event("first valid event with adequate length")) + "\n"
        + "this line is not valid JSON {{{ broken\n"
        + json.dumps(_make_event("never reached because file-level error")) + "\n"
    )
    assert fpath.exists()

    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    assert counts["files_failed"] == 1, counts
    assert counts["files_drained"] == 0, counts
    assert not fpath.exists(), "original must be renamed away on per-file error"
    failed = list(deferred_dir.glob("session-B-12345.failed-*.jsonl"))
    assert len(failed) == 1, f"expected exactly 1 .failed-* file, got {failed}"


def test_drain_skips_future_version(iai_home):
    from iai_mcp.capture import drain_deferred_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    fpath = _write_deferred_jsonl(
        deferred_dir,
        "session-C",
        [_make_event("event from a future format version that we cannot parse")],
        version=99,
    )

    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    assert counts["files_drained"] == 0, counts
    assert counts["files_failed"] == 0, counts
    assert counts["events_inserted"] == 0, counts
    assert counts["events_skipped_insert_failed"] == 0, counts
    assert fpath.exists(), "version>1 file must remain for a future daemon to handle"
    assert not list(deferred_dir.glob("*.failed-*.jsonl"))

    log_dir = iai_home / ".iai-mcp" / "logs"
    log_files = list(log_dir.glob("deferred-drain-*.log"))
    assert log_files, "drain must create a log file when it skips a future version"
    log_content = log_files[0].read_text()
    assert "skip" in log_content
    assert "session-C" in log_content
    assert "version=99" in log_content


def test_drain_no_deferred_dir(iai_home):
    from iai_mcp.capture import drain_deferred_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    assert not deferred_dir.exists()

    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    assert counts["files_drained"] == 0, counts
    assert counts["files_failed"] == 0, counts
    assert counts["events_inserted"] == 0, counts
    assert counts["events_skipped_insert_failed"] == 0, counts
    assert not deferred_dir.exists(), "drain should not create .deferred-captures/"


def test_drain_empty_jsonl(iai_home):
    from iai_mcp.capture import drain_deferred_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    fpath = deferred_dir / "session-E-empty.jsonl"
    fpath.write_text("")
    assert fpath.exists()

    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    assert counts["files_drained"] == 0, counts
    assert counts["files_failed"] == 0, counts
    assert counts["events_inserted"] == 0, counts
    assert counts["events_skipped_insert_failed"] == 0, counts
    assert not fpath.exists(), "0-byte file must be unlinked"


def test_drain_multiple_files_processed_in_order(iai_home):
    from iai_mcp.capture import drain_deferred_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    distinct_texts = [
        "apples are red and grow on trees in orchards across the world",
        "quantum chromodynamics describes the strong nuclear force precisely",
        "hummingbirds beat their wings about eighty times per second in flight",
    ]
    paths = []
    for i, base_ts in enumerate([1000, 2000, 3000]):
        events = [_make_event(distinct_texts[i])]
        paths.append(
            _write_deferred_jsonl(
                deferred_dir, f"session-F-{i}", events, ts_suffix=base_ts,
            )
        )
    assert all(p.exists() for p in paths)

    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    assert counts["files_drained"] == 3, counts
    assert counts["events_inserted"] == 3, counts
    assert counts["events_skipped_insert_failed"] == 0, counts
    assert counts["files_failed"] == 0, counts
    for p in paths:
        assert not p.exists(), f"{p} must be unlinked after drain"


def test_drain_partial_insert_failure_preserves_file(iai_home, monkeypatch):
    from iai_mcp.capture import drain_deferred_captures
    from iai_mcp.hippo import HippoDB

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"

    fpath = _write_deferred_jsonl(
        deferred_dir,
        "session-H",
        [
            _make_event("first good event with adequate length here"),
            _make_event("INSERT_FAIL_SENTINEL_07_9 — this event triggers a failure"),
            _make_event("third good event after the failing one in the middle"),
        ],
        ts_suffix=42,
    )
    assert fpath.exists()

    # The two-phase drain writes each genuinely-new turn as a pending row via
    # HippoDB.insert_pending_row (no synchronous embed). Fail that write for the
    # sentinel event and assert the whole file is preserved as .failed-* — the
    # per-file insert-failure contract is identical, just at the new write site.
    real_insert_pending = HippoDB.insert_pending_row

    def insert_pending_or_fail(self, *, literal_surface, **kwargs):
        if "INSERT_FAIL_SENTINEL_07_9" in literal_surface:
            raise RuntimeError("simulated pending-row write failure")
        return real_insert_pending(self, literal_surface=literal_surface, **kwargs)

    monkeypatch.setattr(HippoDB, "insert_pending_row", insert_pending_or_fail)

    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    assert not fpath.exists(), "original file must be renamed when any insert fails"
    failed_files = list(deferred_dir.glob("session-H-42.failed-*.jsonl"))
    assert len(failed_files) == 1, (
        f"expected 1 .failed-* file; got {failed_files} "
        f"(deferred_dir contents: {list(deferred_dir.iterdir())})"
    )

    assert counts["events_inserted"] == 2, counts
    assert counts["events_skipped_insert_failed"] == 1, counts
    assert counts["events_skipped_intentional"] == 0, counts
    assert counts["files_drained"] == 0, counts
    assert counts["files_failed"] == 1, counts

    log_dir = iai_home / ".iai-mcp" / "logs"
    log_files = list(log_dir.glob("deferred-drain-*.log"))
    assert log_files, "log file must record the insert-failed event"
    log_content = log_files[0].read_text()
    assert "insert-failed" in log_content
    assert "session-H" in log_content


def test_drain_intentional_skip_does_not_fail_file(iai_home):
    from iai_mcp.capture import drain_deferred_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    fpath = _write_deferred_jsonl(
        deferred_dir,
        "session-I",
        [
            _make_event("ok this is a long enough event for the min-length gate"),
            {"cue": "x", "text": "tiny", "tier": "episodic", "role": "user",
             "ts": "2026-04-26T00:00:00Z"},
        ],
        ts_suffix=43,
    )
    assert fpath.exists()

    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    assert not fpath.exists()
    assert list(deferred_dir.glob("*.failed-*.jsonl")) == []
    assert counts["files_drained"] == 1, counts
    assert counts["files_failed"] == 0, counts
    assert counts["events_inserted"] == 1, counts
    assert counts["events_skipped_intentional"] == 1, counts
    assert counts["events_skipped_insert_failed"] == 0, counts


def _spawn_daemon(sock_path: Path, store_dir: Path, home: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["IAI_DAEMON_SOCKET_PATH"] = str(sock_path)
    env["IAI_MCP_STORE"] = str(store_dir)
    env["IAI_DAEMON_IDLE_SHUTDOWN_SECS"] = "99999"
    env["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")
    env["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"
    env["IAI_MCP_CRYPTO_PASSPHRASE"] = "test-drain-integration-pass"
    return subprocess.Popen(
        [sys.executable, "-m", "iai_mcp.daemon"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_for_socket(sock_path: Path, timeout_sec: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if sock_path.exists():
            return True
        time.sleep(0.1)
    return False


def _kill_daemon_by_socket(sock_path: Path) -> None:
    target = str(sock_path)
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cl = " ".join(p.info.get("cmdline") or [])
            if "iai_mcp.daemon" not in cl:
                continue
            try:
                env = p.environ()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
            if env.get("IAI_DAEMON_SOCKET_PATH") == target:
                try:
                    p.send_signal(signal.SIGTERM)
                    p.wait(timeout=3)
                except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                    try:
                        p.send_signal(signal.SIGKILL)
                    except psutil.NoSuchProcess:
                        pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def test_daemon_main_drain_does_not_crash_on_bad_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-drain-integration-pass")
    monkeypatch.setenv("IAI_MCP_STARTUP_DEFERRED_DRAIN_GRACE_SEC", "0")

    iai_dir = tmp_path / ".iai-mcp"
    iai_dir.mkdir(parents=True, exist_ok=True)
    store_dir = iai_dir / "lancedb"
    store_dir.mkdir(parents=True, exist_ok=True)
    deferred_dir = iai_dir / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)

    bad = deferred_dir / "session-G-99999.jsonl"
    bad.write_text(
        json.dumps({"version": 1, "session_id": "session-G",
                    "deferred_at": "2026-04-26T00:00:00Z", "cwd": "/tmp"}) + "\n"
        + "totally not JSON ===invalid===\n"
    )
    assert bad.exists()

    sock_dir_ctx = tempfile.TemporaryDirectory(prefix="iai-sock-")
    sock_dir = Path(sock_dir_ctx.name)
    sock_path = sock_dir / "d.sock"

    proc = None
    try:
        proc = _spawn_daemon(
            sock_path, store_dir, home=Path(os.environ["HOME"])
        )
        assert _wait_for_socket(sock_path, timeout_sec=30), (
            f"daemon never bound socket within 30s; pid={proc.pid} "
            f"poll_status={proc.poll()}"
        )

        time.sleep(2.0)

        assert proc.poll() is None, (
            f"daemon exited unexpectedly with code {proc.returncode} — "
            f"startup-drain probably propagated an exception"
        )

        assert not bad.exists(), (
            "malformed file should have been renamed away by drain"
        )
        failed = list(deferred_dir.glob("session-G-99999.failed-*.jsonl"))
        assert len(failed) == 1, (
            f"expected exactly 1 .failed-* file, got {failed}"
        )
    finally:
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.send_signal(signal.SIGKILL)
                proc.wait(timeout=3)
        _kill_daemon_by_socket(sock_path)
        try:
            if sock_path.exists():
                sock_path.unlink()
        except OSError:
            pass
        try:
            sock_dir.rmdir()
        except OSError:
            pass
        sock_dir_ctx.cleanup()
        import keyring.core
        keyring.core._keyring_backend = None

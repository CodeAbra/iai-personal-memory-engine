from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from _socket_test_helpers import (
    daemon_endpoint,
    daemon_endpoint_ready_path,
    new_daemon_client_socket,
)


@pytest.fixture
def iai_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Path.home() reads USERPROFILE on Windows
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-crash-loop-passphrase")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp" / "hippo"))
    import keyring.core

    keyring.core._keyring_backend = None
    yield tmp_path
    keyring.core._keyring_backend = None


def _open_isolated_store():
    from iai_mcp.store import MemoryStore

    return MemoryStore()


def _make_clean_jsonl(
    deferred_dir: Path,
    session_id: str,
    ts_suffix: int,
    *,
    version: int = 1,
) -> Path:
    deferred_dir.mkdir(parents=True, exist_ok=True)
    out = deferred_dir / f"sess-{session_id}-{ts_suffix}.jsonl"
    header = {
        "version": version,
        "deferred_at": "2026-05-15T00:00:00Z",
        "session_id": session_id,
        "cwd": "/tmp",
    }
    event = {
        "text": "event text long enough to pass MIN_CAPTURE length checks",
        "cue": "test cue crash loop",
        "tier": "episodic",
        "role": "user",
        "ts": "2026-05-15T00:00:00Z",
    }
    out.write_text(json.dumps(header) + "\n" + json.dumps(event) + "\n")
    return out


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_poison_pill_quarantined_after_two_attempts(iai_home, monkeypatch):
    from iai_mcp.capture import (
        QUARANTINE_MAX_ATTEMPTS,
        drain_deferred_captures,
    )

    assert QUARANTINE_MAX_ATTEMPTS == 2

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    fpath = _make_clean_jsonl(deferred_dir, "poison", 1000000000, version=2)

    dead = _dead_pid()
    stale_1 = fpath.with_name(fpath.stem + f".processing-{dead}.jsonl")
    fpath.rename(stale_1)

    store = _open_isolated_store()

    drain_deferred_captures(store)
    crash_1 = list(deferred_dir.glob("sess-poison-1000000000.crash-1.jsonl"))
    assert len(crash_1) == 1, (
        f"first stale-PID rescan should rename to .crash-1.jsonl, "
        f"dir={list(deferred_dir.iterdir())}"
    )

    stale_2 = crash_1[0].with_name(crash_1[0].stem + f".processing-{dead}.jsonl")
    crash_1[0].rename(stale_2)

    drain_deferred_captures(store)
    crash_2 = list(deferred_dir.glob("sess-poison-1000000000.crash-2.jsonl"))
    assert len(crash_2) == 1, (
        f"second stale-PID rescan should rename to .crash-2.jsonl, "
        f"dir={list(deferred_dir.iterdir())}"
    )

    stale_3 = crash_2[0].with_name(crash_2[0].stem + f".processing-{dead}.jsonl")
    crash_2[0].rename(stale_3)

    drain_deferred_captures(store)

    quarantine_dir = deferred_dir / ".quarantine"
    assert quarantine_dir.exists(), "quarantine dir must be created on third strike"
    quarantined = list(quarantine_dir.iterdir())
    assert len(quarantined) == 1, (
        f"exactly one quarantined file expected, got {quarantined}"
    )
    assert quarantined[0].name.endswith("sess-poison-1000000000.jsonl") or (
        "sess-poison-1000000000" in quarantined[0].name
    ), quarantined[0].name


def test_quarantine_emits_warning_event(iai_home, monkeypatch):
    from iai_mcp.capture import drain_deferred_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    fpath = _make_clean_jsonl(deferred_dir, "warn", 1100000000, version=2)

    dead = _dead_pid()
    crash_2 = fpath.with_name(
        fpath.stem + ".crash-2.jsonl"
    )
    fpath.rename(crash_2)
    stale_marker = crash_2.with_name(
        crash_2.stem + f".processing-{dead}.jsonl"
    )
    crash_2.rename(stale_marker)

    write_event_calls: list[tuple[str, dict, dict]] = []

    def _stub_write_event(_store, kind, data, **kwargs):
        write_event_calls.append((kind, data, kwargs))

    import iai_mcp.events as events_mod

    monkeypatch.setattr(events_mod, "write_event", _stub_write_event)

    store = _open_isolated_store()
    drain_deferred_captures(store)

    quarantine_events = [
        c for c in write_event_calls if c[0] == "deferred_captures_quarantined"
    ]
    assert len(quarantine_events) == 1, (
        f"expected 1 deferred_captures_quarantined event, got {write_event_calls}"
    )
    kind, data, kwargs = quarantine_events[0]
    assert kwargs.get("severity") == "warning", kwargs
    assert kwargs.get("domain") == "ops", kwargs
    assert data.get("reason") == "crash_loop", data
    assert isinstance(data.get("attempts"), int), data
    assert data.get("attempts") >= 3, data
    assert "file" in data, data


def test_processing_marker_removed_on_success(iai_home, monkeypatch):
    from iai_mcp.capture import drain_deferred_captures
    import iai_mcp.capture as capture_mod

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    _make_clean_jsonl(deferred_dir, "clean", 1200000000)

    def _stub(*_args: Any, **_kwargs: Any) -> dict:
        return {"status": "inserted", "record_id": None, "reason": "ok"}

    monkeypatch.setattr(capture_mod, "capture_turn", _stub)

    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    assert counts["files_drained"] == 1, counts

    leftovers = [
        p.name
        for p in deferred_dir.iterdir()
        if ".processing-" in p.name
    ]
    assert leftovers == [], (
        f"no .processing-<pid> markers should remain after clean drain, got {leftovers}"
    )


def test_failed_attempt_retry_policy_still_holds(iai_home, monkeypatch):
    from iai_mcp.capture import drain_deferred_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    fpath = _make_clean_jsonl(deferred_dir, "fail3", 2000000000)
    attempt_3 = fpath.with_name(
        "sess-fail3-2000000000.failed-2000000000-attempt-3.jsonl"
    )
    fpath.rename(attempt_3)
    aged = time.time() - 1000
    os.utime(attempt_3, (aged, aged))

    def _stub(*_args: Any, **_kwargs: Any) -> dict:
        return {"status": "skipped", "reason": "insert-failed:test"}

    import iai_mcp.capture as capture_mod

    # The two-phase backlog drain inserts via `_drain_write_pending` (pending row
    # first, embed later); the live path uses `capture_turn`. Force both to report
    # an insert failure so the attempt/permanent-failure policy is exercised.
    monkeypatch.setattr(capture_mod, "_drain_write_pending", _stub)
    monkeypatch.setattr(capture_mod, "capture_turn", _stub)

    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    assert counts["files_failed"] == 1, counts
    perm = list(deferred_dir.glob(
        "sess-fail3-2000000000.permanent-failed-*.jsonl"
    ))
    assert len(perm) == 1, (
        f"expected 1 .permanent-failed-* after attempt-3 retry failure, got {perm}"
    )


def test_socket_binds_before_drain_completes(tmp_path, monkeypatch, request):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Path.home() reads USERPROFILE on Windows
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-bind-first-passphrase")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp" / "hippo"))
    monkeypatch.setenv("IAI_DAEMON_IDLE_SHUTDOWN_SECS", "3600")
    import keyring.core

    keyring.core._keyring_backend = None

    sock_dir = Path(tempfile.mkdtemp(prefix="iai-sock-"))
    tmp_socket = sock_dir / f"iai-test-{os.getpid()}.sock"
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_socket))

    def _cleanup_socket():
        try:
            tmp_socket.unlink()
        except (FileNotFoundError, OSError):
            pass
        try:
            sock_dir.rmdir()
        except OSError:
            pass

    request.addfinalizer(_cleanup_socket)

    drain_state = {"started": False, "finished": False}

    def _slow_drain(_store):
        drain_state["started"] = True
        time.sleep(60)
        drain_state["finished"] = True
        return {
            "files_drained": 0,
            "files_failed": 0,
            "events_inserted": 0,
            "events_reinforced": 0,
            "events_skipped_intentional": 0,
            "events_skipped_insert_failed": 0,
        }

    import iai_mcp.capture as capture_mod

    monkeypatch.setattr(capture_mod, "drain_deferred_captures", _slow_drain)

    from iai_mcp.daemon import main as daemon_main

    snapshot: dict = {}

    async def _scenario() -> bool:
        daemon_task = asyncio.create_task(daemon_main())
        try:
            deadline = time.monotonic() + 60.0
            sock_ok = False
            while time.monotonic() < deadline:
                if daemon_task.done():
                    exc = daemon_task.exception()
                    if exc is not None:
                        raise exc
                    return False
                if daemon_endpoint_ready_path(tmp_socket).exists():
                    try:
                        s = new_daemon_client_socket()
                        s.settimeout(1.0)
                        await asyncio.to_thread(s.connect, daemon_endpoint(tmp_socket))
                        s.close()
                        snapshot["bound_at"] = time.monotonic()
                        snapshot["drain_started"] = drain_state["started"]
                        snapshot["drain_finished"] = drain_state["finished"]
                        sock_ok = True
                        break
                    except (ConnectionRefusedError, FileNotFoundError, OSError):
                        pass
                await asyncio.sleep(0.1)
            return sock_ok
        finally:
            daemon_task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.gather(daemon_task, return_exceptions=True),
                    timeout=5.0,
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

    try:
        sock_ok = asyncio.run(_scenario())
        assert sock_ok, (
            f"socket {tmp_socket} did not accept a connection within 60s "
            f"while drain was busy; socket_exists={tmp_socket.exists()}, "
            f"drain_started={drain_state['started']}, "
            f"drain_finished={drain_state['finished']}"
        )
        assert not snapshot.get("drain_finished", True), (
            f"drain finished before socket bound at "
            f"t={snapshot.get('bound_at'):.2f} — startup is still blocking on it"
        )
    finally:
        keyring.core._keyring_backend = None


def test_atomic_claim_logs_generic_oserror(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Path.home() reads USERPROFILE on Windows
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "p6-1-fix-a-test-passphrase")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp" / "hippo"))
    import keyring.core

    keyring.core._keyring_backend = None

    deferred_dir = tmp_path / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    fname = "sess-fix-a-1700000000.jsonl"
    fpath = deferred_dir / fname
    header = {"version": 1, "session_id": "sess-fix-a"}
    event = {"cue": "x", "text": "y" * 32, "tier": "episodic", "role": "user"}
    fpath.write_text(json.dumps(header) + "\n" + json.dumps(event) + "\n")

    import pathlib as _pathlib

    real_replace = _pathlib.Path.replace

    def boom(self, target):
        if ".processing-" in str(target) and self == fpath:
            raise PermissionError("simulated EACCES on atomic claim")
        return real_replace(self, target)

    # The atomic claim uses Path.replace (os.replace) — not rename — so the
    # claim survives a pre-existing dest on Windows. Patch what the code calls.
    monkeypatch.setattr(_pathlib.Path, "replace", boom)

    from iai_mcp.capture import drain_deferred_captures
    from iai_mcp.store import MemoryStore

    store = MemoryStore()
    try:
        counts = drain_deferred_captures(store)
    finally:
        keyring.core._keyring_backend = None

    assert fpath.exists(), "regression: file should remain after claim failure"
    assert ".processing-" not in fpath.name, (
        f"regression: file basename should NOT have a .processing- segment, "
        f"got {fpath.name}"
    )
    assert ".crash-" not in fpath.name

    from datetime import datetime, timezone

    log_path = tmp_path / ".iai-mcp" / "logs" / (
        f"deferred-drain-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.log"
    )
    assert log_path.exists(), "deferred-drain log should be created"
    log_text = log_path.read_text()
    assert "claim-failed" in log_text, (
        f"regression: log should contain 'claim-failed', got: {log_text!r}"
    )
    assert "PermissionError" in log_text, (
        f"regression: log should name PermissionError, got: {log_text!r}"
    )

    assert counts["files_drained"] == 0
    assert counts["files_failed"] == 0


def test_strip_processing_marker_returns_false_on_rename_failure(
    tmp_path, monkeypatch
):
    from iai_mcp.capture import _strip_processing_marker

    src = tmp_path / "sess-x-1700000000.processing-99999.jsonl"
    src.write_text("{}\n")
    log_path = tmp_path / "drain.log"

    import pathlib as _pathlib

    def boom(self, target):
        raise PermissionError("simulated")

    # _strip_processing_marker uses Path.replace (os.replace), not rename.
    monkeypatch.setattr(_pathlib.Path, "replace", boom)

    new_path, ok = _strip_processing_marker(src, log_path=log_path)
    assert ok is False, "strip MUST report failure"
    assert new_path == src, "on failure, return the input path"
    assert log_path.exists(), "log_path should be written"
    assert "strip-marker-failed" in log_path.read_text()

    log_path2 = tmp_path / "absent.log"
    new_path2, ok2 = _strip_processing_marker(src)
    assert ok2 is False
    assert new_path2 == src
    assert not log_path2.exists()

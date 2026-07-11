from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def iai_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-retry-passphrase")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp" / "lancedb"))
    import keyring.core

    keyring.core._keyring_backend = None
    yield tmp_path
    keyring.core._keyring_backend = None


def _make_clean_jsonl(deferred_dir: Path, session_id: str, ts_suffix: int) -> Path:
    import json

    deferred_dir.mkdir(parents=True, exist_ok=True)
    out = deferred_dir / f"sess-{session_id}-{ts_suffix}.jsonl"
    header = {
        "version": 1,
        "deferred_at": "2026-05-13T00:00:00Z",
        "session_id": session_id,
        "cwd": "/tmp",
    }
    event = {
        "text": "event text long enough to pass MIN_CAPTURE length checks",
        "cue": "test cue retry",
        "tier": "episodic",
        "role": "user",
        "ts": "2026-05-13T00:00:00Z",
    }
    out.write_text(json.dumps(header) + "\n" + json.dumps(event) + "\n")
    return out


def _force_insert_failed(monkeypatch) -> None:
    # The backlog drain routes genuinely-new events through `_drain_write_pending`
    # (the two-phase pending-row write), not the synchronous `capture_turn`. Inject
    # the failure at the live drain path so the drain loop sees the real
    # `insert-failed:` reason that drives the failed-path promotion (attempt
    # advancement, permanent-after-3). Stubbing `capture_turn` would no longer be
    # intercepted by the live drain and the turn would land as a pending row.
    def _stub(*_args: Any, **_kwargs: Any) -> dict:
        return {"status": "skipped", "record_id": None, "reason": "insert-failed:test"}

    import iai_mcp.capture as capture_mod

    monkeypatch.setattr(capture_mod, "_drain_write_pending", _stub)


def _open_isolated_store():
    from iai_mcp.store import MemoryStore

    return MemoryStore()


def test_retry_after_backoff(iai_home, monkeypatch):
    from iai_mcp.capture import drain_deferred_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    fpath = _make_clean_jsonl(deferred_dir, "retry", 1000000000)

    attempt_1 = fpath.with_name("sess-retry-1000000000.failed-1000000000-attempt-1.jsonl")
    fpath.rename(attempt_1)
    aged = time.time() - 70
    os.utime(attempt_1, (aged, aged))

    _force_insert_failed(monkeypatch)
    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    assert counts["files_failed"] == 1, counts
    assert not attempt_1.exists(), "attempt-1 file must be renamed forward"
    attempt_2 = list(deferred_dir.glob("sess-retry-1000000000.failed-*-attempt-2.jsonl"))
    assert len(attempt_2) == 1, f"expected exactly one attempt-2 file, got {attempt_2}"


def test_permanent_after_3_attempts(iai_home, monkeypatch):
    from iai_mcp.capture import drain_deferred_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    fpath = _make_clean_jsonl(deferred_dir, "perm", 2000000000)
    attempt_3 = fpath.with_name("sess-perm-2000000000.failed-2000000000-attempt-3.jsonl")
    fpath.rename(attempt_3)
    aged = time.time() - 300
    os.utime(attempt_3, (aged, aged))

    _force_insert_failed(monkeypatch)

    write_event_calls: list[tuple[str, dict, dict]] = []

    def _stub_write_event(_store, kind, data, **kwargs):
        write_event_calls.append((kind, data, kwargs))

    import iai_mcp.events as events_mod

    monkeypatch.setattr(events_mod, "write_event", _stub_write_event)

    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    assert counts["files_failed"] == 1, counts
    assert not attempt_3.exists(), "attempt-3 file must be renamed away"

    perm = list(deferred_dir.glob("sess-perm-2000000000.permanent-failed-*.jsonl"))
    assert len(perm) == 1, f"expected 1 .permanent-failed-* file, got {perm}"

    perm_events = [c for c in write_event_calls if c[0] == "permanent_capture_failure"]
    assert len(perm_events) == 1, (
        f"expected exactly 1 permanent_capture_failure event, got {write_event_calls}"
    )
    kind, data, kwargs = perm_events[0]
    assert kwargs.get("severity") == "critical", kwargs
    assert data.get("attempts") == 3, data
    assert "file" in data and ".permanent-failed-" in data["file"], data


def test_skip_permanent_failed(iai_home, monkeypatch):
    from iai_mcp.capture import drain_deferred_captures
    import json

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    perm = deferred_dir / "sess-doomed-3000000000.permanent-failed-3000000000.jsonl"
    header = {
        "version": 1,
        "deferred_at": "2026-05-13T00:00:00Z",
        "session_id": "doomed",
        "cwd": "/tmp",
    }
    event = {
        "text": "this content stays preserved on disk forever",
        "cue": "perm cue",
        "tier": "episodic",
        "role": "user",
        "ts": "2026-05-13T00:00:00Z",
    }
    perm.write_text(json.dumps(header) + "\n" + json.dumps(event) + "\n")

    _force_insert_failed(monkeypatch)
    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    assert counts["files_drained"] == 0, counts
    assert counts["files_failed"] == 0, counts
    assert perm.exists(), "permanent-failed file must not be unlinked"
    assert perm.name.startswith("sess-doomed-3000000000.permanent-failed-"), perm


def test_clean_file_first_failure_becomes_attempt_1(iai_home, monkeypatch):
    from iai_mcp.capture import drain_deferred_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    fpath = _make_clean_jsonl(deferred_dir, "first", 4000000000)
    assert ".failed-" not in fpath.name

    _force_insert_failed(monkeypatch)
    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    assert counts["files_failed"] == 1, counts
    assert not fpath.exists(), "clean file must be renamed away on first failure"

    attempt_1 = list(deferred_dir.glob("sess-first-4000000000.failed-*-attempt-1.jsonl"))
    assert len(attempt_1) == 1, f"expected exactly one attempt-1 file, got {attempt_1}"
    attempt_other = list(deferred_dir.glob("sess-first-4000000000.failed-*-attempt-[2-9].jsonl"))
    assert len(attempt_other) == 0, (
        f"first failure must become attempt-1, not {attempt_other}"
    )


def test_legacy_failed_shape_becomes_attempt_2(iai_home, monkeypatch):
    from iai_mcp.capture import drain_deferred_captures
    import json

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    legacy = deferred_dir / "sess-legacy-5000000000.failed-5000000000.jsonl"
    header = {
        "version": 1,
        "deferred_at": "2026-05-13T00:00:00Z",
        "session_id": "legacy",
        "cwd": "/tmp",
    }
    event = {
        "text": "legacy shape from before the retry policy landed",
        "cue": "legacy cue",
        "tier": "episodic",
        "role": "user",
        "ts": "2026-05-13T00:00:00Z",
    }
    legacy.write_text(json.dumps(header) + "\n" + json.dumps(event) + "\n")
    aged = time.time() - 70
    os.utime(legacy, (aged, aged))

    _force_insert_failed(monkeypatch)
    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    assert counts["files_failed"] == 1, counts
    assert not legacy.exists(), "legacy file must be renamed forward"

    attempt_2 = list(deferred_dir.glob("sess-legacy-5000000000.failed-*-attempt-2.jsonl"))
    assert len(attempt_2) == 1, f"expected exactly one attempt-2 file, got {attempt_2}"

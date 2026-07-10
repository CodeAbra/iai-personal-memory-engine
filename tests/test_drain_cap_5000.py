from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX paths + atomic rename",
)

# The large-N cap tests seed thousands of events with UNIQUE tags and drain
# them; each drained event runs an unindexed tag lookup. On the in-tree storage
# engine that lookup is an O(N) table scan walked in Python (the native sqlite3
# backend answers it in C), so over a growing table the drain is O(N^2) and the
# large-N variants are too slow for the default correctness gate. They are
# marked slow ONLY on the engine backend: on the native backend they run in the
# default gate (fast), and on the engine backend they run under --runslow. The
# drain logic itself is correct on both — its assertions are unchanged, and the
# small-file cap test below stays in the default gate on both backends.
_ENGINE_BACKEND = os.environ.get("LILLI_STORAGE_DRIVER") == "lilli"


def _slow_on_lilli(fn):
    """Mark a test slow only on the in-tree engine backend; no-op otherwise."""
    return pytest.mark.slow(fn) if _ENGINE_BACKEND else fn


@pytest.fixture
def fast_drain_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-cap-pass")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp" / "lancedb"))
    import keyring.core
    keyring.core._keyring_backend = None

    from iai_mcp import capture as capture_mod

    def fake_capture_turn(store, *, cue="", text="", tier="episodic",
                          session_id="-", role="user", ts=None, **_):
        return {"status": "inserted", "record_id": "x", "reason": ""}

    monkeypatch.setattr(capture_mod, "capture_turn", fake_capture_turn)

    yield tmp_path
    keyring.core._keyring_backend = None


def _write_big(deferred_dir: Path, session_id: str, n_events: int, ts_suffix: int) -> Path:
    deferred_dir.mkdir(parents=True, exist_ok=True)
    path = deferred_dir / f"{session_id}-{ts_suffix}.jsonl"
    header = {
        "version": 1,
        "deferred_at": "2026-05-12T00:00:00Z",
        "session_id": session_id,
        "cwd": "/tmp",
    }
    with path.open("w") as fh:
        fh.write(json.dumps(header) + "\n")
        for i in range(n_events):
            fh.write(json.dumps({
                "text": f"event {i} with enough text content for the gate",
                "cue": f"cue-{i}",
                "tier": "episodic",
                "role": "user",
                "ts": "2026-05-12T00:00:00Z",
            }) + "\n")
    return path


def _store():
    from iai_mcp.store import MemoryStore
    return MemoryStore()


@_slow_on_lilli
def test_partial_drain_at_5000(fast_drain_env):
    from iai_mcp.capture import MAX_DRAIN_EVENTS_PER_RUN, drain_deferred_captures

    assert MAX_DRAIN_EVENTS_PER_RUN == 5000

    deferred = fast_drain_env / ".iai-mcp" / ".deferred-captures"
    big = _write_big(deferred, "big-session", n_events=6000, ts_suffix=1700000000)

    counts = drain_deferred_captures(_store())

    assert counts["events_inserted"] == 5000, counts
    assert not big.exists(), "original must be unlinked after residual is durable"
    partials = list(deferred.glob("*.partial.jsonl"))
    assert len(partials) == 1, partials
    residual = partials[0]
    lines = residual.read_text().splitlines()
    assert len(lines) == 1 + 1000, f"header + 1000 unprocessed events; got {len(lines)} lines"


@_slow_on_lilli
def test_second_pass_drains_remainder(fast_drain_env):
    from iai_mcp.capture import drain_deferred_captures

    deferred = fast_drain_env / ".iai-mcp" / ".deferred-captures"
    _write_big(deferred, "big-session", n_events=6000, ts_suffix=1700000001)

    store = _store()
    first = drain_deferred_captures(store)
    assert first["events_inserted"] == 5000, first

    second = drain_deferred_captures(store)
    assert second["events_inserted"] == 1000, second

    leftover = [p for p in deferred.iterdir() if p.suffix == ".jsonl"]
    assert leftover == [], f"deferred dir should be empty of .jsonl, got {leftover}"


def test_cap_does_not_apply_to_small_files(fast_drain_env):
    from iai_mcp.capture import drain_deferred_captures

    deferred = fast_drain_env / ".iai-mcp" / ".deferred-captures"
    small = _write_big(deferred, "small-session", n_events=100, ts_suffix=1700000002)

    counts = drain_deferred_captures(_store())

    assert counts["events_inserted"] == 100, counts
    assert counts["files_drained"] == 1, counts
    assert not small.exists()
    assert list(deferred.glob("*.partial.jsonl")) == []


@_slow_on_lilli
def test_partial_file_has_valid_header(fast_drain_env):
    from iai_mcp.capture import drain_deferred_captures

    deferred = fast_drain_env / ".iai-mcp" / ".deferred-captures"
    _write_big(deferred, "head-check", n_events=5500, ts_suffix=1700000003)

    drain_deferred_captures(_store())

    partials = list(deferred.glob("*.partial.jsonl"))
    assert len(partials) == 1
    head = json.loads(partials[0].read_text().splitlines()[0])
    assert head["version"] == 1
    assert head["session_id"] == "head-check"
    assert "cwd" in head
    assert "deferred_at" in head

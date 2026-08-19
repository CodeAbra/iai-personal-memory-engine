"""Regression coverage for the capture-transcript flush guarantee.

`cmd_capture_transcript`'s default path must flush/close the store on every
return path (success and exception unwind) so every reported insert is
actually durable, not just buffered in the process that reported it.
"""
from __future__ import annotations

import argparse
import io
import json
import platform
import uuid
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX paths + UNIX socket semantics",
)

SESSION_ID = "sess-flush-test"
_N_TURNS = 8


@pytest.fixture
def iai_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-capture-flush-passphrase")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")
    import keyring.core
    keyring.core._keyring_backend = None
    yield tmp_path
    keyring.core._keyring_backend = None


def _make_transcript(path: Path, n_turns: int = _N_TURNS) -> Path:
    """Write a JSONL transcript with n_turns alternating user/assistant turns.

    Each turn gets a distinct UUID and distinct text so every turn inserts,
    with no pattern-separation dedup between turns.
    """
    transcript_path = path / "transcript.jsonl"
    lines = []
    base_ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    for i in range(1, n_turns + 1):
        role = "user" if i % 2 == 1 else "assistant"
        ts = base_ts.replace(second=i % 60, minute=i // 60 % 60, hour=i // 3600 % 24)
        turn = {
            "type": role,
            "uuid": str(uuid.uuid4()),
            "timestamp": ts.isoformat(),
            "sessionId": SESSION_ID,
            "message": {
                "role": role,
                "content": f"Flush-gap turn {i} — {role} distinct alice content {uuid.uuid4()}",
            },
        }
        lines.append(json.dumps(turn))
    transcript_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return transcript_path


def _count_episodic_records(store) -> int:
    """Return the number of active episodic records in the store."""
    with store.db._conn_lock:
        row = store.db._conn.execute(
            "SELECT COUNT(*) FROM records"
            " WHERE tombstoned_at IS NULL"
            " AND tier = 'episodic'"
        ).fetchone()
    return int(row[0]) if row else 0


def _make_args(transcript_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        no_spawn=False,
        session_id=SESSION_ID,
        transcript_path=str(transcript_path),
        max_turns=100000,
    )


def test_capture_transcript_persists_all_reported_inserts(iai_home, tmp_path):
    """Every insert cmd_capture_transcript reports must survive process return."""
    from iai_mcp.cli._capture import cmd_capture_transcript
    from iai_mcp.store import MemoryStore

    transcript = _make_transcript(tmp_path)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_capture_transcript(_make_args(transcript))

    assert rc == 0
    counts = json.loads(buf.getvalue())
    reported = counts["inserted"]
    assert reported > 0, f"Expected at least one insert; counts={counts!r}"

    fresh_store = MemoryStore()
    try:
        persisted = _count_episodic_records(fresh_store)
    finally:
        fresh_store.close()

    assert persisted == reported, (
        f"Reported {reported} inserts but only {persisted} rows are actually "
        f"persisted -- unclosed store dropped buffered records on process return."
    )


def test_capture_transcript_flushes_on_exception(iai_home, tmp_path, monkeypatch):
    """Already-buffered turns must survive an in-flight failure after insert."""
    from iai_mcp.cli import _capture as _capture_mod
    from iai_mcp.store import MemoryStore

    transcript = _make_transcript(tmp_path)

    seen: dict = {}

    def _boom(counts):
        seen.update(counts)
        raise RuntimeError("synthetic liveness-stamp failure")

    monkeypatch.setattr(_capture_mod, "_stamp_capture_batch_liveness", _boom)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = _capture_mod.cmd_capture_transcript(_make_args(transcript))

    assert rc == 0
    assert seen.get("inserted", 0) > 0, f"Expected at least one insert; seen={seen!r}"

    fresh_store = MemoryStore()
    try:
        persisted = _count_episodic_records(fresh_store)
    finally:
        fresh_store.close()

    assert persisted == seen["inserted"], (
        f"Reported {seen['inserted']} inserts via the exception path but only "
        f"{persisted} rows are actually persisted -- the store did not flush "
        f"before the exception unwind."
    )

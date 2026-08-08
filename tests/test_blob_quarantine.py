"""Legacy machine-notification blobs must be quarantinable: journaled,
snapshotted, reversible, and blind to real memories."""
from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest

from iai_mcp.store import MemoryStore


@pytest.fixture(autouse=True)
def _passphrase(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "iai-mcp-test-passphrase")
    yield


def _capture(store, text, sid):
    from iai_mcp.capture import capture_turn

    return UUID(capture_turn(
        store, cue="", text=text, tier="episodic",
        session_id=sid, role="user", live_turn=True,
    )["record_id"])


def test_quarantine_tombstones_blobs_and_spares_memories(tmp_path):
    from iai_mcp.migrate._blob_quarantine import quarantine_notification_blobs
    from iai_mcp.store._buffers import flush_record_buffer

    store = MemoryStore(path=tmp_path / "store")
    blob = _capture(
        store,
        "<task-notification>\n<task-id>abc99</task-id>\n"
        "<status>completed</status>\n</task-notification>",
        "s-blob",
    )
    memory = _capture(
        store, "Working on Zephyrbot notification handling today.", "s-real",
    )
    flush_record_buffer(store)

    dry = quarantine_notification_blobs(
        store, apply=False, store_path=tmp_path / "store"
    )
    assert dry["mode"] == "dry-run"
    assert dry["blobs_found"] == 1
    assert dry["tombstoned"] == 0
    assert store.get(blob) is not None

    applied = quarantine_notification_blobs(
        store, apply=True, store_path=tmp_path / "store"
    )
    assert applied["tombstoned"] == 1
    assert applied["snapshot_dir"] and Path(applied["snapshot_dir"]).exists()
    journal = Path(applied["journal"])
    entries = [json.loads(line) for line in journal.read_text().splitlines()]
    assert entries[0]["record_id"] == str(blob)

    def _tombstoned_at(rid):
        with store.db._conn_lock:
            row = store.db._conn.execute(
                "SELECT tombstoned_at FROM records WHERE id = ?", (str(rid),)
            ).fetchone()
        return row["tombstoned_at"]

    assert _tombstoned_at(blob) is not None, "blob must be tombstoned"
    assert _tombstoned_at(memory) is None, "real memory must survive"

    again = quarantine_notification_blobs(
        store, apply=False, store_path=tmp_path / "store"
    )
    assert again["blobs_found"] == 0, "quarantine must be idempotent"


def test_quarantine_spares_pinned_records(tmp_path):
    from iai_mcp.migrate._blob_quarantine import quarantine_notification_blobs
    from iai_mcp.store._buffers import flush_record_buffer

    store = MemoryStore(path=tmp_path / "store")
    rid = _capture(
        store, "<task-notification>pinned for a reason</task-notification>",
        "s-pin",
    )
    flush_record_buffer(store)
    store.db.open_table("records").update(
        where=f"id = '{rid}'", values={"never_merge": 1},
    )

    dry = quarantine_notification_blobs(
        store, apply=False, store_path=tmp_path / "store"
    )
    assert dry["blobs_found"] == 0, "never_merge records are off-limits"

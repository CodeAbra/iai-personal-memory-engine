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


def _count_tombstoned_but_live(store) -> int:
    with store.db._conn_lock:
        row = store.db._conn.execute(
            "SELECT COUNT(*) AS n FROM records"
            " WHERE tombstoned_at IS NOT NULL AND live = 1"
        ).fetchone()
    return int(row["n"])


def test_quarantined_record_still_served_on_direct_recency_rail(tmp_path):
    """A write that removes a record must not leave it reachable on the
    direct-recency read rail: every write that sets ``tombstoned_at`` must
    derive ``live`` from the same value in the same write."""
    from iai_mcp.hippo import direct_recency_rows_from_store
    from iai_mcp.migrate._blob_quarantine import quarantine_notification_blobs
    from iai_mcp.store._buffers import flush_record_buffer

    store = MemoryStore(path=tmp_path / "store")
    blob = _capture(
        store,
        "<task-notification>\n<task-id>abc99</task-id>\n"
        "<status>completed</status>\n</task-notification>",
        "s-blob",
    )
    flush_record_buffer(store)

    pre_rows = direct_recency_rows_from_store(tmp_path / "store")
    pre_ids = {str(r["id"]) for r in pre_rows}
    assert str(blob) in pre_ids, (
        "pre-apply probe did not find the record on the direct-recency rail; "
        "the probe returns an empty list on any internal failure, so an "
        "empty result here is a broken test setup, never a valid pass"
    )

    quarantine_notification_blobs(store, apply=True, store_path=tmp_path / "store")

    post_rows = direct_recency_rows_from_store(tmp_path / "store")
    post_ids = {str(r["id"]) for r in post_rows}
    tombstoned_but_live = _count_tombstoned_but_live(store)

    failures: list[str] = []
    if str(blob) in post_ids:
        failures.append("removed record is still served on the direct-recency rail")
    if tombstoned_but_live != 0:
        failures.append(
            f"{tombstoned_but_live} tombstoned row(s) are still marked live "
            f"(a write that sets tombstoned_at must set live in the same write)"
        )
    assert not failures, "; ".join(failures)


def test_quarantined_record_still_served_from_warm_caches(tmp_path):
    """A record a maintenance verb removes must be evicted from the resident
    in-process caches, not just the SQLite row: a stale recency-buffer entry
    or a stale exact-cosine matrix row keeps serving a removed record until
    the process restarts."""
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
    embedding = store.get(blob).embedding

    buffered_before = {m.id for m in store._recency_buffer}
    assert str(blob) in buffered_before, (
        "warm-up did not load the record into the recency buffer; "
        "fix the warm-up, not the assertion below"
    )
    matrix_before = {str(rid) for rid, _ in store.exact_top_k(embedding, k=5)}
    assert str(blob) in matrix_before, (
        "warm-up did not load the record into the exact-cosine matrix; "
        "fix the warm-up, not the assertion below"
    )

    quarantine_notification_blobs(store, apply=True, store_path=tmp_path / "store")

    buffered_after = {m.id for m in store._recency_buffer}
    matrix_after = {str(rid) for rid, _ in store.exact_top_k(embedding, k=5)}

    failures: list[str] = []
    if str(blob) in buffered_after:
        failures.append("removed record is still resident in the recency buffer")
    if str(blob) in matrix_after:
        failures.append("removed record is still served by the exact-cosine matrix")
    # A surviving record must still come back from both structures — proves
    # the empty/absent case above is a genuine eviction, not an empty result
    # from an unrelated buffer/matrix-rebuild failure.
    if str(memory) not in buffered_after:
        failures.append(
            "the surviving record dropped out of the recency buffer too "
            "— buffer state itself is broken, not just eviction"
        )
    if str(memory) not in matrix_after:
        failures.append(
            "the surviving record dropped out of the exact-cosine matrix too "
            "— matrix rebuild itself is broken, not just eviction"
        )
    assert not failures, "; ".join(failures)


def test_quarantined_record_absent_from_recall_hits_and_anti_hits(tmp_path):
    """A record removed by the sweep must not surface through the full
    recall pipeline, as either a hit or an anti-hit."""
    from iai_mcp.migrate._blob_quarantine import quarantine_notification_blobs
    from iai_mcp.store._buffers import flush_record_buffer

    store = MemoryStore(path=tmp_path / "store")
    blob = _capture(
        store,
        "<task-notification>\n<task-id>abc99</task-id>\n"
        "<status>completed</status>\n</task-notification>",
        "s-blob",
    )
    flush_record_buffer(store)
    embedding = store.get(blob).embedding

    quarantine_notification_blobs(store, apply=True, store_path=tmp_path / "store")

    from iai_mcp import core as _core
    import iai_mcp.pipeline as _pm

    _pm._last_recall_latency_ms = 0.0
    store._build_exact_index_sync()
    resp = _core.dispatch(store, "memory_recall", {
        "cue": "abc99 task notification status completed",
        "session_id": "quarantine-recall-check",
        "budget_tokens": 2000,
        "cue_embedding": embedding,
    })
    hit_ids = {h["record_id"] for h in resp.get("hits", [])}
    anti_hit_ids = {h["record_id"] for h in resp.get("anti_hits", [])}
    assert str(blob) not in hit_ids
    assert str(blob) not in anti_hit_ids

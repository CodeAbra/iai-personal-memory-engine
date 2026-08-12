"""Data-integrity error handlers must be observable, never silent swallows.

Each test forces a failure that the production code previously discarded with a
bare ``except ...: pass``/``continue`` and asserts the anomaly now reaches a log
record (bound to logger name, level, and the distinguishing id/name token) or an
observable connection-state flag, while the availability-preserving skip/return
that keeps the awake recall path answering is deliberately retained.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _unit_vec(i: int = 0) -> list[float]:
    v = [0.0] * EMBED_DIM
    v[i % EMBED_DIM] = 1.0
    return v


def _make_record(
    *,
    text: str = "turn",
    tier: str = "episodic",
    role: str | None = None,
    embedding_pending: int = 0,
    seed: int = 0,
    tags: list[str] | None = None,
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
        aaak_index="",
        embedding=_unit_vec(seed),
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=tags or [],
        language="en",
        role=role,
        embedding_pending=embedding_pending,
    )


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))
    s = MemoryStore(str(tmp_path / "store"))
    try:
        yield s
    finally:
        try:
            s.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Read-only label-map repopulation failure — observable degraded open
# ---------------------------------------------------------------------------


def test_ro_label_map_repopulation_failure_is_observable(tmp_path, caplog, monkeypatch):
    """A read-only open whose label-map repopulation raises still succeeds, but
    logs a warning naming the store and marks the connection degraded — never a
    silent empty vector recall."""
    from iai_mcp.hippo import HippoDB, AccessMode

    # Create and close a real store so a read-only open has a valid file.
    db = HippoDB(tmp_path)
    db_path = db._db_path
    db.close()

    # Control: a healthy read-only open is NOT degraded.
    ro_ok = HippoDB(tmp_path, access_mode=AccessMode.SHARED, read_only=True)
    try:
        assert ro_ok._label_map_degraded is False
    finally:
        ro_ok.close()

    # Force the read-only repopulation to raise on the next open.
    def _boom(self):
        raise RuntimeError("forced label-map repopulation failure")

    monkeypatch.setattr(HippoDB, "_repopulate_label_map_from_sqlite", _boom)

    caplog.set_level(logging.WARNING, logger="iai_mcp.hippo._db")
    ro = HippoDB(tmp_path, access_mode=AccessMode.SHARED, read_only=True)
    try:
        assert ro._label_map_degraded is True
    finally:
        ro.close()

    hits = [
        r
        for r in caplog.records
        if r.name == "iai_mcp.hippo._db"
        and r.levelno == logging.WARNING
        and "read-only label map repopulation failed" in r.getMessage()
        and db_path in r.getMessage()
    ]
    assert hits, "expected a degraded read-only open warning naming the store"


# ---------------------------------------------------------------------------
# _from_row corrupt-row drops become logged (get_batch + recent-pending)
# ---------------------------------------------------------------------------


def test_from_row_skip_in_get_batch_is_logged(store, caplog):
    """get_batch skips a row whose _from_row raises but returns the good rows
    and emits ONE aggregated warning per call — never a silent absence and
    never a per-row flood on the awake recall path. The skipped id stays
    observable at DEBUG."""
    good = _make_record(text="good turn", seed=1)
    bad = _make_record(text="bad turn", seed=2)
    store.insert(good)
    store.insert(bad)

    orig_from_row = store._from_row

    def _selective(row):
        if row.get("id") == str(bad.id):
            raise ValueError("forced deserialization failure")
        return orig_from_row(row)

    store._from_row = _selective

    caplog.set_level(logging.DEBUG, logger="iai_mcp.store._store")
    out = store.get_batch([good.id, bad.id])

    assert good.id in out, "the good row must still be returned (availability)"
    assert bad.id not in out, "the corrupt row must be absent from the result"
    summary = [
        r
        for r in caplog.records
        if r.name == "iai_mcp.store._store"
        and r.levelno == logging.WARNING
        and "undeserializable record" in r.getMessage()
        and "get_batch" in r.getMessage()
    ]
    assert len(summary) == 1, (
        f"expected exactly one aggregated get_batch warning, got {len(summary)}"
    )
    detail = [
        r
        for r in caplog.records
        if r.name == "iai_mcp.store._store"
        and r.levelno == logging.DEBUG
        and str(bad.id) in r.getMessage()
    ]
    assert detail, "the skipped record id must stay observable at DEBUG"


def test_from_row_skip_in_recent_pending_markers_is_logged(store, caplog):
    """Both legs of the recent-pending SQL fallback keep a row whose _from_row
    raises observable at DEBUG (naming its id), then the call emits ONE
    aggregated warning — never a silent drop and never a per-row flood."""
    # Leg B: an episodic role='user' row (over-fetch leg). The role column is
    # derived from the role: tag, so it must be present for the leg to match.
    user_rec = _make_record(
        text="user leg turn", tier="episodic", seed=3, tags=["role:user"]
    )
    store.insert(user_rec)

    # Leg A: a pending row (embedding_pending=1 union leg).
    pending_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()
    store.db.insert_pending_row(
        record_id=pending_id,
        tier="episodic",
        literal_surface="pending leg turn",
        tags_json=json.dumps([]),
        provenance_json=json.dumps([]),
        created_at=now,
        updated_at=now,
    )

    corrupt_ids = {str(user_rec.id), pending_id}
    orig_from_row = store._from_row

    def _selective(row):
        if row.get("id") in corrupt_ids:
            raise ValueError("forced deserialization failure")
        return orig_from_row(row)

    store._from_row = _selective

    caplog.set_level(logging.DEBUG, logger="iai_mcp.store._store")
    result = store._recent_pending_markers_sql(16)

    got_ids = {str(r.id) for r in result}
    assert not (corrupt_ids & got_ids), "corrupt rows must be absent from the result"
    logged_ids = {
        cid
        for cid in corrupt_ids
        if any(
            r.name == "iai_mcp.store._store"
            and r.levelno == logging.DEBUG
            and "undeserializable record" in r.getMessage()
            and cid in r.getMessage()
            for r in caplog.records
        )
    }
    assert logged_ids == corrupt_ids, (
        f"both recent-pending legs must log their skipped id at DEBUG; got {logged_ids}"
    )
    summary = [
        r
        for r in caplog.records
        if r.name == "iai_mcp.store._store"
        and r.levelno == logging.WARNING
        and "undeserializable record" in r.getMessage()
        and "_recent_pending_markers_sql" in r.getMessage()
    ]
    assert len(summary) == 1, (
        f"expected exactly one aggregated recent-pending warning, got {len(summary)}"
    )


# ---------------------------------------------------------------------------
# Resident exact-cosine index feed becomes observable at all three sites
# ---------------------------------------------------------------------------


def test_exact_feed_debug_inner_upsert_failure(store, caplog, monkeypatch):
    """The inner _feed_exact_index guard logs a debug record when the resident
    matrix upsert itself raises."""

    class _RaisingIndex:
        def upsert(self, record_id, vec):
            raise RuntimeError("upsert boom")

    monkeypatch.setattr(store, "_exact_index", _RaisingIndex())
    caplog.set_level(logging.DEBUG, logger="iai_mcp.store._store")
    store._feed_exact_index("exact-inner-1", _unit_vec(0))

    hits = [
        r
        for r in caplog.records
        if r.name == "iai_mcp.store._store"
        and r.levelno == logging.DEBUG
        and "exact-index upsert failed" in r.getMessage()
        and "exact-inner-1" in r.getMessage()
    ]
    assert hits, "inner exact-index upsert failure must log a debug record"


def test_exact_feed_debug_in_buffer_flush(store, caplog, monkeypatch):
    """The buffer-flush guard also wraps argument evaluation: a row dict missing
    its embedding key raises inside the guard and must be logged."""
    from iai_mcp.store import _buffers

    class _NoopTable:
        def add(self, rows):
            return None

    monkeypatch.setattr(store.db, "open_table", lambda name: _NoopTable())
    item = {"id": "buffer-corrupt-1", "embedding_pending": 0}
    _buffers._record_buffer[id(store)] = [item]

    caplog.set_level(logging.DEBUG, logger="iai_mcp.store._buffers")
    try:
        _buffers.flush_record_buffer(store)
    finally:
        _buffers._record_buffer.pop(id(store), None)

    hits = [
        r
        for r in caplog.records
        if r.name == "iai_mcp.store._buffers"
        and r.levelno == logging.DEBUG
        and "exact-index feed failed" in r.getMessage()
        and "buffer-corrupt-1" in r.getMessage()
    ]
    assert hits, "buffer-flush exact-index feed failure must log a debug record"


def test_exact_feed_debug_on_async_flush(store, caplog, monkeypatch):
    """The async on-flush guard logs a debug record when the exact-index feed
    raises for a flushed record."""
    import asyncio

    def _raiser(*args, **kwargs):
        raise RuntimeError("async feed boom")

    # Patch before enable so the on_flushed closure captures the raising feed.
    monkeypatch.setattr(store, "_feed_exact_index", _raiser)
    asyncio.run(store.enable_async_writes(coalesce_ms=1, max_batch=1))

    rec = _make_record(text="async flush turn", seed=9)
    caplog.set_level(logging.DEBUG, logger="iai_mcp.store._store")
    store.insert(rec)
    # Draining on close flushes the queue synchronously, firing on_flushed.
    store.close()

    hits = [
        r
        for r in caplog.records
        if r.name == "iai_mcp.store._store"
        and r.levelno == logging.DEBUG
        and "exact-index feed failed on flush" in r.getMessage()
        and str(rec.id) in r.getMessage()
    ]
    assert hits, "async on-flush exact-index feed failure must log a debug record"


# ---------------------------------------------------------------------------
# Close-commit failure and abandoned lock-demote become observable
# ---------------------------------------------------------------------------


def test_close_commit_failure_is_logged(tmp_path, caplog, monkeypatch):
    """close() still completes when the connection commit raises, but the
    would-be silent data-loss-at-close is now logged."""
    from iai_mcp.hippo import HippoDB

    db = HippoDB(tmp_path)
    db_path = db._db_path

    class _CommitRaises:
        def __init__(self, real):
            self._real = real

        def commit(self):
            raise RuntimeError("commit boom")

        def __getattr__(self, name):
            return getattr(self._real, name)

    monkeypatch.setattr(db, "_conn", _CommitRaises(db._conn))
    caplog.set_level(logging.WARNING, logger="iai_mcp.hippo._db")
    db.close()  # must not raise

    hits = [
        r
        for r in caplog.records
        if r.name == "iai_mcp.hippo._db"
        and r.levelno == logging.WARNING
        and "commit at close failed" in r.getMessage()
        and db_path in r.getMessage()
    ]
    assert hits, "a commit failure at close must be logged"


def test_lock_demote_failure_is_logged(tmp_path, caplog, monkeypatch):
    """When the non-blocking shared-lock demote fails the process stays
    exclusive, the intent file is left in place, and a warning is logged."""
    import errno as _errno

    from iai_mcp import _flock as _flock_mod
    from iai_mcp.hippo import HippoDB, AccessMode

    db = HippoDB(tmp_path, access_mode=AccessMode.EXCLUSIVE)
    db_path = db._db_path
    intent = db._hippo_dir / ".consolidation-pending"
    intent.write_text("held\n")

    real_flock = _flock_mod.flock

    def _selective_flock(fd, op):
        if op & _flock_mod.LOCK_SH:
            raise OSError(_errno.EAGAIN, "forced demote failure")
        return real_flock(fd, op)

    monkeypatch.setattr(_flock_mod, "flock", _selective_flock)
    caplog.set_level(logging.WARNING, logger="iai_mcp.hippo._db")
    try:
        db.downgrade_to_shared()
        assert db._access_mode is AccessMode.EXCLUSIVE, (
            "a failed demote must leave the process exclusive"
        )
        assert intent.exists(), "the intent file must be left in place"
        hits = [
            r
            for r in caplog.records
            if r.name == "iai_mcp.hippo._db"
            and r.levelno == logging.WARNING
            and "shared-lock demote failed" in r.getMessage()
            and db_path in r.getMessage()
        ]
        assert hits, "a failed shared-lock demote must be logged"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Corrupt-header spool file is counted + logged, not a silent skip
# ---------------------------------------------------------------------------


def test_corrupt_header_spool_file_counted_and_logged(store, caplog):
    """A live spool file with a genuinely malformed header is counted and
    logged and left on disk, while well-formed files in the same directory
    still drain in the same pass."""
    from iai_mcp.capture import deferred_captures_dir, drain_active_live_captures

    deferred_dir = deferred_captures_dir()
    deferred_dir.mkdir(parents=True, exist_ok=True)

    # A corrupt, plaintext (non-encrypted, non-JSON) header line.
    corrupt = deferred_dir / "11111111-1111-4111-8111-111111111111.live.jsonl"
    corrupt.write_text("CORRUPT-HEADER-NOT-JSON !!!\n")

    # A well-formed live file with a valid header and one user event.
    good_session = "22222222-2222-4222-8222-222222222222"
    good = deferred_dir / f"{good_session}.live.jsonl"
    header = {
        "version": 1,
        "deferred_at": "2026-05-31T04:45:00.000000+00:00",
        "session_id": good_session,
        "cwd": "/tmp/test",
    }
    event = {
        "text": "a well formed live turn that must still drain",
        "cue": f"session {good_session} turn",
        "tier": "episodic",
        "role": "user",
        "ts": "2026-05-31T04:45:43.000000+00:00",
    }
    good.write_text(
        json.dumps(header, ensure_ascii=False)
        + "\n"
        + json.dumps(event, ensure_ascii=False)
        + "\n"
    )

    caplog.set_level(logging.WARNING, logger="iai_mcp.capture")
    counts = drain_active_live_captures(store, exclude_session_id="-")

    assert counts["files_corrupt"] == 1, f"corrupt header must be counted: {counts}"
    assert corrupt.exists(), "the corrupt file must be left on disk (lossless)"
    assert counts["events_inserted"] >= 1, (
        f"the well-formed file must still drain in the same pass: {counts}"
    )
    hits = [
        r
        for r in caplog.records
        if r.name == "iai_mcp.capture"
        and r.levelno == logging.WARNING
        and "corrupt spool header" in r.getMessage()
        and corrupt.name in r.getMessage()
    ]
    assert hits, "the corrupt spool header must be logged with its file name"

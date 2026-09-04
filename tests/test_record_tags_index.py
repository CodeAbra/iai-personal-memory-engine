"""Differential + seam + backfill + drain-integration coverage for the
``record_tags`` indexed tag lookup (replaces the ``tags_json LIKE '%...%'``
full-table scan in ``MemoryStore.find_record_by_tag``).

Every test in this file runs under both storage drivers: the default stdlib
driver and, where explicitly noted, ``LILLI_STORAGE_DRIVER=lilli``. Tests that
depend on the lilli engine's scan-counter introspection (``full_scan_count``)
skip cleanly under stdlib rather than failing, mirroring the project's
standing "both drivers, single file" testing convention.

Every store here lives under ``tmp_path`` — never a real or prod store.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest

from iai_mcp.hippo._db import DEFAULT_STORAGE_DRIVER

pytestmark = pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX paths; hermetic fixture uses HOME monkeypatching",
)

_LILLI = os.environ.get("LILLI_STORAGE_DRIVER", DEFAULT_STORAGE_DRIVER).lower() == "lilli"

_EMBED_DIM = 384


def _zero_embedding() -> list[float]:
    return [0.0] * _EMBED_DIM


def _make_record(text: str, tags: list[str] | None = None):
    from iai_mcp.types import MemoryRecord

    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=_zero_embedding(),
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"session_id": "test-session", "role": "user"}],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=tags or ["role:user"],
        language="en",
    )


@pytest.fixture
def rtag_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-record-tags-passphrase")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp" / "store"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "no.sock"))

    import keyring.core
    keyring.core._keyring_backend = None

    from iai_mcp import embed as embed_mod

    class _FakeEmbedder:
        DIM = _EMBED_DIM

        def embed(self, text: str) -> list[float]:
            return _zero_embedding()

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [_zero_embedding() for _ in texts]

    monkeypatch.setattr(embed_mod, "embedder_for_store", lambda store: _FakeEmbedder())

    yield tmp_path

    keyring.core._keyring_backend = None


def _open_store(tmp_path: Path):
    from iai_mcp.store import MemoryStore
    return MemoryStore()


def _get_lilli_conn(db):
    """Return the underlying lilli engine Connection, or None if unavailable."""
    conn = getattr(db, "_conn", None)
    if conn is None:
        return None
    if not hasattr(conn, "full_scan_count"):
        return None
    return conn


# ---------------------------------------------------------------------------
# 1. DIFFERENTIAL — new indexed lookup vs. the old LIKE+post-filter reference
# ---------------------------------------------------------------------------


def _old_find_record_by_tag_reference(store, tag: str) -> UUID | None:
    """The exact pre-index LIKE-scan semantics, reproduced independently of
    ``MemoryStore._find_record_by_tag_like_scan`` (which is production code
    kept as the backfill-failure fallback) so the differential has a truly
    independent oracle.
    """
    with store.db._conn_lock:
        rows = store.db._conn.execute(
            "SELECT id, tags_json FROM records WHERE tags_json LIKE :pat",
            {"pat": f"%{json.dumps(tag)}%"},
        ).fetchall()
    for row in rows:
        tags_raw = row["tags_json"] if row["tags_json"] else "[]"
        try:
            tags = json.loads(tags_raw)
        except (ValueError, TypeError):
            continue
        if tag in tags:
            raw_id = row["id"]
            if raw_id is None:
                continue
            try:
                return UUID(str(raw_id))
            except (ValueError, AttributeError):
                continue
    return None


def test_differential_overlapping_multi_unicode_substring_tags(rtag_env: Path) -> None:
    """~300 records with overlapping/multi/unicode/JSON-special-char tags —
    every probed tag must resolve identically between the new indexed lookup
    and the independent LIKE-scan reference, including the adversarial
    substring case a naive LIKE '%"tag"%' pattern exists to guard against
    (``json.dumps`` framing with quotes prevents a bare-substring false
    match, e.g. tag "ab" must not match a stored tag "xaby").
    """
    store = _open_store(rtag_env)
    try:
        rng_tags: list[str] = []
        for i in range(300):
            tag_variants = [
                "role:user",
                f"seed:{i:05d}",
            ]
            if i % 7 == 0:
                tag_variants.append("shared-substring-tag")
            if i % 11 == 0:
                # adversarial: a tag that is a naive substring of another tag
                # below (a bare (non-json-framed) LIKE '%foo%' would conflate
                # these; json.dumps-framed LIKE '%"foo"%' should not)
                tag_variants.append("ab")
            if i % 13 == 0:
                tag_variants.append("xaby")  # contains "ab" as a raw substring
            if i % 5 == 0:
                tag_variants.append("unicode-tag-é中文-\U0001F600")
            if i % 3 == 0:
                tag_variants.append('json-special-"quote"-and-\\backslash')
            store.insert(_make_record(f"seeded record {i:05d}", tags=tag_variants))
            rng_tags.append(tag_variants[-1])

        from iai_mcp.store._buffers import flush_record_buffer
        flush_record_buffer(store)

        probe_tags = [
            "role:user",
            "shared-substring-tag",
            "ab",
            "xaby",
            "unicode-tag-é中文-\U0001F600",
            'json-special-"quote"-and-\\backslash',
            "seed:00042",
            "seed:00299",
            "definitely-absent-tag-zzz",
            "",
        ]
        for tag in probe_tags:
            expected = _old_find_record_by_tag_reference(store, tag)
            actual = store.find_record_by_tag(tag)
            assert actual == expected, (
                f"tag={tag!r}: indexed lookup {actual} != LIKE-scan reference {expected}"
            )
    finally:
        store.close()


def test_differential_none_case_absent_tag(rtag_env: Path) -> None:
    store = _open_store(rtag_env)
    try:
        store.insert(_make_record("one record", tags=["role:user", "only-tag"]))
        from iai_mcp.store._buffers import flush_record_buffer
        flush_record_buffer(store)

        expected = _old_find_record_by_tag_reference(store, "nonexistent-xyz")
        actual = store.find_record_by_tag("nonexistent-xyz")
        assert expected is None
        assert actual is None
    finally:
        store.close()


def test_differential_first_match_ordering(rtag_env: Path) -> None:
    """A tag shared by multiple records must resolve to the SAME record id
    under both implementations (first-match-by-insertion-order parity).
    """
    store = _open_store(rtag_env)
    try:
        from iai_mcp.store._buffers import flush_record_buffer
        ids = []
        for i in range(8):
            r = _make_record(f"shared-tag record {i}", tags=["shared-order-tag", f"only-{i}"])
            store.insert(r)
            flush_record_buffer(store)
            ids.append(r.id)

        expected = _old_find_record_by_tag_reference(store, "shared-order-tag")
        actual = store.find_record_by_tag("shared-order-tag")
        assert expected == ids[0]
        assert actual == expected
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 2. SEAM MAINTENANCE — sync buffered, async flush, tombstone/delete parity
# ---------------------------------------------------------------------------


def test_seam_sync_buffered_insert_immediately_findable(rtag_env: Path) -> None:
    store = _open_store(rtag_env)
    try:
        from iai_mcp.store._buffers import flush_record_buffer
        r = _make_record("sync buffered insert", tags=["role:user", "sync-idem-tag"])
        store.insert(r)
        flush_record_buffer(store)

        found = store.find_record_by_tag("sync-idem-tag")
        assert found == r.id, "record must be findable by tag immediately after a synchronous flush"
    finally:
        store.close()


def test_seam_insert_pending_row_immediately_findable(rtag_env: Path) -> None:
    """``insert_pending_row`` is a separate direct-SQL write path (not
    ``HippoTable.add``) — its own record_tags maintenance seam must make the
    pending row immediately tag-findable, matching the two-phase drain's
    dedup expectation.
    """
    store = _open_store(rtag_env)
    try:
        rid = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()
        store.insert_pending(
            record_id=rid,
            tier="episodic",
            literal_surface="pending row text",
            tags_json=json.dumps(["role:user", "pending-idem-tag"]),
            provenance_json=json.dumps([{"session_id": "test-session"}]),
            created_at=now_iso,
            updated_at=now_iso,
        )
        found = store.find_record_by_tag("pending-idem-tag")
        assert found is not None
        assert str(found) == rid
    finally:
        store.close()


def test_seam_async_write_queue_flush_immediately_findable(rtag_env: Path) -> None:
    """The async write-queue path (``enable_async_writes``) routes through the
    SAME ``HippoTable.add`` choke point as the sync buffered path — its
    record must also become tag-findable once flushed.
    """
    import asyncio

    store = _open_store(rtag_env)
    try:
        loop_holder: dict = {}

        def _run_async() -> None:
            asyncio.run(store.enable_async_writes(coalesce_ms=10, max_batch=8))

        # enable_async_writes spins its own background loop/thread internally;
        # call it from a throwaway coroutine runner so this test stays sync.
        asyncio.run(store.enable_async_writes(coalesce_ms=10, max_batch=8))

        r = _make_record("async queued insert", tags=["role:user", "async-idem-tag"])
        store.insert(r)

        import time
        deadline = time.time() + 5.0
        found = None
        while time.time() < deadline:
            found = store.find_record_by_tag("async-idem-tag")
            if found is not None:
                break
            time.sleep(0.05)

        assert found == r.id, "async-flushed record must become tag-findable"
    finally:
        store.close()


def test_seam_update_by_id_does_not_touch_tags(rtag_env: Path) -> None:
    """``MemoryStore.update`` never mutates ``tags_json`` — confirming no
    additional maintenance seam is needed there (the original tag remains
    findable after an unrelated field update).
    """
    store = _open_store(rtag_env)
    try:
        from iai_mcp.store._buffers import flush_record_buffer
        r = _make_record("pre-update text", tags=["role:user", "update-idem-tag"])
        store.insert(r)
        flush_record_buffer(store)

        r.literal_surface = "post-update text"
        r.centrality = 0.42
        store.update(r)

        found = store.find_record_by_tag("update-idem-tag")
        assert found == r.id, "tag lookup must survive an unrelated field update"
    finally:
        store.close()


def test_seam_hard_delete_purges_record_tags_rows(rtag_env: Path) -> None:
    store = _open_store(rtag_env)
    try:
        from iai_mcp.store._buffers import flush_record_buffer
        r = _make_record("to be deleted", tags=["role:user", "delete-idem-tag"])
        store.insert(r)
        flush_record_buffer(store)
        assert store.find_record_by_tag("delete-idem-tag") == r.id

        store.delete(r.id)

        assert store.find_record_by_tag("delete-idem-tag") is None

        with store.db._conn_lock:
            rows = store.db._conn.execute(
                "SELECT COUNT(*) c FROM record_tags WHERE record_id = ?",
                (str(r.id),),
            ).fetchone()
        assert int(rows["c"]) == 0, "record_tags rows must be purged on hard delete"
    finally:
        store.close()


def test_seam_soft_tombstone_does_not_purge_record_tags(rtag_env: Path) -> None:
    """Soft-tombstoning (``tombstoned_at`` set, row stays in ``records``) must
    NOT remove the record from the tag index — this matches the pre-index
    LIKE-scan, which never filtered on ``tombstoned_at``
    (``test_find_record_by_tag_no_tombstone_filter`` in
    ``test_find_record_by_tag_rss.py`` pins this exact behavior).
    """
    store = _open_store(rtag_env)
    try:
        from iai_mcp.store._buffers import flush_record_buffer
        r = _make_record("soft tombstoned", tags=["role:user", "tombstone-idem-tag"])
        store.insert(r)
        flush_record_buffer(store)

        now = datetime.now(timezone.utc)
        with store.db._conn_lock:
            store.db._conn.execute(
                "UPDATE records SET tombstoned_at = ? WHERE id = ?",
                (now.isoformat(), str(r.id)),
            )
            store.db._conn.commit()

        found = store.find_record_by_tag("tombstone-idem-tag")
        assert found == r.id, (
            "soft-tombstoned records must remain tag-findable (parity with the"
            " pre-index LIKE-scan, which never filtered tombstoned_at)"
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 3. BACKFILL — pre-table store simulation + idempotence
# ---------------------------------------------------------------------------


def test_backfill_repopulates_wiped_record_tags_on_reopen(rtag_env: Path) -> None:
    store = _open_store(rtag_env)
    try:
        from iai_mcp.store._buffers import flush_record_buffer
        r = _make_record("pre-table-era record", tags=["role:user", "backfill-idem-tag"])
        store.insert(r)
        flush_record_buffer(store)
        assert store.find_record_by_tag("backfill-idem-tag") == r.id

        # Simulate a store that predates the record_tags table/backfill:
        # wipe the table and its idempotency stamp.
        with store.db._conn_lock:
            store.db._conn.execute("DELETE FROM record_tags")
            store.db._conn.execute(
                "DELETE FROM _hippo_meta WHERE key = 'record_tags_backfill_done'"
            )
            store.db._conn.commit()
    finally:
        store.close()

    store2 = _open_store(rtag_env)
    try:
        found = store2.find_record_by_tag("backfill-idem-tag")
        assert found == r.id, "backfill must repopulate record_tags on the next open"
    finally:
        store2.close()


def test_backfill_reopen_twice_is_idempotent(rtag_env: Path) -> None:
    store = _open_store(rtag_env)
    try:
        from iai_mcp.store._buffers import flush_record_buffer
        r = _make_record("idempotent backfill record", tags=["role:user", "idempotent-tag"])
        store.insert(r)
        flush_record_buffer(store)
    finally:
        store.close()

    store2 = _open_store(rtag_env)
    try:
        assert store2.find_record_by_tag("idempotent-tag") == r.id
        with store2.db._conn_lock:
            count_after_first_reopen = store2.db._conn.execute(
                "SELECT COUNT(*) c FROM record_tags WHERE record_id = ?",
                (str(r.id),),
            ).fetchone()["c"]
    finally:
        store2.close()

    store3 = _open_store(rtag_env)
    try:
        assert store3.find_record_by_tag("idempotent-tag") == r.id
        with store3.db._conn_lock:
            count_after_second_reopen = store3.db._conn.execute(
                "SELECT COUNT(*) c FROM record_tags WHERE record_id = ?",
                (str(r.id),),
            ).fetchone()["c"]
        # The record carries 2 tags ("role:user", "idempotent-tag") -> 2
        # record_tags rows. The invariant under test is stability across
        # reopens, not the absolute count.
        assert count_after_second_reopen == count_after_first_reopen == 2, (
            "re-running the backfill/open sequence must never duplicate tag rows"
        )
    finally:
        store3.close()


# ---------------------------------------------------------------------------
# 4. DRAIN INTEGRATION — find_record_by_tag -> reinforce_record, zero full scans
# ---------------------------------------------------------------------------


def test_drain_shaped_lookup_then_reinforce_end_to_end(rtag_env: Path) -> None:
    from iai_mcp.capture import _idem_tag

    store = _open_store(rtag_env)
    try:
        from iai_mcp.store._buffers import flush_record_buffer
        idem = _idem_tag("drain-session", "user", "2026-07-03T12:00:00+00:00", "drain turn text")
        r = _make_record("drain turn text", tags=[idem, "role:user"])
        store.insert(r)
        flush_record_buffer(store)

        existing_id = store.find_record_by_tag(idem)
        assert existing_id == r.id

        result = store.reinforce_record(existing_id)
        assert result is not None
    finally:
        store.close()


@pytest.mark.skipif(not _LILLI, reason="full_scan_count is a lilli-engine-only diagnostic")
def test_drain_shaped_lookup_performs_zero_full_scans_on_lilli(rtag_env: Path) -> None:
    """Warm-repeated-calls discipline: a fresh table's very first touch pays a
    one-time engine index-metadata load (measured 2 scans, independent of
    corpus size or query shape — confirmed by probing ``record_tags`` and
    ``records`` in isolation before any lookup has run). That one-time cost is
    NOT what this test guards; the oracle is that a WARM indexed lookup never
    re-triggers a scan, across an arbitrary number of subsequent warm calls.
    """
    store = _open_store(rtag_env)
    try:
        from iai_mcp.store._buffers import flush_record_buffer
        for i in range(50):
            store.insert(_make_record(f"warm corpus {i}", tags=["role:user", f"warm-tag-{i}"]))
        flush_record_buffer(store)

        lconn = _get_lilli_conn(store.db)
        if lconn is None:
            pytest.skip("full_scan_count not available on this connection type")

        # Pay the one-time cold-index-load cost outside the measured window.
        store.find_record_by_tag("warm-tag-0")

        lconn.reset_full_scan_count()
        for i in range(50):
            found = store.find_record_by_tag(f"warm-tag-{i}")
            assert found is not None
        scans = lconn.full_scan_count()

        assert scans == 0, (
            f"the indexed find_record_by_tag lookup triggered {scans} full scan(s)"
            " across 50 warm lookups — expected 0 (the whole point of the index)."
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 5. UNIQUE SAFETY — concurrent duplicate-tag upserts
# ---------------------------------------------------------------------------


def test_concurrent_duplicate_tag_upserts_no_integrity_error_no_lost_rows(
    rtag_env: Path,
) -> None:
    """Two threads inserting records that share an overlapping tag set must
    never raise IntegrityError (the record_tags UNIQUE(record_id, tag)
    constraint is upsert-safe via merge_insert-shaped
    ``ON CONFLICT DO NOTHING``) and must never lose a row: every inserted
    record's tags must resolve correctly afterward.
    """
    store = _open_store(rtag_env)
    try:
        from iai_mcp.store._buffers import flush_record_buffer

        errors: list[BaseException] = []
        n_per_thread = 15
        record_ids: dict[int, list] = {0: [], 1: []}

        def _worker(thread_idx: int) -> None:
            try:
                for i in range(n_per_thread):
                    r = _make_record(
                        f"concurrent record t{thread_idx}-{i}",
                        tags=["role:user", "shared-concurrent-tag", f"t{thread_idx}-{i}"],
                    )
                    store.insert(r)
                    record_ids[thread_idx].append(r.id)
            except BaseException as exc:  # noqa: BLE001 -- capture for the assertion below
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(idx,)) for idx in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        flush_record_buffer(store)

        assert not errors, f"concurrent tag upserts raised: {errors}"

        for thread_idx in range(2):
            for i, rid in enumerate(record_ids[thread_idx]):
                found = store.find_record_by_tag(f"t{thread_idx}-{i}")
                assert found == rid, (
                    f"lost row for t{thread_idx}-{i}: expected {rid}, got {found}"
                )

        shared_hit = store.find_record_by_tag("shared-concurrent-tag")
        assert shared_hit is not None
    finally:
        store.close()

"""Exact-key duplicate hygiene.

Two live records sharing one idem tag are the same captured event stored
twice — historical check-then-insert races plus tag-index holes let them
accumulate. Guards here pin the two halves of the fix: the buffer-aware tag
lookup (no new dups minted between an insert and its flush) and the
idem-dedup sweep (existing extras collapse to the earliest copy).
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord
from tests._helpers import stub_embedder_for_store


def _rec(text: str, i: int, tags: list[str]) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    emb = [0.0] * EMBED_DIM
    emb[i % EMBED_DIM] = 1.0
    emb[(i * 31 + 7) % EMBED_DIM] = 0.5
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=emb,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.5,
        difficulty=0.3,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=tags,
        language="en",
    )


def test_find_record_by_tag_sees_buffered_rows(tmp_path):
    """The exact-key check must see a record still sitting in the insert
    buffer — the same-batch blind spot minted duplicates."""
    store = MemoryStore(path=tmp_path)
    idem = "idem:" + "c" * 64
    rec = _rec("buffered row", 1, ["capture", idem])
    store.insert(rec)

    found = store.find_record_by_tag(idem)
    assert found == rec.id, (
        "tag lookup missed the buffered row — the dedup blind spot is back"
    )


def test_idem_dedup_collapses_extras_to_earliest(tmp_path):
    from iai_mcp.migrate import cleanup_idem_duplicates
    from iai_mcp.store._buffers import flush_record_buffer

    store = MemoryStore(path=tmp_path)
    idem = "idem:" + "d" * 64
    first = _rec("same event", 1, ["capture", idem])
    second = _rec("same event", 2, ["capture", idem])
    second.created_at = first.created_at.replace(
        microsecond=(first.created_at.microsecond + 1) % 1000000
    )
    lone = _rec("different event", 3, ["capture", "idem:" + "e" * 64])
    for r in (first, second, lone):
        store.insert(r)
    flush_record_buffer(store)

    dry = cleanup_idem_duplicates(store, apply=False)
    assert dry["groups"] == 1
    assert dry["extra_copies"] == 1
    assert dry["tombstoned"] == 0

    applied = cleanup_idem_duplicates(store, apply=True, store_path=tmp_path)
    assert applied["tombstoned"] == 1

    with store.db._conn_lock:
        rows = store.db._conn.execute(
            "SELECT id, tombstoned_at FROM records"
        ).fetchall()
    state = {str(r["id"]): r["tombstoned_at"] for r in rows}
    assert state[str(first.id)] is None, "keeper (earliest) must stay live"
    assert state[str(second.id)] is not None, "extra copy must be tombstoned"
    assert state[str(lone.id)] is None, "unrelated record must be untouched"

    rerun = cleanup_idem_duplicates(store, apply=True, store_path=tmp_path)
    assert rerun["groups"] == 0, "sweep must be idempotent"


def _count_tombstoned_but_live(store) -> int:
    with store.db._conn_lock:
        row = store.db._conn.execute(
            "SELECT COUNT(*) AS n FROM records"
            " WHERE tombstoned_at IS NOT NULL AND live = 1"
        ).fetchone()
    return int(row["n"])


def test_tombstoned_extra_copy_still_served_on_direct_recency_rail(tmp_path):
    """A write that removes a record must not leave it reachable on the
    direct-recency read rail: every write that sets ``tombstoned_at`` must
    derive ``live`` from the same value in the same write."""
    from iai_mcp.hippo import direct_recency_rows_from_store
    from iai_mcp.migrate import cleanup_idem_duplicates
    from iai_mcp.store._buffers import flush_record_buffer

    store = MemoryStore(path=tmp_path)
    idem = "idem:" + "1" * 64
    first = _rec("same event", 11, ["capture", "role:user", idem])
    second = _rec("same event", 12, ["capture", "role:user", idem])
    second.created_at = first.created_at.replace(
        microsecond=(first.created_at.microsecond + 1) % 1000000
    )
    store.insert(first)
    store.insert(second)
    flush_record_buffer(store)

    pre_rows = direct_recency_rows_from_store(tmp_path)
    pre_ids = {str(r["id"]) for r in pre_rows}
    assert str(second.id) in pre_ids, (
        "pre-apply probe did not find the extra copy on the direct-recency "
        "rail; the probe returns an empty list on any internal failure, so "
        "an empty result here is a broken test setup, never a valid pass"
    )

    cleanup_idem_duplicates(store, apply=True, store_path=tmp_path)

    post_rows = direct_recency_rows_from_store(tmp_path)
    post_ids = {str(r["id"]) for r in post_rows}
    tombstoned_but_live = _count_tombstoned_but_live(store)

    failures: list[str] = []
    if str(second.id) in post_ids:
        failures.append("removed extra copy is still served on the direct-recency rail")
    if tombstoned_but_live != 0:
        failures.append(
            f"{tombstoned_but_live} tombstoned row(s) are still marked live "
            f"(a write that sets tombstoned_at must set live in the same write)"
        )
    assert not failures, "; ".join(failures)


def test_tombstoned_extra_copy_still_served_from_warm_caches(tmp_path):
    """A record a maintenance verb removes must be evicted from the resident
    in-process caches, not just the SQLite row: a stale recency-buffer entry
    or a stale exact-cosine matrix row keeps serving a removed record until
    the process restarts."""
    from iai_mcp.migrate import cleanup_idem_duplicates
    from iai_mcp.store._buffers import flush_record_buffer

    store = MemoryStore(path=tmp_path)
    idem = "idem:" + "2" * 64
    first = _rec("same event", 21, ["capture", "role:user", idem])
    second = _rec("same event", 22, ["capture", "role:user", idem])
    second.created_at = first.created_at.replace(
        microsecond=(first.created_at.microsecond + 1) % 1000000
    )
    store.insert(first)
    store.insert(second)
    flush_record_buffer(store)
    embedding = store.get(second.id).embedding

    buffered_before = {m.id for m in store._recency_buffer}
    assert str(second.id) in buffered_before, (
        "warm-up did not load the extra copy into the recency buffer; "
        "fix the warm-up, not the assertion below"
    )
    matrix_before = {str(rid) for rid, _ in store.exact_top_k(embedding, k=5)}
    assert str(second.id) in matrix_before, (
        "warm-up did not load the extra copy into the exact-cosine matrix; "
        "fix the warm-up, not the assertion below"
    )

    cleanup_idem_duplicates(store, apply=True, store_path=tmp_path)

    buffered_after = {m.id for m in store._recency_buffer}
    matrix_after = {str(rid) for rid, _ in store.exact_top_k(embedding, k=5)}

    failures: list[str] = []
    if str(second.id) in buffered_after:
        failures.append("removed extra copy is still resident in the recency buffer")
    if str(second.id) in matrix_after:
        failures.append("removed extra copy is still served by the exact-cosine matrix")
    # The kept earliest copy must still come back from both structures —
    # proves the empty/absent case above is a genuine eviction, not an empty
    # result from an unrelated buffer/matrix-rebuild failure.
    if str(first.id) not in buffered_after:
        failures.append(
            "the surviving keeper dropped out of the recency buffer too "
            "— buffer state itself is broken, not just eviction"
        )
    if str(first.id) not in matrix_after:
        failures.append(
            "the surviving keeper dropped out of the exact-cosine matrix too "
            "— matrix rebuild itself is broken, not just eviction"
        )
    assert not failures, "; ".join(failures)


def test_tombstoned_extra_copy_absent_from_recall_hits_and_anti_hits(tmp_path, monkeypatch):
    """A record removed by the sweep must not surface through the full
    recall pipeline, as either a hit or an anti-hit."""
    from iai_mcp.migrate import cleanup_idem_duplicates
    from iai_mcp.store._buffers import flush_record_buffer

    store = MemoryStore(path=tmp_path)
    idem = "idem:" + "3" * 64
    first = _rec("same event", 31, ["capture", "role:user", idem])
    second = _rec("same event", 32, ["capture", "role:user", idem])
    second.created_at = first.created_at.replace(
        microsecond=(first.created_at.microsecond + 1) % 1000000
    )
    store.insert(first)
    store.insert(second)
    flush_record_buffer(store)
    cue_embedding = second.embedding

    cleanup_idem_duplicates(store, apply=True, store_path=tmp_path)

    from iai_mcp import core as _core
    import iai_mcp.pipeline as _pm

    class _StubEmbedder:
        def __init__(self, vec: list[float]) -> None:
            self._vec = vec

        def embed(self, _text: str) -> list[float]:
            return list(self._vec)

    stub_embedder_for_store(monkeypatch, _StubEmbedder(cue_embedding))
    _pm._last_recall_latency_ms = 0.0
    store._build_exact_index_sync()
    resp = _core.dispatch(store, "memory_recall", {
        "cue": "irrelevant text, embedder is stubbed",
        "session_id": "idem-dedup-recall-check",
        "budget_tokens": 2000,
        "cue_embedding": cue_embedding,
    })
    hit_ids = {h["record_id"] for h in resp.get("hits", [])}
    anti_hit_ids = {h["record_id"] for h in resp.get("anti_hits", [])}
    assert str(second.id) not in hit_ids
    assert str(second.id) not in anti_hit_ids

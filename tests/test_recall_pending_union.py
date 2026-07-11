from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord

def _random_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()

def _monkeypatch_env(monkeypatch, tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))
    monkeypatch.setenv("IAI_MCP_RECALL_SAMPLE_RATE", "1.0")

def test_pending_marker_surfaces_via_bounded_recency_union(tmp_path, monkeypatch):
    _monkeypatch_env(monkeypatch, tmp_path)

    store_path = tmp_path / "pending-union-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))

    for i in range(20):
        rec = _make(text=f"User filler record {i}", vec=_random_vec(1000 + i))
        store.insert(rec)

    pending_text = "User pending marker just written test content"
    pending_id = UUID(int=999_001)
    now = datetime.now(timezone.utc)
    store.db.insert_pending_row(
        record_id=str(pending_id),
        tier="episodic",
        literal_surface=pending_text,
        provenance_json="[]",
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
        tags_json="[]",
    )

    with store.db._conn_lock:
        row = store.db._conn.execute(
            "SELECT embedding_pending FROM records WHERE id = ?",
            (str(pending_id),),
        ).fetchone()
    assert row is not None, "pending record not stored"
    assert row[0] == 1, f"record not stored as pending: embedding_pending={row[0]}"

    ann_results = store.query_similar([0.1] * EMBED_DIM, k=200)
    ann_ids = {r.id for r, _ in ann_results}
    assert UUID(int=999_001) not in ann_ids, (
        "pending record should be excluded from query_similar"
    )

    all_records_call_count = {"n": 0}
    original_all_records = store.all_records

    def _spy_all_records(*args, **kwargs):
        all_records_call_count["n"] += 1
        raise AssertionError(
            "all_records() called during recall — QUAL-02 requires recent_pending_markers only"
        )

    monkeypatch.setattr(store, "all_records", _spy_all_records)

    pending_markers_call_count = {"n": 0}
    original_rpm = store.recent_pending_markers

    def _spy_rpm(n=50):
        pending_markers_call_count["n"] += 1
        return original_rpm(n=n)

    monkeypatch.setattr(store, "recent_pending_markers", _spy_rpm)

    import iai_mcp.retrieve as _retrieve
    import iai_mcp.runtime_graph_cache as _rgc

    _graph, _assignment, _rc = _retrieve.build_runtime_graph(store)
    _rgc.save(store, _assignment, _rc)

    from iai_mcp.embed import Embedder
    from iai_mcp.pipeline import recall_for_response
    import iai_mcp.pipeline as _pipeline_mod

    _pipeline_mod._last_recall_latency_ms = 0.0
    embedder = Embedder()

    response = recall_for_response(
        store=store,
        graph=_graph,
        assignment=_assignment,
        rich_club=_rc,
        embedder=embedder,
        cue="User pending marker just written",
        session_id="test-session",
        budget_tokens=2000,
        mode="concept",
    )

    hit_ids = {h.record_id for h in response.hits}
    assert UUID(int=999_001) in hit_ids, (
        f"pending record not surfaced via recency union; hit_ids={hit_ids}"
    )

    assert all_records_call_count["n"] == 0, (
        f"all_records() was called {all_records_call_count['n']} time(s) — "
        "QUAL-02 recency union must use recent_pending_markers only"
    )

    assert pending_markers_call_count["n"] >= 1, (
        "recent_pending_markers() was not called — "
        "QUAL-02 union did not fire"
    )

def test_pending_role_survives_reembed(tmp_path, monkeypatch):
    """A pending ``role:user`` row remains visible to the role query after ``reembed_pending_rows``.

    This guards the deferred-embed bypass landmine: ``insert_pending_row`` writes
    a row with ``embedding_pending=1`` and a ``tags_json`` that carries
    ``"role:user"``. When ``reembed_pending_rows`` flips ``embedding_pending`` to 0
    (making the row leave the pending-markers branch), the row must still be
    reachable via the ``role='user'`` equality predicate.

    On HEAD ``insert_pending_row`` does not set the ``role`` column (the column
    does not yet exist in the schema). After the role column implementation adds the column and wires
    it into ``insert_pending_row``, the row's ``role`` column is set at insert
    time and survives the reembed UPDATE unchanged — this test becomes GREEN.

    RED condition on HEAD: the row has ``role=NULL`` after reembed because
    ``insert_pending_row`` bypasses ``_to_row`` and never sets ``role``.
    The assertion that ``role='user'`` returns the row therefore fails.
    """
    _monkeypatch_env(monkeypatch, tmp_path)

    store_path = tmp_path / "pending-role-reembed-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))

    # Seed some regular records so the store is non-trivial.
    for i in range(5):
        rec = _make(text=f"Filler record {i}", vec=_random_vec(3000 + i))
        store.insert(rec)

    # Write a pending row tagged as role:user — simulating the two-phase
    # capture path that the fast-path drain uses.
    import json as _json
    pending_id_str = str(UUID(int=777_001))
    now = datetime.now(timezone.utc)
    role_tags = ["capture", "role:user"]
    store.db.insert_pending_row(
        record_id=pending_id_str,
        tier="episodic",
        literal_surface="Pending user turn content for role visibility test",
        provenance_json=_json.dumps([{"ts": now.isoformat(), "cue": "test", "session_id": "s1"}]),
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
        tags_json=_json.dumps(role_tags),
    )

    # Confirm the row is pending (embedding_pending=1).
    with store.db._conn_lock:
        before = store.db._conn.execute(
            "SELECT embedding_pending FROM records WHERE id = ?",
            (pending_id_str,),
        ).fetchone()
    assert before is not None, "pending row not stored"
    assert before[0] == 1, f"row not stored as pending: embedding_pending={before[0]}"

    # Run the deferred-embed pass — flips embedding_pending to 0.
    from iai_mcp.embed import Embedder
    embedder = Embedder()
    embedded_count = store.db.reembed_pending_rows(embedder)
    assert embedded_count >= 1, (
        f"reembed_pending_rows embedded {embedded_count} rows; "
        "expected at least 1 (the pending role:user row)"
    )

    # Confirm the row is no longer pending.
    with store.db._conn_lock:
        after_pending = store.db._conn.execute(
            "SELECT embedding_pending FROM records WHERE id = ?",
            (pending_id_str,),
        ).fetchone()
    assert after_pending is not None
    assert after_pending[0] == 0, (
        f"row still pending after reembed: embedding_pending={after_pending[0]}"
    )

    # Core assertion: the row must still be found by the role='user' predicate.
    # On HEAD this fails because insert_pending_row leaves role=NULL; after the role column implementation
    # sets role at insert time the column survives the reembed UPDATE → GREEN.
    #
    # We query directly on the raw connection (the same connection the production
    # recent_pending_markers uses) so there is no ambiguity about which path fired.
    try:
        with store.db._conn_lock:
            role_hits = store.db._conn.execute(
                "SELECT id FROM records WHERE tier='episodic' AND role='user'"
            ).fetchall()
        hit_ids = {r[0] for r in role_hits}
        assert pending_id_str in hit_ids, (
            f"pending role:user row not found by role='user' after reembed — "
            f"insert_pending_row does not set role (expected RED state on HEAD); "
            f"hit ids: {hit_ids!r}"
        )
    except Exception as exc:
        # If the role column does not exist at all (pre-plan-02 schema), the
        # query raises OperationalError. Treat this as the same failure: the
        # row is not findable by role, which is the RED condition.
        raise AssertionError(
            f"role='user' query failed — role column absent or row not found "
            f"(expected RED state on HEAD): {exc}"
        ) from exc


def _make_user_turn(text: str, created_at: datetime) -> MemoryRecord:
    """A non-pending episodic role:user record with an explicit ``created_at``."""
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
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
        created_at=created_at,
        updated_at=created_at,
        tags=["capture", "role:user"],
        language="en",
    )


def test_recent_pending_markers_respects_created_at_over_rowid(tmp_path, monkeypatch):
    """recent_pending_markers returns the n most-recent-by-created_at user turns.

    Byte-identity guard for the pre-166 behaviour: the old code over-fetched
    role:user candidates by rowid (``LIMIT n*4``) and re-ranked them by
    ``created_at`` DESC before truncating to ``n``. When rowid order and
    created_at order disagree (post-reembed / bulk-copy re-keying), a naive
    ``LIMIT n`` by rowid would surface the n highest-rowid rows — the n OLDEST
    by created_at — instead of the n newest. This builds exactly that adversarial
    corpus (rowid order = REVERSE of created_at order) and asserts the result is
    the n newest-by-created_at turns, in created_at-DESC order.
    """
    _monkeypatch_env(monkeypatch, tmp_path)

    store_path = tmp_path / "rowid-vs-created-at-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))

    # Build user turns whose INSERT order (→ ascending rowid) is the REVERSE of
    # created_at order: the first inserted (lowest rowid) is the NEWEST, the last
    # inserted (highest rowid) is the OLDEST.
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    total = 12
    recs: list[MemoryRecord] = []
    for i in range(total):
        # i=0 → newest (base + total minutes); i=total-1 → oldest (base + 1 min).
        ca = base + timedelta(minutes=(total - i))
        recs.append(_make_user_turn(text=f"user turn {i}", created_at=ca))
    for r in recs:
        store.insert(r)

    n = 3
    # n*4 = 12 = total, so the over-fetch gathers every role:user row; the
    # created_at re-rank then yields the true top-n by recency.
    got = store.recent_pending_markers(n=n)
    got_ids = [r.id for r in got]

    # Expected: the n records with the LARGEST created_at, created_at DESC.
    expected = sorted(recs, key=lambda r: r.created_at, reverse=True)[:n]
    expected_ids = [r.id for r in expected]

    assert got_ids == expected_ids, (
        "recent_pending_markers did not return the n newest-by-created_at turns "
        "in created_at-DESC order — a LIMIT-n-by-rowid regression drops newer rows "
        f"that sit at higher rowid positions. got={got_ids}, expected={expected_ids}"
    )

    # These are the newest-by-created_at (the first-inserted, lowest-rowid rows),
    # NOT the highest-rowid rows a naive LIMIT n by rowid would have returned.
    highest_rowid_ids = [r.id for r in recs[-n:]]
    assert set(got_ids).isdisjoint(highest_rowid_ids), (
        "result surfaced the highest-rowid (oldest-by-created_at) turns — "
        "the byte-identity over-fetch + created_at re-rank was not applied"
    )

    # The over-fetch must still ride the indexed role='user' predicate — it must
    # NOT re-introduce a full table scan. Asserted on the lilli engine, which
    # exposes the scan-count probe.
    conn = store.db._conn
    if hasattr(conn, "full_scan_count") and hasattr(conn, "reset_full_scan_count"):
        store.recent_pending_markers(n=n)  # warm the col-index
        conn.reset_full_scan_count()
        store.recent_pending_markers(n=n)
        assert conn.full_scan_count() == 0, (
            f"role:user over-fetch full-scanned records: {conn.full_scan_count()} "
            "scans — the indexed predicate was not used"
        )


def test_pending_union_does_not_double_count_ranked_pending(tmp_path, monkeypatch):
    _monkeypatch_env(monkeypatch, tmp_path)

    store_path = tmp_path / "dedup-test-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))

    for i in range(10):
        rec = _make(text=f"User dedup test filler {i}", vec=_random_vec(2000 + i))
        store.insert(rec)

    pending_id = UUID(int=888_001)
    now = datetime.now(timezone.utc)
    store.db.insert_pending_row(
        record_id=str(pending_id),
        tier="episodic",
        literal_surface="User dedup pending test",
        provenance_json="[]",
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
        tags_json="[]",
    )

    import iai_mcp.retrieve as _retrieve
    import iai_mcp.runtime_graph_cache as _rgc
    _g, _a, _rc = _retrieve.build_runtime_graph(store)
    _rgc.save(store, _a, _rc)

    from iai_mcp.embed import Embedder
    from iai_mcp.pipeline import recall_for_response
    import iai_mcp.pipeline as _pipeline_mod

    _pipeline_mod._last_recall_latency_ms = 0.0
    embedder = Embedder()

    response = recall_for_response(
        store=store,
        graph=_g,
        assignment=_a,
        rich_club=_rc,
        embedder=embedder,
        cue="User dedup pending test",
        session_id="test-session",
        budget_tokens=2000,
        mode="concept",
    )

    hit_ids = [h.record_id for h in response.hits]
    assert len(hit_ids) == len(set(hit_ids)), (
        f"duplicate record_ids in hits: {hit_ids}"
    )

from __future__ import annotations

import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path

from tests._store_raw import open_store_raw


def _zero_vector_blob(embed_dim: int) -> bytes:
    return struct.pack(f"<{embed_dim}f", *([0.0] * embed_dim))

def _make_normal_record(text: str, seed: int = 42):
    import numpy as np
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    rng = np.random.RandomState(seed=seed)
    vec = rng.randn(EMBED_DIM).tolist()
    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=vec,
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
        tags=["role:user"],
        language="en",
    )

def test_semantic_warm_embedder_returns_store_hits(
    hermetic_store: Path, monkeypatch
) -> None:
    import time
    import numpy as np
    from iai_mcp.store import MemoryStore, flush_record_buffer
    from iai_mcp.types import EMBED_DIM
    import iai_mcp.embed as _embed_mod

    from iai_mcp.semantic_recall import recall_semantic_warm

    seed_vec = np.random.RandomState(seed=10).randn(EMBED_DIM).tolist()

    store = MemoryStore(hermetic_store)
    try:
        rec = _make_normal_record("warm semantic probe distinctive text", seed=10)
        store.insert(rec)
        flush_record_buffer(store)
    finally:
        store.close()

    class _FakeEmbedder:
        DIM = EMBED_DIM

        def embed(self, text: str) -> list:
            return list(seed_vec)

    monkeypatch.setattr(_embed_mod, "embedder_for_store", lambda _store: _FakeEmbedder())

    t0 = time.monotonic()
    hits = recall_semantic_warm(
        store_root=hermetic_store,
        cue="warm semantic probe distinctive text",
        n=5,
    )
    elapsed = time.monotonic() - t0

    assert elapsed <= 1.5, f"warm semantic recall took {elapsed:.3f} s (SLO ≤1.5 s)"
    assert hits, "warm semantic recall returned no hits (must be store-backed)"
    surfaces = [h.get("literal_surface", "") or "" for h in hits]
    assert any("warm semantic probe" in s for s in surfaces), (
        "distinctive seeded turn not in warm semantic hits"
    )

def test_semantic_no_embedder_degrades_not_empty(hermetic_store: Path) -> None:
    from iai_mcp.store import MemoryStore, flush_record_buffer

    from iai_mcp.semantic_recall import recall_semantic_degraded  # type: ignore[import]

    store = MemoryStore(hermetic_store)
    try:
        rec = _make_normal_record("degraded store backed probe text store only", seed=20)
        store.insert(rec)
        flush_record_buffer(store)
    finally:
        store.close()

    hits = recall_semantic_degraded(
        store_root=hermetic_store,
        cue="degraded store backed probe",
        n=5,
    )

    assert hits is not None, "degraded recall must return a result, not None"
    assert len(hits) > 0, (
        "degraded recall must return a non-empty result "
        "(recency/temporal direct from store), not empty"
    )
    surfaces = [h.get("literal_surface", "") or "" for h in hits]
    assert any("degraded store backed probe" in s for s in surfaces), (
        "degraded result must be STORE-backed "
        "(bank cannot produce this turn — it was seeded only in the tmp store)"
    )

def test_pending_row_excluded_from_warm_semantic_until_reembed(
    hermetic_store: Path,
) -> None:
    import sqlite3 as _sqlite3

    from iai_mcp.types import EMBED_DIM
    from iai_mcp.store import MemoryStore, flush_record_buffer

    store = MemoryStore(hermetic_store)
    try:
        rec_normal = _make_normal_record("normal embedded row text", seed=30)
        store.insert(rec_normal)
        flush_record_buffer(store)
    finally:
        store.close()

    db_path = hermetic_store / "hippo" / "brain.sqlite3"
    pending_id = str(uuid.uuid4())
    zero_blob = _zero_vector_blob(EMBED_DIM)
    now = datetime.now(timezone.utc).isoformat()

    conn = open_store_raw(db_path)
    conn.row_factory = _sqlite3.Row
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(records)")}
        if "embedding_pending" not in cols:
            conn.execute(
                "ALTER TABLE records ADD COLUMN embedding_pending INTEGER NOT NULL DEFAULT 0"
            )
            conn.commit()
        conn.execute(
            "INSERT INTO records "
            "(id, tier, literal_surface, aaak_index, embedding, embedding_pending, "
            " created_at, updated_at, hv_tier, structure_hv_payload) "
            "VALUES (?, ?, ?, ?, ?, 1, ?, ?, 'bsc', x'')",
            (pending_id, "episodic", "pending probe text", "", zero_blob, now, now),
        )
        conn.commit()
    finally:
        conn.close()

    store2 = MemoryStore(hermetic_store)
    try:
        from iai_mcp.retrieve import build_runtime_graph  # type: ignore[import]
        graph, _assignment, _rich_club = build_runtime_graph(store2)

        from uuid import UUID as _UUID
        pending_uuid = _UUID(pending_id)
        graph_nodes = set(graph.nodes())
        assert pending_uuid not in graph_nodes, (
            "pending row (embedding_pending=1) must NOT be a graph node; "
            "its zero-vector would pollute cosine/graph candidates"
        )

        import numpy as np
        rng = np.random.RandomState(seed=30)
        cue_vec = rng.randn(EMBED_DIM).tolist()
        similar = store2.query_similar(cue_vec, n=10)
        similar_ids = {str(r.id) for r in similar}
        assert pending_id not in similar_ids, (
            "pending row must not appear in query_similar results "
            "(zero-vector must be excluded from ANN candidates)"
        )

        turns = store2.all_records()
        all_ids = {str(r.id) for r in turns}
        assert pending_id in all_ids, (
            "pending row must still be in all_records() — "
            "recency is embedding-independent; pending-inclusive"
        )

        recent = store2.recent_user_turns(n=20)
        recent_ids = {str(r.id) for r in recent}
        assert pending_id in recent_ids, (
            "pending row must appear in recent_user_turns() — "
            "recency must not filter by embedding_pending"
        )

    finally:
        store2.close()

    import numpy as np
    rng2 = np.random.RandomState(seed=31)
    real_vec = rng2.randn(EMBED_DIM).tolist()
    real_blob = struct.pack(f"<{EMBED_DIM}f", *real_vec)

    conn3 = open_store_raw(db_path)
    try:
        conn3.execute(
            "UPDATE records SET embedding = ?, embedding_pending = 0 WHERE id = ?",
            (real_blob, pending_id),
        )
        conn3.commit()
    finally:
        conn3.close()

    store3 = MemoryStore(hermetic_store)
    try:
        from iai_mcp.retrieve import build_runtime_graph as _brg
        graph3, _, _ = _brg(store3)
        graph_nodes3 = set(graph3.nodes())
        from uuid import UUID as _UUID2
        assert _UUID2(pending_id) in graph_nodes3, (
            "after re-embed (embedding_pending=0), row must appear in "
            "the runtime graph as a warm semantic candidate"
        )
    finally:
        store3.close()

def test_embed_cue_in_control_msg_types() -> None:
    from iai_mcp.socket_server import SocketServer

    assert "embed_cue" in SocketServer.CONTROL_MSG_TYPES, (
        "embed_cue must be in SocketServer.CONTROL_MSG_TYPES for the socket "
        "layer to forward it to _dispatch_socket_request (MED routing fix)"
    )

def test_embed_cue_dispatch_warm_stub(hermetic_store: Path) -> None:
    import asyncio
    from unittest.mock import patch

    from iai_mcp.store import MemoryStore
    from iai_mcp.concurrency import _dispatch_socket_request

    store = MemoryStore(hermetic_store)
    try:
        state: dict = {}

        import numpy as np
        fake_vec = np.random.RandomState(42).randn(384).tolist()

        class _FakeEmbedder:
            DIM = 384
            def embed(self, text: str) -> list:
                return list(fake_vec)

        with patch("iai_mcp.embed.Embedder", return_value=_FakeEmbedder()):
            result = asyncio.run(
                _dispatch_socket_request(
                    {"type": "embed_cue", "cue": "test cue for embed"},
                    store,
                    state,
                )
            )

        assert result.get("ok") is True, f"embed_cue dispatch must return ok=True when warm; got {result}"
        embedding = result.get("embedding")
        assert isinstance(embedding, list), "embedding must be a list"
        assert len(embedding) == 384, f"embedding must be 384-d, got {len(embedding)}"

        class _FailEmbedder:
            DIM = 384
            def embed(self, text: str) -> list:
                raise RuntimeError("embedder not ready")

        with patch("iai_mcp.embed.Embedder", return_value=_FailEmbedder()):
            result_fail = asyncio.run(
                _dispatch_socket_request(
                    {"type": "embed_cue", "cue": "test"},
                    store,
                    state,
                )
            )

        assert result_fail.get("ok") is False, "embed_cue must return ok=False when embedder fails"
        assert result_fail.get("reason") == "daemon_not_ready", (
            f"reason must be 'daemon_not_ready', got {result_fail.get('reason')!r}"
        )
    finally:
        store.close()

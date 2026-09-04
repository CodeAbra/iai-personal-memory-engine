"""Opt-in latency evidence for the salience_level rank-fusion multiplier."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp import pipeline
from iai_mcp import retrieve
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _random_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _mk_rec(text: str, embedding: list[float]) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=embedding,
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
        tags=[],
        language="en",
    )


@pytest.mark.perf
def test_salience_level_recall_latency_report(tmp_path):
    """Opt-in latency evidence: the salience_level multiplier is one
    getattr/dict-field passthrough (SALIENCE_LEVEL_RANK.get(rec.salience_level))
    plus one multiply per candidate -- no new store query, decrypt, or I/O.
    Reported for the record, not gated on a wall-clock ceiling."""
    import time as _time

    from iai_mcp.embed import Embedder

    store = MemoryStore(path=tmp_path / "salience-latency-store")
    for i in range(50):
        store.insert(_mk_rec(f"latency filler record {i}", _random_vec(8000 + i)))

    embedder = Embedder()
    g, a, rc = retrieve.build_runtime_graph(store)
    import iai_mcp.runtime_graph_cache as _rgc
    _rgc.save(store, a, rc)

    # Warm call — model + graph caches.
    pipeline.recall_for_response(
        store=store, graph=g, assignment=a, rich_club=rc,
        embedder=embedder, cue="latency filler record 0", session_id="s1",
        budget_tokens=1500, mode="concept",
    )

    t0 = _time.perf_counter()
    pipeline.recall_for_response(
        store=store, graph=g, assignment=a, rich_club=rc,
        embedder=embedder, cue="latency filler record 1", session_id="s1",
        budget_tokens=1500, mode="concept",
    )
    elapsed_ms = (_time.perf_counter() - t0) * 1000.0

    print(f"salience_level passthrough recall latency (warm): {elapsed_ms:.2f}ms")

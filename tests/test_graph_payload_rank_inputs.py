"""The PRODUCTION graph builder must deliver rank inputs to the scorer.

A hand-written test payload can carry any field; what protects live
recall is the real builder. Every graph lane — cold scan, cached replay,
incremental delta, live sync hook — must put aaak_index, created_at and
stability into the node payload, or the aaak and age terms score blanks
for every record while the formula claims otherwise.

Also pins the score-state transplant on the authority merge: a hit whose
score was already stale-halved must carry that state onto the authority
head, or a later stale pass halves the served number a second time while
the reason still claims the first value.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryHit, MemoryRecord

RANK_FIELDS = ("aaak_index", "created_at", "stability")


@pytest.fixture(autouse=True)
def _passphrase(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "iai-mcp-test-passphrase")
    yield


def _rec(vec: list[float], aaak: str, days_old: int, stability: float) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(), tier="episodic",
        literal_surface="rank input probe",
        aaak_index=aaak, embedding=vec,
        community_id=None, centrality=0.0, detail_level=2,
        pinned=False, stability=stability, difficulty=0.0, last_reviewed=None,
        never_decay=False, never_merge=False, provenance=[],
        created_at=now - timedelta(days=days_old), updated_at=now,
        tags=[], language="en",
    )


def test_production_builder_payloads_carry_rank_inputs(tmp_path):
    from iai_mcp.retrieve import _build_runtime_graph_impl

    store = MemoryStore(path=tmp_path / "store")
    rng = np.random.default_rng(11)
    probe = _rec(
        (lambda v: (v / np.linalg.norm(v)).tolist())(rng.normal(size=EMBED_DIM)),
        aaak="W:E/R:abc/E:liveness/T:capture",
        days_old=400, stability=0.9,
    )
    store.insert(probe)
    for _ in range(3):
        v = rng.normal(size=EMBED_DIM)
        store.insert(_rec((v / np.linalg.norm(v)).tolist(), "", 0, 0.5))

    graph, _assignment, _rich = _build_runtime_graph_impl(store)

    payload = graph.get_payload(probe.id)
    assert payload, "the production builder must include the probe record"
    for field in RANK_FIELDS:
        assert field in payload, (
            f"the production graph payload is missing {field!r} — the "
            "scorer will rank this record with a blank"
        )
    assert payload["aaak_index"] == "W:E/R:abc/E:liveness/T:capture"
    assert payload["created_at"].startswith(
        probe.created_at.isoformat()[:10]
    ), f"created_at drifted: {payload['created_at']!r}"
    assert abs(float(payload["stability"]) - 0.9) < 1e-9


def test_authority_merge_transplants_stale_state():
    from iai_mcp.pipeline import merge_authority_hits

    rid = uuid4()
    pipeline_hit = MemoryHit(
        record_id=rid, score=0.5,
        reason="cos 1.000*1 + aaak 0.00*0.3 + deg_norm 0.000*0.1 "
               "- age 0.00*0.05 · stale",
        literal_surface="x", adjacent_suggestions=[],
    )
    pipeline_hit._stale_downweighted = True
    authority_hit = MemoryHit(
        record_id=rid, score=0.3, reason="exact-cosine",
        literal_surface="x", adjacent_suggestions=[],
    )

    merged, _used = merge_authority_hits([pipeline_hit], [authority_hit], 5000)
    head = next(h for h in merged if h.record_id == rid)
    assert head.score == 0.5
    assert getattr(head, "_stale_downweighted", False), (
        "the transplanted score was already stale-halved; without the flag "
        "a later stale pass halves it again while the reason still claims "
        "the printed value"
    )


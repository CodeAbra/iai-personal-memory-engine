"""Recency-union markers must stay within the instant-recall token budget.

When regular hits already fill the budget and there are pending-recency
markers to surface, the recall must not emit more than budget_tokens. The
freshness markers must still be represented (they are the recency signal),
even if a lower-priority regular hit has to be trimmed to make room.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM


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
    monkeypatch.setenv(
        "IAI_MCP_CRYPTO_PASSPHRASE",
        "iai-mcp-test-passphrase-not-secret",
    )


def _token_cost(surface: str) -> int:
    # Mirror the pipeline's per-hit token accounting exactly.
    return len(surface) // 4


def test_recency_union_stays_within_budget_and_marker_survives(tmp_path, monkeypatch):
    _monkeypatch_env(monkeypatch, tmp_path)

    store_path = tmp_path / "budget-union-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))

    # Regular records with long surfaces so a handful already fill the budget.
    # Each surface ~= 400 chars -> ~100 tokens; with budget_tokens=300 only a
    # few regular hits fit, forcing the recency marker to contend for budget.
    long_body = "User long filler record content that is intentionally verbose " * 7
    for i in range(12):
        rec = _make(
            text=f"{long_body} index {i}",
            vec=_random_vec(5000 + i),
        )
        store.insert(rec)

    # A pending-recency marker (excluded from the ANN index, surfaced via the
    # bounded recency union).
    pending_text = "User pending recency marker just written, must survive budget"
    pending_id = UUID(int=777_001)
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

    import iai_mcp.retrieve as _retrieve
    import iai_mcp.runtime_graph_cache as _rgc

    _graph, _assignment, _rc = _retrieve.build_runtime_graph(store)
    _rgc.save(store, _assignment, _rc)

    from iai_mcp.embed import Embedder
    from iai_mcp.pipeline import recall_for_response
    import iai_mcp.pipeline as _pipeline_mod

    _pipeline_mod._last_recall_latency_ms = 0.0
    embedder = Embedder()

    budget_tokens = 300

    response = recall_for_response(
        store=store,
        graph=_graph,
        assignment=_assignment,
        rich_club=_rc,
        embedder=embedder,
        cue="User long filler record content verbose pending recency marker",
        session_id="test-session",
        budget_tokens=budget_tokens,
        mode="concept",
    )

    # The recency marker must still be represented (freshness signal kept).
    hit_ids = {h.record_id for h in response.hits}
    assert UUID(int=777_001) in hit_ids, (
        f"recency marker dropped from budget-constrained recall; hit_ids={hit_ids}"
    )

    # No overflow: the realized emitted token cost must not exceed the budget.
    realized_tokens = sum(_token_cost(h.literal_surface) for h in response.hits)
    assert realized_tokens <= budget_tokens, (
        f"recall emitted {realized_tokens} tokens > budget {budget_tokens}; "
        f"recency union overflowed the budget"
    )

    # The reported budget_used must also honour the cap.
    assert response.budget_used <= budget_tokens, (
        f"reported budget_used={response.budget_used} exceeds budget {budget_tokens}"
    )

    # The marker surface must be byte-exact verbatim (no paraphrase/trim of
    # whatever IS emitted).
    marker_hit = next(h for h in response.hits if h.record_id == UUID(int=777_001))
    assert marker_hit.literal_surface == pending_text, (
        "recency marker surface was mutated — verbatim invariant violated"
    )


def test_single_fat_marker_alone_within_budget(tmp_path, monkeypatch):
    # When there are no regular hits to reclaim and a marker alone fits the
    # budget, it is admitted; total stays within budget.
    _monkeypatch_env(monkeypatch, tmp_path)

    store_path = tmp_path / "fat-marker-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))

    # Only a couple of small regular records.
    for i in range(3):
        rec = _make(text=f"User small {i}", vec=_random_vec(6000 + i))
        store.insert(rec)

    pending_text = "User pending recency single marker content within budget"
    pending_id = UUID(int=666_001)
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

    import iai_mcp.retrieve as _retrieve
    import iai_mcp.runtime_graph_cache as _rgc

    _graph, _assignment, _rc = _retrieve.build_runtime_graph(store)
    _rgc.save(store, _assignment, _rc)

    from iai_mcp.embed import Embedder
    from iai_mcp.pipeline import recall_for_response
    import iai_mcp.pipeline as _pipeline_mod

    _pipeline_mod._last_recall_latency_ms = 0.0
    embedder = Embedder()

    budget_tokens = 2000

    response = recall_for_response(
        store=store,
        graph=_graph,
        assignment=_assignment,
        rich_club=_rc,
        embedder=embedder,
        cue="User pending recency single marker content",
        session_id="test-session",
        budget_tokens=budget_tokens,
        mode="concept",
    )

    hit_ids = {h.record_id for h in response.hits}
    assert UUID(int=666_001) in hit_ids, (
        f"recency marker not surfaced; hit_ids={hit_ids}"
    )

    realized_tokens = sum(_token_cost(h.literal_surface) for h in response.hits)
    assert realized_tokens <= budget_tokens, (
        f"emitted {realized_tokens} tokens > budget {budget_tokens}"
    )

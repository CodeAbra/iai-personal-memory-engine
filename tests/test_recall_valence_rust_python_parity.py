"""Rust-vs-Python valence-multiplier parity.

Proves the Rust primary scoring path (`use_rust_scorer=True`) and the Python
fallback path (`use_rust_scorer=False`) return the SAME fused score for a
record whose valence is above zero. `0.5` is used for the parity assertion
because it is exactly representable in both the f32 resident column (Rust)
and the f64 reference (Python fallback) -- exact equality then isolates the
multiplier-restoration proof from the pre-existing f32-resident/f64-reference
dtype boundary that affects every scalar column (stability, centrality,
valence), rather than conflating the two.

The fixture is deliberately minimal (two records, no edges, no community, no
structural/spread terms) so every non-valence scoring term is either exactly
zero or exactly representable in both f32 and f64 -- a corpus with realistic
degree/community/stability variance carries its OWN pre-existing float32-
resident/float64-reference noise unrelated to valence (see
`test_rust_python_valence_parity_zero_baseline`, which pins this at
valence=0.0 with the fix inert).
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_recall_scoring_differential import _freeze_age_penalty, _wrap_call_counter  # noqa: E402
from test_recall_stage_profile import _monkeypatch_env  # noqa: E402

from iai_mcp import pipeline as _pm  # noqa: E402
from iai_mcp import runtime_graph_cache  # noqa: E402
from iai_mcp.embed import Embedder  # noqa: E402
from iai_mcp.pipeline import recall_for_response  # noqa: E402
from iai_mcp.retrieve import build_runtime_graph  # noqa: E402
from iai_mcp.store import MemoryStore  # noqa: E402
from iai_mcp.types import MemoryRecord  # noqa: E402

_NEUTRAL_CUE = "an unrelated grocery list cue phrase"
# Ten years back: (now - created_at) / AGE_HALF_LIFE_DAYS is far past 1.0 on
# both sides regardless of the few milliseconds of wall-clock drift between
# the two recall calls below -- age_penalty's min(1.0, ...) clamp then makes
# the age term an identical constant on both paths, not a source of noise.
_ANCIENT_DAYS = 3650


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(_keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p))
    monkeypatch.setattr(_keyring, "delete_password", lambda s, u: fake.pop((s, u), None))
    yield fake


def _mk_rec(literal_surface: str, embedding: list[float], valence: float) -> MemoryRecord:
    ancient = datetime.now(timezone.utc) - timedelta(days=_ANCIENT_DAYS)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=literal_surface,
        aaak_index="",
        embedding=embedding,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.5,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=ancient,
        updated_at=ancient,
        tags=[],
        language="en",
        salience_level="unflagged",
        valence=valence,
    )


def _build_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, target_valence: float):
    _monkeypatch_env(monkeypatch, tmp_path)
    store_root = tmp_path / "valence-rust-python-parity"
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))

    embedder = Embedder()
    target_vec = list(embedder.embed("a distinct, isolated recall-parity probe surface"))
    filler_vec = list(embedder.embed("a wholly unrelated filler record about kitchen tiles"))
    target = _mk_rec("a distinct, isolated recall-parity probe surface", target_vec, target_valence)
    filler = _mk_rec("a wholly unrelated filler record about kitchen tiles", filler_vec, 0.0)

    store = MemoryStore(path=store_root)
    store.insert(target)
    store.insert(filler)

    runtime_graph_cache.invalidate(store)
    graph, assignment, rich_club = build_runtime_graph(store)
    return store, target, embedder, graph, assignment, rich_club


def _score_for(hits, record_id: UUID) -> "float | None":
    for h in hits:
        if h.record_id == record_id:
            return h.score
    return None


def _recall(store, graph, assignment, rich_club, embedder, cue_embedding, *, use_rust_scorer: bool, session_id: str):
    _pm._last_recall_latency_ms = 0.0
    response = recall_for_response(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_NEUTRAL_CUE, session_id=session_id,
        budget_tokens=1500, mode="concept",
        cue_embedding=cue_embedding, use_rust_scorer=use_rust_scorer,
    )
    return response.hits


def test_rust_python_valence_parity_f32_exact(tmp_path, monkeypatch):
    _freeze_age_penalty(monkeypatch)
    store, target, embedder, graph, assignment, rich_club = _build_store(
        tmp_path, monkeypatch, target_valence=0.5,
    )
    cue_embedding = list(target.embedding)

    rust_calls = _wrap_call_counter(monkeypatch, _pm, "_t11_t12_flags")
    rust_hits = _recall(
        store, graph, assignment, rich_club, embedder, cue_embedding,
        use_rust_scorer=True, session_id="valence-parity-rust",
    )
    assert len(rust_calls) == 1, (
        "Rust scorer did not fire exactly once -- _t11_t12_flags entry count "
        f"was {len(rust_calls)}, which would make this a Python-vs-Python "
        "comparison (also fails loudly against a stale pre-fix .so, which "
        "never applies the valence multiplier)"
    )

    python_hits = _recall(
        store, graph, assignment, rich_club, embedder, cue_embedding,
        use_rust_scorer=False, session_id="valence-parity-python",
    )

    rust_score = _score_for(rust_hits, target.id)
    python_score = _score_for(python_hits, target.id)
    assert rust_score is not None and python_score is not None, (
        f"target record {target.id} missing from a scorer's hits -- "
        f"rust={[h.record_id for h in rust_hits]} python={[h.record_id for h in python_hits]}"
    )
    assert rust_score == python_score, (
        f"Rust ({rust_score}) and Python ({python_score}) fused scores diverge "
        "for an f32-exact valence=0.5 record on a minimal, zero-noise fixture"
    )


def test_rust_python_valence_parity_zero_baseline(tmp_path, monkeypatch):
    """valence=0.0 baseline: the two paths already agree without the fix
    (the multiplier never fires) -- guards against a test that would pass
    regardless of whether the fix is present."""
    _freeze_age_penalty(monkeypatch)
    store, target, embedder, graph, assignment, rich_club = _build_store(
        tmp_path, monkeypatch, target_valence=0.0,
    )
    cue_embedding = list(target.embedding)

    rust_hits = _recall(
        store, graph, assignment, rich_club, embedder, cue_embedding,
        use_rust_scorer=True, session_id="valence-baseline-rust",
    )
    python_hits = _recall(
        store, graph, assignment, rich_club, embedder, cue_embedding,
        use_rust_scorer=False, session_id="valence-baseline-python",
    )

    rust_score = _score_for(rust_hits, target.id)
    python_score = _score_for(python_hits, target.id)
    assert rust_score is not None and python_score is not None, (
        f"target record {target.id} missing from a scorer's hits -- "
        f"rust={[h.record_id for h in rust_hits]} python={[h.record_id for h in python_hits]}"
    )
    assert rust_score == python_score, (
        f"valence=0.0 baseline diverged: rust={rust_score} python={python_score}"
    )

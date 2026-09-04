"""Non-vacuity + safety proofs for the awake priming seam with the flag ON:
a recognized chunk's visible member surfaces (A), a primed candidate
genuinely bottom-ranked on the full fused score survives the Rust scorer's
`winners.truncate(k+k_margin)` at production pool scale -- not merely at a
pool too small for the truncation to ever fire (B) -- a genuine top hit is
never eclipsed by a primed one (C), and the mechanism never surfaces
procedural chunk text or blows the token budget (D).

All target/filler records are genuinely inserted with distinct UUIDs; none
of the "A"/"B"/"alice"/"bob" placeholder ids are used, since those cannot
round-trip through `UUID(...)` and would make the ON path a silently
swallowed no-op.

Filler/target embeddings are constructed numerically against the cue's own
real embedding (`alpha * cue_unit + beta * orthogonal`) so each candidate's
cosine to the cue is a controlled, known quantity -- the only way to
guarantee a candidate is genuinely below (or above) the Rust truncation
boundary rather than merely low-cosine by chance.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from tests.test_recall_scoring_differential import _freeze_age_penalty
from tests.test_recall_stage_profile import _monkeypatch_env
from tests._synthetic_cue_corpus import insert_corpus

import iai_mcp.pipeline as _pm
from iai_mcp import prime_cache
from iai_mcp.embed import Embedder
from iai_mcp.pipeline import _POST_RANK_MAX_HITS, _recall_core, recall_for_response
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord

_SEED = 0
_CUE_TEXT = "how does the archive cleanup job handle stale sessions this month"
_N_FILLERS_LARGE = 150


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(_keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p))
    monkeypatch.setattr(_keyring, "delete_password", lambda s, u: fake.pop((s, u), None))
    yield fake


# ---------------------------------------------------------------------------
# Numeric embedding construction -- alpha controls cosine-to-cue exactly.
# ---------------------------------------------------------------------------

def _unit(v: "np.ndarray") -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    return arr / n if n > 0 else arr


def _orthogonal(rng: np.random.Generator, basis_unit: np.ndarray, dim: int) -> np.ndarray:
    v = rng.normal(size=dim).astype(np.float32)
    v = v - basis_unit * float(np.dot(v, basis_unit))
    return _unit(v)


def _blend(cue_unit: np.ndarray, ortho: np.ndarray, alpha: float) -> list[float]:
    beta = float(np.sqrt(max(0.0, 1.0 - alpha * alpha)))
    return _unit(alpha * cue_unit + beta * ortho).tolist()


def _mk_record(
    literal_surface: str, embedding: list[float], created_at: datetime, *, tier: str = "episodic",
) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(), tier=tier, literal_surface=literal_surface, aaak_index="",
        embedding=embedding, community_id=None, centrality=0.0, detail_level=2,
        pinned=False, stability=0.5, difficulty=0.0, last_reviewed=None,
        never_decay=False, never_merge=False, provenance=[],
        created_at=created_at, updated_at=created_at, tags=[], language="en",
    )


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, store_name: str):
    _monkeypatch_env(monkeypatch, tmp_path)
    store_root = tmp_path / store_name
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    embedder = Embedder()
    cue_unit = _unit(np.asarray(embedder.embed(_CUE_TEXT), dtype=np.float32))
    return embedder, cue_unit, store_root


def _make_fillers(
    rng: np.random.Generator, cue_unit: np.ndarray, dim: int, n: int,
    alpha_lo: float, alpha_hi: float, ts: datetime,
) -> "list[MemoryRecord]":
    out = []
    for i in range(n):
        alpha = float(rng.uniform(alpha_lo, alpha_hi))
        ortho = _orthogonal(rng, cue_unit, dim)
        emb = _blend(cue_unit, ortho, alpha)
        text = f"unrelated filler passage number {i} static log entry lorem qux zephyr"
        out.append(_mk_record(text, emb, ts))
    return out


def _finalize(store_root: Path, records: "list[MemoryRecord]"):
    store = MemoryStore(path=store_root)
    insert_corpus(store, records)
    from iai_mcp.retrieve import build_runtime_graph
    graph, assignment, rich_club = build_runtime_graph(store)
    return store, graph, assignment, rich_club


def _save_prime_blob(store: MemoryStore, seed_id: UUID, chunk_id: str, dst_id: UUID) -> None:
    blob = {
        "seed_to_chunks": {str(seed_id): [chunk_id]},
        "chunk_members": {chunk_id: [str(seed_id), str(dst_id)]},
    }
    assert prime_cache.save(store, blob) is True
    # load() memoizes per-process -- a save without invalidate would leave a
    # prior (possibly empty) load() result stale for the rest of this test.
    prime_cache.invalidate(store)


def _core(
    store, graph, assignment, rich_club, embedder, cue_unit, *,
    use_rust_scorer: "bool | None", session_id: str,
):
    graph._records_view_cache = None
    return _recall_core(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_CUE_TEXT, session_id=session_id,
        profile_state={}, mode="concept", cue_intent=None,
        contradicts_outgoing={}, use_rust_scorer=use_rust_scorer,
        cue_embedding=cue_unit.tolist(),
    )


def _ids(hits) -> "set[str]":
    return {str(h.record_id) for h in hits}


# ---------------------------------------------------------------------------
# A. Widen-on-recognition (positive control) + trace_mark observability.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_rust_scorer", [True, False], ids=["rust", "python_reference"])
def test_widen_on_recognition_surfaces_visible_member(
    tmp_path, monkeypatch, use_rust_scorer,
):
    _freeze_age_penalty(monkeypatch)
    embedder, cue_unit, store_root = _setup(tmp_path, monkeypatch, f"proc-prime-seam-a-{use_rust_scorer}")
    rng = np.random.default_rng(_SEED)
    dim = cue_unit.shape[0]
    recent = datetime.now(timezone.utc) - timedelta(hours=1)

    src_rec = _mk_record(_CUE_TEXT, cue_unit.tolist(), recent)
    # Descending, close-packed alphas so a 5% nudge can cross exactly one
    # rank boundary -- a large, uniform boost is required here: an isolated
    # near-zero-cosine target needs far more than the default nudge to cross
    # any rank boundary, which would prove nothing about the mechanism.
    filler_alphas = [0.35, 0.33, 0.31, 0.29, 0.27, 0.25]
    fillers = []
    for i, alpha in enumerate(filler_alphas):
        ortho = _orthogonal(rng, cue_unit, dim)
        fillers.append(_mk_record(f"filler passage {i} unrelated static text", _blend(cue_unit, ortho, alpha), recent))
    dst_ortho = _orthogonal(rng, cue_unit, dim)
    dst = _mk_record(
        "isolated target record distinct unrelated topic marker delta",
        _blend(cue_unit, dst_ortho, 0.24), recent,
    )

    store, graph, assignment, rich_club = _finalize(store_root, [src_rec, *fillers, dst])
    _save_prime_blob(store, src_rec.id, "chunk-widen-a", dst.id)

    _K = 7  # src_rec + the 6 fillers; dst (rank 8 unprimed) must fall outside it

    marks_off: list[str] = []
    monkeypatch.delenv("IAI_MCP_PROC_PRIME", raising=False)
    core_off = _core(
        store, graph, assignment, rich_club, embedder, cue_unit,
        use_rust_scorer=use_rust_scorer, session_id="proc-prime-seam-a-off",
    )
    _pm_trace_off = []
    # trace_mark isn't threaded through _recall_core's return value directly
    # in this call shape -- re-run once more with the collector wired, cheap
    # at this pool size, to get both the rank proof and the trace proof.
    graph._records_view_cache = None
    core_off_traced = _recall_core(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_CUE_TEXT, session_id="proc-prime-seam-a-off-trace",
        profile_state={}, mode="concept", cue_intent=None, contradicts_outgoing={},
        use_rust_scorer=use_rust_scorer, cue_embedding=cue_unit.tolist(),
        trace_mark=marks_off.append,
    )
    assert str(dst.id) not in _ids(core_off.scored_hits[:_K]), (
        "fixture does not isolate the widening: dst already ranks inside "
        "the unprimed top-K -- narrow the filler/dst alpha gap"
    )
    assert "proc_prime" not in marks_off

    marks_on: list[str] = []
    monkeypatch.setenv("IAI_MCP_PROC_PRIME", "1")
    core_on = _core(
        store, graph, assignment, rich_club, embedder, cue_unit,
        use_rust_scorer=use_rust_scorer, session_id="proc-prime-seam-a-on",
    )
    graph._records_view_cache = None
    core_on_traced = _recall_core(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_CUE_TEXT, session_id="proc-prime-seam-a-on-trace",
        profile_state={}, mode="concept", cue_intent=None, contradicts_outgoing={},
        use_rust_scorer=use_rust_scorer, cue_embedding=cue_unit.tolist(),
        trace_mark=marks_on.append,
    )
    assert str(dst.id) in _ids(core_on.scored_hits[:_K]), (
        "priming ON did not promote the recognized chunk's visible member "
        "across the rank boundary the nudge is supposed to cross"
    )
    assert "proc_prime" in marks_on
    del core_off_traced, core_on_traced


# ---------------------------------------------------------------------------
# B. Large-pool Rust-axis truncation-survival (the non-vacuity crux).
# ---------------------------------------------------------------------------

def test_large_pool_rust_truncation_survival(tmp_path, monkeypatch):
    _freeze_age_penalty(monkeypatch)
    embedder, cue_unit, store_root = _setup(tmp_path, monkeypatch, "proc-prime-seam-b")
    rng = np.random.default_rng(_SEED)
    dim = cue_unit.shape[0]
    recent = datetime.now(timezone.utc) - timedelta(hours=1)
    old = datetime.now(timezone.utc) - timedelta(days=400)  # >> AGE_HALF_LIFE_DAYS=30 -- max age penalty

    src_rec = _mk_record(_CUE_TEXT, cue_unit.tolist(), recent)
    fillers = _make_fillers(rng, cue_unit, dim, _N_FILLERS_LARGE, 0.55, 0.85, recent)
    dst_ortho = _orthogonal(rng, cue_unit, dim)
    dst_emb = _blend(cue_unit, dst_ortho, 0.0)
    dst = _mk_record(
        "isolated bottom-ranked target unrelated topic marker epsilon", dst_emb, old,
    )

    all_records = [src_rec, *fillers, dst]
    store, graph, assignment, rich_club = _finalize(store_root, all_records)
    _save_prime_blob(store, src_rec.id, "chunk-truncation-b", dst.id)

    # Fixture self-check: the cosine gap alone must dwarf every other
    # pre-truncation term (max possible aaak+degree+age combined << W_COSINE
    # * gap here) -- this converts "deterministically higher" from a hope
    # into a checked precondition.
    dst_cos = float(np.dot(np.asarray(dst_emb, dtype=np.float32), cue_unit))
    filler_coss = [float(np.dot(np.asarray(f.embedding, dtype=np.float32), cue_unit)) for f in fillers]
    min_filler_cos = min(filler_coss)
    assert min_filler_cos - dst_cos > 0.4, (
        f"cosine gap too small to dominate the fused score deterministically: "
        f"min_filler_cos={min_filler_cos:.3f} dst_cos={dst_cos:.3f}"
    )

    assert len(all_records) >= 150

    # P1: dst is genuinely in the pool.
    assert dst.id in set(graph.iter_nodes()), "P1 failed: dst not in graph pool"

    monkeypatch.delenv("IAI_MCP_PROC_PRIME", raising=False)

    # P2: priming OFF, RUST axis -> dst ABSENT (un-widened fused rank is
    # beyond the top-82 cut -- the exact quantity Rust truncates by).
    core_off_rust = _core(
        store, graph, assignment, rich_club, embedder, cue_unit,
        use_rust_scorer=True, session_id="proc-prime-seam-b-off-rust",
    )
    assert str(dst.id) not in _ids(core_off_rust.scored_hits), (
        "P2 failed: dst is present on the Rust axis even without priming -- "
        "either the pool is too small or a pre-truncation term is lifting "
        "dst inside the top-82 cut (re-check the cosine/aaak/degree/age/"
        "community/fts/trigram/lex controls); do not shrink the pool or "
        "raise dst's score to force this green"
    )

    # P3: priming OFF, PYTHON-REFERENCE axis -> dst PRESENT (never truncates).
    core_off_py = _core(
        store, graph, assignment, rich_club, embedder, cue_unit,
        use_rust_scorer=False, session_id="proc-prime-seam-b-off-py",
    )
    assert str(dst.id) in _ids(core_off_py.scored_hits), (
        "P3 failed: dst absent even on the never-truncating Python-reference "
        "axis -- dst is not being scored at all, not merely truncated; "
        "P2's absence would not isolate truncation"
    )

    # CLAIM: priming ON, RUST axis -> dst PRESENT (k_margin widened past the
    # pool size, so winners.truncate is a no-op for this call).
    monkeypatch.setenv("IAI_MCP_PROC_PRIME", "1")
    core_on_rust = _core(
        store, graph, assignment, rich_club, embedder, cue_unit,
        use_rust_scorer=True, session_id="proc-prime-seam-b-on-rust",
    )
    assert str(dst.id) in _ids(core_on_rust.scored_hits), (
        "priming ON did not rescue dst from the Rust truncation -- the "
        "k_margin widening is not surviving to the scorer call"
    )

    # Asymmetry: the Python-reference axis was never truncating in the first
    # place, so its ON presence is unaffected by the k_margin lever -- this
    # is what makes the truncation-survival mutant (bare RUST_SCORER_K_MARGIN
    # restored) redden the Rust axis ONLY, never both.
    core_on_py = _core(
        store, graph, assignment, rich_club, embedder, cue_unit,
        use_rust_scorer=False, session_id="proc-prime-seam-b-on-py",
    )
    assert str(dst.id) in _ids(core_on_py.scored_hits), (
        "dst absent on the Python-reference axis with priming ON -- this "
        "axis never truncates, so this failure is unrelated to k_margin"
    )


# ---------------------------------------------------------------------------
# C. Clamp-safety (bounded control): a genuine top hit is never eclipsed.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_rust_scorer", [True, False], ids=["rust", "python_reference"])
@pytest.mark.parametrize("pool_size", ["small", "large"])
def test_clamp_safety_bounded_control(tmp_path, monkeypatch, use_rust_scorer, pool_size):
    _freeze_age_penalty(monkeypatch)
    store_name = f"proc-prime-seam-c-{pool_size}-{use_rust_scorer}"
    embedder, cue_unit, store_root = _setup(tmp_path, monkeypatch, store_name)
    rng = np.random.default_rng(_SEED)
    dim = cue_unit.shape[0]
    recent = datetime.now(timezone.utc) - timedelta(hours=1)

    strong_genuine = _mk_record(_CUE_TEXT, cue_unit.tolist(), recent)

    weak_ortho = _orthogonal(rng, cue_unit, dim)
    weak_genuine = _mk_record(
        "weak genuine record modest overlap topic marker gamma",
        _blend(cue_unit, weak_ortho, 0.15), recent,
    )
    primed_ortho = _orthogonal(rng, cue_unit, dim)
    primed_rec = _mk_record(
        "primed only candidate reached solely via the chunk marker theta",
        _blend(cue_unit, primed_ortho, 0.1), recent,
    )

    if pool_size == "large":
        fillers = _make_fillers(rng, cue_unit, dim, _N_FILLERS_LARGE, 0.55, 0.85, recent)
    else:
        fillers = _make_fillers(rng, cue_unit, dim, 5, 0.2, 0.4, recent)

    all_records = [strong_genuine, weak_genuine, primed_rec, *fillers]
    store, graph, assignment, rich_club = _finalize(store_root, all_records)
    _save_prime_blob(store, strong_genuine.id, "chunk-clamp-c", primed_rec.id)

    # Inflated so the primed candidate's PRE-CLAMP score would exceed
    # strong_genuine's -- a bare 1.05 default would never cross it, making
    # the clamp mutant vacuous (it would already pass with the clamp removed).
    monkeypatch.setenv("IAI_MCP_PROC_PRIME_BOOST", "50")
    monkeypatch.setenv("IAI_MCP_PROC_PRIME", "1")

    core = _core(
        store, graph, assignment, rich_club, embedder, cue_unit,
        use_rust_scorer=use_rust_scorer, session_id=f"proc-prime-seam-c-{pool_size}",
    )
    by_id = {str(h.record_id): h for h in core.scored_hits}
    assert str(strong_genuine.id) in by_id, "strong_genuine missing from scored output"
    assert str(primed_rec.id) in by_id, (
        "primed_rec missing from scored output -- cannot assert clamp order "
        "against a candidate that was never scored"
    )
    strong_score = by_id[str(strong_genuine.id)].score
    primed_score = by_id[str(primed_rec.id)].score
    assert strong_score > primed_score, (
        f"clamp failed: primed ({primed_score:.6f}) eclipsed the top genuine "
        f"hit strong_genuine ({strong_score:.6f}) -- primed-vs-weak-genuine "
        "order is deliberately unconstrained, but the TOP genuine hit must "
        "never be eclipsed"
    )


# ---------------------------------------------------------------------------
# D. Non-visibility + budget.
# ---------------------------------------------------------------------------

def test_non_visibility_and_budget(tmp_path, monkeypatch):
    _freeze_age_penalty(monkeypatch)
    embedder, cue_unit, store_root = _setup(tmp_path, monkeypatch, "proc-prime-seam-d")
    rng = np.random.default_rng(_SEED)
    dim = cue_unit.shape[0]
    recent = datetime.now(timezone.utc) - timedelta(hours=1)

    src_rec = _mk_record(_CUE_TEXT, cue_unit.tolist(), recent)
    fillers = _make_fillers(rng, cue_unit, dim, 10, 0.2, 0.5, recent)
    dst_ortho = _orthogonal(rng, cue_unit, dim)
    dst = _mk_record(
        "isolated visible member unrelated topic marker sigma", _blend(cue_unit, dst_ortho, 0.05), recent,
    )

    _CHUNK_MARKER = "unmistakably-procedural-chunk-text-never-served-marker-kappa"
    chunk_rec = _mk_record(_CHUNK_MARKER, cue_unit.tolist(), recent, tier="procedural")

    all_records = [src_rec, *fillers, dst, chunk_rec]
    store, graph, assignment, rich_club = _finalize(store_root, all_records)
    _save_prime_blob(store, src_rec.id, "chunk-visibility-d", dst.id)

    monkeypatch.setenv("IAI_MCP_PROC_PRIME", "1")
    budget_tokens = 1500
    graph._records_view_cache = None
    response = recall_for_response(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_CUE_TEXT, session_id="proc-prime-seam-d",
        budget_tokens=budget_tokens, mode="concept", cue_embedding=cue_unit.tolist(),
    )

    served_ids = {str(h.record_id) for h in response.hits}
    served_surfaces = [h.literal_surface for h in response.hits]

    assert str(chunk_rec.id) not in served_ids, (
        "the procedural chunk's own record surfaced in served hits -- the "
        "procedural-tier serve filter must exclude it even with priming widening the seeds"
    )
    assert not any(_CHUNK_MARKER in s for s in served_surfaces), (
        "the procedural chunk's distinctive text leaked into a served hit's "
        "literal_surface"
    )
    assert str(dst.id) in served_ids, (
        "fixture sanity: priming should surface the VISIBLE member -- if "
        "this fails the non-visibility proof above is not exercising a live "
        "widening at all"
    )
    assert response.budget_used <= budget_tokens, (
        f"served budget_used={response.budget_used} exceeds the requested "
        f"cap {budget_tokens} -- the widened k_margin's larger scored_hits "
        "leaked past the budget-pack loop"
    )
    assert len(response.hits) <= _POST_RANK_MAX_HITS

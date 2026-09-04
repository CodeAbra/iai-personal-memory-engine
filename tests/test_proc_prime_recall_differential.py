"""Both no-op paths of the awake priming seam -- flag OFF and flag ON with
an empty/absent prime cache -- must serve byte-identical (hits + budget)
recall to a run without the seam. The mechanism ships gated off; this proves
"off" and "on-but-nothing-to-prime" are both silent no-ops, not merely by
convention.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_recall_scoring_differential import _freeze_age_penalty
from tests.test_recall_stage_profile import _monkeypatch_env

from tests._synthetic_cue_corpus import build_corpus_records, build_cue_set, insert_corpus

import iai_mcp.pipeline as _pm
from iai_mcp.embed import Embedder
from iai_mcp.pipeline import recall_for_response
from iai_mcp.store import MemoryStore

_SEED = 0


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(_keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p))
    monkeypatch.setattr(_keyring, "delete_password", lambda s, u: fake.pop((s, u), None))
    yield fake


def _build_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _monkeypatch_env(monkeypatch, tmp_path)
    store_root = tmp_path / "proc-prime-differential"
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))

    embedder = Embedder()
    records = build_corpus_records(seed=_SEED, embedder=embedder)
    store = MemoryStore(path=store_root)
    insert_corpus(store, records)

    from iai_mcp.retrieve import build_runtime_graph
    graph, assignment, rich_club = build_runtime_graph(store)
    return store, graph, assignment, rich_club, embedder


def _recall(
    store, graph, assignment, rich_club, embedder, cue: str, *, budget_tokens: int = 1500,
) -> "tuple[list[str], int]":
    _pm._last_recall_latency_ms = 0.0
    response = recall_for_response(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=cue, session_id="proc-prime-differential",
        budget_tokens=budget_tokens, mode="concept",
    )
    return [str(h.record_id) for h in response.hits], response.budget_used


def test_off_path_is_deterministic(tmp_path, monkeypatch):
    _freeze_age_penalty(monkeypatch)
    monkeypatch.delenv("IAI_MCP_PROC_PRIME", raising=False)
    store, graph, assignment, rich_club, embedder = _build_store(tmp_path, monkeypatch)
    cue = build_cue_set(seed=_SEED)["specific"][0].text

    first = _recall(store, graph, assignment, rich_club, embedder, cue)
    second = _recall(store, graph, assignment, rich_club, embedder, cue)

    assert first == second, (
        "recall is non-deterministic across repeated calls with "
        "IAI_MCP_PROC_PRIME unset -- the OFF-path baseline itself is unstable"
    )


def test_on_empty_cache_byte_identical_to_off(tmp_path, monkeypatch):
    _freeze_age_penalty(monkeypatch)
    store, graph, assignment, rich_club, embedder = _build_store(tmp_path, monkeypatch)
    cue = build_cue_set(seed=_SEED)["specific"][0].text

    monkeypatch.delenv("IAI_MCP_PROC_PRIME", raising=False)
    off_result = _recall(store, graph, assignment, rich_club, embedder, cue)

    from iai_mcp import prime_cache
    assert prime_cache.load(store) == {}, (
        "precondition failed: the store's prime cache is not empty -- the "
        "ON-empty-cache no-op path is not actually being exercised"
    )

    monkeypatch.setenv("IAI_MCP_PROC_PRIME", "1")
    try:
        on_result = _recall(store, graph, assignment, rich_club, embedder, cue)
    except Exception as exc:  # noqa: BLE001 -- the assertion below documents the crash smoke
        pytest.fail(f"ON-empty-cache recall raised: {exc!r}")

    assert on_result == off_result, (
        f"ON-empty-cache recall diverged from the OFF baseline: "
        f"off={off_result} on={on_result} -- an empty primed_ids set must "
        f"leave k_margin, the nudge, and the clamp all inert"
    )


def test_unconditional_widening_reddens_the_differential(tmp_path, monkeypatch):
    """Non-vacuity: if the priming block ran regardless of the flag (or
    seeded a candidate even with an empty cache), the ON-empty served result
    would diverge from OFF. Simulated here via a monkeypatched prime_cache
    that maps every possible seed to the same priming target -- robust to
    whichever 3 ids `_pick_seeds` actually chooses -- proving the comparator
    in this file is capable of catching a real seam leak, not just recording
    a vacuous pass. The corresponding source-level mutant (dropping the
    `== "1"` gate in pipeline.py) was verified by hand to redden this same
    comparison and was reverted clean.
    """
    _freeze_age_penalty(monkeypatch)
    store, graph, assignment, rich_club, embedder = _build_store(tmp_path, monkeypatch)
    cue = build_cue_set(seed=_SEED)["specific"][0].text
    _budget = 100_000

    monkeypatch.delenv("IAI_MCP_PROC_PRIME", raising=False)
    off_result = _recall(store, graph, assignment, rich_club, embedder, cue, budget_tokens=_budget)

    node_ids = [str(rid) for rid in graph.iter_nodes()]
    assert len(node_ids) >= 5, "corpus fixture too small to construct a priming pair"
    # A mid-ranked hit, not the top or bottom of the baseline order: its
    # unnudged score sits well below max(unprimed), so the 5% boost is
    # never capped away by the clamp, and it is expected to jump ahead of
    # its immediate rank neighbours -- deterministically visible in the
    # served order, unlike an arbitrary (hash-order-dependent) record id.
    target = off_result[0][len(off_result[0]) // 2]
    seed_to_chunks = {rid: [f"fake-chunk-{rid}"] for rid in node_ids if rid != target}
    chunk_members = {f"fake-chunk-{rid}": [rid, target] for rid in node_ids if rid != target}
    fake_blob = {"seed_to_chunks": seed_to_chunks, "chunk_members": chunk_members}

    from iai_mcp import prime_cache
    monkeypatch.setattr(prime_cache, "load", lambda _store: fake_blob)
    monkeypatch.setenv("IAI_MCP_PROC_PRIME", "1")
    on_result = _recall(store, graph, assignment, rich_club, embedder, cue, budget_tokens=_budget)

    assert on_result != off_result, (
        "a non-empty prime cache under the flag produced a byte-identical "
        "result to OFF -- the differential in this file is vacuous and "
        "would not catch a real leak of the priming mechanism"
    )

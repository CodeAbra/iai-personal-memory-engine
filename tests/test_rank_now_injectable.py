"""The Rust scorer's `now` comes from `pipeline._rank_now()`.

This proves the seam is injectable: frozen, two Rust-scored recalls over an
unchanged store/graph are byte-identical; advanced, they differ; and
unfrozen, the seam returns the real wall clock so production is unaffected.
The Rust scorer's own call counter proves it fired on every recall in both
arms, so byte-identity is never trivially true via the Python reference path.
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("iai_mcp_native")

import iai_mcp.pipeline as pipeline_mod
from iai_mcp.pipeline import recall_for_response
from iai_mcp.retrieve import build_runtime_graph
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.store._rank_index import _RankIndexHandle

from tests.rss_rca.corpus import build_corpus

_N_RECORDS = 60
_SEED = 20260917
_CUE = "alice daily routine note 5"
_ADVANCE_SECONDS = 100 * 86400


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(_keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p))
    monkeypatch.setattr(_keyring, "delete_password", lambda s, u: fake.pop((s, u), None))
    yield fake


class _DeterministicEmbedder:
    """One deterministic unit vector per cue text, sha256-seeded for cross-process reproducibility."""

    def __init__(self, dim: int) -> None:
        self.DIM = dim

    def _vec_for(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], "big")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.DIM).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed(self, text: str) -> list[float]:
        return self._vec_for(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._vec_for(t) for t in texts]


def _build_fixture(tmp_path: Path):
    store = MemoryStore(path=tmp_path / "store")
    build_corpus(
        store, n=_N_RECORDS, seed=_SEED, dim=store.embed_dim,
        varied_tags=True, with_hv_payload=False,
    )
    flush_record_buffer(store)
    graph, assignment, rich_club = build_runtime_graph(store)
    embedder = _DeterministicEmbedder(dim=store.embed_dim)
    return store, graph, assignment, rich_club, embedder


def _wrap_score_counter(monkeypatch: pytest.MonkeyPatch) -> list:
    """Counts `_RankIndexHandle.score` calls at the class level; proves the Rust scorer fired."""
    original = _RankIndexHandle.score
    calls: list = []

    def _wrapped(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(_RankIndexHandle, "score", _wrapped)
    return calls


def _run_recall(store, graph, assignment, rich_club, embedder, cue: str):
    return recall_for_response(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=cue, session_id="rank-now-injectable",
        budget_tokens=1500, use_rust_scorer=True,
    )


def _serialize(response) -> bytes:
    return json.dumps(dataclasses.asdict(response), sort_keys=True, default=str).encode("utf-8")


def test_unfrozen_seam_matches_wall_clock() -> None:
    before = int(time.time())
    value = pipeline_mod._rank_now()
    after = int(time.time())
    assert before - 1 <= value <= after + 1


def test_frozen_seam_produces_byte_identical_rust_recalls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("IAI_MCP_RECALL_RUST_SCORER_OFF", raising=False)
    monkeypatch.delenv("IAI_MCP_STAGE_PROFILE", raising=False)
    store, graph, assignment, rich_club, embedder = _build_fixture(tmp_path)

    monkeypatch.setattr(pipeline_mod, "_age_penalty", lambda created_at: 0.0)
    fixed_now = int(time.time())
    monkeypatch.setattr(pipeline_mod, "_rank_now", lambda: fixed_now)
    calls = _wrap_score_counter(monkeypatch)

    n0 = len(calls)
    first = _run_recall(store, graph, assignment, rich_club, embedder, _CUE)
    n1 = len(calls)
    assert n1 - n0 >= 1, "the Rust scorer did not fire on the first recall"

    second = _run_recall(store, graph, assignment, rich_club, embedder, _CUE)
    n2 = len(calls)
    assert n2 - n1 >= 1, "the Rust scorer did not fire on the second recall"

    assert _serialize(first) == _serialize(second), (
        "two Rust-scored recalls with a frozen clock seam must be byte-identical"
    )


def test_advancing_seam_changes_rust_recall_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("IAI_MCP_RECALL_RUST_SCORER_OFF", raising=False)
    monkeypatch.delenv("IAI_MCP_STAGE_PROFILE", raising=False)
    store, graph, assignment, rich_club, embedder = _build_fixture(tmp_path)

    monkeypatch.setattr(pipeline_mod, "_age_penalty", lambda created_at: 0.0)
    base_now = int(time.time())
    seam_values = itertools.chain(
        [base_now, base_now + _ADVANCE_SECONDS],
        itertools.repeat(base_now + _ADVANCE_SECONDS),
    )
    monkeypatch.setattr(pipeline_mod, "_rank_now", lambda: next(seam_values))
    calls = _wrap_score_counter(monkeypatch)

    n0 = len(calls)
    first = _run_recall(store, graph, assignment, rich_club, embedder, _CUE)
    n1 = len(calls)
    assert n1 - n0 >= 1, "the Rust scorer did not fire on the first recall"

    second = _run_recall(store, graph, assignment, rich_club, embedder, _CUE)
    n2 = len(calls)
    assert n2 - n1 >= 1, "the Rust scorer did not fire on the second recall"

    assert _serialize(first) != _serialize(second), (
        "advancing the clock seam between two Rust-scored recalls must "
        "change the serialized output -- otherwise the seam is not wired "
        "into the Rust rank call at all"
    )

"""Recalls in one session, driven through the real `core.dispatch` entry,
must never trip the `GenerationRegression` -> `recall_pipeline_fallback`
collision: the handle-owned generation reconciles to `max(local, engine)`
on every `snapshot()`, so a wrong sequence would silently degrade every
later recall to the Python fallback. The byte-parity harness bypasses this
entry entirely (it calls `recall_for_response` directly), so this is the
only coverage for that collision class.

The fallback is silent by design (`core/__init__.py`'s
`rank_index_snapshot_failed` debug log, never a raised exception or a
response field) -- asserted here via `caplog` on the emitting logger, not
by inferring it from hit counts alone.
"""
from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

import iai_mcp.pipeline as _pipeline_mod
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import MemoryRecord
from tests._helpers import stub_embedder_for_store

_DIM = 16  # small synthetic dim; avoids loading the Rust embedder
_FALLBACK_MARKER = "rank_index_snapshot_failed"


@pytest.fixture(autouse=True)
def _small_embed_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(_DIM))


@pytest.fixture(autouse=True)
def _crypto_passphrase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase-not-secret")


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(_keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p))
    monkeypatch.setattr(_keyring, "delete_password", lambda s, u: fake.pop((s, u), None))
    yield fake


def _select_driver(driver: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


class _StubEmbedder:
    """Deterministic stand-in embedder -- a fixed cue vector regardless of text."""

    def __init__(self, vec: list[float]) -> None:
        self._vec = vec

    def embed(self, _text: str) -> list[float]:
        return list(self._vec)


def _seeded_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _make_rec(rid: UUID, seed: int, surface: str, embedding: "list[float] | None" = None) -> MemoryRecord:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=rid,
        tier="episodic",
        literal_surface=surface,
        aaak_index="",
        embedding=embedding if embedding is not None else _seeded_vec(seed),
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
        tags=["capture"],
        language="en",
    )


def _seed_corpus(store: MemoryStore, cue_vec: list[float]) -> None:
    target = _make_rec(uuid4(), seed=1, surface="associative target record", embedding=cue_vec)
    store.insert(target)
    for i in range(6):
        store.insert(_make_rec(uuid4(), seed=100 + i, surface=f"unrelated filler {i}"))
    flush_record_buffer(store)


def _dispatch_recall(store: MemoryStore, cue_vec: list[float], session_id: str) -> dict:
    from iai_mcp import core as _core

    store._build_exact_index_sync()
    _pipeline_mod._last_recall_latency_ms = 0.0
    return _core.dispatch(store, "memory_recall", {
        "cue": "two-pass generation-safe probe",
        "session_id": session_id,
        "budget_tokens": 2000,
        "cue_embedding": cue_vec,
    })


@pytest.fixture
def _store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def _make(driver: str) -> MemoryStore:
        _select_driver(driver, monkeypatch)
        return MemoryStore(path=tmp_path / f"store-{driver}")
    return _make


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_second_recall_in_one_session_does_not_trip_generation_regression(
    driver, _store, monkeypatch, caplog,
):
    """First recall builds the resident rank index via `_rank_handle`'s
    `.snapshot()` call. Both this recall and a second one in the SAME
    session, driven through `core.dispatch`, must return hits and leave no
    `rank_index_snapshot_failed` trace -- the debug log `core/__init__.py`
    emits when `_rank_handle.snapshot()` raises `GenerationRegression`.
    This protects the persistent rank-index handle staying generation-safe
    across repeated recalls."""
    store = _store(driver)
    cue_vec = _seeded_vec(2)
    _seed_corpus(store, cue_vec)
    stub_embedder_for_store(monkeypatch, _StubEmbedder(cue_vec))

    caplog.set_level(logging.DEBUG, logger="iai_mcp.core")

    caplog.clear()
    resp1 = _dispatch_recall(store, cue_vec, session_id="two-pass-gen-safe")
    hits1 = resp1.get("hits", [])
    assert hits1, "first recall produced zero hits -- fixture is broken"
    fallback1 = [r for r in caplog.records if _FALLBACK_MARKER in r.getMessage()]
    assert not fallback1, (
        f"first recall unexpectedly tripped the fallback: "
        f"{[r.getMessage() for r in fallback1]}"
    )

    caplog.clear()
    resp2 = _dispatch_recall(store, cue_vec, session_id="two-pass-gen-safe")
    hits2 = resp2.get("hits", [])
    assert hits2, "second recall produced zero hits -- fixture is broken"
    fallback2 = [r for r in caplog.records if _FALLBACK_MARKER in r.getMessage()]
    assert not fallback2, (
        f"second recall tripped the rank-index generation-regression "
        f"fallback (GenerationRegression -> recall_pipeline_fallback): "
        f"{[r.getMessage() for r in fallback2]}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
@pytest.mark.parametrize("force_fold", [False, True])
def test_three_dispatches_with_interleaved_write_stay_generation_safe(
    driver, force_fold, _store, monkeypatch, caplog,
):
    """Three `core.dispatch` recalls in one session, with a hook-fed write
    (`store.insert`, the same path `_make_graph_sync_hook` feeds through)
    interleaved between recall 1 and 2, must never trip the fallback and
    must keep the Rust scorer engaged on every recall -- a wrong generation
    sequence would silently route every later recall to the Python
    fallback instead. `force_fold=True` additionally sets
    `IAI_MCP_RANK_OVERLAY_FOLD_THRESHOLD=1` so the interleaved write's
    `maybe_fold()` call runs a real fold before recall 2, proving fold
    never trips the fallback either.

    The handle's generation advances on every dispatch (every hydrated
    candidate is re-fed each call, not only on an external write) -- the
    assertion here is that it never regresses, not that it is flat between
    writes.
    """
    if force_fold:
        monkeypatch.setenv("IAI_MCP_RANK_OVERLAY_FOLD_THRESHOLD", "1")
    store = _store(driver)
    cue_vec = _seeded_vec(3)
    _seed_corpus(store, cue_vec)
    stub_embedder_for_store(monkeypatch, _StubEmbedder(cue_vec))

    from iai_mcp.store._rank_index import rank_index_for

    rust_calls = {"n": 0}
    orig_rank_now = _pipeline_mod._rank_now

    def _spy_rank_now() -> int:
        rust_calls["n"] += 1
        return orig_rank_now()

    monkeypatch.setattr(_pipeline_mod, "_rank_now", _spy_rank_now)

    caplog.set_level(logging.DEBUG, logger="iai_mcp.core")
    handle = rank_index_for(store)
    generations: list[int] = []
    rust_call_counts: list[int] = []

    def _run_recall(label: str) -> None:
        caplog.clear()
        rust_calls["n"] = 0
        resp = _dispatch_recall(store, cue_vec, session_id="three-pass-gen-safe")
        assert resp.get("hits"), f"{label} produced zero hits -- fixture is broken"
        fallback = [r for r in caplog.records if _FALLBACK_MARKER in r.getMessage()]
        assert not fallback, (
            f"{label} tripped the rank-index generation-regression fallback: "
            f"{[r.getMessage() for r in fallback]}"
        )
        generations.append(handle.generation)
        rust_call_counts.append(rust_calls["n"])

    _run_recall("recall 1")

    new_rec = _make_rec(uuid4(), seed=777, surface="interleaved write probe", embedding=cue_vec)
    store.insert(new_rec)

    _run_recall("recall 2")
    _run_recall("recall 3")

    assert all(c > 0 for c in rust_call_counts), (
        f"the Rust scorer must fire on every recall -- got call counts {rust_call_counts}"
    )
    assert generations[1] > generations[0], (
        f"generation must advance across the interleaved write: {generations}"
    )
    assert generations[2] > generations[1], (
        f"generation must keep advancing monotonically: {generations}"
    )

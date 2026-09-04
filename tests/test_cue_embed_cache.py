from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

import iai_mcp.embed as embed_mod
from iai_mcp import concurrency
from iai_mcp.concurrency import _dispatch_socket_request


@pytest.fixture(autouse=True)
def _reset_cue_cache() -> None:
    concurrency._cue_embed_vecs.clear()
    yield
    concurrency._cue_embed_vecs.clear()


class _FakeEmbedder:
    DIM = 4


def _make_spy(dim: int = 4, calls: "list[str] | None" = None):
    calls = calls if calls is not None else []

    def _spy(embedder: Any, text: str) -> list[float]:
        calls.append(text)
        return [float(len(text)), float(len(calls)), 0.5, 0.25][:dim]

    _spy.calls = calls  # type: ignore[attr-defined]
    return _spy


def _patch_embedder(monkeypatch: pytest.MonkeyPatch, spy, dim: int = 4) -> None:
    fake = _FakeEmbedder()
    fake.DIM = dim
    monkeypatch.setattr(embed_mod, "embedder_for_store", lambda _store, **_kw: fake)
    monkeypatch.setattr(embed_mod, "embed_query", spy)


def _dispatch(cue: str, store: Any = None) -> dict:
    store = store if store is not None else MagicMock()
    state: dict = {"fsm_state": "WAKE"}
    return asyncio.run(
        _dispatch_socket_request({"type": "embed_cue", "cue": cue}, store, state)
    )


def test_repeat_cue_hits_cache_byte_identical_and_single_embedder_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    spy = _make_spy(calls=calls)
    _patch_embedder(monkeypatch, spy)

    resp1 = _dispatch("hello world")
    resp2 = _dispatch("hello world")

    assert resp1["ok"] is True
    assert resp2["ok"] is True
    assert resp1["embedding"] == resp2["embedding"]
    assert len(calls) == 1
    assert resp2["embedding"] is not concurrency._cue_embed_vecs["hello world"]


def test_different_cues_get_distinct_embeddings_and_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    spy = _make_spy(calls=calls)
    _patch_embedder(monkeypatch, spy)

    resp1 = _dispatch("cue one")
    resp2 = _dispatch("cue two")

    assert resp1["ok"] is True
    assert resp2["ok"] is True
    assert resp1["embedding"] != resp2["embedding"]
    assert len(calls) == 2
    assert len(concurrency._cue_embed_vecs) == 2


def test_dim_mismatch_not_cached_and_reembeds_on_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    spy = _make_spy(dim=3, calls=calls)
    _patch_embedder(monkeypatch, spy, dim=4)

    resp1 = _dispatch("bad dim cue")
    assert resp1["ok"] is False
    assert resp1["reason"] == "embed_dim_mismatch"
    assert "bad dim cue" not in concurrency._cue_embed_vecs
    assert len(calls) == 1

    resp2 = _dispatch("bad dim cue")
    assert resp2["ok"] is False
    assert resp2["reason"] == "embed_dim_mismatch"
    assert len(calls) == 2


def test_dim_mismatch_dropped_on_cache_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    spy = _make_spy(calls=calls)
    _patch_embedder(monkeypatch, spy)

    resp1 = _dispatch("hit dim cue")
    assert resp1["ok"] is True
    assert "hit dim cue" in concurrency._cue_embed_vecs

    concurrency._cue_embed_vecs["hit dim cue"] = [0.0, 0.0]

    resp2 = _dispatch("hit dim cue")

    assert resp2["ok"] is True
    assert len(resp2["embedding"]) == 4
    assert len(calls) == 2
    assert len(concurrency._cue_embed_vecs["hit dim cue"]) == 4


def test_cache_bounded_clears_at_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    spy = _make_spy(calls=calls)
    _patch_embedder(monkeypatch, spy)

    cap = concurrency._CUE_EMBED_VEC_CAP
    for i in range(cap + 5):
        _dispatch(f"cue-{i}")
        assert len(concurrency._cue_embed_vecs) <= cap


def test_kill_switch_disables_caching(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_CUE_EMBED_CACHE", "0")
    calls: list[str] = []
    spy = _make_spy(calls=calls)
    _patch_embedder(monkeypatch, spy)

    _dispatch("same cue")
    _dispatch("same cue")

    assert len(calls) == 2
    assert concurrency._cue_embed_vecs == {}


def test_kill_switch_purges_stale_cache_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    spy = _make_spy(calls=calls)
    _patch_embedder(monkeypatch, spy)

    _dispatch("stale cue")
    assert "stale cue" in concurrency._cue_embed_vecs

    monkeypatch.setenv("IAI_MCP_CUE_EMBED_CACHE", "0")
    _dispatch("stale cue")

    assert concurrency._cue_embed_vecs == {}


def test_identity_guard_reruns_on_cache_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    spy = _make_spy(calls=calls)
    _patch_embedder(monkeypatch, spy)

    resp1 = _dispatch("guarded cue")
    assert resp1["ok"] is True

    guard_calls: list[Any] = []
    real_embedder_for_store = embed_mod.embedder_for_store

    def _failing_embedder_for_store(store: Any, **kw: Any) -> Any:
        guard_calls.append(store)
        raise embed_mod.EmbedIdentityMismatch("store vectors are a foreign generation")

    monkeypatch.setattr(embed_mod, "embedder_for_store", _failing_embedder_for_store)

    resp2 = _dispatch("guarded cue")

    assert len(guard_calls) == 1
    assert resp2["ok"] is False
    assert resp2["reason"] == "daemon_not_ready"
    monkeypatch.setattr(embed_mod, "embedder_for_store", real_embedder_for_store)

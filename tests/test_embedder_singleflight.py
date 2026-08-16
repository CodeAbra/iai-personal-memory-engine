"""Process-wide embedder construction single-flight.

``embedder_for_store`` has no in-process construction guard: every call built
a brand-new ``Embedder()``. A second concurrent caller during a cold build
started an independent, uncoordinated construction racing the first for the
same on-disk model resource. These tests discriminate the guard directly
against stub factories (model-free, fast) and against the real entry points
with a monkeypatched ``Embedder`` class (no Rust construction).
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _clean_singleflight_cache():
    from iai_mcp import embed as embed_mod

    embed_mod._reset_embedder_singleton()
    yield
    embed_mod._reset_embedder_singleton()


def test_share_single_build_across_threads():
    from iai_mcp import embed as embed_mod

    calls = {"n": 0}
    proceed = threading.Event()

    def factory():
        calls["n"] += 1
        proceed.wait(timeout=2.0)
        return object()

    key = ("share-key",)
    results: dict[str, object] = {}

    def builder():
        results["builder"] = embed_mod._build_or_get_shared_embedder(key, factory)

    def waiter():
        results["waiter"] = embed_mod._build_or_get_shared_embedder(key, factory)

    t1 = threading.Thread(target=builder)
    t1.start()
    time.sleep(0.05)  # let t1 acquire the lock and enter factory()
    t2 = threading.Thread(target=waiter)
    t2.start()
    time.sleep(0.05)  # let t2 block on the same lock
    proceed.set()
    t1.join(timeout=3)
    t2.join(timeout=3)

    assert calls["n"] == 1, "factory must fire exactly once for a shared key"
    assert results["builder"] is results["waiter"]


def test_key_discriminates_on_text_prefix(monkeypatch: pytest.MonkeyPatch):
    from iai_mcp import embed as embed_mod

    monkeypatch.delenv("IAI_MCP_EMBED_TEXT_PREFIX", raising=False)
    key_a = embed_mod._embedder_construction_key(
        provider="native", model_key="bge-small-en-v1.5"
    )
    monkeypatch.setenv("IAI_MCP_EMBED_TEXT_PREFIX", "query: ")
    key_b = embed_mod._embedder_construction_key(
        provider="native", model_key="bge-small-en-v1.5"
    )
    assert key_a != key_b


def test_key_discriminates_on_quantize_mode(monkeypatch: pytest.MonkeyPatch):
    from iai_mcp import embed as embed_mod

    monkeypatch.delenv("IAI_MCP_EMBED_QUANTIZE", raising=False)
    key_a = embed_mod._embedder_construction_key(
        provider="native", model_key="bge-small-en-v1.5"
    )
    monkeypatch.setenv("IAI_MCP_EMBED_QUANTIZE", "int8")
    key_b = embed_mod._embedder_construction_key(
        provider="native", model_key="bge-small-en-v1.5"
    )
    assert key_a != key_b


def test_key_discriminates_on_resolved_model_key():
    from iai_mcp import embed as embed_mod

    key_a = embed_mod._embedder_construction_key(
        provider="native", model_key="bge-small-en-v1.5"
    )
    key_b = embed_mod._embedder_construction_key(
        provider="native", model_key="multilingual-e5-small"
    )
    assert key_a != key_b


def test_key_discriminates_http_config_from_native():
    from iai_mcp import embed as embed_mod

    native_key = embed_mod._embedder_construction_key(
        provider="native", model_key="bge-small-en-v1.5"
    )
    http_key = embed_mod._embedder_construction_key(
        provider="http", http_config=("http://127.0.0.1:1/embed", 3, 30.0, "m")
    )
    assert native_key != http_key


def test_failed_build_does_not_poison_cache_and_retries_fresh():
    from iai_mcp import embed as embed_mod

    calls = {"n": 0}

    def bad_then_good():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return "instance"

    key = ("poison-key",)
    with pytest.raises(RuntimeError, match="boom"):
        embed_mod._build_or_get_shared_embedder(key, bad_then_good)

    # the slot must be EMPTY after a failure -- the next caller retries fresh
    result = embed_mod._build_or_get_shared_embedder(key, bad_then_good)
    assert result == "instance"
    assert calls["n"] == 2


def test_reset_singleton_clears_cache_for_retry():
    from iai_mcp import embed as embed_mod

    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return object()

    key = ("reset-key",)
    first = embed_mod._build_or_get_shared_embedder(key, factory)
    assert calls["n"] == 1
    second = embed_mod._build_or_get_shared_embedder(key, factory)
    assert second is first
    assert calls["n"] == 1  # cache hit, no rebuild

    embed_mod._reset_embedder_singleton()
    third = embed_mod._build_or_get_shared_embedder(key, factory)
    assert calls["n"] == 2
    assert third is not first


def test_bounded_acquire_shares_build_when_ready_within_timeout():
    from iai_mcp import embed as embed_mod

    calls = {"n": 0}
    proceed = threading.Event()

    def factory():
        calls["n"] += 1
        proceed.wait(timeout=2.0)
        return "shared-instance"

    key = ("bounded-share-key",)
    results: dict[str, object] = {}

    def builder():
        results["builder"] = embed_mod._build_or_get_shared_embedder(key, factory)

    def waiter():
        results["waiter"] = embed_mod._build_or_get_shared_embedder(
            key, factory, timeout=2.0
        )

    t1 = threading.Thread(target=builder)
    t1.start()
    time.sleep(0.05)
    t2 = threading.Thread(target=waiter)
    t2.start()
    time.sleep(0.05)
    proceed.set()
    t1.join(timeout=3)
    t2.join(timeout=3)

    assert calls["n"] == 1
    assert results["builder"] == results["waiter"] == "shared-instance"


def test_bounded_acquire_signals_not_ready_when_wait_exceeds_timeout():
    from iai_mcp import embed as embed_mod

    proceed = threading.Event()

    def factory():
        proceed.wait(timeout=2.0)
        return "instance"

    key = ("bounded-timeout-key",)
    t = threading.Thread(
        target=lambda: embed_mod._build_or_get_shared_embedder(key, factory)
    )
    t.start()
    time.sleep(0.05)  # let the builder acquire the lock and enter factory()

    with pytest.raises(embed_mod._EmbedderBuildNotReady):
        embed_mod._build_or_get_shared_embedder(key, factory, timeout=0.01)

    proceed.set()
    t.join(timeout=3)


def test_try_embedder_for_store_returns_none_when_build_not_ready(
    monkeypatch: pytest.MonkeyPatch,
):
    from iai_mcp import embed as embed_mod

    proceed = threading.Event()

    class _SlowStub:
        DIM = 384

        def __init__(self, *_a, **_kw) -> None:
            proceed.wait(timeout=2.0)

    monkeypatch.setattr(embed_mod, "Embedder", _SlowStub)
    store = SimpleNamespace(embed_dim=None)

    t = threading.Thread(target=lambda: embed_mod.embedder_for_store(store))
    t.start()
    time.sleep(0.05)  # let the builder thread take the lock

    result = embed_mod.try_embedder_for_store(store, build_timeout=0.01)
    assert result is None

    proceed.set()
    t.join(timeout=3)


def test_try_embedder_for_store_returns_shared_instance_when_ready(
    monkeypatch: pytest.MonkeyPatch,
):
    from iai_mcp import embed as embed_mod

    class _FastStub:
        DIM = 384

        def __init__(self, *_a, **_kw) -> None:
            pass

    monkeypatch.setattr(embed_mod, "Embedder", _FastStub)
    store = SimpleNamespace(embed_dim=None)

    result = embed_mod.try_embedder_for_store(store, build_timeout=2.0)
    assert isinstance(result, _FastStub)


def test_try_embedder_for_store_propagates_identity_mismatch_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
):
    from iai_mcp import embed as embed_mod

    class _FastStub:
        DIM = 384

        def __init__(self, *_a, **_kw) -> None:
            pass

    def _raise_mismatch(_store, _embedder, *, allow_mismatch):
        raise embed_mod.EmbedIdentityMismatch("stub mismatch")

    monkeypatch.setattr(embed_mod, "Embedder", _FastStub)
    monkeypatch.setattr(embed_mod, "_enforce_store_embed_identity", _raise_mismatch)
    store = SimpleNamespace(embed_dim=None)

    with pytest.raises(embed_mod.EmbedIdentityMismatch):
        embed_mod.try_embedder_for_store(store, build_timeout=1.0)

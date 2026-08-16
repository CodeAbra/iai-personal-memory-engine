"""The boot-window embedder-build-degrade backstop must bound TOTAL recall
latency, not just the embedder-acquire wait.

The bounded-wait acquire (``try_embedder_for_store(..., build_timeout=...)``)
already bounds how long a recall waits to SHARE an in-flight embedder build
before it degrades. Once degraded, the fallback recall must return fast and
never re-enter the same unbounded construction path the acquire just gave up
on -- otherwise the "bounded" backstop's total latency is unbounded in
practice, tracking whatever the concurrent construction happens to cost.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


@pytest.fixture(autouse=True)
def _clean_singleflight_cache():
    from iai_mcp import embed as embed_mod

    embed_mod._reset_embedder_singleton()
    yield
    embed_mod._reset_embedder_singleton()


def _seed_one_record(store: MemoryStore) -> None:
    _seed_records(store, n=1)


def _seed_records(store: MemoryStore, *, n: int) -> None:
    # Seeded with n>1 for the degrade-path test: a zero-norm query vector
    # (the degraded path's cue) must still surface real hits on a non-trivial
    # corpus -- one record is not enough to distinguish "returns fast" from
    # "returns fast and empty", which would violate Hippo always answers.
    now = datetime.now(timezone.utc)
    for i in range(n):
        store.insert(
            MemoryRecord(
                id=uuid4(),
                tier="episodic",
                literal_surface=f"boot window degrade backstop probe record {i}",
                aaak_index="",
                embedding=[1.0] + [0.0] * (EMBED_DIM - 1),
                community_id=None,
                centrality=0.0,
                detail_level=1,
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
        )


def test_degraded_recall_returns_fast_despite_concurrent_slow_construction(
    tmp_path, monkeypatch,
):
    """Attribution + regression: a slow embedder construction held on another
    thread must not make the DEGRADED recall's total latency track the
    construction's duration. Injects the construction slowness directly at
    the embedder single-flight lock (embed.py::_embedder_lock) -- the same
    lock a real cold model load holds for its full duration -- so this does
    not depend on a real model or a real cold download.
    """
    import iai_mcp.embed as _embed_mod
    from iai_mcp import core

    store = MemoryStore(path=tmp_path)
    _seed_records(store, n=20)

    hold_sec = 1.2
    lock_held = threading.Event()
    may_release = threading.Event()

    def _hold_lock():
        with _embed_mod._embedder_lock:
            lock_held.set()
            may_release.wait(10)

    holder = threading.Thread(target=_hold_lock, daemon=True)
    holder.start()
    assert lock_held.wait(5), "helper thread never acquired the embedder lock"
    # Released by its own timer, independent of when dispatch() returns --
    # releasing only after dispatch() would deadlock if dispatch blocks on
    # this same lock, which is exactly the mechanism under test.
    threading.Timer(hold_sec, may_release.set).start()

    # Attribution probe (throwaway): time wall-clock actually spent inside
    # the single-flight construction helper during this call, to attribute
    # the degraded recall's cost to that specific function rather than to
    # unspecified "scheduling contention".
    probe_ms: list[float] = []
    real_build_or_get = _embed_mod._build_or_get_shared_embedder

    def _timed_build_or_get(*args, **kwargs):
        _t0 = time.perf_counter()
        try:
            return real_build_or_get(*args, **kwargs)
        finally:
            probe_ms.append((time.perf_counter() - _t0) * 1000.0)

    monkeypatch.setattr(
        _embed_mod, "_build_or_get_shared_embedder", _timed_build_or_get
    )

    try:
        t0 = time.monotonic()
        resp = core.dispatch(
            store, "memory_recall",
            {"cue": "boot window degrade backstop probe record", "session_id": "probe"},
        )
        elapsed = time.monotonic() - t0
    finally:
        may_release.set()
        holder.join(5)

    assert "error" not in resp, f"memory_recall returned an error: {resp.get('error')}"
    result = resp
    assert result.get("_source") == "embedder-build-degrade", (
        f"expected the degrade indicator to fire, got _source={result.get('_source')!r}"
    )
    # A fast but EMPTY degrade answer is not "Hippo always answers" -- a
    # zero-norm query vector must still surface real hits on a non-trivial
    # corpus, not a degenerate empty result indistinguishable from a bug.
    assert result.get("hits"), (
        "degraded recall returned zero hits on a 20-record corpus -- a fast "
        "but empty answer is not an honest degrade"
    )

    # The degrade fallback must not re-enter the same unbounded single-flight
    # lock the bounded acquire just gave up sharing: total latency must not
    # track the concurrent construction's hold duration (~1.2s here).
    assert elapsed < 0.9, (
        f"degraded recall took {elapsed:.3f}s while a concurrent construction "
        f"held the embedder lock for {hold_sec}s -- the degrade backstop must "
        f"bound TOTAL latency, not track the concurrent construction's cost"
    )

    # Attribution: the single-flight helper is entered exactly once (the
    # bounded acquire itself, bounded near its own 0.44s budget) -- never a
    # second, unbounded time for a re-embed attempt. A second entry, or one
    # whose duration tracks hold_sec instead of the acquire's own bound,
    # would mean the fallback re-entered the construction wait it already
    # gave up sharing.
    assert len(probe_ms) == 1, (
        f"expected exactly one single-flight construction attempt (the "
        f"bounded acquire), got {len(probe_ms)}: {probe_ms}"
    )
    assert probe_ms[0] < 0.5 * hold_sec * 1000.0, (
        f"the single-flight construction attempt took {probe_ms[0]:.1f}ms, "
        f"tracking the concurrent hold duration ({hold_sec}s) instead of its "
        f"own bounded acquire budget -- the degrade path re-entered the "
        f"unbounded construction wait"
    )


def test_recall_skips_reembed_when_allow_cue_reembed_is_false(tmp_path, monkeypatch):
    """allow_cue_reembed=False must skip the re-embed fallback entirely for
    an invalid (e.g. all-zero) cue vector -- the caller's vector is searched
    as-is, never upgraded through embedder_for_store."""
    import iai_mcp.embed as _embed_mod
    from iai_mcp import retrieve
    from iai_mcp.types import EMBED_DIM

    store = MemoryStore(path=tmp_path)
    _seed_one_record(store)

    calls = {"n": 0}

    def _spy_embedder_for_store(_store, **_kwargs):
        calls["n"] += 1
        raise AssertionError("embedder_for_store must not be called")

    monkeypatch.setattr(_embed_mod, "embedder_for_store", _spy_embedder_for_store)

    resp = retrieve.recall(
        store=store,
        cue_embedding=[0.0] * EMBED_DIM,
        cue_text="boot window degrade backstop probe record",
        session_id="probe",
        allow_cue_reembed=False,
    )

    assert calls["n"] == 0, "embedder_for_store was called despite allow_cue_reembed=False"
    assert resp is not None


def test_sleep_cortex_fallback_bounds_embedder_acquire_under_concurrent_construction(
    tmp_path, monkeypatch,
):
    """The SLEEP-state cortex-fallback recall site must bound its embedder
    acquire exactly like the primary boot-window path: a recall through this
    site cannot block on a concurrent single-flight construction beyond the
    boot-window bound."""
    import iai_mcp.embed as _embed_mod
    from iai_mcp import core

    store = MemoryStore(path=tmp_path)
    _seed_records(store, n=20)

    monkeypatch.setattr(
        "iai_mcp.daemon_state.load_state", lambda: {"current_state": "SLEEP"},
    )
    monkeypatch.setattr("iai_mcp.daemon_state.save_state", lambda s: None)
    monkeypatch.setattr(
        core, "_CRISIS_STATE_CACHE", {"crisis": {"crisis_mode": False}},
    )

    hold_sec = 1.2
    lock_held = threading.Event()
    may_release = threading.Event()

    def _hold_lock():
        with _embed_mod._embedder_lock:
            lock_held.set()
            may_release.wait(10)

    holder = threading.Thread(target=_hold_lock, daemon=True)
    holder.start()
    assert lock_held.wait(5), "helper thread never acquired the embedder lock"
    threading.Timer(hold_sec, may_release.set).start()

    try:
        t0 = time.monotonic()
        resp = core.dispatch(
            store, "memory_recall",
            {"cue": "boot window degrade backstop probe record", "session_id": "probe"},
        )
        elapsed = time.monotonic() - t0
    finally:
        may_release.set()
        holder.join(5)

    assert "error" not in resp, f"memory_recall returned an error: {resp.get('error')}"
    assert resp.get("_source") == "cortex-fallback", (
        f"expected the cortex-fallback indicator, got _source={resp.get('_source')!r}"
    )
    assert resp.get("hits"), (
        "cortex-fallback recall returned zero hits on a 20-record corpus"
    )
    assert elapsed < 0.9, (
        f"cortex-fallback recall took {elapsed:.3f}s while a concurrent "
        f"construction held the embedder lock for {hold_sec}s -- the sibling "
        f"site must bound its acquire, not track the concurrent build's cost"
    )


def test_generic_fallback_bounds_embedder_acquire_under_concurrent_construction(
    tmp_path, monkeypatch,
):
    """The generic exception-fallback recall site must bound its embedder
    acquire exactly like the primary boot-window path, for a genuine
    non-embedder pipeline exception."""
    import iai_mcp.embed as _embed_mod
    import iai_mcp.pipeline as _pipeline_mod
    from iai_mcp import core

    store = MemoryStore(path=tmp_path)
    _seed_records(store, n=20)

    # A genuine non-embedder pipeline failure -- routes this recall into the
    # generic except handler (site 3) before the primary path's own embedder
    # acquire ever runs, so this site's bounded acquire is the FIRST attempt.
    # `iai_mcp.pipeline` is already imported by this point in dispatch() (the
    # unrelated `recall_for_response` import runs first), so this only
    # removes the one name the inner try block needs.
    monkeypatch.delattr(_pipeline_mod, "K_CANDIDATES", raising=True)

    hold_sec = 1.2
    lock_held = threading.Event()
    may_release = threading.Event()

    def _hold_lock():
        with _embed_mod._embedder_lock:
            lock_held.set()
            may_release.wait(10)

    holder = threading.Thread(target=_hold_lock, daemon=True)
    holder.start()
    assert lock_held.wait(5), "helper thread never acquired the embedder lock"
    threading.Timer(hold_sec, may_release.set).start()

    try:
        t0 = time.monotonic()
        resp = core.dispatch(
            store, "memory_recall",
            {"cue": "boot window degrade backstop probe record", "session_id": "probe"},
        )
        elapsed = time.monotonic() - t0
    finally:
        may_release.set()
        holder.join(5)

    assert "error" not in resp, f"memory_recall returned an error: {resp.get('error')}"
    assert resp.get("hits"), (
        "generic-fallback recall returned zero hits on a 20-record corpus"
    )
    assert elapsed < 0.9, (
        f"generic-fallback recall took {elapsed:.3f}s while a concurrent "
        f"construction held the embedder lock for {hold_sec}s -- the sibling "
        f"site must bound its acquire, not track the concurrent build's cost"
    )


def test_sibling_fallback_reembeds_cue_when_embedder_ready(tmp_path, monkeypatch):
    """With the embedder resident (no concurrent construction), the SLEEP
    cortex-fallback site must still re-embed the cue -- non-boot recall
    quality preserved, no blind allow_cue_reembed=False."""
    import iai_mcp.embed as _embed_mod
    from iai_mcp import core

    store = MemoryStore(path=tmp_path)
    _seed_records(store, n=5)

    calls = {"n": 0}

    def _spy_embedder_for_store(_store, **_kwargs):
        calls["n"] += 1

        class _S:
            def embed(self, text):
                return [1.0] + [0.0] * (EMBED_DIM - 1)

        return _S()

    monkeypatch.setattr(_embed_mod, "embedder_for_store", _spy_embedder_for_store)
    monkeypatch.setattr(
        "iai_mcp.daemon_state.load_state", lambda: {"current_state": "SLEEP"},
    )
    monkeypatch.setattr("iai_mcp.daemon_state.save_state", lambda s: None)
    monkeypatch.setattr(
        core, "_CRISIS_STATE_CACHE", {"crisis": {"crisis_mode": False}},
    )

    resp = core.dispatch(
        store, "memory_recall",
        {"cue": "boot window degrade backstop probe record", "session_id": "probe"},
    )

    assert "error" not in resp, f"memory_recall returned an error: {resp.get('error')}"
    assert resp.get("_source") == "cortex-fallback"
    assert calls["n"] >= 1, (
        "embedder_for_store was never called -- the sibling site did not "
        "re-embed the cue despite the embedder being resident"
    )
    assert resp.get("hits"), "expected real hits from the reembedded cue"


def test_generic_fallback_propagates_identity_mismatch_instead_of_degrading(
    tmp_path, monkeypatch,
):
    """A mismatched-identity store must raise through the generic-fallback
    site rather than degrading silently -- consistent with core:997's
    philosophy that a vector-space refusal must surface."""
    import iai_mcp.embed as _embed_mod
    import iai_mcp.pipeline as _pipeline_mod
    from iai_mcp import core

    store = MemoryStore(path=tmp_path)
    _seed_records(store, n=5)

    # Routes this recall into the generic except handler (site 3) via a
    # genuine non-embedder pipeline failure, same technique as the bounded-
    # acquire sibling test above.
    monkeypatch.delattr(_pipeline_mod, "K_CANDIDATES", raising=True)

    def _refuse(_store, **_kwargs):
        raise _embed_mod.EmbedIdentityMismatch("simulated vector space mismatch")

    monkeypatch.setattr(_embed_mod, "embedder_for_store", _refuse)

    with pytest.raises(_embed_mod.EmbedIdentityMismatch):
        core.dispatch(
            store, "memory_recall",
            {"cue": "boot window degrade backstop probe record", "session_id": "probe"},
        )


def test_recall_reembeds_by_default_when_cue_vector_invalid(tmp_path, monkeypatch):
    """The default (allow_cue_reembed=True, unchanged) still re-embeds an
    invalid cue vector via the cue text -- byte-identical to pre-existing
    callers that never pass the new keyword."""
    import iai_mcp.embed as _embed_mod
    from iai_mcp import retrieve
    from iai_mcp.types import EMBED_DIM

    store = MemoryStore(path=tmp_path)
    _seed_one_record(store)

    calls = {"n": 0}

    def _spy_embedder_for_store(_store, **_kwargs):
        calls["n"] += 1

        class _S:
            def embed(self, text):
                return [1.0] + [0.0] * (EMBED_DIM - 1)

        return _S()

    monkeypatch.setattr(_embed_mod, "embedder_for_store", _spy_embedder_for_store)

    retrieve.recall(
        store=store,
        cue_embedding=[0.0] * EMBED_DIM,
        cue_text="boot window degrade backstop probe record",
        session_id="probe",
    )

    assert calls["n"] == 1, "default behavior must still re-embed an invalid cue vector"

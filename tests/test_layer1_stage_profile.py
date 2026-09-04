"""Guards for the hydrate/fetch stage bucket and the repeat-seen candidate-id
overlap counter (``IAI_MCP_STAGE_PROFILE=1``), added to candidate assembly in
``core.dispatch``'s ``memory_recall`` branch (``core/__init__.py``). Mirrors
the ``degree`` stage bucket pattern already covered for ``recall_for_response``
in ``tests/test_stage_profile_instrumentation.py``.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

import iai_mcp.pipeline as _pipeline_mod
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import MemoryRecord
from tests._helpers import stub_embedder_for_store

_DIM = 16  # small synthetic dim; avoids loading the Rust embedder
_FIXED_CREATED_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)

# The sub-stage timers threaded through RecallResponse.stage_timings ->
# response["_stage_timings"]: t11_t12 (written directly in _recall_core's
# rust-scorer branch) plus the 10 keys merged in from hydrate_stage_timings.
NEW_SUBSTAGE_KEYS = frozenset({
    "t11_t12",
    "ann_scan", "ann_inlist", "ann_decode",
    "ann_rows_fetched", "ann_rows_served",
    "ge_populate", "ge_incident", "ge_split", "ge_contr_fetch",
    "hops_snapshot",
})


@pytest.fixture(autouse=True)
def _small_embed_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(_DIM))


def _select_driver(driver: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


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


@pytest.fixture(autouse=True)
def _clear_authority_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IAI_MCP_EXACT_AUTHORITY_OFF", raising=False)


class _StubEmbedder:
    """Deterministic stand-in embedder — a fixed cue vector regardless of text."""

    def __init__(self, vec: list[float]) -> None:
        self._vec = vec

    def embed(self, _text: str) -> list[float]:
        return list(self._vec)


def _seeded_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _make_rec(
    rid: UUID, seed: int, surface: str, *,
    embedding: list[float] | None = None,
    created_at: datetime | None = None,
) -> MemoryRecord:
    ts = created_at or _FIXED_CREATED_AT
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
        created_at=ts,
        updated_at=ts,
        tags=["capture"],
        language="en",
    )


def _stub_embedder_for_store(monkeypatch: pytest.MonkeyPatch, vec: list[float]) -> None:
    stub_embedder_for_store(monkeypatch, _StubEmbedder(vec))


def _seed_corpus(store: MemoryStore, cue_vec: list[float]) -> None:
    target = _make_rec(uuid4(), seed=1, surface="associative target record", embedding=cue_vec)
    store.insert(target)
    for i in range(6):
        store.insert(_make_rec(uuid4(), seed=100 + i, surface=f"unrelated filler {i}"))
    flush_record_buffer(store)


def _seed_deterministic_corpus(store: MemoryStore, cue_vec: list[float], base: int) -> None:
    """Same shape as ``_seed_corpus`` but with STABLE ids/timestamps, so two
    separately-built stores produce byte-identical candidate content — used
    by the flag-off-vs-flag-on differential, which must never confound
    "the instrumentation changed nothing" with "two random corpora differ"."""
    target_id = UUID(int=base + 1)
    store.insert(_make_rec(target_id, seed=1, surface="associative target record", embedding=cue_vec))
    for i in range(6):
        store.insert(_make_rec(UUID(int=base + 100 + i), seed=100 + i, surface=f"unrelated filler {i}"))
    flush_record_buffer(store)


def _dispatch_recall(store: MemoryStore, cue_vec: list[float], session_id: str = "layer1-stage-profile") -> dict:
    from iai_mcp import core as _core

    store._build_exact_index_sync()
    _pipeline_mod._last_recall_latency_ms = 0.0
    return _core.dispatch(store, "memory_recall", {
        "cue": "layer1 stage profile probe",
        "session_id": session_id,
        "budget_tokens": 2000,
        "cue_embedding": cue_vec,
    })


@pytest.fixture
def _store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def _make(driver: str, suffix: str = "") -> MemoryStore:
        _select_driver(driver, monkeypatch)
        return MemoryStore(path=tmp_path / f"store-{driver}{suffix}")
    return _make


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_hydrate_bucket_reports_ann_getbatch_split(driver, _store, monkeypatch):
    """IAI_MCP_STAGE_PROFILE=1 MUST report a hydrate bucket split into the
    ANN-seed decode (hydrate_ann) and the get_batch decode (hydrate_getbatch),
    with hydrate the sum of both — mirrors the shape of the existing degree
    stage bucket."""
    monkeypatch.setenv("IAI_MCP_STAGE_PROFILE", "1")
    store = _store(driver)
    cue_vec = _seeded_vec(1)
    _seed_corpus(store, cue_vec)
    _stub_embedder_for_store(monkeypatch, cue_vec)

    _pipeline_mod._last_stage_timings_ms.clear()
    _dispatch_recall(store, cue_vec)

    timings = _pipeline_mod._last_stage_timings_ms
    assert "hydrate_ann" in timings, timings
    assert "hydrate_getbatch" in timings, timings
    assert "hydrate" in timings, timings
    assert timings["hydrate_ann"] >= 0.0
    assert timings["hydrate_getbatch"] >= 0.0
    assert timings["hydrate"] == pytest.approx(
        timings["hydrate_ann"] + timings["hydrate_getbatch"]
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_overlap_fraction_zero_on_first_call_full_on_identical_repeat(driver, _store, monkeypatch):
    """First call on a store has no prior candidate set -- overlap is 0.0.
    An identical repeat call (same store, same cue, no inserts/deletes in
    between) resolves to the SAME candidate id set, so overlap is 1.0. The
    fraction stays in [0, 1] on both calls."""
    monkeypatch.setenv("IAI_MCP_STAGE_PROFILE", "1")
    store = _store(driver)
    cue_vec = _seeded_vec(2)
    _seed_corpus(store, cue_vec)
    _stub_embedder_for_store(monkeypatch, cue_vec)

    _pipeline_mod._last_stage_timings_ms.clear()
    _dispatch_recall(store, cue_vec)
    first_overlap = _pipeline_mod._last_stage_timings_ms.get("candidate_overlap_fraction")
    assert first_overlap is not None
    assert first_overlap == pytest.approx(0.0)

    _pipeline_mod._last_stage_timings_ms.clear()
    _dispatch_recall(store, cue_vec)
    second_overlap = _pipeline_mod._last_stage_timings_ms.get("candidate_overlap_fraction")
    assert second_overlap is not None
    assert 0.0 <= second_overlap <= 1.0
    assert second_overlap == pytest.approx(1.0)


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_stage_timings_surface_in_response(driver, _store, monkeypatch):
    """IAI_MCP_STAGE_PROFILE=1 sub-stage timers (t11_t12, ann_*, ge_*,
    hops_snapshot) must reach the memory_recall JSON-RPC response as
    response["_stage_timings"], not just the in-process
    pipeline._last_stage_timings_ms global that no out-of-process caller
    can read."""
    monkeypatch.setenv("IAI_MCP_STAGE_PROFILE", "1")
    store = _store(driver)
    cue_vec = _seeded_vec(6)
    _seed_corpus(store, cue_vec)
    _stub_embedder_for_store(monkeypatch, cue_vec)

    _pipeline_mod._last_stage_timings_ms.clear()
    resp = _dispatch_recall(store, cue_vec)

    assert "_stage_timings" in resp, resp
    st = resp["_stage_timings"]
    for key in NEW_SUBSTAGE_KEYS:
        assert key in st, (key, st)
        assert st[key] >= 0.0, (key, st)
    # This single in-process call populated both the response's per-call
    # copy and the legacy global identically -- the fix reproduces the
    # global's content faithfully, it does not fork a second measurement.
    assert st == _pipeline_mod._last_stage_timings_ms


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_stage_timings_response_survives_global_clobber_not_torn(driver, _store, monkeypatch):
    """response["_stage_timings"] must come from a per-call structure, not
    a live read of the pipeline._last_stage_timings_ms module global --
    the daemon's asyncio.to_thread concurrent-request fan-out means a
    second in-flight recall can clear/overwrite that global before this
    call's response is built. Reproduced by clobbering the global from
    inside the last hook that still runs before the response is built
    (_backfill_hit_metadata) -- exactly the race window a module-global
    read at the response site would fall into. A response built from the
    global would observe the clobbered sentinel values; one built from the
    per-call structure must not."""
    monkeypatch.setenv("IAI_MCP_STAGE_PROFILE", "1")
    store = _store(driver)
    cue_vec = _seeded_vec(7)
    _seed_corpus(store, cue_vec)
    _stub_embedder_for_store(monkeypatch, cue_vec)

    real_backfill = _pipeline_mod._backfill_hit_metadata

    def _clobbering_backfill(*args, **kwargs):
        real_backfill(*args, **kwargs)
        _pipeline_mod._last_stage_timings_ms.clear()
        _pipeline_mod._last_stage_timings_ms.update(
            {k: -999.0 for k in NEW_SUBSTAGE_KEYS}
        )

    monkeypatch.setattr(_pipeline_mod, "_backfill_hit_metadata", _clobbering_backfill)

    resp = _dispatch_recall(store, cue_vec)
    assert "_stage_timings" in resp
    st = resp["_stage_timings"]
    assert all(k in st for k in NEW_SUBSTAGE_KEYS), st
    assert all(st[k] != -999.0 for k in NEW_SUBSTAGE_KEYS), st


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_stage_timings_no_cross_thread_leak_under_concurrent_dispatch(driver, _store, monkeypatch):
    """Real concurrent dispatch (ThreadPoolExecutor with a synchronization
    barrier, matching the daemon's asyncio.to_thread request fan-out shape):
    two recalls in flight at once must each surface their own per-call
    stage timings in their own response, never a torn mix with the other
    call's."""
    monkeypatch.setenv("IAI_MCP_STAGE_PROFILE", "1")

    cue_a = _seeded_vec(11)
    cue_b = _seeded_vec(22)
    store_a = _store(driver, suffix="-concurrent-a")
    store_b = _store(driver, suffix="-concurrent-b")
    _seed_deterministic_corpus(store_a, cue_a, base=5000)
    _seed_deterministic_corpus(store_b, cue_b, base=6000)
    store_a._build_exact_index_sync()
    store_b._build_exact_index_sync()

    import iai_mcp.embed as _embed_mod
    _embedder_by_store_id = {
        id(store_a): _StubEmbedder(cue_a),
        id(store_b): _StubEmbedder(cue_b),
    }
    monkeypatch.setattr(
        _embed_mod, "embedder_for_store",
        lambda _store, **_kwargs: _embedder_by_store_id[id(_store)],
    )

    barrier = threading.Barrier(2)

    def _run(store: MemoryStore, cue_vec: list[float], session_id: str) -> dict:
        from iai_mcp import core as _core

        barrier.wait(timeout=10)
        return _core.dispatch(store, "memory_recall", {
            "cue": "concurrent stage profile probe",
            "session_id": session_id,
            "budget_tokens": 2000,
            "cue_embedding": cue_vec,
        })

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_a = ex.submit(_run, store_a, cue_a, "concurrent-a")
        fut_b = ex.submit(_run, store_b, cue_b, "concurrent-b")
        resp_a = fut_a.result(timeout=60)
        resp_b = fut_b.result(timeout=60)

    for resp in (resp_a, resp_b):
        assert "_stage_timings" in resp, resp
    # A coherent per-call invariant (hydrate == ann + getbatch) must hold
    # independently in BOTH responses -- a torn read across the two
    # concurrent calls would break this pairing in at least one of them.
    for resp in (resp_a, resp_b):
        st = resp["_stage_timings"]
        assert "hydrate_ann" in st and "hydrate_getbatch" in st and "hydrate" in st
        assert st["hydrate"] == pytest.approx(st["hydrate_ann"] + st["hydrate_getbatch"])


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_stage_profile_off_reports_neither_bucket_and_results_byte_identical(driver, _store, monkeypatch):
    """With the flag unset, neither the hydrate bucket nor the overlap
    fraction is computed, and recall results are byte-identical to a
    twin store/corpus run with the flag on -- the instrumentation must be
    a pure observer, never a behavior change. Two separately-built,
    deterministically-seeded stores are compared (rather than two sequential
    calls on one store) so write-side effects on the recall path
    (write_event/boost_edges/coactivation) never confound the comparison,
    and `_age_penalty`'s wall-clock `now()` read is frozen so the two calls,
    made microseconds apart, do not diverge on that alone."""
    monkeypatch.setattr(_pipeline_mod, "_age_penalty", lambda _created_at: 0.5)

    cue_vec = _seeded_vec(3)
    store_off = _store(driver, suffix="-off")
    _seed_deterministic_corpus(store_off, cue_vec, base=1000)
    _stub_embedder_for_store(monkeypatch, cue_vec)

    monkeypatch.delenv("IAI_MCP_STAGE_PROFILE", raising=False)
    _pipeline_mod._last_stage_timings_ms.clear()
    resp_off = _dispatch_recall(store_off, cue_vec, session_id="byte-identical-check")
    assert _pipeline_mod._last_stage_timings_ms == {}

    store_on = _store(driver, suffix="-on")
    _seed_deterministic_corpus(store_on, cue_vec, base=1000)
    _stub_embedder_for_store(monkeypatch, cue_vec)

    monkeypatch.setenv("IAI_MCP_STAGE_PROFILE", "1")
    _pipeline_mod._last_stage_timings_ms.clear()
    resp_on = _dispatch_recall(store_on, cue_vec, session_id="byte-identical-check")
    assert "hydrate" in _pipeline_mod._last_stage_timings_ms
    assert "candidate_overlap_fraction" in _pipeline_mod._last_stage_timings_ms

    assert resp_off["hits"] == resp_on["hits"]
    assert resp_off["anti_hits"] == resp_on["anti_hits"]
    assert resp_off["activation_trace"] == resp_on["activation_trace"]
    assert resp_off["budget_used"] == resp_on["budget_used"]
    assert resp_off.get("ann_path_used") == resp_on.get("ann_path_used")
    assert resp_off.get("exact_authority_used") == resp_on.get("exact_authority_used")

    # The response-surfaced stage timings follow the flag exactly: absent
    # off, present (and carrying every sub-stage key) on.
    assert "_stage_timings" not in resp_off
    assert "_stage_timings" in resp_on
    for key in NEW_SUBSTAGE_KEYS:
        assert key in resp_on["_stage_timings"], (key, resp_on["_stage_timings"])

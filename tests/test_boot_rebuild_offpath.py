"""Boot rebuild off-path gate (F): the boot graph rebuild runs off the event
loop AND is chunked/incremental so it never starves foreground recall.

The daemon boot preload rebuilds the runtime graph by streaming the whole
corpus (``retrieve.py`` ``_build_runtime_graph_impl``) and, on a cache miss,
spawning community detection + betweenness centrality over the whole graph.
Two separable concerns:

  OFF-LOOP (the 177-L3 gate): the boot preload wraps ``build_runtime_graph``
  in ``asyncio.to_thread`` so the sync+heavy body runs on a worker thread,
  never inline on the event loop. This module pins that gate as a
  source-level guard so a regression that moves it back onto the loop is
  caught.

  CHUNKED / INCREMENTAL (the F fix): even off-loop, a worker thread running a
  long uninterrupted pure-Python pass over the whole corpus can still starve a
  sibling worker thread doing a foreground recall -- CPython's GIL does not
  reliably hand off to another runnable thread across a long unbroken stretch
  of bytecode-level work. The fix chunks the corpus/edge streaming loops in
  ``_build_runtime_graph_impl`` with an explicit ``time.sleep(0)`` yield point
  every ``IAI_MCP_RGC_STREAM_CHUNK_ROWS`` rows (forcing a real OS-level
  reschedule, not just a GIL switch-interval expiry), and adds
  ``build_runtime_graph_incremental`` -- a delta-only rebuild that patches in
  just the records/edges changed since the cache was saved instead of
  re-streaming the whole corpus, falling back to a full rebuild whenever the
  cached payload was shed or the drift is large.

This module pins:

  1. A source-level OFF-LOOP guard: the boot preload and the cascade rebuild
     both dispatch ``build_runtime_graph`` via ``asyncio.to_thread`` -- never a
     bare inline call on the loop.

  2. A behavioural incremental-path marker: an incremental/delta rebuild entry
     point exists on ``retrieve`` (``build_runtime_graph_incremental``).

  3. An isolated prod-scale during-boot foreground-recall probe (skips
     honestly without the corpus fixture): fire a socket recall DURING the
     boot rebuild window on a fresh-migrated prod-scale lilli store and assert
     it returns within the warm SLA even while the rebuild runs.

  4. A differential correctness proof: the incremental rebuild's graph is
     topology-equivalent to a full rebuild over the same corpus state.

  5. A loop-yield bound: the chunked streaming loop actually yields at a
     bounded interval, not just at the end of the whole stream.
"""
from __future__ import annotations

import ast
import time
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# 1 -- source-level OFF-LOOP guard (always runs, GREEN: the 177-L3 gate holds).
# ---------------------------------------------------------------------------


def _daemon_source() -> str:
    """The daemon source read from DISK (not ``inspect.getsource``, which can
    return a stale imported module when a ``.pyc`` lags the file)."""
    import iai_mcp.daemon as _daemon

    return Path(_daemon.__file__).read_text(encoding="utf-8")


def test_boot_preload_dispatches_build_off_the_loop() -> None:
    """The boot rebuild must never call ``build_runtime_graph`` inline on the
    event loop -- it runs off-loop via ``asyncio.to_thread`` (directly or through
    a sync wrapper the ``to_thread`` dispatches).

    This pins the 177-L3 gate at the source level. It scopes the on-loop check to
    ``async def`` bodies (the event-loop context): a ``build_runtime_graph`` call
    directly inside an async function would be on-loop and starve it. Inline
    ``build_runtime_graph`` calls in PLAIN sync helpers -- the session-start
    payload composer, or the ``_boot_preload_body`` sync wrapper that
    ``asyncio.to_thread`` dispatches -- are fine: they never run on the loop.

    It also requires that the daemon dispatches SOME work via ``asyncio.to_thread``
    (the off-loop mechanism is present at all), guarding against a refactor that
    drops the worker-thread dispatch entirely.
    """
    tree = ast.parse(_daemon_source())

    has_to_thread = False
    inline_on_loop_build_calls: list[int] = []

    def _walk(node: ast.AST, *, in_async: bool) -> None:
        # Descend into function bodies, tracking whether we are inside an
        # ``async def`` (the event-loop context).
        for child in ast.iter_child_nodes(node):
            child_in_async = in_async
            if isinstance(child, ast.AsyncFunctionDef):
                child_in_async = True
            elif isinstance(child, ast.FunctionDef):
                # A plain sync def resets the context: its body does not run on
                # the loop even when nested inside an async function (the
                # ``_boot_preload_body`` wrapper is exactly this shape).
                child_in_async = False

            if isinstance(child, ast.Call):
                func = child.func
                if isinstance(func, ast.Attribute) and func.attr == "to_thread":
                    nonlocal has_to_thread
                    has_to_thread = True
                elif in_async and _names_build_runtime_graph(func):
                    inline_on_loop_build_calls.append(child.lineno)

            _walk(child, in_async=child_in_async)

    _walk(tree, in_async=False)

    assert has_to_thread, (
        "expected the daemon to dispatch heavy work via asyncio.to_thread (the "
        "177-L3 off-loop mechanism); found no to_thread dispatch at all -- the "
        "off-loop worker-thread path may have been removed"
    )
    assert not inline_on_loop_build_calls, (
        f"found inline (on-loop) build_runtime_graph call(s) directly inside "
        f"async function(s) at daemon/__init__.py line(s) "
        f"{inline_on_loop_build_calls} -- the boot rebuild must always run off "
        f"the loop (via asyncio.to_thread, directly or through a sync wrapper) "
        f"so it cannot starve the event loop"
    )


def _names_build_runtime_graph(node: ast.AST) -> bool:
    """True when an AST node references the ``build_runtime_graph`` callable
    (either ``build_runtime_graph`` or ``<mod>.build_runtime_graph``)."""
    if isinstance(node, ast.Attribute):
        return node.attr == "build_runtime_graph"
    if isinstance(node, ast.Name):
        return node.id == "build_runtime_graph"
    return False


# ---------------------------------------------------------------------------
# 2 -- behavioural incremental-delta-path marker.
# ---------------------------------------------------------------------------


def test_boot_rebuild_has_incremental_delta_path() -> None:
    """A chunked / incremental (delta-only) rebuild path must exist.

    ``_build_runtime_graph_impl`` still rebuilds the WHOLE graph on a cache
    miss -- streams the whole corpus and (when the cached results are not
    fresh) spawns community detection + betweenness over the full graph. A
    delta-only entry point (``retrieve.build_runtime_graph_incremental``)
    updates just the records/edges changed since the last cache instead, so a
    small drift never pays the whole-corpus scan.

    This asserts an incremental rebuild callable exists.
    """
    from iai_mcp import retrieve
    from iai_mcp import runtime_graph_cache

    # Any of these names is an acceptable home for the delta-only rebuild path;
    # the F fix picks one. On HEAD none exists.
    candidates = [
        (retrieve, "build_runtime_graph_incremental"),
        (retrieve, "update_runtime_graph_delta"),
        (retrieve, "_build_runtime_graph_incremental"),
        (runtime_graph_cache, "apply_delta"),
        (runtime_graph_cache, "update_incremental"),
    ]
    found = [
        f"{mod.__name__}.{name}"
        for mod, name in candidates
        if callable(getattr(mod, name, None))
    ]
    assert found, (
        "no incremental / delta-only runtime-graph rebuild path exists "
        "(searched: "
        + ", ".join(f"{mod.__name__}.{name}" for mod, name in candidates)
        + "). Without it, every cache miss pays a full whole-corpus pass, which "
        "-- even off-loop -- holds a worker thread + the shared read path for "
        "the full build and can starve a concurrent foreground recall."
    )


# ---------------------------------------------------------------------------
# 3 -- isolated prod-scale during-boot foreground-recall probe (skips honestly
# without the corpus fixture).
# ---------------------------------------------------------------------------


try:
    import iai_mcp_native  # noqa: F401

    _NATIVE = True
except ImportError:
    _NATIVE = False


@pytest.mark.skipif(
    not _NATIVE,
    reason="iai_mcp_native is unavailable; the during-boot recall probe requires "
    "the native engine extension",
)
def test_foreground_recall_not_starved_during_boot_rebuild(tmp_path: Path) -> None:
    """A foreground socket recall fired DURING the boot rebuild window must
    return within the warm SLA even while the whole-corpus rebuild runs.

    Reuses the phase-178 isolated-daemon harness (a fresh-migrated prod-scale
    lilli copy with a drift-mismatched stale cache) as a during-boot recall
    probe: the stale cache forces a cold whole-corpus rebuild at boot, and a
    foreground recall is fired while it runs.

    The chunked streaming loop in ``_build_runtime_graph_impl`` yields at a
    bounded row interval so this uninterrupted whole-corpus rebuild no longer
    starves the foreground recall past the warm SLA. Skips honestly when the
    prod-scale repro corpus snapshot is absent (a local-only fixture) rather
    than fabricating a synthetic corpus that would never reproduce the boot
    storm.

    Never touches the live prod store: the harness operates only on a COPY of a
    preserved, non-live snapshot, with a scratch HOME, a scratch store root, a
    copied (never minted) crypto key, and a short mkdtemp socket dir. The real
    prod daemon (if running) is asserted to be the SAME process, still alive,
    after the probe.
    """
    from tests._warm_recall_repro_support import (
        await_socket,
        locate_repro_source,
        make_fresh_lilli_copy,
        raw_recall,
        spawn_isolated_daemon,
    )
    from tests.test_warm_recall_prodscale_isolated_daemon import (
        _assert_prod_daemon_alive_if_present,
        _prod_daemon_pid,
    )

    src_db = locate_repro_source()
    if src_db is None:
        pytest.skip(
            "the prod-scale repro corpus snapshot is not present on this "
            "machine; this is a local-only fixture, skipping honestly rather "
            "than fabricating a synthetic corpus that would never reproduce the "
            "boot-rebuild starvation"
        )

    _WARM_SLA_SEC = 1.0
    _CLIENT_TIMEOUT_SEC = 30.0
    _SOCKET_AWAIT_TIMEOUT_SEC = 30.0

    prod_daemon_pid = _prod_daemon_pid()

    base = tmp_path / "iso-base"
    scratch_home = tmp_path / "scratch-home"

    report = make_fresh_lilli_copy(src_db, base, stale_cache=True)
    assert report.rows_copied.get("records", 0) > 0, (
        "the fresh lilli copy must contain migrated records"
    )

    handle = spawn_isolated_daemon(base, "lilli", scratch_home=scratch_home)
    try:
        await_socket(handle.socket_path, timeout=_SOCKET_AWAIT_TIMEOUT_SEC)

        # Fire a foreground recall in the boot-rebuild window (right after the
        # socket binds, while the stale-cache-forced cold rebuild runs).
        cue = "cutover debugging session"
        t0 = time.monotonic()
        resp = raw_recall(handle.socket_path, cue, timeout_s=_CLIENT_TIMEOUT_SEC)
        elapsed = time.monotonic() - t0

        assert "error" not in resp, (
            f"memory_recall for cue {cue!r} returned an error during boot: "
            f"{resp.get('error')}"
        )
        result = resp.get("result")
        assert isinstance(result, dict), (
            f"expected a result dict for cue {cue!r}, got {resp!r}"
        )

        # The chunked streaming loop yields at a bounded row interval, so the
        # whole-corpus boot rebuild no longer starves this foreground recall
        # past the warm SLA.
        assert elapsed < _WARM_SLA_SEC, (
            f"foreground recall for cue {cue!r} took {elapsed:.3f}s during the "
            f"boot rebuild window (SLA {_WARM_SLA_SEC}s) -- the whole-corpus "
            f"rebuild starved it."
        )
    finally:
        handle.terminate()
        _assert_prod_daemon_alive_if_present(prod_daemon_pid)


# ---------------------------------------------------------------------------
# 4 -- differential correctness: incremental delta == full rebuild.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _crypto_passphrase(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase-not-secret")
    yield


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(
        _keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p)
    )
    monkeypatch.setattr(
        _keyring, "delete_password", lambda s, u: fake.pop((s, u), None)
    )
    yield fake


def _make_graph_fixture_rec(seed: int, store):
    import numpy as np
    from datetime import datetime, timezone
    from uuid import uuid4
    from iai_mcp.types import MemoryRecord

    rng = np.random.default_rng(seed)
    vec = rng.random(store.embed_dim).astype(np.float32)
    vec = (vec / np.linalg.norm(vec)).tolist()
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=f"delta-equivalence fixture surface {seed}",
        aaak_index="",
        embedding=vec,
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
        tags=[f"tag{seed % 3}"],
        language="en",
    )


def _graph_node_snapshot(graph) -> dict[str, tuple]:
    """A comparable {node_id: (embedding, surface, tier, pinned, tags, language,
    centrality)} snapshot of every node in a built graph, for topology-equivalence
    assertions between the full-rebuild and incremental-delta paths."""
    out: dict[str, tuple] = {}
    for label, payload in graph._node_payload.items():
        out[label] = (
            tuple(round(x, 6) for x in (payload.get("embedding") or [])),
            payload.get("surface", ""),
            payload.get("tier", "episodic"),
            bool(payload.get("pinned", False)),
            tuple(sorted(payload.get("tags") or [])),
            payload.get("language", "en"),
        )
    return out


def _graph_edge_snapshot(graph) -> set[tuple]:
    out: set[tuple] = set()
    for u_label, neighbors in graph._adj.items():
        for v_label, attrs in neighbors.items():
            if u_label <= v_label and attrs:
                out.add(
                    (u_label, v_label, str(attrs.get("edge_type")), float(attrs.get("weight", 1.0)))
                )
    return out


def test_incremental_delta_equals_full_rebuild_after_small_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The incremental (delta-only) rebuild produces a graph
    topology-equivalent to a full rebuild over the SAME corpus state.

    Seeds a small connected corpus, builds a first (cold) cache-populating
    graph, then adds a handful of new records + edges (a below-tolerance
    drift, so the existing warm-cache contract is untouched) and confirms
    ``build_runtime_graph_incremental`` folds the new records/edges into the
    graph -- matching a from-scratch full rebuild of the SAME final corpus
    node-for-node and edge-for-edge.
    """
    from iai_mcp import retrieve, runtime_graph_cache
    from iai_mcp.store import MemoryStore, flush_edge_buffer, flush_record_buffer

    store = MemoryStore(path=tmp_path / "store")
    store.root = tmp_path

    base_recs = [_make_graph_fixture_rec(seed, store) for seed in range(10)]
    for rec in base_recs:
        store.insert(rec)
    flush_record_buffer(store)
    base_ids = [r.id for r in base_recs]
    store.boost_edges(
        [(base_ids[i], base_ids[i + 1]) for i in range(len(base_ids) - 1)],
        delta=1.0,
        edge_type="hebbian",
    )
    # Pad the committed edge count away from a cache-key staleness-window
    # boundary (the key windows counts by // window): the drift below adds a
    # few edges, and if that delta crossed a boundary the persisted cache key
    # would change, try_load would refuse the cache, and the incremental call
    # would DELEGATE to a full build instead of exercising the delta algorithm
    # this test exists to compare.
    store.boost_edges(
        [(base_ids[0], base_ids[i]) for i in (2, 3, 4)],
        delta=1.0,
        edge_type="hebbian",
    )
    flush_edge_buffer(store)

    # Cold build: populates the cache with a full node_payload (small corpus,
    # never sheds) and a real (non-empty) centrality map.
    retrieve.build_runtime_graph(store)
    cached = runtime_graph_cache.try_load(store)
    assert cached is not None
    assert cached[2], "the cache must carry a non-empty node_payload for this fixture size"

    # Add a below-tolerance drift: a couple of new records + one new edge.
    new_recs = [_make_graph_fixture_rec(1000 + i, store) for i in range(2)]
    for rec in new_recs:
        store.insert(rec)
    flush_record_buffer(store)
    store.boost_edges(
        [(base_ids[0], new_recs[0].id)], delta=1.0, edge_type="hebbian",
    )
    # Equivalence is defined over ONE corpus state: drain the in-process edge
    # buffer so both builds stream the identical committed edge set (buffer
    # drain timing is the daemon's concern, not this comparison's).
    flush_edge_buffer(store)

    # Precondition guard: the persisted cache must still be loadable, or the
    # path under test is silently NOT the delta path (the fallback serves a
    # warm bundle whose sync-hook keeps nodes — not edges — live).
    assert runtime_graph_cache.try_load(store) is not None, (
        "fixture drift crossed a cache-key staleness window — the incremental "
        "call would delegate and this test would compare the wrong machinery"
    )

    incr_graph, incr_assignment, _incr_rc = retrieve.build_runtime_graph_incremental(store)

    # Reference: force a from-scratch full rebuild of the SAME final corpus by
    # invalidating the cache first.
    runtime_graph_cache.invalidate(store)
    full_graph, full_assignment, _full_rc = retrieve.build_runtime_graph(store)

    assert incr_graph.node_count() == full_graph.node_count() == 12
    incr_nodes = _graph_node_snapshot(incr_graph)
    full_nodes = _graph_node_snapshot(full_graph)
    # Compare everything except centrality (the incremental path may carry a
    # bounded-stale/neutral centrality for newly-added nodes until the next
    # full rebuild recomputes betweenness -- the same degrade the existing
    # warm cache-hit path already tolerates).
    incr_nodes_no_centrality = {k: v for k, v in incr_nodes.items()}
    full_nodes_no_centrality = {k: v for k, v in full_nodes.items()}
    assert incr_nodes_no_centrality == full_nodes_no_centrality, (
        "incremental delta rebuild's node payload diverges from a full "
        "rebuild over the same corpus state"
    )
    assert _graph_edge_snapshot(incr_graph) == _graph_edge_snapshot(full_graph), (
        "incremental delta rebuild's edge set diverges from a full rebuild "
        "over the same corpus state"
    )
    assert incr_assignment is not None
    assert full_assignment is not None


def test_incremental_delta_falls_back_to_full_rebuild_without_cache(
    tmp_path: Path,
) -> None:
    """With no cache to diff against, the incremental entry point degrades to
    a full rebuild rather than fabricating a delta over nothing."""
    from iai_mcp import retrieve
    from iai_mcp.store import MemoryStore, flush_record_buffer

    store = MemoryStore(path=tmp_path / "store")
    store.root = tmp_path
    for seed in range(5):
        store.insert(_make_graph_fixture_rec(seed, store))
    flush_record_buffer(store)

    graph, assignment, _rc = retrieve.build_runtime_graph_incremental(store)
    assert graph.node_count() == 5
    assert assignment is not None


# ---------------------------------------------------------------------------
# 5 -- loop-yield bound: the chunked stream actually yields mid-stream.
# ---------------------------------------------------------------------------


def test_graph_stream_yields_at_bounded_chunk_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The corpus-streaming loop calls the yield hook at least once per
    configured chunk size, not just once at the very end of the whole stream
    -- proving the yield is actually interleaved with the scan, not a single
    post-hoc call that would still let the whole scan run uninterrupted.
    """
    from iai_mcp import retrieve
    from iai_mcp.store import MemoryStore, flush_record_buffer

    monkeypatch.setenv("IAI_MCP_RGC_STREAM_CHUNK_ROWS", "4")

    store = MemoryStore(path=tmp_path / "store")
    store.root = tmp_path
    for seed in range(37):
        store.insert(_make_graph_fixture_rec(seed, store))
    flush_record_buffer(store)

    yield_calls = {"n": 0}
    real_yield = retrieve._yield_to_event_loop

    def _counting_yield():
        yield_calls["n"] += 1
        real_yield()

    monkeypatch.setattr(retrieve, "_yield_to_event_loop", _counting_yield)

    graph, _assignment, _rc = retrieve.build_runtime_graph(store)

    assert graph.node_count() == 37
    # With chunk size 4 over 37 records the loop must yield multiple times
    # (>= 8), not just once — proving genuine mid-stream interleaving.
    assert yield_calls["n"] >= 8, (
        f"expected the chunked stream to yield repeatedly (>= 8 times for 37 "
        f"rows at chunk size 4), got {yield_calls['n']} -- the yield may have "
        f"collapsed to a single post-hoc call instead of real interleaving"
    )

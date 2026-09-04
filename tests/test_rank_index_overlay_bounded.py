"""Overlay fold trigger: fires off the recall read path, keeps the bounded
delta overlay from growing unbounded over daemon uptime.

`RankIndex.fold()` is the ~174ms wholesale CSR rebuild -- calling it during
a recall would reintroduce exactly the cost the bounded delta overlay
design moved off-path. The trigger (`_RankIndexHandle.maybe_fold()`) fires
only from the store's write-time `graph_sync_hook` (`retrieve.py`), never
from `snapshot()`/`score()`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from test_store import _make  # noqa: E402

from iai_mcp.pipeline import recall_for_response  # noqa: E402
from iai_mcp.retrieve import build_runtime_graph  # noqa: E402
from iai_mcp.store import MemoryStore  # noqa: E402
from iai_mcp.store._rank_index import rank_index_for  # noqa: E402
from iai_mcp.types import EMBED_DIM  # noqa: E402

_FOLD_THRESHOLD = 20


def _unit_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _seed_store(store: MemoryStore, n: int, seed: int) -> list[UUID]:
    rng = np.random.default_rng(seed)
    ids: list[UUID] = []
    for i in range(n):
        v = rng.standard_normal(EMBED_DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        rec = _make(text=f"overlay fixture record {i}", vec=v.tolist())
        store.insert(rec)
        ids.append(rec.id)
    return ids


def test_overlay_bounded_across_post_write_recalls(tmp_path, monkeypatch):
    """`feed()` enqueues a pending op; it is only drained into the overlay
    by the next `snapshot()` -- the real production caller (`core/__init__.py`'s
    MCP dispatch) drains right before scoring on every recall.
    `recall_for_response`/`_recall_core` (called directly here, as the
    differential harness does) does not itself own that drain step, so this
    test drives it explicitly alongside a real recall call, proving recall
    itself never disturbs the overlay while the write-time hook's
    `maybe_fold()` keeps the drained overlay bounded. Across many
    write-then-recall cycles the drained overlay length, sampled right
    after each cycle, never grows past the threshold plus the small margin
    one un-folded write-and-drain cycle can add (the trigger fires on the
    WRITE after crossing, one cycle behind)."""
    monkeypatch.setenv("IAI_MCP_RANK_OVERLAY_FOLD_THRESHOLD", str(_FOLD_THRESHOLD))

    store = MemoryStore(path=str(tmp_path / "store"))
    _seed_store(store, n=10, seed=1)
    graph, assignment, rich_club = build_runtime_graph(store)

    handle = rank_index_for(store, graph)
    # Force the first build (mirrors a real recall's snapshot() call) before
    # any writes -- feed()/maybe_fold() are no-ops before the index exists.
    handle.snapshot(graph)

    cue_vec = _unit_vec(2)
    overlay_samples: list[int] = []
    n_cycles = _FOLD_THRESHOLD * 3
    for i in range(n_cycles):
        rec = _make(text=f"post-build write {i}", vec=_unit_vec(100 + i))
        store.insert(rec)
        handle.snapshot(graph)
        recall_for_response(
            store=store, graph=graph, assignment=assignment, rich_club=rich_club,
            embedder=None, cue="overlay bound probe", session_id="overlay-bounded",
            budget_tokens=1500, cue_embedding=list(cue_vec),
        )
        overlay_samples.append(handle.overlay_len())

    margin = 2  # one write-and-drain cycle's worth of lag on the trigger
    assert max(overlay_samples) <= _FOLD_THRESHOLD + margin, (
        f"overlay grew past the fold threshold ({_FOLD_THRESHOLD}) + margin "
        f"({margin}) across {n_cycles} write-then-recall cycles: "
        f"samples={overlay_samples}"
    )
    # Non-vacuity: the overlay must have actually carried entries at some
    # point -- otherwise "stays bounded" would be trivially true because
    # nothing was ever fed.
    assert max(overlay_samples) > 0, (
        "overlay_len() was always 0 -- this run never exercised the write "
        "path this test is supposed to bound"
    )


def test_fold_never_runs_on_recall_read_path(tmp_path, monkeypatch):
    """Spy on the handle's fold() across a run that mixes writes (which may
    legitimately fold) and recalls (which never may) -- then isolate a
    recall-only window and assert zero fold() calls landed in it, including
    through the hydrate-stage feed() calls a real recall can drive."""
    monkeypatch.setenv("IAI_MCP_RANK_OVERLAY_FOLD_THRESHOLD", str(_FOLD_THRESHOLD))

    store = MemoryStore(path=str(tmp_path / "store"))
    _seed_store(store, n=10, seed=3)
    graph, assignment, rich_club = build_runtime_graph(store)

    handle = rank_index_for(store, graph)
    handle.snapshot(graph)

    original_fold = handle.fold
    fold_calls: list[int] = []

    def _counting_fold():
        fold_calls.append(1)
        return original_fold()

    monkeypatch.setattr(handle, "fold", _counting_fold)

    # Drive enough write-then-drain cycles to guarantee at least one fold
    # happened on the write path -- proves the spy is wired to a fold that
    # genuinely fires, not a dead spy that would pass vacuously. The drain
    # (snapshot()) is called directly here, mirroring what a real recall's
    # read path does, without paying a full recall_for_response per cycle.
    for i in range(_FOLD_THRESHOLD * 3):
        store.insert(_make(text=f"pre-recall write {i}", vec=_unit_vec(200 + i)))
        handle.snapshot(graph)

    assert fold_calls, (
        "the write-path fold trigger never fired across the pre-recall "
        "writes -- this test cannot isolate a meaningful recall-only window"
    )
    fold_calls.clear()

    cue_vec = _unit_vec(4)
    for _ in range(10):
        # snapshot() is the real recall read path's own drain call (the
        # production caller in core/__init__.py runs it right before
        # scoring, on every recall) -- included here, with NO write in
        # between, so this window exercises the actual drain mechanism,
        # not just recall_for_response's own code path (which never calls
        # snapshot() itself when invoked directly, as this harness does).
        handle.snapshot(graph)
        recall_for_response(
            store=store, graph=graph, assignment=assignment, rich_club=rich_club,
            embedder=None, cue="fold-off-recall-path probe", session_id="fold-guard",
            budget_tokens=1500, cue_embedding=list(cue_vec),
        )

    assert fold_calls == [], (
        f"fold() was called {len(fold_calls)} time(s) during recall-only "
        "calls (including their own snapshot() drain) -- fold() must "
        "never run on the recall read path"
    )

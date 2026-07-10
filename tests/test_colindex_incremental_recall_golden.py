"""Golden cross-driver recall parity around an inter-recall write.

The records col-index is maintained incrementally on write rather than dropped
and lazily full-scan-rebuilt. This pins the user-visible contract that the
maintenance change keeps recall output byte-identical:

- a write coalesced between two recalls (the provenance/event shape) must not
  change the second recall's top-k, and
- the top-k record-id + score sequence must be identical whether the store runs
  on the stdlib sqlite3 driver or the lilli engine.

The harness drives the SAME deterministic corpus + cue through both drivers and
asserts the (record_id, rounded-score) sequences match exactly, before and after
the inter-recall write.
"""
from __future__ import annotations

import importlib
import os
from uuid import UUID

import pytest

from _recall_helpers import _deterministic_vec, _populate_store


def _topk_signature(store, cue_vec):
    """The recall top-k as a byte-comparable signature.

    The deterministic gold records use fixed ``UUID(int=N)`` ids; the 300 filler
    records carry random ids (the test corpus helper does not seed them), so the
    cross-driver contract is the engine-determined ordering — the score sequence
    plus the positions of the deterministic gold ids — NOT the random filler ids,
    which differ run-to-run on either driver. The signature pairs each hit's
    rounded score with its id ONLY when the id is a deterministic gold id (so a
    reordering of gold records or a score drift is caught), and a sentinel for a
    filler hit (whose random id is not a stable comparison target)."""
    from uuid import UUID
    from iai_mcp.retrieve import recall

    gold_ids = {str(UUID(int=i)) for i in (1, 2, 3, 4, 5)}
    resp = recall(
        store,
        cue_embedding=cue_vec,
        cue_text="reference gold",
        session_id="golden-sess",
        k_hits=5,
        k_anti=3,
    )
    sig = []
    for h in resp.hits:
        rid = str(h.record_id)
        marker = rid if rid in gold_ids else "<filler>"
        sig.append((marker, round(h.score, 6)))
    return sig


def _run_one_driver(driver: str, tmp_path):
    """Build the corpus, recall, write between recalls, recall again — return
    (before, after) top-k signatures for the given storage driver."""
    os.environ["LILLI_STORAGE_DRIVER"] = driver
    # Re-import the store module under the selected driver so the routing flag is
    # read fresh (the driver is resolved at store construction).
    import iai_mcp.store as _store_mod

    importlib.reload(_store_mod)
    from iai_mcp.store import MemoryStore

    path = tmp_path / f"golden-{driver}"
    path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(path=path)
    cue_vec = _deterministic_vec()
    _populate_store(store, cue_vec)

    before = _topk_signature(store, cue_vec)

    # A coalesced provenance-style write between recalls: append provenance to the
    # top hit (the records UPSERT that DROPPED the col-index under the old
    # drop-and-rebuild discipline). It must not alter the next recall's top-k.
    from datetime import datetime, timezone

    store.append_provenance_batch(
        [(UUID(int=1), {"ts": datetime.now(timezone.utc).isoformat(), "cue": "x"})]
    )

    after = _topk_signature(store, cue_vec)
    store.close()
    return before, after


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_inter_recall_write_does_not_change_topk(driver, tmp_path):
    """Within one driver, a write between recalls leaves the top-k unchanged —
    the incremental col-index reflects the write without a rebuild that could
    reorder results."""
    before, after = _run_one_driver(driver, tmp_path)
    assert before, "the gold corpus must yield a non-empty top-k"
    assert before == after, (
        f"driver={driver}: a write between recalls changed the top-k\n"
        f"before={before}\nafter ={after}"
    )


def test_topk_byte_identical_across_drivers(tmp_path):
    """The recall top-k must be byte-identical on the stdlib sqlite3 driver and
    the lilli engine, both before and after the inter-recall write."""
    std_before, std_after = _run_one_driver("stdlib", tmp_path)
    lil_before, lil_after = _run_one_driver("lilli", tmp_path)
    assert std_before == lil_before, (
        f"pre-write top-k diverged across drivers\n"
        f"stdlib={std_before}\nlilli ={lil_before}"
    )
    assert std_after == lil_after, (
        f"post-write top-k diverged across drivers\n"
        f"stdlib={std_after}\nlilli ={lil_after}"
    )

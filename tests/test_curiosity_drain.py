from __future__ import annotations

from uuid import uuid4

from iai_mcp.core import dispatch
from iai_mcp.events import write_event
from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline
from iai_mcp.store import MemoryStore


def _write_deferred(store, *, session_id, turn, n_hits=5, with_scores=True):
    hit_ids = [str(uuid4()) for _ in range(n_hits)]
    data = {
        "hit_ids": hit_ids,
        "cue": f"ambiguous cue {session_id}",
        "session_id": session_id,
        "turn": turn,
    }
    if with_scores:
        # Uniform scores across many hits -> high Shannon entropy -> fires.
        data["scores"] = [1.0] * n_hits
    write_event(
        store,
        kind="deferred_curiosity_input",
        data=data,
        severity="info",
        session_id=session_id,
    )


def test_drain_fires_questions_and_populates_pending(tmp_path):
    store = MemoryStore(path=tmp_path)

    # Two producers in distinct sessions (avoids the per-session cooldown).
    _write_deferred(store, session_id="sA", turn=1)
    _write_deferred(store, session_id="sB", turn=1)

    # Baseline: the queue is empty until the drain runs.
    assert dispatch(store, "curiosity_pending", {}) == {"questions": [], "count": 0}

    pipe = SleepPipeline(store)
    done, payload = pipe._step_curiosity_drain(None)

    assert done is True
    assert payload["deferred_inputs"] == 2
    assert payload["inputs_processed"] == 2
    assert payload["questions_fired"] == 2

    out = dispatch(store, "curiosity_pending", {})
    assert out["count"] == 2


def test_drain_is_idempotent_via_watermark(tmp_path):
    store = MemoryStore(path=tmp_path)
    _write_deferred(store, session_id="sA", turn=1)

    pipe = SleepPipeline(store)
    _, p1 = pipe._step_curiosity_drain(None)
    assert p1["questions_fired"] == 1

    # Second run with no new inputs must fire nothing and report zero inputs.
    _, p2 = pipe._step_curiosity_drain(None)
    assert p2["deferred_inputs"] == 0
    assert p2["questions_fired"] == 0

    # Total questions unchanged.
    assert dispatch(store, "curiosity_pending", {})["count"] == 1


def test_drain_processes_only_new_inputs_after_watermark(tmp_path):
    store = MemoryStore(path=tmp_path)
    _write_deferred(store, session_id="sA", turn=1)

    pipe = SleepPipeline(store)
    _, p1 = pipe._step_curiosity_drain(None)
    assert p1["questions_fired"] == 1

    # A fresh input written after the first drain must be picked up next run.
    _write_deferred(store, session_id="sB", turn=1)
    _, p2 = pipe._step_curiosity_drain(None)
    assert p2["deferred_inputs"] == 1
    assert p2["questions_fired"] == 1
    assert dispatch(store, "curiosity_pending", {})["count"] == 2


def test_drain_skips_inputs_without_scores(tmp_path):
    store = MemoryStore(path=tmp_path)
    # Pre-feature event: no scores -> must be skipped (not fired spuriously),
    # but should still advance the watermark so it is never revisited.
    _write_deferred(store, session_id="sOld", turn=1, with_scores=False)

    pipe = SleepPipeline(store)
    _, payload = pipe._step_curiosity_drain(None)

    assert payload["deferred_inputs"] == 1
    assert payload["inputs_processed"] == 0
    assert payload["questions_fired"] == 0
    assert dispatch(store, "curiosity_pending", {})["count"] == 0

    # Watermark advanced -> a second run sees no eligible inputs.
    _, p2 = pipe._step_curiosity_drain(None)
    assert p2["deferred_inputs"] == 0


def test_drain_empty_store_noop(tmp_path):
    store = MemoryStore(path=tmp_path)
    pipe = SleepPipeline(store)
    done, payload = pipe._step_curiosity_drain(None)
    assert done is True
    assert payload == {"deferred_inputs": 0, "questions_fired": 0}

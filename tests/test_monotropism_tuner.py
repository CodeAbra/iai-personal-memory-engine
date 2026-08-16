"""B1 recall-signal plumbing (pre-rank community-gate top-1 + K + backend
threaded out of ``_recall_core`` to the ``retrieval_used`` emit) and the
nightly step's gate-id -> name remap seam + all-rows ``profile_tuned``
redaction.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp import core
from iai_mcp.events import flush_event_buffer, query_events, write_event
from iai_mcp.graph import MemoryGraph
from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline
from iai_mcp.lilli.cycle.sleep_pipeline._knob_tune_specs import K_MIN, MAX_AUTO_DEPTH
from iai_mcp.lilli.profile.community_names import load_community_names, save_community_names
from iai_mcp.lilli.profile.knobs import default_state
from iai_mcp.lilli.profile.persistence import save_profile_state
from iai_mcp.pipeline import _RecallCoreResult, _apply_post_rank_pipeline
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryHit

from test_store import _make


@pytest.fixture(autouse=True)
def _restore_live_profile():
    saved_state = dict(core._profile_state)
    saved_posterior = dict(core._posterior_state)
    saved_hydrated = set(core._profile_hydrated_stores)
    core._profile_hydrated_stores.clear()
    yield
    core._profile_state.clear()
    core._profile_state.update(saved_state)
    core._posterior_state.clear()
    core._posterior_state.update(saved_posterior)
    core._profile_hydrated_stores.clear()
    core._profile_hydrated_stores.update(saved_hydrated)


# ---------------------------------------------------------------------------
# _RecallCoreResult defaults (l0 fast-path never threads a gate)
# ---------------------------------------------------------------------------


def test_recall_core_result_gate_fields_default_null() -> None:
    result = _RecallCoreResult()
    assert result.cue_community_id is None
    assert result.community_k is None
    assert result.community_backend is None


# ---------------------------------------------------------------------------
# _recall_core threads the pre-rank gate top-1/K/backend to the result
# ---------------------------------------------------------------------------


def test_recall_core_threads_gate_top1_k_backend(tmp_path, monkeypatch) -> None:
    import iai_mcp.pipeline as _pipeline_mod
    from iai_mcp.embed import Embedder
    from iai_mcp.retrieve import build_runtime_graph

    store = MemoryStore(path=tmp_path / "store")
    rng = np.random.default_rng(7)
    for i in range(8):
        v = rng.standard_normal(EMBED_DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        store.insert(_make(text=f"filler record {i}", vec=v.tolist()))

    graph, assignment, rich_club = build_runtime_graph(store)

    fake_top1 = uuid4()
    # community_scores (the gate's per-query candidate-pool scoring) and
    # mid_regions (the corpus-stable community grouping) are deliberately
    # different sizes here -- community_k must come from mid_regions, never
    # from the pool-restricted scores dict.
    fake_scores = {fake_top1: 0.9, uuid4(): 0.5, uuid4(): 0.4, uuid4(): 0.1}
    monkeypatch.setattr(
        _pipeline_mod, "_community_gate_scored",
        lambda *a, **k: ([fake_top1], fake_scores),
    )
    assignment.backend = "leiden-custom"
    assignment.mid_regions = {uuid4(): [uuid4()] for _ in range(6)}

    embedder = Embedder()
    result = _pipeline_mod._recall_core(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue="probe cue text for the gate", session_id="test",
    )

    assert result.cue_community_id == str(fake_top1)
    assert result.community_k == 6
    assert result.community_k != len(fake_scores)
    assert result.community_backend == "leiden-custom"
    store.close()


# ---------------------------------------------------------------------------
# _apply_post_rank_pipeline: threading + null-gating at the emit
# ---------------------------------------------------------------------------


def _seed_hit(store: MemoryStore) -> MemoryHit:
    rid = uuid4()
    return MemoryHit(
        record_id=rid,
        score=0.9,
        reason="test",
        literal_surface="alice asked about the schema migration",
        adjacent_suggestions=[],
        session_id="alice-session-1",
    )


def _run_pipeline_for_gate(
    store: MemoryStore,
    *,
    cue_community_id,
    community_k,
    community_backend,
) -> dict:
    hit = _seed_hit(store)
    _apply_post_rank_pipeline(
        [hit],
        store=store,
        graph=MemoryGraph(),
        records_cache={},
        cue="what did we decide about the schema",
        session_id="alice-session-1",
        profile_state={},
        turn=1,
        mode="verbatim",
        budget_used=10,
        path_label="test_monotropism_tuner",
        cue_community_id=cue_community_id,
        community_k=community_k,
        community_backend=community_backend,
    )
    flush_event_buffer(store)
    events = query_events(store, kind="retrieval_used", limit=10)
    assert events, "expected a buffered retrieval_used event"
    return events[0]["data"]


def test_emit_carries_gate_id_and_k_when_backend_not_flat_and_k_at_floor(tmp_path) -> None:
    store = MemoryStore(path=tmp_path / "store")
    cid = str(uuid4())
    data = _run_pipeline_for_gate(
        store, cue_community_id=cid, community_k=K_MIN, community_backend="leiden",
    )
    assert data["cue_community_id"] == cid
    assert data["community_k"] == K_MIN


def test_emit_nulls_gate_id_on_flat_backend(tmp_path) -> None:
    store = MemoryStore(path=tmp_path / "store")
    cid = str(uuid4())
    data = _run_pipeline_for_gate(
        store, cue_community_id=cid, community_k=K_MIN + 5, community_backend="flat",
    )
    assert data["cue_community_id"] is None
    assert data["community_k"] is None


def test_emit_nulls_gate_id_below_k_floor(tmp_path) -> None:
    store = MemoryStore(path=tmp_path / "store")
    cid = str(uuid4())
    data = _run_pipeline_for_gate(
        store, cue_community_id=cid, community_k=K_MIN - 1, community_backend="leiden",
    )
    assert data["cue_community_id"] is None
    assert data["community_k"] is None


def test_emit_defaults_null_when_no_gate_args_passed(tmp_path) -> None:
    """Mirrors the pre-existing task_support_probe call, which omits the
    three new kwargs entirely -- the keyword-only defaults must not raise."""
    store = MemoryStore(path=tmp_path / "store")
    hit = _seed_hit(store)
    _apply_post_rank_pipeline(
        [hit],
        store=store,
        graph=MemoryGraph(),
        records_cache={},
        cue="what did we decide about the schema",
        session_id="alice-session-1",
        profile_state={},
        turn=1,
        mode="verbatim",
        budget_used=10,
        path_label="test_monotropism_tuner",
    )
    flush_event_buffer(store)
    data = query_events(store, kind="retrieval_used", limit=10)[0]["data"]
    assert data["cue_community_id"] is None
    assert data["community_k"] is None


# ---------------------------------------------------------------------------
# Nightly step: gate-id -> name remap seam
# ---------------------------------------------------------------------------


def _seed_retrieval_used_gate_rows(
    store: MemoryStore, *, gate_cid: str, k: int, count: int, now: datetime,
) -> None:
    for i in range(count):
        write_event(
            store,
            kind="retrieval_used",
            data={
                "hit_ids": [], "query": "q", "used": True,
                "cue_community_id": gate_cid, "community_k": k,
                "timestamp": (now - timedelta(minutes=i)).isoformat(),
            },
            severity="info",
        )


def test_dominant_community_moves_a_name_keyed_depth(tmp_path, monkeypatch) -> None:
    store = MemoryStore(path=tmp_path / "store")
    now = datetime.now(timezone.utc)
    monkeypatch.setattr("iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: now)

    gate_cid = str(uuid4())
    save_community_names(store, reverse_index={gate_cid: "music"}, provenance={})

    _seed_retrieval_used_gate_rows(
        store, gate_cid=gate_cid, k=K_MIN, count=20, now=now,
    )

    core._profile_state.clear()
    core._profile_state.update(default_state())

    pipe = SleepPipeline(store, lifecycle_state_path=tmp_path / "lifecycle.json")
    done, _payload = pipe._step_knob_tune(None)
    assert done is True

    events = query_events(store, kind="profile_tuned", limit=10)
    row = next(r for r in events[0]["data"]["knobs"] if r["knob"] == "monotropism_depth")
    assert row["reason"] == "moved"
    assert core._profile_state["monotropism_depth"] == {"music": pytest.approx(0.15)}


def test_first_night_empty_map_is_skipped_no_signal(tmp_path, monkeypatch) -> None:
    store = MemoryStore(path=tmp_path / "store")
    now = datetime.now(timezone.utc)
    monkeypatch.setattr("iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: now)

    gate_cid = str(uuid4())
    # No community_names saved at all -- the honest first-night state.
    assert load_community_names(store) == {}
    _seed_retrieval_used_gate_rows(
        store, gate_cid=gate_cid, k=K_MIN, count=20, now=now,
    )

    core._profile_state.clear()
    core._profile_state.update(default_state())

    pipe = SleepPipeline(store, lifecycle_state_path=tmp_path / "lifecycle.json")
    done, _payload = pipe._step_knob_tune(None)
    assert done is True

    events = query_events(store, kind="profile_tuned", limit=10)
    row = next(r for r in events[0]["data"]["knobs"] if r["knob"] == "monotropism_depth")
    assert row["reason"] == "skipped_no_signal"
    assert core._profile_state["monotropism_depth"] == {}


def test_gate_id_absent_from_map_contributes_nothing(tmp_path, monkeypatch) -> None:
    store = MemoryStore(path=tmp_path / "store")
    now = datetime.now(timezone.utc)
    monkeypatch.setattr("iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: now)

    gate_cid = str(uuid4())
    other_cid = str(uuid4())
    save_community_names(store, reverse_index={other_cid: "film"}, provenance={})
    _seed_retrieval_used_gate_rows(
        store, gate_cid=gate_cid, k=K_MIN, count=20, now=now,
    )

    core._profile_state.clear()
    core._profile_state.update(default_state())

    pipe = SleepPipeline(store, lifecycle_state_path=tmp_path / "lifecycle.json")
    done, _payload = pipe._step_knob_tune(None)
    assert done is True

    events = query_events(store, kind="profile_tuned", limit=10)
    row = next(r for r in events[0]["data"]["knobs"] if r["knob"] == "monotropism_depth")
    assert row["reason"] == "skipped_no_signal"


# ---------------------------------------------------------------------------
# B6: single all-rows redaction, including a populated dict on a no-signal row
# ---------------------------------------------------------------------------


def test_populated_dict_redacted_on_skipped_no_signal_row(tmp_path, monkeypatch) -> None:
    store = MemoryStore(path=tmp_path / "store")
    now = datetime.now(timezone.utc)
    monkeypatch.setattr("iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: now)

    # No retrieval_used events at all this window -- guaranteed no signal --
    # but the knob already carries a populated, previously-tuned dict.
    core._profile_state.clear()
    core._profile_state.update(default_state())
    core._profile_state["monotropism_depth"] = {"music": 0.4}

    pipe = SleepPipeline(store, lifecycle_state_path=tmp_path / "lifecycle.json")
    done, _payload = pipe._step_knob_tune(None)
    assert done is True

    events = query_events(store, kind="profile_tuned", limit=10)
    data = events[0]["data"]
    row = next(r for r in data["knobs"] if r["knob"] == "monotropism_depth")
    assert row["reason"] == "skipped_no_signal"
    assert row["from"] == {"keys": 1}
    assert row["to"] == {"keys": 1}
    # No topic name string anywhere in the whole written event.
    assert "music" not in json.dumps(data)


def test_moved_row_dict_redacted_to_key_count(tmp_path, monkeypatch) -> None:
    store = MemoryStore(path=tmp_path / "store")
    now = datetime.now(timezone.utc)
    monkeypatch.setattr("iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: now)

    gate_cid = str(uuid4())
    save_community_names(store, reverse_index={gate_cid: "music"}, provenance={})
    _seed_retrieval_used_gate_rows(
        store, gate_cid=gate_cid, k=K_MIN, count=20, now=now,
    )

    core._profile_state.clear()
    core._profile_state.update(default_state())

    pipe = SleepPipeline(store, lifecycle_state_path=tmp_path / "lifecycle.json")
    pipe._step_knob_tune(None)

    events = query_events(store, kind="profile_tuned", limit=10)
    data = events[0]["data"]
    row = next(r for r in data["knobs"] if r["knob"] == "monotropism_depth")
    assert row["reason"] == "moved"
    assert row["from"] == {"keys": 0}
    assert row["to"] == {"keys": 1}
    assert "music" not in json.dumps(data)


def test_user_pin_freezes_the_whole_dict_even_above_the_auto_cap(tmp_path, monkeypatch) -> None:
    """A knob-level pin (must_haves truth #6) is a promise about the
    DURABLE value -- the generic pin branch short-circuits before the spec
    ever runs, so a manual >0.7 set survives untouched, uncapped, undecayed,
    and its from/to are still redacted (site #1, the only append fed by the
    durable blob rather than work_state)."""
    store = MemoryStore(path=tmp_path / "store")
    now = datetime.now(timezone.utc)
    monkeypatch.setattr("iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: now)

    pinned_value = {"jazz": 0.9}
    save_profile_state(
        store, knobs={"monotropism_depth": pinned_value}, posterior={},
        pins={"monotropism_depth": now.isoformat()},
    )

    gate_cid = str(uuid4())
    save_community_names(store, reverse_index={gate_cid: "jazz"}, provenance={})
    _seed_retrieval_used_gate_rows(
        store, gate_cid=gate_cid, k=K_MIN, count=20, now=now,
    )

    core._profile_state.clear()
    core._profile_state.update(default_state())
    core._profile_state["monotropism_depth"] = dict(pinned_value)

    pipe = SleepPipeline(store, lifecycle_state_path=tmp_path / "lifecycle.json")
    done, _payload = pipe._step_knob_tune(None)
    assert done is True

    assert core._profile_state["monotropism_depth"] == pinned_value

    events = query_events(store, kind="profile_tuned", limit=10)
    data = events[0]["data"]
    row = next(r for r in data["knobs"] if r["knob"] == "monotropism_depth")
    assert row["reason"] == "skipped_pinned_by_user"
    assert row["from"] == {"keys": 1}
    assert row["to"] == {"keys": 1}
    assert "jazz" not in json.dumps(data)

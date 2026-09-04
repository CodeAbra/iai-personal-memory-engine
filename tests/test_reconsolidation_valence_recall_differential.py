"""Label-free recall drop-out differential gate for the reconsolidation
valence write.

Modeled on the token-economy-deprefix comparator (`dropped = (p1 & p2) - b`,
`tests/test_token_economy_deprefix_recall_differential.py`) and the CI-safe
synthetic cue corpus (`tests/_synthetic_cue_corpus.py`) -- a wholly
synthetic, non-owner corpus, never the real store.

The A/A floor below dispatches two probes against ONE shared, unrebuilt
graph object: with no rebuild between them, exact top-k equality bounds
DISPATCH determinism on a fixed graph -- it does not by itself license
attributing every B-leg drop-out to the valence write, since the B leg
rebuilds its own graph and inherits that rebuild's own jitter. The A/B
no-drop assertion below is what carries the safety claim: an id A/A-stable
in the OFF leg must not be absent from the ON leg (a gain is benign; only a
drop is a regression).

Diagnostics on failure carry record ids, cue indices, and counts only, never
cue text or stored content.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_recall_stage_profile import _monkeypatch_env  # noqa: E402
from test_recall_scoring_differential import _freeze_age_penalty  # noqa: E402

from tests._synthetic_cue_corpus import (  # noqa: E402
    build_live_path_shaped_corpus,
    build_production_shaped_cue_set,
    flatten_cues,
    insert_live_path_corpus,
)

from iai_mcp import runtime_graph_cache  # noqa: E402
from iai_mcp.embed import Embedder  # noqa: E402
from iai_mcp.lifecycle_state import default_state, save_state  # noqa: E402
from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline  # noqa: E402
from iai_mcp.pipeline import recall_for_response  # noqa: E402
from iai_mcp.retrieve import build_runtime_graph  # noqa: E402
from iai_mcp.store import MemoryStore  # noqa: E402

_SEED = 0
_TOP_K = 10
# Wider window used only to pick a topic-local planted-labile target: a
# candidate absent from every other cue's full ranked response (not merely
# its top-k) is far enough from that cue's k-boundary that a bounded
# STABILITY_BOOST_ON_RECALL delta cannot plausibly push it across it. Sized
# against the live-path-shaped corpus (800 records, 40 activity clusters of
# 20 near-duplicate paraphrases each) -- a 64-record corpus returns most of
# itself per cue and admits no uncontested candidate at all.
_CONTESTED_WINDOW = 40
_RECORDS_PER_TOPIC = 20


# ---------------------------------------------------------------------------
# Pure comparator (store-independent) -- copies the drop-out shape from
# test_token_economy_deprefix_recall_differential.py's `_comparator`.
# ---------------------------------------------------------------------------


def _comparator(
    per_cue: "dict[object, dict[str, set[UUID]]]",
) -> "dict[object, set[UUID]]":
    """An id is a regression for a cue iff it is A/A-stable (present in both
    p1 and p2) AND absent from b. An id present in p1 but not p2 is excused
    as dispatch noise; a reshuffle within the retrieved set is never a
    regression -- only drop-out is. Returns cue-key -> dropped id set,
    omitting cues with no drop-out.
    """
    dropped_by_cue: "dict[object, set[UUID]]" = {}
    for cue_key, sets in per_cue.items():
        stable = sets["p1"] & sets["p2"]
        dropped = stable - sets["b"]
        if dropped:
            dropped_by_cue[cue_key] = dropped
    return dropped_by_cue


def test_comparator_stable_drop_is_flagged():
    per_cue = {"cue-a": {"p1": {"s1", "n1"}, "p2": {"s1", "n1"}, "b": {"n1"}}}
    assert _comparator(per_cue) == {"cue-a": {"s1"}}


def test_comparator_unstable_drop_is_excused():
    per_cue = {"cue-a": {"p1": {"s1"}, "p2": set(), "b": set()}}
    assert _comparator(per_cue) == {}


def test_comparator_within_set_reshuffle_is_benign():
    per_cue = {"cue-a": {"p1": {"s1", "n1"}, "p2": {"s1", "n1"}, "b": {"n1", "s1"}}}
    assert _comparator(per_cue) == {}


def test_comparator_gain_is_benign():
    per_cue = {"cue-a": {"p1": {"s1"}, "p2": {"s1"}, "b": {"s1", "n1"}}}
    assert _comparator(per_cue) == {}


# ---------------------------------------------------------------------------
# Shared harness helpers
# ---------------------------------------------------------------------------


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
            pytest.skip("iai_mcp_native not built -- lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _build_gate_store(driver: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Fresh live-path-shaped synthetic corpus (800 records, 40 activity
    clusters), valence never written. This scale is required (a smaller
    64-record corpus returns most of itself per cue, leaving no
    genuinely topic-local candidate to plant valence on). Zero
    read-consistency is required so the B leg's sleep step observes the
    just-written `labile_until` value within the same process."""
    _select_driver(driver, monkeypatch)
    _monkeypatch_env(monkeypatch, tmp_path)
    store_root = tmp_path / f"valence-differential-{driver}"
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    monkeypatch.setenv("IAI_MCP_VALENCE_WRITE_OFF", "1")

    embedder = Embedder()
    records, edges = build_live_path_shaped_corpus(seed=_SEED, embedder=embedder)
    cues = flatten_cues(build_production_shaped_cue_set())
    store = MemoryStore(path=store_root, read_consistency_interval=timedelta(seconds=0))
    insert_live_path_corpus(store, records, edges)
    return store, records, cues, embedder


def _dispatch_hits(store, graph, assignment, rich_club, embedder, cue, session_id: str):
    """Full ranked response, unsliced -- callers truncate to `_TOP_K` for the
    drop-out comparator or `_CONTESTED_WINDOW` for target selection."""
    response = recall_for_response(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=cue.text, session_id=session_id,
        budget_tokens=1500, mode=cue.mode,
    )
    return response.hits


def _hit_for(hits, record_id: UUID):
    for h in hits:
        if h.record_id == record_id:
            return h
    return None


def _mark_labile(store: MemoryStore, record_id: UUID) -> None:
    tbl = store.db.open_table("records")
    tbl.update(
        where=f"id = '{record_id}'",
        values={"labile_until": datetime.now(timezone.utc) + timedelta(seconds=300)},
    )


# ---------------------------------------------------------------------------
# Recall drop-out differential gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_recall_drop_out_differential_floor_and_non_vacuity(tmp_path, monkeypatch, driver):
    """A/A exact floor on one shared OFF graph, A/B no-drop across every
    cue, and non-vacuity (planted valence moves rank and fires "xval")."""
    store, records, cues, embedder = _build_gate_store(driver, tmp_path, monkeypatch)
    _freeze_age_penalty(monkeypatch)

    runtime_graph_cache.invalidate(store)
    graph, assignment, rich_club = build_runtime_graph(store)
    assert getattr(store, "_warm_graph_bundle", None) is not None, (
        "graph build did not memoize a warm bundle -- the shared-graph A/A "
        "floor below would silently rebuild instead of reusing one object"
    )

    specific_indices = [i for i, c in enumerate(cues) if c.band == "specific"]
    topic_records = [
        records[i * _RECORDS_PER_TOPIC:(i + 1) * _RECORDS_PER_TOPIC]
        for i in range(len(specific_indices))
    ]

    p1_hits = {
        i: _dispatch_hits(store, graph, assignment, rich_club, embedder, cue, "gate-off")
        for i, cue in enumerate(cues)
    }
    p1_contested_sets = {
        i: {h.record_id for h in p1_hits[i][:_CONTESTED_WINDOW]} for i in range(len(cues))
    }

    # Locate a record ranked 2nd or 3rd in its own topic cue's top-k (solidly
    # inside, not on the k-boundary) and absent from every OTHER cue's wider
    # ranked window (topic-local, not a globally-borderline record whose
    # boost would evict unrelated cues' tail candidates) -- the
    # planted-labile target below.
    target_id: "UUID | None" = None
    target_cue_index: "int | None" = None
    for cue_index, group in zip(specific_indices, topic_records):
        if cues[cue_index].mode != "concept":
            continue
        group_ids = {r.id for r in group}
        for rank, hit in enumerate(p1_hits[cue_index][:_TOP_K]):
            if not (0 < rank <= 2 and hit.record_id in group_ids):
                continue
            contested = any(
                hit.record_id in p1_contested_sets[j]
                for j in range(len(cues))
                if j != cue_index
            )
            if not contested:
                target_id = hit.record_id
                target_cue_index = cue_index
                break
        if target_id is not None:
            break
    assert target_id is not None, (
        "fixture precondition: no topic-local corpus record found ranked "
        "2nd or 3rd, and absent from every other cue's wider ranked window "
        "-- cannot select a non-vacuous planted-labile target"
    )

    p2_hits = {
        i: _dispatch_hits(store, graph, assignment, rich_club, embedder, cue, "gate-off")
        for i, cue in enumerate(cues)
    }

    for i in range(len(cues)):
        p1_ids = [h.record_id for h in p1_hits[i][:_TOP_K]]
        p2_ids = [h.record_id for h in p2_hits[i][:_TOP_K]]
        assert p1_ids == p2_ids, (
            f"A/A floor violated on cue index {i} against a single shared, "
            f"unrebuilt graph object (p1 count={len(p1_ids)}, "
            f"p2 count={len(p2_ids)})"
        )

    monkeypatch.delenv("IAI_MCP_VALENCE_WRITE_OFF", raising=False)
    _mark_labile(store, target_id)

    lifecycle_path = tmp_path / "lifecycle.json"
    save_state(default_state(), lifecycle_path)
    pipeline = SleepPipeline(store=store, lifecycle_state_path=lifecycle_path)
    done, payload = pipeline._step_reconsolidation_valence(interrupt_check=None)
    assert done is True
    assert payload["valence_writes"] >= 1, (
        "non-vacuity precondition: the sleep step must actually write a "
        "valence delta for the planted-labile target"
    )

    runtime_graph_cache.invalidate(store)
    assert getattr(store, "_warm_graph_bundle", None) is None, (
        "invalidate did not clear the warm bundle -- the B rebuild would "
        "silently re-serve the stale (pre-write) graph"
    )
    graph_b, assignment_b, rich_club_b = build_runtime_graph(store)

    b_hits = {
        i: _dispatch_hits(store, graph_b, assignment_b, rich_club_b, embedder, cue, "gate-on")
        for i, cue in enumerate(cues)
    }

    per_cue = {
        i: {
            "p1": {h.record_id for h in p1_hits[i][:_TOP_K]},
            "p2": {h.record_id for h in p2_hits[i][:_TOP_K]},
            "b": {h.record_id for h in b_hits[i][:_TOP_K]},
        }
        for i in range(len(cues))
    }
    dropped = _comparator(per_cue)
    assert dropped == {}, f"A/A-stable id(s) dropped out of the ON leg: {dropped}"

    target_before = _hit_for(p2_hits[target_cue_index][:_TOP_K], target_id)
    target_after = _hit_for(b_hits[target_cue_index][:_TOP_K], target_id)
    assert target_before is not None and target_after is not None
    assert "xval" not in target_before.reason
    assert "xval" in target_after.reason, (
        f"xval marker missing from post-write reason for {target_id}"
    )
    assert target_after.score > target_before.score, (
        f"planted valence did not raise {target_id}'s score: "
        f"before={target_before.score} after={target_after.score}"
    )

    order_differs = any(
        [h.record_id for h in p2_hits[i][:_TOP_K]] != [h.record_id for h in b_hits[i][:_TOP_K]]
        for i in range(len(cues))
    )
    assert order_differs, (
        "non-vacuity: the ON leg's rank order is byte-identical to the OFF "
        "leg across every cue -- the planted valence write had no "
        "observable effect on ranking"
    )


# ---------------------------------------------------------------------------
# Cannot-false-green control
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_comparator_flags_planted_tombstone_regression(tmp_path, monkeypatch, driver):
    """A genuine storage-mutation regression (deleting an A/A-stable
    record, NOT writing valence) must be flagged by the same comparator the
    differential gate above trusts -- proving the harness can fail before it
    is trusted to pass."""
    store, records, cues, embedder = _build_gate_store(driver, tmp_path, monkeypatch)
    monkeypatch.delenv("IAI_MCP_VALENCE_WRITE_OFF", raising=False)

    runtime_graph_cache.invalidate(store)
    graph, assignment, rich_club = build_runtime_graph(store)

    cue_index, cue = next((i, c) for i, c in enumerate(cues) if c.band == "specific")
    p1 = _dispatch_hits(store, graph, assignment, rich_club, embedder, cue, "tombstone-p1")

    runtime_graph_cache.invalidate(store)
    graph2, assignment2, rich_club2 = build_runtime_graph(store)
    p2 = _dispatch_hits(store, graph2, assignment2, rich_club2, embedder, cue, "tombstone-p2")

    p1_ids = {h.record_id for h in p1[:_TOP_K]}
    p2_ids = {h.record_id for h in p2[:_TOP_K]}
    stable = p1_ids & p2_ids
    assert stable, "fixture precondition: no A/A-stable id for the tombstone control"
    target_id = sorted(stable, key=str)[0]

    store.delete(target_id)
    runtime_graph_cache.invalidate(store)
    graph3, assignment3, rich_club3 = build_runtime_graph(store)
    b = _dispatch_hits(store, graph3, assignment3, rich_club3, embedder, cue, "tombstone-b")
    b_ids = {h.record_id for h in b[:_TOP_K]}

    per_cue = {cue_index: {"p1": p1_ids, "p2": p2_ids, "b": b_ids}}
    dropped = _comparator(per_cue)
    assert target_id in dropped.get(cue_index, set()), (
        f"planted tombstone control did not flag deleted id {target_id} as "
        "a drop-out -- the comparator or id flow is broken"
    )

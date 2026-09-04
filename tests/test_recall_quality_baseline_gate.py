"""Default-gate accuracy-regression gate for recall quality.

Loads the committed recall-quality baseline fixture, recomputes the same
reference-cue recalls at n=1000, and fails if any field regresses beyond
its tolerance. This is the numeric consumer the fixture was missing: the
slow fixture-producing test only validated structural presence, so a real
regression could silently overwrite the committed baseline and pass.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_recall_stage_profile import (
    LEXICAL_GENERIC_CUE,
    LEXICAL_SPECIFIC_CUE,
    _ann_top_k,
    _estimate_ann_top200_cosine_threshold,
    _exact_top_k,
    _monkeypatch_env,
    _recall_at_k,
    build_gate_b_reference_store,
)
from _recall_helpers import diff_recall_quality_baseline_entry

from iai_mcp.embed import Embedder

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "recall_quality_baseline.json"

# The fixture's own small_k_ef_blast_radius.change_direction is a fixed
# qualitative label produced by the fixture-generating test, not derived
# from a live comparison -- mirrored here so the diff exercises the field.
_BLAST_RADIUS_CHANGE_DIRECTION = "toward-exact-or-unchanged"


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _compute_fresh_entry(
    store, graph, assignment, rich_club, embedder, gold_ids,
    cue_vec_generic, cue_vec_specific,
) -> dict:
    from iai_mcp.pipeline import recall_for_response
    import iai_mcp.pipeline as _pm

    hub_gold_id = gold_ids["hub_gold"]
    seed_id = gold_ids["seed"]
    two_hop_gold_id = gold_ids["two_hop_gold"]
    contradict_a_id = gold_ids["contradict_a"]
    contradict_b_id = gold_ids["contradict_b"]

    cue_spec_arr = np.asarray(cue_vec_specific, dtype=np.float32)
    cue_spec_arr /= np.linalg.norm(cue_spec_arr)

    ann_boundary = _estimate_ann_top200_cosine_threshold(store, cue_vec_specific)
    two_hop_vec = np.asarray(store.get(two_hop_gold_id).embedding, dtype=np.float32)
    two_hop_vec /= np.linalg.norm(two_hop_vec)
    gold_cosine_vs_cue = float(np.dot(two_hop_vec, cue_spec_arr))
    two_hop_outside_ann_top200 = gold_cosine_vs_cue < ann_boundary

    spread_from_seed = graph.two_hop_neighborhood([seed_id], top_k=5)
    two_hop_reachable = two_hop_gold_id in spread_from_seed

    reference_cues_meta = [
        {"cue": LEXICAL_GENERIC_CUE, "cue_label": "lexical-generic", "must_hit": True,
         "expected_stable_keys": [str(hub_gold_id)]},
        {"cue": LEXICAL_SPECIFIC_CUE, "cue_label": "lexical-specific", "must_hit": True,
         "expected_stable_keys": [str(two_hop_gold_id)]},
    ]

    cue_records = []
    for meta in reference_cues_meta:
        _pm._last_recall_latency_ms = 0.0
        response = recall_for_response(
            store=store, graph=graph, assignment=assignment, rich_club=rich_club,
            embedder=embedder, cue=meta["cue"], session_id="user",
            budget_tokens=1500, mode="concept",
        )
        returned_hit_ids = [str(h.record_id) for h in response.hits]
        expected = meta["expected_stable_keys"]
        r5 = sum(1 for ek in expected if ek in returned_hit_ids[:5]) / max(len(expected), 1)
        r10 = sum(1 for ek in expected if ek in returned_hit_ids[:10]) / max(len(expected), 1)
        anti_hit_ids = [str(h.record_id) for h in response.anti_hits]
        anti_hit_surfaced = (
            str(contradict_a_id) in anti_hit_ids or str(contradict_b_id) in anti_hit_ids
        )
        cue_records.append({
            "cue_label": meta["cue_label"],
            "must_hit": meta["must_hit"],
            "recall_at_5": round(r5, 4),
            "recall_at_10": round(r10, 4),
            "anti_hit_surfaced": anti_hit_surfaced,
        })

    recall_at_200 = {}
    for cue_vec, cue_label in [
        (cue_vec_generic, "lexical-generic"),
        (cue_vec_specific, "lexical-specific"),
    ]:
        ann_ids = _ann_top_k(store, cue_vec, k=200)
        exact_ids = _exact_top_k(store, cue_vec, k=200)
        overlap = _recall_at_k(ann_ids, exact_ids, k=min(200, len(exact_ids), len(ann_ids)))
        recall_at_200[cue_label] = round(overlap, 4)

    return {
        "reference_cues": cue_records,
        "recall_at_200": recall_at_200,
        "two_hop_gold_reachable_via_2hop": two_hop_reachable,
        "two_hop_gold_outside_ann_top200": two_hop_outside_ann_top200,
        "small_k_ef_blast_radius": {"change_direction": _BLAST_RADIUS_CHANGE_DIRECTION},
    }


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_n1k_recall_quality_matches_committed_baseline(tmp_path, monkeypatch, driver):
    _select_driver(driver, monkeypatch)
    _monkeypatch_env(monkeypatch, tmp_path)

    assert FIXTURE_PATH.exists(), f"committed baseline fixture not found: {FIXTURE_PATH}"
    with open(FIXTURE_PATH) as f:
        committed = json.load(f)
    committed_n1k = committed["n1k"]
    n_records = committed_n1k["n_records"]

    embedder = Embedder()
    cue_vec_generic = embedder.embed(LEXICAL_GENERIC_CUE)
    cue_vec_specific = embedder.embed(LEXICAL_SPECIFIC_CUE)

    store_path = tmp_path / "quality-gate-n1k-store"
    # Must build from the SAME reference-store shape the committed fixture
    # was produced from -- a differently-shaped store (different gold-record
    # graph degree) changes graph-based ranking and makes this diff compare
    # two different systems, not the same system at two points in time.
    store, gold_ids = build_gate_b_reference_store(
        store_path, n_records, cue_vec_generic, cue_vec_specific,
    )
    monkeypatch.setenv("IAI_MCP_STORE", str(store_path))

    from iai_mcp.retrieve import build_runtime_graph
    graph, assignment, rich_club = build_runtime_graph(store)

    fresh_entry = _compute_fresh_entry(
        store, graph, assignment, rich_club, embedder, gold_ids,
        cue_vec_generic, cue_vec_specific,
    )

    violations = diff_recall_quality_baseline_entry(fresh_entry, committed_n1k)
    assert not violations, (
        f"recall-quality regression vs the committed baseline ({FIXTURE_PATH}):\n"
        + "\n".join(violations)
    )


def _load_committed_n1k() -> dict:
    with open(FIXTURE_PATH) as f:
        return json.load(f)["n1k"]


def test_diff_helper_no_violations_when_unchanged():
    committed = _load_committed_n1k()
    fresh = copy.deepcopy(committed)
    violations = diff_recall_quality_baseline_entry(fresh, committed)
    assert not violations, (
        f"diff helper flagged spurious violations on unchanged data: {violations}"
    )


def test_diff_helper_flags_planted_recall_at_200_regression():
    """Positive control: prove the diff helper actually flags a planted
    regression, not just passes vacuously because nothing trips it."""
    committed = _load_committed_n1k()
    fresh = copy.deepcopy(committed)
    cue_label = next(iter(fresh["recall_at_200"]))
    fresh["recall_at_200"][cue_label] = max(
        0.0, committed["recall_at_200"][cue_label] - 0.5
    )

    violations = diff_recall_quality_baseline_entry(fresh, committed)
    assert violations, "diff helper failed to flag a planted recall_at_200 regression"
    assert any("recall_at_200" in v for v in violations)


def test_diff_helper_flags_planted_must_hit_recall_drop():
    """Positive control: an exact-match field (recall_at_5 for a must-hit
    cue) must be flagged, not just the epsilon-banded recall_at_200."""
    committed = _load_committed_n1k()
    fresh = copy.deepcopy(committed)
    assert fresh["reference_cues"][0]["must_hit"] is True
    fresh["reference_cues"][0]["recall_at_5"] = 0.0

    violations = diff_recall_quality_baseline_entry(fresh, committed)
    assert violations, "diff helper failed to flag a planted recall_at_5 regression"


def test_diff_helper_flags_planted_boolean_flip():
    """Positive control: a boolean field flip must be flagged exactly, not
    tolerated the way recall_at_200's epsilon band tolerates noise."""
    committed = _load_committed_n1k()
    fresh = copy.deepcopy(committed)
    fresh["two_hop_gold_reachable_via_2hop"] = not committed["two_hop_gold_reachable_via_2hop"]

    violations = diff_recall_quality_baseline_entry(fresh, committed)
    assert violations, "diff helper failed to flag a planted boolean-field flip"


def test_diff_helper_flags_planted_qualitative_change():
    """Positive control: the qualitative small_k_ef_blast_radius field must
    be flagged on an exact mismatch."""
    committed = _load_committed_n1k()
    fresh = copy.deepcopy(committed)
    fresh["small_k_ef_blast_radius"]["change_direction"] = "away-from-exact"

    violations = diff_recall_quality_baseline_entry(fresh, committed)
    assert violations, "diff helper failed to flag a planted qualitative-field change"


def test_diff_helper_flags_cue_dropped_entirely_from_fresh_side():
    """Positive control: a cue present in the committed baseline but absent
    from the fresh recompute (e.g. a caller bug that drops a must-hit cue)
    must be flagged, not silently pass because the loop never visits it."""
    committed = _load_committed_n1k()
    fresh = copy.deepcopy(committed)
    dropped_label = fresh["reference_cues"][0]["cue_label"]
    fresh["reference_cues"].pop(0)

    violations = diff_recall_quality_baseline_entry(fresh, committed)
    assert violations, "diff helper failed to flag a cue dropped entirely from the fresh side"
    assert any(dropped_label in v for v in violations)

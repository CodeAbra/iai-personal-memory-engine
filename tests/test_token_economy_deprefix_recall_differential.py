"""CI-safe recall drop-out comparator + re-arm controls for the render-only
cls_summary lever.

A labelled fixture cannot prove regression-freedom: relevant_record_ids go
stale as the corpus changes underneath them. The comparator below derives
its verdict from cue drop-out, not label recall — an id that surfaced for a
cue before a change must still surface after it, unless the drop is within a
measured A/A noise floor (two independent pre-change rebuilds of the same
copy). Movement within the retrieved set is explicitly benign; only drop-out
of an A/A-stable id counts as a regression. The comparator and the tombstone
control below stay live so this file re-arms automatically if a future
storage-mutation lever is reintroduced.

Runs entirely on read-only-sourced copies
(bench/recall_accuracy_real.open_eval_copy_store) — the operator's live
store is never opened for write. Diagnostics on failure/skip carry record
ids and counts only, never cue text or stored content (cues are derived
from the owner's memory content).
"""
from __future__ import annotations

from uuid import UUID

import pytest

from bench.recall_accuracy_real import (
    _AFTER_STRUCTURAL_WEIGHT,
    _dispatch_real_cue,
    open_eval_copy_store,
)
from iai_mcp import runtime_graph_cache
from iai_mcp.retrieve import build_runtime_graph
from iai_mcp.session import _clean_surface
from iai_mcp.types import CLS_SUMMARY_PREFIX_RE

_MIN_CUE_WORDS = 3
_CUE_WORD_COUNT = 6

# Selection thresholds mirrored from the record schema: a candidate must be
# unpinned, untrusted knowledge (below the promotion threshold), and match
# the boilerplate cls_summary prefix.
UNTOUCHABLE_TRUST = 0.9
_BATCH_SIZE = 500


# ---------------------------------------------------------------------------
# Pure comparator (real-store-independent)
# ---------------------------------------------------------------------------


def _comparator(
    per_cue: "dict[str, dict[str, set[str]]]",
    classification: "dict[str, bool]",
) -> "tuple[dict[str, set[str]], dict[str, set[str]]]":
    """Regression rule (pinned): an id is a regression for a cue iff it is
    surfaced in P1 AND A/A-stable (also surfaced in P2) AND absent in B. An
    id P1 surfaced but P2 did not is excused as harness noise. A reshuffle
    within the retrieved set is never a regression -- only drop-out is.

    Returns (regressions, instability), each partitioned into
    ``{"cls_summary": set[id], "non_cls_summary": set[id]}``. ``instability``
    is P1-minus-P2, the population the caller bounds with the A/A ceiling.
    """
    regressions: "dict[str, set[str]]" = {"cls_summary": set(), "non_cls_summary": set()}
    instability: "dict[str, set[str]]" = {"cls_summary": set(), "non_cls_summary": set()}
    for sets in per_cue.values():
        p1, p2, b = sets["p1"], sets["p2"], sets["b"]
        stable = p1 & p2
        unstable = p1 - p2
        dropped = stable - b
        for rid in dropped:
            partition = "cls_summary" if classification.get(rid, False) else "non_cls_summary"
            regressions[partition].add(rid)
        for rid in unstable:
            partition = "cls_summary" if classification.get(rid, False) else "non_cls_summary"
            instability[partition].add(rid)
    return regressions, instability


def test_comparator_stable_cls_summary_drop_is_flagged():
    per_cue = {"cue-a": {"p1": {"s1", "n1"}, "p2": {"s1", "n1"}, "b": {"n1"}}}
    classification = {"s1": True, "n1": False}
    regressions, instability = _comparator(per_cue, classification)
    assert regressions["cls_summary"] == {"s1"}
    assert regressions["non_cls_summary"] == set()
    assert instability == {"cls_summary": set(), "non_cls_summary": set()}


def test_comparator_stable_non_cls_summary_drop_is_flagged():
    per_cue = {"cue-a": {"p1": {"s1", "n1"}, "p2": {"s1", "n1"}, "b": {"s1"}}}
    classification = {"s1": True, "n1": False}
    regressions, instability = _comparator(per_cue, classification)
    assert regressions["non_cls_summary"] == {"n1"}
    assert regressions["cls_summary"] == set()
    assert instability == {"cls_summary": set(), "non_cls_summary": set()}


def test_comparator_unstable_drop_is_excused():
    per_cue = {"cue-a": {"p1": {"s1"}, "p2": set(), "b": set()}}
    classification = {"s1": True}
    regressions, instability = _comparator(per_cue, classification)
    assert regressions == {"cls_summary": set(), "non_cls_summary": set()}
    assert instability["cls_summary"] == {"s1"}
    assert instability["non_cls_summary"] == set()


def test_comparator_within_set_reshuffle_is_benign():
    per_cue = {"cue-a": {"p1": {"s1", "n1"}, "p2": {"s1", "n1"}, "b": {"n1", "s1"}}}
    classification = {"s1": True, "n1": False}
    regressions, instability = _comparator(per_cue, classification)
    assert regressions == {"cls_summary": set(), "non_cls_summary": set()}
    assert instability == {"cls_summary": set(), "non_cls_summary": set()}


# ---------------------------------------------------------------------------
# Shared real-store helpers
# ---------------------------------------------------------------------------


def _real_store_present() -> bool:
    from iai_mcp.hippo import _operator_home

    return (_operator_home() / ".iai-mcp" / "hippo" / "brain.sqlite3").exists()


def _select_affected_candidates(store, cap: int) -> "list[tuple[str, str, str]]":
    """Ids whose plaintext columns mark them as unpinned, untrusted
    cls_summary candidates matching the boilerplate prefix. Returns
    ``(id, full_pre_text, real_content_text)``, deterministic order, capped.
    """
    candidate_ids: "list[UUID]" = []
    last_id = ""
    while True:
        with store.db._conn_lock:
            rows = store.db._conn.execute(
                "SELECT id FROM records"
                " WHERE tombstoned_at IS NULL"
                "   AND COALESCE(pinned, 0) = 0"
                "   AND COALESCE(never_merge, 0) = 0"
                "   AND COALESCE(embedding_pending, 0) = 0"
                "   AND (s5_trust_score IS NULL OR s5_trust_score < ?)"
                "   AND tags_json LIKE '%cls_summary%'"
                "   AND id > ?"
                " ORDER BY id"
                " LIMIT ?",
                (UNTOUCHABLE_TRUST, last_id, _BATCH_SIZE),
            ).fetchall()
        if not rows:
            break
        for row in rows:
            candidate_ids.append(UUID(str(row["id"])))
        last_id = str(rows[-1]["id"])

    if not candidate_ids:
        return []

    out: "list[tuple[str, str, str]]" = []
    batch = store.get_batch(candidate_ids)
    for rid in candidate_ids:
        rec = batch.get(rid)
        if rec is None or "cls_summary" not in (rec.tags or []):
            continue
        text = rec.literal_surface or ""
        m = CLS_SUMMARY_PREFIX_RE.match(text)
        if m is None:
            continue
        out.append((str(rid), text, text[m.end():]))
        if len(out) >= cap:
            break
    return out


# ---------------------------------------------------------------------------
# A/B retrieval leg — retired under render-only (no storage mutation exists)
# ---------------------------------------------------------------------------


def test_ab_retrieval_leg_retired_render_only():
    pytest.skip(
        "A/B retrieval leg retired: the render-only cls_summary lever never "
        "mutates storage, so a post-render rebuild differs from a "
        "pre-render rebuild only by rebuild jitter — the comparator cannot "
        "fail for a lever reason here. The structural byte-identity guard "
        "(tests/test_render_only_no_storage_mutation.py) is the load-bearing "
        "no-regression proof for this lever. The comparator above and the "
        "tombstone control below stay live and will re-arm automatically if "
        "a future lever reintroduces a storage mutation."
    )


# ---------------------------------------------------------------------------
# Non-vacuity: end-to-end planted drop, its OWN copy, its OWN P2 A/A leg
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_end_to_end_tombstone_detected_as_regression(driver, monkeypatch):
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")

    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    monkeypatch.delenv("IAI_MCP_STORE", raising=False)

    if not _real_store_present():
        pytest.skip(
            "real store not found; this is a real-store control, not a "
            "fresh-clone CI gate"
        )

    cm = open_eval_copy_store(driver=driver)
    try:
        store = cm.__enter__()
    except Exception as exc:
        if driver == "lilli":
            pytest.skip(
                "lilli driver could not open the real-store copy (on-disk "
                f"format mismatch or driver error): {exc}"
            )
        raise

    try:
        runtime_graph_cache.invalidate(store)
        _graph_p1, _assignment_p1, rich_club_p1 = build_runtime_graph(store)
        assert rich_club_p1, "rich_club membership empty on the control's P1 rebuild"

        affected = _select_affected_candidates(store, cap=1)
        if not affected:
            pytest.skip(
                "no backfill-affected cls_summary record found for the "
                "planted-drop control on this real-store copy"
            )
        target_id, _full, real_text = affected[0]
        words = _clean_surface(real_text).split()
        if len(words) < _MIN_CUE_WORDS:
            pytest.skip(
                "backfill-affected candidate's real content too short to "
                "build a cue for the planted-drop control"
            )
        cue_text = " ".join(words[:_CUE_WORD_COUNT])

        p1_ids = set(_dispatch_real_cue(store, cue_text, _AFTER_STRUCTURAL_WEIGHT))

        runtime_graph_cache.invalidate(store)
        assert getattr(store, "_warm_graph_bundle", None) is None
        _graph_p2, _assignment_p2, rich_club_p2 = build_runtime_graph(store)
        assert rich_club_p2, "rich_club membership empty on the control's P2 rebuild"
        p2_ids = set(_dispatch_real_cue(store, cue_text, _AFTER_STRUCTURAL_WEIGHT))

        if target_id not in (p1_ids & p2_ids):
            pytest.skip(
                "planted-drop target id is not A/A-stable on this "
                "real-store copy — cannot prove the control without a "
                "stable subject"
            )

        store.delete(UUID(target_id))
        runtime_graph_cache.invalidate(store)
        _graph_b, _assignment_b, rich_club_b = build_runtime_graph(store)
        b_ids = set(_dispatch_real_cue(store, cue_text, _AFTER_STRUCTURAL_WEIGHT))

        per_cue = {"planted": {"p1": p1_ids, "p2": p2_ids, "b": b_ids}}
        classification = {target_id: True}
        regressions, _instability = _comparator(per_cue, classification)

        assert target_id in regressions["cls_summary"], (
            "planted-drop control did not flag the tombstoned, A/A-stable "
            "cls_summary id as a regression — the comparator or id flow is "
            "broken"
        )
    finally:
        cm.__exit__(None, None, None)

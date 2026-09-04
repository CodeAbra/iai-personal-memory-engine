"""SC-4 witness: real-store before/after token reduction with no recall
regression for the compact rich_club label lever
(``IAI_MCP_RICH_CLUB_COMPACT_LABEL``).

Reuses the per-component token instrument (``bench/token_breakdown.py``,
``bench/token_recall_guard.py``) and the read-only real-store-copy helper
(``bench/recall_accuracy_real.py::open_eval_copy_store``) — never opens or
mutates the operator's live store, never re-implements store copying or
token counting.

Both the before render (env toggle OFF, legacy label+budget) and the after
render (default ON) recompose through the exact SAME production path
(``iai_mcp.session._compose_session_start_payload`` at
``wake_depth="standard"``) against the SAME store copy and the SAME
``rich_club`` ordering from a single ``build_runtime_graph`` call — only the
env toggle differs between the two renders. This sidesteps a measured
cross-run community-partition drift between two independently rebuilt
graphs: both sides of this comparison see the identical cold-rebuilt graph.

The load-bearing SC-4 gate is the admitted-record-id-set check: the
skip filter inside ``_rich_club_segment_with_budget`` (``rec is None`` /
empty ``cleaned`` surface) is label-independent, so the Nth surviving
rendered line always maps to the Nth surviving uid in ``rich_club`` order,
identically OFF and ON. Recovering admitted ids by POSITION (never by
matching each line's content-text back to a record) is required — repeated
near-constant tags/content prefixes in the real corpus make content-text
matching vacuous (a dropped record can leave the recovered set unchanged).

Real-store witness, not a fresh-clone CI gate: every test here skips
gracefully when the operator's real store (or the local-only labelled
fixture, for the supplementary check) is absent.
"""
from __future__ import annotations

import pytest

from bench.recall_accuracy_real import open_eval_copy_store
from bench.token_breakdown import compose_and_measure
from iai_mcp.retrieve import build_runtime_graph
from iai_mcp.session import _clean_surface, _compose_session_start_payload

_COMPACT_ENV = "IAI_MCP_RICH_CLUB_COMPACT_LABEL"


def _real_store_present() -> bool:
    from iai_mcp.hippo import _operator_home

    return (_operator_home() / ".iai-mcp" / "hippo" / "brain.sqlite3").exists()


@pytest.fixture(params=["stdlib", "lilli"])
def driver(request, monkeypatch):
    if request.param == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    return request.param


def test_real_store_token_reduction_and_admitted_id_subset(driver, monkeypatch):
    """Load-bearing SC-4 gate: same-session live OFF-vs-ON tiktoken total is
    strictly reduced, and the OFF-admitted-id-set is a subset of the
    ON-admitted-id-set (identity on the saturated branch), recovered by
    position."""
    if not _real_store_present():
        pytest.skip(
            "real store not found; this is a real-store witness test, not a "
            "fresh-clone CI gate"
        )

    from iai_mcp.hippo import _operator_home

    real_db = _operator_home() / ".iai-mcp" / "hippo" / "brain.sqlite3"
    mtime_before = real_db.stat().st_mtime
    size_before = real_db.stat().st_size

    with open_eval_copy_store(driver=driver) as store:
        _graph, assignment, rich_club = build_runtime_graph(store)
        assert rich_club, "real store's rich_club membership is empty — nothing to measure"

        monkeypatch.setenv(_COMPACT_ENV, "0")
        report_off = compose_and_measure(store, assignment, rich_club, session_id="token-economy-off")
        payload_off = _compose_session_start_payload(
            store, assignment, rich_club,
            session_id="token-economy-off",
            profile_state={"wake_depth": "standard"},
        )

        monkeypatch.delenv(_COMPACT_ENV, raising=False)
        report_on = compose_and_measure(store, assignment, rich_club, session_id="token-economy-on")
        payload_on = _compose_session_start_payload(
            store, assignment, rich_club,
            session_id="token-economy-on",
            profile_state={"wake_depth": "standard"},
        )

        by_uuid = store.get_batch(rich_club)

    assert real_db.stat().st_mtime == mtime_before, "real store mtime changed — write leaked to the live store"
    assert real_db.stat().st_size == size_before, "real store size changed — write leaked to the live store"

    # Position-based admitted-id recovery: the render skip filter (record
    # missing / empty cleaned surface) never depends on the aaak label, so
    # it is identical OFF vs ON — the Nth surviving rich_club uid is the
    # source of the Nth rendered line on BOTH sides.
    survivors = [
        uid
        for uid in rich_club
        if by_uuid.get(uid) is not None and _clean_surface(by_uuid[uid].literal_surface)
    ]
    assert survivors, "no rich_club record survived the render skip filter"

    lines_off = [ln for ln in payload_off.rich_club.split("\n") if ln]
    lines_on = [ln for ln in payload_on.rich_club.split("\n") if ln]
    assert lines_off, "OFF-lever (legacy) render produced no rich_club lines"
    assert lines_on, "ON-lever (default, compact) render produced no rich_club lines"

    ids_off = survivors[: len(lines_off)]
    ids_on = survivors[: len(lines_on)]

    assert set(ids_off) <= set(ids_on), (
        "SC-4 REGRESSION: a record shown before the compact-label lever is "
        "missing after it — the before-admitted-id-set must be a subset of "
        "the after-admitted-id-set"
    )

    # Saturated branch: the render loop broke before exhausting the
    # candidate pool on BOTH sides -> the proportional budget reduction
    # must SAVE the same set, not widen it.
    saturated_off = len(lines_off) < len(survivors)
    saturated_on = len(lines_on) < len(survivors)
    if saturated_off and saturated_on:
        assert ids_off == ids_on, (
            "saturated-branch admitted-id-set is not IDENTICAL before vs "
            "after — the proportional budget reduction must SAVE, not widen "
            f"(off={len(ids_off)}, on={len(ids_on)})"
        )

    assert report_on["total"]["tiktoken_tokens"] < report_off["total"]["tiktoken_tokens"], (
        f"after-lever (compact label, ON) tiktoken total "
        f"({report_on['total']['tiktoken_tokens']}) is not strictly below the "
        f"before-lever (legacy label, OFF) total "
        f"({report_off['total']['tiktoken_tokens']}) in this same session"
    )


def test_supplementary_dispatch_fnr_unaffected_by_render_lever(driver):
    """Supplementary signal ONLY — NOT the SC-4 gate. The compact-label
    lever is a render-site transform that never becomes an input to
    ``core.dispatch``; identical weight_for callables on both sides must
    yield an identical false_negative_rate by construction, regardless of
    the lever. Retained as a sanity witness that dispatch still runs
    against the real corpus, not as a regression detector for this class of
    change (the labelled fixture's cues have near-zero overlap with the
    rich_club tier's membership on the real corpus, so this check has
    limited power to catch a rich_club-only render regression)."""
    from bench.recall_accuracy_real import fixture_path

    if not fixture_path().exists():
        pytest.skip(
            f"local labelled fixture not found at {fixture_path()} — "
            "this fixture is LOCAL-ONLY and never committed"
        )
    if not _real_store_present():
        pytest.skip("real store not found; this is a real-store witness test")

    from bench.token_recall_guard import run_before_after

    result = run_before_after(
        before_weight_for=lambda _c: 0.0,
        after_weight_for=lambda _c: 0.0,
        driver=driver,
    )
    if result.get("skipped"):
        pytest.skip(result.get("reason", "fixture unavailable"))

    assert result["before"]["false_negative_rate"] == result["after"]["false_negative_rate"], (
        "supplementary signal only, not the SC-4 gate: a render-site label "
        "lever is never an input to core.dispatch, so identical weight_for "
        "callables on both sides must yield an identical false_negative_rate "
        "regardless of the lever's own toggle state"
    )

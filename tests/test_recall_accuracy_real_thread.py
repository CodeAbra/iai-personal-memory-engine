"""RACC-EVAL-01 SC-1 witness gate: real-thread-following recall accuracy.

Locks the CONTRACT for the labelled real-corpus eval set before any
downstream code exists (fixture schema, harness wiring through
``core.dispatch``, trace-mark coverage, verbatim preservation, and the
strict ``after_184.false_negative_rate < baseline.false_negative_rate``
improvement bar).

The labelled fixture (``~/.iai-mcp/eval-fixtures/labelled_real_thread_cues.json``
by default, override via ``IAI_MCP_EVAL_FIXTURE_PATH``) is LOCAL-ONLY and is
NEVER committed to this repo — every test below degrades to a graceful skip
when it is absent, so a fresh clone or CI environment without the owner's
personal fixture never hard-fails. The harness module
(``bench/recall_accuracy_real.py``) is a separate, later deliverable; tests
that depend on it also skip gracefully while it does not yet exist.

Dual-driver: every test is parametrized over ``driver in {"stdlib", "lilli"}``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

_DEFAULT_FIXTURE_PATH = Path("~/.iai-mcp/eval-fixtures/labelled_real_thread_cues.json").expanduser()
_HARNESS_PATH = Path("bench/recall_accuracy_real.py")


def _fixture_path() -> Path:
    override = os.environ.get("IAI_MCP_EVAL_FIXTURE_PATH")
    if override:
        return Path(override).expanduser()
    return _DEFAULT_FIXTURE_PATH


def _fixture_absent_reason() -> str:
    return (
        "Local labelled real-thread fixture not found at "
        f"{_fixture_path()}. This fixture is LOCAL-ONLY (never committed). "
        "Build it by running: python scripts/label_real_thread_cues.py"
    )


def _harness_absent_reason() -> str:
    return (
        f"Harness module not found at {_HARNESS_PATH}."
    )


def _load_fixture() -> dict[str, Any]:
    with _fixture_path().open("r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(autouse=True)
def _clear_toggle_env(monkeypatch):
    # None of the three membership-affecting toggles should leak between tests.
    for var in (
        "IAI_MCP_EXACT_AUTHORITY_OFF",
        "IAI_MCP_CONF_ESCALATE_OFF",
        "IAI_MCP_MULTI_SEED_OFF",
    ):
        monkeypatch.delenv(var, raising=False)


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


def test_fixture_schema_valid_when_present(driver):
    """The labelled fixture, once built, must satisfy the RACC-EVAL-01 schema:
    30-50 cues, each with a non-empty cue_text and non-empty relevant_record_ids."""
    if not _fixture_path().exists():
        pytest.skip(_fixture_absent_reason())

    data = _load_fixture()
    assert data.get("schema_version") == 1, "fixture must declare schema_version == 1"
    cues = data.get("cues")
    assert isinstance(cues, list), "fixture 'cues' field must be a list"
    assert 30 <= len(cues) <= 50, (
        f"expected 30-50 labelled cues per RACC-EVAL-01, got {len(cues)}"
    )
    for entry in cues:
        assert isinstance(entry.get("cue_id"), str) and entry["cue_id"], (
            "every cue entry needs a non-empty string cue_id"
        )
        assert isinstance(entry.get("cue_text"), str) and entry["cue_text"], (
            "every cue entry needs a non-empty string cue_text"
        )
        relevant_ids = entry.get("relevant_record_ids")
        assert isinstance(relevant_ids, list) and relevant_ids, (
            "every cue entry needs a non-empty relevant_record_ids list — "
            "a cue with no relevant ids is not a usable eval case"
        )
        assert all(isinstance(rid, str) for rid in relevant_ids)
        if "notes" in entry:
            assert isinstance(entry["notes"], str)


def test_harness_uses_core_dispatch(driver):
    """Static wiring gate: the harness must dispatch through core.dispatch,
    never call recall_for_response directly — the exact 184-04-SUMMARY.md
    escalation mistake this phase must not repeat."""
    if not _HARNESS_PATH.exists():
        pytest.skip(_harness_absent_reason())

    src = _HARNESS_PATH.read_text(encoding="utf-8")
    assert "core.dispatch(" in src, (
        "harness must route recall through core.dispatch(...), not "
        "recall_for_response directly"
    )
    assert "from iai_mcp.pipeline import recall_for_response" not in src, (
        "harness must not import recall_for_response directly — this bypasses "
        "D-A (merge_authority_hits), which lives exclusively in core/__init__.py"
    )
    assert "recall_for_response(" not in src, (
        "harness must not call recall_for_response directly — route through "
        "core.dispatch instead"
    )


def test_trace_marks_fire_on_real_cue(driver):
    """At least one real cue's traced dispatch must fire the soft_gate,
    conf_escalate, and multi_seed marks; a cue flagged expects_structural
    must additionally fire cleanup_attractor."""
    if not _fixture_path().exists():
        pytest.skip(_fixture_absent_reason())
    if not _HARNESS_PATH.exists():
        pytest.skip(_harness_absent_reason())

    from bench.recall_accuracy_real import dispatch_cue_traced

    data = _load_fixture()
    cues = data["cues"]
    structural_cues = [c for c in cues if c.get("expects_structural")]
    target_cue = structural_cues[0] if structural_cues else cues[0]

    _returned_ids, marks_set = dispatch_cue_traced(target_cue, driver=driver)

    assert marks_set >= {"soft_gate", "conf_escalate", "multi_seed"}, (
        f"expected soft_gate/conf_escalate/multi_seed to fire, got {sorted(marks_set)}"
    )

    if structural_cues:
        found_cleanup = False
        for cue in structural_cues:
            _ids, cue_marks = dispatch_cue_traced(cue, driver=driver)
            if "cleanup_attractor" in cue_marks:
                found_cleanup = True
                break
        assert found_cleanup, (
            "expected at least one expects_structural cue to fire cleanup_attractor"
        )


def test_verbatim_literal_surface_preserved(driver):
    """literal_surface must be byte-identical before and after a real-cue
    dispatch — recall's reinforce side effect must never touch stored text."""
    if not _fixture_path().exists():
        pytest.skip(_fixture_absent_reason())
    if not _HARNESS_PATH.exists():
        pytest.skip(_harness_absent_reason())

    from bench.recall_accuracy_real import (
        dispatch_cue_traced,
        open_eval_copy_store,
    )

    data = _load_fixture()
    cue = data["cues"][0]

    with open_eval_copy_store(driver=driver) as store:
        # The fixture labels outlive the store's lifecycle: erasure sweeps
        # legitimately delete long-tombstoned records, so resolve the label
        # set against the live copy and prove the invariant on the survivors.
        before = {}
        for rid in cue["relevant_record_ids"]:
            rec = store.get(rid)
            if rec is not None:
                before[rid] = rec.literal_surface
        if not before:
            pytest.skip(
                "every labelled record for this cue has been erased from the "
                "real store — relabel the fixture"
            )

        dispatch_cue_traced(cue, driver=driver, store=store)

        after = {}
        for rid in before:
            rec = store.get(rid)
            assert rec is not None, (
                f"record {rid} vanished during a read-path dispatch — recall "
                f"must never delete stored records"
            )
            after[rid] = rec.literal_surface

    assert after == before, (
        "literal_surface changed after dispatching a real cue — verbatim/"
        "lossless invariant violated"
    )


def test_after_184_strictly_better_than_baseline(driver):
    """The SC-1 witness: after-184 false_negative_rate must be STRICTLY
    lower than the baseline (all mechanisms disabled) on the same real
    labelled corpus. Opt-in — requires the real Rust embedder against a
    copy of the owner's actual Hippo store."""
    if not _fixture_path().exists():
        pytest.skip(_fixture_absent_reason())
    if not _HARNESS_PATH.exists():
        pytest.skip(_harness_absent_reason())
    if os.environ.get("IAI_MCP_DRYRUN_REAL_STORE") != "1":
        pytest.skip(
            "real-corpus SC-1 witness is opt-in: set IAI_MCP_DRYRUN_REAL_STORE=1 "
            "and have a labelled fixture present"
        )

    from bench.recall_accuracy_real import run_both_passes

    data = _load_fixture()
    result = run_both_passes(data, driver=driver)

    baseline = result["baseline"]
    after_184 = result["after_184"]

    assert after_184["false_negative_rate"] < baseline["false_negative_rate"], (
        f"expected after_184 false_negative_rate ({after_184['false_negative_rate']}) "
        f"to be STRICTLY lower than baseline ({baseline['false_negative_rate']})"
    )
    assert len(after_184["per_cue"]) == len(data["cues"]), (
        "every labelled cue must be measured in the after_184 pass"
    )
    for entry in after_184["per_cue"]:
        assert "recall_at_k" in entry
        assert isinstance(entry.get("hit"), bool)

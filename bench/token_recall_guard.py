"""Recall-regression guard for the token-economy phase.

Thin before/after wrapper reusing bench/recall_accuracy_real.py's read-only-
copy shape: runs the labelled fixture's cues through the UNCHANGED recall
dispatch path twice against the SAME store copy (before-lever config vs
after-lever config) and reports false_negative_rate for both. Also exposes
a fixture-coverage verdict for rich-club-dependent cues.

Never modifies retrieve.py / core.dispatch -- this only reads them. Every
emitted report carries ids/counts only, never cue text or stored content.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable

_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
_REPO_PATH = str(Path(__file__).resolve().parent.parent)
if _REPO_PATH not in sys.path:
    sys.path.insert(0, _REPO_PATH)

from bench.recall_accuracy_real import (  # noqa: E402
    _run_one_pass,
    _validate_fixture_dict,
    fixture_path,
    open_eval_copy_store,
)


def _fixture_absent_reason() -> str:
    return (
        "Local labelled real-thread fixture not found at "
        f"{fixture_path()}. This fixture is LOCAL-ONLY (never committed)."
    )


def _load_fixture_cues(fixture: "dict | Path | None" = None) -> list[dict]:
    if isinstance(fixture, dict):
        return _validate_fixture_dict(fixture)
    resolved = fixture if fixture is not None else fixture_path()
    if not resolved.exists():
        raise FileNotFoundError(f"labelled fixture not found: {resolved}")
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    return _validate_fixture_dict(raw)


def run_before_after(
    *,
    before_weight_for: "Callable[[dict], float]",
    after_weight_for: "Callable[[dict], float]",
    fixture: "dict | Path | None" = None,
    driver: "str | None" = None,
) -> dict:
    """Run the labelled fixture's cues twice against ONE store copy: once
    under the before-lever config, once under the after-lever config. Both
    weight_for callables receive a cue dict and return the structural_weight
    to dispatch with. Returns per_cue detail (cue_id/hit/returned_ids -- ids
    only, never cue text) alongside the aggregate false_negative_rate, so a
    later wave can derive a before-subset-of-after id-set check from the
    same run without re-dispatching.
    """
    if fixture is None and not fixture_path().exists():
        return {"skipped": True, "reason": _fixture_absent_reason()}

    cues = _load_fixture_cues(fixture)
    resolved_driver = driver if driver is not None else os.environ.get("LILLI_STORAGE_DRIVER", "stdlib")

    with open_eval_copy_store(driver=driver) as store:
        before_result = _run_one_pass(store, cues, structural_weight_for=before_weight_for)
        after_result = _run_one_pass(store, cues, structural_weight_for=after_weight_for)

    return {
        "skipped": False,
        "driver": resolved_driver,
        "cue_count": len(cues),
        "before": {
            "false_negative_rate": before_result["false_negative_rate"],
            "per_cue": [
                {"cue_id": c["cue_id"], "hit": c["hit"], "returned_ids": c["returned_ids"]}
                for c in before_result["per_cue"]
            ],
        },
        "after": {
            "false_negative_rate": after_result["false_negative_rate"],
            "per_cue": [
                {"cue_id": c["cue_id"], "hit": c["hit"], "returned_ids": c["returned_ids"]}
                for c in after_result["per_cue"]
            ],
        },
        "regression": after_result["false_negative_rate"] > before_result["false_negative_rate"],
    }


def fixture_coverage_verdict(*, driver: "str | None" = None, fixture: "dict | Path | None" = None) -> dict:
    """Reports how many labelled cues' relevant_record_ids intersect the
    CURRENT rich_club tier, plus a gate-sufficiency note: false_negative_rate
    reflects SELECTION into the rich_club tier, not the aaak_index render
    FORMAT within it -- a display-format-only lever may leave this metric
    unchanged even if it hurts orientation. Coverage here is necessary, not
    sufficient, for gating an index-format lever.
    """
    if fixture is None and not fixture_path().exists():
        return {"skipped": True, "reason": _fixture_absent_reason()}

    from iai_mcp.retrieve import build_runtime_graph

    cues = _load_fixture_cues(fixture)
    resolved_driver = driver if driver is not None else os.environ.get("LILLI_STORAGE_DRIVER", "stdlib")

    with open_eval_copy_store(driver=driver) as store:
        _graph, _assignment, rich_club = build_runtime_graph(store)
        rich_club_set = {str(uid) for uid in rich_club}

        covered = 0
        for cue in cues:
            relevant = {str(rid) for rid in cue["relevant_record_ids"]}
            if relevant & rich_club_set:
                covered += 1

    return {
        "skipped": False,
        "driver": resolved_driver,
        "cue_count": len(cues),
        "rich_club_size": len(rich_club_set),
        "cues_covering_rich_club": covered,
        "coverage_fraction": covered / len(cues) if cues else 0.0,
        "gate_sufficiency_note": (
            "false_negative_rate reflects recall SELECTION into the rich_club "
            "tier, not the aaak_index render FORMAT within it. A display-"
            "format-only lever may leave this metric unchanged even if it "
            "hurts a reading model's ability to orient from the index -- "
            "treat this coverage number as necessary, not sufficient, for "
            "gating an index-format lever; a qualitative index-usefulness "
            "check is also needed for that class of change."
        ),
    }


def main(argv: "list[str] | None" = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Recall-regression guard + fixture-coverage verdict for the token-economy phase."
    )
    parser.add_argument("--driver", choices=["stdlib", "lilli"], default="stdlib")
    parser.add_argument("--coverage-only", action="store_true")
    args = parser.parse_args(argv)

    if args.coverage_only:
        result = fixture_coverage_verdict(driver=args.driver)
    else:
        # No lever exists yet this wave -- identical before/after config is
        # the no-op self-test the lever waves will replace with the real
        # before-lever vs after-lever weight functions.
        result = run_before_after(
            before_weight_for=lambda _c: 0.0,
            after_weight_for=lambda _c: 0.0,
            driver=args.driver,
        )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

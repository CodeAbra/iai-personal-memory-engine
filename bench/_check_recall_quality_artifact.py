"""Verify a recall-quality baseline JSON artifact and its protocol doc.

Membership checks only, via ``json.loads``/attribute access -- never
``grep``. Run as ``python bench/_check_recall_quality_artifact.py
<baseline.json> <protocol.md>``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_NON_GOAL_SENTENCE = (
    "A promised numeric recall-quality target for #85 -- the milestone "
    "delivers measurement infrastructure + methodology only; recall "
    "quality is not 'solved' in one milestone."
)
_HOLDOUT_DECISION_PHRASE = "holdout"


def _fail(reasons: "list[str]") -> int:
    for reason in reasons:
        print(f"CHECK FAILED: {reason}", file=sys.stderr)
    return 1


def check_baseline_json(path: "Path | str") -> "list[str]":
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    reasons: "list[str]" = []

    for split in ("dev", "holdout"):
        block = payload.get(split)
        if not isinstance(block, dict):
            reasons.append(f"missing {split!r} block")
            continue
        if "recall_at_k" not in block:
            reasons.append(f"{split!r} block missing recall_at_k")
        if "cause_distribution" not in block:
            reasons.append(f"{split!r} block missing cause_distribution")

    required_null_arm_keys = (
        "null_arm_delta", "null_arm_ci_floor_band", "null_arm_n",
        "null_arm_discordant_n", "null_arm_verdict",
    )
    for key in required_null_arm_keys:
        if key not in payload:
            reasons.append(f"missing top-level key {key!r}")

    discordant_n = payload.get("null_arm_discordant_n")
    if not isinstance(discordant_n, int) or discordant_n <= 0:
        reasons.append(
            f"null_arm_discordant_n must be a genuine positive count, got {discordant_n!r}"
        )

    verdict = payload.get("null_arm_verdict")
    if verdict not in ("NON-PASS", "UNDERPOWERED"):
        reasons.append(
            f"null_arm_verdict must be NON-PASS or UNDERPOWERED for a baseline-only "
            f"run with no candidate lever, got {verdict!r}"
        )

    floor_band = payload.get("null_arm_ci_floor_band")
    if not (isinstance(floor_band, list) and len(floor_band) == 2):
        reasons.append(f"null_arm_ci_floor_band must be a 2-element band, got {floor_band!r}")

    return reasons


def check_protocol_doc(path: "Path | str") -> "list[str]":
    text = Path(path).read_text(encoding="utf-8")
    reasons: "list[str]" = []
    if _NON_GOAL_SENTENCE not in text:
        reasons.append("protocol doc missing the verbatim non-goal sentence")
    if _HOLDOUT_DECISION_PHRASE not in text.lower():
        reasons.append("protocol doc never mentions holdout as the decision number")
    return reasons


def main(argv: "list[str] | None" = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(
            "usage: python bench/_check_recall_quality_artifact.py <baseline.json> <protocol.md>",
            file=sys.stderr,
        )
        return 2

    baseline_path, protocol_path = argv
    reasons = check_baseline_json(baseline_path) + check_protocol_doc(protocol_path)
    if reasons:
        return _fail(reasons)

    print("recall-quality artifact + protocol doc: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

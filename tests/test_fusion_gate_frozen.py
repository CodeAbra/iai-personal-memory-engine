"""Pins fusion.rs's FROZEN_GATE_THRESHOLD by parsing the shipped source --
fails the moment the constant changes without this test being updated in
the same commit as rust/iai_mcp_tools_core/src/fusion.rs's own
`frozen_gate_threshold_is_pinned` cargo test.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FUSION_RS = REPO_ROOT / "rust" / "iai_mcp_tools_core" / "src" / "fusion.rs"

_CONST_RE = re.compile(r"pub const FROZEN_GATE_THRESHOLD:\s*f64\s*=\s*([0-9.eE+-]+);")


def _shipped_frozen_gate_threshold() -> float:
    text = FUSION_RS.read_text()
    match = _CONST_RE.search(text)
    if not match:
        raise AssertionError("FROZEN_GATE_THRESHOLD constant not found in fusion.rs")
    return float(match.group(1))


def test_shipped_frozen_gate_threshold_matches_pinned_value() -> None:
    assert _shipped_frozen_gate_threshold() == 0.03

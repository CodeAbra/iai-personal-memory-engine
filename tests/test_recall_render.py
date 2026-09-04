"""Unit tests for the stdlib-only per-turn <iai-mcp-recall> render helper.

The helper is the gate-wired proof for the arbitration fix: a superseded
hit must never render as plainly as a current one, and when the recall
result carries a corrector (anti_hits), the block must carry an explicit
supersession line so a reader can outrank a competing unhedged stale claim.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HOOKS_DIR = Path(__file__).resolve().parents[1] / "src/iai_mcp/_deploy/hooks"
if str(_HOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_HOOKS_DIR))

from _recall_render import render_recall_block  # noqa: E402


def _hit(literal_surface: str, valid_to=None, record_id="r1") -> dict:
    return {
        "record_id": record_id,
        "literal_surface": literal_surface,
        "valid_to": valid_to,
    }


def test_current_hit_renders_plainly():
    result = {"hits": [_hit("alice prefers dark mode")]}
    block = render_recall_block(result)
    assert "<iai-mcp-recall>" in block
    assert "alice prefers dark mode" in block
    assert "superseded" not in block


def test_superseded_hit_renders_dated_staleness_marker():
    result = {"hits": [_hit("alice used the old dashboard layout",
                             valid_to="2026-08-15T00:00:00+00:00")]}
    block = render_recall_block(result)
    assert "alice used the old dashboard layout" in block
    assert "superseded" in block
    assert "2026-08-15" in block


def test_result_with_corrector_renders_supersedes_line():
    result = {
        "hits": [_hit("alice's team ships on sonnet for review stages")],
        "anti_hits": [_hit(
            "alice's team ships on opus for review stages",
            valid_to="2026-08-20T00:00:00+00:00",
        )],
    }
    block = render_recall_block(result)
    assert "supersedes prior version dated 2026-08-20" in block


def test_no_hits_renders_empty_block():
    assert render_recall_block({"hits": []}) == ""
    assert render_recall_block({}) == ""


def test_output_stays_within_budget():
    long_text = "alice " + ("x" * 2000)
    result = {"hits": [_hit(long_text, record_id=f"r{i}") for i in range(10)]}
    block = render_recall_block(result)
    lines = [ln for ln in block.splitlines() if ln.startswith("- ")]
    assert len(lines) <= 3, "must stay within the existing 3-hit budget"
    for ln in lines:
        # per-hit literal_surface stays truncated to 400 chars plus a short
        # marker suffix -- never the full 2000-char raw text.
        assert len(ln) < 500


def test_rendering_is_deterministic():
    result = {"hits": [
        _hit("alice's first fact"),
        _hit("alice's second fact", valid_to="2026-01-01T00:00:00+00:00"),
    ]}
    first = render_recall_block(result)
    second = render_recall_block(result)
    assert first == second


def test_current_hit_valid_to_none_never_marked_superseded():
    result = {"hits": [_hit("alice's evergreen preference", valid_to=None)]}
    block = render_recall_block(result)
    assert "superseded" not in block
    assert "supersedes" not in block


def test_hit_with_future_valid_to_not_marked_superseded():
    result = {"hits": [_hit("alice's still-current fact",
                             valid_to="2099-01-01T00:00:00+00:00")]}
    block = render_recall_block(result)
    assert "superseded" not in block

from __future__ import annotations


TRAILER = "[... payload truncated to fit Claude Code 10000-char limit ...]"

def test_input_above_cap_truncated_with_trailer():
    from iai_mcp.cli import _truncate_for_claude_code_hook

    out = _truncate_for_claude_code_hook("A" * 50000, cap=10000)
    assert len(out) <= 10000
    assert out.endswith(TRAILER)

def test_input_below_cap_unchanged():
    from iai_mcp.cli import _truncate_for_claude_code_hook

    body = "A" * 100
    assert _truncate_for_claude_code_hook(body, cap=10000) == body

def test_empty_input_unchanged():
    from iai_mcp.cli import _truncate_for_claude_code_hook

    assert _truncate_for_claude_code_hook("", cap=10000) == ""

def test_input_exactly_at_cap_unchanged():
    from iai_mcp.cli import _truncate_for_claude_code_hook

    body = "A" * 10000
    out = _truncate_for_claude_code_hook(body, cap=10000)
    assert out == body
    assert len(out) == 10000

"""Capture-time budget gate rejects on count OR token overflow, distinct
reason, still captured as ordinary memory -- accepted set stays intact and
within the render budget."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from iai_mcp.capture import capture_turn
from iai_mcp.directive_budget import (
    DIRECTIVE_BUDGET_TOKENS,
    DIRECTIVE_LINE_CHAR_CAP,
    DIRECTIVE_MAX_COUNT,
    projected_directive_tokens,
)
from iai_mcp.store import MemoryStore

_FILLER_WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet "


def _near_cap_text(tag: str) -> str:
    prefix = f"standing order {tag}: "
    body = (_FILLER_WORDS * 20)[: DIRECTIVE_LINE_CHAR_CAP - len(prefix)]
    text = prefix + body
    assert len(text) == DIRECTIVE_LINE_CHAR_CAP
    return text


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "lancedb")


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_count_boundary_rejects_eleventh_directive(driver, store, monkeypatch):
    _select_driver(driver, monkeypatch)

    accepted_ids = []
    for i in range(DIRECTIVE_MAX_COUNT):
        result = capture_turn(
            store=store, cue="c",
            text=f"standing order number {i}: keep answers under five sentences",
            directive=True, session_id="s1", role="user",
        )
        assert result["status"] == "inserted", result
        rec = store.get(UUID(result["record_id"]))
        assert rec is not None
        assert rec.directive is True
        accepted_ids.append(rec.id)

    overflow_result = capture_turn(
        store=store, cue="c",
        text="the eleventh standing order that should overflow the count cap",
        directive=True, session_id="s1", role="user",
    )
    assert overflow_result["status"] == "inserted", overflow_result
    overflow_rec = store.get(UUID(overflow_result["record_id"]))
    assert overflow_rec is not None
    assert overflow_rec.directive is False
    assert "count cap" in overflow_result["reason"], overflow_result

    for rid in accepted_ids:
        rec = store.get(rid)
        assert rec is not None
        assert rec.directive is True


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_token_boundary_rejects_overflow_while_count_stays_under_cap(driver, store, monkeypatch):
    """Fewer than DIRECTIVE_MAX_COUNT live directives, but their projected
    rendered tokens already sit near DIRECTIVE_BUDGET_TOKENS: one more
    near-max-length directive must be rejected on TOKEN grounds, proving the
    gate bounds tokens independently of count."""
    _select_driver(driver, monkeypatch)

    n_seed = 9
    assert n_seed < DIRECTIVE_MAX_COUNT

    accepted_texts: list[str] = []
    accepted_ids = []
    for i in range(n_seed):
        text = _near_cap_text(f"seed{i:02d}")
        result = capture_turn(
            store=store, cue="c", text=text,
            directive=True, session_id="s1", role="user",
        )
        assert result["status"] == "inserted", result
        rec = store.get(UUID(result["record_id"]))
        assert rec is not None
        assert rec.directive is True
        accepted_texts.append(text)
        accepted_ids.append(rec.id)

    assert projected_directive_tokens(accepted_texts) <= DIRECTIVE_BUDGET_TOKENS

    overflow_text = _near_cap_text("overflow")
    overflow_result = capture_turn(
        store=store, cue="c", text=overflow_text,
        directive=True, session_id="s1", role="user",
    )
    assert overflow_result["status"] == "inserted", overflow_result
    overflow_rec = store.get(UUID(overflow_result["record_id"]))
    assert overflow_rec is not None
    assert overflow_rec.directive is False
    assert "token cap" in overflow_result["reason"], overflow_result

    for rid in accepted_ids:
        rec = store.get(rid)
        assert rec is not None
        assert rec.directive is True

    assert projected_directive_tokens(accepted_texts) <= DIRECTIVE_BUDGET_TOKENS, (
        "the accepted set must stay within the render budget -- no downstream truncation possible"
    )

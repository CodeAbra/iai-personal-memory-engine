"""Injected memory is advice with explicit staleness signals: every hint
carries its age, a fact that already replaced earlier beliefs carries a
revision marker, and both the pack and the session payload invite the agent
to re-verify — an old or volatile memory must prompt a fresh search, never
substitute for one.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from iai_mcp import foresight
from iai_mcp.capture import capture_turn
from iai_mcp.store import MemoryStore, flush_record_buffer

_NOW = datetime(2026, 7, 27, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _foresight_floor(monkeypatch):
    monkeypatch.setenv(foresight.FORESIGHT_MIN_COS_ENV, "0.45")
    monkeypatch.setenv(foresight.FORESIGHT_GOAL_WEIGHT_ENV, "0")


@pytest.fixture(autouse=True)
def _fresh_working_tier():
    from iai_mcp import working_tier

    working_tier._reset()
    yield
    working_tier._reset()


def _turn(store, text, session):
    result = capture_turn(
        store, cue="", text=text, tier="episodic",
        session_id=session, role="user", live_turn=True,
    )
    flush_record_buffer(store)
    return result


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        (timedelta(seconds=10), "1m"),
        (timedelta(minutes=5), "5m"),
        (timedelta(hours=7), "7h"),
        (timedelta(days=1), "24h"),
        (timedelta(days=3), "3d"),
        (timedelta(days=30), "4w"),
        (timedelta(days=120), "3mo"),
        (timedelta(days=800), "2y"),
    ],
)
def test_age_label_single_unit(delta, expected):
    assert foresight.age_label(_NOW - delta, _NOW) == expected


def test_age_label_refuses_future_and_garbage():
    assert foresight.age_label(_NOW + timedelta(days=1), _NOW) == ""
    assert foresight.age_label(None, _NOW) == ""
    assert foresight.age_label("not a date", _NOW) == ""


def test_pack_hint_carries_relative_age(tmp_path):
    store = MemoryStore(path=tmp_path)
    _turn(store, "The hippocampus consolidates memories during sleep.", "yesterday")

    _turn(store, "How does sleep consolidation strengthen hippocampal memory?", "today")

    body = foresight.pack_path(store, "today").read_text(encoding="utf-8")
    hint = next(l for l in body.splitlines() if "hippocampus consolidates" in l)
    assert re.search(r"\(\d+(m|h|d|w|mo|y) ago\) · cos", hint), hint


def test_pack_marks_corrector_as_revised(tmp_path):
    store = MemoryStore(path=tmp_path)
    stale = _turn(
        store, "The vector project stores embeddings in the plum database.", "day1"
    )
    current = _turn(
        store,
        "Correction: the vector project stores embeddings in the pearl database.",
        "day2",
    )
    store.add_contradicts_edge(
        UUID(stale["record_id"]), UUID(current["record_id"])
    )

    _turn(store, "Which database does the vector project store embeddings in?", "today")

    body = foresight.pack_path(store, "today").read_text(encoding="utf-8")
    current_lines = [l for l in body.splitlines() if "pearl database" in l]
    assert current_lines, body
    # The stale belief travels only under the superseded flag; the corrector
    # itself is a fact that already changed once — it must carry ↻1.
    plain = [l for l in current_lines if "superseded" not in l]
    if plain:
        assert any("↻1 ·" in l for l in plain), current_lines
    superseded = [l for l in body.splitlines() if "superseded belief" in l]
    if superseded:
        assert re.search(r"current \(\d+(m|h|d|w|mo|y) ago\):", superseded[0]), (
            superseded[0]
        )
    assert plain or superseded, body


def test_pack_footer_carries_volatility_advice(tmp_path):
    store = MemoryStore(path=tmp_path)
    _turn(store, "The hippocampus consolidates memories during sleep.", "yesterday")

    _turn(store, "How does sleep consolidation strengthen hippocampal memory?", "today")

    body = foresight.pack_path(store, "today").read_text(encoding="utf-8")
    assert "volatility signals" in body
    assert "re-verification" in body


def test_recent_thread_lines_carry_age(tmp_path):
    from iai_mcp.session import _recent_thread_segment

    store = MemoryStore(path=tmp_path)
    _turn(store, "Working through the migration script for the archive.", "s1")

    seg = _recent_thread_segment(store)
    assert seg, "captured turn must surface in the recent thread"
    assert re.match(r"- \(\d+(m|h|d|w|mo|y)\) ", seg.splitlines()[0]), seg


def test_rich_club_lines_carry_age(tmp_path):
    from iai_mcp.session import _rich_club_segment

    store = MemoryStore(path=tmp_path)
    rec = _turn(store, "The archive migration runs every Sunday night.", "s1")

    seg = _rich_club_segment(store, [UUID(rec["record_id"])])
    assert seg, "rich-club member must render"
    assert re.search(r" \(\d+(m|h|d|w|mo|y)\): ", seg), seg


def test_payload_markdown_advisory_footer_only_with_content():
    from iai_mcp.session import format_payload_as_markdown

    md = format_payload_as_markdown(
        {"l0": "identity", "l1": "", "l2": [], "rich_club": "", "recent_thread": ""}
    )
    assert "re-verification" in md
    assert "starting points" in md

    empty = format_payload_as_markdown(
        {"l0": "", "l1": "", "l2": [], "rich_club": "", "recent_thread": ""}
    )
    assert empty == ""

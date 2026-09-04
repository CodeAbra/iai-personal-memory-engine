"""Pure-function unit coverage for `_payload_created_at`'s unknown-input
branches, `_age_penalty`'s None-guard, and `matches_mentions`'s existing
None handling. Touches no store -- driver-agnostic, must not skip when
`iai_mcp_native` is absent.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from iai_mcp.pipeline import AGE_HALF_LIFE_DAYS, _age_penalty, _payload_created_at
from iai_mcp.temporal_cue import DateMention, matches_mentions


def test_payload_created_at_none_returns_none():
    assert _payload_created_at(None) is None


def test_payload_created_at_empty_string_returns_none():
    assert _payload_created_at("") is None


def test_payload_created_at_unparseable_string_returns_none():
    assert _payload_created_at("not-a-timestamp") is None


def test_payload_created_at_real_tz_aware_datetime_unchanged():
    dt = datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc)
    assert _payload_created_at(dt) == dt


def test_payload_created_at_real_naive_datetime_coerced_to_utc():
    naive = datetime(2024, 1, 15, 12, 30)
    result = _payload_created_at(naive)
    assert result == naive.replace(tzinfo=timezone.utc)
    assert result.tzinfo is timezone.utc


def test_payload_created_at_valid_isoformat_string_parsed():
    iso = "2024-01-15T12:30:00+00:00"
    result = _payload_created_at(iso)
    assert result == datetime.fromisoformat(iso)


def test_age_penalty_none_returns_zero():
    assert _age_penalty(None) == 0.0


def test_age_penalty_real_distinctly_past_datetime_unchanged():
    # AGE_HALF_LIFE_DAYS=30.0, W_AGE applied elsewhere -- a date roughly one
    # half-life old lands near 0.5, numerically distinguishable from the
    # None-branch's 0.0.
    now = datetime.now(timezone.utc)
    half_life_ago = now - timedelta(days=AGE_HALF_LIFE_DAYS / 2)
    penalty = _age_penalty(half_life_ago)
    assert 0.4 < penalty < 0.6, f"expected penalty near 0.5, got {penalty}"


def test_matches_mentions_none_created_at_returns_false():
    mention = DateMention(year=2024, month=1, day=15)
    assert matches_mentions(None, [mention]) is False

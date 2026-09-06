"""Unknown/unparseable created_at and updated_at must decode as maximally old.

Covers the two decode tiers that share the same fallback idiom: MemoryStore's
plain-record decode (``_from_row``) and its rank-only decode
(``_from_row_rank_view``). Both must substitute the epoch-min sentinel for a
missing or unparseable timestamp, never a value near "now" -- a corrupt row
must never be able to rank as freshly captured.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.store import MemoryStore

EPOCH_MIN = datetime.min.replace(tzinfo=timezone.utc)


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


@pytest.fixture
def store(driver, tmp_path):
    return MemoryStore(path=tmp_path)


def _minimal_row(**overrides) -> dict:
    row: dict = {"id": str(uuid4())}
    row.update(overrides)
    return row


def test_from_row_missing_created_at_returns_epoch_min_sentinel(store):
    rec = store._from_row(_minimal_row(created_at=None))
    assert rec.created_at == EPOCH_MIN


def test_from_row_unparseable_created_at_returns_epoch_min_sentinel(store):
    rec = store._from_row(_minimal_row(created_at="not-a-timestamp"))
    assert rec.created_at == EPOCH_MIN


def test_from_row_missing_updated_at_returns_epoch_min_sentinel(store):
    rec = store._from_row(_minimal_row(updated_at=None))
    assert rec.updated_at == EPOCH_MIN


def test_from_row_unparseable_updated_at_returns_epoch_min_sentinel(store):
    rec = store._from_row(_minimal_row(updated_at="also-not-a-timestamp"))
    assert rec.updated_at == EPOCH_MIN


def test_from_row_real_created_at_decodes_unchanged(store):
    real = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rec = store._from_row(
        _minimal_row(created_at=real.isoformat(), updated_at=real.isoformat())
    )
    assert rec.created_at == real
    assert rec.updated_at == real


def test_from_row_rank_view_missing_created_at_returns_epoch_min_sentinel(store):
    rv = store._from_row_rank_view(_minimal_row(created_at=None))
    assert rv.created_at == EPOCH_MIN


def test_from_row_rank_view_unparseable_created_at_returns_epoch_min_sentinel(store):
    rv = store._from_row_rank_view(_minimal_row(created_at="garbage"))
    assert rv.created_at == EPOCH_MIN


def test_from_row_rank_view_real_created_at_decodes_unchanged(store):
    real = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rv = store._from_row_rank_view(_minimal_row(created_at=real.isoformat()))
    assert rv.created_at == real


def test_unknown_sentinel_sorts_as_oldest_in_most_recent_first_order(store):
    real = datetime(2026, 1, 1, tzinfo=timezone.utc)
    fresh = store._from_row(_minimal_row(created_at=real.isoformat()))
    unknown = store._from_row(_minimal_row(created_at=None))

    cands = [unknown, fresh]
    cands.sort(key=lambda r: r.created_at or EPOCH_MIN, reverse=True)

    assert cands[0] is fresh
    assert cands[-1] is unknown


def test_unknown_sentinel_rank_view_sorts_as_oldest_in_most_recent_first_order(store):
    real = datetime(2026, 1, 1, tzinfo=timezone.utc)
    fresh = store._from_row_rank_view(_minimal_row(created_at=real.isoformat()))
    unknown = store._from_row_rank_view(_minimal_row(created_at=None))

    cands = [unknown, fresh]
    cands.sort(key=lambda r: r.created_at or EPOCH_MIN, reverse=True)

    assert cands[0] is fresh
    assert cands[-1] is unknown


def test_sentinel_age_penalty_stays_finite_and_clamped():
    from iai_mcp.pipeline import _age_penalty

    penalty = _age_penalty(EPOCH_MIN)
    assert math.isfinite(penalty)
    assert penalty == 1.0

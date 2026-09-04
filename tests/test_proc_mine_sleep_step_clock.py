"""PROC_MINE sleep step routes decay through the pipeline clock.

step_proc_mine must call decay_proc_chunks(..., now=self._now()) like every
other timestamp-sensitive sibling step, so a monkeypatched pipeline clock
reaches decay's AND-gated age/staleness window.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pandas as pd
import pytest

from iai_mcp.lifecycle_state import default_state, save_state
from iai_mcp.lilli.cycle import sleep_pipeline as _sleep_pipeline_pkg
from iai_mcp.lilli.cycle.chunk import persist_proc_chunk
from iai_mcp.lilli.cycle.proc_mine import MIN_DISTINCT_SESSIONS, PAIR_COUNT_FLOOR, CofirePairCandidate
from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline
from iai_mcp.store import RECORDS_TABLE, MemoryStore

# Timestamps track the real wall clock at test-run time, not a fixed
# calendar date -- a hardcoded past mint date eventually drifts far enough
# behind real time that decay's own datetime.now() fallback would retire the
# chunk too, silently defeating the RED/GREEN differential.
_REAL_NOW = datetime.now(timezone.utc)
_MINT_TS = _REAL_NOW - timedelta(days=10)
_FUTURE_NOW = _REAL_NOW + timedelta(days=250)  # clears both decay windows


def _candidate(pair: tuple[str, str]) -> CofirePairCandidate:
    return CofirePairCandidate(
        pair=pair,
        source="retrieval_cofired",
        count=PAIR_COUNT_FLOOR,
        session_count=MIN_DISTINCT_SESSIONS,
        sessions=frozenset({"s1", "s2", "s3"}),
        first_ts=_MINT_TS,
        last_ts=_MINT_TS + timedelta(minutes=5),
    )


def _backdate(store: MemoryStore, record_id, *, created_at: datetime, last_reviewed: datetime) -> None:
    tbl = store.db.open_table(RECORDS_TABLE)
    tbl.update(
        where=f"id = '{record_id}'",
        values={
            "created_at": created_at.isoformat(),
            "last_reviewed": last_reviewed.isoformat(),
        },
    )


def _row_for(df: pd.DataFrame, rid) -> dict | None:
    sub = df[df["id"] == str(rid)]
    return None if sub.empty else sub.iloc[0].to_dict()


def _is_tombstoned(row: dict) -> bool:
    val = row.get("tombstoned_at")
    return val is not None and not pd.isna(val)


def _plant_stale_chunk(store: MemoryStore, pair: tuple[str, str]) -> UUID:
    chunk_id = persist_proc_chunk(store, _candidate(pair))
    assert chunk_id is not None
    _backdate(store, chunk_id, created_at=_MINT_TS, last_reviewed=_MINT_TS)
    return chunk_id


def _run_step(store: MemoryStore, tmp_path: Path, label: str) -> dict:
    lifecycle_path = tmp_path / f"lifecycle-{label}.json"
    save_state(default_state(), lifecycle_path)
    pipeline = SleepPipeline(store=store, lifecycle_state_path=lifecycle_path)

    done, payload = pipeline._step_proc_mine(interrupt_check=None)
    assert done is True
    return payload


def _tombstoned_row(store: MemoryStore, chunk_id: UUID) -> dict:
    tbl = store.db.open_table(RECORDS_TABLE)
    row = _row_for(tbl.to_pandas(), chunk_id)
    assert row is not None, "decay never deletes rows, only tombstones them"
    return row


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_monkeypatched_pipeline_clock_reaches_decay(tmp_path, monkeypatch, driver):
    """The pipeline's _now() must be the clock decay actually evaluates
    against. Two legs on the same fixture pin the differential: under the
    real (unpatched) clock the chunk is only 10 days old and must survive;
    monkeypatching the pipeline clock 250 days forward must retire it --
    without now=self._now() wired through, decay would fall back to real
    wall-clock time and neither leg would move."""
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built -- lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)

    store = MemoryStore(path=tmp_path / "operator-home" / ".iai-mcp")

    control_id = _plant_stale_chunk(store, ("alice", "bob"))
    control_payload = _run_step(store, tmp_path, "control")
    control_row = _tombstoned_row(store, control_id)
    assert not _is_tombstoned(control_row), (
        "control leg: a 10-day-old chunk must survive under the real clock"
    )
    assert bool(control_row["live"])
    assert control_payload["chunks_persisted"] == 0

    future_id = _plant_stale_chunk(store, ("carol", "dave"))
    monkeypatch.setattr(_sleep_pipeline_pkg, "_utc_now", lambda: _FUTURE_NOW)
    future_payload = _run_step(store, tmp_path, "future")
    future_row = _tombstoned_row(store, future_id)
    assert _is_tombstoned(future_row), (
        "the pipeline clock was monkeypatched past both decay windows, so "
        "the backdated chunk must retire -- decay is reading real "
        "wall-clock time instead of self._now()"
    )
    assert not bool(future_row["live"])
    assert future_payload["chunks_persisted"] == 0

"""A concurrent WAL checkpoint fences an open read-only snapshot mid-scan.

The streaming drain must survive: reopen a fresh snapshot, re-execute, and
serve every row exactly once (dedup by id). A scan without an id column that
already served rows must fail loud rather than silently re-yield.
"""
from __future__ import annotations

import sqlite3

from iai_mcp import errors

import pytest

from iai_mcp.daemon import _sleep_backoff_active
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.capture import capture_turn


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401, PLC0415
        except ImportError:
            pytest.skip("iai_mcp_native not built")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _seed_records(store: MemoryStore, n: int) -> None:
    for i in range(n):
        capture_turn(
            store, cue="", text=f"fence recovery record number {i} unique",
            tier="episodic", session_id="s", role="user",
        )
    flush_record_buffer(store)


class _FencingConn:
    """Snapshot-conn stand-in that dies with the fence error after serving
    a fixed number of fetchmany batches (0 = dies at execute time)."""

    def __init__(self, real, die_after_batches: int):
        self._real = real
        self._die_after = die_after_batches

    def execute(self, sql, *args):
        if self._die_after == 0:
            raise errors.OperationalError(
                "storage error: integrity violation: read-only snapshot "
                "invalidated: main file size changed under a concurrent "
                "checkpoint"
            )
        return _FencingCursor(self._real.execute(sql, *args), self._die_after)

    def close(self):
        self._real.close()

    def __getattr__(self, name):
        return getattr(self._real, name)


class _FencingCursor:
    def __init__(self, real, die_after_batches: int):
        self._real = real
        self._left = die_after_batches
        self.description = real.description

    def fetchmany(self, n):
        if self._left <= 0:
            raise errors.OperationalError(
                "storage error: integrity violation: read-only snapshot "
                "invalidated: main file size changed under a concurrent "
                "checkpoint"
            )
        self._left -= 1
        return self._real.fetchmany(n)

    def close(self):
        self._real.close()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
@pytest.mark.parametrize("die_after", [0, 1])
def test_fenced_scan_recovers_exactly_once(driver, die_after, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    _seed_records(store, 7)
    from iai_mcp.hippo._table import HippoQuery

    real_open = HippoQuery._open_snapshot_conn
    fenced_once = {"done": False}

    def fenced_open(self, db_path):
        real = real_open(self, db_path)
        if fenced_once["done"]:
            return real
        fenced_once["done"] = True
        return _FencingConn(real, die_after)

    monkeypatch.setattr(HippoQuery, "_open_snapshot_conn", fenced_open)

    from iai_mcp.store import RECORDS_TABLE

    q = store.db.open_table(RECORDS_TABLE).search().select(["id", "tags_json"])
    rows = [r for batch in q.to_batches(batch_size=3) for r in batch.to_pylist()]
    ids = [r["id"] for r in rows]
    assert fenced_once["done"], "the fenced snapshot path must have been hit"
    assert len(ids) == 7, f"[{driver}] every record exactly once, got {len(ids)}"
    assert len(set(ids)) == 7, f"[{driver}] duplicates after fence recovery"
    store.close()


@pytest.mark.parametrize("driver", ["stdlib"])
def test_fenced_scan_without_id_fails_loud_after_rows_served(
    driver, tmp_path, monkeypatch
):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    _seed_records(store, 7)
    from iai_mcp.hippo._table import HippoQuery

    real_open = HippoQuery._open_snapshot_conn

    def always_fenced_open(self, db_path):
        return _FencingConn(real_open(self, db_path), 1)

    monkeypatch.setattr(HippoQuery, "_open_snapshot_conn", always_fenced_open)

    from iai_mcp.store import RECORDS_TABLE

    q = store.db.open_table(RECORDS_TABLE).search().select(["tags_json"])
    with pytest.raises(errors.OperationalError, match="snapshot invalidated"):
        [r for batch in q.to_batches(batch_size=3) for r in batch.to_pylist()]
    store.close()


def test_sleep_backoff_blocks_and_force_rem_overrides():
    assert _sleep_backoff_active({}, backoff_until=100.0, now=50.0) is True
    assert _sleep_backoff_active({}, backoff_until=100.0, now=150.0) is False
    assert _sleep_backoff_active(None, backoff_until=100.0, now=50.0) is True
    assert (
        _sleep_backoff_active(
            {"force_rem_request": {"pending": True}}, backoff_until=100.0, now=50.0
        )
        is False
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_backlog_drain_sweeps_marathon_live_spools(driver, tmp_path, monkeypatch):
    """A session that never ends must not hoard its memory on disk: the
    composed drain folds still-open live spools in incrementally (offset-
    tracked), so a second pass re-inserts nothing."""
    import json as _json
    from pathlib import Path

    from iai_mcp.capture import drain_capture_backlog

    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    spool_dir = Path.home() / ".iai-mcp" / ".deferred-captures"
    spool_dir.mkdir(parents=True, exist_ok=True)
    spool = spool_dir / "marathon-session.live.jsonl"
    header = {"version": 1, "session_id": "marathon-session",
              "deferred_at": "2026-07-07T10:00:00+00:00", "cwd": "/tmp"}
    events = [
        {"cue": "", "text": "Marathon sessions still deserve fresh memory.",
         "tier": "episodic", "role": "user"},
        {"cue": "", "text": "The drowsy pause is when turns fold in.",
         "tier": "episodic", "role": "assistant"},
    ]
    spool.write_text(
        "\n".join(_json.dumps(x) for x in [header, *events]) + "\n",
        encoding="utf-8",
    )

    first = drain_capture_backlog(store)
    assert first.get("live_events_inserted", 0) == 2, first
    flush_record_buffer(store)
    row = store.db._conn.execute(
        "SELECT COUNT(*) FROM records WHERE tombstoned_at IS NULL"
    ).fetchone()
    assert int(row[0]) == 2

    again = drain_capture_backlog(store)
    assert again.get("live_events_inserted", 0) == 0, (
        f"[{driver}] offset tracking must make the sweep incremental: {again}"
    )
    store.close()


def test_drain_rss_threshold_is_growth_relative():
    """A daemon whose standing footprint exceeds the absolute floor must
    still be allowed to drain: the stop threshold is the HIGHER of the
    configured floor and baseline + growth budget."""
    from iai_mcp.capture import (
        DRAIN_RSS_GROWTH_BUDGET_DEFAULT_BYTES,
        DRAIN_RSS_SOFT_CAP_DEFAULT_BYTES,
        _drain_rss_stop_threshold,
    )

    small = 500 * 1024 * 1024
    assert _drain_rss_stop_threshold(small) == max(
        DRAIN_RSS_SOFT_CAP_DEFAULT_BYTES,
        small + DRAIN_RSS_GROWTH_BUDGET_DEFAULT_BYTES,
    )
    big = 3 * 1024 * 1024 * 1024
    assert (
        _drain_rss_stop_threshold(big)
        == big + DRAIN_RSS_GROWTH_BUDGET_DEFAULT_BYTES
    ), "a big-but-legitimate baseline must not silence the drain"


def test_live_compaction_trims_even_when_daemon_holds_writer(tmp_path, monkeypatch):
    """Write-side cap bounds a marathon .live.jsonl even while the daemon owns
    the exclusive writer: the store-open fails (daemon alive) but the offset-
    based trim still runs on the lines the daemon already promoted."""
    import json as _json

    import iai_mcp.capture as cap

    monkeypatch.setenv("HOME", str(tmp_path))
    deferred = tmp_path / ".iai-mcp" / ".deferred-captures"
    state = tmp_path / ".iai-mcp" / ".capture-state"
    deferred.mkdir(parents=True)
    state.mkdir(parents=True)

    sid = "marathon"
    live = deferred / f"{sid}.live.jsonl"
    header = {"version": 1, "session_id": sid, "deferred_at": "t", "cwd": "/tmp"}
    lines = [_json.dumps(header)] + [
        _json.dumps({"text": f"turn {i} of a session that never ends"})
        for i in range(6000)
    ]
    live.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (state / f"{sid}.drain-offset").write_text("5500")

    def boom(*a, **k):
        raise RuntimeError("store is locked: daemon owns the writer")
    # the function does a local `from iai_mcp.store import MemoryStore`, so
    # the patch must target the source symbol, not capture's namespace
    import iai_mcp.store as _store_mod
    monkeypatch.setattr(_store_mod, "MemoryStore", boom)

    cap._compact_live_file_if_oversized(live, sid)
    after = live.read_text(encoding="utf-8").splitlines()
    assert len(after) == 1 + (6000 - 5500), (
        f"trim must run despite the daemon-held store open: got {len(after)}"
    )
    assert _json.loads(after[0])["session_id"] == sid


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_iter_record_columns_keyset_resumes_on_fence(driver, tmp_path, monkeypatch):
    """The durable fix: a mid-scan checkpoint fence must not restart the scan
    from row 1. Keyset pagination resumes from the last id, so every record is
    yielded exactly once and the scan completes even under repeated fencing."""
    import sqlite3

    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    _seed_records(store, 25)

    real_open = MemoryStore._open_keyset_snapshot
    fence_budget = {"n": 3}  # fence the first 3 page queries

    class _FenceOnce:
        def __init__(self, real):
            self._real = real
        def execute(self, sql, *a):
            if fence_budget["n"] > 0 and "id > ?" in sql:
                fence_budget["n"] -= 1
                raise errors.OperationalError(
                    "storage error: integrity violation: read-only snapshot "
                    "invalidated: main file size changed under a concurrent "
                    "checkpoint")
            return self._real.execute(sql, *a)
        def close(self):
            self._real.close()
        def __getattr__(self, n):
            return getattr(self._real, n)
        def __setattr__(self, n, v):
            if n == "_real":
                object.__setattr__(self, n, v)
            else:
                setattr(self._real, n, v)

    def fenced_open(self, db_path):
        return _FenceOnce(real_open(self, db_path))
    monkeypatch.setattr(MemoryStore, "_open_keyset_snapshot", fenced_open)

    rows = list(store.iter_record_columns(["id", "tags_json"], batch_size=5))
    ids = [r["id"] for r in rows]
    assert len(ids) == 25, f"[{driver}] every record once, got {len(ids)}"
    assert len(set(ids)) == 25, f"[{driver}] duplicates after fence resume"
    assert fence_budget["n"] == 0, "the fence path must have been exercised"

    # a projection WITHOUT id must not leak the cursor column
    rows2 = list(store.iter_record_columns(["tags_json"], batch_size=5))
    assert len(rows2) == 25 and "id" not in rows2[0]
    store.close()

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock



def _clear_buffer(store) -> None:
    from iai_mcp import events

    events._event_buffer.pop(id(store), None)
    events._last_flush_at.pop(id(store), None)


def test_write_event_buffered_does_not_write_to_lancedb(tmp_path):
    from iai_mcp import events
    from iai_mcp.events import write_event
    from iai_mcp.store import EVENTS_TABLE, MemoryStore

    with MemoryStore(path=tmp_path) as store:
        _clear_buffer(store)

        tbl = store.db.open_table(EVENTS_TABLE)
        n_before = len(tbl.to_pandas())

        event_id = write_event(store, kind="test_buf", data={"x": 1}, buffered=True)
        assert event_id is not None

        tbl = store.db.open_table(EVENTS_TABLE)
        n_after = len(tbl.to_pandas())
        assert n_after == n_before, (
            f"buffered=True wrote to LanceDB: {n_before} -> {n_after}"
        )

        assert len(events._event_buffer.get(id(store), [])) == 1


def test_flush_event_buffer_writes_batch_and_clears(tmp_path):
    from iai_mcp import events
    from iai_mcp.events import flush_event_buffer, write_event
    from iai_mcp.store import EVENTS_TABLE, MemoryStore

    with MemoryStore(path=tmp_path) as store:
        _clear_buffer(store)

        tbl = store.db.open_table(EVENTS_TABLE)
        n_before = len(tbl.to_pandas())

        for i in range(3):
            write_event(store, kind="batch_flush", data={"i": i}, buffered=True)

        assert len(events._event_buffer.get(id(store), [])) == 3

        flushed = flush_event_buffer(store)
        assert flushed == 3

        assert not events._event_buffer.get(id(store))

        tbl = store.db.open_table(EVENTS_TABLE)
        n_after = len(tbl.to_pandas())
        assert n_after == n_before + 3


def test_flush_event_buffer_failure_logs_and_doesnt_raise(tmp_path, caplog):
    from iai_mcp import events
    from iai_mcp.events import flush_event_buffer, write_event
    from iai_mcp.store import MemoryStore

    with MemoryStore(path=tmp_path) as store:
        _clear_buffer(store)

        write_event(store, kind="will_fail", data={"i": 0}, buffered=True)
        write_event(store, kind="will_fail", data={"i": 1}, buffered=True)
        assert len(events._event_buffer.get(id(store), [])) == 2

        real_open_table = store.db.open_table

        def _raising(name):
            tbl = real_open_table(name)
            mock = MagicMock(wraps=tbl)
            mock.add.side_effect = RuntimeError("simulated lance failure")
            return mock

        store.db.open_table = _raising

        with caplog.at_level(logging.WARNING, logger="iai_mcp.events"):
            flushed = flush_event_buffer(store)
            assert flushed == 2

        msgs = [r.message for r in caplog.records if r.name == "iai_mcp.events"]
        assert any("flush_event_buffer_failed" in m for m in msgs), (
            f"expected flush_event_buffer_failed warning; got: {msgs}"
        )


def test_should_flush_size_threshold(tmp_path, monkeypatch):
    from iai_mcp.events import should_flush, write_event
    from iai_mcp.store import MemoryStore

    with MemoryStore(path=tmp_path) as store:
        _clear_buffer(store)

        monkeypatch.setenv("IAI_MCP_EVENT_BUFFER_MAX", "10")

        assert should_flush(id(store)) is False

        for i in range(9):
            write_event(store, kind="sz", data={"i": i}, buffered=True)
        assert should_flush(id(store)) is False

        write_event(store, kind="sz", data={"i": 9}, buffered=True)
        assert should_flush(id(store)) is True

        assert should_flush(id(store), max_size=100) is False


def test_should_flush_time_threshold(tmp_path):
    from iai_mcp.events import should_flush_by_time, write_event
    from iai_mcp.store import MemoryStore

    with MemoryStore(path=tmp_path) as store:
        _clear_buffer(store)

        assert should_flush_by_time(id(store), None) is False
        assert should_flush_by_time(id(store), datetime.now(timezone.utc) - timedelta(seconds=60)) is False

        write_event(store, kind="tm", data={"i": 0}, buffered=True)

        assert should_flush_by_time(id(store), None) is True

        recent = datetime.now(timezone.utc) - timedelta(seconds=1)
        assert should_flush_by_time(id(store), recent) is False

        old = datetime.now(timezone.utc) - timedelta(seconds=6)
        assert should_flush_by_time(id(store), old) is True


def test_store_pattern_separation_pass_uses_buffered_writes():
    store_py = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "store" / "_store.py"
    text = store_py.read_text(encoding="utf-8")

    pattern = re.compile(
        r'write_event\(\s*self,\s*"pattern_separation_pass"\s*,'
    )
    starts = [m.start() for m in pattern.finditer(text)]
    assert len(starts) == 4, (
        f"expected 4 pattern_separation_pass call sites (plan claimed 5 — drift); got {len(starts)}"
    )

    lines = text.splitlines()
    line_index = []
    cursor = 0
    for ln in lines:
        line_index.append(cursor)
        cursor += len(ln) + 1

    for s in starts:
        line_no = next(i for i, c in enumerate(line_index) if c > s) - 1
        window = "\n".join(lines[line_no : line_no + 30])
        assert "buffered=True" in window, (
            f"pattern_separation_pass call at store.py line {line_no + 1} lacks buffered=True"
        )


def test_daemon_wake_wires_flush_event_buffer():
    daemon_py = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "daemon" / "__init__.py"
    text = daemon_py.read_text(encoding="utf-8")

    assert text.count("flush_event_buffer") >= 3, (
        f"expected >= 3 flush_event_buffer references (periodic + shutdown); "
        f"found {text.count('flush_event_buffer')}"
    )
    assert "should_flush_by_time" in text, (
        "should_flush_by_time gate not found in daemon.py — per-tick time-threshold missing"
    )
    gate_idx = text.find("should_flush_by_time")
    flush_idx = text.find("flush_event_buffer", gate_idx)
    assert flush_idx > gate_idx, (
        "flush_event_buffer must appear after should_flush_by_time in daemon.py; "
        f"gate_idx={gate_idx}, flush_idx={flush_idx}"
    )
    tick_region_start = text.find("should_flush_by_time")
    tick_region_end = text.find("flush_event_buffer", tick_region_start) + len("flush_event_buffer")
    tick_region = text[max(0, tick_region_start - 200): tick_region_end + 200]
    assert "try:" in tick_region or "except" in tick_region, (
        "per-tick events flush block should be guarded by try/except"
    )


def test_daemon_periodic_tick_wires_should_flush_by_time():
    daemon_py = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "daemon" / "__init__.py"
    text = daemon_py.read_text(encoding="utf-8")

    assert "should_flush_by_time" in text, (
        "periodic-tick wiring uses should_flush_by_time helper — missing"
    )

    import tempfile

    from iai_mcp.events import should_flush_by_time, write_event
    from iai_mcp.store import MemoryStore

    with tempfile.TemporaryDirectory() as td:
        with MemoryStore(path=Path(td)) as store:
            _clear_buffer(store)

            write_event(store, kind="tk", data={"i": 0}, buffered=True)
            assert should_flush_by_time(
                id(store), datetime.now(timezone.utc) - timedelta(seconds=6)
            ) is True

            assert should_flush_by_time(
                id(store), datetime.now(timezone.utc) - timedelta(seconds=1)
            ) is False


def test_daemon_shutdown_wires_flush_event_buffer_sync(tmp_path):
    daemon_py = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "daemon" / "__init__.py"
    text = daemon_py.read_text(encoding="utf-8")

    assert "flush_event_buffer" in text, "shutdown flush missing"

    from iai_mcp.events import flush_event_buffer, write_event
    from iai_mcp.store import EVENTS_TABLE, MemoryStore

    with MemoryStore(path=tmp_path) as store:
        _clear_buffer(store)

        tbl = store.db.open_table(EVENTS_TABLE)
        n_before = len(tbl.to_pandas())

        write_event(store, kind="shutdown_test", data={"i": 0}, buffered=True)
        write_event(store, kind="shutdown_test", data={"i": 1}, buffered=True)

        flushed = flush_event_buffer(store)
        assert flushed == 2

        tbl = store.db.open_table(EVENTS_TABLE)
        assert len(tbl.to_pandas()) == n_before + 2

"""Unit coverage for the watchdog resident-set breadcrumb.

The watchdog tick emits a throttled JSON breadcrumb (RSS + vmmap region count +
VM_ALLOCATE) to the watchdog log: every tick (30 s) during SLEEP, every ~10th
tick (300 s) otherwise. The JSON breadcrumb lines coexist with the watchdog's
space-delimited kill-lines without corrupting recovery-timestamp parsing, and a
breadcrumb failure can never crash the tick.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from iai_mcp.daemon import _watchdog
from iai_mcp.events import DAEMON_MEMORY_PRESSURE_KILL
from iai_mcp.lifecycle_state import LifecycleState


def _pkg():
    import sys

    return sys.modules["iai_mcp.daemon"]


@pytest.fixture
def breadcrumb_log(tmp_path: Path, monkeypatch):
    """Open a real FD on a tmp watchdog log and reset the throttle clock."""
    log_path = tmp_path / ".daemon-watchdog.log"
    fd = os.open(str(log_path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setattr(_pkg(), "_WATCHDOG_LOG_FD", fd, raising=False)
    monkeypatch.setattr(_pkg(), "_last_rss_breadcrumb_at", 0.0, raising=False)
    try:
        yield log_path
    finally:
        os.close(fd)


def _read_breadcrumbs(log_path: Path) -> list[dict]:
    rows: list[dict] = []
    if not log_path.exists():
        return rows
    for raw in log_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw.startswith("{"):
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if row.get("kind") == "rss_breadcrumb":
            rows.append(row)
    return rows


def test_breadcrumb_written_when_due(breadcrumb_log: Path) -> None:
    _watchdog._maybe_emit_rss_breadcrumb(
        rss=200 * 1024 * 1024,
        pressure_level=1,
        fsm_state="WAKE",
        now_mono=10_000.0,
    )
    rows = _read_breadcrumbs(breadcrumb_log)
    assert len(rows) == 1
    row = rows[0]
    for key in ("rss_kib", "vmmap_region_count", "vm_allocate_kib", "fsm_state", "ts"):
        assert key in row, f"breadcrumb missing {key!r}: {row}"
    assert row["fsm_state"] == "WAKE"
    # 200 MiB rss -> 204800 KiB.
    assert row["rss_kib"] == 200 * 1024


def test_breadcrumb_throttled_when_not_due(breadcrumb_log: Path) -> None:
    # First emit at t=10000 fires; second at t=10010 (10 s later, well inside the
    # 300 s normal interval) is throttled.
    _watchdog._maybe_emit_rss_breadcrumb(
        rss=100 * 1024 * 1024, pressure_level=0, fsm_state="WAKE", now_mono=10_000.0,
    )
    _watchdog._maybe_emit_rss_breadcrumb(
        rss=100 * 1024 * 1024, pressure_level=0, fsm_state="WAKE", now_mono=10_010.0,
    )
    rows = _read_breadcrumbs(breadcrumb_log)
    assert len(rows) == 1


def test_breadcrumb_sleep_cadence_is_faster_than_normal(breadcrumb_log: Path) -> None:
    sleep_state = LifecycleState.SLEEP.value
    # During SLEEP the interval is 30 s: a first emit at t=10000 fires, a second
    # 31 s later also fires (gap exceeds the 30 s SLEEP interval).
    _watchdog._maybe_emit_rss_breadcrumb(
        rss=100 * 1024 * 1024, pressure_level=0, fsm_state=sleep_state, now_mono=10_000.0,
    )
    _watchdog._maybe_emit_rss_breadcrumb(
        rss=100 * 1024 * 1024, pressure_level=0, fsm_state=sleep_state, now_mono=10_031.0,
    )
    assert len(_read_breadcrumbs(breadcrumb_log)) == 2

    # Reset and prove the normal interval (300 s) suppresses the same 31 s gap.
    breadcrumb_log.write_text("", encoding="utf-8")
    setattr(_pkg(), "_last_rss_breadcrumb_at", 0.0)
    _watchdog._maybe_emit_rss_breadcrumb(
        rss=100 * 1024 * 1024, pressure_level=0, fsm_state="WAKE", now_mono=10_000.0,
    )
    _watchdog._maybe_emit_rss_breadcrumb(
        rss=100 * 1024 * 1024, pressure_level=0, fsm_state="WAKE", now_mono=10_031.0,
    )
    assert len(_read_breadcrumbs(breadcrumb_log)) == 1


def test_breadcrumb_never_raises_when_vmmap_explodes(breadcrumb_log: Path, monkeypatch) -> None:
    import iai_mcp.lilli.cycle.sleep_pipeline._rss_probe as probe

    def _boom(*_a, **_k):
        raise RuntimeError("vmmap exploded")

    monkeypatch.setattr(probe, "_vmmap_summary_counts", _boom)
    # Must not propagate.
    _watchdog._maybe_emit_rss_breadcrumb(
        rss=100 * 1024 * 1024, pressure_level=0, fsm_state="WAKE", now_mono=10_000.0,
    )


def test_throttle_global_present_on_package_from_import() -> None:
    """_last_rss_breadcrumb_at must exist on the daemon package namespace from
    import — not seeded by a first successful write.  The re-export block in
    daemon/__init__.py is the mechanism; this test catches any future omission.
    """
    import importlib
    import sys

    # Re-import to reflect the current module state; do NOT rely on a fixture
    # that uses raising=False (which would silently create the attribute even if
    # it were absent from the re-export block).
    pkg = sys.modules.get("iai_mcp.daemon")
    if pkg is None:
        import iai_mcp.daemon as pkg  # type: ignore[assignment]
        pkg = sys.modules["iai_mcp.daemon"]

    assert hasattr(pkg, "_last_rss_breadcrumb_at"), (
        "_last_rss_breadcrumb_at absent from iai_mcp.daemon package namespace; "
        "add it to the _watchdog re-export block in daemon/__init__.py"
    )
    assert isinstance(pkg._last_rss_breadcrumb_at, float), (
        f"_last_rss_breadcrumb_at should be float, got {type(pkg._last_rss_breadcrumb_at)}"
    )


def test_recovery_parse_ignores_breadcrumb_lines(tmp_path: Path) -> None:
    # A mixed log: one real space-delimited kill-line + one JSON breadcrumb line.
    # _load_recovery_timestamps must return exactly the one kill timestamp.
    log_path = tmp_path / ".daemon-watchdog.log"
    kill_iso = "2026-06-20T10:00:00+00:00"
    breadcrumb = json.dumps(
        {
            "kind": "rss_breadcrumb",
            "ts": "2026-06-20T10:00:30+00:00",
            "fsm_state": "SLEEP",
            "rss_kib": 204800,
            "vmmap_region_count": 289,
            "vm_allocate_kib": 16,
            "pressure_level": 1,
        }
    )
    log_path.write_text(
        f"{kill_iso} {DAEMON_MEMORY_PRESSURE_KILL} reason=leak rss=999\n{breadcrumb}\n",
        encoding="utf-8",
    )

    stamps = _watchdog._load_recovery_timestamps(
        log_path, (DAEMON_MEMORY_PRESSURE_KILL,)
    )
    assert len(stamps) == 1, f"expected exactly the kill timestamp, got {stamps}"

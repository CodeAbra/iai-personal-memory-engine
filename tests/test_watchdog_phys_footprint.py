"""The watchdog grades its hard cap against macOS phys_footprint, not RSS.

resident_size (Mach ``resident_size``, what ``psutil.memory_info().rss`` reads)
counts reusable MADV_FREE pages an allocator has freed but not yet returned to
the kernel. phys_footprint is the kernel's "charged" memory — the number the
memory-pressure / jetsam machinery actually grades a process against — and it
EXCLUDES those reusable pages. Grading the 4 GiB cap against resident_size means
grading against the wrong number; this suite pins that the watchdog reads
phys_footprint for the kill decision on macOS, falls back to resident_size when
phys_footprint is unavailable (non-macOS / unreadable), and logs BOTH so the
phantom delta (rss - phys_footprint) stays visible in the watchdog log.
"""

from __future__ import annotations

import json
import os
import platform
import signal
import time
from pathlib import Path

import pytest

from iai_mcp import daemon
from iai_mcp.daemon import _watchdog as wd

DARWIN = platform.system() == "Darwin"

HARD_CAP = wd.WATCHDOG_RSS_HARD_CAP_BYTES
OVER_CAP = HARD_CAP + (512 * 1024 * 1024)
UNDER_CAP = 300 * 1024 * 1024
RSS_LOW = 300 * 1024 * 1024

NORMAL = 1


# --- the phys_footprint reader -----------------------------------------------


@pytest.mark.skipif(not DARWIN, reason="phys_footprint is a macOS metric")
def test_phys_footprint_reads_positive_int_on_macos() -> None:
    phys = wd._phys_footprint_bytes()
    assert isinstance(phys, int)
    assert phys > 0


@pytest.mark.skipif(not DARWIN, reason="phys_footprint is a macOS metric")
def test_phys_footprint_not_above_rss_on_clean_process() -> None:
    # phys_footprint excludes reusable (MADV_FREE) pages that still count toward
    # resident_size, so it can only be <= RSS (a small slack absorbs the race
    # between the two reads on a live, churning heap).
    rss = wd._own_rss_bytes()
    phys = wd._phys_footprint_bytes()
    assert rss is not None and phys is not None
    assert phys <= rss * 1.05, (
        f"phys_footprint ({phys}) should not exceed resident_size ({rss}); "
        "phys excludes reusable pages RSS still counts"
    )


def test_phys_footprint_returns_none_off_darwin(monkeypatch) -> None:
    # On any non-Darwin host there is no proc_pid_rusage syscall, so the reader
    # must return None and let the caller fall back to resident_size.
    monkeypatch.setattr(wd.platform, "system", lambda: "Linux")
    assert wd._phys_footprint_bytes() is None


def test_own_charged_prefers_phys_when_available(monkeypatch) -> None:
    monkeypatch.setattr(wd, "_own_rss_bytes", lambda: 200 * 1024 * 1024)
    monkeypatch.setattr(wd, "_phys_footprint_bytes", lambda: 150 * 1024 * 1024)
    charged, rss, phys = wd._own_charged_bytes()
    assert charged == 150 * 1024 * 1024  # phys wins the decision
    assert rss == 200 * 1024 * 1024  # rss kept for logging
    assert phys == 150 * 1024 * 1024


def test_own_charged_falls_back_to_rss_when_phys_none(monkeypatch) -> None:
    monkeypatch.setattr(wd, "_own_rss_bytes", lambda: 200 * 1024 * 1024)
    monkeypatch.setattr(wd, "_phys_footprint_bytes", lambda: None)
    charged, rss, phys = wd._own_charged_bytes()
    assert charged == 200 * 1024 * 1024  # rss is the decision number
    assert rss == 200 * 1024 * 1024
    assert phys is None


# --- the kill decision uses phys_footprint, logs RSS too ----------------------


@pytest.fixture
def tick_env(tmp_path, monkeypatch):
    log_path = tmp_path / ".daemon-watchdog.log"
    sock_path = str(tmp_path / ".daemon.sock")
    fd = os.open(str(log_path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setattr(daemon, "_WATCHDOG_LOG_FD", fd)
    monkeypatch.setattr(daemon, "_last_rss_breadcrumb_at", 0.0, raising=False)
    # Boot timestamp inside the cold-start grace — the cap must fire regardless.
    monkeypatch.setattr(daemon, "_daemon_started_monotonic", time.monotonic())

    kill_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        daemon.os, "kill", lambda pid, sig: kill_calls.append((pid, sig))
    )

    class _Ns:
        pass

    ns = _Ns()
    ns.log_path = log_path
    ns.sock_path = sock_path
    ns.kill_calls = kill_calls
    yield ns
    try:
        os.close(fd)
    except OSError:
        pass


def _probe(result: bool):
    async def _p(_sock, _timeout):
        return result

    return _p


def _last_breadcrumb(log_path: Path) -> dict:
    rows: list[dict] = []
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
    assert rows, "expected at least one rss_breadcrumb line"
    return rows[-1]


def test_tick_kills_when_phys_over_cap_while_rss_low(tick_env) -> None:
    # phys_footprint over the cap drives the kill EVEN THOUGH rss reads under it.
    daemon._watchdog_tick(
        object(),
        tick_env.sock_path,
        tick_env.log_path,
        0,
        probe_fn=_probe(True),
        pressure_fn=lambda: NORMAL,
        rss_fn=lambda: RSS_LOW,
        phys_fn=lambda: OVER_CAP,
    )
    assert tick_env.kill_calls == [(os.getpid(), signal.SIGKILL)], (
        "an over-cap phys_footprint must SIGKILL on the first tick even when "
        "resident_size reads under the cap"
    )
    crumb = tick_env.log_path.read_text(encoding="utf-8")
    assert daemon.DAEMON_MEMORY_PRESSURE_KILL in crumb
    assert "reason=leak" in crumb


def test_tick_no_kill_when_phys_under_cap_while_rss_over(tick_env) -> None:
    # The inverse of the bug: resident_size reads OVER the cap (phantom reusable
    # pages), but the real charged memory (phys_footprint) is well under it — so
    # the watchdog must NOT kill. This is the phantom-RSS false-positive the
    # switch to phys_footprint eliminates.
    daemon._watchdog_tick(
        object(),
        tick_env.sock_path,
        tick_env.log_path,
        0,
        probe_fn=_probe(True),
        pressure_fn=lambda: NORMAL,
        rss_fn=lambda: OVER_CAP,
        phys_fn=lambda: UNDER_CAP,
    )
    assert tick_env.kill_calls == [], (
        "phantom resident_size over the cap must NOT kill when phys_footprint "
        "(the real charged memory) is under it"
    )


def test_tick_logs_both_rss_and_phys(tick_env) -> None:
    daemon._watchdog_tick(
        object(),
        tick_env.sock_path,
        tick_env.log_path,
        0,
        probe_fn=_probe(True),
        pressure_fn=lambda: NORMAL,
        rss_fn=lambda: RSS_LOW,
        phys_fn=lambda: UNDER_CAP,
    )
    row = _last_breadcrumb(tick_env.log_path)
    assert row["rss_kib"] == RSS_LOW // 1024
    assert row["phys_footprint_kib"] == UNDER_CAP // 1024
    # The phantom delta is reconstructable from the two logged numbers.
    assert row["rss_kib"] - row["phys_footprint_kib"] == (RSS_LOW - UNDER_CAP) // 1024


# --- revert-proof: phys unavailable -> decision falls back to RSS -------------


def test_tick_falls_back_to_rss_when_phys_none(tick_env) -> None:
    # Force the phys reader to None (the non-macOS / unreadable path). The kill
    # decision must then grade the cap against resident_size, exactly as it did
    # before phys_footprint existed — an over-cap RSS still kills.
    daemon._watchdog_tick(
        object(),
        tick_env.sock_path,
        tick_env.log_path,
        0,
        probe_fn=_probe(True),
        pressure_fn=lambda: NORMAL,
        rss_fn=lambda: OVER_CAP,
        phys_fn=lambda: None,
    )
    assert tick_env.kill_calls == [(os.getpid(), signal.SIGKILL)], (
        "with phys_footprint unavailable the cap must fall back to "
        "resident_size — an over-cap RSS still kills"
    )
    row = _last_breadcrumb(tick_env.log_path)
    assert row["rss_kib"] == OVER_CAP // 1024
    assert row["phys_footprint_kib"] is None

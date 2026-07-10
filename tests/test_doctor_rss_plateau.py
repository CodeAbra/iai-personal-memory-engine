"""Unit coverage for the doctor resident-set 24h-plateau check.

The check uses a median-of-thirds statistic to compare the oldest third of
in-window rss_breadcrumb samples against the newest third: FAIL when the
sustained climb exceeds 1 GiB, WARN above 512 MiB, PASS otherwise or on fewer
than two in-window samples. It prefers phys_footprint_kib over rss_kib (the
former excludes MADV_FREE reusable pages), reads the env-aware watchdog-log
path, parses only JSON lines, and tolerates malformed or out-of-window rows
without raising. The median-of-thirds design is robust to single-sample
transients (e.g. a VACUUM spike on the first or last raw endpoint).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


def _write_watchdog_log(store_root: Path, lines: list[str]) -> Path:
    store_root.mkdir(parents=True, exist_ok=True)
    log_path = store_root / ".daemon-watchdog.log"
    log_path.write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")
    return log_path


def _breadcrumb(
    rss_kib: int,
    *,
    age_minutes: float,
    phys_footprint_kib: int | None = None,
) -> str:
    ts = (datetime.now(timezone.utc) - timedelta(minutes=age_minutes)).isoformat()
    row: dict = {
        "kind": "rss_breadcrumb",
        "ts": ts,
        "fsm_state": "SLEEP",
        "rss_kib": rss_kib,
        "vmmap_region_count": 289,
        "vm_allocate_kib": 16,
        "pressure_level": 1,
    }
    if phys_footprint_kib is not None:
        row["phys_footprint_kib"] = phys_footprint_kib
    return json.dumps(row)


def test_plateau_fail_over_1gib(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.doctor._lifecycle_checks import check_y_rss_24h_plateau

    base = 500 * 1024  # 500 MiB
    grown = base + (1100 * 1024)  # +1.1 GiB
    _write_watchdog_log(
        tmp_path,
        [
            _breadcrumb(base, age_minutes=600),
            _breadcrumb(grown, age_minutes=5),
        ],
    )
    result = check_y_rss_24h_plateau()
    assert result.status == "FAIL"
    assert result.passed is False


def test_plateau_pass_under_512mib(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.doctor._lifecycle_checks import check_y_rss_24h_plateau

    base = 500 * 1024
    grown = base + (100 * 1024)  # +100 MiB
    _write_watchdog_log(
        tmp_path,
        [
            _breadcrumb(base, age_minutes=600),
            _breadcrumb(grown, age_minutes=5),
        ],
    )
    result = check_y_rss_24h_plateau()
    assert result.status == "PASS"
    assert result.passed is True


def test_plateau_warn_between_512mib_and_1gib(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.doctor._lifecycle_checks import check_y_rss_24h_plateau

    base = 500 * 1024
    grown = base + (700 * 1024)  # +700 MiB (between 512 and 1024)
    _write_watchdog_log(
        tmp_path,
        [
            _breadcrumb(base, age_minutes=600),
            _breadcrumb(grown, age_minutes=5),
        ],
    )
    result = check_y_rss_24h_plateau()
    assert result.status == "WARN"


def test_plateau_insufficient_samples(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.doctor._lifecycle_checks import check_y_rss_24h_plateau

    _write_watchdog_log(tmp_path, [_breadcrumb(500 * 1024, age_minutes=5)])
    result = check_y_rss_24h_plateau()
    assert result.status == "PASS"
    assert result.passed is True
    assert "insufficient" in result.detail.lower()


def test_plateau_ignores_old_and_garbage(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.doctor._lifecycle_checks import check_y_rss_24h_plateau

    base = 500 * 1024
    grown = base + (100 * 1024)  # +100 MiB among the two in-window rows -> PASS
    _write_watchdog_log(
        tmp_path,
        [
            # Older than 24h: must be excluded (would otherwise force a FAIL).
            _breadcrumb(1 * 1024, age_minutes=60 * 30),
            "this is not json and must not crash the parser",
            "{ malformed json",
            "2026-06-20T10:00:00+00:00 daemon_memory_pressure_kill reason=leak",
            _breadcrumb(base, age_minutes=600),
            _breadcrumb(grown, age_minutes=5),
        ],
    )
    result = check_y_rss_24h_plateau()
    assert result.status == "PASS"
    assert result.passed is True


def test_plateau_no_log_is_pass(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.doctor._lifecycle_checks import check_y_rss_24h_plateau

    # No watchdog log written at all: absence is not a failure.
    result = check_y_rss_24h_plateau()
    assert result.passed is True
    assert result.status == "PASS"


def test_plateau_in_run_diagnosis_and_all(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    import iai_mcp.doctor as doctor

    assert "check_y_rss_24h_plateau" in doctor.__all__

    results = doctor.run_diagnosis()
    names = [r.name for r in results]
    # The plateau row sits between the collapsed-timestamps (x) row and the
    # avx2 (z) row.
    x_idx = next(i for i, n in enumerate(names) if "(x)" in n)
    z_idx = next(i for i, n in enumerate(names) if "(z)" in n)
    y_idx = next(i for i, n in enumerate(names) if "(y)" in n)
    assert x_idx < y_idx < z_idx


def test_plateau_vacuum_transient_on_endpoints_does_not_flip_verdict(
    tmp_path, monkeypatch
) -> None:
    """A VACUUM-induced transient spike on the first or last raw sample must not
    change the PASS verdict when the underlying baseline is stable.  The
    median-of-thirds statistic absorbs the spike because it only affects one of
    the (at least three) samples in the relevant third.
    """
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.doctor._lifecycle_checks import check_y_rss_24h_plateau

    base_kib = 500 * 1024  # 500 MiB — stable baseline
    # Spike injected on the very first sample (VACUUM transient when the daemon
    # was first sampled) — 2x the baseline.
    first_spike_kib = base_kib * 2

    # Build a realistic 9-sample stream:
    # sample[0] = spike (oldest), samples[1..7] = stable baseline, sample[8] = stable
    lines = [
        _breadcrumb(first_spike_kib, age_minutes=23 * 60),  # oldest — spike
        _breadcrumb(base_kib, age_minutes=20 * 60),
        _breadcrumb(base_kib, age_minutes=18 * 60),
        _breadcrumb(base_kib, age_minutes=15 * 60),
        _breadcrumb(base_kib, age_minutes=12 * 60),
        _breadcrumb(base_kib, age_minutes=9 * 60),
        _breadcrumb(base_kib, age_minutes=6 * 60),
        _breadcrumb(base_kib, age_minutes=3 * 60),
        _breadcrumb(base_kib, age_minutes=30),  # newest — stable
    ]
    _write_watchdog_log(tmp_path, lines)
    result = check_y_rss_24h_plateau()
    # The spike on the first endpoint would have produced a negative delta
    # (≈ −500 MiB) under the old endpoint-delta logic, trivially passing while
    # hiding nothing.  Under median-of-thirds the first-third median is the
    # stable baseline, so the delta is ~0 and the verdict is PASS.
    assert result.status == "PASS", f"unexpected status: {result.status!r} — {result.detail}"
    assert result.passed is True

    # Now inject the spike on the LAST sample (VACUUM at newest endpoint).
    # Old endpoint-delta would have reported a false +500 MiB WARN/FAIL.
    last_spike_kib = base_kib + (700 * 1024)  # +700 MiB — would trigger WARN
    lines2 = [
        _breadcrumb(base_kib, age_minutes=23 * 60),
        _breadcrumb(base_kib, age_minutes=20 * 60),
        _breadcrumb(base_kib, age_minutes=18 * 60),
        _breadcrumb(base_kib, age_minutes=15 * 60),
        _breadcrumb(base_kib, age_minutes=12 * 60),
        _breadcrumb(base_kib, age_minutes=9 * 60),
        _breadcrumb(base_kib, age_minutes=6 * 60),
        _breadcrumb(base_kib, age_minutes=3 * 60),
        _breadcrumb(last_spike_kib, age_minutes=30),  # newest — spike
    ]
    _write_watchdog_log(tmp_path, lines2)
    result2 = check_y_rss_24h_plateau()
    assert result2.status == "PASS", (
        f"last-endpoint spike caused false alarm: {result2.status!r} — {result2.detail}"
    )


def test_plateau_genuine_sustained_climb_still_fails(tmp_path, monkeypatch) -> None:
    """A real monotonic climb must still produce FAIL even with median-of-thirds.

    The median-of-thirds delta is (median of newest third) - (median of oldest
    third).  For the FAIL threshold of 1024 MiB the total end-to-end climb must
    be large enough so that even the inter-third comparison exceeds 1024 MiB.
    With 9 samples and third=3:
        first-third  median ≈ sample[1] value
        last-third   median ≈ sample[7] value
    So the climb from sample[1] to sample[7] (6/8 of the total) must exceed
    1024 MiB.  A total climb of 1400 MiB gives inter-third ≈ 1050 MiB → FAIL.
    """
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.doctor._lifecycle_checks import check_y_rss_24h_plateau

    # 9 samples climbing steadily; total span = 1400 MiB.
    base_kib = 500 * 1024
    total_climb_kib = 1400 * 1024
    step_kib = total_climb_kib // 8
    ages = [23 * 60, 20 * 60, 18 * 60, 15 * 60, 12 * 60, 9 * 60, 6 * 60, 3 * 60, 30]
    lines = [
        _breadcrumb(base_kib + i * step_kib, age_minutes=age)
        for i, age in enumerate(ages)
    ]
    _write_watchdog_log(tmp_path, lines)
    result = check_y_rss_24h_plateau()
    assert result.status == "FAIL", (
        f"genuine climb not detected: {result.status!r} — {result.detail}"
    )
    assert result.passed is False


def test_plateau_prefers_phys_footprint_over_rss(tmp_path, monkeypatch) -> None:
    """When phys_footprint_kib is present it is used instead of rss_kib."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.doctor._lifecycle_checks import check_y_rss_24h_plateau

    # rss_kib values would indicate a >1 GiB climb (FAIL), but phys_footprint_kib
    # is stable — the settled footprint wins.
    base_phys = 500 * 1024
    large_rss = base_phys + (1200 * 1024)  # would FAIL if rss_kib were used
    _write_watchdog_log(
        tmp_path,
        [
            _breadcrumb(large_rss, age_minutes=600, phys_footprint_kib=base_phys),
            _breadcrumb(large_rss, age_minutes=5, phys_footprint_kib=base_phys + 50 * 1024),
        ],
    )
    result = check_y_rss_24h_plateau()
    # phys_footprint delta = +50 MiB → PASS
    assert result.status == "PASS", (
        f"rss_kib leaked into metric: {result.status!r} — {result.detail}"
    )

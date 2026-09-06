"""Daemon, socket, lifecycle, sleep-cycle, heartbeat, idle, subscription and CLI-reachability health checks.

Read-only probes of the running daemon and its lifecycle surface. The store is
never written; a store held by the live daemon is reported as a normal, passing
condition (the daemon is sleep/consolidation only, never a gatekeeper).
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from iai_mcp.doctor import (
    CheckResult,
    _extract_binder_pids,
    _extract_binder_pids_ss,
    _format_relative_short,
    _resolve_lifecycle_log_dir,
    _resolve_lifecycle_state_path,
    _resolve_socket_path,
    _resolve_wrappers_dir,
)

logger = logging.getLogger(__name__)


def _resolve_hippo_db_path(*args, **kwargs):
    # re-fetch the package attribute per call so monkeypatches stay visible
    from iai_mcp import doctor as _pkg

    return _pkg._resolve_hippo_db_path(*args, **kwargs)


def check_a_daemon_alive() -> CheckResult:
    from iai_mcp.daemon_state import load_state

    try:
        state = load_state() or {}
    except Exception as e:
        logger.debug("check_a: daemon-state.json unreadable: %s", e)
        return CheckResult(
            "(a) daemon process alive",
            False,
            f"daemon-state.json unreadable: {type(e).__name__}: {e}",
        )

    pid = state.get("daemon_pid")
    if pid is None:
        return CheckResult(
            "(a) daemon process alive",
            False,
            "ABSENT (no daemon_pid in state — daemon never booted or already shut down)",
        )

    if not isinstance(pid, int) or pid < 1 or pid > 2**31 - 1:
        return CheckResult(
            "(a) daemon process alive",
            False,
            f"daemon_pid={pid!r} is not a valid PID (corrupt state?)",
        )

    from iai_mcp.lifecycle_lock import _is_pid_alive

    try:
        if not _is_pid_alive(pid):
            return CheckResult(
                "(a) daemon process alive",
                False,
                f"PID {pid} is not a live iai_mcp.daemon process",
            )
    except OSError as e:
        return CheckResult(
            "(a) daemon process alive",
            False,
            f"liveness probe failed: {type(e).__name__}: {e}",
        )

    try:
        import psutil

        proc = psutil.Process(pid)
        cmdline = " ".join(proc.cmdline() or [])
        if "iai_mcp.daemon" not in cmdline:
            return CheckResult(
                "(a) daemon process alive",
                False,
                f"PID {pid} is NOT iai_mcp.daemon (got: {proc.name()!r})",
            )
    except Exception as e:  # noqa: BLE001 — psutil edge cases all roll up here
        logger.debug("check_a: psutil verify PID %d failed: %s", pid, e)
        return CheckResult(
            "(a) daemon process alive",
            False,
            f"could not verify PID {pid}: {type(e).__name__}: {e}",
        )

    return CheckResult(
        "(a) daemon process alive",
        True,
        f"PID {pid} (iai_mcp.daemon)",
    )


async def _socket_connect_probe(socket_path: Path, timeout: float) -> str | None:
    try:
        from iai_mcp._ipc import open_ipc_connection
        reader, writer = await asyncio.wait_for(
            open_ipc_connection(str(socket_path)),
            timeout=timeout,
        )
    except FileNotFoundError:
        return "FileNotFoundError"
    except ConnectionRefusedError:
        return "ConnectionRefusedError"
    except asyncio.TimeoutError:
        return f"TimeoutError after {int(timeout * 1000)} ms"
    except OSError as e:
        return f"OSError errno={e.errno}: {e.strerror or e}"
    try:
        writer.close()
        await writer.wait_closed()
    except Exception:
        pass
    return None


def check_b_socket_fresh() -> CheckResult:
    socket_path = _resolve_socket_path()
    if sys.platform == "win32":
        # Windows serves the daemon over loopback TCP; the bound port lives
        # in PORT_FILE and there is no unix-socket path to stat.
        from iai_mcp._ipc import PORT_FILE

        if not PORT_FILE.exists():
            return CheckResult(
                "(b) socket file fresh",
                False,
                f"{PORT_FILE} does not exist",
            )
    elif not socket_path.exists():
        return CheckResult(
            "(b) socket file fresh",
            False,
            f"{socket_path} does not exist",
        )

    t0 = time.monotonic()
    try:
        err = asyncio.run(_socket_connect_probe(socket_path, timeout=1.0))
    except Exception as e:  # noqa: BLE001 — surface any unexpected probe failure
        logger.debug("check_b: socket probe failed: %s", e)
        return CheckResult(
            "(b) socket file fresh",
            False,
            f"connect failed: {type(e).__name__}: {e}",
        )
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    if err is not None:
        return CheckResult(
            "(b) socket file fresh",
            False,
            f"{socket_path} present but unreachable: {err}",
        )
    return CheckResult(
        "(b) socket file fresh",
        True,
        f"{socket_path} connected in {elapsed_ms} ms",
    )


def check_c_lock_healthy() -> CheckResult:
    import errno as _errno
    try:
        import fcntl as _fcntl
    except ImportError:
        return CheckResult("(c) lock file healthy", True, "skipped on Windows")

    lock_path = _resolve_hippo_db_path().parent / ".lock"
    if not lock_path.exists():
        return CheckResult(
            "(c) lock file healthy",
            True,
            f"{lock_path} absent (store not yet initialized)",
        )
    fd = None
    try:
        fd = os.open(str(lock_path), os.O_RDONLY)
        try:
            _fcntl.flock(fd, _fcntl.LOCK_SH | _fcntl.LOCK_NB)
            _fcntl.flock(fd, _fcntl.LOCK_UN)
            return CheckResult(
                "(c) lock file healthy",
                True,
                f"{lock_path} acquirable (store idle)",
            )
        except OSError as exc:
            if exc.errno in (_errno.EAGAIN, _errno.EWOULDBLOCK):
                return CheckResult(
                    "(c) lock file healthy",
                    True,
                    f"{lock_path} held (consolidating or recall active — normal)",
                )
            raise
    except Exception as e:  # noqa: BLE001 — fcntl/OSError/permission all FAIL
        logger.debug("check_c: store-lock probe failed: %s", e)
        return CheckResult(
            "(c) lock file healthy",
            False,
            f"store-lock probe failed: {type(e).__name__}: {e}",
        )
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def check_d_no_orphan_core() -> CheckResult:
    try:
        import psutil

        orphans: list[int] = []
        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                cl = " ".join(p.info.get("cmdline") or [])
                if "iai_mcp.core" in cl:
                    orphans.append(p.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if not orphans:
            return CheckResult(
                "(d) no orphan iai_mcp.core procs",
                True,
                "0 found",
            )
        return CheckResult(
            "(d) no orphan iai_mcp.core procs",
            False,
            f"{len(orphans)} found: PIDs {orphans}",
        )
    except Exception as e:  # noqa: BLE001 — psutil edge cases
        logger.debug("check_d: psutil probe failed: %s", e)
        return CheckResult(
            "(d) no orphan iai_mcp.core procs",
            False,
            f"psutil probe failed: {type(e).__name__}: {e}",
        )


def check_e_state_file_valid() -> CheckResult:
    from iai_mcp.daemon_state import load_state

    try:
        state = load_state() or {}
    except Exception as e:  # noqa: BLE001 — corrupt JSON / IO error
        logger.debug("check_e: daemon state unreadable: %s", e)
        return CheckResult(
            "(e) daemon state file valid",
            False,
            f"unreadable: {type(e).__name__}: {e}",
        )

    fsm_state = state.get("fsm_state")
    if fsm_state is None:
        return CheckResult(
            "(e) daemon state file valid",
            True,
            "no state file (daemon never booted — not a bug)",
        )

    # TRANSITIONING is the legacy-file spelling the reconciler writes for
    # canonical DROWSY — a healthy daemon parked in DROWSY must not FAIL here.
    valid = {
        "WAKE", "DROWSY", "TRANSITIONING", "SLEEP", "SLEEPING",
        "DREAMING", "HIBERNATION",
    }
    if fsm_state in valid:
        return CheckResult(
            "(e) daemon state file valid",
            True,
            f"fsm_state={fsm_state}",
        )
    return CheckResult(
        "(e) daemon state file valid",
        False,
        f"fsm_state={fsm_state!r} not in {sorted(valid)}",
    )


def check_g_no_dup_binders() -> CheckResult:
    socket_path = _resolve_socket_path()
    if not socket_path.exists():
        return CheckResult(
            "(g) no dup binders",
            True,
            "no socket file (skip)",
        )
    try:
        result = subprocess.run(
            ["lsof", "-U", "-F", "pn"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return CheckResult(
            "(g) no dup binders",
            True,
            f"lsof unavailable: {e} (skip)",
        )
    binder_pids = _extract_binder_pids(result.stdout, socket_path)
    if not binder_pids and platform.system() == "Linux":
        # Non-root Linux cannot read other procs' /proc/<pid>/fd/ via lsof; fall back to
        # `ss -lxp`, which reads the globally-readable /proc/net/unix.
        try:
            ss_result = subprocess.run(
                ["ss", "-lxp"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            binder_pids = _extract_binder_pids_ss(ss_result.stdout, socket_path)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    if len(binder_pids) <= 1:
        return CheckResult(
            "(g) no dup binders",
            True,
            f"{len(binder_pids)} binder(s)",
        )
    return CheckResult(
        "(g) no dup binders",
        False,
        f"{len(binder_pids)} processes bound to socket: {sorted(binder_pids)}",
    )


def check_m_heartbeat_scanner() -> CheckResult:
    from iai_mcp.heartbeat_scanner import HeartbeatScanner, HeartbeatStatus

    wrappers_dir = _resolve_wrappers_dir()
    if not wrappers_dir.exists():
        return CheckResult(
            name="(m) heartbeat scanner",
            passed=True,
            detail=(
                f"{wrappers_dir} not present yet (fresh install or no "
                "wrapper has refreshed yet)"
            ),
            status="PASS",
        )

    scanner = HeartbeatScanner(wrappers_dir)
    try:
        entries = scanner.scan()
    except OSError as exc:
        return CheckResult(
            name="(m) heartbeat scanner",
            passed=False,
            detail=(
                f"could not scan {wrappers_dir}: "
                f"{type(exc).__name__}: {exc}"
            ),
            status="FAIL",
        )

    fresh = sum(1 for e in entries if e.status is HeartbeatStatus.FRESH)
    stale = sum(1 for e in entries if e.status is HeartbeatStatus.STALE)
    orphan = sum(1 for e in entries if e.status is HeartbeatStatus.ORPHAN)
    return CheckResult(
        name="(m) heartbeat scanner",
        passed=True,
        detail=f"n={fresh} fresh, {stale} stale, {orphan} orphan",
        status="PASS",
    )


def _daemon_process_alive() -> bool:
    from iai_mcp.daemon_state import load_state as _load_ds
    from iai_mcp.lifecycle_lock import _is_pid_alive

    try:
        pid = (_load_ds() or {}).get("daemon_pid")
        return isinstance(pid, int) and pid > 0 and _is_pid_alive(pid)
    except Exception:  # noqa: BLE001 — unreadable state reads as not-alive
        return False


def check_j_lifecycle_current_state() -> CheckResult:
    from iai_mcp.lifecycle_state import load_state

    state_path = _resolve_lifecycle_state_path()
    record = load_state(state_path)
    current = record.get("current_state", "WAKE")
    since_ts = record.get("since_ts", "?")
    elapsed = _format_relative_short(since_ts)
    shadow_run = record.get("shadow_run", True)

    detail = f"{current} since {elapsed} (shadow_run={'true' if shadow_run else 'false'})"

    # A parked engine (persisted HIBERNATION, or SLEEP after a mid-cycle
    # death) with no live daemon cannot wake itself on engines that need
    # the wake-signal file — someone running doctor IS the demand.
    if current in ("HIBERNATION", "SLEEP", "SLEEPING") and not _daemon_process_alive():
        return CheckResult(
            name="(j) lifecycle current state",
            passed=False,
            detail=(
                f"{detail}; engine parked in {current} with no live daemon "
                "— --apply/--auto writes wake.signal and starts it"
            ),
            status="FAIL",
        )
    return CheckResult(
        name="(j) lifecycle current state",
        passed=True,
        detail=detail,
        status="PASS",
    )


def check_k_lifecycle_history_24h() -> CheckResult:
    from datetime import datetime as _dt
    from datetime import timedelta as _td
    from datetime import timezone as _tz

    from iai_mcp.lifecycle_event_log import LifecycleEventLog

    log_dir = _resolve_lifecycle_log_dir()
    if not log_dir.exists():
        return CheckResult(
            name="(k) lifecycle history 24h",
            passed=True,
            detail="no event log yet (fresh install or daemon never run)",
            status="PASS",
        )

    log = LifecycleEventLog(log_dir=log_dir)
    now = _dt.now(_tz.utc)
    today = now.strftime("%Y-%m-%d")
    yesterday = (now - _td(days=1)).strftime("%Y-%m-%d")

    transitions: list[dict[str, Any]] = []
    for date_str in (yesterday, today):
        try:
            events = log.read_all(date_str=date_str)
        except OSError:
            continue
        for ev in events:
            if ev.get("event") == "state_transition":
                transitions.append(ev)

    counts: dict[str, int] = {}
    for ev in transitions:
        to = ev.get("to") or "?"
        counts[to] = counts.get(to, 0) + 1

    if not transitions:
        return CheckResult(
            name="(k) lifecycle history 24h",
            passed=True,
            detail="0 transitions in last 24h",
            status="PASS",
        )

    summary = ", ".join(f"{state}={n}" for state, n in sorted(counts.items()))
    return CheckResult(
        name="(k) lifecycle history 24h",
        passed=True,
        detail=f"{len(transitions)} transitions ({summary})",
        status="PASS",
    )


def check_y_rss_24h_plateau() -> CheckResult:
    import statistics as _stats
    from datetime import datetime as _dt
    from datetime import timezone as _tz
    import json as _json

    name = "(y) RSS 24h plateau"

    # Env-aware watchdog-log path (honors IAI_MCP_STORE), resolved locally to
    # avoid importing the daemon package from the doctor surface.
    root = os.environ.get("IAI_MCP_STORE")
    log_path = (Path(root) if root else Path.home() / ".iai-mcp") / ".daemon-watchdog.log"

    if not log_path.exists():
        return CheckResult(
            name=name,
            passed=True,
            detail="no watchdog log yet (fresh install or daemon never run)",
            status="PASS",
        )

    def _settled_kib(row: dict) -> int | None:
        """Prefer phys_footprint_kib (excludes MADV_FREE reusable pages); fall back to rss_kib."""
        v = row.get("phys_footprint_kib")
        if isinstance(v, int):
            return v
        v = row.get("rss_kib")
        return v if isinstance(v, int) else None

    now = _dt.now(_tz.utc).timestamp()
    cutoff = now - 24 * 3600
    samples: list[dict] = []
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.strip()
                if not line.startswith("{"):
                    continue
                try:
                    row = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                if row.get("kind") != "rss_breadcrumb":
                    continue
                ts_raw = row.get("ts")
                if not isinstance(ts_raw, str):
                    continue
                try:
                    ts = _dt.fromisoformat(ts_raw).timestamp()
                except (ValueError, TypeError):
                    continue
                if ts < cutoff:
                    continue
                kib = _settled_kib(row)
                if kib is None:
                    continue
                samples.append({"ts": ts, "kib": kib})
    except OSError:
        return CheckResult(
            name=name,
            passed=True,
            detail="watchdog log unreadable",
            status="WARN",
        )

    if len(samples) < 2:
        return CheckResult(
            name=name,
            passed=True,
            detail="insufficient samples (need >= 2 within 24h)",
            status="PASS",
        )

    # Median-of-thirds: compare the median of the oldest third of samples against
    # the median of the newest third. This is robust to single-sample transients
    # (e.g. a VACUUM spike landing on the first or last raw endpoint) while still
    # detecting a genuine sustained climb across the full window.
    samples.sort(key=lambda s: s["ts"])
    third = max(1, len(samples) // 3)
    base_kib = _stats.median(s["kib"] for s in samples[:third])
    recent_kib = _stats.median(s["kib"] for s in samples[-third:])
    delta_mib = (recent_kib - base_kib) / 1024.0
    span_hr = (samples[-1]["ts"] - samples[0]["ts"]) / 3600.0
    detail = (
        f"{delta_mib:+.0f} MiB over {span_hr:.1f}h "
        f"({base_kib / 1024:.0f} -> {recent_kib / 1024:.0f} MiB, median-of-thirds, n={len(samples)})"
    )

    if delta_mib > 1024:
        return CheckResult(
            name=name,
            passed=False,
            detail=f"{detail}; resident set climbing — inspect 'iai-mcp daemon rss-stats'",
            status="FAIL",
        )
    if delta_mib > 512:
        return CheckResult(
            name=name,
            passed=True,
            detail=f"{detail}; resident set drifting up",
            status="WARN",
        )
    return CheckResult(name=name, passed=True, detail=detail, status="PASS")


def check_l_sleep_cycle_status() -> CheckResult:
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    from iai_mcp.lifecycle_state import load_state

    state_path = _resolve_lifecycle_state_path()
    record = load_state(state_path)
    quarantine = record.get("quarantine")
    if quarantine is None:
        return CheckResult(
            name="(l) sleep cycle quarantine",
            passed=True,
            detail="no quarantine active",
            status="PASS",
        )

    reason = quarantine.get("reason", "?")
    until_ts = quarantine.get("until_ts", "?")
    since_ts = quarantine.get("since_ts", "?")

    now = _dt.now(_tz.utc)
    try:
        since = _dt.fromisoformat(since_ts)
        if since.tzinfo is None:
            since = since.replace(tzinfo=_tz.utc)
        age_hours = (now - since).total_seconds() / 3600.0
    except (TypeError, ValueError):
        age_hours = 0.0

    try:
        until = _dt.fromisoformat(until_ts)
        if until.tzinfo is None:
            until = until.replace(tzinfo=_tz.utc)
        expired = until <= now
    except (TypeError, ValueError):
        expired = False

    if expired:
        return CheckResult(
            name="(l) sleep cycle quarantine",
            passed=True,
            detail=(
                f"quarantine expired (until={until_ts}); will clear on next "
                f"sleep-cycle run; reason={reason}"
            ),
            status="PASS",
        )

    detail = (
        f"quarantined for {age_hours:.1f}h; until={until_ts}; reason={reason}"
    )

    if age_hours >= 12.0:
        return CheckResult(
            name="(l) sleep cycle quarantine",
            passed=False,
            detail=(
                f"{detail}; run `iai-mcp maintenance sleep-cycle "
                "--reset-quarantine` to clear"
            ),
            status="FAIL",
        )
    return CheckResult(
        name="(l) sleep cycle quarantine",
        passed=True,
        detail=detail,
        status="WARN",
    )


def check_n_hid_idle_source() -> CheckResult:
    from iai_mcp.idle_detector import IdleDetector

    detail, result_status = IdleDetector().describe()

    return CheckResult(
        name="(n) HID idle source",
        passed=True,
        detail=detail,
        status=result_status,
    )


def check_o_subscription_credentials() -> CheckResult:
    try:
        from iai_mcp.claude_cli import verify_credentials_subscription
    except Exception as exc:  # noqa: BLE001 -- defensive
        return CheckResult(
            name="(o) Claude subscription credentials",
            passed=True,
            detail=f"unable to import claude_cli ({exc}); skipping",
            status="WARN",
        )

    result = verify_credentials_subscription()
    if result.get("ok"):
        sub_type = result.get("subscription_type") or result.get("billing_type") or "unknown"
        return CheckResult(
            name="(o) Claude subscription credentials",
            passed=True,
            detail=f"valid {sub_type} subscription with inference scope",
            status="PASS",
        )

    reason = result.get("reason", "unknown")
    return CheckResult(
        name="(o) Claude subscription credentials",
        passed=True,
        detail=(
            f"reason={reason}; daemon will fall back to local Tier-0 "
            "consolidation (no LLM critic, no nightly insight). Run "
            "`claude /login` to restore subscription path."
        ),
        status="WARN",
    )


def check_q_iai_cli_reachable() -> CheckResult:
    import shutil
    import os
    import sys

    iai_path = shutil.which("iai")
    if iai_path is None:
        bin_dir = os.path.dirname(sys.executable)
        exe_name = "iai.exe" if sys.platform == "win32" else "iai"
        fallback = os.path.join(bin_dir, exe_name)
        if os.path.exists(fallback):
            iai_path = fallback

    if iai_path is None:
        return CheckResult(
            name="(q) iai CLI reachable",
            passed=True,
            detail=(
                "iai not in PATH. Re-run `pip install -e .` from the repo "
                "root to register the entry point."
            ),
            status="WARN",
        )

    try:
        completed = subprocess.run(  # noqa: S603 -- argv list, no shell
            [iai_path, "--version"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return CheckResult(
            name="(q) iai CLI reachable",
            passed=True,
            detail=f"iai found at {iai_path} but invocation failed: {exc}",
            status="WARN",
        )

    if completed.returncode != 0:
        return CheckResult(
            name="(q) iai CLI reachable",
            passed=True,
            detail=(
                f"iai --version exited {completed.returncode}: "
                f"{completed.stderr.strip()[:120]}"
            ),
            status="WARN",
        )

    version_line = (completed.stdout or completed.stderr).strip().splitlines()[0:1]
    version = version_line[0] if version_line else "?"
    return CheckResult(
        name="(q) iai CLI reachable",
        passed=True,
        detail=f"{iai_path} -> {version}",
        status="PASS",
    )


# Consecutive elapsed nightly windows with no minted insight before this check
# alarms.
_INSIGHT_MINT_ALARM_THRESHOLD_NIGHTS: int = 3
_INSIGHT_MINT_QUERY_LIMIT: int = 200


_INSIGHT_MINT_CHECK_NAME: str = "(bb) nightly insight mint"


def _read_last_sleep_step_failure() -> str | None:
    """Best-effort read of the last recorded sleep-step failure.

    Reads the lifecycle-state record directly -- a local file, independent
    of daemon reachability -- and returns the ``sleep_cycle_progress.last_error``
    string when present. Every failure mode (missing file, unreadable file,
    malformed payload, missing sub-dict, wrong types) returns ``None``; this
    helper must never raise, since a diagnostic that crashes the checklist
    is worse than one that omits a hint.
    """
    try:
        from iai_mcp import lifecycle_state

        record = lifecycle_state.load_state()
        progress = record.get("sleep_cycle_progress")
        if not isinstance(progress, dict):
            return None
        last_error = progress.get("last_error")
        return last_error if isinstance(last_error, str) and last_error else None
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None


def _with_last_sleep_step_failure(detail: str) -> str:
    """Append the last recorded sleep-step failure to a WARN/FAIL detail."""
    failure = _read_last_sleep_step_failure()
    if not failure:
        return detail
    return f"{detail}; last recorded nightly-step failure: {failure}"


def _evaluate_nightly_insight_events(
    events: list[dict],
    now: Any,
    *,
    no_history: bool,
) -> CheckResult:
    """Shared FAIL/WARN/PASS verdict logic for both read paths.

    ``events`` are ``rem_cycle_completed`` rows shaped ``{"ts": datetime,
    "data": dict}``, already windowed to the alarm's lookback. ``no_history``
    signals a confirmed empty history (fresh install or daemon never run),
    bypassing the bucketing entirely.
    """
    from datetime import timedelta as _td
    from datetime import timezone as _tz

    name = _INSIGHT_MINT_CHECK_NAME

    if no_history:
        return CheckResult(
            name=name,
            passed=True,
            detail="no nightly cycle history yet (fresh install or daemon never run)",
            status="PASS",
        )

    buckets: dict[Any, list[dict]] = {}
    for ev in events:
        ts = ev["ts"]
        day = ts.astimezone(_tz.utc).date() if ts.tzinfo else ts.date()
        buckets.setdefault(day, []).append(ev)

    today = now.date()

    today_events = buckets.get(today, [])
    if any(ev["data"].get("claude_call_used") is True for ev in today_events):
        return CheckResult(
            name=name,
            passed=True,
            detail=f"last minted insight {today.isoformat()}",
            status="PASS",
        )

    elapsed_buckets = {day: evs for day, evs in buckets.items() if day < today}

    if not elapsed_buckets:
        return CheckResult(
            name=name,
            passed=True,
            detail="no elapsed nightly window recorded yet",
            status="PASS",
        )

    earliest_known = min(elapsed_buckets)
    consecutive = 0
    saw_unreadable_night = False
    cursor = today - _td(days=1)
    last_mint_date = None

    while cursor >= earliest_known and consecutive < _INSIGHT_MINT_ALARM_THRESHOLD_NIGHTS:
        night_events = elapsed_buckets.get(cursor, [])
        if any(ev["data"].get("claude_call_used") is True for ev in night_events):
            last_mint_date = cursor
            break
        if night_events and all(ev["data"] == {} for ev in night_events):
            saw_unreadable_night = True
        consecutive += 1
        cursor -= _td(days=1)

    if last_mint_date is not None:
        return CheckResult(
            name=name,
            passed=True,
            detail=f"last minted insight {last_mint_date.isoformat()}",
            status="PASS",
        )

    if consecutive == 0:
        return CheckResult(
            name=name,
            passed=True,
            detail="no elapsed nightly window recorded yet",
            status="PASS",
        )

    if consecutive >= _INSIGHT_MINT_ALARM_THRESHOLD_NIGHTS:
        if saw_unreadable_night:
            return CheckResult(
                name=name,
                passed=True,
                detail=_with_last_sleep_step_failure(
                    f"{consecutive} consecutive nights without a confirmed mint; "
                    "some event data was unreadable"
                ),
                status="WARN",
            )
        return CheckResult(
            name=name,
            passed=False,
            detail=_with_last_sleep_step_failure(
                f"{consecutive} consecutive nights without a minted insight"
            ),
            status="FAIL",
        )

    return CheckResult(
        name=name,
        passed=True,
        detail=_with_last_sleep_step_failure(
            f"{consecutive} consecutive night(s) without a minted insight"
        ),
        status="WARN",
    )


class _SocketUnavailable(Exception):
    """Sentinel: the daemon socket could not be reached at all -- caller must
    fall back to a direct store open, not treat this as a genuine failure."""


def _normalize_socket_events(events_raw: list[Any]) -> tuple[list[dict], int]:
    """Coerce wire-shaped event rows to ``{"ts": datetime, "data": dict}``.

    Returns ``(events, dropped)``. A row is dropped only when its ``ts``
    cannot be parsed at all (or the row itself is not a dict) -- the caller
    must treat any drop as an ambiguous read, never as if the row never
    existed, or the verdict gets computed on data known to be incomplete.
    """
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    out: list[dict] = []
    dropped = 0
    for e in events_raw:
        if not isinstance(e, dict):
            dropped += 1
            continue
        try:
            ts = _dt.fromisoformat(str(e.get("ts")))
        except (TypeError, ValueError):
            dropped += 1
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=_tz.utc)
        data = e.get("data")
        if not isinstance(data, dict):
            data = {}
        out.append({"ts": ts, "data": data})
    return out, dropped


def _rem_events_via_socket(now: Any, *, since: Any) -> CheckResult:
    """Read ``rem_cycle_completed`` events through the daemon's socket.

    The daemon serves a generic JSON-RPC dispatch (``events_query`` ->
    ``core.dispatch``) over its unix socket even in steady state, when a
    same-process direct store open would contend for the OS lock the daemon
    already holds. Raises ``_SocketUnavailable`` when the daemon cannot be
    reached at all, so the caller falls back to the direct-open path; any
    other failure (malformed response, dispatch error) is a genuine WARN --
    the daemon answered, but the answer was not usable.
    """
    from iai_mcp.cli import _send_jsonrpc_request

    name = _INSIGHT_MINT_CHECK_NAME

    probe_resp = _send_jsonrpc_request(
        "events_query",
        {"kind": "rem_cycle_completed", "limit": 1},
        connect_timeout=1.0,
        read_timeout=5.0,
    )
    if probe_resp is None:
        raise _SocketUnavailable()
    if not isinstance(probe_resp, dict) or "result" not in probe_resp:
        return CheckResult(
            name=name,
            passed=True,
            detail="unable to read nightly cycle history via daemon socket",
            status="WARN",
        )
    probe_result = probe_resp["result"]
    probe_events = probe_result.get("events") if isinstance(probe_result, dict) else None
    if probe_events is None:
        return CheckResult(
            name=name,
            passed=True,
            detail="unable to read nightly cycle history via daemon socket",
            status="WARN",
        )
    if not probe_events:
        return _evaluate_nightly_insight_events([], now, no_history=True)

    windowed_resp = _send_jsonrpc_request(
        "events_query",
        {
            "kind": "rem_cycle_completed",
            "since": since.isoformat(),
            "limit": _INSIGHT_MINT_QUERY_LIMIT,
        },
        connect_timeout=1.0,
        read_timeout=5.0,
    )
    if not isinstance(windowed_resp, dict) or "result" not in windowed_resp:
        # The daemon already answered the probe -- it is up. A now-failing
        # second call is a genuine WARN, not grounds to fall back to a
        # direct open that would contend for the same lock this path exists
        # to avoid.
        return CheckResult(
            name=name,
            passed=True,
            detail="unable to read nightly cycle history via daemon socket",
            status="WARN",
        )
    windowed_result = windowed_resp["result"]
    events_raw = windowed_result.get("events") if isinstance(windowed_result, dict) else None
    if events_raw is None:
        return CheckResult(
            name=name,
            passed=True,
            detail="unable to read nightly cycle history via daemon socket",
            status="WARN",
        )
    events, dropped = _normalize_socket_events(events_raw)
    if dropped:
        # Some windowed rows could not be interpreted (bad ts shape) -- the
        # verdict below would be computed on data known to be incomplete.
        # An empty windowed response (dropped == 0) still falls through to
        # the legitimate "no elapsed nightly window" PASS.
        return CheckResult(
            name=name,
            passed=True,
            detail="unable to interpret nightly cycle history via daemon socket",
            status="WARN",
        )
    return _evaluate_nightly_insight_events(events, now, no_history=False)


def check_bb_nightly_insight_mint(*, store: Any = None, now: Any = None) -> CheckResult:
    """Alarm when the nightly synthesis step stops producing insights.

    Counts consecutive fully-elapsed nightly windows with no confirmed
    successful mint. A confirmed mint in the current, not-yet-elapsed
    window still clears the alarm immediately; an empty or insight-less
    bucket for that same window is never counted toward the alarm.

    A caller-supplied ``store`` (tests, or a future in-process caller) is
    always read directly and never routes through the socket. Otherwise the
    daemon's steady-state lock hold means a same-process direct open would
    contend for the lock the daemon already has; the socket is tried first
    and a direct SHARED read-only open is the fallback for when the daemon
    is unreachable.
    """
    from datetime import datetime as _dt
    from datetime import timedelta as _td
    from datetime import timezone as _tz

    from iai_mcp import errors as _errors
    from iai_mcp.hippo import HippoLockHeldError

    name = _INSIGHT_MINT_CHECK_NAME
    now = now or _dt.now(_tz.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=_tz.utc)

    window_days = _INSIGHT_MINT_ALARM_THRESHOLD_NIGHTS + 10
    since = now - _td(days=window_days)

    if store is None:
        try:
            return _rem_events_via_socket(now, since=since)
        except _SocketUnavailable:
            pass
        except Exception as exc:  # noqa: BLE001 -- a diagnostic must never crash the run
            logger.debug("check_bb: socket read failed: %s", exc)
            return CheckResult(
                name=name,
                passed=True,
                detail=f"unable to read nightly cycle history: {type(exc).__name__}",
                status="WARN",
            )

    _store = store
    _owns_store = False

    try:
        if _store is None:
            from iai_mcp.doctor._storage_checks import _store_file_present

            if not _store_file_present():
                return CheckResult(
                    name=name,
                    passed=True,
                    detail="no store yet (skip)",
                    status="PASS",
                )
            from iai_mcp.hippo import AccessMode
            from iai_mcp.store import MemoryStore

            # SHARED + read-only coexists with the daemon's steady-state
            # SHARED lock (it downgrades from EXCLUSIVE after boot) instead of
            # contending for the EXCLUSIVE lock a default open would demand.
            # Hippo reads must stay daemon-independent — never gate a read on
            # the daemon holding the store's lock.
            _store = MemoryStore(access_mode=AccessMode.SHARED, read_only=True)
            _owns_store = True

        from iai_mcp.events import query_events

        probe = query_events(_store, kind="rem_cycle_completed", limit=1)
        if not probe:
            return _evaluate_nightly_insight_events([], now, no_history=True)

        events = query_events(
            _store,
            kind="rem_cycle_completed",
            since=since,
            limit=_INSIGHT_MINT_QUERY_LIMIT,
        )
        return _evaluate_nightly_insight_events(events, now, no_history=False)
    except HippoLockHeldError as exc:
        # A SHARED read-only open coexists with the daemon's steady-state
        # SHARED lock, so reaching this branch means something genuinely
        # unexpected is holding the store exclusively -- a real WARN, never
        # a blind PASS-defer.
        logger.debug("check_bb: store lock unavailable: %s", exc)
        return CheckResult(
            name=name,
            passed=True,
            detail=f"unable to read nightly cycle history: {type(exc).__name__}",
            status="WARN",
        )
    except _errors.OperationalError as exc:
        logger.debug("check_bb: query failed: %s", exc)
        return CheckResult(
            name=name,
            passed=True,
            detail=f"unable to read nightly cycle history: {type(exc).__name__}",
            status="WARN",
        )
    except Exception as exc:  # noqa: BLE001 -- a diagnostic must never crash the run
        logger.debug("check_bb: unexpected failure: %s", exc)
        return CheckResult(
            name=name,
            passed=True,
            detail=f"unable to read nightly cycle history: {type(exc).__name__}",
            status="WARN",
        )
    finally:
        if _owns_store and _store is not None and hasattr(_store, "close"):
            try:
                _store.close()
            except Exception:  # noqa: BLE001
                pass


_BACKGROUND_LIVENESS_CHECK_NAME: str = "(cc) background liveness"
_BACKGROUND_LIVENESS_LOOKBACK_DAYS: int = 7
_BACKGROUND_LIVENESS_ALARM_ROWS: int = 3


def _row_identity(row: dict[str, Any]) -> str | None:
    kind = row.get("event")
    if kind == "sleep_step_completed":
        step = row.get("step")
        return str(step) if step else None
    if kind == "promise_liveness":
        promise = row.get("promise")
        return str(promise) if promise else None
    return None


def _row_fully_saturated(row: dict[str, Any], candidates: int) -> bool:
    """True when a row's zero-processed candidates are all accounted for by
    a producer-reported saturation count (every labile candidate was already
    at its value ceiling, so the write was correctly declined) rather than
    left unexplained."""
    saturated = row.get("valence_saturated")
    return isinstance(saturated, int) and saturated == candidates


def _classify_liveness_window(rows: list[dict[str, Any]], alarm_rows: int) -> str:
    """Verdict for one identity's last ``alarm_rows`` recorded rows.

    Returns one of ``"fail"``, ``"warn"``, ``"unknown"``, ``"pass"``. A row
    carrying ``error`` is excluded from the window entirely (the crashed-step
    domain is the watchdog/quarantine check, not this one) rather than
    counted as evidence either way; a row with an unreadable
    candidate/processed/spec-version triple is excluded the same way but
    marks the identity unknown so an ambiguous window can never resolve to a
    silent PASS. A zero-processed window where every candidate is
    saturation-accounted-for (see ``_row_fully_saturated``) is healthy
    operation, not a stall.
    """
    window = sorted(rows, key=lambda r: r.get("ts") or "")[-alarm_rows:]

    usable: list[tuple[int, int, dict[str, Any]]] = []
    saw_unknown = False
    for row in window:
        if row.get("error") is not None:
            continue
        candidates = row.get("liveness_candidates")
        processed = row.get("liveness_processed")
        if (
            not isinstance(candidates, int)
            or not isinstance(processed, int)
            or not row.get("liveness_spec_version")
        ):
            saw_unknown = True
            continue
        usable.append((candidates, processed, row))

    if len(usable) == alarm_rows and all(c > 0 and p == 0 for c, p, _r in usable):
        if all(_row_fully_saturated(r, c) for c, _p, r in usable):
            return "pass"
        return "fail"
    if len(usable) == alarm_rows and all(c == 0 for c, _p, _r in usable):
        return "warn"
    if saw_unknown:
        return "unknown"
    return "pass"


def check_cc_background_liveness(*, now: Any = None) -> CheckResult:
    """Row-relative verdict over ``sleep_step_completed`` and ``promise_liveness`` rows.

    Reads the JSONL lifecycle log directly -- no store open, daemon-independent.
    Each identity (a sleep step by name, or a promise by id) is evaluated over
    its own last few recorded rows, never a wall-clock day bucket, because the
    covered promises run on unrelated clocks (per-cycle, hourly, per-session).
    """
    from datetime import datetime as _dt
    from datetime import timedelta as _td
    from datetime import timezone as _tz

    from iai_mcp.lifecycle_event_log import LifecycleEventLog

    name = _BACKGROUND_LIVENESS_CHECK_NAME
    now = now or _dt.now(_tz.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=_tz.utc)

    log_dir = _resolve_lifecycle_log_dir()
    if not log_dir.exists():
        return CheckResult(
            name=name,
            passed=True,
            detail="no event log yet (fresh install or daemon never run)",
            status="PASS",
        )

    log = LifecycleEventLog(log_dir=log_dir)

    rows_by_identity: dict[str, list[dict[str, Any]]] = {}
    for offset in range(_BACKGROUND_LIVENESS_LOOKBACK_DAYS):
        date_str = (now - _td(days=offset)).strftime("%Y-%m-%d")
        try:
            events = log.read_all(date_str=date_str)
        except OSError:
            # Absent or rotated (.jsonl.gz) day-file -- unreadable, not "no work".
            continue
        for ev in events:
            identity = _row_identity(ev)
            if identity is None:
                continue
            rows_by_identity.setdefault(identity, []).append(ev)

    if not rows_by_identity:
        return CheckResult(
            name=name,
            passed=True,
            detail="no liveness rows recorded yet",
            status="PASS",
        )

    fails: list[str] = []
    warns: list[str] = []
    unknowns: list[str] = []
    for identity, rows in sorted(rows_by_identity.items()):
        verdict = _classify_liveness_window(rows, _BACKGROUND_LIVENESS_ALARM_ROWS)
        if verdict == "fail":
            fails.append(identity)
        elif verdict == "warn":
            warns.append(identity)
        elif verdict == "unknown":
            unknowns.append(identity)

    if fails:
        detail = (
            f"{len(fails)} identity(ies) with {_BACKGROUND_LIVENESS_ALARM_ROWS} consecutive "
            f"no-op runs over a non-empty input: {', '.join(fails)}"
        )
        if unknowns:
            detail += f"; unknown effect: {', '.join(unknowns)}"
        return CheckResult(name=name, passed=False, detail=detail, status="FAIL")

    if warns:
        detail = f"possibly-starved input: {', '.join(warns)}"
        if unknowns:
            detail += f"; unknown effect: {', '.join(unknowns)}"
        return CheckResult(name=name, passed=True, detail=detail, status="WARN")

    if unknowns:
        return CheckResult(
            name=name,
            passed=True,
            detail=f"unknown effect: {', '.join(unknowns)}",
            status="PASS",
        )

    return CheckResult(
        name=name,
        passed=True,
        detail=f"{len(rows_by_identity)} identity(ies) healthy",
        status="PASS",
    )


_CODE_CURRENT_CHECK_NAME = "(ff) daemon sleep-path code current"


def check_ff_daemon_code_current() -> CheckResult:
    """Compare the daemon's boot-time source stamp against the source on disk.

    Never FAILs: the only remedy is a daemon stop+start, an operator decision.
    """
    from iai_mcp.daemon_state import load_state

    name = _CODE_CURRENT_CHECK_NAME
    try:
        state = load_state() or {}
    except Exception as e:  # noqa: BLE001 -- an unreadable state file is advisory here
        logger.debug("check_ff: daemon-state.json unreadable: %s", e)
        return CheckResult(
            name, True, f"daemon-state.json unreadable: {type(e).__name__}", status="WARN",
        )

    if state.get("daemon_pid") is None:
        return CheckResult(name, True, "no daemon booted")

    boot_stamp = state.get("code_stamp")
    try:
        from iai_mcp.code_stamp import stamp_divergence

        divergence = stamp_divergence(boot_stamp)
    except OSError as e:
        return CheckResult(
            name, True, f"sleep-path source unreadable: {type(e).__name__}: {e}", status="WARN",
        )

    if divergence == "missing":
        return CheckResult(
            name,
            True,
            "daemon booted without a code stamp — stop and start it to enable this check",
            status="WARN",
        )
    if divergence == "roots":
        booted_from = ", ".join(str(r) for r in (boot_stamp.get("roots") or []))
        return CheckResult(
            name,
            True,
            f"daemon booted from a different sleep-path source root: {booted_from}",
            status="WARN",
        )
    if divergence == "digest":
        return CheckResult(
            name,
            True,
            "daemon is running sleep-path code older than the source on disk — run "
            "`iai-mcp daemon stop && iai-mcp daemon start` to load it",
            status="WARN",
        )

    return CheckResult(
        name,
        True,
        f"sleep-path source matches the daemon boot stamp "
        f"({boot_stamp.get('files', 0)} files)",
    )


_BUILD_SKEW_CHECK_NAME = "(hh) daemon build matches installed package"


def check_hh_daemon_build_skew() -> CheckResult:
    """The daemon's own reported package version vs the installed one.

    Never FAILs; the only remedy is an operator-decided stop and start.
    Warns on the package-version dimension only -- the source-digest
    dimension stays owned by the check above, so one underlying skew never
    produces two warnings.
    """
    name = _BUILD_SKEW_CHECK_NAME
    from iai_mcp import doctor as _pkg

    status = None
    try:
        status = asyncio.run(
            _pkg._socket_status_probe(_pkg._resolve_socket_path(), 2.0)
        )
    except Exception as exc:  # noqa: BLE001 -- any probe failure reads as no-daemon
        logger.debug("check_hh: socket probe failed: %s", exc)

    if not isinstance(status, dict):
        return CheckResult(name, True, "no daemon reachable to compare against")

    try:
        from iai_mcp import __version__ as installed_version
    except (ImportError, AttributeError):
        installed_version = None

    daemon_version = status.get("version")

    if daemon_version is None:
        return CheckResult(
            name,
            True,
            "the daemon's status reply carries no version field -- stop and "
            "start it (`iai-mcp daemon stop && iai-mcp daemon start`) to "
            "make this row meaningful",
        )

    if installed_version is None:
        return CheckResult(
            name,
            True,
            f"daemon reports version {daemon_version}; the installed "
            "package version could not be determined",
        )

    if daemon_version != installed_version:
        return CheckResult(
            name,
            True,
            f"daemon is running version {daemon_version}, installed "
            f"package is {installed_version} -- run `iai-mcp daemon stop "
            "&& iai-mcp daemon start` to bring the daemon onto the "
            "installed build",
            status="WARN",
        )

    detail = f"daemon and installed package both report version {daemon_version}"
    try:
        from iai_mcp.code_stamp import stamp_divergence

        divergence = stamp_divergence(status.get("code_stamp"))
        stamp_state = "matches" if divergence is None else f"diverges ({divergence})"
        detail += f"; sleep-path source stamp {stamp_state}"
    except OSError as exc:
        logger.debug("check_hh: source stamp comparison failed: %s", exc)

    return CheckResult(name, True, detail)

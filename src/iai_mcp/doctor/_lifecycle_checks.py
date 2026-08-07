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

    iai_path = shutil.which("iai")
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

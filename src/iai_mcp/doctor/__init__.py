from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import re
import signal
import sqlite3
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


_LAUNCHD_REACT_DELAY_SEC = 2.0
_RESPAWN_BIND_TIMEOUT_SEC = 8.0
_RESPAWN_POLL_INTERVAL_SEC = 0.1


@dataclass
class CheckResult:

    name: str
    passed: bool
    detail: str
    status: str = ""

    def __post_init__(self) -> None:
        if not self.status:
            self.status = "PASS" if self.passed else "FAIL"


@dataclass
class RepairAction:

    label: str
    description: str
    destructive: bool
    execute: Callable[[], tuple[bool, str, int]]


def _resolve_socket_path() -> Path:
    env_path = os.environ.get("IAI_DAEMON_SOCKET_PATH")
    if env_path:
        return Path(env_path)
    from iai_mcp.cli import SOCKET_PATH

    return Path(SOCKET_PATH)


async def _socket_status_probe(socket_path: Path, timeout: float) -> dict | None:
    from iai_mcp._ipc import open_ipc_connection
    try:
        reader, writer = await open_ipc_connection(str(socket_path), timeout=timeout)
    except (FileNotFoundError, ConnectionRefusedError, asyncio.TimeoutError, OSError):
        return None
    try:
        writer.write((json.dumps({"type": "status"}) + "\n").encode("utf-8"))
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        if not line:
            return None
        return json.loads(line.decode("utf-8"))
    except Exception as exc:
        logger.debug("socket status probe failed: %s", exc)
        return None
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


def check_f_hippo_readable() -> CheckResult:
    import sqlite3

    from iai_mcp.hippo import HippoLockHeldError

    _s = None
    try:
        from iai_mcp.store import MemoryStore

        _s = MemoryStore()
        return CheckResult(
            "(f) hippo storage readable",
            True,
            "Hippo storage opens without error",
        )
    except HippoLockHeldError as e:
        logger.debug("check_f: store held by running daemon: %s", e)
        return CheckResult(
            "(f) hippo storage readable",
            True,
            "store held by the live daemon — normal",
        )
    except sqlite3.OperationalError as e:
        if "database is locked" in str(e).lower():
            logger.debug("check_f: store held by running daemon (sqlite): %s", e)
            return CheckResult(
                "(f) hippo storage readable",
                True,
                "store held by the live daemon — normal",
            )
        logger.debug("check_f: hippo storage open failed: %s", e)
        return CheckResult(
            "(f) hippo storage readable",
            False,
            f"open failed: {type(e).__name__}: {e}",
        )
    except Exception as e:  # noqa: BLE001 — surface any open failure
        logger.debug("check_f: hippo storage open failed: %s", e)
        return CheckResult(
            "(f) hippo storage readable",
            False,
            f"open failed: {type(e).__name__}: {e}",
        )
    finally:
        if _s is not None and hasattr(_s, "close"):
            try:
                _s.close()
            except Exception:  # noqa: BLE001
                pass


def _extract_binder_pids(lsof_output: str, target_socket: Path) -> set[int]:
    pids: set[int] = set()
    current_pid: int | None = None
    target = str(target_socket)
    for line in lsof_output.splitlines():
        if line.startswith("p"):
            try:
                current_pid = int(line[1:])
            except ValueError:
                current_pid = None
        elif line.startswith("n") and current_pid is not None:
            name = line[1:]
            if name == target:
                pids.add(current_pid)
    return pids


def _extract_binder_pids_ss(ss_output: str, target_socket: Path) -> set[int]:
    # Linux: lsof requires root to introspect other processes' /proc/<pid>/fd/;
    # `ss -lxp` reads the globally-readable /proc/net/unix and embeds the
    # binder pid in a `users:(("...",pid=N,fd=K))` field on the matching path line.
    #
    # Scope: the `pid=` field is only populated for sockets owned by the invoking
    # UID — `/proc/<pid>/fd/` is per-process-privileged. Cross-UID binders appear
    # in `ss` output but without a `users:(...)` field, so they are silently
    # skipped. This is acceptable: duplicate daemon processes always run under the
    # same UID as the doctor invocation.
    pids: set[int] = set()
    target = str(target_socket)
    for line in ss_output.splitlines():
        # `ss -lxp` columns: Netid State Recv-Q Send-Q LocalAddr:Port PeerAddr:Port Process
        # The Unix socket path is a whitespace-delimited token; require an exact
        # token match so /a/b.sock does not match /a/b.sock-old or /a/b.sock2.
        if target not in line.split():
            continue
        for m in re.finditer(r"pid=(\d+)", line):
            pids.add(int(m.group(1)))
    return pids


_HIPPO_EXPECTED_SCHEMA_VERSION = "1"


def _resolve_hippo_db_path() -> Path:
    env_path = os.environ.get("IAI_MCP_STORE")
    root = Path(env_path) if env_path else (Path.home() / ".iai-mcp")
    return root / "hippo" / "brain.sqlite3"


def _resolve_wrappers_dir() -> Path:
    env_path = os.environ.get("IAI_MCP_STORE")
    root = Path(env_path) if env_path else (Path.home() / ".iai-mcp")
    return root / "wrappers"


def _resolve_lifecycle_state_path() -> Path:
    env_path = os.environ.get("IAI_MCP_STORE")
    root = Path(env_path) if env_path else (Path.home() / ".iai-mcp")
    return root / "lifecycle_state.json"


def _resolve_lifecycle_log_dir() -> Path:
    env_path = os.environ.get("IAI_MCP_STORE")
    root = Path(env_path) if env_path else (Path.home() / ".iai-mcp")
    return root / "logs"


def _format_relative_short(ts_iso: str, *, now: Any = None) -> str:
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    try:
        ts = _dt.fromisoformat(ts_iso)
    except (TypeError, ValueError):
        return "?"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=_tz.utc)
    moment = now if now is not None else _dt.now(_tz.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=_tz.utc)
    seconds = int((moment - ts).total_seconds())
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min"
    hours = minutes // 60
    if hours < 48:
        return f"{hours} h"
    days = hours // 24
    return f"{days} d"


def _format_top_of_output_hint(results: list[CheckResult]) -> str | None:
    for r in results:
        if r.name == "(h) crypto key file state" and r.status == "WARN":
            flat = " ".join(line.strip() for line in r.detail.splitlines() if line.strip())
            return f"> hint: {flat}"
    return None


_HEADLESS_DOWNGRADE_ROWS: frozenset[str] = frozenset({
    "(b) socket file fresh",
    "(n) HID idle source",
})


def is_headless(*, force: bool = False) -> bool:
    if force:
        return True
    if platform.system() != "Linux":
        return False
    return (
        os.environ.get("DISPLAY") is None
        and os.environ.get("WAYLAND_DISPLAY") is None
    )


def _apply_headless_downgrade(
    results: list[CheckResult], headless: bool
) -> list[CheckResult]:
    if not headless:
        return results
    for r in results:
        if r.name in _HEADLESS_DOWNGRADE_ROWS and r.status == "FAIL":
            r.passed = True
            r.status = "WARN"
    return results


def run_diagnosis() -> list[CheckResult]:
    return [
        check_a_daemon_alive(),
        check_b_socket_fresh(),
        check_c_lock_healthy(),
        check_d_no_orphan_core(),
        check_e_state_file_valid(),
        check_f_hippo_readable(),
        check_g_no_dup_binders(),
        check_h_crypto_file_state(),
        check_i_hippo_db_size(),
        check_j_lifecycle_current_state(),
        check_k_lifecycle_history_24h(),
        check_l_sleep_cycle_status(),
        check_m_heartbeat_scanner(),
        check_n_hid_idle_source(),
        check_o_subscription_credentials(),
        check_p_anthropic_sdk_absent(),
        check_q_iai_cli_reachable(),
        check_r_hippo_hnsw_loadable(),
        check_s_hippo_schema_version(),
        check_t_hippo_compacted_freshness(),
        check_u_recall_centrality_regression(),
        check_v_native_embedder(),
        check_w_no_permanent_failed(),
        check_x_no_collapsed_timestamps(),
        check_y_rss_24h_plateau(),
        check_z_avx2_support(),
    ]


def print_checklist(results: list[CheckResult]) -> None:
    print("iai doctor — daemon health check\n")
    for r in results:
        if r.status == "WARN":
            tag = "[WARN]"
        elif r.passed:
            tag = "[PASS]"
        else:
            tag = "[FAIL]"
        print(f"  {tag} {r.name:<40} {r.detail}")


def _kill_orphan_cores() -> tuple[bool, str, int]:
    import psutil

    t0 = time.monotonic()
    killed: list[int] = []
    failed: list[tuple[int, str]] = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cl = " ".join(p.info.get("cmdline") or [])
            if "iai_mcp.core" not in cl:
                continue
            pid = p.info["pid"]
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        except OSError as e:
            failed.append((p.info.get("pid", -1), str(e)))
    duration_ms = int((time.monotonic() - t0) * 1000)
    if failed:
        return (
            False,
            f"killed {len(killed)} ({killed}); FAILED on {failed}",
            duration_ms,
        )
    return True, f"killed {len(killed)} orphan(s): {killed}", duration_ms


def _unlink_stale_socket() -> tuple[bool, str, int]:
    socket_path = _resolve_socket_path()
    t0 = time.monotonic()
    if not socket_path.exists():
        return True, "no stale socket to unlink", int((time.monotonic() - t0) * 1000)
    try:
        socket_path.unlink()
        return True, f"unlinked {socket_path}", int((time.monotonic() - t0) * 1000)
    except OSError as e:
        return False, f"unlink failed: {e}", int((time.monotonic() - t0) * 1000)


def _respawn_daemon() -> tuple[bool, str, int]:
    from iai_mcp.cli import LAUNCHD_TARGET, SYSTEMD_TARGET, SERVICE_NAME

    t0 = time.monotonic()
    socket_path = _resolve_socket_path()

    using_default_socket = os.environ.get("IAI_DAEMON_SOCKET_PATH") is None
    if (
        using_default_socket
        and LAUNCHD_TARGET
        and Path(LAUNCHD_TARGET).expanduser().exists()
    ):
        time.sleep(_LAUNCHD_REACT_DELAY_SEC)
        return (
            True,
            "launchd-managed (KeepAlive will respawn)",
            int((time.monotonic() - t0) * 1000),
        )

    if (
        using_default_socket
        and platform.system() == "Linux"
        and SYSTEMD_TARGET
        and Path(SYSTEMD_TARGET).expanduser().exists()
    ):
        subprocess.run(
            ["systemctl", "--user", "start", SERVICE_NAME],
            check=False, capture_output=True,
        )

    try:
        spawn_env = os.environ.copy()
        spawn_env["IAI_DAEMON_RESPAWN_BY"] = "doctor"
        subprocess.Popen(
            [sys.executable, "-m", "iai_mcp.daemon"],
            env=spawn_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as e:  # noqa: BLE001 — spawn failure is a recovery error
        logger.debug("respawn daemon failed: %s", e)
        return (
            False,
            f"respawn failed: {type(e).__name__}: {e}",
            int((time.monotonic() - t0) * 1000),
        )

    deadline = time.monotonic() + _RESPAWN_BIND_TIMEOUT_SEC
    while time.monotonic() < deadline:
        if socket_path.exists():
            duration_ms = int((time.monotonic() - t0) * 1000)
            return (
                True,
                f"daemon respawned (socket bound in {duration_ms} ms)",
                duration_ms,
            )
        time.sleep(_RESPAWN_POLL_INTERVAL_SEC)
    duration_ms = int((time.monotonic() - t0) * 1000)
    return (
        False,
        f"daemon respawn timed out (socket not bound after {_RESPAWN_BIND_TIMEOUT_SEC}s)",
        duration_ms,
    )


def _kill_dup_binders() -> tuple[bool, str, int]:
    import psutil

    t0 = time.monotonic()
    socket_path = _resolve_socket_path()
    try:
        result = subprocess.run(
            ["lsof", "-U", "-F", "pn"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return (
            False,
            f"lsof unavailable: {e}",
            int((time.monotonic() - t0) * 1000),
        )
    binder_pids = _extract_binder_pids(result.stdout, socket_path)
    if not binder_pids and platform.system() == "Linux" and socket_path.exists():
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
        return (
            True,
            f"{len(binder_pids)} dup binders to kill",
            int((time.monotonic() - t0) * 1000),
        )

    pid_etimes: list[tuple[int, float]] = []
    for pid in binder_pids:
        try:
            p = psutil.Process(pid)
            create_time = p.create_time()
            pid_etimes.append((pid, time.time() - create_time))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if not pid_etimes:
        return (
            False,
            "all binders disappeared between lsof and psutil",
            int((time.monotonic() - t0) * 1000),
        )

    pid_etimes.sort(key=lambda x: x[1], reverse=True)
    keep_pid = pid_etimes[0][0]
    kill_candidates = [pid for pid, _ in pid_etimes[1:]]

    killed: list[int] = []
    for pid in kill_candidates:
        try:
            p = psutil.Process(pid)
            cmdline = " ".join(p.cmdline() or [])
            if "iai_mcp.daemon" not in cmdline:
                continue
            p.kill()
            killed.append(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    time.sleep(_LAUNCHD_REACT_DELAY_SEC)
    return (
        True,
        f"kept PID {keep_pid} (oldest); killed {killed}",
        int((time.monotonic() - t0) * 1000),
    )


def _plan_repair_actions(results: list[CheckResult]) -> list[RepairAction]:
    actions: list[RepairAction] = []
    fail_names = {r.name for r in results if not r.passed}

    if "(b) socket file fresh" in fail_names:
        actions.append(
            RepairAction(
                label="unlink_stale_socket",
                description="unlink stale ~/.iai-mcp/.daemon.sock",
                destructive=True,
                execute=_unlink_stale_socket,
            )
        )

    if "(g) no dup binders" in fail_names:
        actions.append(
            RepairAction(
                label="kill_dup_binders",
                description="keep oldest-etime daemon binder, SIGKILL the rest",
                destructive=True,
                execute=_kill_dup_binders,
            )
        )

    if "(d) no orphan iai_mcp.core procs" in fail_names:
        actions.append(
            RepairAction(
                label="kill_orphan_cores",
                description="SIGTERM every orphan iai_mcp.core process",
                destructive=True,
                execute=_kill_orphan_cores,
            )
        )

    if "(a) daemon process alive" in fail_names:
        actions.append(
            RepairAction(
                label="respawn_daemon",
                description="spawn `python -m iai_mcp.daemon` detached",
                destructive=True,
                execute=_respawn_daemon,
            )
        )

    return actions


def _prompt_action(action: RepairAction) -> bool:
    try:
        response = input(f"  [y/N] {action.description}: ")
    except EOFError:
        response = ""
    return response.strip().lower() == "y"


def cmd_doctor(args: argparse.Namespace) -> int:
    apply = bool(getattr(args, "apply", False))
    yes = bool(getattr(args, "yes", False))
    if yes and not apply:
        print(
            "[warn] --yes without --apply is meaningless; ignoring --yes.",
            file=sys.stderr,
        )

    results = run_diagnosis()
    headless = is_headless(force=bool(getattr(args, "headless", False)))
    results = _apply_headless_downgrade(results, headless)
    total = len(results)
    hint = _format_top_of_output_hint(results)
    if hint is not None:
        print(hint)
        print()
    print_checklist(results)
    fail_count = sum(1 for r in results if not r.passed)

    if fail_count == 0:
        print("\nAll checks passed. Exit 0.")
        return 0

    if not apply:
        print(
            f"\n{fail_count}/{total} FAIL. Run with --apply to attempt recovery. Exit 1."
        )
        return 1

    print(
        f"\n{fail_count}/{total} FAIL. Attempting recovery (--apply{' --yes' if yes else ''}):\n"
    )
    actions = _plan_repair_actions(results)
    if not actions:
        print(
            "(no automated repair actions for the FAILs above; manual intervention required)"
        )
    for action in actions:
        if action.destructive and not yes:
            if not _prompt_action(action):
                print(f"  [skipped] {action.description}")
                continue
        ok, msg, ms = action.execute()
        tag = "[done]" if ok else "[FAIL]"
        print(f"  {tag} {action.label}: {msg} ({ms} ms)")
        try:
            from iai_mcp.events import write_event
            from iai_mcp.store import MemoryStore

            with MemoryStore() as _audit_store:
                write_event(
                    _audit_store,
                    kind="doctor_action",
                    data={
                        "action": action.label,
                        "target": action.description,
                        "success": ok,
                        "duration_ms": ms,
                        "detail": msg,
                    },
                )
        except Exception as e:
            logger.debug("doctor audit event write failed: %s", e)

    print("\nRe-running checks ...")
    final_results = run_diagnosis()
    print_checklist(final_results)
    final_fails = [r.name for r in final_results if not r.passed]
    if not final_fails:
        print(f"\nFIXED. All {len(final_results)} checks pass. Exit 0.")
        return 0
    print(f"\nSTILL BROKEN: {final_fails}. Exit 2.")
    return 2


# Check functions are split into two concern-grouped sub-modules and re-exported
# here. The import runs after the spine above is defined, so the sub-modules can
# import the stable spine helpers from this partially-initialized package.
from iai_mcp.doctor._lifecycle_checks import (
    _socket_connect_probe,
    check_a_daemon_alive,
    check_b_socket_fresh,
    check_c_lock_healthy,
    check_d_no_orphan_core,
    check_e_state_file_valid,
    check_g_no_dup_binders,
    check_j_lifecycle_current_state,
    check_k_lifecycle_history_24h,
    check_l_sleep_cycle_status,
    check_m_heartbeat_scanner,
    check_n_hid_idle_source,
    check_o_subscription_credentials,
    check_q_iai_cli_reachable,
    check_y_rss_24h_plateau,
)
from iai_mcp.doctor._storage_checks import (
    check_h_crypto_file_state,
    check_i_hippo_db_size,
    check_p_anthropic_sdk_absent,
    check_r_hippo_hnsw_loadable,
    check_s_hippo_schema_version,
    check_t_hippo_compacted_freshness,
    check_u_recall_centrality_regression,
    check_v_native_embedder,
    check_w_no_permanent_failed,
    check_x_no_collapsed_timestamps,
    check_z_avx2_support,
)

__all__ = [
    "CheckResult",
    "RepairAction",
    "cmd_doctor",
    "run_diagnosis",
    "print_checklist",
    "is_headless",
    "_apply_headless_downgrade",
    "_format_top_of_output_hint",
    "_extract_binder_pids",
    "_extract_binder_pids_ss",
    "_resolve_hippo_db_path",
    "_kill_dup_binders",
    "check_a_daemon_alive",
    "check_b_socket_fresh",
    "check_c_lock_healthy",
    "check_d_no_orphan_core",
    "check_e_state_file_valid",
    "check_f_hippo_readable",
    "check_g_no_dup_binders",
    "check_h_crypto_file_state",
    "check_i_hippo_db_size",
    "check_j_lifecycle_current_state",
    "check_k_lifecycle_history_24h",
    "check_l_sleep_cycle_status",
    "check_m_heartbeat_scanner",
    "check_n_hid_idle_source",
    "check_o_subscription_credentials",
    "check_p_anthropic_sdk_absent",
    "check_q_iai_cli_reachable",
    "check_r_hippo_hnsw_loadable",
    "check_s_hippo_schema_version",
    "check_t_hippo_compacted_freshness",
    "check_u_recall_centrality_regression",
    "check_v_native_embedder",
    "check_w_no_permanent_failed",
    "check_x_no_collapsed_timestamps",
    "check_y_rss_24h_plateau",
    "check_z_avx2_support",
]

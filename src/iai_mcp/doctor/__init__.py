from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import re
import signal
from iai_mcp import errors
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
    # Eligible for the unattended `doctor --auto` reflex. The auto subset
    # must never kill a process, never mutate the store, and never lose
    # data — quarantine-renames and derived-file heals only.
    auto_safe: bool = False


def _resolve_socket_path() -> Path:
    env_path = os.environ.get("IAI_DAEMON_SOCKET_PATH")
    if env_path:
        return Path(env_path)
    from iai_mcp.cli import SOCKET_PATH

    return Path(SOCKET_PATH)


async def _socket_status_probe(socket_path: Path, timeout: float) -> dict | None:
    try:
        from iai_mcp._ipc import open_ipc_connection
        reader, writer = await asyncio.wait_for(
            open_ipc_connection(str(socket_path)),
            timeout=timeout,
        )
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


_HIPPO_READABLE_CHECK_NAME = "(f) hippo storage readable"


def _check_f_via_socket() -> CheckResult:
    """Prove Hippo is readable through the daemon socket.

    ``session_started`` is already whitelisted for ``events_query`` — a
    non-error reply carrying an ``events`` list (even empty) proves the
    daemon executed a real ``query_events(store, ...)`` against the store.
    Deliberately NOT ``_socket_status_probe``: the ``status`` reply is
    assembled from the daemon's in-memory state and never reads the store,
    so it would reproduce the same blind PASS in new clothes. Raises
    ``_SocketUnavailable`` when the daemon cannot be reached at all, so the
    caller falls back to a direct store open.
    """
    from iai_mcp.cli import _send_jsonrpc_request
    from iai_mcp.doctor._lifecycle_checks import _SocketUnavailable

    resp = _send_jsonrpc_request(
        "events_query",
        {"kind": "session_started", "limit": 1},
        connect_timeout=1.0,
        read_timeout=5.0,
    )
    if resp is None:
        raise _SocketUnavailable()
    if not isinstance(resp, dict) or "result" not in resp:
        return CheckResult(
            _HIPPO_READABLE_CHECK_NAME,
            True,
            "unable to confirm Hippo readability via the daemon socket",
            status="WARN",
        )
    result = resp["result"]
    events = result.get("events") if isinstance(result, dict) else None
    if events is None:
        return CheckResult(
            _HIPPO_READABLE_CHECK_NAME,
            True,
            "unable to confirm Hippo readability via the daemon socket",
            status="WARN",
        )
    return CheckResult(
        _HIPPO_READABLE_CHECK_NAME,
        True,
        "Hippo storage readable via the running daemon",
    )


def check_f_hippo_readable() -> CheckResult:
    from iai_mcp.doctor._lifecycle_checks import _SocketUnavailable
    from iai_mcp.hippo import HippoLockHeldError

    from iai_mcp.doctor._storage_checks import _store_file_present

    if not _store_file_present():
        try:
            from iai_mcp import store as _store_mod

            _root = str(
                os.environ.get("IAI_MCP_STORE") or _store_mod.DEFAULT_STORAGE_PATH
            )
        except Exception:  # noqa: BLE001 -- naming the path is best-effort
            _root = "the configured store root"
        return CheckResult(
            _HIPPO_READABLE_CHECK_NAME,
            True,
            f"no store yet at {_root} — nothing to read",
            status="WARN",
        )

    try:
        return _check_f_via_socket()
    except _SocketUnavailable:
        pass
    except Exception as exc:  # noqa: BLE001 — a diagnostic must never crash the run
        logger.debug("check_f: socket read failed: %s", exc)
        return CheckResult(
            _HIPPO_READABLE_CHECK_NAME,
            True,
            "unable to confirm Hippo readability via the daemon socket",
            status="WARN",
        )

    _s = None
    try:
        from iai_mcp.hippo import AccessMode
        from iai_mcp.store import MemoryStore

        # SHARED + read-only coexists with the daemon's steady-state SHARED
        # lock; reaching a lock error here means something unexpected holds
        # the store exclusively, not a "daemon holds it, normal" condition.
        _s = MemoryStore(access_mode=AccessMode.SHARED, read_only=True)
        return CheckResult(
            _HIPPO_READABLE_CHECK_NAME,
            True,
            "Hippo storage opens without error",
        )
    except HippoLockHeldError as e:
        logger.debug("check_f: store lock unavailable: %s", e)
        return CheckResult(
            _HIPPO_READABLE_CHECK_NAME,
            True,
            f"unable to open Hippo storage: {type(e).__name__}",
            status="WARN",
        )
    except errors.OperationalError as e:
        if "database is locked" in str(e).lower():
            logger.debug("check_f: store lock unavailable (sqlite): %s", e)
            return CheckResult(
                _HIPPO_READABLE_CHECK_NAME,
                True,
                f"unable to open Hippo storage: {type(e).__name__}",
                status="WARN",
            )
        logger.debug("check_f: hippo storage open failed: %s", e)
        return CheckResult(
            _HIPPO_READABLE_CHECK_NAME,
            False,
            f"open failed: {type(e).__name__}: {e}",
        )
    except Exception as e:  # noqa: BLE001 — surface any open failure
        logger.debug("check_f: hippo storage open failed: %s", e)
        return CheckResult(
            _HIPPO_READABLE_CHECK_NAME,
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


def check_plus_update_available(*, fetch: bool = True) -> CheckResult:
    # Notify-only: never FAILs. Fetch respects the daily TTL and the env
    # kill switch; offline keeps the last known answer silently. The
    # unattended reflex passes fetch=False — a degraded-daemon heal must
    # not wait on a PyPI lookup.
    from iai_mcp.version_check import (
        check_enabled,
        installed_version,
        is_editable_install,
        pending_update,
        refresh_cache,
    )

    name = "(+) update available"
    if not check_enabled():
        return CheckResult(name, True, "version check disabled", status="PASS")
    if is_editable_install():
        return CheckResult(
            name, True, "source checkout — updates via git, not pip", status="PASS",
        )
    if fetch:
        try:
            refresh_cache()
        except Exception as e:  # noqa: BLE001 — probe failure is advisory
            logger.debug("check_plus: refresh failed: %s", e)
    pair = pending_update()
    if pair is not None:
        current, latest = pair
        return CheckResult(
            name,
            True,
            f"iai-pme {latest} available (installed {current}) — "
            f"run `iai-mcp self-update`",
            status="WARN",
        )
    cur = installed_version() or "?"
    return CheckResult(name, True, f"up to date ({cur}) or no version data", status="PASS")


def run_diagnosis(*, fetch_update: bool = True) -> list[CheckResult]:
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
        check_ii_embed_identity(),
        check_w_no_permanent_failed(),
        check_x_no_collapsed_timestamps(),
        check_aa_capture_state_hygiene(),
        check_y_rss_24h_plateau(),
        check_bb_nightly_insight_mint(),
        check_cc_background_liveness(),
        check_z_avx2_support(),
        check_plus_update_available(fetch=fetch_update),
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


def _store_root() -> Path:
    env_path = os.environ.get("IAI_MCP_STORE")
    return Path(env_path) if env_path else (Path.home() / ".iai-mcp")


def _quarantine_rename(path: Path) -> tuple[bool, str, int]:
    # Never delete: rename aside with a timestamp so the file stays
    # recoverable and the consumer regenerates a fresh one.
    from datetime import datetime, timezone

    t0 = time.monotonic()
    if not path.exists():
        return True, f"{path} already absent", int((time.monotonic() - t0) * 1000)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = path.with_name(f"{path.name}.corrupt-{stamp}")
    try:
        path.rename(target)
    except OSError as e:
        return False, f"rename failed: {e}", int((time.monotonic() - t0) * 1000)
    return True, f"quarantined to {target.name}", int((time.monotonic() - t0) * 1000)


def _write_wake_signal() -> tuple[bool, str, int]:
    from datetime import datetime, timezone

    t0 = time.monotonic()
    from iai_mcp.wake_handler import wake_signal_path

    path = wake_signal_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".doctor.tmp")
        tmp.write_text(json.dumps({
            "requested_at": datetime.now(timezone.utc).isoformat(),
            "source": "doctor",
        }), encoding="utf-8")
        tmp.rename(path)
    except OSError as e:
        return False, f"wake.signal write failed: {e}", int((time.monotonic() - t0) * 1000)
    return True, f"wake.signal written at {path}", int((time.monotonic() - t0) * 1000)


#: Cold boot binds the socket late (index load); give the parked-engine
#: wake more headroom than the plain respawn probe.
_PARKED_WAKE_BIND_TIMEOUT_SEC = 20.0


def _wake_parked_engine() -> tuple[bool, str, int]:
    # Signal first, then start: a daemon booted before the signal lands
    # would restore the parked state and exit again. The start must be an
    # EXPLICIT one (cmd_daemon_start = launchctl bootstrap+kickstart /
    # systemctl start / spawn): a hibernation exit is exit 0, so launchd's
    # KeepAlive leaves the job down and trusting it is a no-op.
    import argparse as _ap

    t0 = time.monotonic()
    ok_sig, msg_sig, _ = _write_wake_signal()
    if not ok_sig:
        return False, msg_sig, int((time.monotonic() - t0) * 1000)
    try:
        from iai_mcp.cli._daemon import cmd_daemon_start

        rc = cmd_daemon_start(_ap.Namespace())
    except Exception as e:  # noqa: BLE001 — start failure is a recovery error
        return (
            False,
            f"{msg_sig}; daemon start failed: {type(e).__name__}: {e}",
            int((time.monotonic() - t0) * 1000),
        )
    if rc != 0:
        return (
            False,
            f"{msg_sig}; daemon start exited {rc}",
            int((time.monotonic() - t0) * 1000),
        )
    socket_path = _resolve_socket_path()
    deadline = time.monotonic() + _PARKED_WAKE_BIND_TIMEOUT_SEC
    while time.monotonic() < deadline:
        if socket_path.exists():
            duration_ms = int((time.monotonic() - t0) * 1000)
            return True, f"{msg_sig}; daemon up (socket bound)", duration_ms
        time.sleep(_RESPAWN_POLL_INTERVAL_SEC)
    # A slow cold boot may still finish after this window — report honestly.
    return (
        False,
        f"{msg_sig}; start requested but socket not bound in "
        f"{_PARKED_WAKE_BIND_TIMEOUT_SEC:.0f}s",
        int((time.monotonic() - t0) * 1000),
    )


def _cleanup_stale_heartbeats() -> tuple[bool, str, int]:
    from iai_mcp.heartbeat_scanner import HeartbeatScanner

    t0 = time.monotonic()
    try:
        deleted = HeartbeatScanner(_resolve_wrappers_dir()).cleanup_stale_orphans()
    except OSError as e:
        return False, f"cleanup failed: {e}", int((time.monotonic() - t0) * 1000)
    return (
        True,
        f"removed {deleted} stale/orphan heartbeat file(s)",
        int((time.monotonic() - t0) * 1000),
    )


def _quarantine_state_file() -> tuple[bool, str, int]:
    from iai_mcp.daemon_state import daemon_state_path

    return _quarantine_rename(daemon_state_path())


def _quarantine_vec_index() -> tuple[bool, str, int]:
    return _quarantine_rename(_resolve_hippo_db_path().parent / "records.hnsw")


def _reset_sleep_quarantine() -> tuple[bool, str, int]:
    t0 = time.monotonic()
    try:
        from iai_mcp.lifecycle_event_log import LifecycleEventLog
        from iai_mcp.lifecycle_state import lifecycle_state_path
        from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline

        # The quarantine reset path touches only the lifecycle state file
        # and the event log — store=None is safe here. Paths MUST go through
        # the env-honoring resolvers: the module constants are home-rooted
        # and would clobber the production record under IAI_MCP_STORE.
        pipeline = SleepPipeline(
            store=None,
            lifecycle_state_path=lifecycle_state_path(),
            event_log=LifecycleEventLog(log_dir=_resolve_lifecycle_log_dir()),
        )
        pipeline.reset_quarantine()
    except Exception as e:  # noqa: BLE001 — surface any reset failure
        return False, f"quarantine reset failed: {e}", int((time.monotonic() - t0) * 1000)
    return True, "sleep-cycle quarantine cleared", int((time.monotonic() - t0) * 1000)


def _run_own_cli(argv: list[str], timeout_sec: float) -> tuple[bool, str, int]:
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "iai_mcp.cli", *argv],
            capture_output=True, text=True, timeout=timeout_sec, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"{argv[0]} failed: {e}", int((time.monotonic() - t0) * 1000)
    duration_ms = int((time.monotonic() - t0) * 1000)
    tail = (proc.stdout or proc.stderr or "").strip().splitlines()[-1:]
    detail = tail[0][:160] if tail else ""
    if proc.returncode != 0:
        return False, f"exit {proc.returncode}: {detail}", duration_ms
    return True, detail or "done", duration_ms


def _drain_permanent_failed() -> tuple[bool, str, int]:
    return _run_own_cli(["drain-permanent-failed"], timeout_sec=300.0)


def _compact_hippo() -> tuple[bool, str, int]:
    # Explicit --store-path: the maintenance verb's default is home-rooted
    # and would target the wrong store under IAI_MCP_STORE.
    return _run_own_cli(
        [
            "maintenance", "compact-hippo", "--apply", "--yes",
            "--store-path", str(_store_root()),
        ],
        timeout_sec=1800.0,
    )


def _rederive_timestamps() -> tuple[bool, str, int]:
    return _run_own_cli(["migrate", "--rederive-timestamps"], timeout_sec=1800.0)


def _sweep_capture_state_heal() -> tuple[bool, str, int]:
    from iai_mcp.capture import sweep_capture_state

    counts = sweep_capture_state(apply=True)
    return True, f"removed {counts['removed']} stale capture-state file(s)", 0


def _plan_repair_actions(results: list[CheckResult]) -> list[RepairAction]:
    actions: list[RepairAction] = []
    fail_names = {r.name for r in results if not r.passed}
    by_name = {r.name: r for r in results}

    if "(aa) capture-state hygiene" in fail_names:
        actions.append(
            RepairAction(
                label="sweep_capture_state",
                description=(
                    "remove month-old capture-state files and day-old "
                    "crashed-writer tmp debris"
                ),
                destructive=False,
                execute=_sweep_capture_state_heal,
                # Derived-file class: a swept offset re-walks an idempotent
                # transcript, never loses store data.
                auto_safe=True,
            )
        )

    if "(b) socket file fresh" in fail_names:
        actions.append(
            RepairAction(
                label="unlink_stale_socket",
                description="unlink stale ~/.iai-mcp/.daemon.sock",
                destructive=True,
                execute=_unlink_stale_socket,
                # (b) alone fails on a 1s connect timeout — a daemon merely
                # busy in consolidation. Unattended unlink is allowed only
                # when the daemon process is provably gone too.
                auto_safe="(a) daemon process alive" in fail_names,
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

    if "(j) lifecycle current state" in fail_names:
        # Parked-engine deadlock: persisted HIBERNATION/SLEEP with no live
        # daemon. Signal-first wake, then start — respawn alone would boot
        # back into the parked state on an engine without the demand wake.
        actions.append(
            RepairAction(
                label="wake_parked_engine",
                description="write wake.signal, then start the daemon",
                destructive=False,
                execute=_wake_parked_engine,
                auto_safe=True,
            )
        )

    if "(a) daemon process alive" in fail_names and (
        "(j) lifecycle current state" not in fail_names
    ):
        actions.append(
            RepairAction(
                label="respawn_daemon",
                description="spawn `python -m iai_mcp.daemon` detached",
                destructive=True,
                execute=_respawn_daemon,
                auto_safe=True,
            )
        )

    _e = by_name.get("(e) daemon state file valid")
    if _e is not None and not _e.passed and _e.detail.startswith("unreadable"):
        # Only the unreadable/corrupt case — a merely-unknown fsm_state may
        # be a version skew a newer daemon still parses.
        actions.append(
            RepairAction(
                label="quarantine_state_file",
                description=(
                    "rename the corrupt daemon state file aside "
                    "(daemon regenerates it on next boot)"
                ),
                destructive=False,
                execute=_quarantine_state_file,
                auto_safe=True,
            )
        )

    _r = by_name.get("(r) hippo hnsw index")
    if _r is not None and not _r.passed and "stat failed" not in _r.detail:
        actions.append(
            RepairAction(
                label="quarantine_vec_index",
                description=(
                    "rename the corrupt vector index aside "
                    "(rebuilds from SQLite on next daemon boot)"
                ),
                destructive=False,
                execute=_quarantine_vec_index,
                auto_safe=True,
            )
        )

    _m = by_name.get("(m) heartbeat scanner")
    if _m is not None and _m.passed and re.search(
        r"\b[1-9]\d* (?:stale|orphan)", _m.detail
    ):
        actions.append(
            RepairAction(
                label="cleanup_stale_heartbeats",
                description="delete stale/orphan wrapper heartbeat files",
                destructive=False,
                execute=_cleanup_stale_heartbeats,
                auto_safe=True,
            )
        )

    if "(l) sleep cycle quarantine" in fail_names:
        actions.append(
            RepairAction(
                label="reset_sleep_quarantine",
                description="clear the stuck (>=12h) sleep-cycle quarantine",
                destructive=False,
                execute=_reset_sleep_quarantine,
                auto_safe=True,
            )
        )

    _w = by_name.get("(w) no permanent-failed captures")
    if _w is not None and _w.status == "WARN" and "permanent-failed capture" in _w.detail:
        actions.append(
            RepairAction(
                label="drain_permanent_failed",
                description="recover permanent-failed capture files into the store",
                destructive=False,
                execute=_drain_permanent_failed,
            )
        )

    _x = by_name.get("(x) no collapsed-timestamp groups")
    if _x is not None and _x.status == "WARN" and "group(s) with" in _x.detail:
        actions.append(
            RepairAction(
                label="rederive_timestamps",
                description="repair collapsed timestamps (store mutation)",
                destructive=True,
                execute=_rederive_timestamps,
            )
        )

    if "(i) hippo db size" in fail_names:
        actions.append(
            RepairAction(
                label="compact_hippo",
                description="compact the oversized store (long-running)",
                destructive=True,
                execute=_compact_hippo,
            )
        )

    return actions


def _prompt_action(action: RepairAction) -> bool:
    try:
        response = input(f"  [y/N] {action.description}: ")
    except EOFError:
        response = ""
    return response.strip().lower() == "y"


#: Minimum spacing between unattended auto-heal runs. Damps heal loops: a
#: fault the auto subset cannot fix must not be re-poked every session start.
DOCTOR_AUTO_COOLDOWN_SEC: float = 6 * 3600.0


def _auto_damper_path() -> Path:
    return _store_root() / ".doctor-auto-last"


def _auto_recently_ran(now: float | None = None) -> bool:
    path = _auto_damper_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return False
    moment = now if now is not None else time.time()
    return (moment - mtime) < DOCTOR_AUTO_COOLDOWN_SEC


def _stamp_auto_run() -> None:
    path = _auto_damper_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    except OSError as e:
        logger.debug("auto damper stamp failed: %s", e)


def _execute_action(action: RepairAction) -> tuple[bool, str, int]:
    ok, msg, ms = action.execute()
    try:
        from iai_mcp.doctor._storage_checks import _store_file_present
        from iai_mcp.events import write_event
        from iai_mcp.store import MemoryStore

        if not _store_file_present():
            # The audit trail is best-effort; minting a store to hold it
            # is worse than skipping it.
            logger.debug(
                "doctor heal audit skipped: no store yet (action=%s)",
                action.label,
            )
            return ok, msg, ms
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
    except Exception as e:  # noqa: BLE001 — audit is best-effort
        logger.debug("doctor audit event write failed: %s", e)
    return ok, msg, ms


def cmd_doctor_auto() -> int:
    """Unattended reflex: run only the auto-safe heal subset, no prompts.

    Never kills a process, never mutates the store, never loses data —
    quarantine-renames, signal writes, heartbeat cleanup, daemon starts.
    """
    if _auto_recently_ran():
        print("doctor --auto: cooldown active, skipping.")
        return 0
    # Stamp IMMEDIATELY after passing the cooldown: every branch below —
    # heals, no-auto-heal-available, even all-green — must be damped, or a
    # persistent unfixable fault re-runs the full diagnosis on every session
    # start forever. This also shrinks the concurrent-session window.
    _stamp_auto_run()

    # fetch_update=False: the heal reflex fires when the daemon is down,
    # often together with a broken network — never wait on PyPI here.
    results = run_diagnosis(fetch_update=False)
    results = _apply_headless_downgrade(results, is_headless())
    actions = [a for a in _plan_repair_actions(results) if a.auto_safe]
    if not actions:
        fails = [r.name for r in results if not r.passed]
        if fails:
            print(f"doctor --auto: no auto-safe heal for {fails}; run `iai-mcp doctor --apply`.")
            return 1
        print("doctor --auto: all checks green, nothing to heal.")
        return 0

    for action in actions:
        ok, msg, ms = _execute_action(action)
        tag = "[done]" if ok else "[FAIL]"
        print(f"  {tag} {action.label}: {msg} ({ms} ms)")

    final_results = _apply_headless_downgrade(
        run_diagnosis(fetch_update=False), is_headless(),
    )
    final_fails = [r.name for r in final_results if not r.passed]
    if not final_fails:
        print("doctor --auto: healed, all checks pass.")
        return 0
    print(f"doctor --auto: still failing: {final_fails}; run `iai-mcp doctor --apply`.")
    return 1


def cmd_doctor(args: argparse.Namespace) -> int:
    if bool(getattr(args, "auto", False)):
        return cmd_doctor_auto()
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
        ok, msg, ms = _execute_action(action)
        tag = "[done]" if ok else "[FAIL]"
        print(f"  {tag} {action.label}: {msg} ({ms} ms)")

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
    check_bb_nightly_insight_mint,
    check_c_lock_healthy,
    check_cc_background_liveness,
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
    check_ii_embed_identity,
    check_p_anthropic_sdk_absent,
    check_r_hippo_hnsw_loadable,
    check_s_hippo_schema_version,
    check_t_hippo_compacted_freshness,
    check_u_recall_centrality_regression,
    check_v_native_embedder,
    check_w_no_permanent_failed,
    check_x_no_collapsed_timestamps,
    check_z_avx2_support,
    check_aa_capture_state_hygiene,
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
    "check_aa_capture_state_hygiene",
    "check_y_rss_24h_plateau",
    "check_z_avx2_support",
    "check_bb_nightly_insight_mint",
    "check_cc_background_liveness",
]

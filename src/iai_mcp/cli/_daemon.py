from __future__ import annotations

import argparse
import importlib.resources as _res
import json
import logging
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


STOP_TERM_TIMEOUT_S: float = 3.0
STOP_POLL_INTERVAL_S: float = 0.1


def _stop_escalation_bound() -> float:
    raw = os.environ.get("IAI_DAEMON_STOP_TIMEOUT_S")
    if raw:
        try:
            val = float(raw)
            if val >= 0:
                return val
        except ValueError:
            pass
    return STOP_TERM_TIMEOUT_S


def _stop_poll_interval() -> float:
    raw = os.environ.get("IAI_DAEMON_STOP_POLL_S")
    if raw:
        try:
            val = float(raw)
            if val > 0:
                return val
        except ValueError:
            pass
    return STOP_POLL_INTERVAL_S


def _launchd_template():
    return _res.files("iai_mcp") / "_deploy" / "launchd" / "com.iai-mcp.daemon.plist"


def _render_launchd_plist() -> str:
    from iai_mcp import cli as _cli
    text = _launchd_template().read_text()
    username = os.environ.get("USER") or Path.home().name
    text = text.replace("/usr/local/bin/python3", _cli.sys.executable)
    text = text.replace("{USERNAME}", username)
    return text


def _render_systemd_unit() -> str:
    from iai_mcp import cli as _cli
    tmpl = _res.files("iai_mcp") / "_deploy" / "systemd" / "iai-mcp-daemon.service"
    text = tmpl.read_text()
    text = text.replace("/usr/bin/python3", _cli.sys.executable)
    return text


def _find_pythonw() -> str:
    # Prefer pythonw.exe so the daemon runs with no console window; fall back to
    # the current interpreter if the windowless variant is not alongside it.
    exe = Path(sys.executable)
    pythonw = exe.parent / "pythonw.exe"
    if pythonw.exists():
        return str(pythonw)
    return sys.executable


def _render_schtasks_xml() -> str:
    from xml.sax.saxutils import escape as _xml_escape
    # Escape the interpolated values: a Windows account name or interpreter path
    # may legally contain '&' (and, defensively, '<'/'>'), which would otherwise
    # produce malformed XML that schtasks /Create rejects.
    pythonw = _xml_escape(_find_pythonw())
    username = _xml_escape(os.environ.get("USERNAME", ""))
    # No <WorkingDirectory>: the Task Scheduler engine rejects a working
    # directory set via XML when the path contains spaces (e.g. the default
    # %APPDATA% under "C:\\Users\\First Last\\..."), failing the launch with
    # 0x8007010B "The directory name is invalid" — even though the path exists
    # and CreateProcess accepts it fine outside the scheduler. The daemon never
    # depends on cwd (all state lives under ~/.iai-mcp via absolute paths), so
    # we omit it and let the task default to %windir%\\system32.
    return f"""\
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>iai-mcp sleep daemon — background memory consolidation for Claude Code</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
      <UserId>{username}</UserId>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>{username}</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Hidden>true</Hidden>
  </Settings>
  <Actions>
    <Exec>
      <Command>{pythonw}</Command>
      <Arguments>-m iai_mcp.daemon</Arguments>
    </Exec>
  </Actions>
</Task>"""


def _prompt_consent(stream_out=None) -> bool:
    from iai_mcp import cli as _cli
    if stream_out is None:
        stream_out = sys.stderr
    print(_cli.CONSENT_BANNER, file=stream_out, end="")
    stream_out.flush()
    try:
        response = input("")
    except EOFError:
        return False
    return response.strip().lower() == "y"


def _record_consent_receipt() -> None:
    from iai_mcp import cli as _cli
    state_dir = _cli.LOCK_PATH.parent
    state_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    payload = {
        "consent": True,
        "ts": ts,
        "executable": sys.executable,
        "platform": platform.system(),
        "user": os.environ.get("USER") or "",
    }
    safe_ts = ts.replace(":", "").replace("-", "").replace(".", "")
    receipt = state_dir / f".consent-{safe_ts}.json"
    try:
        receipt.write_text(json.dumps(payload, indent=2))
        os.chmod(receipt, 0o600)
    except OSError as exc:
        print(f"warning: could not write consent receipt: {exc}", file=sys.stderr)


def _remove_state_files() -> None:
    from iai_mcp import cli as _cli
    for p in (_cli.LOCK_PATH, _cli.SOCKET_PATH, _cli.STATE_PATH):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"warning: could not remove {p}: {exc}", file=sys.stderr)


def cmd_daemon_install(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    dry_run = bool(getattr(args, "dry_run", False))
    yes = bool(getattr(args, "yes", False))

    if not yes and not dry_run:
        if not _prompt_consent():
            print("Install cancelled.", file=sys.stderr)
            return 1
        _record_consent_receipt()

    if _cli._is_macos():
        content = _render_launchd_plist()
        target = _cli.LAUNCHD_TARGET
    elif _cli._is_linux():
        content = _render_systemd_unit()
        target = _cli.SYSTEMD_TARGET
    elif _cli._is_windows():
        content = _render_schtasks_xml()
        target = None
    else:
        print(f"Unsupported OS: {platform.system()}", file=sys.stderr)
        return 1

    if dry_run:
        if target is not None:
            print(f"# Would install to: {target}")
        else:
            print(f"# Would create scheduled task: {_cli.SCHTASKS_TASK_NAME}")
        print(content)
        return 0

    if _cli._is_windows():
        _cli._ensure_crypto_key_present()

        import subprocess as _sp
        import tempfile as _tmpmod

        # schtasks /Create ingests the task definition as a UTF-16 XML file.
        fd, xml_path = _tmpmod.mkstemp(suffix=".xml", prefix="iai-mcp-task-")
        try:
            with os.fdopen(fd, "w", encoding="utf-16") as f:
                f.write(content)
            result = _sp.run(
                [
                    "schtasks", "/Create",
                    "/TN", _cli.SCHTASKS_TASK_NAME,
                    "/XML", xml_path,
                    "/F",
                ],
                check=False, capture_output=True, text=True,
            )
            if result.returncode != 0:
                print(
                    f"schtasks /Create failed ({result.returncode}): "
                    f"{result.stderr.strip()}",
                    file=sys.stderr,
                )
                return 1
        finally:
            try:
                os.unlink(xml_path)
            except OSError:
                pass

        _sp.run(
            ["schtasks", "/Run", "/TN", _cli.SCHTASKS_TASK_NAME],
            check=False, capture_output=True,
        )
        print(f"Installed scheduled task: {_cli.SCHTASKS_TASK_NAME}")
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    try:
        os.chmod(target, 0o644)
    except OSError:
        pass

    _cli._ensure_crypto_key_present()

    uid = os.getuid() if hasattr(os, "getuid") else 0
    if _cli._is_macos():
        _cli.subprocess.run(
            ["launchctl", "bootout", f"gui/{uid}", str(target)],
            check=False, capture_output=True,
        )
        result = _cli.subprocess.run(
            ["launchctl", "bootstrap", f"gui/{uid}", str(target)],
            check=False, capture_output=True, text=True,
        )
        if result.returncode != 0 and result.stderr:
            print(
                f"warning: launchctl bootstrap returned {result.returncode}: "
                f"{result.stderr.strip()}",
                file=sys.stderr,
            )
        _cli.subprocess.run(
            ["launchctl", "kickstart", f"gui/{uid}/{_cli.DAEMON_LABEL}"],
            check=False, capture_output=True,
        )
    else:
        user = os.environ.get("USER") or ""
        linger_probe = _cli.subprocess.run(
            ["loginctl", "show-user", user, "--property=Linger"],
            check=False, capture_output=True, text=True,
        )
        if "Linger=yes" not in linger_probe.stdout:
            _cli.subprocess.run(
                ["loginctl", "enable-linger", user],
                check=False, capture_output=True,
            )
            linger_recheck = _cli.subprocess.run(
                ["loginctl", "show-user", user, "--property=Linger"],
                check=False, capture_output=True, text=True,
            )
            if "Linger=yes" not in linger_recheck.stdout:
                print(
                    "WARNING: loginctl enable-linger did not take effect -- "
                    "daemon may die at logout",
                    file=sys.stderr,
                )
        _cli.subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            check=False, capture_output=True,
        )
        _cli.subprocess.run(
            ["systemctl", "--user", "enable", "--now", _cli.SERVICE_NAME],
            check=False, capture_output=True,
        )

    print(f"Installed to {target}")
    return 0


def cmd_daemon_uninstall(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    yes = bool(getattr(args, "yes", False))
    if not yes:
        try:
            response = input(
                "Uninstall iai daemon? "
                "(removes plist/unit + state files) [y/N]: "
            )
        except EOFError:
            response = ""
        if response.strip().lower() != "y":
            print("Uninstall cancelled.", file=sys.stderr)
            return 1

    uid = os.getuid() if hasattr(os, "getuid") else 0
    if _cli._is_macos():
        if _cli.LAUNCHD_TARGET.exists():
            _cli.subprocess.run(
                ["launchctl", "bootout", f"gui/{uid}", str(_cli.LAUNCHD_TARGET)],
                check=False, capture_output=True,
            )
            try:
                _cli.LAUNCHD_TARGET.unlink()
            except OSError as exc:
                print(f"warning: could not remove plist: {exc}", file=sys.stderr)
    elif _cli._is_linux():
        if _cli.SYSTEMD_TARGET.exists():
            _cli.subprocess.run(
                ["systemctl", "--user", "disable", "--now", _cli.SERVICE_NAME],
                check=False, capture_output=True,
            )
            try:
                _cli.SYSTEMD_TARGET.unlink()
            except OSError as exc:
                print(f"warning: could not remove unit: {exc}", file=sys.stderr)
            _cli.subprocess.run(
                ["systemctl", "--user", "daemon-reload"],
                check=False, capture_output=True,
            )
    elif _cli._is_windows():
        import subprocess as _sp
        _sp.run(
            ["schtasks", "/End", "/TN", _cli.SCHTASKS_TASK_NAME],
            check=False, capture_output=True,
        )
        _sp.run(
            ["schtasks", "/Delete", "/TN", _cli.SCHTASKS_TASK_NAME, "/F"],
            check=False, capture_output=True,
        )

    _remove_state_files()
    print("Daemon uninstalled. State files removed.")
    return 0


def cmd_daemon_start(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    uid = os.getuid() if hasattr(os, "getuid") else 0
    if _cli._is_macos():
        target = _cli.LAUNCHD_TARGET
        _cli.subprocess.run(
            ["launchctl", "bootout", f"gui/{uid}", str(target)],
            check=False, capture_output=True,
        )
        _cli.subprocess.run(
            ["launchctl", "bootstrap", f"gui/{uid}", str(target)],
            check=False, capture_output=True,
        )
        _cli.subprocess.run(
            ["launchctl", "kickstart", f"gui/{uid}/{_cli.DAEMON_LABEL}"],
            check=False, capture_output=True,
        )
    elif _cli._is_linux():
        _cli.subprocess.run(
            ["systemctl", "--user", "start", _cli.SERVICE_NAME],
            check=False,
        )
    elif _cli._is_windows():
        import subprocess as _sp
        _sp.run(
            ["schtasks", "/Run", "/TN", _cli.SCHTASKS_TASK_NAME],
            check=False, capture_output=True,
        )
    else:
        print(f"Unsupported OS: {platform.system()}", file=sys.stderr)
        return 1
    return 0


def cmd_daemon_stop(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    import signal as _signal
    import time as _time

    try:
        from iai_mcp.daemon_state import load_state, save_state

        state = load_state()
        state["user_requested_shutdown"] = True
        save_state(state)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.debug("sentinel write failed (non-blocking): %s", exc)

    if _cli._is_macos():
        from iai_mcp.lifecycle_lock import LifecycleLock, _is_pid_alive

        uid = os.getuid()
        payload = LifecycleLock().read()
        pid = payload["pid"] if payload else None

        _cli.subprocess.run(
            ["launchctl", "bootout", f"gui/{uid}", str(_cli.LAUNCHD_TARGET)],
            check=False, capture_output=True,
        )

        if pid is None:
            return 0

        if _is_pid_alive(pid):
            try:
                os.kill(pid, _signal.SIGTERM)
            except (ProcessLookupError, PermissionError) as exc:
                logger.debug("SIGTERM to daemon pid=%d failed: %s", pid, exc)
                return 0

            deadline = _time.monotonic() + _stop_escalation_bound()
            interval = _stop_poll_interval()
            while _time.monotonic() < deadline:
                if not _is_pid_alive(pid):
                    return 0
                _time.sleep(interval)

            if _is_pid_alive(pid):
                try:
                    os.kill(pid, _signal.SIGKILL)
                except (ProcessLookupError, PermissionError) as exc:
                    logger.debug("SIGKILL to daemon pid=%d failed: %s", pid, exc)
        return 0
    elif _cli._is_linux():
        _cli.subprocess.run(
            ["systemctl", "--user", "stop", _cli.SERVICE_NAME],
            check=False,
        )
    elif os.name == "nt":
        # Windows: signals cannot stop the tree — a plain terminate orphans
        # the child that still holds the hippo lock. taskkill /T fells the
        # whole process tree; /F because there is no SIGTERM grace concept.
        from iai_mcp.lifecycle_lock import LifecycleLock, _is_pid_alive

        payload = LifecycleLock().read()
        pid = payload["pid"] if payload else None
        if pid is not None and _is_pid_alive(pid):
            _cli.subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                check=False, capture_output=True,
            )
    else:
        print(f"Unsupported OS: {platform.system()}", file=sys.stderr)
        return 1
    return 0


def _compute_p90_from_events(events: list[dict]) -> dict[str, int | None]:
    import statistics

    samples = [
        int(e["data"]["total_cached_tokens"])
        for e in events
        if isinstance(e.get("data"), dict) and "total_cached_tokens" in e["data"]
    ]
    if not samples:
        return {"p90": None, "n_samples": 0}
    if len(samples) == 1:
        return {"p90": samples[0], "n_samples": 1}
    q = statistics.quantiles(samples, n=10, method="inclusive")
    p90 = int(round(q[8]))
    return {"p90": p90, "n_samples": len(samples)}


def _render_daemon_stats(result: dict[str, int | None]) -> None:
    p90_str = str(result["p90"]) if result["p90"] is not None else "no-data"
    print(f"session_start_tokens_p90: {p90_str}")
    print(f"n_samples: {result['n_samples']}")
    if 0 < (result["n_samples"] or 0) < 100:
        print(f"note: rolling window under-filled (have {result['n_samples']}, need 100)")


def cmd_daemon_stats(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    resp = _cli._send_jsonrpc_request("events_query", {"kind": "session_started", "limit": 100})
    if isinstance(resp, dict) and "result" in resp:
        payload = resp["result"]
        if isinstance(payload, dict) and "events" in payload:
            result = _compute_p90_from_events(payload["events"])
            _render_daemon_stats(result)
            return 0

    from iai_mcp.hippo import HippoLockHeldError
    from iai_mcp.store import MemoryStore

    try:
        store_dir = Path(os.environ.get("IAI_MCP_STORE", Path.home() / ".iai-mcp"))
        store = MemoryStore(path=store_dir)
        result = _cli.compute_session_start_tokens_p90(store)
    except HippoLockHeldError:
        print("daemon holds store lock; retry when daemon is idle")
        return 0

    _render_daemon_stats(result)
    return 0


def _read_recent_step_deltas(store_root: Path, max_rows: int = 65) -> list[dict]:
    """Read recent completed-step rows carrying an rss_delta_kib from the event log."""
    log_dir = store_root / "logs"
    if not log_dir.exists():
        return []
    files = sorted(log_dir.glob("lifecycle-events-*.jsonl"), reverse=True)
    rows: list[dict] = []
    for path in files:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw.startswith("{"):
                        continue
                    try:
                        row = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if (
                        row.get("event") == "sleep_step_completed"
                        and row.get("rss_delta_kib") is not None
                    ):
                        rows.append(row)
        except OSError:
            continue
        if len(rows) >= max_rows:
            break
    return rows[-max_rows:]


def _render_step_delta_table(rows: list[dict]) -> None:
    if not rows:
        print("(no per-step resident-set deltas found yet)")
        return
    # Group by an hour-grain bucket of the row timestamp; print the most recent
    # few buckets with each step's gross (pre-relief) delta.
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        bucket = str(row.get("ts"))[:13]
        buckets.setdefault(bucket, []).append(row)
    print("recent per-step resident-set deltas (gross, pre-relief):")
    for bucket in sorted(buckets)[-5:]:
        members = buckets[bucket]
        total_mib = sum(
            (m.get("rss_delta_kib") or 0) for m in members
        ) / 1024.0
        print(f"  {bucket} — total {total_mib:+.1f} MiB over {len(members)} steps")
        for m in members:
            step = m.get("step", "?")
            delta_mib = (m.get("rss_delta_kib") or 0) / 1024.0
            dur = m.get("duration_sec", "?")
            print(f"    {step:<24} {delta_mib:+8.1f} MiB   {dur}s")


def cmd_daemon_rss_stats(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli

    print("resident-set picture")
    resp = _cli._send_jsonrpc_request("rss_stats", {})
    if isinstance(resp, dict) and "result" in resp:
        snap = resp["result"]
        rss_kib = snap.get("rss_kib", "?")
        regions = snap.get("vmmap_region_count", "?")
        va_kib = snap.get("vm_allocate_kib", "?")
        pa_bytes = snap.get("pa_pool_bytes", "?")
        pa_max = snap.get("pa_pool_max", "?")
        nrt_alloc = snap.get("numba_nrt_alloc_count", -1)
        nrt_free = snap.get("numba_nrt_free_count", -1)
        nrt_outstanding = (
            nrt_alloc - nrt_free
            if isinstance(nrt_alloc, int)
            and isinstance(nrt_free, int)
            and nrt_alloc >= 0
            and nrt_free >= 0
            else "?"
        )
        print(f"  rss_kib:             {rss_kib}")
        print(f"  vmmap_region_count:  {regions}")
        print(f"  vm_allocate_kib:     {va_kib}")
        print(f"  pyarrow_pool_bytes:  {pa_bytes} (high-water {pa_max})")
        print(f"  numba_nrt_outstanding: {nrt_outstanding}")
    else:
        print("(daemon socket unreachable; live stats omitted)")

    store_root = Path(os.environ.get("IAI_MCP_STORE", Path.home() / ".iai-mcp"))
    try:
        rows = _read_recent_step_deltas(store_root)
    except OSError as exc:
        print(f"error reading event log: {exc}", file=sys.stderr)
        return 1
    _render_step_delta_table(rows)
    return 0


def cmd_daemon_status(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    import asyncio
    try:
        resp = _cli._send_socket_request({"type": "status"}, timeout=10.0)
    except asyncio.TimeoutError:
        print("daemon not responding", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 -- surface socket errors cleanly
        logger.error("daemon status failed: %s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if resp is None:
        print("daemon not running")
        return 1

    try:
        from iai_mcp import __version__ as installed_version
    except (ImportError, AttributeError):
        installed_version = "unknown"
    daemon_version = resp.get("version", "unknown")
    if (
        daemon_version != "unknown"
        and installed_version != "unknown"
        and daemon_version != installed_version
    ):
        print(
            f"WARNING: daemon version {daemon_version} != "
            f"installed {installed_version} -- run iai-mcp daemon "
            f"stop && iai-mcp daemon start to restart",
            file=sys.stderr,
        )

    for k, v in resp.items():
        print(f"{k}: {v}")
    return 0


def cmd_daemon_logs(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    follow = bool(getattr(args, "follow", False))
    lines = int(getattr(args, "lines", 50))
    if _cli._is_macos():
        path = Path.home() / ".iai-mcp" / "logs" / "launchd-stderr.log"
        argv = ["tail"]
        if follow:
            argv.append("-f")
        argv.extend(["-n", str(lines), str(path)])
        _cli.subprocess.run(argv, check=False)
    elif _cli._is_linux():
        argv = ["journalctl", "--user", "-u", _cli.SERVICE_NAME, "-n", str(lines)]
        if follow:
            argv.append("-f")
        _cli.subprocess.run(argv, check=False)
    else:
        print(f"Unsupported OS: {platform.system()}", file=sys.stderr)
        return 1
    return 0


def cmd_daemon_force_rem(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    import asyncio
    try:
        resp = _cli._send_socket_request(
            {"type": "force_rem", "ts": datetime.now(timezone.utc).isoformat()},
            timeout=15 * 60,
        )
    except asyncio.TimeoutError:
        print("force_rem timed out after 15 minutes", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.error("force_rem failed: %s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if resp is None:
        print("daemon not running")
        return 1
    print(json.dumps(resp))
    return 0


def cmd_daemon_pause(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    seconds = int(args.seconds)
    try:
        resp = _cli._send_socket_request(
            {"type": "pause", "seconds": seconds}, timeout=10.0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("pause failed: %s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if resp is None:
        print("daemon not running")
        return 1
    print(f"paused for {seconds}s")
    return 0


def cmd_daemon_resume(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    try:
        resp = _cli._send_socket_request({"type": "resume"}, timeout=10.0)
    except Exception as exc:  # noqa: BLE001
        logger.error("resume failed: %s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if resp is None:
        print("daemon not running")
        return 1
    print("resumed")
    return 0


def cmd_daemon_configure(args: argparse.Namespace) -> int:
    from iai_mcp.daemon_state import load_state, save_state

    key = args.key
    value = getattr(args, "value", None)
    state = load_state()

    if key == "set-budget":
        if value is None:
            print("set-budget requires a float value", file=sys.stderr)
            return 2
        state["daily_quota_pct_override"] = float(value)
    elif key == "set-cycle-count":
        if value is None:
            print("set-cycle-count requires an int value", file=sys.stderr)
            return 2
        state["cycle_count_override"] = int(value)
    elif key == "set-quiet-window":
        if value is None or "-" not in value:
            print(
                "set-quiet-window requires HH:MM-HH:MM format",
                file=sys.stderr,
            )
            return 2
        start, end = value.split("-", 1)
        state["quiet_window_manual_override"] = [start.strip(), end.strip()]
    elif key == "disable-claude":
        state["claude_enabled"] = False
    elif key == "enable-claude":
        state["claude_enabled"] = True
    else:
        print(f"unknown configure key: {key}", file=sys.stderr)
        return 2

    save_state(state)
    print(f"{key} -> {value if value is not None else 'toggled'}")
    return 0

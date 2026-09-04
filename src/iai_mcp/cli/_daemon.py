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

#: Emergency disable for the orphan sweep. The test suite sets it in
#: conftest so no unit test of the stop verb can ever signal a real
#: daemon on the development host.
ORPHAN_SWEEP_DISABLE_ENV = "IAI_MCP_DISABLE_ORPHAN_SWEEP"


def _matches_daemon_title(cmdline: list[str], target_title: str) -> bool:
    # The title must BE argv[0] — never an argument and never a substring:
    # an any-field or containment match fells innocent carriers of the
    # title (pgrep -f, pkill -f, grep -r) or a sibling store whose path
    # extends this one. setproctitle rewrites argv into one padded string,
    # so argv[0] equals the full title on macOS and Linux; without
    # setproctitle the daemon carries no title and the sweep is inert.
    if not cmdline:
        return False
    return (cmdline[0] or "").strip() == target_title


def _pid_serves_this_store(pid: int) -> bool:
    """Whether `pid` is a daemon process of THIS invocation's store.

    Identity, not context: the lifecycle-lock pid is only evidence of what
    the file says, and a stale record can name a recycled pid belonging to
    an unrelated process — or to another store's daemon. Signalling on that
    evidence alone is how a stop aimed at one store fells another. The
    sweep already matches by process-title equality; the primary stop path
    must hold to the same standard. Unknown identity is NOT a match: a
    daemon we cannot identify is one we must not signal.
    """
    try:
        import psutil
    except ImportError:
        return False

    from iai_mcp.lifecycle_lock import daemon_process_title
    from iai_mcp.tz import store_root

    try:
        cmdline = psutil.Process(int(pid)).cmdline()
    except (psutil.Error, ValueError, TypeError):
        return False
    return _matches_daemon_title(
        [str(x) for x in (cmdline or [])], daemon_process_title(store_root())
    )


def _live_daemon_pid_for_this_store() -> "int | None":
    """The lifecycle-lock pid when it is alive AND serves this store."""
    from iai_mcp.lifecycle_lock import LifecycleLock, _is_pid_alive

    try:
        payload = LifecycleLock().read()
    except (OSError, ValueError, RuntimeError):
        return None
    pid = payload.get("pid") if payload else None
    if pid is None:
        return None
    try:
        if not _is_pid_alive(int(pid)):
            return None
    except (TypeError, ValueError):
        return None
    return int(pid) if _pid_serves_this_store(int(pid)) else None


def _sweep_orphan_daemon_processes(
    exclude_pids: set[int],
    *,
    proc_iter=None,
    term_timeout: float | None = None,
) -> list[int]:
    """Fell every daemon process of THIS store that the primary stop missed.

    The platform stop targets only the lifecycle-lock pid; a daemon that
    lost (or never held) the lock survives every restart, keeps retrying
    the hippo lock, and floods stderr. Matching is scoped by the
    store-path-bearing process title, so stopping one store's daemon can
    never fell another store's.
    """
    _disable_raw = os.environ.get(ORPHAN_SWEEP_DISABLE_ENV, "").strip().lower()
    if _disable_raw in ("1", "true", "yes", "on"):
        return []
    try:
        import psutil
    except ImportError:
        return []

    from iai_mcp.lifecycle_lock import daemon_process_title
    from iai_mcp.tz import store_root

    target_title = daemon_process_title(store_root())

    me = os.getpid()
    victims = []
    iterator = (
        proc_iter
        if proc_iter is not None
        else psutil.process_iter(["pid", "name", "cmdline"])
    )
    for p in iterator:
        try:
            info = getattr(p, "info", None) or {}
            cmdline = info.get("cmdline")
            if cmdline is None:
                cmdline = p.cmdline()
            pid_val = int(p.pid)
        except (psutil.Error, ValueError, TypeError):
            continue
        if pid_val == me or pid_val in exclude_pids:
            continue
        if not _matches_daemon_title([str(x) for x in (cmdline or [])], target_title):
            continue
        victims.append(p)
    if not victims:
        return []

    swept: list[int] = []
    for p in victims:
        try:
            p.terminate()
            swept.append(int(p.pid))
        except psutil.Error:
            continue
    if term_timeout is None:
        term_timeout = _stop_escalation_bound()
    try:
        _gone, alive = psutil.wait_procs(victims, timeout=term_timeout)
    except psutil.Error:
        alive = [p for p in victims if _safe_is_running(p)]
    for p in alive:
        try:
            p.kill()
        except psutil.Error:
            pass
    if swept:
        print(
            f"swept {len(swept)} orphan daemon process(es): {swept}",
            file=sys.stderr,
        )
    return swept


def _safe_is_running(p) -> bool:
    try:
        return bool(p.is_running())
    except Exception:  # noqa: BLE001 -- process may vanish mid-check
        return False


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
    text = _launchd_template().read_text(encoding="utf-8")
    username = os.environ.get("USER") or Path.home().name
    text = text.replace("/usr/local/bin/python3", _cli.sys.executable)
    text = text.replace("{USERNAME}", username)
    return text


def _render_systemd_unit() -> str:
    from iai_mcp import cli as _cli
    tmpl = _res.files("iai_mcp") / "_deploy" / "systemd" / "iai-mcp-daemon.service"
    text = tmpl.read_text(encoding="utf-8")
    text = text.replace("/usr/bin/python3", _cli.sys.executable)
    return text


def _render_windows_task_xml() -> str:
    from iai_mcp import cli as _cli
    tmpl = _res.files("iai_mcp") / "_deploy" / "windows" / "iai-mcp-daemon.xml"
    text = tmpl.read_text(encoding="utf-8")
    text = text.replace("{START_CMD}", str(_cli.WINDOWS_START_CMD))
    text = text.replace("{WORK_DIR}", str(Path.home() / ".iai-mcp"))
    return text


def _render_windows_start_cmd() -> str:
    """Start wrapper the scheduled task runs: the Task Scheduler XML cannot
    set environment variables or redirect streams, so this .cmd is the
    Windows counterpart of the plist's EnvironmentVariables +
    StandardErrorPath blocks."""
    from iai_mcp import cli as _cli
    home = Path.home()
    log_dir = home / ".iai-mcp" / "logs"
    return (
        "@echo off\r\n"
        f'set "IAI_MCP_STORE={home / ".iai-mcp"}"\r\n'
        'set "IAI_MCP_LAUNCHD_MANAGED=1"\r\n'
        f'if not exist "{log_dir}" mkdir "{log_dir}"\r\n'
        f'"{_cli.sys.executable}" -m iai_mcp.daemon '
        f'>> "{log_dir / "task-stdout.log"}" '
        f'2>> "{log_dir / "task-stderr.log"}"\r\n'
    )


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
        receipt.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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
    elif os.name == "nt":
        return _install_windows_task(dry_run=dry_run)
    else:
        print(f"Unsupported OS: {platform.system()}", file=sys.stderr)
        return 1

    if dry_run:
        print(f"# Would install to: {target}")
        print(content)
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    try:
        os.chmod(target, 0o644)
    except OSError:
        pass

    _cli._ensure_crypto_key_present()

    # os.getuid() is POSIX-only; resolve it inside the branch that uses it.
    if _cli._is_macos():
        uid = os.getuid()
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


def _install_windows_task(*, dry_run: bool) -> int:
    """Register the daemon as a per-user Task Scheduler task (logon trigger,
    restart-on-failure) — the Windows counterpart of launchd/systemd install."""
    from iai_mcp import cli as _cli

    cmd_content = _render_windows_start_cmd()
    xml_content = _render_windows_task_xml()

    if dry_run:
        print(f"# Would install start wrapper to: {_cli.WINDOWS_START_CMD}")
        print(cmd_content)
        print(f"# Would register task '{_cli.WINDOWS_TASK_NAME}' from: {_cli.WINDOWS_TASK_XML}")
        print(xml_content)
        return 0

    _cli.WINDOWS_START_CMD.parent.mkdir(parents=True, exist_ok=True)
    _cli.WINDOWS_START_CMD.write_text(cmd_content, encoding="utf-8")
    # Task Scheduler expects its XML in UTF-16, matching its own export format.
    _cli.WINDOWS_TASK_XML.write_text(xml_content, encoding="utf-16")

    _cli._ensure_crypto_key_present()

    result = _cli.subprocess.run(
        [
            "schtasks", "/Create",
            "/TN", _cli.WINDOWS_TASK_NAME,
            "/XML", str(_cli.WINDOWS_TASK_XML),
            "/F",
        ],
        check=False, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(
            f"schtasks /Create failed ({result.returncode}): "
            f"{(result.stderr or result.stdout).strip()}",
            file=sys.stderr,
        )
        return 1
    _cli.subprocess.run(
        ["schtasks", "/Run", "/TN", _cli.WINDOWS_TASK_NAME],
        check=False, capture_output=True,
    )

    print(f"Installed scheduled task '{_cli.WINDOWS_TASK_NAME}' (start wrapper: {_cli.WINDOWS_START_CMD})")
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

    if os.name == "nt":
        _cli.subprocess.run(
            ["schtasks", "/End", "/TN", _cli.WINDOWS_TASK_NAME],
            check=False, capture_output=True,
        )
        _cli.subprocess.run(
            ["schtasks", "/Delete", "/TN", _cli.WINDOWS_TASK_NAME, "/F"],
            check=False, capture_output=True,
        )
        for artifact in (_cli.WINDOWS_TASK_XML, _cli.WINDOWS_START_CMD):
            try:
                artifact.unlink(missing_ok=True)
            except OSError as exc:
                print(f"warning: could not remove {artifact.name}: {exc}", file=sys.stderr)
        _remove_state_files()
        print("Daemon uninstalled. State files removed.")
        return 0

    # os.getuid() is POSIX-only; resolve it inside the branch that uses it.
    if _cli._is_macos():
        uid = os.getuid()
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

    _remove_state_files()
    print("Daemon uninstalled. State files removed.")
    return 0


def cmd_daemon_start(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    from iai_mcp.migrate import refuse_if_marker_present
    from iai_mcp.tz import store_root as _resolve_start_store_root

    _swap_marker_reason = refuse_if_marker_present(_resolve_start_store_root())
    if _swap_marker_reason is not None:
        print(_swap_marker_reason, file=sys.stderr)
        return 1

    if os.name == "nt":
        result = _cli.subprocess.run(
            ["schtasks", "/Run", "/TN", _cli.WINDOWS_TASK_NAME],
            check=False, capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(
                f"schtasks /Run failed ({result.returncode}): "
                f"{(result.stderr or result.stdout).strip()} — "
                "run `iai-mcp daemon install` first",
                file=sys.stderr,
            )
            return 1
        print("Daemon start requested via Task Scheduler.")
        return 0
    # os.getuid() is POSIX-only; resolve it inside the branch that uses it.
    if _cli._is_macos():
        uid = os.getuid()
        target = _cli.LAUNCHD_TARGET
        # bootout is the ONLY step here that SIGTERMs a live instance, and
        # callers that reach start as a heal (the unattended doctor reflex,
        # a wrapper wake) would kill a daemon that is merely busy — losing
        # the consolidation step it was running. Skip just that step;
        # bootstrap must still run, or a daemon alive while the job is
        # unloaded (stop's own bootout→SIGTERM window, a hand-run daemon)
        # would leave the job permanently uninstalled.
        if _live_daemon_pid_for_this_store() is None:
            _cli.subprocess.run(
                ["launchctl", "bootout", f"gui/{uid}", str(target)],
                check=False, capture_output=True,
            )
        _cli.subprocess.run(
            ["launchctl", "bootstrap", f"gui/{uid}", str(target)],
            check=False, capture_output=True,
        )
        kick = _cli.subprocess.run(
            ["launchctl", "kickstart", f"gui/{uid}/{_cli.DAEMON_LABEL}"],
            check=False, capture_output=True,
        )
        rc = getattr(kick, "returncode", 0)
        if rc:
            # A silent success here is how an uninstalled job hides.
            print(
                f"launchctl kickstart returned {rc} for {_cli.DAEMON_LABEL}",
                file=sys.stderr,
            )
    elif _cli._is_linux():
        _cli.subprocess.run(
            ["systemctl", "--user", "start", _cli.SERVICE_NAME],
            check=False,
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
        from iai_mcp.daemon_state import update_state

        update_state(lambda d: d.__setitem__("user_requested_shutdown", True))
    except (OSError, ValueError, RuntimeError) as exc:
        logger.debug("sentinel write failed (non-blocking): %s", exc)

    if _cli._is_macos():
        from iai_mcp.lifecycle_lock import _is_pid_alive

        uid = os.getuid()
        # Signal only a pid whose process identifies as THIS store's daemon.
        pid = _live_daemon_pid_for_this_store()

        _cli.subprocess.run(
            ["launchctl", "bootout", f"gui/{uid}", str(_cli.LAUNCHD_TARGET)],
            check=False, capture_output=True,
        )

        # Liveness and identity were both established above; re-checking
        # here would only widen the window between decision and signal.
        if pid is not None:
            try:
                os.kill(pid, _signal.SIGTERM)
            except (ProcessLookupError, PermissionError) as exc:
                logger.debug("SIGTERM to daemon pid=%d failed: %s", pid, exc)
            else:
                deadline = _time.monotonic() + _stop_escalation_bound()
                interval = _stop_poll_interval()
                while _time.monotonic() < deadline:
                    if not _is_pid_alive(pid):
                        break
                    _time.sleep(interval)

                if _is_pid_alive(pid):
                    try:
                        os.kill(pid, _signal.SIGKILL)
                    except (ProcessLookupError, PermissionError) as exc:
                        logger.debug(
                            "SIGKILL to daemon pid=%d failed: %s", pid, exc
                        )
        # No exclusions beyond self: if the primary somehow survived both
        # signals, the sweep is the mechanism that catches it.
        _sweep_orphan_daemon_processes(set())
        return 0
    elif _cli._is_linux():
        _cli.subprocess.run(
            ["systemctl", "--user", "stop", _cli.SERVICE_NAME],
            check=False,
        )
        _sweep_orphan_daemon_processes(set())
    elif os.name == "nt":
        # Windows: signals cannot stop the tree — a plain terminate orphans
        # the child that still holds the hippo lock. taskkill /T fells the
        # whole process tree; /F because there is no SIGTERM grace concept.
        pid = _live_daemon_pid_for_this_store()
        if pid is not None:
            _cli.subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                check=False, capture_output=True,
            )
        else:
            # The pid IS the whole stop mechanism here (no service manager
            # to fall back on), so a stop that identified nothing must say
            # so rather than report success over a daemon still running.
            logger.warning(
                "daemon stop: no live daemon of this store identified; "
                "nothing was signalled",
            )
        _sweep_orphan_daemon_processes(set())
    else:
        print(f"Unsupported OS: {platform.system()}", file=sys.stderr)
        return 1
    return 0


def cmd_daemon_restart(args: argparse.Namespace) -> int:
    # stop waits through the SIGTERM window; the SIGKILL escalation returns
    # without re-confirming death, so a start racing the dying process is
    # possible — the O_CREAT|O_EXCL lifecycle lock degrades that to a failed
    # start, never a double writer.
    rc = cmd_daemon_stop(args)
    if rc != 0:
        return rc
    rc = cmd_daemon_start(args)
    if rc != 0:
        return rc
    print("Daemon restart requested (stop completed, start issued).")
    return 0


_SELF_UPDATE_PIP_TIMEOUT_SEC = 600.0
_SELF_UPDATE_SOCKET_WAIT_SEC = 30.0


def cmd_self_update(args: argparse.Namespace) -> int:
    """Upgrade the wheel AND restart the daemon — the two halves that
    `pip install -U` alone silently leaves apart: the wheel changes on
    disk while the old engine keeps serving recall."""
    import subprocess
    import time as _time

    from iai_mcp.version_check import (
        PACKAGE_NAME,
        fetch_latest_version,
        installed_version,
        is_editable_install,
        is_newer,
        refresh_cache,
    )

    check_only = bool(getattr(args, "check", False))
    yes = bool(getattr(args, "yes", False))

    current = installed_version()
    if current is None:
        print(
            f"cannot determine the installed {PACKAGE_NAME} version — "
            "is this a packaged install?",
            file=sys.stderr,
        )
        return 1
    if is_editable_install():
        print(
            "this is a source (editable) checkout — self-update would replace "
            "it with the wheel. Update via git + your build script instead.",
            file=sys.stderr,
        )
        return 1

    latest = fetch_latest_version()
    if latest is None:
        print("PyPI is unreachable — try again when online.", file=sys.stderr)
        return 1
    if not is_newer(latest, current):
        refresh_cache(force=True)
        print(f"already up to date ({current}).")
        return 0

    print(f"{PACKAGE_NAME} {current} -> {latest}")
    if check_only:
        return 0
    if not yes:
        try:
            answer = input("  [y/N] upgrade the package and restart the daemon: ")
        except EOFError:
            answer = ""
        if answer.strip().lower() != "y":
            print("aborted — nothing changed.")
            return 1

    # Everything needed AFTER pip must be imported BEFORE pip: the upgrade
    # rewrites this package on disk under the running interpreter, and a
    # post-upgrade import could load new-layout modules into old code.
    import asyncio as _asyncio

    from iai_mcp import daemon_state as _preload_ds  # noqa: F401
    from iai_mcp import lifecycle_lock as _preload_ll  # noqa: F401
    from iai_mcp.doctor import _resolve_socket_path, _socket_status_probe

    pip_cmd = [
        sys.executable, "-m", "pip", "install", "--upgrade",
        f"{PACKAGE_NAME}=={latest}",
    ]
    try:
        proc = subprocess.run(
            pip_cmd, capture_output=True, text=True,
            timeout=_SELF_UPDATE_PIP_TIMEOUT_SEC, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"pip upgrade failed to run: {e}", file=sys.stderr)
        print("daemon NOT restarted — the old engine keeps serving.", file=sys.stderr)
        return 1
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        for line in tail:
            print(f"  {line}", file=sys.stderr)
        print(
            f"pip exited {proc.returncode}; daemon NOT restarted — "
            "the old engine keeps serving.",
            file=sys.stderr,
        )
        return 1
    print(f"package upgraded to {latest}; restarting the daemon ...")

    rc = cmd_daemon_restart(args)
    if rc != 0:
        print(
            "daemon restart failed — start it manually: `iai-mcp daemon start`",
            file=sys.stderr,
        )
        return 1

    # "Is live" must be proven by a status round-trip reporting the NEW
    # version — a SIGKILLed daemon leaves its socket file on disk, so a
    # bare stat() would call a corpse alive.
    socket_path = _resolve_socket_path()
    served: str | None = None
    deadline = _time.monotonic() + _SELF_UPDATE_SOCKET_WAIT_SEC
    while _time.monotonic() < deadline:
        try:
            resp = _asyncio.run(_socket_status_probe(socket_path, timeout=2.0))
        except Exception:  # noqa: BLE001 — probe failure = not up yet
            resp = None
        if isinstance(resp, dict) and resp.get("version"):
            served = str(resp["version"])
            if served == latest:
                refresh_cache(force=True)
                print(
                    f"done — {PACKAGE_NAME} {latest} is live (status "
                    "round-trip). Run `iai-mcp doctor` for a full checkup."
                )
                return 0
        _time.sleep(1.0)
    if served is not None:
        print(
            f"daemon is up but serves {served}, expected {latest} — the "
            "restart may not have picked up the new code; check "
            "`iai-mcp daemon status`.",
            file=sys.stderr,
        )
    else:
        print(
            f"upgraded to {latest}, restart issued, but the daemon did not "
            f"answer a status round-trip in {_SELF_UPDATE_SOCKET_WAIT_SEC:.0f}s "
            "— check `iai-mcp daemon status` / `iai-mcp doctor --apply`.",
            file=sys.stderr,
        )
    return 1


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

    try:
        from iai_mcp.code_stamp import stamp_divergence

        if stamp_divergence(resp.get("code_stamp")) == "digest":
            print(
                "WARNING: daemon is running sleep-path code older than the "
                "source on disk -- run iai-mcp daemon stop && iai-mcp daemon "
                "start to load it",
                file=sys.stderr,
            )
    except OSError as exc:
        logger.debug("code stamp comparison failed: %s", exc)

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
    from iai_mcp.daemon_state import update_state

    key = args.key
    value = getattr(args, "value", None)

    if key == "set-budget":
        if value is None:
            print("set-budget requires a float value", file=sys.stderr)
            return 2
        mutation = {"daily_quota_pct_override": float(value)}
    elif key == "set-cycle-count":
        if value is None:
            print("set-cycle-count requires an int value", file=sys.stderr)
            return 2
        mutation = {"cycle_count_override": int(value)}
    elif key == "set-quiet-window":
        if value == "auto":
            mutation = {"quiet_window_manual_override": None}
        else:
            from iai_mcp.quiet_window import parse_window_spec

            if value is None or parse_window_spec(value) is None:
                print(
                    "set-quiet-window requires HH:MM-HH:MM (non-empty span) "
                    "or 'auto' to return to the learned window",
                    file=sys.stderr,
                )
                return 2
            start, end = value.split("-", 1)
            mutation = {"quiet_window_manual_override": [start.strip(), end.strip()]}
    elif key == "disable-claude":
        mutation = {"claude_enabled": False}
    elif key == "enable-claude":
        mutation = {"claude_enabled": True}
    else:
        print(f"unknown configure key: {key}", file=sys.stderr)
        return 2

    update_state(lambda d: d.update(mutation))
    print(f"{key} -> {value if value is not None else 'toggled'}")
    return 0

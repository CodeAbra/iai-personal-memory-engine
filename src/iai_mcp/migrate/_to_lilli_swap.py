"""In-place swap of a verified native copy into the live store root.

Closes the gap between "a verified native copy exists" (``migrate_sqlite_to_lilli``
+ ``verify_store_equality``) and "the daemon now runs on it": replaces only the
``hippo/`` subdirectory inside the live store root, so a root-level crypto key
file never moves and the staging tree (also inside the live root by default)
makes the final rename same-filesystem by construction.

Dry-run by default; refuses loudly rather than racing a live daemon or
clobbering an existing backup. The function has no process authority -- a
blocker names the sanctioned stop/start commands as text; it never calls them.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from iai_mcp import _flock

_log = logging.getLogger(__name__)

MARKER_FILE_NAME = ".swap-in-progress"
LOCK_FILE_NAME = ".swap.lock"


def _device_id(path: Path) -> int:
    """The filesystem device id for ``path`` (or its nearest existing parent).

    A named seam so a test can force a cross-filesystem refusal without a
    second real filesystem.
    """
    probe = path
    while not probe.exists():
        probe = probe.parent
    return os.stat(probe).st_dev


def _same_filesystem(live_root: Path, staging_root: Path) -> bool:
    return _device_id(live_root) == _device_id(staging_root)


def _rename(src: Path, dst: Path) -> None:
    """The atomic rename primitive both swap renames go through.

    A named seam so a test can key a controlled failure on the destination
    argument alone -- the predicate that is unique to the second swap rename.
    """
    os.replace(str(src), str(dst))


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_marker(live_root: Path) -> Path:
    """Write the swap-in-progress marker at the live ROOT, atomically.

    Root-level placement is load-bearing: the first rename moves the whole
    hippo/ subdirectory to the dated backup, so a marker written inside it
    would ride into the backup and be absent from the live root for exactly
    the interval it exists to cover.
    """
    marker = live_root / MARKER_FILE_NAME
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
    }
    tmp = marker.parent / f"{marker.name}.tmp.{os.getpid()}"
    fd = os.open(str(tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(fd, json.dumps(payload).encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(str(tmp), str(marker))
    return marker


def _read_marker(live_root: Path) -> Optional[dict]:
    """The marker's recorded fields, or ``None`` when no marker is present.

    An unreadable or non-JSON marker still counts as present -- it returns a
    dict with ``malformed=True`` rather than ``None``, so a caller can never
    mistake "cannot parse this" for "nothing is here."
    """
    marker = live_root / MARKER_FILE_NAME
    if not marker.exists():
        return None
    try:
        raw = marker.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("marker payload is not a JSON object")
    except (OSError, ValueError):
        return {"path": str(marker), "malformed": True, "timestamp": None, "pid": None}
    result = dict(data)
    result["path"] = str(marker)
    result["malformed"] = False
    return result


def _remove_marker(live_root: Path) -> None:
    """Remove the marker this swap itself wrote -- the last step of the
    mutation window. Never called by anything but the swap that wrote it."""
    marker = live_root / MARKER_FILE_NAME
    try:
        marker.unlink()
    except OSError:
        pass


def refuse_if_marker_present(live_root: str | Path) -> Optional[str]:
    """A store root carrying the swap marker -> an operator-facing refusal
    reason; a root without one -> ``None``. Never deletes the marker -- a
    marker found here means a swap was interrupted, and clearing it is an
    operator decision, never an automatic repair.
    """
    root = Path(live_root)
    info = _read_marker(root)
    if info is None:
        return None

    marker_path = info["path"]
    if info.get("malformed"):
        return (
            f"a swap-in-progress marker exists at {marker_path} but its "
            "contents could not be read; a swap was interrupted before it "
            "finished. Confirm the store at this root is whole, then remove "
            "the marker file yourself -- it is never removed automatically."
        )

    timestamp = info.get("timestamp", "unknown")
    pid = info.get("pid")
    alive = False
    if isinstance(pid, int):
        try:
            from iai_mcp.lifecycle_lock import _is_pid_alive

            alive = _is_pid_alive(pid)
        except (TypeError, ValueError, OSError):
            alive = False
    liveness = "still alive" if alive else "not alive"
    return (
        f"a swap-in-progress marker exists at {marker_path} "
        f"(written {timestamp} by pid={pid}, {liveness}); a prior swap was "
        "interrupted before it finished. Confirm the store at this root is "
        "whole, then remove the marker file yourself -- it is never removed "
        "automatically; clearing it is an operator decision."
    )


def _default_live_pid_probe() -> "int | None":
    from iai_mcp.cli._daemon import _live_daemon_pid_for_this_store

    return _live_daemon_pid_for_this_store()


def _prepare_swap_plan(
    src_db: Path,
    live_root: Path,
    dst_root: str | Path | None,
    probe: Callable[[], "int | None"],
) -> dict:
    """Sniff the on-disk format and, for a legacy store, resolve the staging
    and backup paths and compute every blocker. Read-only, no filesystem
    write.
    """
    from iai_mcp.hippo._db import _resolve_effective_driver

    plan: dict = {
        "source_format": None,
        "staging_root": None,
        "backup_dir": None,
        "daemon_pid": None,
        "blockers": [],
    }

    detected = _resolve_effective_driver(str(src_db))
    plan["source_format"] = detected
    if detected == "lilli":
        # Already native: success, not a refusal -- no blockers, no work.
        return plan

    if dst_root is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dst_root = live_root / f".migrate-staging-{ts}"
    dst_root = Path(dst_root)
    plan["staging_root"] = dst_root

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    backup_dir = live_root / f"hippo.sqlite-backup-{today}"
    plan["backup_dir"] = backup_dir

    pid = probe()
    plan["daemon_pid"] = pid

    blockers: list[str] = []
    if pid is not None:
        blockers.append(
            f"a live daemon (pid={pid}) serves this store; stop it first "
            "with `iai-mcp daemon stop`, run the swap, then bring it back "
            "with `iai-mcp daemon start`."
        )

    marker_reason = refuse_if_marker_present(live_root)
    if marker_reason is not None:
        blockers.append(marker_reason)

    if backup_dir.exists():
        blockers.append(
            f"a backup directory for today already exists at {backup_dir}; "
            "remove or rename it before swapping again today."
        )

    if not _same_filesystem(live_root, dst_root):
        blockers.append(
            f"the staging root {dst_root} would be on a different "
            f"filesystem than the live root {live_root}; the final rename "
            "requires the same filesystem."
        )

    plan["blockers"] = blockers
    return plan


def swap_migrated_store(
    src_db: str | Path,
    *,
    apply: bool = False,
    dst_root: str | Path | None = None,
    batch: int = 500,
    live_pid_probe: Callable[[], "int | None"] | None = None,
) -> dict:
    """Replace the live ``hippo/`` subdirectory with a verified native copy.

    ``src_db`` is the live store's ``hippo/brain.sqlite3`` (or a path to it
    through a symlinked store root, resolved to its real target before any
    check runs). Dry-run (the default) performs no filesystem write of any
    kind and reports what an apply would do -- no lock needed, since it
    writes nothing. Apply refuses loudly -- and changes nothing -- while any
    blocker applies; otherwise it copies into a staging root, verifies, and
    replaces ``hippo/`` in place, keeping the previous store under a dated
    backup. Apply serializes the entire mutating body behind an exclusive,
    non-blocking lock on a root-level lock file: a second concurrent apply
    against the same root fails immediately with a blocker instead of racing
    into the copy/verify/rename sequence.
    """
    src_db = Path(src_db).resolve()
    live_hippo_dir = src_db.parent
    if live_hippo_dir.name != "hippo":
        raise ValueError(
            f"expected the parent directory of {src_db} to be named 'hippo' "
            f"(a store's hippo subdirectory), got {live_hippo_dir.name!r}"
        )
    live_root = live_hippo_dir.parent

    summary: dict = {
        "mode": "apply" if apply else "dry-run",
        "source_format": None,
        "live_root": str(live_root),
        "live_hippo_dir": str(live_hippo_dir),
        "staging_root": None,
        "backup_dir": None,
        "daemon_pid": None,
        "blockers": [],
        "verify_ok": None,
        "rows_copied": None,
        "swapped": False,
    }

    probe = live_pid_probe if live_pid_probe is not None else _default_live_pid_probe

    def _apply_plan(plan: dict) -> None:
        summary["source_format"] = plan["source_format"]
        if plan["staging_root"] is not None:
            summary["staging_root"] = str(plan["staging_root"])
        if plan["backup_dir"] is not None:
            summary["backup_dir"] = str(plan["backup_dir"])
        summary["daemon_pid"] = plan["daemon_pid"]
        summary["blockers"] = plan["blockers"]

    if not apply:
        _apply_plan(_prepare_swap_plan(src_db, live_root, dst_root, probe))
        return summary

    lock_path = live_root / LOCK_FILE_NAME
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    except OSError as exc:
        summary["blockers"] = [
            f"could not open the swap lock file at {lock_path}: {exc}"
        ]
        return summary
    try:
        try:
            _flock.flock(fd, _flock.LOCK_EX | _flock.LOCK_NB)
        except OSError:
            summary["blockers"] = [
                "another swap_migrated_store apply is already running "
                f"against {live_root}; wait for it to finish, then retry."
            ]
            return summary

        plan = _prepare_swap_plan(src_db, live_root, dst_root, probe)
        _apply_plan(plan)

        if plan["source_format"] == "lilli":
            # Already native (possibly swapped by the invocation that just
            # released this lock): success, not a refusal -- no work.
            return summary

        if plan["blockers"]:
            return summary

        staging_root: Path = plan["staging_root"]
        backup_dir: Path = plan["backup_dir"]

        from iai_mcp.migrate._to_lilli import migrate_sqlite_to_lilli
        from iai_mcp.migrate._to_lilli_verify import verify_store_equality

        report = migrate_sqlite_to_lilli(src_db, staging_root, batch=batch)
        summary["rows_copied"] = report.rows_copied

        from iai_mcp.crypto import CryptoKey

        key = CryptoKey(store_root=live_root).get_or_create()
        verify_report = verify_store_equality(
            str(src_db), str(staging_root), key, src_root=live_root
        )
        summary["verify_ok"] = verify_report.ok
        if not verify_report.ok:
            # Abort before any rename and before the marker; the staging
            # tree is left in place for inspection.
            summary["blockers"] = [
                f"{name}: {dim.reason}"
                for name, dim in verify_report.dimensions.items()
                if not dim.ok
            ]
            return summary

        # Re-check liveness immediately before the first rename to narrow the
        # window between the check and the mutation. The marker (written
        # next) is what actually closes it -- a respawn arriving after this
        # point reads the marker and refuses instead of opening a store in
        # motion.
        pid2 = probe()
        if pid2 is not None:
            summary["daemon_pid"] = pid2
            summary["blockers"] = [
                f"a live daemon (pid={pid2}) started during verification; "
                "stop it first with `iai-mcp daemon stop`, run the swap, "
                "then bring it back with `iai-mcp daemon start`."
            ]
            return summary

        _write_marker(live_root)

        # Rename ordering is load-bearing: backup FIRST, then the staged
        # tree into the vacated path -- replacing a non-empty directory
        # cannot work the other way round.
        _rename(live_hippo_dir, backup_dir)
        _rename(staging_root / "hippo", live_hippo_dir)
        _fsync_dir(live_root)

        # Marker removal is the LAST step of the mutation window -- removing
        # it any earlier reopens the race it exists to close.
        _remove_marker(live_root)

        # The runtime graph cache is root-relative (a sibling of hippo/),
        # so the hippo/-only rename above never carries a freshly built
        # one along -- relocate it to where the swapped-in store looks.
        try:
            staging_cache = staging_root / "runtime_graph_cache.json"
            live_cache = live_root / "runtime_graph_cache.json"
            if staging_cache.exists():
                os.replace(str(staging_cache), str(live_cache))
        except OSError as exc:
            _log.warning(
                "post-swap runtime graph cache relocation failed; first "
                "daemon boot will perform a cold rebuild: %s",
                exc,
            )

        try:
            if staging_root.exists() and not any(staging_root.iterdir()):
                staging_root.rmdir()
        except OSError:
            pass

        summary["swapped"] = True

        try:
            from iai_mcp.events import write_event
            from iai_mcp.store import MemoryStore

            audit_store = MemoryStore(live_root)
            try:
                write_event(
                    audit_store,
                    kind="migrate_swap_run",
                    data={
                        "mode": summary["mode"],
                        "backup_dir": summary["backup_dir"],
                        "rows_copied": summary["rows_copied"],
                    },
                    severity="info",
                    session_id="system",
                )
            finally:
                audit_store.db.close()
        except Exception as exc:  # noqa: BLE001 -- an audit failure never fails a completed swap
            _log.warning("post-swap audit event failed: %s", exc, exc_info=True)

        return summary
    finally:
        try:
            _flock.flock(fd, _flock.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass

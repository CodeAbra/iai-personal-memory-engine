"""Salvage `.processing-<owner_pid>.jsonl` files whose owner PID is dead OR is alive but
does not match the currently-running daemon.

When the daemon wedges with a `.processing-<pid>` marker still held by an owner PID
(dead from a previous instance, OR alive but belonging to a pre-restart daemon), the
drain pipeline at ``capture.py`` skips the file forever — the drain's predicate trusts
``_pid_is_alive(owner_pid)``. This migration walks the deferred-captures dir and renames
each such file back to its bare ``<basename>.jsonl`` form so the next drain tick claims it
as a normal candidate. Idempotent — after the rename, the marker is gone; a second run
finds nothing to do. Collision-safe — when the bare ``<basename>.jsonl`` already exists,
the unlocked file is renamed to ``<basename>.recovered-<utc_ts>-<unlock_pid>.jsonl`` (also
``.jsonl``-suffixed and drain-eligible) so the pre-existing target is preserved verbatim.

All renames are atomic-create-no-clobber: ``os.link`` reserves the target name exclusively
before the source link is dropped, so a concurrent second invocation or a racing drain can
never silently overwrite an already-present file.

The currently-running daemon's owner-PID files are NEVER touched. Documented operator
workflow: stop the wedged daemon → start a fresh daemon (the ``daemon_pid`` in
``.daemon-state.json`` becomes the new pid; the wedge's owner pid is now "foreign") → run
this migration → next drain tick reclaims the files.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from iai_mcp.capture import _PROCESSING_MARKER_RE, _pid_is_alive

log = logging.getLogger(__name__)


class _Sentinel:
    """Distinguishes 'auto-detect live_daemon_pid' from explicit ``None``."""


_NOT_GIVEN = _Sentinel()


_LIVE_PID_UNKNOWN = object()
"""Sentinel returned by ``_read_live_daemon_pid`` when the state file is absent or
corrupt (distinguishable from ``None`` which is reserved for "no live daemon known"
in explicit call sites)."""


def _read_live_daemon_pid() -> int | None:
    """Return the live daemon's PID from ``.daemon-state.json``.

    Returns
    -------
    int
        The daemon's PID when the state file exists, ``daemon_pid`` is a valid integer,
        and that process is currently alive.
    None
        The state file is absent *and* no live daemon is detectable from it, OR the
        recorded PID is dead — safe to treat as "no live daemon".

    Never raises.  Callers that need to distinguish "absent/corrupt state" from
    "known-dead PID" should call ``_read_live_daemon_pid_with_source`` instead.
    """
    try:
        from iai_mcp.daemon_state import load_state
        state = load_state() or {}
    except Exception:  # noqa: BLE001 -- load_state already swallows; defence-in-depth
        return None
    pid = state.get("daemon_pid")
    if not isinstance(pid, int) or pid < 1 or pid > 2**31 - 1:
        return None
    if not _pid_is_alive(pid):
        return None
    return pid


def _atomic_claim_target(
    src: Path,
    bare_target: Path,
    *,
    deferred_dir: Path,
    stripped: str,
    unlock_pid: int,
) -> tuple[Path | None, bool]:
    """Move ``src`` onto an exclusively-reserved target, without clobbering.

    Uses ``os.link`` (atomic create-or-EEXIST on POSIX) to reserve the target name
    before dropping the source link via ``os.unlink``.  A concurrent second invocation
    or a racing drain can never silently overwrite a pre-existing file — on EEXIST the
    reservation is retried with a unique ``.recovered-<ts>-<unlock_pid>-<n>.jsonl``
    name.

    Parameters
    ----------
    src
        The ``.processing-<pid>.jsonl`` file to move.
    bare_target
        The preferred destination (stripped bare ``.jsonl`` name).
    deferred_dir
        Directory that contains both ``src`` and the target.
    stripped
        ``src.name`` with the ``.processing-<pid>`` marker removed (= ``bare_target.name``).
    unlock_pid
        PID of the salvaging process (this CLI invocation), embedded in ``.recovered-``
        names for uniqueness across concurrent invocations.

    Returns
    -------
    tuple[Path | None, bool]
        ``(final_target, collision_safe)`` — ``final_target`` is the path the file now
        lives at, or ``None`` on unrecoverable error; ``collision_safe`` is ``True`` when a
        ``.recovered-`` fallback name was used.
    """
    stem = stripped.removesuffix(".jsonl")
    collision_safe = False

    for attempt in range(64):
        if attempt == 0 and not bare_target.exists():
            cand = bare_target
        else:
            # Bare target exists (or a prior link attempt raced) — use a unique fallback.
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            cand = deferred_dir / f"{stem}.recovered-{ts}-{unlock_pid}-{attempt}.jsonl"
            collision_safe = True

        try:
            os.link(src, cand)  # atomic create-or-EEXIST; never clobbers a pre-existing file
        except FileExistsError:
            continue  # another racer claimed this name — try a fresh one
        except FileNotFoundError:
            # src was raced away by the drain after we entered this function.
            return (None, collision_safe)
        except OSError as exc:
            log.warning(
                "dead-pid-unlock: link %s -> %s failed: %s",
                src.name,
                cand.name,
                exc,
            )
            return (None, collision_safe)

        # We exclusively own ``cand`` (the hardlink succeeded).  Drop the source link.
        try:
            os.unlink(src)
        except FileNotFoundError:
            pass  # src raced away after we hardlinked — cand still holds the data; fine
        return (cand, collision_safe)

    log.warning(
        "dead-pid-unlock: could not reserve any target name for %s after 64 attempts",
        src.name,
    )
    return (None, collision_safe)


def _unlock_one(
    src: Path,
    deferred_dir: Path,
    *,
    unlock_pid: int,
    dry_run: bool,
) -> tuple[Path | None, bool]:
    """Rename one locked file back to its bare ``.jsonl`` form, atomically.

    Returns ``(target_path, collision_safe)``. ``target_path`` is ``None`` on
    rename failure (raced or OSError) — caller treats as no-op for counter accounting.
    """
    stripped = _PROCESSING_MARKER_RE.sub(".jsonl", src.name)
    if stripped == src.name:
        # Defensive: shouldn't reach here — caller already matched the marker.
        return (None, False)
    bare_target = deferred_dir / stripped

    if dry_run:
        # Dry-run: report what *would* happen without touching the filesystem.
        collision_safe = bare_target.exists()
        return (bare_target, collision_safe)

    return _atomic_claim_target(
        src,
        bare_target,
        deferred_dir=deferred_dir,
        stripped=stripped,
        unlock_pid=unlock_pid,
    )


def migrate_unlock_dead_pid_processing_files(
    *,
    deferred_dir: Path | None = None,
    live_daemon_pid: int | None | _Sentinel = _NOT_GIVEN,
    dry_run: bool = False,
) -> dict:
    """Rename ``.processing-<owner_pid>.jsonl`` -> ``.jsonl`` for owner PIDs that are
    DEAD, or ALIVE-but-foreign (``owner_pid != live_daemon_pid``).

    Parameters
    ----------
    deferred_dir
        Override target dir (test injection). Default ``~/.iai-mcp/.deferred-captures``.
    live_daemon_pid
        Override the live-daemon PID (test injection). Sentinel ``_NOT_GIVEN`` means
        "load from ``.daemon-state.json``"; an explicit ``None`` means the live daemon PID
        is unknown — only provably-dead owner PIDs are unlocked (fail-closed).
    dry_run
        If True, report counts only; no renames.

    Returns
    -------
    dict
        ``files_scanned``, ``files_unlocked``, ``skipped_live_current_daemon``,
        ``skipped_live_unknown``, ``skipped_unparseable``, ``collision_safe_renames``,
        ``live_daemon_pid``, ``dry_run``.
    """
    if deferred_dir is None:
        deferred_dir = Path.home() / ".iai-mcp" / ".deferred-captures"
    deferred_dir = Path(deferred_dir)

    if isinstance(live_daemon_pid, _Sentinel):
        live_daemon_pid = _read_live_daemon_pid()

    result: dict = {
        "files_scanned": 0,
        "files_unlocked": 0,
        "skipped_live_current_daemon": 0,
        "skipped_live_unknown": 0,
        "skipped_unparseable": 0,
        "collision_safe_renames": 0,
        "live_daemon_pid": live_daemon_pid,
        "dry_run": dry_run,
    }
    if not deferred_dir.exists():
        return result

    unlock_pid = os.getpid()

    for entry in sorted(deferred_dir.iterdir()):
        if not entry.is_file():
            continue
        m = _PROCESSING_MARKER_RE.search(entry.name)
        if not m:
            continue
        result["files_scanned"] += 1
        try:
            owner_pid = int(m.group(1))
        except (TypeError, ValueError):
            result["skipped_unparseable"] += 1
            continue

        owner_alive = _pid_is_alive(owner_pid)

        if live_daemon_pid is not None:
            if owner_pid == live_daemon_pid:
                # ALIVE owner == current daemon: never touch (the drain owns the claim).
                result["skipped_live_current_daemon"] += 1
                continue
            # else: owner is dead OR alive-but-foreign with a known live daemon → unlock.
        else:
            # Live daemon PID is indeterminate (state absent/corrupt/dead PID recorded).
            # Fail-closed: only unlock owners that are provably dead.  An alive owner in
            # this state could be the wedged daemon itself holding a valid claim.
            if owner_alive:
                result["skipped_live_unknown"] += 1
                continue

        # Owner is provably dead, or alive-but-foreign with a known live daemon: unlock.
        target, collision_safe = _unlock_one(
            entry,
            deferred_dir,
            unlock_pid=unlock_pid,
            dry_run=dry_run,
        )
        if target is not None:
            result["files_unlocked"] += 1
            if collision_safe:
                result["collision_safe_renames"] += 1
            if not dry_run:
                log.info(
                    "dead-pid-unlock: %s -> %s",
                    entry.name,
                    target.name,
                )

    return result

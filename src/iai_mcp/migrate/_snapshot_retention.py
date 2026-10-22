"""Retention and orphan sweep for the pre-write snapshots every migration
command leaves behind, plus aux directories from a superseded storage layout.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

# Reviewed allowlist of obsolete-layout directory names. Extending it
# requires review, never a runtime "not in my known-good list" inference.
_ORPHAN_PATTERNS = ("lancedb", "lancedb-orphan-*", "lance")


@dataclass(frozen=True)
class SnapshotInfo:
    path: Path
    mtime: float
    size_bytes: int


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        if not entry.is_file():
            continue
        try:
            total += entry.stat().st_size
        except OSError:
            continue
    return total


def _live_hippo_dir(store_root: Path) -> Path:
    return Path(store_root) / "hippo"


def _is_live_store(candidate: Path, live_dir: Path) -> bool:
    """Identity check, never a name comparison -- a symlink or a case-fold
    alias must still resolve to the same inode as the live store."""
    try:
        return candidate.samefile(live_dir)
    except (FileNotFoundError, OSError):
        return candidate.resolve() == live_dir.resolve()


def _probe_store_lock(store_root: Path) -> str | None:
    """None when the store is free (absent, or not currently locked); the
    holder identifier when something else holds it exclusively. Never opens
    a store that doesn't exist yet -- there is nothing to prune-guard, and
    opening a MemoryStore would create the hippo directory as a side effect
    even in read-only mode. SHARED + read-only coexists with the daemon's
    steady-state SHARED lock -- reaching a lock error here means an
    EXCLUSIVE holder (a running migration, another prune) is active right
    now, mirroring the probe iai-mcp doctor already uses."""
    db_path = Path(store_root) / "hippo" / "brain.sqlite3"
    if not db_path.exists():
        return None

    from iai_mcp.hippo import AccessMode, ConsolidationPendingError, HippoLockHeldError
    from iai_mcp.store import MemoryStore

    store = None
    try:
        store = MemoryStore(
            path=store_root, access_mode=AccessMode.SHARED, read_only=True
        )
    except (HippoLockHeldError, ConsolidationPendingError) as exc:
        return str(getattr(exc, "holding_pid", "locked"))
    except Exception as exc:  # noqa: BLE001 -- probe is advisory, must never crash the run
        return f"probe failed: {type(exc).__name__}"
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:  # noqa: BLE001 -- probe cleanup must never crash the run
                pass
    return None


def list_snapshots(store_root: Path) -> list[SnapshotInfo]:
    """Newest first, ordered by directory mtime -- never by parsing the
    trailing timestamp out of the name, since a manually-named snapshot can
    lack the full timestamp shape. Raises OSError on a genuine enumeration
    failure; the caller decides fail-open vs fail-closed."""
    out: list[SnapshotInfo] = []
    for p in Path(store_root).glob("hippo-pre-*"):
        if not p.is_dir():
            continue
        out.append(SnapshotInfo(path=p, mtime=p.stat().st_mtime, size_bytes=_dir_size_bytes(p)))
    return sorted(out, key=lambda s: s.mtime, reverse=True)


def list_orphans(store_root: Path) -> list[Path]:
    """Directories matching the reviewed obsolete-layout allowlist only --
    never an unrecognized-therefore-delete inference."""
    store_root = Path(store_root)
    out: list[Path] = []
    for pattern in _ORPHAN_PATTERNS:
        out.extend(p for p in store_root.glob(pattern) if p.is_dir())
    return out


def prune_snapshots(
    store_root: Path,
    *,
    keep: int = 3,
    apply: bool = False,
    prune_orphans: bool = True,
) -> dict:
    """Plan (dry-run) or execute (apply) removal of snapshots beyond the
    newest ``keep`` plus allowlisted orphan directories. ``keep`` is
    clamped to at least 1 so a rollback point always survives; the live
    store is excluded by inode identity, never by name. Refuses a
    snapshot-shaped ``store_root`` outright, before any enumeration, and
    skips a store whose lock is held by another process."""
    store_root = Path(store_root)
    if store_root.name.startswith("hippo-pre-"):
        raise ValueError(f"refusing snapshot-shaped store root: {store_root}")
    if store_root.name == "hippo":
        raise ValueError(f"refusing hippo-shaped store root: {store_root}")

    mode = "apply" if apply else "dry-run"
    prune_key = "pruned" if apply else "planned_prune"
    orphan_key = "orphans_removed" if apply else "planned_orphan_removal"

    holder = _probe_store_lock(store_root)
    if holder is not None:
        return {
            "mode": mode,
            "store_root": str(store_root),
            "locked": True,
            "holder": holder,
            "snapshots_found": 0,
            "kept": [],
            prune_key: [],
            "orphans_found": [],
            orphan_key: [],
            "sizes": {},
            "reclaimed_bytes": 0,
            "errors": [],
        }

    live_dir = _live_hippo_dir(store_root)
    keep = max(keep, 1)

    snapshots = [
        s for s in list_snapshots(store_root) if not _is_live_store(s.path, live_dir)
    ]
    to_keep = snapshots[:keep]
    to_prune = snapshots[keep:]
    orphans = list_orphans(store_root) if prune_orphans else []

    sizes = {s.path.name: s.size_bytes for s in snapshots}
    sizes.update({o.name: _dir_size_bytes(o) for o in orphans})

    pruned: list[str] = []
    orphans_acted: list[str] = []
    errors: list[str] = []
    reclaimed_bytes = 0

    if apply:
        for s in to_prune:
            try:
                shutil.rmtree(s.path)
                pruned.append(s.path.name)
                reclaimed_bytes += sizes.get(s.path.name, 0)
            except OSError as exc:
                errors.append(f"{s.path.name}: {exc}")
        for o in orphans:
            try:
                shutil.rmtree(o)
                orphans_acted.append(o.name)
                reclaimed_bytes += sizes.get(o.name, 0)
            except OSError as exc:
                errors.append(f"{o.name}: {exc}")
    else:
        pruned = [s.path.name for s in to_prune]
        orphans_acted = [o.name for o in orphans]

    return {
        "mode": mode,
        "store_root": str(store_root),
        "locked": False,
        "snapshots_found": len(snapshots),
        "kept": [s.path.name for s in to_keep],
        prune_key: pruned,
        "orphans_found": [o.name for o in orphans],
        orphan_key: orphans_acted,
        "sizes": sizes,
        "reclaimed_bytes": reclaimed_bytes,
        "errors": errors,
    }

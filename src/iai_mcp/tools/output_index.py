"""Opt-in indexing of a completed run's captured output, plus a bounded
retention cap on the command-output store as a whole.

Imports only stdlib + `iai_mcp.tools` (+ `iai_mcp.tools.secret_gate` at
call time); no brain import. `index_command_output` is a CONVENIENCE
constructor, NOT the security boundary: the load-bearing secret gate
lives in `iai_mcp.tools.index_paths`/`index_all`, so neither a direct
`index_paths` call nor the retention rebuild below can bypass it --
`_rebuild_store` re-indexes retained runs through that SAME call.
`index_command_output` additionally refuses (raises) rather than proceed
if its own `store_root` is not command-output-shaped, so this convenience
constructor cannot silently degrade to an ungated store.

The manifest read-modify-write and the rebuild-then-repopulate rotation
below are each one process's local view of shared on-disk state -- both
run under `_retention_lock` (an exclusive flock on a sibling lockfile) so
two concurrent `index_command_output` calls against the same store_root
serialize instead of racing a lost manifest update or a rmtree-vs-write
collision.
"""

from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

from iai_mcp import _flock as fcntl

RETENTION_MANIFEST_FILENAME = "retention_manifest.json"
RETENTION_LOCK_FILENAME = ".retention.lock"

# Bounded default so the store cannot grow forever under sustained opt-in
# use; overridable per call (e.g. by a test) via `retention_max_runs`.
COMMAND_OUTPUT_RETENTION_MAX_RUNS = 200


def _manifest_path(store_root: Path) -> Path:
    return store_root / RETENTION_MANIFEST_FILENAME


def _retention_lock_path(store_root: Path) -> Path:
    """A SIBLING of `store_root`, never a child of it -- `_rebuild_store`
    rmtree's `store_root` itself while this lock is held, and a lock file
    living inside the directory it protects would be deleted out from
    under its own held file descriptor, silently breaking mutual exclusion
    for the next contender (who would open a fresh inode at the same path
    and never contend with the still-open, now-unlinked one)."""
    return store_root.parent / f".{store_root.name}{RETENTION_LOCK_FILENAME}"


@contextmanager
def _retention_lock(store_root: Path):
    """Exclusive, blocking flock serializing the read-modify-write-and-
    maybe-rebuild critical section below across processes sharing the
    same `store_root` -- see `lock_protocol.py` for the same `_flock`
    idiom used elsewhere in this codebase."""
    lock_path = _retention_lock_path(store_root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def load_retention_manifest(store_root: Path) -> "list[dict]":
    """Reads the small indexed-run manifest. A missing/corrupt file reads
    as an empty manifest rather than raising."""
    try:
        data = json.loads(_manifest_path(store_root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def _write_manifest(store_root: Path, entries: "list[dict]") -> None:
    _manifest_path(store_root).write_text(json.dumps(entries), encoding="utf-8")


def _run_started_at(run_dir: Path) -> "int | None":
    """Reads the run's own `started_at` from its real `meta.json` --
    retention orders on the same field the executor itself recorded,
    never a wall-clock guess taken at index time."""
    try:
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    started_at = meta.get("started_at")
    return started_at if isinstance(started_at, int) else None


def _rebuild_store(store_root: Path, retained_entries: "list[dict]") -> None:
    """Rebuilds `store_root` from scratch, re-indexing ONLY the artifact
    files named by `retained_entries` against the SAME store root --
    the same call `index_command_output` itself uses, so the structural
    secret gate at that boundary applies again on rebuild, not a second,
    divergent check. The caller writes the manifest AFTER this returns,
    since this clears the directory the manifest also lives in.
    """
    from iai_mcp.tools import index_paths

    retained_paths: "list[str]" = []
    for entry in retained_entries:
        for key in ("stdout_path", "stderr_path"):
            path = entry.get(key)
            if path and Path(path).exists():
                retained_paths.append(path)

    shutil.rmtree(store_root, ignore_errors=True)
    store_root.mkdir(parents=True, exist_ok=True)
    if retained_paths:
        index_paths(retained_paths, store_root=store_root)


def index_command_output(
    run_dir: Path,
    *,
    store_root: "Path | None" = None,
    retention_max_runs: int = COMMAND_OUTPUT_RETENTION_MAX_RUNS,
) -> dict:
    """Indexes `run_dir`'s stdout.log/stderr.log into the command-output
    store, then enforces the retention cap on the store as a whole.

    Returns `{"indexed": <int>, "refused": [<paths>]}`. `refused` is a
    reporting convenience computed via the SAME gate the store boundary
    enforces -- this call's own indexing still goes through that boundary,
    so a refused path is never written regardless of this report.

    Retention: this run's entry (run_id/started_at/stdout_path/stderr_path)
    is appended to a small on-disk manifest after indexing, UNLESS every
    candidate was refused (`indexed == 0`) -- a fully-refused run never
    wrote anything into the searchable store, so it must not consume a
    retention slot that a genuinely indexed run would otherwise keep.
    Once the manifest holds more than `retention_max_runs` entries, the
    store is REBUILT from only the newest-N runs by `started_at` -- there
    is no un-index API, so rebuild-from-retained-set is the only path
    that structurally cannot leave a posting pointing at a rotated-out
    artifact file.

    The whole read-modify-write-and-maybe-rebuild body runs under an
    exclusive lock on `store_root` so concurrent calls against the same
    store serialize instead of racing.
    """
    from iai_mcp.tools import (
        COMMAND_OUTPUT_STORE_DIRNAME,
        default_command_output_store_root,
        index_paths,
        is_command_output_store_root,
    )
    from iai_mcp.tools import secret_gate

    root = store_root or default_command_output_store_root()
    if not is_command_output_store_root(root):
        raise ValueError(
            f"index_command_output requires a command-output-shaped store_root "
            f"(directory named {COMMAND_OUTPUT_STORE_DIRNAME!r}), got {root!r}"
        )

    candidates = []
    for name in ("stdout.log", "stderr.log"):
        candidate = run_dir / name
        if candidate.exists():
            candidates.append(str(candidate))

    with _retention_lock(root):
        passing, refused = secret_gate.gate_paths_for_command_output(candidates)
        indexed = index_paths(passing, store_root=root)

        if indexed == 0:
            return {"indexed": indexed, "refused": refused}

        entries = load_retention_manifest(root)
        entries.append(
            {
                "run_id": run_dir.name,
                "started_at": _run_started_at(run_dir),
                "stdout_path": str(run_dir / "stdout.log"),
                "stderr_path": str(run_dir / "stderr.log"),
            }
        )

        if len(entries) > retention_max_runs:
            entries.sort(key=lambda e: e["started_at"] if e["started_at"] is not None else -1, reverse=True)
            entries = entries[:retention_max_runs]
            _rebuild_store(root, entries)

        _write_manifest(root, entries)

    return {"indexed": indexed, "refused": refused}

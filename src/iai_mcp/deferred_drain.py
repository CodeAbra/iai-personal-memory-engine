"""Offline byte-bounded recovery of the deferred-capture backlog.

Enumerates every drain-eligible capture file — plain ``.jsonl``,
``.crash-*.jsonl``, and the ``.quarantine/*.jsonl`` sink that no other path
reads — groups them into byte-bounded batches, and spawns ONE spawn-context
subprocess per batch. Each child constructs its own store, drains its files via
``capture_turn`` (forwarding ``source_uuid``/``ts`` unchanged for idempotency),
reports its counts and PID, and exits.

Per-batch subprocesses are the whole point: in-process draining accumulates the
peak resident set across the entire backlog, while a child that exits after its
batch lets the OS reclaim that peak before the next batch begins.

The parent's per-file event count (``_count_events`` — line 0 is the header, so
the count is ``len(non_blank_lines) - 1``) is the single reconciliation
authority. Children independently re-count their drained events and the two
sums must agree.
"""

from __future__ import annotations

import multiprocessing
import time
from pathlib import Path

from iai_mcp.capture import _LIVE_ACTIVE_RE, _PROCESSING_MARKER_RE

_DEFAULT_BATCH_BYTES = 100 * 1024 * 1024

# Per-batch wall-clock budget. Scaled by the batch's event count and capped so a
# wedged child cannot block recovery forever.
_BATCH_TIMEOUT_BASE_S = 120.0
_BATCH_TIMEOUT_PER_EVENT_S = 0.2
_BATCH_TIMEOUT_CAP_S = 3600.0


def _count_events(path: Path) -> int:
    """Return the number of event lines in a backlog file.

    Line 0 is the JSON header (verified live keys ``{cwd, deferred_at,
    session_id, version}``), so a header-only or empty file contributes 0. This
    is the single reconciliation authority for the parent's denominator.
    """
    try:
        with path.open("r", encoding="utf-8") as fh:
            lines = [ln for ln in fh if ln.strip()]
    except OSError:
        return 0
    return max(0, len(lines) - 1)


def enumerate_backlog(deferred_dir: Path) -> list[Path]:
    """Return the deterministically-sorted drain-eligible backlog files.

    Includes top-level ``.jsonl`` (excluding ``.live``, ``.processing-<pid>``,
    and ``.permanent-failed-*``), ``.crash-*.jsonl`` (its suffix is ``.jsonl``),
    and ``.quarantine/*.jsonl``. Returns ``[]`` if the directory is absent.
    """
    deferred_dir = Path(deferred_dir)
    if not deferred_dir.exists():
        return []

    files: list[Path] = []
    for entry in deferred_dir.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix != ".jsonl":
            continue
        name = entry.name
        if _LIVE_ACTIVE_RE.search(name):
            continue
        if _PROCESSING_MARKER_RE.search(name):
            continue
        if ".permanent-failed-" in name:
            # Delegated to drain-permanent-failed — keep this command additive.
            continue
        files.append(entry)

    quarantine_dir = deferred_dir / ".quarantine"
    if quarantine_dir.exists():
        files.extend(p for p in quarantine_dir.glob("*.jsonl") if p.is_file())

    return sorted(files, key=str)


def plan_batches(files: list[Path], batch_bytes: int) -> list[list[Path]]:
    """Greedily pack files into batches whose cumulative size <= ``batch_bytes``.

    A single file larger than ``batch_bytes`` becomes its own over-budget batch.
    Every file appears in exactly one batch.
    """
    batches: list[list[Path]] = []
    current: list[Path] = []
    current_bytes = 0

    for f in files:
        try:
            size = f.stat().st_size
        except OSError:
            size = 0

        if size > batch_bytes:
            if current:
                batches.append(current)
                current = []
                current_bytes = 0
            batches.append([f])
            continue

        if current and current_bytes + size > batch_bytes:
            batches.append(current)
            current = []
            current_bytes = 0

        current.append(f)
        current_bytes += size

    if current:
        batches.append(current)

    return batches


def _worker_entry_dd(conn) -> None:  # noqa: ANN001
    """Picklable spawn target for the deferred-drain worker."""
    from iai_mcp.deferred_drain_worker import _worker_entry
    _worker_entry(conn)


def _run_one_batch(store_root: Path, batch: list[Path], event_count: int) -> dict:
    """Drain one batch in a spawn-context child and return its done payload."""
    timeout_s = min(
        _BATCH_TIMEOUT_BASE_S + _BATCH_TIMEOUT_PER_EVENT_S * event_count,
        _BATCH_TIMEOUT_CAP_S,
    )

    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=True)
    process = ctx.Process(
        target=_worker_entry_dd,
        args=(child_conn,),
        daemon=True,
    )
    process.start()
    child_conn.close()

    parent_conn.send((
        "files",
        {"store_root": str(store_root), "paths": [str(p) for p in batch]},
    ))

    deadline = time.monotonic() + timeout_s
    result_kind: str | None = None
    result_payload = None
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process.terminate()
                process.join(timeout=2.0)
                raise RuntimeError(
                    f"deferred-drain child timed out after {timeout_s:.0f}s"
                )
            if parent_conn.poll(min(remaining, 1.0)):
                result_kind, result_payload = parent_conn.recv()
                break
    finally:
        parent_conn.close()

    process.join(timeout=5.0)

    if result_kind == "error":
        raise RuntimeError(f"deferred-drain child error: {result_payload}")
    if result_kind != "done":
        raise RuntimeError(f"deferred-drain child unexpected envelope: {result_kind!r}")
    if process.exitcode != 0:
        raise RuntimeError(f"deferred-drain child exited with code {process.exitcode}")

    return result_payload or {}


def run_deferred_drain(
    *,
    deferred_dir: Path,
    store_root: Path,
    batch_bytes: int = _DEFAULT_BATCH_BYTES,
    dry_run: bool = False,
) -> dict:
    """Recover the full deferred-capture backlog into the store, offline.

    ``dry_run`` enumerates and reports without spawning or mutating anything.
    A real run spawns one child per byte-bounded batch, accumulating counts,
    child-reported event totals, and distinct child PIDs, then reconciles the
    drained total against the parent's authoritative event count.
    """
    deferred_dir = Path(deferred_dir)
    store_root = Path(store_root)

    # Pre-flight: reclaim stale .processing-<dead-pid> markers. No live daemon by
    # design (offline recovery), so unlock everything.
    from iai_mcp.migrate import migrate_unlock_dead_pid_processing_files

    migrate_unlock_dead_pid_processing_files(
        deferred_dir=deferred_dir,
        live_daemon_pid=None,
        dry_run=dry_run,
    )

    files = enumerate_backlog(deferred_dir)
    events_seen = sum(_count_events(p) for p in files)
    total_bytes = sum(_safe_size(p) for p in files)
    batches = plan_batches(files, batch_bytes)

    result: dict = {
        "dry_run": dry_run,
        "files_enumerated": len(files),
        "events_seen": events_seen,
        "total_bytes": total_bytes,
        "batches": [[p.name for p in b] for b in batches],
        "inserted": 0,
        "reinforced": 0,
        "skipped": 0,
        "skipped_existing": 0,
        "count_before": 0,
        "count_after": 0,
        "remaining_events": events_seen,
        "child_events_seen": 0,
        "child_pids": [],
        "reconciled": False,
    }

    if dry_run:
        return result

    from iai_mcp.store import MemoryStore

    # Read the baseline count and ensure the store schema exists, then RELEASE
    # the handle before spawning any writer child (single-writer invariant). A
    # writable open is required here: a read-only open on a never-initialized
    # root has no `records` table yet. The flock is dropped at close() below,
    # well before the first child takes the exclusive writer lock.
    init = MemoryStore(path=store_root)
    try:
        count_before = init.active_records_count()
    finally:
        init.close()

    inserted = reinforced = skipped = skipped_existing = child_events_seen = 0
    child_pids: list[int] = []

    for batch in batches:
        batch_events = sum(_count_events(p) for p in batch)
        payload = _run_one_batch(store_root, batch, batch_events)
        inserted += int(payload.get("inserted", 0))
        reinforced += int(payload.get("reinforced", 0))
        skipped += int(payload.get("skipped", 0))
        skipped_existing += int(payload.get("skipped_existing", 0))
        child_events_seen += int(payload.get("events_seen", 0))
        child_pids.append(int(payload.get("pid", 0)))

    ro2 = MemoryStore(path=store_root, read_only=True)
    try:
        count_after = ro2.active_records_count()
    finally:
        ro2.close()

    remaining_events = sum(_count_events(p) for p in enumerate_backlog(deferred_dir))

    result.update({
        "inserted": inserted,
        "reinforced": reinforced,
        "skipped": skipped,
        "skipped_existing": skipped_existing,
        "count_before": count_before,
        "count_after": count_after,
        "remaining_events": remaining_events,
        "child_events_seen": child_events_seen,
        "child_pids": child_pids,
        "reconciled": (
            inserted + reinforced + skipped_existing + skipped == events_seen
        ),
    })
    return result


def _safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0

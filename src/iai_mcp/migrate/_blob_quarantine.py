"""Quarantine machine-notification blobs captured before the noise filter.

Harness notifications (task notifications, system notices, command
envelopes) are machine traffic, not memories: they carry no recall value
and drown real records on any cue sharing their vocabulary. The capture
noise filter drops them going forward; this sweep tombstones the legacy
ones already stored. Tombstoning is reversible and the whole hippo tree
is snapshotted first, mirroring the idem-dedup sweep.
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from iai_mcp.capture import _NOISE_PATTERNS

log = logging.getLogger(__name__)

# Tombstone criteria for this (destructive, --apply mutates the store) sweep.
# Deliberately narrower than the full noise filter and NOT auto-widened: a
# false positive here silently destroys a record, so growing this set is a
# deliberate, reviewed decision, never an automatic side effect of widening
# the noise filter elsewhere.
_BLOB_PREFIXES: tuple[str, ...] = (
    "<task-notification>",
    "[SYSTEM NOTIFICATION",
    "<local-command-stdout>",
    "<local-command-caveat>",
    "<command-name>",
    "<command-message>",
    "<system-reminder>",
)

# Guard-only superset: the directive-marker blob guard in capture.py must
# never fire on any signature the noise filter (iai_mcp.capture._NOISE_PATTERNS)
# recognizes -- refusing to mint a directive is safe/non-destructive, so this
# set is deliberately widened to union in every noise-filter prefix (startswith
# AND the one equals-type signature, folded in as a prefix -- startswith is a
# strict superset of an equals check for a single literal here), sourced from
# _NOISE_PATTERNS so it can never drift narrower than the filter it backstops.
# NEVER use this tuple for the destructive tombstone sweep above.
_MARKER_GUARD_PREFIXES: tuple[str, ...] = tuple(dict.fromkeys(
    _BLOB_PREFIXES
    + tuple(pattern for kind, pattern in _NOISE_PATTERNS if kind in ("startswith", "equals"))
))


def _is_notification_blob(text: str) -> bool:
    head = text.lstrip()[:80]
    return any(head.startswith(p) for p in _BLOB_PREFIXES)


def quarantine_notification_blobs(
    store,
    apply: bool = False,
    store_path: Path | None = None,
) -> dict:
    from iai_mcp.store._buffers import flush_record_buffer

    flush_record_buffer(store)

    with store.db._conn_lock:
        rows = store.db._conn.execute(
            "SELECT id FROM records WHERE tombstoned_at IS NULL"
        ).fetchall()
        ids = [str(r[0]) for r in rows]

    scanned = 0
    errors: list[str] = []
    to_tombstone: list[tuple[str, str]] = []
    for rid_s in ids:
        scanned += 1
        try:
            rec = store.get(UUID(rid_s))
            if rec is None or not rec.literal_surface:
                continue
            if rec.never_merge or rec.never_decay:
                continue
            if _is_notification_blob(rec.literal_surface):
                # 400 chars in the journal: an audit must be able to see
                # whether user text follows the machine envelope.
                to_tombstone.append((rid_s, rec.literal_surface[:400]))
        except Exception as exc:  # noqa: BLE001 -- one bad record must not abort the sweep
            errors.append(f"{rid_s}: scan: {type(exc).__name__}: {exc}")

    snapshot_dir: str | None = None
    journal_path: str | None = None
    tombstoned = 0
    if apply and to_tombstone:
        iai_root = Path(store_path) if store_path is not None else Path(store.root)
        src_hippo = iai_root / "hippo"
        if not src_hippo.exists():
            # Falling back to the store root would copytree the root into
            # a directory inside itself.
            raise FileNotFoundError(
                f"refusing to apply without a hippo dir to snapshot: {src_hippo}"
            )
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snap = iai_root / f"hippo-pre-blob-quarantine-{ts}"
        shutil.copytree(src_hippo, snap)
        snapshot_dir = str(snap)

        journal_dir = iai_root / "backfills"
        journal_dir.mkdir(parents=True, exist_ok=True)
        journal = journal_dir / f"blob-quarantine-{ts}.jsonl"
        journal_path = str(journal)

        now_iso = datetime.now(timezone.utc).isoformat()
        tbl = store.db.open_table("records")
        tombstoned_ids: list[str] = []
        with journal.open("x", encoding="utf-8") as jf:
            for rid_s, head in to_tombstone:
                try:
                    tbl.update(
                        where=f"id = '{rid_s}'",
                        values={"tombstoned_at": now_iso, "live": 0},
                    )
                    tombstoned += 1
                    tombstoned_ids.append(rid_s)
                    jf.write(json.dumps({
                        "record_id": rid_s,
                        "tombstoned_at": now_iso,
                        "head": head,
                    }, ensure_ascii=False) + "\n")
                    jf.flush()
                except Exception as exc:  # noqa: BLE001 -- one bad record must not abort the sweep
                    errors.append(
                        f"{rid_s}: apply: {type(exc).__name__}: {exc}"
                    )
        try:
            store._invalidate_corpus_count("active")
        except Exception:  # noqa: BLE001 -- count cache refresh is best-effort
            pass
        _inv_x = getattr(store, "invalidate_exact_index", None)
        if callable(_inv_x):
            _inv_x()  # no-raise contract; see MemoryStore.invalidate_exact_index
        buf = getattr(store, "_recency_buffer", None)
        if buf is not None:
            for rid_s in tombstoned_ids:
                try:
                    buf.evict(str(rid_s))
                except Exception as exc:  # noqa: BLE001 -- eviction is best-effort
                    log.debug("recency buffer evict failed for %s: %s", rid_s, exc)

    return {
        "mode": "apply" if apply else "dry-run",
        "records_scanned": scanned,
        "blobs_found": len(to_tombstone),
        "tombstoned": tombstoned,
        "snapshot_dir": snapshot_dir,
        "journal": journal_path,
        "errors": errors,
    }

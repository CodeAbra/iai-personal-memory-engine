"""Retire phantom standing-order directives planted before explicit-only
capture (the retired fuzzy classify_is_directive write branch).

Mirrors the idem-dedup / blob-quarantine sweep shape: dry-run by default,
snapshot-first --apply. The retire target is every live directive record
whose provenance carries NO explicit-declaration stamp (directive_source
neither "explicit-marker" nor "explicit-command") -- at first run that is
every pre-fix record, since none were stamped. literal_surface is never
read for selection and never written by the apply step; only the
directive flag column is flipped.
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

_EXPLICIT_DIRECTIVE_SOURCES = frozenset({"explicit-marker", "explicit-command"})


def _is_explicitly_stamped(provenance: "list | None") -> bool:
    if not isinstance(provenance, list):
        return False
    return any(
        isinstance(entry, dict)
        and entry.get("directive_source") in _EXPLICIT_DIRECTIVE_SOURCES
        for entry in provenance
    )


def sweep_phantom_directives(
    store,
    *,
    apply: bool = False,
    store_path: "Path | str | None" = None,
) -> dict:
    from uuid import UUID

    from iai_mcp.store import RECORDS_TABLE, _uuid_literal, flush_record_buffer

    flush_record_buffer(store)

    # Per-record scan (mirrors _blob_quarantine.quarantine_notification_blobs):
    # a raw id list first, then one store.get() per id inside try/except, so a
    # single undecryptable/malformed directive record is skipped and reported
    # rather than aborting the whole batch -- list(store.iter_records(...))
    # eagerly drives the decrypting generator to completion in one call, so
    # one bad record's decrypt failure used to propagate out of that call and
    # crash the sweep before any record was even classified.
    with store.db._conn_lock:
        rows = store.db._conn.execute(
            "SELECT id FROM records WHERE directive = 1 AND tombstoned_at IS NULL"
        ).fetchall()
        ids = [str(r[0]) for r in rows]

    directives_found = 0
    targets = []
    scan_errors: list[str] = []
    for rid_s in ids:
        try:
            rec = store.get(UUID(rid_s))
            if rec is None:
                continue
            directives_found += 1
            if not _is_explicitly_stamped(rec.provenance):
                targets.append(rec)
        except Exception as exc:  # noqa: BLE001 -- one bad record must not abort the sweep
            scan_errors.append(f"{rid_s}: scan: {type(exc).__name__}: {exc}")
    unstamped = len(targets)
    failed = len(scan_errors)

    if not apply or not targets:
        return {
            "mode": "apply" if apply else "dry-run",
            "directives_found": directives_found,
            "unstamped": unstamped,
            "retired": 0,
            "failed": failed,
            "errors": scan_errors,
            "snapshot_dir": None,
            "cache_refreshed": False,
        }

    root = Path(store_path) if store_path is not None else Path(store.root)
    src_hippo = root / "hippo"
    if not src_hippo.exists():
        # Falling back to the store root would copytree the root into a
        # directory inside itself.
        raise FileNotFoundError(
            f"refusing to apply without a hippo dir to snapshot: {src_hippo}"
        )
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snap = root / f"hippo-pre-directive-sweep-{ts}"
    shutil.copytree(src_hippo, snap)
    snapshot_dir = str(snap)

    tbl = store.db.open_table(RECORDS_TABLE)
    retired = 0
    for rec in targets:
        try:
            tbl.update(
                where=f"id = '{_uuid_literal(rec.id)}'",
                values={"directive": False},
            )
            retired += 1
        except Exception as exc:  # noqa: BLE001 -- one bad record must not abort the sweep
            log.warning("directive_sweep_clear_failed id=%s: %s", rec.id, exc)
            scan_errors.append(f"{rec.id}: apply: {type(exc).__name__}: {exc}")
    failed = len(scan_errors)

    # Load-bearing: DIRECTIVES_CACHE_PATH is computed from the home directory
    # at import time -- a bare write_directives_cache(store) call would
    # rewrite the live home cache even during a --store-path run.
    cache_path = root / ".directives.cached.md"
    cache_refreshed = False
    try:
        from iai_mcp.directive_cache import write_directives_cache

        write_directives_cache(store, cache_path=cache_path)
        cache_refreshed = True
    except Exception as exc:  # noqa: BLE001 -- cache refresh must never break the sweep
        log.warning("directive_sweep_cache_write_failed: %s", exc, exc_info=True)

    return {
        "mode": "apply",
        "directives_found": directives_found,
        "unstamped": unstamped,
        "retired": retired,
        "failed": failed,
        "errors": scan_errors,
        "snapshot_dir": snapshot_dir,
        "cache_refreshed": cache_refreshed,
    }

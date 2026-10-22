"""Tombstone internal-prompt/template-pollution records, plus any operator-
listed leak records a content predicate cannot safely generalize from.

Mirrors the blob-quarantine / directive-sweep shape: dry-run by default,
snapshot-first ``--apply``, tombstone via a ``records`` table update, never a
hard-DELETE. Two independent targeting modes run in the same pass: predicate
(a narrow reviewed prefix match on ``literal_surface``) and explicit-id (an
operator-supplied record-id list). After tombstoning, a full-store orphan-edge
sweep runs synchronously -- the same routine the nightly sleep cycle uses, not
scoped to only the rows this pass just tombstoned -- and the ANN index is
rebuilt, so a post-purge query never serves a purged row without waiting for
the nightly cycle.
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

log = logging.getLogger(__name__)

# Own narrow, reviewed tuple -- a false positive here silently destroys a
# record, so growing this set is a deliberate, reviewed decision, never
# derived from _NOISE_PATTERNS or any other filter's vocabulary.
_POLLUTION_PREFIXES: tuple[str, ...] = (
    "Below is one record from a developer",
)


def _is_pollution_template(text: str) -> bool:
    head = text.lstrip()[:80]
    return any(head.startswith(p) for p in _POLLUTION_PREFIXES)


def sweep_template_pollution(
    store,
    apply: bool = False,
    store_path: Path | None = None,
    explicit_ids: "list[str] | None" = None,
) -> dict:
    from iai_mcp.store import RECORDS_TABLE, _uuid_literal
    from iai_mcp.store._buffers import flush_record_buffer

    flush_record_buffer(store)

    explicit_id_set = {str(x) for x in (explicit_ids or [])}

    with store.db._conn_lock:
        rows = store.db._conn.execute(
            "SELECT id FROM records WHERE tombstoned_at IS NULL"
        ).fetchall()
        ids = [str(r[0]) for r in rows]

    scanned = 0
    errors: list[str] = []
    # (record_id, journal_head, reason, was_directive)
    to_tombstone: list[tuple[str, str, str, bool]] = []
    skipped_pinned = 0
    matched_explicit_ids: set[str] = set()
    explicit_ids_protected: list[str] = []

    for rid_s in ids:
        scanned += 1
        try:
            rec = store.get(UUID(rid_s))
            if rec is None or not rec.literal_surface:
                continue
            is_predicate_match = _is_pollution_template(rec.literal_surface)
            is_explicit_match = rid_s in explicit_id_set
            if not (is_predicate_match or is_explicit_match):
                continue
            # Found is decided FIRST, independent of the pinned outcome below --
            # an explicit id that exists but is pinned must never read as
            # "missing" (a leak record could otherwise look purged when it is
            # still live and deliberately protected).
            if is_explicit_match:
                matched_explicit_ids.add(rid_s)
            # Predicate/id match decided FIRST -- a pinned record that
            # coincidentally matches must be visible as skipped_pinned, never
            # folded into a scan of every pinned record in the store.
            # Tombstone-protection gate is never_merge OR never_decay --
            # broader than capture.py's fold-protection gate (never_merge
            # only): tombstoning is destructive/irreversible, folding is not.
            if rec.never_merge or rec.never_decay:
                skipped_pinned += 1
                if is_explicit_match:
                    explicit_ids_protected.append(rid_s)
                continue
            reason = "predicate" if is_predicate_match else "explicit_id"
            to_tombstone.append(
                (rid_s, rec.literal_surface[:400], reason, bool(rec.directive))
            )
        except Exception as exc:  # noqa: BLE001 -- one bad record must not abort the sweep
            errors.append(f"{rid_s}: scan: {type(exc).__name__}: {exc}")

    snapshot_dir: str | None = None
    journal_path: str | None = None
    tombstoned = 0
    orphan_sweep_payload: dict | None = None
    ann_rebuild_invoked = False
    exact_index_invalidated = False
    directive_cache_refreshed = False

    if apply and to_tombstone:
        iai_root = Path(store_path) if store_path is not None else Path(store.root)
        src_hippo = iai_root / "hippo"
        if not src_hippo.exists():
            # Falling back to the store root would copytree the root into a
            # directory inside itself.
            raise FileNotFoundError(
                f"refusing to apply without a hippo dir to snapshot: {src_hippo}"
            )
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snap = iai_root / f"hippo-pre-template-pollution-{ts}"
        shutil.copytree(src_hippo, snap)
        snapshot_dir = str(snap)

        journal_dir = iai_root / "backfills"
        journal_dir.mkdir(parents=True, exist_ok=True)
        # Exclusive-create: a same-second retried --apply needs a >1s gap
        # (mirrors _blob_quarantine.py's journal discipline).
        journal = journal_dir / f"template-pollution-{ts}.jsonl"
        journal_path = str(journal)

        now_iso = datetime.now(timezone.utc).isoformat()
        tbl = store.db.open_table(RECORDS_TABLE)
        tombstoned_ids: list[str] = []
        had_directive = False
        with journal.open("x", encoding="utf-8") as jf:
            for rid_s, head, reason, was_directive in to_tombstone:
                try:
                    # _uuid_literal enforces the strict UUID shape at the
                    # interpolation site itself, not only via the upstream
                    # UUID(rid_s) validation this id already passed.
                    tbl.update(
                        where=f"id = '{_uuid_literal(rid_s)}'",
                        values={"tombstoned_at": now_iso, "live": 0},
                    )
                    tombstoned += 1
                    tombstoned_ids.append(rid_s)
                    if was_directive:
                        had_directive = True
                    jf.write(json.dumps({
                        "record_id": rid_s,
                        "tombstoned_at": now_iso,
                        "reason": reason,
                        "head": head,
                    }, ensure_ascii=False) + "\n")
                    jf.flush()
                except Exception as exc:  # noqa: BLE001 -- one bad record must not abort the sweep
                    errors.append(f"{rid_s}: apply: {type(exc).__name__}: {exc}")

        try:
            store._invalidate_corpus_count("active")
        except Exception as exc:  # noqa: BLE001 -- count cache refresh is best-effort
            log.debug("template_pollution_corpus_count_invalidate_failed: %s", exc)

        # Each maintenance call below is independently guarded: a failure
        # here must not lose the operator's own rollback pointers
        # (snapshot_dir/journal, already computed above) by propagating out
        # of this function before the summary is returned.
        try:
            from iai_mcp.lilli.cycle.sleep_pipeline._compact import _sweep_orphan_edges

            orphan_sweep_payload = _sweep_orphan_edges(store)
        except Exception as exc:  # noqa: BLE001 -- see guard note above
            errors.append(f"orphan_edge_sweep: {type(exc).__name__}: {exc}")

        try:
            _inv_exact = getattr(store, "invalidate_exact_index", None)
            if callable(_inv_exact):
                _inv_exact()  # no-raise contract; see MemoryStore.invalidate_exact_index
                exact_index_invalidated = True
        except Exception as exc:  # noqa: BLE001 -- see guard note above
            errors.append(f"invalidate_exact_index: {type(exc).__name__}: {exc}")

        try:
            _rebuild = getattr(store.db, "_rebuild_index_from_sqlite", None)
            if callable(_rebuild):
                _rebuild()
                ann_rebuild_invoked = True
        except Exception as exc:  # noqa: BLE001 -- see guard note above
            errors.append(f"ann_rebuild: {type(exc).__name__}: {exc}")

        try:
            from iai_mcp import runtime_graph_cache as _rgc
            _rgc.invalidate(store)
        except Exception as exc:  # noqa: BLE001 -- invalidation must never break the sweep
            log.debug("template_pollution_runtime_graph_cache_invalidate_failed: %s", exc)

        buf = getattr(store, "_recency_buffer", None)
        if buf is not None:
            for rid_s in tombstoned_ids:
                try:
                    buf.evict(str(rid_s))
                except Exception as exc:  # noqa: BLE001 -- eviction is best-effort
                    log.debug("recency buffer evict failed for %s: %s", rid_s, exc)

        if had_directive:
            # Load-bearing: the cache path is computed from the same root the
            # apply ran against, so a --store-path run never rewrites the
            # live home cache (mirrors _directive_sweep.py's cache write).
            cache_path = iai_root / ".directives.cached.md"
            try:
                from iai_mcp.directive_cache import write_directives_cache

                write_directives_cache(store, cache_path=cache_path)
                directive_cache_refreshed = True
            except Exception as exc:  # noqa: BLE001 -- cache refresh must never break the sweep
                log.warning("template_pollution_cache_write_failed: %s", exc, exc_info=True)

    # An explicit id absent from the live-id scan is either already
    # tombstoned by a prior run (excluded from the `tombstoned_at IS NULL`
    # scan at the top of this function) or genuinely never existed -- a
    # point lookup per unresolved id (a handful at most) distinguishes the
    # two so "missing" means only "not found in the store at all."
    already_tombstoned_explicit_ids: list[str] = []
    unresolved_explicit_ids = sorted(explicit_id_set - matched_explicit_ids)
    if unresolved_explicit_ids:
        for candidate in unresolved_explicit_ids:
            try:
                lit = _uuid_literal(candidate)
            except ValueError:
                continue
            with store.db._conn_lock:
                row = store.db._conn.execute(
                    "SELECT tombstoned_at FROM records WHERE id = ?", (lit,),
                ).fetchone()
            if row is not None and row["tombstoned_at"] is not None:
                already_tombstoned_explicit_ids.append(candidate)

    missing_explicit_ids = sorted(
        explicit_id_set - matched_explicit_ids - set(already_tombstoned_explicit_ids)
    )

    return {
        "mode": "apply" if apply else "dry-run",
        "records_scanned": scanned,
        "pollution_found": len(to_tombstone),
        "tombstoned": tombstoned,
        "skipped_pinned": skipped_pinned,
        "explicit_ids_protected": sorted(set(explicit_ids_protected)),
        "explicit_ids_already_tombstoned": already_tombstoned_explicit_ids,
        "missing_explicit_ids": missing_explicit_ids,
        "snapshot_dir": snapshot_dir,
        "journal": journal_path,
        "orphan_edge_sweep": orphan_sweep_payload,
        "ann_rebuild_invoked": ann_rebuild_invoked,
        "exact_index_invalidated": exact_index_invalidated,
        "directive_cache_refreshed": directive_cache_refreshed,
        "errors": errors,
    }

"""Noise exclusion at the ``memory_recall`` serving choke point.

Reuses the capture-time noise predicate and the tombstone-sweep carve-out
gate so a noise record already in the store can never leak through the
recall-serving path, regardless of whether it was swept at capture time.
"""

from __future__ import annotations


def is_excludable_noise(rec) -> bool:
    from iai_mcp.capture import _is_noise

    if not _is_noise(rec.literal_surface or ""):
        return False
    # Carve-out gate verbatim from migrate/_template_pollution.py /
    # migrate/_blob_quarantine.py -- a locked record is never excluded.
    return not (rec.never_merge or rec.never_decay)


def exclude_serving_noise(store, hits: list) -> list:
    from iai_mcp.capture import _is_noise

    flagged_ids = [h.record_id for h in hits if _is_noise(h.literal_surface or "")]
    if not flagged_ids:
        return hits

    try:
        records = store.get_batch(flagged_ids, decode="full")
    except Exception:  # noqa: BLE001 -- fail toward leak: a broken lookup
        # must never drop a hit that would otherwise be served.
        return hits

    excluded_ids = {
        rid
        for rid in flagged_ids
        if (rec := records.get(rid)) is not None and is_excludable_noise(rec)
    }
    if not excluded_ids:
        return hits
    return [h for h in hits if h.record_id not in excluded_ids]

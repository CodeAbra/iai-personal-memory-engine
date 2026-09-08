"""Shared retire/resolve seam for standing-order directives.

Both the CLI `directive remove` handler and the drain-worker REMOVE-marker
path retire through `retire_directive()` -- one write path, one cache
refresh, so every flip stays consistent across callers and drivers.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from uuid import UUID

from iai_mcp.store import RECORDS_TABLE, MemoryStore, _uuid_literal, flush_record_buffer


class ResolveOutcome(Enum):
    LIVE = "live"
    ALREADY_RETIRED = "already_retired"
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class ResolveResult:
    outcome: ResolveOutcome
    record_id: "UUID | None" = None


def iter_live_directives(store: MemoryStore) -> "list[tuple[str, UUID, str]]":
    """Ordered (short_id, record_id, text) triples for live directives.

    short_id is the first 8 hex chars of the record's dashless UUID --
    stable because it derives from the immutable id, not list position.
    """
    from iai_mcp.session import _live_directive_ids

    ids = _live_directive_ids(store)
    out: list[tuple[str, UUID, str]] = []
    for rid in ids:
        rec = store.get(rid)
        if rec is None:
            continue
        out.append((rid.hex[:8], rid, rec.literal_surface or ""))
    return out


def _all_record_ids(store: MemoryStore) -> "list[UUID]":
    flush_record_buffer(store)
    db = store.db
    with db._conn_lock:
        rows = db._conn.execute("SELECT id FROM records").fetchall()
    out: list[UUID] = []
    for row in rows:
        try:
            out.append(UUID(row["id"]))
        except (ValueError, TypeError):
            continue
    return out


def resolve_directive_short_id(store: MemoryStore, token: "str | None") -> ResolveResult:
    """Resolve `token` (a full UUID or a short-id prefix) against live
    directives first, then against all records so the CLI/drain caller can
    distinguish an already-retired id from a genuinely unknown one.
    """
    token_norm = (token or "").strip().lower().replace("-", "")
    if not token_norm:
        return ResolveResult(ResolveOutcome.UNKNOWN)

    live = iter_live_directives(store)
    live_matches = [rid for _short, rid, _text in live if rid.hex.startswith(token_norm)]
    if len(live_matches) == 1:
        return ResolveResult(ResolveOutcome.LIVE, live_matches[0])
    if len(live_matches) > 1:
        return ResolveResult(ResolveOutcome.AMBIGUOUS)

    for rid in _all_record_ids(store):
        if rid.hex.startswith(token_norm):
            return ResolveResult(ResolveOutcome.ALREADY_RETIRED)
    return ResolveResult(ResolveOutcome.UNKNOWN)


def retire_directive(store: MemoryStore, record_id: UUID) -> None:
    """Flip `directive` -> False for one record, then synchronously refresh
    the directive cache from the store root -- never `tombstoned_at`, never
    `live`. Same shape as `retrieve.contradict()` and
    `migrate/_directive_sweep.py::sweep_phantom_directives`.
    """
    flush_record_buffer(store)
    tbl = store.db.open_table(RECORDS_TABLE)
    tbl.update(
        where=f"id = '{_uuid_literal(record_id)}'",
        values={"directive": False},
    )

    from iai_mcp.directive_cache import write_directives_cache

    # Derived from the store root, not the bare default -- a bare call would
    # write the operator's real ~/.iai-mcp cache even during a tmp-store test.
    cache_path = store.root / ".directives.cached.md"
    write_directives_cache(store, cache_path=cache_path)


__all__ = [
    "ResolveOutcome",
    "ResolveResult",
    "iter_live_directives",
    "resolve_directive_short_id",
    "retire_directive",
]

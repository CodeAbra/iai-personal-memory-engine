"""Procedural chunk minter: a gated CofirePairCandidate becomes a real
tier="procedural" MemoryRecord plus one directed proc_transitions row.

Payload is ids/role/source only -- never another record's literal_surface.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from iai_mcp.events import write_event
from iai_mcp.lilli.cycle.proc_mine import (
    MIN_DISTINCT_SESSIONS,
    PAIR_COUNT_FLOOR,
    CofirePairCandidate,
)
from iai_mcp.store import RECORDS_TABLE, MemoryStore, _uuid_literal, flush_record_buffer
from iai_mcp.types import MemoryRecord, SCHEMA_VERSION_CURRENT

#: Below tier=3, __post_init__ leaves never_decay caller-controlled -- a
#: chunk must stay retirable, unlike a permanent schema record.
CHUNK_DETAIL_LEVEL: int = 1

_TRANSITIONS_TABLE = "proc_transitions"

#: A chunk becomes decay-eligible only once BOTH windows have elapsed
#: (AND, not OR): it must be old enough (age) AND not recently reinforced
#: (staleness). Reinforcement (persist_proc_chunk's repeat-sighting path)
#: advances last_reviewed and resets the staleness clause independently of
#: age, so a chunk that keeps recurring never decays regardless of its
#: original mint date.
CHUNK_DECAY_AGE_DAYS: int = 180
CHUNK_DECAY_STALENESS_DAYS: int = 90


def _chunk_label(candidate: CofirePairCandidate) -> str:
    return f"chunk {candidate.pair[0]}->{candidate.pair[1]} via {candidate.source}"


def _live_transition_chunk_id(
    store: MemoryStore, src: str, dst: str, source: str
) -> UUID | None:
    with store.db._conn_lock:
        row = store.db._conn.execute(
            f"SELECT chunk_id FROM {_TRANSITIONS_TABLE} WHERE src = ? AND dst = ? AND source = ?",
            (src, dst, source),
        ).fetchone()
    if row is None or not row["chunk_id"]:
        return None
    candidate_id = UUID(row["chunk_id"])
    # store.insert() buffers rows until a flush threshold; a same-process
    # sighting of a just-minted chunk must see it as live, or dedup mints
    # a second chunk for the same (src,dst,source) instead of reinforcing.
    flush_record_buffer(store)
    tbl = store.db.open_table(RECORDS_TABLE)
    live = (
        tbl.count_rows(
            filter=f"id = '{_uuid_literal(candidate_id)}' AND tombstoned_at IS NULL"
        )
        == 1
    )
    return candidate_id if live else None


def _write_transition_row(
    store: MemoryStore, candidate: CofirePairCandidate, chunk_id: UUID, now: datetime
) -> None:
    src, dst = candidate.pair
    store.db.open_table(_TRANSITIONS_TABLE).merge_insert(
        ["src", "dst", "source"]
    ).execute(
        [
            {
                "src": src,
                "dst": dst,
                "source": candidate.source,
                "count": candidate.count,
                "session_count": candidate.session_count,
                "first_ts": candidate.first_ts.isoformat(),
                "last_ts": candidate.last_ts.isoformat(),
                "chunk_id": str(chunk_id),
                "updated_at": now.isoformat(),
            }
        ]
    )


def persist_proc_chunk(store: MemoryStore, candidate: CofirePairCandidate) -> UUID | None:
    if candidate.count < PAIR_COUNT_FLOOR or candidate.session_count < MIN_DISTINCT_SESSIONS:
        return None

    from iai_mcp.aaak import enforce_language_tagged, generate_aaak_index
    from iai_mcp.embed import embedder_for_store

    src, dst = candidate.pair
    now = datetime.now(timezone.utc)

    existing_chunk_id = _live_transition_chunk_id(store, src, dst, candidate.source)
    if existing_chunk_id is not None:
        _write_transition_row(store, candidate, existing_chunk_id, now)
        store.reinforce_record(existing_chunk_id, is_retrieval=True)
        write_event(
            store,
            kind="proc_chunk_reinforced",
            data={
                "chunk_id": str(existing_chunk_id),
                "pair": [src, dst],
                "source": candidate.source,
                "count": candidate.count,
                "session_count": candidate.session_count,
            },
            severity="info",
            source_ids=[existing_chunk_id],
        )
        return existing_chunk_id

    label = _chunk_label(candidate)
    emb = embedder_for_store(store).embed(label)
    chunk_id = uuid4()
    chunk_rec = MemoryRecord(
        id=chunk_id,
        tier="procedural",
        literal_surface=label,
        aaak_index="",
        embedding=emb,
        community_id=None,
        centrality=0.0,
        detail_level=CHUNK_DETAIL_LEVEL,
        pinned=False,
        stability=0.7,
        difficulty=0.3,
        last_reviewed=now,
        never_decay=False,
        never_merge=False,
        provenance=[
            {"ts": now.isoformat(), "cue": "proc_chunk_mint", "session_id": "system"}
        ],
        created_at=now,
        updated_at=now,
        tags=["chunk", f"source:{candidate.source}"],
        language="en",
        s5_trust_score=0.5,
        profile_modulation_gain={},
        schema_version=SCHEMA_VERSION_CURRENT,
    )
    enforce_language_tagged(chunk_rec)
    chunk_rec.aaak_index = generate_aaak_index(chunk_rec)
    store.insert(chunk_rec)

    _write_transition_row(store, candidate, chunk_id, now)

    write_event(
        store,
        kind="proc_chunk_minted",
        data={
            "chunk_id": str(chunk_id),
            "pair": [src, dst],
            "source": candidate.source,
            "count": candidate.count,
            "session_count": candidate.session_count,
        },
        severity="info",
        source_ids=[chunk_id],
    )
    return chunk_id


def decay_proc_chunks(
    store: MemoryStore, *, now: datetime | None = None, dry_run: bool = False,
) -> dict[str, int]:
    """Tombstone stale procedural chunks (tombstoned_at set, live=0, row
    kept). Mirrors step_erasure_agent's tombstone mechanism but scoped to
    tier='procedural' and driven by its own AND-gated age/staleness
    constants rather than the shared erasure config -- reinforcement
    (persist_proc_chunk's repeat-sighting path) resets last_reviewed
    independently, so a still-recurring chunk never decays.
    """
    now = now or datetime.now(timezone.utc)
    age_cutoff = now - timedelta(days=CHUNK_DECAY_AGE_DAYS)
    staleness_cutoff = now - timedelta(days=CHUNK_DECAY_STALENESS_DAYS)
    # Lexical TEXT comparison against stored timestamps: match the stored
    # column format (space-separated, no 'T'), not .isoformat().
    age_cutoff_str = age_cutoff.strftime("%Y-%m-%d %H:%M:%S")
    staleness_cutoff_str = staleness_cutoff.strftime("%Y-%m-%d %H:%M:%S")

    tbl = store.db.open_table(RECORDS_TABLE)
    eligibility_where = (
        f"tier = 'procedural' "
        f"AND (last_reviewed IS NULL OR last_reviewed < '{staleness_cutoff_str}') "
        f"AND created_at < '{age_cutoff_str}' "
        f"AND pinned = false "
        f"AND never_decay = false "
        f"AND tombstoned_at IS NULL "
        f"AND (directive = 0 OR directive IS NULL)"
    )

    quarantined = int(tbl.count_rows(filter=eligibility_where))
    retired = 0
    if not dry_run and quarantined > 0:
        tbl.update(where=eligibility_where, values={"tombstoned_at": now, "live": 0})
        retired = quarantined

    return {"quarantined": quarantined, "retired": retired}

"""Signal-A miner: retrieval_cofired events -> gated directed-pair candidates.

Reads only `used_ids` (the assistant's own first-mention-order reflection),
never `hit_ids` (the ranker's own output) -- letting a ranker's rank order
influence the mined signal is the self-confirmation trap.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from iai_mcp.events import flush_event_buffer, is_attributed_session, query_events
from iai_mcp.lilli.profile.retrieval_tuning import RETRIEVAL_MIN_SAMPLES
from iai_mcp.store._store import _normalize_ts_for_compare

COFIRE_MINE_SOURCE: str = "retrieval_cofired"

# Bound to the sibling tuner's sample floor -- never re-inline this literal.
PAIR_COUNT_FLOOR = RETRIEVAL_MIN_SAMPLES

# Reused-not-re-derived starter value, overridable by a later census.
MIN_DISTINCT_SESSIONS: int = 3

# Defeats query_events' default limit=100, which silently truncates.
_EVENT_LOAD_LIMIT: int = 10_000_000


@dataclass(frozen=True)
class CofirePairCandidate:
    pair: tuple[str, str]
    source: str
    count: int
    session_count: int
    sessions: frozenset[str]
    first_ts: datetime
    last_ts: datetime
    boundary_count: int = 0


def mine_cofired_pairs(
    events: list[dict],
    *,
    min_count: int = PAIR_COUNT_FLOOR,
    min_distinct_sessions: int = MIN_DISTINCT_SESSIONS,
) -> list[CofirePairCandidate]:
    """Directed consecutive-pair counts over `used_ids`, gated on intra-event
    count AND distinct-session spread (both inclusive). Pure -- no store, no
    clock, no randomness; touches only the passed-in event dicts.

    Inter-event pairs (last id of event N -> first id of event N+1, same
    session) are reported as boundary_count on an already-eligible pair only
    -- they never contribute to the gating count or session spread.
    """
    by_session: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        by_session[event.get("session_id", "-")].append(event)

    intra_counts: "Counter[tuple[str, str]]" = Counter()
    intra_sessions: dict[tuple[str, str], set[str]] = defaultdict(set)
    intra_ts: dict[tuple[str, str], list[datetime]] = defaultdict(list)
    inter_counts: "Counter[tuple[str, str]]" = Counter()

    for session_id, sess_events in by_session.items():
        ordered = sorted(sess_events, key=lambda e: (e["ts"], str(e.get("id") or "")))
        for event in ordered:
            ids = (event.get("data") or {}).get("used_ids") or []
            ts = event["ts"]
            for a, b in zip(ids, ids[1:]):
                pair = (a, b)
                intra_counts[pair] += 1
                intra_sessions[pair].add(session_id)
                intra_ts[pair].append(ts)
        for prev_event, next_event in zip(ordered, ordered[1:]):
            prev_ids = (prev_event.get("data") or {}).get("used_ids") or []
            next_ids = (next_event.get("data") or {}).get("used_ids") or []
            if not prev_ids or not next_ids:
                continue
            inter_counts[(prev_ids[-1], next_ids[0])] += 1

    candidates: list[CofirePairCandidate] = []
    for pair, count in intra_counts.items():
        session_count = len(intra_sessions[pair])
        if count >= min_count and session_count >= min_distinct_sessions:
            ts_list = intra_ts[pair]
            candidates.append(
                CofirePairCandidate(
                    pair=pair,
                    source=COFIRE_MINE_SOURCE,
                    count=count,
                    session_count=session_count,
                    sessions=frozenset(intra_sessions[pair]),
                    first_ts=min(ts_list),
                    last_ts=max(ts_list),
                    boundary_count=inter_counts.get(pair, 0),
                )
            )

    candidates.sort(key=lambda c: (-c.count, c.pair))
    return candidates


def _count_matching_events(store: Any, kind: str, since: "datetime | None") -> int:
    where_parts = ["kind = ?"]
    params: list[Any] = [kind]
    if since is not None:
        where_parts.append("ts >= ?")
        params.append(_normalize_ts_for_compare(since).replace("T", " "))
    sql = "SELECT COUNT(*) FROM events WHERE " + " AND ".join(where_parts)
    with store.db._conn_lock:
        row = store.db._conn.execute(sql, params).fetchone()
    return int(row[0]) if row else 0


def load_cofired_events(store: Any, since: "datetime | None" = None) -> list[dict]:
    """Every retrieval_cofired event, ascending by row ts, attributed sessions
    only, with the ranker-authored hit_ids field stripped before the pure
    miner ever sees the dict.
    """
    flush_event_buffer(store)
    expected = _count_matching_events(store, COFIRE_MINE_SOURCE, since)
    events = query_events(store, kind=COFIRE_MINE_SOURCE, since=since, limit=_EVENT_LOAD_LIMIT)
    # A writer landing between the two reads above can only grow the row
    # count; only fewer rows than counted is a genuine _EVENT_LOAD_LIMIT cut.
    if len(events) < expected:
        raise RuntimeError(
            f"retrieval_cofired load truncated: table has {expected} "
            f"matching rows, loaded {len(events)} -- raise _EVENT_LOAD_LIMIT"
        )
    events = [e for e in events if is_attributed_session(e.get("session_id"))]
    for event in events:
        data = event.get("data")
        if isinstance(data, dict):
            data.pop("hit_ids", None)
    events.sort(key=lambda e: (e["ts"], str(e.get("id") or "")))
    return events

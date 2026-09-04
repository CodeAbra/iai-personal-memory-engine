"""Signal-B miner: sleep-time tool-call-sequence bigrams -> gated directed-pair
candidates, feeding the same signal-agnostic chunk/transition sink Signal A
(proc_mine.py) built.

Reads only the `[tools: ...]` trailer riding an assistant turn's own verbatim
surface -- producer-appended provenance text, never a retrieval artifact and
never rewritten by this module.

Trailer text is not structurally verified: a zero-tool-call turn whose own
generated text happens to end with `\n[tools: X, Y]` is byte-indistinguishable
from a genuine producer-appended trailer -- no field on the record
cross-checks parsed tool names against an actual tool-call log. Impact is
bounded to an inert chunk label (`literal_surface` is never touched) and
needs a sustained repeated pattern to clear the mint gate; prefer a
structured tool-call log over trailer-text parsing for anything that mints
durable state, if one is ever added.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from iai_mcp.crypto import is_encrypted
from iai_mcp.lilli.cycle.proc_mine import (
    MIN_DISTINCT_SESSIONS,
    PAIR_COUNT_FLOOR,
    CofirePairCandidate,
)
from iai_mcp.store._store import _parse_ts_field

TOOLSEQ_MINE_SOURCE: str = "tool_sequence"

# \Z-anchored so a mid-text `[tools: ...]`-shaped decoy cannot match.
_TRAILER_RE = re.compile(r"\n\[tools: ([^\]]+)\]\Z")

_ASSISTANT_WHERE = "tags_json LIKE '%role:assistant%' AND tombstoned_at IS NULL"


def parse_tool_sequence(literal_surface: str) -> list[str]:
    """Read-only regex parse of the trailing `[tools: ...]` trailer. Never a
    write; returns names, never assigns back onto its input."""
    match = _TRAILER_RE.search(literal_surface or "")
    if not match:
        return []
    body = re.sub(r" \+\d+\Z", "", match.group(1))
    names = [n.strip() for n in body.split(", ") if n.strip()]
    return names[:80]  # defensive bound -- never trust the producer's own cap blindly


def _normalize_created_at(value: Any) -> "datetime | None":
    """None on a missing or unparseable value -- never raises. A turn with
    no valid created_at cannot be ordered, so mine_tool_ngrams excludes it
    from sequence grouping rather than let the whole step abort on it."""
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if value is None:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def load_assistant_turns(store: Any) -> list[dict]:
    """Every assistant-tagged record's provenance/created_at/literal_surface.

    Reads via store.iter_record_columns -- a keyset-paginated column
    projection over the primary key (id > last_id ORDER BY id),
    fence-resumable across a concurrent WAL checkpoint. The projection is
    narrower than a full MemoryRecord row (no embedding/metadata overhead),
    but this function still returns one list with every matching turn's
    decrypted literal_surface resident at once -- it is not a streaming
    consumer.

    A column projection is NOT decrypted automatically (unlike
    store.iter_records, whose row->record path decrypts internally) -- this
    loader decrypts literal_surface/provenance_json itself via
    is_encrypted + store._decrypt_for_record, the exact pair of calls
    _store.py::_from_row makes for the same two fields, never a hand-rolled
    scheme. Reconciles the loaded count against SELECT COUNT(*) on the same
    WHERE clause before returning -- a truncated read must raise, never pass
    silently, mirroring proc_mine.load_cofired_events.
    """
    with store.db._conn_lock:
        count_row = store.db._conn.execute(
            f"SELECT COUNT(*) FROM records WHERE {_ASSISTANT_WHERE}"
        ).fetchone()
    expected = int(count_row[0]) if count_row and count_row[0] is not None else 0

    out: list[dict] = []
    for row in store.iter_record_columns(
        ["id", "provenance_json", "created_at", "literal_surface"],
        where=_ASSISTANT_WHERE,
    ):
        row_uuid = UUID(str(row["id"]))
        literal_raw = row.get("literal_surface") or ""
        if is_encrypted(literal_raw):
            literal_raw = store._decrypt_for_record(row_uuid, literal_raw)
        provenance_raw = row.get("provenance_json") or "[]"
        if is_encrypted(provenance_raw):
            provenance_raw = store._decrypt_for_record(row_uuid, provenance_raw)
        try:
            provenance_list = json.loads(provenance_raw) if provenance_raw else []
        except (TypeError, ValueError):
            provenance_list = []
        out.append(
            {
                "provenance": provenance_list,
                "created_at": _parse_ts_field(row.get("created_at")),
                "literal_surface": literal_raw,
            }
        )
    # A writer landing between the two reads above can only grow the row
    # count; only fewer rows than counted is a genuine truncated read.
    if len(out) < expected:
        raise RuntimeError(
            f"assistant turns load truncated: table has {expected} "
            f"matching rows, loaded {len(out)}"
        )
    return out


def mine_tool_ngrams(
    turns: list[dict],
    *,
    min_count: int = PAIR_COUNT_FLOOR,
    min_distinct_sessions: int = MIN_DISTINCT_SESSIONS,
) -> list[CofirePairCandidate]:
    """Intra-turn tool-name bigrams (N=2), gated on intra-turn count AND
    distinct-session spread (both inclusive) -- the structural twin of
    proc_mine.mine_cofired_pairs, bound to TOOLSEQ_MINE_SOURCE at candidate
    construction rather than mine_cofired_pairs' COFIRE_MINE_SOURCE.

    A turn captured twice under the same (session_id, created_at) key is
    deduped before its trailer is parsed. Turns are ordered by created_at
    within the whole population before pairing so first_ts/last_ts reflect
    genuine occurrence order.

    Inter-turn boundary pairs (last tool of one turn -> first tool of the
    NEXT turn in the SAME session, ordered by created_at) are reported as
    boundary_count on an already-eligible pair only -- mirroring
    mine_cofired_pairs' inter-event rule. They never feed count or
    session_count: gating on cross-turn pairs would widen the counted
    population past the intra-turn census the count-floor was calibrated
    against.
    """
    seen_turns: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for turn in turns:
        provenance = turn.get("provenance") or []
        session_id = "-"
        if provenance:
            try:
                session_id = provenance[0].get("session_id", "-")
            except (AttributeError, TypeError, KeyError):
                # provenance[0] parsed but is not a dict (e.g. a bare string,
                # int, or a top-level dict indexed like a list) -- unusable
                # for grouping, so this turn is excluded rather than aborting
                # the whole step, same disposition as an unparseable created_at.
                continue
        created_at = _normalize_created_at(turn.get("created_at"))
        if created_at is None:
            continue
        turn_key = (session_id, created_at.isoformat())
        if turn_key in seen_turns:
            continue
        seen_turns.add(turn_key)
        deduped.append(
            {
                "session_id": session_id,
                "created_at": created_at,
                "tools": parse_tool_sequence(turn.get("literal_surface", "")),
            }
        )
    deduped.sort(key=lambda t: t["created_at"])

    counts: "Counter[tuple[str, str]]" = Counter()
    sessions: dict[tuple[str, str], set[str]] = defaultdict(set)
    ts_lists: dict[tuple[str, str], list[datetime]] = defaultdict(list)

    for turn in deduped:
        tools = turn["tools"]
        for a, b in zip(tools, tools[1:]):
            pair = (a, b)
            counts[pair] += 1
            sessions[pair].add(turn["session_id"])
            ts_lists[pair].append(turn["created_at"])

    boundary_counts: "Counter[tuple[str, str]]" = Counter()
    by_session: dict[str, list[dict]] = defaultdict(list)
    for turn in deduped:
        by_session[turn["session_id"]].append(turn)
    for sess_turns in by_session.values():
        ordered = sorted(sess_turns, key=lambda t: t["created_at"])
        for prev_turn, next_turn in zip(ordered, ordered[1:]):
            prev_tools, next_tools = prev_turn["tools"], next_turn["tools"]
            if not prev_tools or not next_tools:
                continue
            boundary_counts[(prev_tools[-1], next_tools[0])] += 1

    candidates: list[CofirePairCandidate] = []
    for pair, count in counts.items():
        session_count = len(sessions[pair])
        if count >= min_count and session_count >= min_distinct_sessions:
            ts_list = ts_lists[pair]
            candidates.append(
                CofirePairCandidate(
                    pair=pair,
                    source=TOOLSEQ_MINE_SOURCE,
                    count=count,
                    session_count=session_count,
                    sessions=frozenset(sessions[pair]),
                    first_ts=min(ts_list),
                    last_ts=max(ts_list),
                    boundary_count=boundary_counts.get(pair, 0),
                )
            )

    candidates.sort(key=lambda c: (-c.count, c.pair))
    return candidates

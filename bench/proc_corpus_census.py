"""Read-only census: retrieval-support signal set for the proc-memory corpus.

Opens a filesystem COPY of a store (never the live store) and reports three
signals plus a retention diagnostic: (a) retrieval_reinforced ordered-pair
support under both intra-event and inter-event ordering, with per-pair
distinct-session distributions -- the gating signal; (b) retrieval_used
hit_ids co-occurrence, context only, never unioned into (a); (c) [tools: ...]
trailer bigram support from a read-only regex parse; and an events-table
row-count + oldest-ts sanity check. Every counting function here is pure --
it takes already-loaded events/records and returns counts. Only main() opens
a store, and it opens the copy only.

Before/after value metric: for each directed pair (A, B) whose FIRST
occurrence falls in the history half of the time-held-out split and recurs
at least once in the held-out half, a per-pair delta contrasts the rank<=K
hit rate of its REPEAT-session (held-out) observations against its
FIRST-occurrence (history) observation:
    per_pair_deltas(...)      -> dict[(A, B), float]  one signed delta per pair
    value_metric = mean(per_pair_deltas(...).values())
The scalar and the vector it is the mean of share the exact same units, so a
should-be-null contrast of the identical shape can later be bootstrapped
against it on equal footing.

Population, repetition, and eligibility come from retrieval_reinforced ONLY.
The rank readout is the ordered hit_ids LIST POSITION only -- never the
`used` boolean, never counted as support. A rank that cannot be resolved
(the paired record absent from the preceding hits, or no preceding recall
in that session) counts as a miss on the rank<=K indicator, never dropped
from the denominator.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import zlib
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
_REPO_PATH = str(Path(__file__).resolve().parent.parent)
if _REPO_PATH not in sys.path:
    sys.path.insert(0, _REPO_PATH)

from bench.recall_accuracy_real import open_eval_copy_store  # noqa: E402
from iai_mcp.events import query_events  # noqa: E402
from iai_mcp.lilli.profile.retrieval_tuning import RETRIEVAL_MIN_SAMPLES  # noqa: E402
from iai_mcp.store._store import _normalize_ts_for_compare  # noqa: E402

RETRIEVAL_REINFORCED_KIND = "retrieval_reinforced"
RETRIEVAL_USED_KIND = "retrieval_used"

_EVENT_LOAD_LIMIT = 10_000_000

TRAILER_RE = re.compile(r"\n\[tools: ([^\]]+)\]\Z")

# Fixed top-of-recall-page rank cut; committed pre-registration. [ASSUMED] --
# no cited canonical recall top-K constant exists in src for this population;
# same [ASSUMED] treatment as MIN_DISTINCT_SESSIONS. Never changes at runtime
# from the data.
RANK_K = 10

# T_cutoff = the median row-ts of the retained reinforced history -- a
# deterministic 50/50 time-held-out split that maximizes both history and
# held-out sample sizes without hand-picking a date. The RULE is committed
# here; derive_t_cutoff() derives the actual timestamp deterministically,
# never a runtime choice made to flatter a result.
T_CUTOFF_RULE = "median"

# Minimum count of distinct pairs (or, by the same rule, recurring bigrams)
# a signal needs to clear a PARK gate. Imported, not a duplicated literal --
# the same floor the sibling event-stream tuner uses, so this gate is never
# more permissive than the mechanism it sits beside.
PAIR_COUNT_FLOOR = RETRIEVAL_MIN_SAMPLES

# Minimum distinct-session spread that counts a pair as "recurring" rather
# than "happened twice". [ASSUMED] -- pre-committed but not independently
# verified against a precedent; the census reports the full session-count
# distribution so this floor stays re-derivable from real data.
MIN_DISTINCT_SESSIONS = 3


def _count_matching_events(store: Any, kind: str, since: datetime | None) -> int:
    where_parts = ["kind = ?"]
    params: list[Any] = [kind]
    if since is not None:
        where_parts.append("ts >= ?")
        params.append(_normalize_ts_for_compare(since).replace("T", " "))
    sql = "SELECT COUNT(*) FROM events WHERE " + " AND ".join(where_parts)
    with store.db._conn_lock:
        row = store.db._conn.execute(sql, params).fetchone()
    return int(row[0]) if row else 0


def load_reinforced_events(store: Any, since: datetime | None = None) -> list[dict]:
    """Load every retrieval_reinforced event, ascending by row ts.

    Reconciles the loaded count against the events-table row count for this
    kind before applying any filter -- a truncated read must raise, never
    pass silently. Rows with session_id == "-" (unset) are excluded.
    """
    expected = _count_matching_events(store, RETRIEVAL_REINFORCED_KIND, since)
    events = query_events(
        store, kind=RETRIEVAL_REINFORCED_KIND, since=since, limit=_EVENT_LOAD_LIMIT
    )
    if len(events) != expected:
        raise RuntimeError(
            f"retrieval_reinforced load truncated: table has {expected} "
            f"matching rows, loaded {len(events)} -- raise _EVENT_LOAD_LIMIT"
        )
    events = [e for e in events if e.get("session_id") != "-"]
    events.sort(key=lambda e: e["ts"])
    return events


def ordered_pairs_intra_event(events: list[dict]) -> "Counter[tuple[str, str]]":
    """Directed-pair support within a single retrieval_reinforced call.

    Pure: takes already-loaded events, no store access.
    """
    pairs: "Counter[tuple[str, str]]" = Counter()
    for event in events:
        ids = event.get("data", {}).get("reinforced_ids") or []
        for a, b in zip(ids, ids[1:]):
            pairs[(a, b)] += 1
    return pairs


def intra_event_pairs_with_session(
    events: list[dict],
) -> list[tuple[tuple[str, str], str]]:
    """Directed pairs within a single event, each tagged with its session_id.

    Pure: takes already-loaded events, no store access.
    """
    out: list[tuple[tuple[str, str], str]] = []
    for event in events:
        ids = event.get("data", {}).get("reinforced_ids") or []
        session_id = event.get("session_id", "-")
        for a, b in zip(ids, ids[1:]):
            out.append(((a, b), session_id))
    return out


def inter_event_pairs_with_session(
    events: list[dict],
) -> list[tuple[tuple[str, str], str]]:
    """Directed pairs across consecutive same-session events, by ascending ts.

    Groups events by session_id, sorts each group ascending by row ts, then
    pairs the LAST id of event N with the FIRST id of event N+1. Never pairs
    across different sessions. Pure: no store access.
    """
    out: list[tuple[tuple[str, str], str]] = []
    by_session: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        by_session[event.get("session_id", "-")].append(event)
    for session_id, sess_events in by_session.items():
        ordered = sorted(sess_events, key=lambda e: e["ts"])
        for prev_event, next_event in zip(ordered, ordered[1:]):
            prev_ids = prev_event.get("data", {}).get("reinforced_ids") or []
            next_ids = next_event.get("data", {}).get("reinforced_ids") or []
            if not prev_ids or not next_ids:
                continue
            out.append(((prev_ids[-1], next_ids[0]), session_id))
    return out


def ordered_pairs_inter_event(events: list[dict]) -> "Counter[tuple[str, str]]":
    """Directed-pair support across consecutive same-session events.

    Pure: takes already-loaded events, no store access. Reported alongside
    ordered_pairs_intra_event, never collapsed into it -- the two ordering
    sources measure distinct candidate pair populations.
    """
    return Counter(pair for pair, _session_id in inter_event_pairs_with_session(events))


def pair_session_spread(
    events: list[dict],
    ordering: "Callable[[list[dict]], list[tuple[tuple[str, str], str]]]",
) -> dict[tuple[str, str], set[str]]:
    """Map each directed pair (under the given ordering) to its distinct session_ids."""
    spread: dict[tuple[str, str], set[str]] = defaultdict(set)
    for pair, session_id in ordering(events):
        spread[pair].add(session_id)
    return dict(spread)


def distinct_session_distribution(
    spread: dict[tuple[str, str], set[str]],
) -> dict[str, int]:
    """Histogram of per-pair distinct-session counts, bucketed 1/2/3/4/5+."""
    buckets = {"1": 0, "2": 0, "3": 0, "4": 0, "5+": 0}
    for sessions in spread.values():
        n = len(sessions)
        key = str(n) if n < 5 else "5+"
        buckets[key] += 1
    return buckets


def pairs_meeting_session_floor(distribution: dict[str, int], *, min_sessions: int) -> int:
    """Count of pairs/bigrams whose distinct-session spread is >= min_sessions,
    from a distinct_session_distribution(...) bucket histogram (1/2/3/4/5+).
    """
    total = 0
    for key, count in distribution.items():
        n = 5 if key == "5+" else int(key)
        if n >= min_sessions:
            total += count
    return total


def load_used_events(store: Any, since: datetime | None = None) -> list[dict]:
    """Load every retrieval_used event, ascending by row ts.

    Exhausts the full table like load_reinforced_events. `data["query"]` is
    dropped at load, before any filter or return, so cue text can never
    reach a downstream output -- this is the only field this kind carries
    that retrieval_reinforced does not.
    """
    expected = _count_matching_events(store, RETRIEVAL_USED_KIND, since)
    events = query_events(
        store, kind=RETRIEVAL_USED_KIND, since=since, limit=_EVENT_LOAD_LIMIT
    )
    if len(events) != expected:
        raise RuntimeError(
            f"retrieval_used load truncated: table has {expected} "
            f"matching rows, loaded {len(events)} -- raise _EVENT_LOAD_LIMIT"
        )
    for event in events:
        data = event.get("data")
        if isinstance(data, dict):
            data.pop("query", None)
        assert "query" not in (event.get("data") or {}), "retrieval_used.query survived load"
    events = [e for e in events if e.get("session_id") != "-"]
    events.sort(key=lambda e: e["ts"])
    return events


def used_hitids_repetition(events: list[dict]) -> "Counter[tuple[str, str]]":
    """CONTEXT ONLY -- hit_ids co-occurrence per distinct session, never `used`.

    Pure: never reads event["data"]["used"]; a session-level "recall was
    non-empty" flag is not a per-record usefulness signal. Each pair is
    canonicalized by sorting (hit_ids order is a rank artifact, not a
    directed claim) and counted once per distinct session it appears in --
    directly comparable to (a)'s per-pair distinct-session counts, and must
    never be unioned into them.
    """
    pair_sessions: dict[tuple[str, str], set[str]] = defaultdict(set)
    for event in events:
        ids = event.get("data", {}).get("hit_ids") or []
        session_id = event.get("session_id", "-")
        for a, b in combinations(sorted(set(ids)), 2):
            pair_sessions[(a, b)].add(session_id)
    return Counter({pair: len(sessions) for pair, sessions in pair_sessions.items()})


def parse_tool_sequence(literal_surface: str) -> list[str]:
    """Read-only regex parse of the LAST [tools: ...] trailer. Never a write.

    \\Z-anchored so a mid-text `[tools: ...]`-shaped decoy cannot match.
    """
    match = TRAILER_RE.search(literal_surface)
    if not match:
        return []
    body = re.sub(r" \+\d+\Z", "", match.group(1))
    names = [n.strip() for n in body.split(", ") if n.strip()]
    return names[:80]  # defensive bound -- never trust the "8+N" cap blindly


def _normalize_created_at(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def load_assistant_records(store: Any) -> list[dict]:
    """Every assistant-tagged record's provenance/created_at/literal_surface.

    Reconciles the loaded count against SELECT COUNT(*) on the same WHERE
    clause before returning -- a truncated read must raise, never pass
    silently, mirroring load_reinforced_events/load_used_events.
    """
    where = "tags_json LIKE '%role:assistant%'"
    with store.db._conn_lock:
        count_row = store.db._conn.execute(
            f"SELECT COUNT(*) FROM records WHERE {where}"
        ).fetchone()
    expected = int(count_row[0]) if count_row and count_row[0] is not None else 0

    out: list[dict] = []
    for record in store.iter_records(where=where):
        out.append(
            {
                "provenance": record.provenance,
                "created_at": record.created_at,
                "literal_surface": record.literal_surface,
            }
        )
    if len(out) != expected:
        raise RuntimeError(
            f"assistant records load truncated: table has {expected} "
            f"matching rows, loaded {len(out)}"
        )
    return out


def tool_bigram_spread(records: list[dict]) -> dict[tuple[str, str], set[str]]:
    """Order-2 tool-name bigram support, per distinct session.

    Groups by (session_id, created_at) to dedupe a turn captured twice
    before parsing its trailer once via parse_tool_sequence.
    """
    seen_turns: set[tuple[str, str]] = set()
    spread: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        provenance = record.get("provenance") or []
        session_id = provenance[0].get("session_id", "-") if provenance else "-"
        created_at = _normalize_created_at(record.get("created_at"))
        turn_key = (session_id, created_at.isoformat())
        if turn_key in seen_turns:
            continue
        seen_turns.add(turn_key)
        tools = parse_tool_sequence(record.get("literal_surface", ""))
        for a, b in zip(tools, tools[1:]):
            spread[(a, b)].add(session_id)
    return dict(spread)


def events_sanity(store: Any) -> dict[str, Any]:
    """Row count + oldest ts for the events table -- the tell for a past prune.

    Diagnostic only, never a blocker: a suspiciously-recent oldest event
    relative to the store's own age (oldest record created_at) flags a
    possible past manual --prune-telemetry-before migration run. Nothing
    auto-prunes the events table in normal operation. Uses COUNT(*) + an
    ORDER BY/LIMIT 1 scan rather than MIN() -- the lilli engine's SQL
    dialect does not support MIN() in the SELECT list.
    """
    with store.db._conn_lock:
        count_row = store.db._conn.execute("SELECT COUNT(*) FROM events").fetchone()
        oldest_event_row = store.db._conn.execute(
            "SELECT ts FROM events ORDER BY ts ASC LIMIT 1"
        ).fetchone()
        oldest_record_row = store.db._conn.execute(
            "SELECT created_at FROM records ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
    return {
        "events_count": int(count_row[0]) if count_row and count_row[0] is not None else 0,
        "oldest_event_ts": oldest_event_row[0] if oldest_event_row else None,
        "oldest_record_created_at": oldest_record_row[0] if oldest_record_row else None,
    }


def derive_t_cutoff(reinforced_events: list[dict]) -> datetime:
    """Median row-ts of the retained reinforced history. Deterministic:
    always the same value for the same event set, never a hand-picked date.
    """
    if not reinforced_events:
        raise ValueError("derive_t_cutoff requires at least one reinforced event")
    ts_sorted = sorted(e["ts"] for e in reinforced_events)
    n = len(ts_sorted)
    mid = n // 2
    if n % 2 == 1:
        return ts_sorted[mid]
    lo, hi = ts_sorted[mid - 1], ts_sorted[mid]
    return lo + (hi - lo) / 2


def repetition_count(
    events: list[dict], *, t_cutoff: datetime
) -> dict[tuple[str, str], int]:
    """Distinct-session count per (A, B) pair, intra-event ordering,
    restricted to history (ts < t_cutoff) ONLY -- the establishment window.
    """
    history = [e for e in events if e["ts"] < t_cutoff]
    spread = pair_session_spread(history, intra_event_pairs_with_session)
    return {pair: len(sessions) for pair, sessions in spread.items()}


def rank_at_recall(
    reinforced_events: list[dict],
    used_events: list[dict],
    pair: tuple[str, str],
    session: str,
) -> int | None:
    """hit_ids list-position of B in the retrieval_used event immediately
    PRECEDING session's retrieval_reinforced co-firing of pair -- the
    nearest-preceding-same-session bisect join. None if the pair never
    co-fired in session, no preceding retrieval_used event exists, or B is
    absent from that event's hit_ids (a MISS, left to the caller to count).
    """
    a, b = pair
    reinforce_ts: datetime | None = None
    for event in reinforced_events:
        if event.get("session_id") != session:
            continue
        ids = event.get("data", {}).get("reinforced_ids") or []
        if any(x == a and y == b for x, y in zip(ids, ids[1:])):
            ts = event["ts"]
            if reinforce_ts is None or ts > reinforce_ts:
                reinforce_ts = ts
    if reinforce_ts is None:
        return None

    same_session_used = sorted(
        (
            (event["ts"], event.get("data", {}).get("hit_ids") or [])
            for event in used_events
            if event.get("session_id") == session
        ),
        key=lambda row: row[0],
    )
    ts_list = [ts for ts, _hit_ids in same_session_used]
    idx = bisect_right(ts_list, reinforce_ts) - 1
    if idx < 0:
        return None
    hit_ids = same_session_used[idx][1]
    if b not in hit_ids:
        return None
    return hit_ids.index(b)


def _rank_indicator(rank: int | None, k: int) -> float:
    """1.0 iff rank is a resolved list-position inside the top-K, else 0.0
    -- a None rank (miss) is never dropped, only scored 0.0.
    """
    return 1.0 if rank is not None and rank < k else 0.0


def _pair_sessions_ordered(
    events: list[dict],
) -> dict[tuple[str, str], list[tuple[datetime, str]]]:
    """pair -> (ts, session_id) list, intra-event ordering, ascending by ts."""
    out: dict[tuple[str, str], list[tuple[datetime, str]]] = defaultdict(list)
    for event in events:
        ids = event.get("data", {}).get("reinforced_ids") or []
        session_id = event.get("session_id", "-")
        ts = event["ts"]
        for a, b in zip(ids, ids[1:]):
            out[(a, b)].append((ts, session_id))
    for pair in out:
        out[pair].sort(key=lambda row: row[0])
    return dict(out)


def _distinct_sessions_ordered(
    ts_sessions: list[tuple[datetime, str]],
) -> list[tuple[datetime, str]]:
    """Dedupe (ts, session_id) rows to one per session_id, keeping the
    EARLIEST ts each session_id first appears at (input must be ts-sorted).
    """
    seen: set[str] = set()
    out: list[tuple[datetime, str]] = []
    for ts, session_id in ts_sessions:
        if session_id in seen:
            continue
        seen.add(session_id)
        out.append((ts, session_id))
    return out


def _mean_indicator(
    reinforced_events: list[dict],
    used_events: list[dict],
    pair: tuple[str, str],
    sessions: list[str],
    k: int,
) -> float:
    indicators = [
        _rank_indicator(rank_at_recall(reinforced_events, used_events, pair, sid), k)
        for sid in sessions
    ]
    return sum(indicators) / len(indicators)


def per_pair_deltas(
    reinforced_events: list[dict],
    used_events: list[dict],
    *,
    t_cutoff: datetime,
    k: int,
) -> dict[tuple[str, str], float]:
    """One signed rank<=k-indicator delta per established (A, B) pair:
    mean(indicator over REPEAT-session observations) minus the indicator of
    the FIRST-occurrence observation. A pair is established only if its
    FIRST occurrence (earliest session, intra-event ordering) falls in the
    HISTORY half (ts < t_cutoff) AND it recurs in at least one session in
    the HELD-OUT half (ts >= t_cutoff) -- the frozen time-held-out split:
    history establishes the pair, held-out measures the outcome, so no
    REPEAT observation is drawn from the same window used to establish it.
    A None rank (B absent from the preceding hits) scores 0.0 on the
    indicator -- it stays in the mean, never dropped.
    """
    ordered = _pair_sessions_ordered(reinforced_events)
    deltas: dict[tuple[str, str], float] = {}
    for pair, ts_sessions in ordered.items():
        distinct = _distinct_sessions_ordered(ts_sessions)
        if len(distinct) < 2:
            continue
        first_ts, first_session = distinct[0]
        if first_ts >= t_cutoff:
            continue
        repeat_sessions = [
            session_id for ts, session_id in distinct[1:] if ts >= t_cutoff
        ]
        if not repeat_sessions:
            continue
        first_indicator = _mean_indicator(
            reinforced_events, used_events, pair, [first_session], k
        )
        repeat_mean = _mean_indicator(
            reinforced_events, used_events, pair, repeat_sessions, k
        )
        deltas[pair] = repeat_mean - first_indicator
    return deltas


def value_metric(
    reinforced_events: list[dict],
    used_events: list[dict],
    *,
    t_cutoff: datetime,
    k: int,
) -> float:
    """The mean of per_pair_deltas(...) -- the scalar IS the mean of that
    exact vector, provably (see the identity test).
    """
    deltas = per_pair_deltas(reinforced_events, used_events, t_cutoff=t_cutoff, k=k)
    values = list(deltas.values())
    return statistics.fmean(values) if values else 0.0


def split_odd_even(session_ids: list[str]) -> tuple[list[str], list[str]]:
    """Deterministic odd/even parity split of session_ids by a stable hash
    (crc32, not the randomized-per-process built-in `hash()`) -- two
    disjoint slices of the SAME input, never two independent RNG draws.
    """
    odd: list[str] = []
    even: list[str] = []
    for session_id in session_ids:
        bucket = odd if (zlib.crc32(session_id.encode("utf-8")) % 2) else even
        bucket.append(session_id)
    return odd, even


def null_per_pair_deltas(
    reinforced_events: list[dict],
    used_events: list[dict],
    *,
    t_cutoff: datetime,
    k: int,
) -> dict[tuple[str, str], float]:
    """Same per-pair-delta SHAPE as per_pair_deltas, but the contrast is an
    odd/even session-parity split of the SAME held-out window (ts >=
    t_cutoff) -- a should-be-null quantity, never repeat-vs-first. A pair is
    included only if it has at least one occurrence in EACH half.
    """
    held_out = [e for e in reinforced_events if e["ts"] >= t_cutoff]
    ordered = _pair_sessions_ordered(held_out)
    session_ids = sorted({e.get("session_id", "-") for e in held_out})
    odd_sessions, even_sessions = split_odd_even(session_ids)
    odd_set, even_set = set(odd_sessions), set(even_sessions)

    deltas: dict[tuple[str, str], float] = {}
    for pair, ts_sessions in ordered.items():
        distinct_sessions = [sid for _ts, sid in _distinct_sessions_ordered(ts_sessions)]
        odd_pair_sessions = [sid for sid in distinct_sessions if sid in odd_set]
        even_pair_sessions = [sid for sid in distinct_sessions if sid in even_set]
        if not odd_pair_sessions or not even_pair_sessions:
            continue
        odd_mean = _mean_indicator(
            reinforced_events, used_events, pair, odd_pair_sessions, k
        )
        even_mean = _mean_indicator(
            reinforced_events, used_events, pair, even_pair_sessions, k
        )
        deltas[pair] = odd_mean - even_mean
    return deltas


def _as_values(deltas: Any) -> list[float]:
    if isinstance(deltas, dict):
        return list(deltas.values())
    return list(deltas)


def bootstrap_lower_ci(
    deltas: Any, *, iters: int = 2000, seed: int = 0, alpha: float = 0.05
) -> float:
    """Lower one-sided (1 - alpha) bootstrap bound on the MEAN of a per-pair
    signed-delta vector -- resamples PAIRS. Consumes exactly the
    dict[tuple, float].values() shape per_pair_deltas/null_per_pair_deltas
    produce. Deterministic under a fixed seed.
    """
    values = _as_values(deltas)
    if not values:
        return 0.0
    rng = random.Random(seed)
    n = len(values)
    boot_means: list[float] = []
    for _ in range(iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    idx = min(max(int(alpha * iters), 0), iters - 1)
    return boot_means[idx]


def two_sided_aa_floor(
    deltas: Any, *, iters: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """(low, high) two-sided floor -- proof the floor MACHINERY behaves, not
    the real corpus floor (that comes from a live run against real data).
    The SAME bootstrap_lower_ci is called twice, once on the signed deltas
    and once on the negated deltas, negated back. No second statistics
    function.
    """
    values = _as_values(deltas)
    ci_low = bootstrap_lower_ci(values, iters=iters, seed=seed)
    ci_high = -bootstrap_lower_ci([-v for v in values], iters=iters, seed=seed)
    return ci_low, ci_high


def park_verdict(
    *,
    gating_pairs_over_floor: int,
    value_metric: float,
    established_pairs: int,
    real_aa_floor: tuple[float, float],
    signalb_bigrams_over_floor: int,
) -> dict[str, Any]:
    """The pre-committed go/PARK rule.

    Proceeds only if (a)'s gating pair count clears PAIR_COUNT_FLOOR AND
    value_metric clears real_aa_floor's upper null bound -- strictly
    greater, so a value equal to the bound does not clear the null band.
    Either failing alone PARKs the whole milestone. Signal B's recurring-
    bigram count is judged against the same PAIR_COUNT_FLOOR but never
    affects the milestone verdict -- a thin Signal B parks Signal B alone.

    established_pairs distinguishes a PARK from zero measured pairs (no
    data to compute value_metric over) from a PARK on a genuinely measured,
    flat/negative value_metric -- both can print value_metric == 0.0, and
    only established_pairs tells them apart. underpowered is reported, not
    gated: it never substitutes a looser threshold for zero data.
    """
    gating_clears_floor = gating_pairs_over_floor >= PAIR_COUNT_FLOOR
    value_metric_clears_floor = value_metric > real_aa_floor[1]
    milestone_verdict = (
        "proceed" if (gating_clears_floor and value_metric_clears_floor) else "PARK"
    )
    signal_b_verdict = "proceed" if signalb_bigrams_over_floor >= PAIR_COUNT_FLOOR else "PARK"
    return {
        "milestone_verdict": milestone_verdict,
        "gating_clears_floor": gating_clears_floor,
        "value_metric_clears_floor": value_metric_clears_floor,
        "signal_b_verdict": signal_b_verdict,
        "established_pairs": established_pairs,
        "underpowered": established_pairs == 0,
        "pair_count_floor": PAIR_COUNT_FLOOR,
        "min_distinct_sessions": MIN_DISTINCT_SESSIONS,
    }


def run_census(store: Any) -> dict[str, Any]:
    """Single assembly: loads reinforced+used events once, derives
    t_cutoff, and computes the (a)/(b)/(c) census numbers plus value_metric
    and real_corpus_aa_floor -- every verdict field from one event load.
    """
    reinforced_events = load_reinforced_events(store)
    used_events = load_used_events(store)

    intra_pairs = ordered_pairs_intra_event(reinforced_events)
    inter_pairs = ordered_pairs_inter_event(reinforced_events)
    intra_spread = pair_session_spread(reinforced_events, intra_event_pairs_with_session)
    inter_spread = pair_session_spread(reinforced_events, inter_event_pairs_with_session)
    intra_distribution = distinct_session_distribution(intra_spread)
    inter_distribution = distinct_session_distribution(inter_spread)

    used_pairs = used_hitids_repetition(used_events)

    assistant_records = load_assistant_records(store)
    bigram_spread = tool_bigram_spread(assistant_records)
    bigram_distribution = distinct_session_distribution(bigram_spread)

    sanity = events_sanity(store)

    # Established-pair population is INTRA-EVENT ordering only, matching
    # per_pair_deltas/repetition_count's own population -- the gating count
    # park_verdict consumes must count the same pairs those functions do.
    gating_pairs_over_floor = pairs_meeting_session_floor(
        intra_distribution, min_sessions=MIN_DISTINCT_SESSIONS
    )
    signalb_bigrams_over_floor = pairs_meeting_session_floor(
        bigram_distribution, min_sessions=MIN_DISTINCT_SESSIONS
    )

    if reinforced_events:
        t_cutoff = derive_t_cutoff(reinforced_events)
        vm = value_metric(reinforced_events, used_events, t_cutoff=t_cutoff, k=RANK_K)
        established_pairs = len(
            per_pair_deltas(reinforced_events, used_events, t_cutoff=t_cutoff, k=RANK_K)
        )
        aa_floor = two_sided_aa_floor(
            null_per_pair_deltas(reinforced_events, used_events, t_cutoff=t_cutoff, k=RANK_K)
        )
    else:
        t_cutoff = None
        vm = 0.0
        established_pairs = 0
        aa_floor = (0.0, 0.0)

    return {
        "a_intra_event": {
            "distinct_pairs": len(intra_pairs),
            "session_distribution": intra_distribution,
        },
        "a_inter_event": {
            "distinct_pairs": len(inter_pairs),
            "session_distribution": inter_distribution,
        },
        "b_context_only_not_a_gate": {
            "distinct_pairs": len(used_pairs),
        },
        "c_tool_bigrams": {
            "distinct_bigrams": len(bigram_spread),
            "session_distribution": bigram_distribution,
        },
        "events_sanity": sanity,
        "value_metric": vm,
        "established_pairs": established_pairs,
        "real_corpus_aa_floor": aa_floor,
        "gating_pairs_over_floor": gating_pairs_over_floor,
        "signalb_bigrams_over_floor": signalb_bigrams_over_floor,
        "rank_k": RANK_K,
        "t_cutoff": t_cutoff.isoformat() if t_cutoff is not None else None,
        "t_cutoff_rule": T_CUTOFF_RULE,
    }


def _print_top_pair(pairs: "Counter[tuple[str, str]]", driver: str) -> None:
    payload: dict[str, Any] = {"driver": driver}
    if pairs:
        top_pair, top_count = pairs.most_common(1)[0]
        payload["top_pair"] = list(top_pair)
        payload["count"] = top_count
    else:
        payload["top_pair"] = None
        payload["count"] = 0
    print(json.dumps(payload))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only census of the proc-memory corpus retrieval-support signal set."
    )
    parser.add_argument("--driver", choices=["lilli", "stdlib"], default="lilli")
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()

    with open_eval_copy_store(driver=args.driver) as store:
        report = run_census(store)
        # Reload once more: run_census's own dict is ids/counts/metrics-only,
        # never a raw Counter (not JSON-shaped) -- the top-pair line below
        # needs the Counter directly.
        reinforced_events = load_reinforced_events(store)

    report["driver"] = args.driver
    report["park_verdict"] = park_verdict(
        gating_pairs_over_floor=report["gating_pairs_over_floor"],
        value_metric=report["value_metric"],
        established_pairs=report["established_pairs"],
        real_aa_floor=report["real_corpus_aa_floor"],
        signalb_bigrams_over_floor=report["signalb_bigrams_over_floor"],
    )
    print(json.dumps(report, default=str))

    intra_pairs = ordered_pairs_intra_event(reinforced_events)
    _print_top_pair(intra_pairs, driver=args.driver)


if __name__ == "__main__":
    main()

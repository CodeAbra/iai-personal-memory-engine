from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from iai_mcp.events import query_events, write_event
from iai_mcp.store import MemoryStore


ENTROPY_LOW: float = 0.4
ENTROPY_MID: float = 0.7
ENTROPY_HIGH: float = 0.9
COOLDOWN_TURNS: int = 3

# An unresolved question older than this stops counting as "pending" on the
# recall projection path. Without it the pending set only ever grows: every
# curiosity_question that is never explicitly resolved lingers forever, so the
# surfaced signal (and the nightly report that reads it) drifts upward without
# bound. The live ``curiosity_pending`` MCP tool still passes ``None`` for a
# full, unexpired read.
PENDING_MAX_AGE_DAYS: float = 30.0

# Kind marker for the drain watermark. The deferred-curiosity consumer records
# the timestamp of the newest ``deferred_curiosity_input`` it has processed so
# the next sleep cycle only fires on inputs that arrived since — no event is
# ever double-fired, and old inputs are not re-scanned every cycle.
_DRAIN_CURSOR_KIND: str = "curiosity_drain_cursor"
_DRAIN_BATCH: int = 200

# How many recent questions the resolution pass considers per drain. Bounded so
# a large historical backlog cannot make the pass scan without limit.
_RESOLVE_SCAN_LIMIT: int = 200

_log = logging.getLogger(__name__)


def _cue_key(cue: str) -> str:
    """Normalized match key for a retrieval cue.

    Curiosity fires on an ambiguous cue and resolves when the *same* cue later
    comes back confident, so the two events must agree on what "same" means.
    Casefold + whitespace collapse absorbs the incidental variation between two
    phrasings of one question without pulling in a similarity model.
    """
    return " ".join(str(cue).split()).casefold()[:200]

# Curiosity signals are advisory: the recall dispatch does not need
# per-recall freshness. A cached projection older than this threshold
# triggers a background single-flight refresh instead of blocking the
# caller on the two live query_events scans.
_CURIOSITY_CACHE_REFRESH_SEC: float = 30.0


@dataclass
class CuriosityQuestion:

    id: UUID
    text: str
    triggered_by_record_ids: list[UUID] = field(default_factory=list)
    entropy: float = 0.0
    tier: str = "question"
    resolved: bool = False


def compute_entropy(scores: list[float]) -> float:
    """Normalized Shannon entropy of a score distribution, in ``[0, 1]``.

    The raw Shannon sum is in *bits* and ranges over ``[0, log2(k)]`` for ``k``
    non-zero outcomes, so it inflates with the number of hits. The curiosity
    tiers (``ENTROPY_LOW/MID/HIGH`` = 0.4/0.7/0.9) are calibrated against a
    unit scale, so the raw sum is divided by ``log2(k)`` to land in ``[0, 1]``
    regardless of hit count. For ``k <= 1`` (a degenerate, single-outcome
    distribution) entropy is exactly 0. Two equal outcomes still yield 1.0.
    """
    if not scores:
        return 0.0
    positive = [max(0.0, float(s)) for s in scores]
    total = sum(positive)
    if total <= 0:
        return 0.0
    probs = [p / total for p in positive if p > 0]
    if len(probs) <= 1:
        return 0.0
    h = 0.0
    for p in probs:
        h -= p * math.log2(p)
    return h / math.log2(len(probs))


def _last_curiosity_turn(store: MemoryStore, session_id: str) -> int | None:
    # Session-scoped at the SQL layer: the previous implementation pulled the
    # 20 most-recent *global* curiosity_question events and filtered in Python,
    # so on a multi-session engine a given session's last question routinely
    # fell outside the window and the cooldown silently stopped firing. Filter
    # by session in the query and take the single newest row.
    events = query_events(
        store, kind="curiosity_question", session_id=session_id, limit=1,
    )
    for e in events:
        try:
            return int(e["data"].get("turn", 0))
        except (TypeError, ValueError):
            return None
    return None


def fire_curiosity(
    store: MemoryStore,
    hits: list,
    cue: str,
    entropy: float,
    session_id: str,
    turn: int,
) -> CuriosityQuestion | None:
    if entropy < ENTROPY_LOW:
        return None

    if entropy < ENTROPY_MID:
        write_event(
            store,
            kind="curiosity_silent_log",
            data={
                "cue": cue[:200],
                "entropy": float(entropy),
                "source_ids": [str(h.record_id) for h in hits[:3]],
            },
            severity="info",
            session_id=session_id,
        )
        return None

    last = _last_curiosity_turn(store, session_id)
    if last is not None and (turn - last) < COOLDOWN_TURNS:
        return None

    q_id = uuid4()
    if entropy < ENTROPY_HIGH:
        tier = "inline"
        text = f"I'm not fully sure -- did you mean {cue!r}?"
    else:
        tier = "question"
        text = f"Could you clarify: {cue!r}?"

    trigger_ids: list[UUID] = [h.record_id for h in hits[:5]]
    question = CuriosityQuestion(
        id=q_id,
        text=text,
        triggered_by_record_ids=trigger_ids,
        entropy=float(entropy),
        tier=tier,
    )

    bridge_pairs = [(tid, q_id) for tid in trigger_ids]
    if bridge_pairs:
        try:
            store.boost_edges(
                bridge_pairs,
                edge_type="curiosity_bridge",
                delta=float(entropy),
            )
        except (OSError, RuntimeError, ValueError):
            pass

    write_event(
        store,
        kind="curiosity_question",
        data={
            "question_id": str(q_id),
            "text": text,
            # The normalized cue is recorded alongside the rendered text so a
            # later confident recall on the same cue can resolve this question
            # without re-parsing the question prose.
            "cue_key": _cue_key(cue),
            "tier": tier,
            "entropy": float(entropy),
            "turn": int(turn),
            "triggered_by": [str(t) for t in trigger_ids],
        },
        severity="info",
        session_id=session_id,
        source_ids=trigger_ids,
    )
    return question


def pending_questions(
    store: MemoryStore,
    session_id: str | None = None,
    *,
    max_age_days: float | None = None,
) -> list[CuriosityQuestion]:
    events = query_events(store, kind="curiosity_question", limit=200)
    resolved_events = query_events(store, kind="curiosity_resolved", limit=500)
    resolved_ids = {
        r["data"].get("question_id")
        for r in resolved_events
        if r["data"].get("question_id")
    }
    cutoff: datetime | None = None
    if max_age_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=float(max_age_days))
    out: list[CuriosityQuestion] = []
    for e in events:
        if session_id is not None and e.get("session_id") != session_id:
            continue
        if cutoff is not None:
            ts = e.get("ts")
            if isinstance(ts, datetime) and ts < cutoff:
                continue
        data = e["data"]
        qid_raw = data.get("question_id")
        if not qid_raw:
            continue
        if qid_raw in resolved_ids:
            continue
        try:
            qid = UUID(qid_raw)
        except (TypeError, ValueError):
            continue
        triggered: list[UUID] = []
        for t in data.get("triggered_by", []):
            try:
                triggered.append(UUID(t))
            except (TypeError, ValueError):
                continue
        out.append(
            CuriosityQuestion(
                id=qid,
                text=data.get("text", ""),
                triggered_by_record_ids=triggered,
                entropy=float(data.get("entropy", 0.0)),
                tier=data.get("tier", "question"),
                resolved=False,
            )
        )
    return out


def get_pending_questions(store: MemoryStore, limit: int = 2) -> list[dict]:
    # Recall-projection read: apply the pending-age bound so a never-resolved
    # backlog cannot inflate the surfaced signal indefinitely.
    qs = pending_questions(store, max_age_days=PENDING_MAX_AGE_DAYS)
    return [
        {"text": q.text, "entropy": q.entropy, "tier": q.tier}
        for q in qs[:limit]
    ]


class _CuriosityCache:
    """Refresh-ahead, single-flight, in-process cache of the pending-questions
    projection for one ``MemoryStore``.

    Curiosity signals are advisory -- the recall dispatch does not need a
    per-recall-fresh read. This cache mirrors the corpus-count-cache
    discipline (lock guards only the dict/state access, never the SQL): a
    warm read returns the last computed projection immediately; a stale or
    cold read also returns immediately (the current value, or ``[]`` if
    never computed) and, if no refresh is already in flight, schedules
    exactly one background refresh via a daemon thread. The refresh reuses
    ``get_pending_questions`` verbatim, so the underlying question-set
    correctness (resolved-filtering, session scoping) is unchanged -- only
    WHEN it runs moves off the caller's thread.

    No-raise contract: a refresh failure is caught and logged; the cache
    keeps (or reverts to) its last-known value rather than propagating the
    exception into the caller.
    """

    def __init__(self, refresh_after_sec: float = _CURIOSITY_CACHE_REFRESH_SEC) -> None:
        self._lock = threading.Lock()
        self._value: list[dict] = []
        self._computed_at: float | None = None
        self._refresh_in_flight: bool = False
        self._refresh_after_sec = float(refresh_after_sec)

    def get(self, store: MemoryStore, limit: int = 2) -> list[dict]:
        """Return the cached projection, triggering a background refresh if
        stale/cold. Never blocks on ``query_events``; never raises.
        """
        with self._lock:
            value = list(self._value)
            is_stale = (
                self._computed_at is None
                or (time.monotonic() - self._computed_at) >= self._refresh_after_sec
            )
            should_start_refresh = is_stale and not self._refresh_in_flight
            if should_start_refresh:
                self._refresh_in_flight = True

        if should_start_refresh:
            self._start_background_refresh(store)

        return value[:limit]

    def _start_background_refresh(self, store: MemoryStore) -> None:
        thread = threading.Thread(
            target=self._refresh_once,
            args=(store,),
            name="iai-mcp-curiosity-refresh",
            daemon=True,
        )
        thread.start()

    def _refresh_once(self, store: MemoryStore) -> None:
        try:
            fresh = get_pending_questions(store, limit=200)
        except Exception as exc:  # noqa: BLE001 -- refresh must never raise into a bg thread
            _log.debug("curiosity_cache_refresh_failed: %s", exc)
            with self._lock:
                self._refresh_in_flight = False
            return
        with self._lock:
            self._value = fresh
            self._computed_at = time.monotonic()
            self._refresh_in_flight = False


# Module-level cache keyed by store identity so multiple stores in one
# process (e.g. tests, or a future multi-store host) do not share state.
_caches: dict[int, _CuriosityCache] = {}
_caches_lock = threading.Lock()


def _cache_for(store: MemoryStore) -> _CuriosityCache:
    key = id(store)
    with _caches_lock:
        cache = _caches.get(key)
        if cache is None:
            cache = _CuriosityCache()
            _caches[key] = cache
        return cache


def get_pending_questions_cached(store: MemoryStore, limit: int = 2) -> list[dict]:
    """Recall-path variant of ``get_pending_questions``.

    Serves the projection from a refresh-ahead, single-flight, in-process
    cache instead of running the two synchronous ``query_events`` scans on
    the caller's thread. A cold cache returns ``[]`` immediately and
    schedules the first background refresh. Advisory only: a slightly stale
    signal is acceptable, and any refresh failure degrades to the
    last-known (or empty) value -- never raises, never blocks.

    Non-recall callers that want a live read (e.g. the ``curiosity_pending``
    MCP tool) should keep calling ``get_pending_questions`` directly.
    """
    return _cache_for(store).get(store, limit=limit)


class _DrainHit:
    """Minimal ``MemoryHit`` stand-in for the deferred-curiosity consumer.

    ``fire_curiosity`` only ever reads ``.record_id`` off its hits, so the
    drain reconstructs that single field from the ``hit_ids`` recorded on the
    deferred event rather than re-fetching whole records.
    """

    __slots__ = ("record_id",)

    def __init__(self, record_id: UUID) -> None:
        self.record_id = record_id


def _load_drain_cursor(store: MemoryStore) -> datetime | None:
    events = query_events(store, kind=_DRAIN_CURSOR_KIND, limit=1)
    if not events:
        return None
    raw = events[0]["data"].get("through_ts")
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def prune_consumed_curiosity_events(store: MemoryStore) -> dict[str, int]:
    """Drop deferred inputs the drain has already consumed, and stale cursors.

    The drain consumes ``deferred_curiosity_input`` *logically* by advancing a
    watermark, which leaves the rows themselves in the events table forever —
    one per non-verbatim recall, plus one ``curiosity_drain_cursor`` per sleep
    cycle. Both are pure residue once the watermark has passed them: deleting
    inputs at or before the cursor cannot un-consume anything (the cursor is
    what gates re-firing), and only the newest cursor is ever read.

    Returns ``{"inputs_pruned": n, "cursors_pruned": m}``. Never raises.
    """
    from iai_mcp.store import EVENTS_TABLE

    out = {"inputs_pruned": 0, "cursors_pruned": 0}
    cursor = _load_drain_cursor(store)
    if cursor is None:
        # Nothing has been consumed yet; deleting inputs now would drop
        # questions that were never fired.
        return out

    from iai_mcp.store._store import _normalize_ts_for_compare

    boundary = _normalize_ts_for_compare(cursor).replace("T", " ")
    table = store.db.open_table(EVENTS_TABLE)
    try:
        with store.db._conn_lock:
            rows = store.db._conn.execute(
                f"SELECT COUNT(*) AS n FROM {EVENTS_TABLE} "
                "WHERE kind = 'deferred_curiosity_input' AND ts <= ?",
                [boundary],
            ).fetchone()
        consumed = int(rows["n"]) if rows else 0
        if consumed:
            table.delete(
                "kind = 'deferred_curiosity_input' "
                f"AND ts <= '{boundary}'",
            )
            out["inputs_pruned"] = consumed
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        _log.debug("curiosity_input_prune_failed: %s", exc)

    # Keep only the newest cursor: it is the only one _load_drain_cursor reads.
    # Keyed on the newest cursor row's own write time, not on the watermark
    # above -- a cursor is written after the inputs it covers, so its ts always
    # sorts later than its own through_ts.
    try:
        with store.db._conn_lock:
            newest = store.db._conn.execute(
                f"SELECT MAX(ts) AS newest FROM {EVENTS_TABLE} "
                f"WHERE kind = '{_DRAIN_CURSOR_KIND}'",
            ).fetchone()
            raw_newest = newest["newest"] if newest else None
            if raw_newest is None:
                return out
            # MAX(ts) comes back as a datetime or a TEXT string depending on the
            # connection's converters; normalize to the stored space-form shape
            # either way so the comparison below is exact.
            newest_cursor_ts = _normalize_ts_for_compare(raw_newest).replace(
                "T", " ",
            )
            stale = store.db._conn.execute(
                f"SELECT COUNT(*) AS n FROM {EVENTS_TABLE} "
                f"WHERE kind = '{_DRAIN_CURSOR_KIND}' AND ts < ?",
                [newest_cursor_ts],
            ).fetchone()
        n_stale = int(stale["n"]) if stale else 0
        if n_stale:
            table.delete(
                f"kind = '{_DRAIN_CURSOR_KIND}' "
                f"AND ts < '{newest_cursor_ts}'",
            )
            out["cursors_pruned"] = n_stale
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        _log.debug("curiosity_cursor_prune_failed: %s", exc)

    return out


def resolve_answered_questions(
    store: MemoryStore,
    cue_key: str,
    session_id: str,
) -> int:
    """Close pending questions whose cue has since come back confident.

    A curiosity question is a recorded ambiguity: the engine could not tell
    which memory a cue meant. When a later recall on that same cue returns a
    confident (low-entropy) distribution, the ambiguity is gone, so the
    question is answered in substance whether or not the user ever replied to
    it. This writes the ``curiosity_resolved`` event that ``pending_questions``
    already filters on -- previously nothing in the engine ever did, so
    questions could only age out.

    Returns the number of questions resolved. Never raises.
    """
    if not cue_key:
        return 0
    try:
        questions = query_events(
            store,
            kind="curiosity_question",
            session_id=session_id,
            limit=_RESOLVE_SCAN_LIMIT,
        )
        if not questions:
            return 0
        resolved_events = query_events(
            store, kind="curiosity_resolved", limit=_RESOLVE_SCAN_LIMIT * 2,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        _log.debug("curiosity_resolve_scan_failed: %s", exc)
        return 0

    already = {
        r["data"].get("question_id")
        for r in resolved_events
        if r["data"].get("question_id")
    }
    resolved = 0
    for e in questions:
        data = e["data"]
        qid = data.get("question_id")
        if not qid or qid in already:
            continue
        # Questions written before cue_key existed carry no key and can never
        # be matched; they age out via the pending-age bound instead.
        if data.get("cue_key") != cue_key:
            continue
        try:
            write_event(
                store,
                kind="curiosity_resolved",
                data={
                    "question_id": qid,
                    "cue_key": cue_key,
                    "reason": "confident_recall",
                },
                severity="info",
                session_id=session_id,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            _log.debug("curiosity_resolved_write_failed: %s", exc)
            continue
        already.add(qid)
        resolved += 1
    return resolved


def drain_deferred_curiosity(
    store: MemoryStore,
    *,
    batch: int = _DRAIN_BATCH,
) -> dict[str, int]:
    """Turn buffered ``deferred_curiosity_input`` events into questions.

    The recall hot path only records a cheap ``deferred_curiosity_input`` event
    (hit ids, cue, precomputed normalized entropy, turn) so it never blocks on
    curiosity work. This consumer -- invoked once per sleep cycle -- reads the
    inputs that have arrived since the last watermark, and replays each through
    ``fire_curiosity``, which applies the entropy tiers and the per-session
    cooldown and, above threshold, writes the ``curiosity_question`` event plus
    the ``curiosity_bridge`` edges. The watermark advances to the newest input
    seen so no event is ever fired twice.

    Inputs below the firing floor are not discarded: an unambiguous recall on a
    cue that previously fired a question resolves that question, which is the
    engine's only path to closing one.

    Returns ``{"drained": n, "fired": m, "resolved": k}``. Never raises: a
    malformed event is skipped, and a store-level failure degrades to a no-op
    count.
    """
    try:
        cursor = _load_drain_cursor(store)
        # query_events returns newest-first, so under a backlog larger than the
        # batch this keeps the freshest ``batch`` inputs and lets the very
        # oldest overflow age out. That is deliberate for an advisory signal:
        # the watermark still advances monotonically to the newest input seen,
        # so the drain always makes forward progress and never re-fires.
        deferred = query_events(
            store,
            kind="deferred_curiosity_input",
            since=cursor,
            since_exclusive=True,
            limit=batch,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        _log.debug("curiosity_drain_scan_failed: %s", exc)
        return {"drained": 0, "fired": 0, "resolved": 0}

    if not deferred:
        return {"drained": 0, "fired": 0, "resolved": 0}

    deferred_sorted = sorted(deferred, key=lambda e: e["ts"])
    newest_ts = deferred_sorted[-1]["ts"]
    fired = 0
    resolved = 0
    for e in deferred_sorted:
        data = e["data"]
        try:
            entropy = float(data.get("entropy", 0.0))
        except (TypeError, ValueError):
            continue
        session_id = data.get("session_id") or e.get("session_id") or "-"
        cue = str(data.get("cue", ""))
        # Below the firing floor the recall was unambiguous. A legacy input
        # (written before entropy was recorded) also lands here and carries no
        # usable cue signal, so resolution is keyed on the cue and no-ops when
        # absent. Either way the input is consumed, never fired.
        if entropy < ENTROPY_LOW:
            resolved += resolve_answered_questions(
                store, _cue_key(cue), session_id,
            )
            continue
        try:
            turn = int(data.get("turn", 0))
        except (TypeError, ValueError):
            turn = 0
        hits: list[_DrainHit] = []
        for hid in data.get("hit_ids", []):
            try:
                hits.append(_DrainHit(UUID(str(hid))))
            except (TypeError, ValueError):
                continue
        try:
            question = fire_curiosity(store, hits, cue, entropy, session_id, turn)
        except (OSError, RuntimeError, ValueError) as exc:
            _log.debug("curiosity_drain_fire_failed: %s", exc)
            continue
        if question is not None:
            fired += 1

    try:
        write_event(
            store,
            kind=_DRAIN_CURSOR_KIND,
            data={"through_ts": newest_ts.isoformat()},
            severity="info",
        )
    except (OSError, RuntimeError, ValueError) as exc:
        _log.debug("curiosity_drain_cursor_write_failed: %s", exc)

    return {
        "drained": len(deferred_sorted),
        "fired": fired,
        "resolved": resolved,
    }

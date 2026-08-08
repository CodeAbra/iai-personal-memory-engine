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

# The miner resolves a pending question when its topic gains records after
# the question was minted (the conversation moved on); the TTL is the
# backstop for questions whose topic simply went quiet.
PENDING_TTL_DAYS: int = 7

MINE_MAX_PER_SESSION: int = 2
MINE_MAX_PER_RUN: int = 10
MINE_SCAN_LIMIT: int = 500

_log = logging.getLogger(__name__)

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
    cue: str = ""


def compute_entropy(scores: list[float]) -> float:
    # Normalized to [0, 1] by log2(n): the tier thresholds (ENTROPY_LOW /
    # silent / question) are calibrated on that scale — raw Shannon entropy
    # grows with candidate count and would push every multi-hit recall past
    # the top threshold.
    if not scores:
        return 0.0
    positive = [max(0.0, float(s)) for s in scores]
    total = sum(positive)
    if total <= 0:
        return 0.0
    probs = [p / total for p in positive]
    h = 0.0
    n_pos = 0
    for p in probs:
        if p > 0:
            n_pos += 1
            h -= p * math.log2(p)
    if n_pos <= 1:
        return 0.0
    return h / math.log2(n_pos)


def _last_curiosity_turn(store: MemoryStore, session_id: str) -> int | None:
    events = query_events(store, kind="curiosity_question", limit=20)
    for e in events:
        if e.get("session_id") == session_id:
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

    trigger_ids: list[UUID] = [h.record_id for h in hits[:5]]
    return _mint_question(
        store, trigger_ids, cue=cue, entropy=entropy,
        session_id=session_id, turn=turn,
    )


def _mint_question(
    store: MemoryStore,
    trigger_ids: list[UUID],
    *,
    cue: str,
    entropy: float,
    session_id: str,
    turn: int,
    text: str | None = None,
) -> CuriosityQuestion:
    q_id = uuid4()
    if text is not None:
        tier = "question"
    elif entropy < ENTROPY_HIGH:
        tier = "inline"
        text = f"I'm not fully sure -- did you mean {cue!r}?"
    else:
        tier = "question"
        text = f"Could you clarify: {cue!r}?"

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
            "tier": tier,
            "entropy": float(entropy),
            "turn": int(turn),
            "cue": cue[:200],
            "triggered_by": [str(t) for t in trigger_ids],
        },
        severity="info",
        session_id=session_id,
        source_ids=trigger_ids,
    )
    return question


def _fresh_topic_scan(
    store: MemoryStore, cue: str,
) -> "tuple[list[UUID], dict[UUID, object]]":
    """Rank the cue against the CURRENT store: (top ids, id -> record).

    Empty result on any failure — a question needs present-tense evidence,
    so a failed scan means no question, never a stale one.
    """
    try:
        from iai_mcp.embed import embed_query, embedder_for_store

        vec = list(embed_query(embedder_for_store(store), cue))
        pairs = store.exact_top_k(vec, k=10)
        ids = [rid for rid, _ in pairs[:5]]
        if not ids:
            return [], {}
        recs = store.get_batch(ids)
        return ids, recs
    except Exception as exc:  # noqa: BLE001 -- nightly step must degrade, not crash
        _log.debug("curiosity_fresh_scan_failed: %s", exc)
        return [], {}


def _contradiction_pair(
    store: MemoryStore, top_ids: "list[UUID]", recs: dict,
) -> "tuple[UUID, UUID] | None":
    """First contradicts edge incident to the fresh top candidates."""
    try:
        edges = store.incident_edges(
            top_ids, edge_types=["contradicts"], top_k=None,
        )
    except Exception as exc:  # noqa: BLE001 -- evidence probe must degrade
        _log.debug("curiosity_contradiction_probe_failed: %s", exc)
        return None
    for src, lst in edges.items():
        for item in lst:
            try:
                neighbor = item[0] if isinstance(item[0], UUID) else UUID(str(item[0]))
            except (TypeError, ValueError):
                continue
            return src, neighbor
    return None


def _one_line(text: str, limit: int = 100) -> str:
    return " ".join((text or "").split())[:limit]


def process_deferred_inputs(
    store: MemoryStore,
    *,
    scan_limit: int = MINE_SCAN_LIMIT,
) -> dict:
    """Replay buffered ``deferred_curiosity_input`` events and mint pending
    questions from PRESENT contradictions.

    The deferred event is only a pointer at a topic; every decision is made
    against the current store. Per event: re-rank the cue now; if the topic
    gained records after the snapshot it is being actively worked — skip,
    and resolve any pending question on that cue (the conversation moved
    on). A question mints only when the fresh top candidates carry a live
    ``contradicts`` edge — the question then names both sides verbatim.
    High-entropy topics without a contradiction are dense knowledge, not
    confusion: they earn at most a silent log.

    A ``curiosity_mine_run`` watermark event records the newest processed
    timestamp, so re-running (including a WAL-recovery replay of the sleep
    cycle) never double-mints. Events written before scores were added to
    the payload carry no entropy signal and are skipped past by the
    watermark.
    """
    since: datetime | None = None
    runs = query_events(store, kind="curiosity_mine_run", limit=1)
    if runs:
        raw = runs[0]["data"].get("through_ts")
        try:
            since = datetime.fromisoformat(str(raw))
        except (TypeError, ValueError):
            since = None

    events = query_events(
        store,
        kind="deferred_curiosity_input",
        since=since,
        since_exclusive=True,
        limit=scan_limit,
    )
    events.reverse()

    asked_cues: set[str] = set()
    pending_by_cue: dict[str, str] = {}
    for prior in query_events(store, kind="curiosity_question", limit=200):
        prior_cue = prior["data"].get("cue")
        if prior_cue:
            asked_cues.add(prior_cue)
    for q in pending_questions(store):
        if q.cue:
            pending_by_cue[q.cue] = str(q.id)

    minted = 0
    silent = 0
    resolved = 0
    skipped_active = 0
    per_session: dict[str, int] = {}
    through_ts: datetime | None = None
    # The same cue recurs across a day's deferred events; embed + rank +
    # edge-probe once per distinct cue, not once per event.
    scan_memo: dict[str, tuple] = {}

    for e in events:
        through_ts = e["ts"]
        data = e["data"]
        cue = str(data.get("cue") or "")[:200]
        scores_raw = data.get("scores")
        if not cue or not isinstance(scores_raw, list) or not scores_raw:
            continue
        try:
            scores = [float(s) for s in scores_raw]
        except (TypeError, ValueError):
            continue
        entropy = compute_entropy(scores)
        if entropy < ENTROPY_LOW:
            continue
        sid = e.get("session_id") or "-"

        memo = scan_memo.get(cue)
        if memo is None:
            memo = _fresh_topic_scan(store, cue)
            scan_memo[cue] = memo
        top_ids, recs = memo

        event_ts = e.get("ts")
        topic_active = False
        if isinstance(event_ts, datetime):
            for rec in recs.values():
                ca = getattr(rec, "created_at", None)
                if ca is not None and ca > event_ts:
                    topic_active = True
                    break
        if topic_active:
            skipped_active += 1
            qid = pending_by_cue.pop(cue, None)
            if qid is not None:
                write_event(
                    store,
                    kind="curiosity_resolved",
                    data={"question_id": qid, "reason": "topic_active"},
                    severity="info",
                    session_id=sid,
                )
                resolved += 1
            continue

        pair = _contradiction_pair(store, top_ids, recs) if top_ids else None
        if pair is None:
            write_event(
                store,
                kind="curiosity_silent_log",
                data={
                    "cue": cue,
                    "entropy": float(entropy),
                    "source_ids": [str(x) for x in data.get("hit_ids", [])[:3]],
                },
                severity="info",
                session_id=sid,
            )
            silent += 1
            continue

        if cue in asked_cues:
            continue
        if minted >= MINE_MAX_PER_RUN or per_session.get(sid, 0) >= MINE_MAX_PER_SESSION:
            continue

        a_id, b_id = pair
        both = store.get_batch([a_id, b_id])
        a_rec, b_rec = both.get(a_id), both.get(b_id)
        if a_rec is None or b_rec is None:
            continue
        text = (
            f"Two memories disagree — which is current: "
            f"\"{_one_line(a_rec.literal_surface)}\" or "
            f"\"{_one_line(b_rec.literal_surface)}\"?"
        )
        _mint_question(
            store, [a_id, b_id], cue=cue, entropy=entropy,
            session_id=sid, turn=int(data.get("turn", 0) or 0),
            text=text,
        )
        asked_cues.add(cue)
        per_session[sid] = per_session.get(sid, 0) + 1
        minted += 1

    if through_ts is not None:
        write_event(
            store,
            kind="curiosity_mine_run",
            data={
                "through_ts": through_ts.isoformat(),
                "scanned": len(events),
                "scan_limit": int(scan_limit),
                "minted": minted,
                "silent": silent,
                "skipped_active": skipped_active,
                "resolved": resolved,
            },
            severity="info",
        )

    return {
        "curiosity_scanned": len(events),
        "curiosity_minted": minted,
        "curiosity_silent": silent,
        "curiosity_resolved": resolved,
    }


def pending_questions(
    store: MemoryStore,
    session_id: str | None = None,
) -> list[CuriosityQuestion]:
    events = query_events(store, kind="curiosity_question", limit=200)
    resolved_events = query_events(store, kind="curiosity_resolved", limit=500)
    resolved_ids = {
        r["data"].get("question_id")
        for r in resolved_events
        if r["data"].get("question_id")
    }
    ttl_floor = datetime.now(timezone.utc) - timedelta(days=PENDING_TTL_DAYS)
    out: list[CuriosityQuestion] = []
    for e in events:
        if session_id is not None and e.get("session_id") != session_id:
            continue
        ts = e.get("ts")
        if isinstance(ts, datetime) and ts < ttl_floor:
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
                cue=str(data.get("cue", "")),
            )
        )
    return out


def get_pending_questions(store: MemoryStore, limit: int = 2) -> list[dict]:
    qs = pending_questions(store)
    return [
        {
            "id": str(q.id),
            "text": q.text,
            "entropy": q.entropy,
            "tier": q.tier,
            "cue": q.cue,
        }
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

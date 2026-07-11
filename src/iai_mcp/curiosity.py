from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from iai_mcp.events import query_events, write_event
from iai_mcp.store import MemoryStore


ENTROPY_LOW: float = 0.4
ENTROPY_MID: float = 0.7
ENTROPY_HIGH: float = 0.9
COOLDOWN_TURNS: int = 3

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


def compute_entropy(scores: list[float]) -> float:
    if not scores:
        return 0.0
    positive = [max(0.0, float(s)) for s in scores]
    total = sum(positive)
    if total <= 0:
        return 0.0
    probs = [p / total for p in positive]
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log2(p)
    return h


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
) -> list[CuriosityQuestion]:
    events = query_events(store, kind="curiosity_question", limit=200)
    resolved_events = query_events(store, kind="curiosity_resolved", limit=500)
    resolved_ids = {
        r["data"].get("question_id")
        for r in resolved_events
        if r["data"].get("question_id")
    }
    out: list[CuriosityQuestion] = []
    for e in events:
        if session_id is not None and e.get("session_id") != session_id:
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
    qs = pending_questions(store)
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

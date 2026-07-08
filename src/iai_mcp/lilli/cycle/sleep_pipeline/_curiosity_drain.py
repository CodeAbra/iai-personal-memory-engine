from __future__ import annotations

import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable
from uuid import UUID

from iai_mcp.lilli.cycle.sleep_pipeline import SleepStep

logger = logging.getLogger(__name__)

# The recall hot-path (pipeline._apply_post_rank_pipeline) buffers a
# `deferred_curiosity_input` event per non-verbatim recall instead of computing
# curiosity inline. This step is the deferred consumer: it drains those inputs,
# recomputes the recall-hit entropy, and calls fire_curiosity — the only writer
# of `curiosity_question` events that `curiosity_pending` reads. Without this
# step the queue is structurally always empty (regressed in v1.0.0, which moved
# the producer to the deferred model but never added this consumer).
_DEFERRED_KIND = "deferred_curiosity_input"
_WATERMARK_KIND = "curiosity_drained"
_MAX_INPUTS_PER_RUN = 200


def _last_drain_watermark(store: Any) -> datetime | None:
    from iai_mcp.events import query_events

    markers = query_events(store, kind=_WATERMARK_KIND, limit=1)
    if not markers:
        return None
    wm = markers[0].get("data", {}).get("watermark")
    if not wm:
        return None
    try:
        dt = datetime.fromisoformat(str(wm))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def step_curiosity_drain(
    self, interrupt_check: Callable[[], bool] | None,
) -> tuple[bool, dict[str, Any]]:
    from iai_mcp.curiosity import compute_entropy, fire_curiosity
    from iai_mcp.events import flush_event_buffer, query_events, write_event

    if self._check_interrupt(SleepStep.CURIOSITY_DRAIN, 0, interrupt_check):
        return False, {}

    store = self._store

    # Producer writes deferred inputs with buffered=True, so flush first to make
    # this cycle's inputs visible to the events table before we query.
    try:
        flush_event_buffer(store)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.debug("curiosity_drain flush_event_buffer failed: %s", exc)

    watermark = _last_drain_watermark(store)
    inputs = query_events(
        store,
        kind=_DEFERRED_KIND,
        since=watermark,
        limit=_MAX_INPUTS_PER_RUN,
        since_exclusive=True,
    )
    if not inputs:
        return True, {"deferred_inputs": 0, "questions_fired": 0}

    # query_events returns newest-first; process oldest-first so fire_curiosity's
    # per-session cooldown (COOLDOWN_TURNS) advances monotonically with `turn`.
    inputs.reverse()

    max_ts = watermark
    processed = 0
    fired = 0
    fallback_turn = 0
    for ev in inputs:
        ev_ts = ev.get("ts")
        if isinstance(ev_ts, datetime) and (max_ts is None or ev_ts > max_ts):
            max_ts = ev_ts

        data = ev.get("data") or {}
        scores = data.get("scores")
        hit_ids = data.get("hit_ids") or []
        # Pre-feature inputs (buffered before this step existed) and empty
        # recalls carry no scores — entropy is undefined, so skip rather than
        # fire spuriously. They still advance the watermark and are never
        # revisited.
        if not scores or not hit_ids:
            continue

        cue = data.get("cue", "")
        session_id = data.get("session_id") or ev.get("session_id") or "-"
        turn = data.get("turn")
        if not isinstance(turn, int):
            fallback_turn += 1
            turn = fallback_turn

        hits: list[Any] = []
        for hid in hit_ids:
            try:
                hits.append(SimpleNamespace(record_id=UUID(str(hid))))
            except (TypeError, ValueError):
                continue
        if not hits:
            continue

        try:
            entropy = compute_entropy([float(s) for s in scores])
        except (TypeError, ValueError) as exc:
            logger.debug("curiosity_drain entropy failed for %s: %s", ev.get("id"), exc)
            continue

        processed += 1
        try:
            q = fire_curiosity(
                store, hits, cue=cue, entropy=entropy,
                session_id=session_id, turn=turn,
            )
        except Exception as exc:  # noqa: BLE001 -- one bad input must not abort the drain
            logger.debug("curiosity_drain fire_curiosity failed for %s: %s", ev.get("id"), exc)
            continue
        if q is not None:
            fired += 1

    # Advance the watermark to the max ts actually observed (not "now") so any
    # input written mid-run with an earlier ts remains eligible next cycle.
    if max_ts is not None:
        try:
            write_event(
                store,
                kind=_WATERMARK_KIND,
                data={
                    "watermark": max_ts.isoformat(),
                    "inputs_seen": len(inputs),
                    "inputs_processed": processed,
                    "questions_fired": fired,
                },
                severity="info",
                session_id="system",
            )
        except Exception as exc:  # noqa: BLE001 -- watermark write is best-effort
            logger.debug("curiosity_drain watermark write failed: %s", exc)

    return True, {
        "deferred_inputs": len(inputs),
        "inputs_processed": processed,
        "questions_fired": fired,
    }

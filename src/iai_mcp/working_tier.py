"""Working tier: the single active-task structured state.

Process-global, in-process RAM only — never touches Hippo directly, never
an HD-encoded persisted representation. The awake recall path must never
import this module. Text fields are the verbatim source of truth; the BSC
structure hypervector is an optional similarity accelerator only and is
never reconstructed back into text.

A plaintext snapshot of the active task is additionally emitted to a cache
file (0600, atomic replace) on every feed — the same daemon-written /
reader-consumed pattern as the session-start payload cache — so external
per-turn readers (shell hooks) can surface the active task WITHOUT a daemon
socket round-trip and without importing this module.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

WORKING_TIER_MAX_SLOTS: int = 5
"""Bound on each open-ended list on WorkingSetEntry (the ~4+-1 chunk cap)."""

WORKING_TIER_IDLE_CLOSE_SEC: float = 3600
"""Idle-gap task boundary: elapsed seconds since the last turn past which
the active task is considered stale and is closed on the next turn."""

WORKING_TIER_MAX_GOAL_CHARS: int = 512
"""Verbatim (never smoothed) upper bound on a first-turn-seeded goal."""

_WORKING_TIER_ROLES: tuple[str, ...] = ("GOAL", "SUBGOAL", "HYPOTHESIS", "RESULT", "FOCUS")
"""Private BSC role names for the structure encoder. Never added to
BSC_ROLE_VOCABULARY — this vocabulary is scoped to this module only."""

_CONSOLIDATION_CUE: str = "working-tier consolidation"
"""Provenance cue on this module's own consolidation writes. update_from_record
recognizes it and no-ops — otherwise the consolidation insert would re-enter
the feed and immediately reopen a task with the just-closed content."""


@dataclass
class WorkingSetEntry:
    goal: str
    open_subgoals: list[str] = field(default_factory=list)
    closed_subgoals: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    results: list[str] = field(default_factory=list)
    focus: str = ""
    raw_sensory: list[str] = field(default_factory=list)
    session_id: str = "-"
    last_turn_ts: float = 0.0


_lock = threading.Lock()
_ACTIVE_TASK: WorkingSetEntry | None = None


WORKING_TIER_CACHE_ENV = "IAI_MCP_WORKING_TIER_CACHE"
"""Env override for the snapshot cache path (tests / non-default roots)."""


def _cache_path(store: Any = None) -> Path:
    env = os.environ.get(WORKING_TIER_CACHE_ENV)
    if env:
        return Path(env)
    root = getattr(store, "root", None)
    if root is not None:
        return Path(root) / ".working-tier.cached.md"
    return Path.home() / ".iai-mcp" / ".working-tier.cached.md"


def _render_snapshot(entry: WorkingSetEntry) -> str:
    lines = [
        "# Working tier — active task",
        f"session: {entry.session_id}",
        f"goal: {entry.goal}",
    ]
    if entry.open_subgoals:
        lines.append("open subgoals:")
        lines += [f"- {s}" for s in entry.open_subgoals]
    if entry.hypotheses:
        lines.append("hypotheses:")
        lines += [f"- {h}" for h in entry.hypotheses]
    if entry.results:
        lines.append("results:")
        lines += [f"- {r}" for r in entry.results]
    if entry.focus:
        lines.append(f"focus: {entry.focus}")
    if entry.raw_sensory:
        lines.append("recent turns:")
        lines += [f"- {t}" for t in entry.raw_sensory]
    return "\n".join(lines) + "\n"


def _persist_snapshot(store: Any = None) -> None:
    """Fail-soft cache emission; absence of a task removes the file so a
    reader never surfaces a closed task as active."""
    try:
        path = _cache_path(store)
        with _lock:
            entry = _ACTIVE_TASK
            text = _render_snapshot(entry) if entry is not None else ""
        if not text:
            path.unlink(missing_ok=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001 -- cache emission must never fail a write
        logger.debug("working_tier snapshot persist failed: %s", exc)


def _reset() -> None:
    """Test-only: clear the singleton so state never leaks across tests."""
    global _ACTIVE_TASK
    with _lock:
        _ACTIVE_TASK = None


def _bounded_append(items: list[str], value: str, *, max_slots: int, allow_evict: bool) -> None:
    items.append(value)
    if len(items) > max_slots:
        if allow_evict:
            del items[0]
        else:
            raise ValueError(
                f"working tier slot overflow: {len(items)} > {max_slots}; "
                "durable results must be consolidated via close_task, not dropped"
            )


def open_task(goal: str, *, session_id: str = "-") -> WorkingSetEntry:
    """Open a fresh task. A second open_task closes-then-reopens — a single
    active task only (never two live tasks)."""
    global _ACTIVE_TASK
    with _lock:
        stale = _ACTIVE_TASK
        entry = WorkingSetEntry(goal=goal.strip(), session_id=session_id, last_turn_ts=time.time())
        _ACTIVE_TASK = entry

    # Consolidation may re-enter store.insert() -> _feed_working ->
    # update_from_record, which must be able to acquire _lock again — never
    # call _consolidate while holding it (lock is not re-entrant).
    if stale is not None:
        _consolidate_detached(stale, store=None)

    return entry


def update_task(
    *,
    sub_goal: str | None = None,
    hypothesis: str | None = None,
    result: str | None = None,
    focus: str | None = None,
    close_sub_goal: str | None = None,
) -> WorkingSetEntry | None:
    """Fold new structured state into the active task. Text is stored
    verbatim (strip only, never paraphrased)."""
    with _lock:
        entry = _ACTIVE_TASK
        if entry is None:
            return None
        if sub_goal is not None:
            _bounded_append(
                entry.open_subgoals, sub_goal.strip(),
                max_slots=WORKING_TIER_MAX_SLOTS, allow_evict=True,
            )
        if hypothesis is not None:
            _bounded_append(
                entry.hypotheses, hypothesis.strip(),
                max_slots=WORKING_TIER_MAX_SLOTS, allow_evict=True,
            )
        if result is not None:
            _bounded_append(
                entry.results, result.strip(),
                max_slots=WORKING_TIER_MAX_SLOTS, allow_evict=False,
            )
        if focus is not None:
            entry.focus = focus.strip()
        if close_sub_goal is not None:
            marker = close_sub_goal.strip()
            if marker in entry.open_subgoals:
                entry.open_subgoals.remove(marker)
            # A closed sub-goal is a durable outcome (consolidated on close),
            # never silently dropped — mirrors the results-list overflow
            # policy, unlike still-open sub-goals/hypotheses.
            _bounded_append(
                entry.closed_subgoals, marker,
                max_slots=WORKING_TIER_MAX_SLOTS, allow_evict=False,
            )
        return entry


def read_task(*, session_id: str | None = None) -> WorkingSetEntry | None:
    """Return the live active task, or None. No store/db parameter — the
    awake-read contract. Folds a live sensory tail so per-turn continuity
    does not lag the promotion cadence."""
    # Fold under the lock, matching populate_from_sensory — the sensory fold
    # MUTATES entry.raw_sensory, so it must not race a concurrent
    # update_from_record folding the same list. _fold_sensory_tail never
    # re-enters _lock (the sensory tier does not call back into the working
    # tier), so holding the lock across it cannot deadlock.
    with _lock:
        entry = _ACTIVE_TASK
        if entry is None:
            return None
        target_session = session_id if session_id is not None else entry.session_id
        try:
            _fold_sensory_tail(entry, target_session)
        except Exception as exc:  # noqa: BLE001 -- read must never crash on a sensory hiccup
            logger.debug("working_tier read_task sensory fold failed: %s", exc)
        return entry


def _fold_sensory_tail(entry: WorkingSetEntry, session_id: str) -> None:
    from iai_mcp.sensory import sensory_pending

    pending = sensory_pending(session_id=session_id)
    for item in pending:
        text = None
        if isinstance(item, dict):
            text = item.get("text") or item.get("literal_surface")
        elif isinstance(item, str):
            text = item
        if text and text not in entry.raw_sensory:
            _bounded_append(
                entry.raw_sensory, text,
                max_slots=WORKING_TIER_MAX_SLOTS, allow_evict=True,
            )


def populate_from_sensory(session_id: str) -> WorkingSetEntry | None:
    """Read-only pull from the sensory buffer into the active task's raw
    view. Never mutates the sensory tier or triggers promotion."""
    with _lock:
        entry = _ACTIVE_TASK
        if entry is None:
            return None
        _fold_sensory_tail(entry, session_id)
        return entry


def _turn_ts(record: Any) -> float:
    """Normalize record.created_at (a datetime) to epoch seconds, tz-aware
    first, mirroring _feed_recency's tz-normalize. Falls back to time.time()
    only when created_at is None. There is no record.ts field."""
    created_at = getattr(record, "created_at", None)
    if created_at is None:
        return time.time()
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return created_at.timestamp()


def _literal_surface_of(record: Any) -> str:
    text = getattr(record, "literal_surface", None)
    return text if isinstance(text, str) else ""


def _is_own_consolidation_write(record: Any) -> bool:
    """True if record was written by _consolidate_detached itself. Such a
    write must not re-enter the working tier's feed — otherwise closing a
    task would immediately reopen one seeded with the just-closed content."""
    provenance = getattr(record, "provenance", None) or []
    if not provenance:
        return False
    try:
        return provenance[0].get("cue") == _CONSOLIDATION_CUE
    except Exception:  # noqa: BLE001
        return False


_DURABLE_PROVENANCE: dict[str, bool] = {"working_tier_durable": True}
"""Out-of-band durability marker on a consolidation write. Relaxes capture's
length floor for a short-but-real durable result WITHOUT touching the stored
text — literal_surface stays byte-identical to the original result."""


def _consolidate_detached(entry: WorkingSetEntry, *, store: Any) -> dict[str, Any]:
    """Consolidate durable results from entry into the declarative store
    (promote-then-clear), then return a status dict. entry must already be
    detached from _ACTIVE_TASK and _lock must NOT be held by the caller —
    this calls into store.insert(), which re-enters _feed_working. Never
    loses a result even if store is unavailable — in that case results stay
    undelivered and are logged, since there is nothing durable to write
    into. A mid-loop store failure never silently drops the remaining
    results: they are surfaced in the returned status."""
    if store is None:
        if entry.results:
            logger.debug(
                "working_tier consolidation skipped (no store available); "
                "%d result(s) not persisted", len(entry.results),
            )
        return {
            "status": "closed",
            "consolidated": [],
            "undelivered": list(entry.results) + list(entry.closed_subgoals),
            "reason": "no_store",
        }

    from iai_mcp import capture

    consolidated: list[str] = []
    undelivered: list[str] = []
    for durable_text in list(entry.results) + list(entry.closed_subgoals):
        try:
            outcome = capture.capture_turn(
                store,
                cue=_CONSOLIDATION_CUE,
                text=durable_text,
                tier="episodic",
                role="assistant",
                provenance_extra=_DURABLE_PROVENANCE,
            )
        except Exception as exc:  # noqa: BLE001 -- a store hiccup on one result
            # must never silently swallow the rest — surface it.
            logger.warning(
                "working_tier consolidation failed for a durable result: %s", exc
            )
            undelivered.append(durable_text)
            continue
        if outcome.get("status") in ("inserted", "reinforced"):
            record_id = outcome.get("record_id")
            if record_id:
                consolidated.append(record_id)
        else:
            undelivered.append(durable_text)

    if undelivered:
        logger.warning(
            "working_tier consolidation: %d durable result(s) undelivered",
            len(undelivered),
        )
        return {
            "status": "partial",
            "consolidated": consolidated,
            "undelivered": undelivered,
        }
    return {"status": "closed", "consolidated": consolidated}


def close_task(store: Any = None) -> dict[str, Any]:
    """Consolidate durable results via ordinary awake store.insert() (through
    capture_turn), THEN clear the singleton. Never clears a result before it
    is written. Also the session-end / teardown close path."""
    global _ACTIVE_TASK
    with _lock:
        entry = _ACTIVE_TASK
        _ACTIVE_TASK = None

    if entry is None:
        _persist_snapshot(store)
        return {"status": "no_active_task", "consolidated": []}

    # Consolidation may re-enter store.insert() -> _feed_working ->
    # update_from_record, which must acquire _lock again — the lock is
    # already released above before this call.
    result = _consolidate_detached(entry, store=store)
    _persist_snapshot(store)
    return result


def _open_fresh_entry_locked(record: Any) -> WorkingSetEntry:
    """Caller must hold _lock. Installs and returns a fresh WorkingSetEntry
    for the given record's session, stamped at the record's turn time. The
    opening turn of a monotropic burst is the task's goal/intent, so the
    first folded turn's verbatim content (bounded to the slot cap, never
    smoothed) seeds the goal."""
    global _ACTIVE_TASK
    now = _turn_ts(record)
    session_id = "-"
    provenance = getattr(record, "provenance", None) or []
    if provenance:
        try:
            session_id = provenance[0].get("session_id", session_id) or session_id
        except Exception:  # noqa: BLE001
            pass
    goal = _literal_surface_of(record).strip()[:WORKING_TIER_MAX_GOAL_CHARS]
    entry = WorkingSetEntry(goal=goal, session_id=session_id, last_turn_ts=now)
    _ACTIVE_TASK = entry
    return entry


def update_from_record(record: Any, *, store: Any = None) -> None:
    """The per-turn feed target and the lazy idle-gap boundary detector.
    Never crashes a write — every failure is caught and logged."""
    try:
        if _is_own_consolidation_write(record):
            return

        now = _turn_ts(record)
        # The working tier is the LIVE attention scratchpad: only turns near
        # wall-clock now may touch it. A replayed historical turn (backlog
        # drain, transcript import, recovery) would otherwise overwrite the
        # active task with a past that already happened — and the per-turn
        # hook would inject that stale task as the current one.
        if (time.time() - now) > WORKING_TIER_IDLE_CLOSE_SEC:
            return
        stale: WorkingSetEntry | None = None

        with _lock:
            entry = _ACTIVE_TASK
            if entry is not None and (now - entry.last_turn_ts) > WORKING_TIER_IDLE_CLOSE_SEC:
                stale = entry
                entry = None

            if entry is None:
                entry = _open_fresh_entry_locked(record)

            text = _literal_surface_of(record)
            if text:
                _bounded_append(
                    entry.raw_sensory, text,
                    max_slots=WORKING_TIER_MAX_SLOTS, allow_evict=True,
                )
            # Advance the idle clock monotonically. The async write-queue flush
            # can deliver records out of created_at order; a bare assignment
            # would rewind last_turn_ts on an older record and spuriously
            # re-trip (or fail to trip) the idle gap on the next in-order turn.
            entry.last_turn_ts = max(entry.last_turn_ts, now)

        # Consolidation may re-enter store.insert() -> _feed_working ->
        # update_from_record, which must acquire _lock again — never call
        # _consolidate while holding it (lock is not re-entrant).
        if stale is not None:
            _consolidate_detached(stale, store=store)
        _persist_snapshot(store)
    except Exception as exc:  # noqa: BLE001 -- hook isolation, never crash a write
        logger.debug("working_tier update_from_record failed: %s", exc)


def encode_structure(entry: WorkingSetEntry) -> bytes:
    """Optional BSC bundle of the current structure. A similarity
    accelerator only — text is never reconstructed from this hypervector."""
    from iai_mcp.lilli.errors import BundleCapacityError
    from iai_mcp.lilli.tiers import bsc

    goal_role, subgoal_role, hypothesis_role, result_role, focus_role = _WORKING_TIER_ROLES

    pairs: list[tuple[str, bytes]] = []
    if entry.goal:
        pairs.append((goal_role, bsc.filler_hv(entry.goal)))
    for sub_goal in entry.open_subgoals:
        pairs.append((subgoal_role, bsc.filler_hv(sub_goal)))
    for hypothesis in entry.hypotheses:
        pairs.append((hypothesis_role, bsc.filler_hv(hypothesis)))
    for result_text in entry.results:
        pairs.append((result_role, bsc.filler_hv(result_text)))
    if entry.focus:
        pairs.append((focus_role, bsc.filler_hv(entry.focus)))

    try:
        return bsc.bundle(pairs)
    except BundleCapacityError:
        logger.debug("working_tier encode_structure: structure-HV unavailable this turn")
        return b""


__all__ = [
    "WorkingSetEntry",
    "open_task",
    "update_task",
    "read_task",
    "close_task",
    "populate_from_sensory",
    "update_from_record",
    "encode_structure",
    "WORKING_TIER_MAX_SLOTS",
    "WORKING_TIER_IDLE_CLOSE_SEC",
    "WORKING_TIER_MAX_GOAL_CHARS",
]

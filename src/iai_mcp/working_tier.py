"""Working tier: one focal active task plus a bounded park of suspended tasks.

Process-global, in-process RAM only — never touches Hippo directly, never
an HD-encoded persisted representation. The awake recall path must never
import this module. Text fields are the verbatim source of truth; the BSC
structure hypervector is an optional similarity accelerator only and is
never reconstructed back into text.

Attention is unitary: exactly one focal task accumulates turns at any
moment. A turn arriving from a different session is an explicit task
switch — the focal task is parked (its durable results written through to
the episodic store first), and the parked task for the arriving session is
restored or a fresh one opened. The park is a small LRU keyed by session;
eviction closes the task fully. A parked task is never fed and never
injected into any session but its own.

A plaintext snapshot of each task is additionally emitted to a per-session
cache file (0600, atomic replace) — the same daemon-written /
reader-consumed pattern as the session-start payload cache — so external
per-turn readers (shell hooks) can surface their own session's task WITHOUT
a daemon socket round-trip and without importing this module.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import timezone
from pathlib import Path
from typing import Any

from iai_mcp.model_attribution import normalize_model

logger = logging.getLogger(__name__)

WORKING_TIER_MAX_SLOTS: int = 5
"""Bound on each open-ended list on WorkingSetEntry (the ~4+-1 chunk cap)."""

WORKING_TIER_PARK_SLOTS: int = 4
"""Bound on suspended task-sets parked for other live sessions (the same
~4+-1 capacity, applied to the region of direct access)."""

WORKING_TIER_IDLE_CLOSE_SEC: float = 3600
"""Idle-gap task boundary: elapsed seconds since the last turn past which
a task (focal or parked) is considered stale and is closed on the next turn."""

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
    next_action: str = ""
    raw_sensory_models: list[str | None] = field(
        default_factory=list,
        kw_only=True,
    )


_lock = threading.Lock()
_FOCAL: WorkingSetEntry | None = None
_PARKED: "OrderedDict[str, WorkingSetEntry]" = OrderedDict()


WORKING_TIER_CACHE_ENV = "IAI_MCP_WORKING_TIER_CACHE"
"""Env override for the snapshot cache path (tests / non-default roots).
When set, ALL sessions share this one file verbatim — an explicit
single-consumer setup; the per-session layout applies only by default."""

_LEGACY_SNAPSHOT_NAME = ".working-tier.cached.md"

_legacy_swept = False


def _sanitize_session_id(session_id: str) -> str:
    # Session ids come from record provenance (external input) and become a
    # filesystem path component — allowlist, never trust.
    sid = "".join(ch for ch in (session_id or "-") if ch.isalnum() or ch in "-_")[:64]
    return sid or "-"


def _cache_path(store: Any = None, session_id: str = "-") -> Path:
    env = os.environ.get(WORKING_TIER_CACHE_ENV)
    if env:
        return Path(env)
    root = getattr(store, "root", None)
    base = Path(root) if root is not None else Path.home() / ".iai-mcp"
    return base / f".working-tier.{_sanitize_session_id(session_id)}.cached.md"


def _record_model(record: Any) -> str | None:
    try:
        for entry in getattr(record, "provenance", None) or []:
            value = entry.get("model") if isinstance(entry, dict) else None
            model = normalize_model(value)
            if model:
                return model
    except Exception:  # noqa: BLE001 -- capture feed must remain fail-soft
        pass
    return None


def _repair_raw_sensory_models(entry: WorkingSetEntry) -> list[str | None]:
    """Bound and align the sidecar list to raw_sensory length before any duplicate check."""
    models = getattr(entry, "raw_sensory_models", None)
    if not isinstance(models, list):
        models = []
        entry.raw_sensory_models = models
    while len(entry.raw_sensory) > WORKING_TIER_MAX_SLOTS:
        del entry.raw_sensory[0]
    while len(models) > len(entry.raw_sensory):
        del models[0]
    if len(models) < len(entry.raw_sensory):
        models[:0] = [None] * (len(entry.raw_sensory) - len(models))
    for index, model in enumerate(models):
        models[index] = normalize_model(model)
    return models


def _append_raw_sensory(
    entry: WorkingSetEntry,
    text: str,
    model: object = None,
) -> None:
    """Append literal sensory text and its bounded render-only sidecar."""
    sidecar_was_malformed = not isinstance(
        getattr(entry, "raw_sensory_models", None), list
    )
    models = _repair_raw_sensory_models(entry)
    if text in entry.raw_sensory:
        index = entry.raw_sensory.index(text)
        normalized_model = normalize_model(model)
        if (
            not sidecar_was_malformed
            and models[index] is None
            and normalized_model is not None
        ):
            models[index] = normalized_model
        return
    entry.raw_sensory.append(text)
    models.append(normalize_model(model))
    while len(entry.raw_sensory) > WORKING_TIER_MAX_SLOTS:
        del entry.raw_sensory[0]
        del models[0]


def _render_snapshot(entry: WorkingSetEntry) -> str:
    lines = [
        "# Working tier — active task",
        f"session: {entry.session_id}",
        f"goal: {entry.goal}",
        f"next action: {entry.next_action or '(none)'}",
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
        models = _repair_raw_sensory_models(entry)
        lines += [
            f"- {'[model:' + model + '] ' if model else ''}{text}"
            for text, model in zip(entry.raw_sensory, models)
        ]
    return "\n".join(lines) + "\n"


def _sweep_legacy_snapshot(base: Path) -> None:
    global _legacy_swept
    if _legacy_swept:
        return
    _legacy_swept = True
    try:
        (base / _LEGACY_SNAPSHOT_NAME).unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001 -- cache hygiene must never fail a write
        logger.debug("working_tier legacy snapshot sweep failed: %s", exc)


def _persist_entry(store: Any, entry: WorkingSetEntry, *, allow_downgrade: bool = False) -> None:
    """Fail-soft cache emission of one task's per-session snapshot, plus a
    session-agnostic continuity-cache refresh reached by every focal
    mutation (update_task AND update_from_record). allow_downgrade=True
    authorizes an explicit caller-cleared focus/next_action to render thin
    in the eager file instead of resurrecting the prior substantive block —
    an incidental task-switch thin-park must never pass True here."""
    try:
        path = _cache_path(store, entry.session_id)
        with _lock:
            text = _render_snapshot(entry)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not os.environ.get(WORKING_TIER_CACHE_ENV):
            _sweep_legacy_snapshot(path.parent)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001 -- cache emission must never fail a write
        logger.debug("working_tier snapshot persist failed: %s", exc)

    # No _lock is held here. Lazy import: session.py imports working_tier at
    # module scope, so a module-level reverse import would be circular.
    try:
        from iai_mcp import session

        session.write_continuity_cache(store, allow_downgrade=allow_downgrade)
    except Exception as exc:  # noqa: BLE001 -- continuity refresh must never fail a persist
        logger.debug("working_tier continuity cache refresh failed: %s", exc)


def _remove_snapshot(store: Any, session_id: str) -> None:
    """A closed task must not stay readable as active."""
    try:
        _cache_path(store, session_id).unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001 -- cache removal must never fail a close
        logger.debug("working_tier snapshot removal failed: %s", exc)


def _reset() -> None:
    """Test-only: clear the tier so state never leaks across tests."""
    global _FOCAL
    with _lock:
        _FOCAL = None
        _PARKED.clear()


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


def _record_session_id(record: Any) -> str:
    provenance = getattr(record, "provenance", None) or []
    if provenance:
        try:
            return provenance[0].get("session_id", "-") or "-"
        except Exception:  # noqa: BLE001
            return "-"
    return "-"


def _park_focal_locked() -> tuple[WorkingSetEntry, list[str], list[WorkingSetEntry]]:
    """Caller must hold _lock and guarantee _FOCAL is not None. Moves the
    focal task into the park, popping its durable texts for write-through
    consolidation, and returns (parked_entry, durable_texts, evicted).
    Durables are popped under the lock so a switch-back can never see them
    twice; on delivery failure the caller re-attaches them."""
    global _FOCAL
    entry = _FOCAL
    assert entry is not None
    durables = list(entry.results) + list(entry.closed_subgoals)
    entry.results.clear()
    entry.closed_subgoals.clear()
    _PARKED[entry.session_id] = entry
    _PARKED.move_to_end(entry.session_id)
    _FOCAL = None
    evicted: list[WorkingSetEntry] = []
    while len(_PARKED) > WORKING_TIER_PARK_SLOTS:
        _, old = _PARKED.popitem(last=False)
        evicted.append(old)
    return entry, durables, evicted


def open_task(goal: str, *, session_id: str = "-") -> WorkingSetEntry:
    """Open a fresh focal task. A second open_task for the SAME session
    closes-then-reopens (one live task per session); a different session's
    prior focal task is parked, not destroyed."""
    global _FOCAL
    stale: WorkingSetEntry | None = None
    evicted: list[WorkingSetEntry] = []
    with _lock:
        prior = _FOCAL
        if prior is not None:
            if prior.session_id == session_id:
                stale = prior
                _FOCAL = None
            else:
                # No store is reachable here, so durables stay on the parked
                # entry; they consolidate at its eventual close.
                _PARKED[prior.session_id] = prior
                _PARKED.move_to_end(prior.session_id)
                _FOCAL = None
                while len(_PARKED) > WORKING_TIER_PARK_SLOTS:
                    _, old = _PARKED.popitem(last=False)
                    evicted.append(old)
        prior_parked = _PARKED.pop(session_id, None)
        if prior_parked is not None:
            evicted.append(prior_parked)
        entry = WorkingSetEntry(goal=goal.strip(), session_id=session_id, last_turn_ts=time.time())
        _FOCAL = entry

    # Consolidation may re-enter store.insert() -> _feed_working ->
    # update_from_record, which must be able to acquire _lock again — never
    # call _consolidate while holding it (lock is not re-entrant).
    if stale is not None:
        _consolidate_detached(stale, store=None)
    for old in evicted:
        _consolidate_detached(old, store=None)
        _remove_snapshot(None, old.session_id)

    return entry


def update_task(
    *,
    sub_goal: str | None = None,
    hypothesis: str | None = None,
    result: str | None = None,
    focus: str | None = None,
    close_sub_goal: str | None = None,
    next_action: str | None = None,
    session_id: str | None = None,
    store: Any = None,
    explicit_clear: bool = False,
) -> WorkingSetEntry | None:
    """Fold new structured state into the target task. Text is stored
    verbatim (strip and WORKING_TIER_MAX_GOAL_CHARS bound for focus/
    next_action, never paraphrased). session_id routes the target entry
    through _select_entry_locked (None == today's global focal task), so a
    fold from one session can never land on another session's entry.
    When store is given, the folded entry's snapshot is persisted in the
    same call. explicit_clear signals a caller-initiated focus=""/
    next_action="" (not an incidental task-switch thin-park) and is the
    ONLY case that authorizes the eager continuity file's downgrade path."""
    with _lock:
        entry = _select_entry_locked(session_id)
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
            entry.focus = focus.strip()[:WORKING_TIER_MAX_GOAL_CHARS]
        if next_action is not None:
            entry.next_action = next_action.strip()[:WORKING_TIER_MAX_GOAL_CHARS]
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

    # _persist_entry acquires _lock itself -- must run after release above,
    # never inside the with-block (the lock is not re-entrant).
    if store is not None:
        _persist_entry(store, entry, allow_downgrade=explicit_clear)
    return entry


def _select_entry_locked(session_id: str | None) -> WorkingSetEntry | None:
    """Caller must hold _lock. session_id=None means the focal task; a named
    session sees ONLY its own task (focal or parked) — reading never switches
    focus and never surfaces another session's task."""
    if session_id is None:
        return _FOCAL
    if _FOCAL is not None and _FOCAL.session_id == session_id:
        return _FOCAL
    return _PARKED.get(session_id)


def read_task(
    *, session_id: str | None = None, fold_sensory: bool = True,
) -> WorkingSetEntry | None:
    """Return the live task for the given session (focal when None), or None.
    No store/db parameter — the awake-read contract. Folds a live sensory
    tail so per-turn continuity does not lag the promotion cadence, unless
    fold_sensory=False: then _lock is held ONLY for the select and the cheap
    in-memory field read, never across the fold's disk-touching scan — for
    a caller (e.g. the write-triggered continuity render) that never reads
    raw_sensory and must not pay that cost on a high-frequency path."""
    # Fold under the lock, matching populate_from_sensory — the sensory fold
    # MUTATES entry.raw_sensory, so it must not race a concurrent
    # update_from_record folding the same list. _fold_sensory_tail never
    # re-enters _lock (the sensory tier does not call back into the working
    # tier), so holding the lock across it cannot deadlock.
    with _lock:
        entry = _select_entry_locked(session_id)
        if entry is None:
            return None
        if not fold_sensory:
            return entry
        try:
            _fold_sensory_tail(entry, entry.session_id)
        except Exception as exc:  # noqa: BLE001 -- read must never crash on a sensory hiccup
            logger.debug("working_tier read_task sensory fold failed: %s", exc)
        return entry


def _fold_sensory_tail(entry: WorkingSetEntry, session_id: str) -> None:
    from iai_mcp.sensory import sensory_pending

    pending = sensory_pending(session_id=session_id)
    for item in pending:
        text = None
        model = None
        if isinstance(item, dict):
            text = item.get("text") or item.get("literal_surface")
            model = item.get("model")
        elif isinstance(item, str):
            text = item
        if text:
            _append_raw_sensory(entry, text, model)


def populate_from_sensory(session_id: str) -> WorkingSetEntry | None:
    """Caller-directed pull from the named session's sensory buffer into the
    FOCAL task's raw view — an explicit statement that this sensory belongs
    to the current focus. Never mutates the sensory tier or triggers
    promotion."""
    with _lock:
        entry = _FOCAL
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
    """True if record was written by this module's own consolidation. Such a
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


def _consolidate_durables(texts: list[str], store: Any) -> tuple[list[str], list[str]]:
    """Write durable texts to the episodic store via the ordinary capture
    path. Returns (consolidated_record_ids, undelivered_texts) — a mid-loop
    store failure never silently drops the remaining texts."""
    from iai_mcp import capture

    consolidated: list[str] = []
    undelivered: list[str] = []
    for durable_text in texts:
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
    return consolidated, undelivered


def _consolidate_detached(entry: WorkingSetEntry, *, store: Any) -> dict[str, Any]:
    """Consolidate durable results from entry into the declarative store
    (promote-then-clear), then return a status dict. entry must already be
    detached from the tier and _lock must NOT be held by the caller —
    this calls into store.insert(), which re-enters _feed_working. Never
    loses a result even if store is unavailable — in that case results stay
    undelivered and are logged, since there is nothing durable to write
    into."""
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

    consolidated, undelivered = _consolidate_durables(
        list(entry.results) + list(entry.closed_subgoals), store
    )

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


def _refresh_continuity_cache_on_close(store: Any) -> None:
    """close_task empties the focal state entirely -- unlike an incidental
    thin-park mid-session, this IS the new ground truth, so the eager file's
    live-state block must clear even though write_continuity_cache normally
    refuses to downgrade a substantive block. No resolvable store root means
    nothing meaningful to refresh (and no root to scope the write under),
    so this stays inert rather than touching the default-home fallback."""
    if getattr(store, "root", None) is None:
        return
    try:
        from iai_mcp import session

        session.write_continuity_cache(store, allow_downgrade=True)
    except Exception as exc:  # noqa: BLE001 -- continuity refresh must never fail a close
        logger.debug("working_tier close_task continuity cache refresh failed: %s", exc)


def close_task(store: Any = None) -> dict[str, Any]:
    """Consolidate durable results of the focal AND every parked task via
    ordinary awake store.insert() (through capture_turn), THEN clear the
    tier. Never clears a result before it is written. Also the session-end /
    teardown close path."""
    global _FOCAL
    with _lock:
        entries: list[WorkingSetEntry] = []
        if _FOCAL is not None:
            entries.append(_FOCAL)
        entries.extend(_PARKED.values())
        _FOCAL = None
        _PARKED.clear()

    if not entries:
        _remove_snapshot(store, "-")
        _refresh_continuity_cache_on_close(store)
        return {"status": "no_active_task", "consolidated": []}

    # Consolidation may re-enter store.insert() -> _feed_working ->
    # update_from_record, which must acquire _lock again — the lock is
    # already released above before this call.
    consolidated: list[str] = []
    undelivered: list[str] = []
    for entry in entries:
        outcome = _consolidate_detached(entry, store=store)
        consolidated.extend(outcome.get("consolidated") or [])
        undelivered.extend(outcome.get("undelivered") or [])
        _remove_snapshot(store, entry.session_id)

    _refresh_continuity_cache_on_close(store)

    if store is None:
        result: dict[str, Any] = {
            "status": "closed",
            "consolidated": consolidated,
            "undelivered": undelivered,
            "reason": "no_store",
        }
        return result
    if undelivered:
        return {
            "status": "partial",
            "consolidated": consolidated,
            "undelivered": undelivered,
        }
    return {"status": "closed", "consolidated": consolidated}


def _open_fresh_entry_locked(record: Any) -> WorkingSetEntry:
    """Caller must hold _lock. Installs and returns a fresh focal
    WorkingSetEntry for the given record's session, stamped at the record's
    turn time. The opening turn of a monotropic burst is the task's
    goal/intent, so the first folded turn's verbatim content (bounded to the
    slot cap, never smoothed) seeds the goal."""
    global _FOCAL
    now = _turn_ts(record)
    goal = _literal_surface_of(record).strip()[:WORKING_TIER_MAX_GOAL_CHARS]
    entry = WorkingSetEntry(
        goal=goal, session_id=_record_session_id(record), last_turn_ts=now
    )
    _FOCAL = entry
    return entry


def update_from_record(record: Any, *, store: Any = None) -> None:
    """The per-turn feed target, the lazy idle-gap boundary detector, and
    the task-switch point: a turn from a non-focal session parks the focal
    task and restores (or opens) that session's own. Never crashes a write —
    every failure is caught and logged."""
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

        sid = _record_session_id(record)
        closed: list[WorkingSetEntry] = []
        parked: WorkingSetEntry | None = None
        parked_durables: list[str] = []

        with _lock:
            global _FOCAL
            entry = _FOCAL
            if entry is not None and (now - entry.last_turn_ts) > WORKING_TIER_IDLE_CLOSE_SEC:
                closed.append(entry)
                _FOCAL = None
                entry = None
            for key in [
                k for k, e in _PARKED.items()
                if (now - e.last_turn_ts) > WORKING_TIER_IDLE_CLOSE_SEC
            ]:
                closed.append(_PARKED.pop(key))

            if entry is not None and entry.session_id != sid:
                parked, parked_durables, evicted = _park_focal_locked()
                closed.extend(evicted)
                restored = _PARKED.pop(sid, None)
                if restored is not None:
                    _FOCAL = restored
                entry = _FOCAL

            if entry is None:
                entry = _open_fresh_entry_locked(record)

            text = _literal_surface_of(record)
            if text:
                _append_raw_sensory(entry, text, _record_model(record))
            # Advance the idle clock monotonically. The async write-queue flush
            # can deliver records out of created_at order; a bare assignment
            # would rewind last_turn_ts on an older record and spuriously
            # re-trip (or fail to trip) the idle gap on the next in-order turn.
            entry.last_turn_ts = max(entry.last_turn_ts, now)

        # Consolidation may re-enter store.insert() -> _feed_working ->
        # update_from_record, which must acquire _lock again — never call
        # _consolidate while holding it (lock is not re-entrant).
        for old in closed:
            _consolidate_detached(old, store=store)
            _remove_snapshot(store, old.session_id)
        if parked is not None:
            undelivered = parked_durables
            if store is not None and parked_durables:
                _, undelivered = _consolidate_durables(parked_durables, store)
            if undelivered:
                # Custody: undelivered durables go back on the parked task so
                # its eventual close retries them — never dropped.
                with _lock:
                    parked.results.extend(undelivered)
            _persist_entry(store, parked)
        _persist_entry(store, entry)
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
    "WORKING_TIER_PARK_SLOTS",
    "WORKING_TIER_IDLE_CLOSE_SEC",
    "WORKING_TIER_MAX_GOAL_CHARS",
]

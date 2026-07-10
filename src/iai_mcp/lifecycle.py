from __future__ import annotations

import asyncio
import errno
import fcntl
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from iai_mcp.lifecycle_event_log import LifecycleEventLog
from iai_mcp.lifecycle_state import (
    LIFECYCLE_STATE_PATH,
    LifecycleState,
    LifecycleStateRecord,
    load_state,
    save_state,
)

DEFAULT_LOCK_PATH: Path = Path.home() / ".iai-mcp" / ".lifecycle.lock"


class LifecycleStateLocked(RuntimeError):
    pass


class LifecycleEvent(str, Enum):

    HEARTBEAT_REFRESH = "heartbeat_refresh"
    IDLE_5MIN = "idle_5min"
    IDLE_30MIN = "idle_30min"
    SLEEP_ELIGIBLE = "sleep_eligible"
    REQUEST_ARRIVED = "request_arrived"
    SLEEP_CYCLE_DONE = "sleep_cycle_done"
    HIBERNATION_GRACE_EXPIRED = "hibernation_grace_expired"
    WAKE_SIGNAL = "wake_signal"
    TICK = "tick"
    FORCE_SLEEP = "force_sleep"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_transition(
    state: LifecycleState,
    event: LifecycleEvent,
    payload: dict[str, Any] | None = None,
) -> LifecycleState | None:
    payload = payload if payload is not None else {}

    if event is LifecycleEvent.REQUEST_ARRIVED:
        return LifecycleState.WAKE

    if event is LifecycleEvent.FORCE_SLEEP:
        if state is LifecycleState.DROWSY:
            return LifecycleState.SLEEP
        if state in (LifecycleState.SLEEP, LifecycleState.HIBERNATION):
            return None
        return LifecycleState.DROWSY

    if state is LifecycleState.WAKE:
        if event is LifecycleEvent.IDLE_5MIN:
            return LifecycleState.DROWSY
        return None

    if state is LifecycleState.DROWSY:
        if event is LifecycleEvent.HEARTBEAT_REFRESH:
            return LifecycleState.WAKE
        if event is LifecycleEvent.IDLE_30MIN and payload.get("sleep_eligible"):
            return LifecycleState.SLEEP
        return None

    if state is LifecycleState.SLEEP:
        if event is LifecycleEvent.SLEEP_CYCLE_DONE and payload.get("still_idle"):
            return LifecycleState.HIBERNATION
        if event is LifecycleEvent.WAKE_SIGNAL:
            # Morning wake: the consolidation window closed (or a boot
            # restored SLEEP out of hours) — sleep yields to the day.
            return LifecycleState.WAKE
        return None

    if state is LifecycleState.HIBERNATION:
        if event is LifecycleEvent.WAKE_SIGNAL:
            return LifecycleState.WAKE
        return None

    return None


@contextmanager
def _lifecycle_lock(lock_path: Path) -> Iterator[int]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                raise LifecycleStateLocked(
                    f"another process holds {lock_path}"
                ) from exc
            raise
        try:
            yield fd
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        os.close(fd)


class LifecycleStateMachine:

    def __init__(
        self,
        state_path: Path | None = None,
        event_log: LifecycleEventLog | None = None,
        lock_path: Path | None = None,
        shadow_run: bool = False,
        *,
        coordinator: "S2Coordinator | None" = None,  # noqa: F821 — forward ref
    ) -> None:
        self._state_path = state_path if state_path is not None else LIFECYCLE_STATE_PATH
        self._event_log = event_log if event_log is not None else LifecycleEventLog()
        self._lock_path = lock_path if lock_path is not None else DEFAULT_LOCK_PATH
        self._shadow_run = shadow_run
        self._coordinator = coordinator


    @property
    def shadow_run(self) -> bool:
        return self._shadow_run

    @property
    def current_state(self) -> LifecycleState:
        record = load_state(self._state_path)
        return LifecycleState(record["current_state"])

    def snapshot(self) -> LifecycleStateRecord:
        return load_state(self._state_path)


    def compute_transition(
        self,
        state: LifecycleState,
        event: LifecycleEvent,
        payload: dict[str, Any] | None = None,
    ) -> LifecycleState | None:
        return compute_transition(state, event, payload)


    async def dispatch(
        self,
        event: LifecycleEvent,
        *,
        reason: str | None = None,
        **payload: Any,
    ) -> LifecycleState:
        if self._coordinator is None:
            raise RuntimeError(
                "LifecycleStateMachine.dispatch requires a coordinator. "
                "Production callers in daemon.main inject one; tests should "
                "construct an S2Coordinator with state_path=tmp_path."
            )

        current_record = await asyncio.to_thread(load_state, self._state_path)
        from_state = LifecycleState(current_record["current_state"])
        payload_dict = dict(payload)
        target = compute_transition(from_state, event, payload_dict)

        if event in {
            LifecycleEvent.HEARTBEAT_REFRESH,
            LifecycleEvent.REQUEST_ARRIVED,
            LifecycleEvent.WAKE_SIGNAL,
        }:
            now_iso = _utc_now_iso()
            updated_record: LifecycleStateRecord = dict(current_record)  # type: ignore[assignment]
            updated_record["last_activity_ts"] = now_iso
            updated_record["wrapper_event_seq"] = (
                current_record.get("wrapper_event_seq", 0) + 1
            )
            updated_record["shadow_run"] = self._shadow_run
            if updated_record != current_record:
                await asyncio.to_thread(save_state, updated_record, self._state_path)
                current_record = updated_record

        if target is None or target == from_state:
            return from_state

        resolved_reason = reason if reason is not None else event.value
        new_state = await self._coordinator.transition(
            from_state, target, resolved_reason,
        )

        self._event_log.append(
            {
                "event": "state_transition",
                "from": from_state.value,
                "to": new_state.value,
                "trigger": resolved_reason,
            }
        )

        if new_state is LifecycleState.HIBERNATION and self._shadow_run:
            self._event_log.append(
                {
                    "event": "shadow_run_warning",
                    "would_action": "hibernate_kill_process",
                    "blocked_by": "shadow_run=True",
                    "note": (
                        "shadow_run=True is a test-only legacy guard "
                        "preserved for transition tests; production "
                        "daemons run with shadow_run=False where this "
                        "branch never fires."
                    ),
                }
            )

        return new_state

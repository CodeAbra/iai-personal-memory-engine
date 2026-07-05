from __future__ import annotations

import asyncio
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from iai_mcp.events import write_event
from iai_mcp.lifecycle_state import (
    LIFECYCLE_STATE_PATH,
    LifecycleState,
    LifecycleStateRecord,
    load_state,
    save_state,
)

_RING_BUFFER_DEFAULT_SIZE: int = 8


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _wake_transition_bypasses_oscillation(
    to_state: LifecycleState,
    reason: str,
) -> bool:
    return to_state is LifecycleState.WAKE and (
        reason == "request_arrived"
        or reason == "wake_signal"
        or reason.startswith("wake_on_")
    )


class S2OscillationConflict(RuntimeError):

    actual_state: LifecycleState
    attempted_from: LifecycleState
    attempted_to: LifecycleState

    def __init__(
        self,
        *,
        actual_state: LifecycleState,
        attempted_from: LifecycleState,
        attempted_to: LifecycleState,
    ) -> None:
        self.actual_state = actual_state
        self.attempted_from = attempted_from
        self.attempted_to = attempted_to
        super().__init__(
            f"S2 CAS conflict: tried "
            f"{attempted_from.value}->{attempted_to.value} but actual="
            f"{actual_state.value}"
        )


class S2OscillationBlocked(RuntimeError):

    first_transition: dict
    second_transition: dict
    interval_sec: float

    def __init__(
        self,
        *,
        first_transition: dict,
        second_transition: dict,
        interval_sec: float,
    ) -> None:
        self.first_transition = first_transition
        self.second_transition = second_transition
        self.interval_sec = interval_sec
        super().__init__(
            f"S2 oscillation blocked: reverse of "
            f"{first_transition['from_state']}->{first_transition['to_state']}"
            f" within {interval_sec:.3f}s (MIN_INTERVAL_SEC)"
        )


class S2Coordinator:

    def __init__(
        self,
        *,
        store: Any,
        state_path: Path | None = None,
        legacy_path: Path | None = None,
        min_interval_sec: float = 5.0,
        dry_run: bool = False,
        ring_buffer_size: int = _RING_BUFFER_DEFAULT_SIZE,
    ) -> None:
        self._store = store
        self._state_path: Path = (
            state_path if state_path is not None else LIFECYCLE_STATE_PATH
        )
        if legacy_path is not None:
            self._legacy_path: Path | None = legacy_path
        else:
            from iai_mcp.daemon_state import STATE_PATH as _LEGACY_STATE_PATH
            self._legacy_path = _LEGACY_STATE_PATH
        self.lock: asyncio.Lock = asyncio.Lock()
        self.version: int = 0
        self._min_interval_sec: float = float(min_interval_sec)
        self._dry_run: bool = bool(dry_run)
        self._ring_buffer: deque[dict] = deque(maxlen=ring_buffer_size)

    async def transition(
        self,
        from_state: LifecycleState,
        to_state: LifecycleState,
        reason: str,
    ) -> LifecycleState:
        version_before = self.version
        now_mono = time.monotonic()

        async with self.lock:
            rec: LifecycleStateRecord = load_state(self._state_path)
            actual_state = LifecycleState(rec["current_state"])

            if actual_state != from_state:
                cas_body = {
                    "from_state": from_state.value,
                    "to_state": to_state.value,
                    "reason": reason,
                    "version_before": version_before,
                    "version_after": version_before,
                    "succeeded": False,
                    "conflict_reason": "cas_mismatch",
                }
                try:
                    write_event(
                        self._store,
                        "s2_transition_attempt",
                        cas_body,
                        severity="info",
                    )
                except Exception:  # noqa: BLE001 -- event emit must never crash FSM  # noqa: BLE001 -- event-store failure must never shadow FSM
                    pass
                raise S2OscillationConflict(
                    actual_state=actual_state,
                    attempted_from=from_state,
                    attempted_to=to_state,
                )

            oscillation_first: dict | None = None
            for entry in reversed(self._ring_buffer):
                if (
                    entry["from_state"] == to_state.value
                    and entry["to_state"] == from_state.value
                    and (now_mono - entry["ts_monotonic"]) < self._min_interval_sec
                ):
                    oscillation_first = entry
                    break

            if (
                oscillation_first is not None
                and not _wake_transition_bypasses_oscillation(to_state, reason)
            ):
                second_transition = {
                    "from_state": from_state.value,
                    "to_state": to_state.value,
                    "reason": reason,
                    "ts_monotonic": now_mono,
                }
                interval_sec = now_mono - oscillation_first["ts_monotonic"]
                block_body = {
                    "first_transition": oscillation_first,
                    "second_transition": second_transition,
                    "interval_sec": interval_sec,
                    "dry_run_mode": self._dry_run,
                }
                try:
                    write_event(
                        self._store,
                        "s2_oscillation_blocked",
                        block_body,
                        severity="warning",
                    )
                except Exception:  # noqa: BLE001 -- event emit must never crash FSM
                    pass

                if not self._dry_run:
                    block_attempt_body = {
                        "from_state": from_state.value,
                        "to_state": to_state.value,
                        "reason": reason,
                        "version_before": version_before,
                        "version_after": version_before,
                        "succeeded": False,
                        "conflict_reason": "oscillation_blocked",
                    }
                    try:
                        write_event(
                            self._store,
                            "s2_transition_attempt",
                            block_attempt_body,
                            severity="info",
                        )
                    except Exception:  # noqa: BLE001 -- event emit must never crash FSM
                        pass
                    raise S2OscillationBlocked(
                        first_transition=oscillation_first,
                        second_transition=second_transition,
                        interval_sec=interval_sec,
                    )

            rec["current_state"] = to_state.value
            rec["since_ts"] = _utc_now_iso()
            save_state(rec, self._state_path)

            if self._legacy_path is not None:
                try:
                    from iai_mcp.fsm_reconcile import _auto_correct_legacy
                    _auto_correct_legacy(self._legacy_path, to_state.value)
                except Exception:  # noqa: BLE001 -- mirror write is best-effort
                    pass

            self._ring_buffer.append(
                {
                    "from_state": from_state.value,
                    "to_state": to_state.value,
                    "reason": reason,
                    "ts_monotonic": now_mono,
                }
            )
            self.version += 1
            version_after = self.version

            success_body = {
                "from_state": from_state.value,
                "to_state": to_state.value,
                "reason": reason,
                "version_before": version_before,
                "version_after": version_after,
                "succeeded": True,
                "conflict_reason": None,
            }
            try:
                write_event(
                    self._store,
                    "s2_transition_attempt",
                    success_body,
                    severity="info",
                )
            except Exception:  # noqa: BLE001 -- event emit must never crash FSM
                pass

            return to_state

    async def set_crisis_mode(self, value: bool, reason: str) -> None:
        async with self.lock:
            rec: LifecycleStateRecord = load_state(self._state_path)
            rec["crisis_mode"] = bool(value)
            rec["crisis_mode_since_ts"] = _utc_now_iso() if bool(value) else None
            save_state(rec, self._state_path)
            self.version += 1

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import NotRequired, TypedDict

LIFECYCLE_STATE_FILENAME: str = "lifecycle_state.json"
LIFECYCLE_STATE_PATH: Path = Path.home() / ".iai-mcp" / LIFECYCLE_STATE_FILENAME


def lifecycle_state_path(store_root: Path | str | None = None) -> Path:
    """Resolve the lifecycle-state file path under the active store root.

    Single source of truth shared by the daemon writer and every reader so the
    write side and the read side can never diverge. With no explicit
    ``store_root``, the ``IAI_MCP_STORE`` environment variable is honored;
    absent that, the module-level ``LIFECYCLE_STATE_PATH`` is returned as the
    default вЂ” read at call time so it stays the single authoritative default for
    the home-rooted store. For the default store the resolved path is
    byte-identical to ``LIFECYCLE_STATE_PATH``.
    """
    if store_root is not None:
        return Path(store_root) / LIFECYCLE_STATE_FILENAME
    env = os.environ.get("IAI_MCP_STORE")
    if env:
        return Path(env) / LIFECYCLE_STATE_FILENAME
    return LIFECYCLE_STATE_PATH


class LifecycleState(str, Enum):

    WAKE = "WAKE"
    DROWSY = "DROWSY"
    SLEEP = "SLEEP"
    HIBERNATION = "HIBERNATION"


class SleepCycleProgress(TypedDict, total=False):

    last_completed_index: int
    last_completed_step: int
    attempt: int
    last_error: str | None
    started_at: str


class Quarantine(TypedDict):

    until_ts: str
    reason: str
    since_ts: str


class LifecycleStateRecord(TypedDict):

    current_state: str
    since_ts: str
    last_activity_ts: str
    wrapper_event_seq: int
    sleep_cycle_progress: SleepCycleProgress | None
    quarantine: Quarantine | None
    shadow_run: bool
    crisis_mode: bool
    crisis_mode_since_ts: NotRequired[str | None]
    essential_variable_consecutive_breaches: NotRequired[int]
    essential_variable_consecutive_clears: NotRequired[int]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_state() -> LifecycleStateRecord:
    now = _utc_now_iso()
    return {
        "current_state": LifecycleState.WAKE.value,
        "since_ts": now,
        "last_activity_ts": now,
        "wrapper_event_seq": 0,
        "sleep_cycle_progress": None,
        "quarantine": None,
        "shadow_run": False,
        "crisis_mode": False,
        "crisis_mode_since_ts": None,
        "essential_variable_consecutive_breaches": 0,
        "essential_variable_consecutive_clears": 0,
    }


def _validate_record(raw: object) -> LifecycleStateRecord:
    if not isinstance(raw, dict):
        raise ValueError(
            f"lifecycle_state record must be a JSON object, got {type(raw).__name__}"
        )

    required_str_keys = ("current_state", "since_ts", "last_activity_ts")
    for k in required_str_keys:
        v = raw.get(k)
        if not isinstance(v, str) or not v:
            raise ValueError(f"lifecycle_state.{k} must be a non-empty string, got {v!r}")

    state_value = raw["current_state"]
    if state_value not in {s.value for s in LifecycleState}:
        raise ValueError(
            f"lifecycle_state.current_state {state_value!r} is not a valid LifecycleState"
        )

    seq = raw.get("wrapper_event_seq")
    if not isinstance(seq, int) or seq < 0:
        raise ValueError(
            f"lifecycle_state.wrapper_event_seq must be a non-negative int, got {seq!r}"
        )

    shadow = raw.get("shadow_run")
    if not isinstance(shadow, bool):
        raise ValueError(
            f"lifecycle_state.shadow_run must be a bool, got {shadow!r}"
        )

    crisis_mode = raw.get("crisis_mode", False)
    if not isinstance(crisis_mode, bool):
        raise ValueError(
            f"lifecycle_state.crisis_mode must be a bool, got {crisis_mode!r}"
        )
    raw["crisis_mode"] = crisis_mode

    since_ts_value = raw.get("crisis_mode_since_ts", None)
    if since_ts_value is not None and not isinstance(since_ts_value, str):
        raise ValueError(
            f"lifecycle_state.crisis_mode_since_ts must be a string or null, got {since_ts_value!r}"
        )
    raw["crisis_mode_since_ts"] = since_ts_value

    consecutive_breaches = raw.get("essential_variable_consecutive_breaches", 0)
    if not isinstance(consecutive_breaches, int) or consecutive_breaches < 0:
        raise ValueError(
            f"lifecycle_state.essential_variable_consecutive_breaches must be a "
            f"non-negative int, got {consecutive_breaches!r}"
        )
    raw["essential_variable_consecutive_breaches"] = consecutive_breaches

    consecutive_clears = raw.get("essential_variable_consecutive_clears", 0)
    if not isinstance(consecutive_clears, int) or consecutive_clears < 0:
        raise ValueError(
            f"lifecycle_state.essential_variable_consecutive_clears must be a "
            f"non-negative int, got {consecutive_clears!r}"
        )
    raw["essential_variable_consecutive_clears"] = consecutive_clears

    progress = raw.get("sleep_cycle_progress")
    if progress is not None and not isinstance(progress, dict):
        raise ValueError(
            f"lifecycle_state.sleep_cycle_progress must be dict or null, got {progress!r}"
        )

    quarantine = raw.get("quarantine")
    if quarantine is not None:
        if not isinstance(quarantine, dict):
            raise ValueError(
                f"lifecycle_state.quarantine must be dict or null, got {quarantine!r}"
            )
        for k in ("until_ts", "reason", "since_ts"):
            if not isinstance(quarantine.get(k), str):
                raise ValueError(
                    f"lifecycle_state.quarantine.{k} must be string"
                )

    return raw  # type: ignore[return-value]


def load_state(path: Path | None = None) -> LifecycleStateRecord:
    target = path if path is not None else LIFECYCLE_STATE_PATH
    if not target.exists():
        return default_state()
    try:
        raw = json.loads(target.read_text())
    except (OSError, json.JSONDecodeError):
        return default_state()
    try:
        return _validate_record(raw)
    except ValueError:
        return default_state()


def save_state(record: LifecycleStateRecord, path: Path | None = None) -> None:
    target = path if path is not None else LIFECYCLE_STATE_PATH
    _validate_record(record)

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".lifecycle_state.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    replaced = False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, target)
        replaced = True
    finally:
        if not replaced:
            try:
                os.unlink(tmp)
            except OSError:
                pass


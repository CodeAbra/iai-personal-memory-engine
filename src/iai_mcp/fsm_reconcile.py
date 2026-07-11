from __future__ import annotations

import json
from pathlib import Path

_NO_DRIFT_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        ("WAKE", "WAKE"),
        ("SLEEP", "SLEEP"),
        ("SLEEP", "DREAMING"),
        ("DROWSY", "TRANSITIONING"),
    }
)


def _read_canonical(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    value = raw.get("current_state")
    return value if isinstance(value, str) and value else None


def _read_legacy(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    value = raw.get("fsm_state")
    return value if isinstance(value, str) and value else None


_CANONICAL_TO_LEGACY: dict[str, str] = {
    "WAKE": "WAKE",
    "DROWSY": "TRANSITIONING",
    "SLEEP": "SLEEP",
    "HIBERNATION": "SLEEP",
}


def _auto_correct_legacy(legacy_path: Path, canonical_state: str) -> bool:
    import os
    import tempfile

    target_legacy = _CANONICAL_TO_LEGACY.get(canonical_state, "WAKE")
    try:
        raw: dict = {}
        if legacy_path.exists():
            raw = json.loads(legacy_path.read_text())
            if not isinstance(raw, dict):
                raw = {}
    except (OSError, json.JSONDecodeError):
        raw = {}

    raw["fsm_state"] = target_legacy
    raw["fsm_corrected_at"] = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    ).isoformat()

    try:
        fd, tmp = tempfile.mkstemp(dir=str(legacy_path.parent), suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(raw, f)
        os.replace(tmp, str(legacy_path))
        return True
    except OSError:
        return False


def reconcile_fsm_state(
    canonical_path: Path | None = None,
    legacy_path: Path | None = None,
    *,
    auto_correct: bool = False,
) -> dict[str, str | bool | None]:
    if canonical_path is None:
        from iai_mcp.lifecycle_state import LIFECYCLE_STATE_PATH

        canonical_path = LIFECYCLE_STATE_PATH
    if legacy_path is None:
        from iai_mcp.daemon_state import STATE_PATH

        legacy_path = STATE_PATH

    canonical = _read_canonical(canonical_path)
    legacy = _read_legacy(legacy_path)

    if canonical is None or legacy is None:
        drift = False
    elif canonical == "HIBERNATION":
        drift = False
    else:
        drift = (canonical, legacy) not in _NO_DRIFT_PAIRS

    corrected = False
    if drift and auto_correct and canonical is not None:
        corrected = _auto_correct_legacy(legacy_path, canonical)

    return {"canonical": canonical, "legacy": legacy, "drift": drift, "corrected": corrected}

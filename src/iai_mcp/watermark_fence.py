"""Cross-layer staleness fence: source_watermark == derived_watermark.

Compares the episodic sidecar against consolidated_watermark
(lifecycle_state.json) and the session-start pack's leading
source_watermark line. Daemon-independent: no import of the daemon
package, pure stdlib + store_watermark + lifecycle_state.

Status rules:
- derived > source (pack or consolidation ahead of episodic): hard-FAIL,
  independent of any threshold.
- Consolidation: current when equal or within CONSOLIDATION_LAG_WARN_SEC,
  lagging (WARN) beyond it, never_run when no consolidation watermark
  exists yet.
- Pack: any inequality -> lagging. The hour-granularity grace for a
  healthy in-flight pack lives in the session-start shell hook's own
  marker, not here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

CONSOLIDATION_LAG_WARN_SEC: int = 48 * 3600
"""WARN threshold for consolidation lag. Tunable -- never affects the
derived>source hard-FAIL, only the current/lagging boundary."""

_STATUS_CURRENT = "current"
_STATUS_LAGGING = "lagging"
_STATUS_IMPOSSIBLE = "impossible"
_STATUS_NEVER_RUN = "never_run"

# Anchored to the exact leading line format format_payload_as_markdown emits
# and the session-start shell hook's own sed parse matches -- a crafted
# cache cannot inject a fake sentinel anywhere but the leading line.
_PACK_WATERMARK_RE = re.compile(r"^<!-- iai-mcp:source_watermark=(.*) -->$")

# Duplicated literal, not an import of daemon.SESSION_START_CACHE_PATH --
# daemon-independence forbids importing the daemon package here. A known,
# accepted duplication shared with cli/_capture.py's own precache-path
# helper; MUST be kept in sync by hand on any rename.
_DEFAULT_PACK_CACHE_PATH = Path.home() / ".iai-mcp" / ".session-start-payload.cached.md"


@dataclass(frozen=True)
class LayerStatus:
    status: str
    value: "str | None"


@dataclass(frozen=True)
class FenceResult:
    episodic: "str | None"
    pack: LayerStatus
    consolidation: LayerStatus
    ok: bool


def read_episodic_watermark(store_root: "Path | str") -> "str | None":
    """Thin wrapper over store_watermark.read, resolved the same way
    session.py's composer and the sleep pipeline's pre-cycle read do:
    the sidecar lives under <store_root>/hippo, not <store_root> directly."""
    from iai_mcp import store_watermark

    return store_watermark.read(Path(store_root) / "hippo")


def read_consolidation_watermark(store_root: "Path | str | None" = None) -> "str | None":
    from iai_mcp.lifecycle_state import lifecycle_state_path, load_state

    record = load_state(lifecycle_state_path(store_root))
    value = record.get("consolidated_watermark")
    return value if isinstance(value, str) and value else None


def read_pack_watermark(cache_path: "Path | str | None" = None) -> "str | None":
    """Extract the leading sentinel watermark line, anchored to line 1 only
    -- mirrors the session-start shell hook's parse so the two can never
    diverge on what counts as a valid marker."""
    path = Path(cache_path) if cache_path is not None else _DEFAULT_PACK_CACHE_PATH
    try:
        with path.open("r", encoding="utf-8") as fh:
            first_line = fh.readline().rstrip("\n")
    except OSError:
        return None
    match = _PACK_WATERMARK_RE.match(first_line)
    return match.group(1) if match else None


def _normalize_ts(value: "str | None") -> "datetime | None":
    """datetime.fromisoformat normalization -- handles Z / +00:00 /
    microsecond suffix differences so a format-only divergence never flips
    a lagging/impossible verdict (unlike the shell hook's hour-prefix
    string equality, this comparator has datetime available)."""
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _pack_layer_status(
    episodic_dt: "datetime | None", pack: "str | None",
) -> "tuple[str, bool]":
    if not pack:
        return _STATUS_CURRENT, True
    pack_dt = _normalize_ts(pack)
    if pack_dt is None or episodic_dt is None:
        # Malformed/absent source -- degrade to a safe, non-crashing,
        # non-escalating status; cannot rule impossible without a
        # comparable source timestamp.
        return _STATUS_CURRENT, True
    if pack_dt > episodic_dt:
        return _STATUS_IMPOSSIBLE, False
    if pack_dt == episodic_dt:
        return _STATUS_CURRENT, True
    return _STATUS_LAGGING, True


def _consolidation_layer_status(
    episodic_dt: "datetime | None", consolidation: "str | None", warn_lag_sec: float,
) -> "tuple[str, bool]":
    if not consolidation:
        return _STATUS_NEVER_RUN, True
    cons_dt = _normalize_ts(consolidation)
    if cons_dt is None:
        return _STATUS_NEVER_RUN, True
    if episodic_dt is None:
        return _STATUS_CURRENT, True
    if cons_dt > episodic_dt:
        return _STATUS_IMPOSSIBLE, False
    lag_sec = (episodic_dt - cons_dt).total_seconds()
    if lag_sec > warn_lag_sec:
        return _STATUS_LAGGING, True
    return _STATUS_CURRENT, True


def check_fence(
    episodic: "str | None",
    consolidation: "str | None",
    pack: "str | None",
    *,
    now: "datetime | None" = None,
    warn_lag_sec: float = CONSOLIDATION_LAG_WARN_SEC,
) -> FenceResult:
    """now is accepted for API symmetry with other fence-style checks but
    unused -- every comparison here is source-vs-derived, never wall-clock
    relative."""
    del now
    episodic_dt = _normalize_ts(episodic)

    pack_status, pack_ok = _pack_layer_status(episodic_dt, pack)
    consolidation_status, consolidation_ok = _consolidation_layer_status(
        episodic_dt, consolidation, warn_lag_sec,
    )

    return FenceResult(
        episodic=episodic,
        pack=LayerStatus(status=pack_status, value=pack),
        consolidation=LayerStatus(status=consolidation_status, value=consolidation),
        ok=pack_ok and consolidation_ok,
    )


__all__ = [
    "CONSOLIDATION_LAG_WARN_SEC",
    "LayerStatus",
    "FenceResult",
    "read_episodic_watermark",
    "read_consolidation_watermark",
    "read_pack_watermark",
    "check_fence",
]

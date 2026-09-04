from __future__ import annotations

"""In-RAM labile-entry staging for the valence reconsolidation writer.

Retrieval-reconsolidation itself (the labile-window concept, `labile_until`)
is driven end-to-end by `store.reinforce_record(is_retrieval=True)` and the
critic step; this buffer stages per-invocation valence modifications for a
single sleep-step call and persists them via `persist_valence`, then is
discarded -- it is not a second cross-cycle source of truth for labile state.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)

LABILE_WINDOW_SEC = 300
MAX_RECONSOLIDATION_DEPTH = 3
STABILITY_BOOST_ON_RECALL = 0.05
STABILITY_PENALTY_ON_CONTRADICTION = 0.2


@dataclass
class LabileEntry:
    record_id: UUID
    entered_at: float
    recall_context: str
    modifications: list[dict] = field(default_factory=list)
    reconsolidation_count: int = 0


class ReconsolidationBuffer:

    def __init__(self, window_sec: float = LABILE_WINDOW_SEC):
        self._window_sec = window_sec
        self._labile: dict[UUID, LabileEntry] = {}

    def enter_labile(self, record_id: UUID, context: str = "") -> LabileEntry:
        now = time.time()
        if record_id in self._labile:
            existing = self._labile[record_id]
            if now - existing.entered_at < self._window_sec:
                return existing
        entry = LabileEntry(
            record_id=record_id,
            entered_at=now,
            recall_context=context,
        )
        self._labile[record_id] = entry
        return entry

    def is_labile(self, record_id: UUID) -> bool:
        entry = self._labile.get(record_id)
        if entry is None:
            return False
        if time.time() - entry.entered_at > self._window_sec:
            del self._labile[record_id]
            return False
        return True

    def modify_valence(self, record_id: UUID, delta: float, reason: str) -> bool:
        if not self.is_labile(record_id):
            return False
        entry = self._labile[record_id]
        if entry.reconsolidation_count >= MAX_RECONSOLIDATION_DEPTH:
            logger.debug("reconsolidation depth limit reached for %s", record_id)
            return False
        entry.modifications.append({
            "type": "valence",
            "delta": delta,
            "reason": reason,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        entry.reconsolidation_count += 1
        return True

    def modify_confidence(self, record_id: UUID, new_confidence: float, reason: str) -> bool:
        if not self.is_labile(record_id):
            return False
        entry = self._labile[record_id]
        if entry.reconsolidation_count >= MAX_RECONSOLIDATION_DEPTH:
            return False
        entry.modifications.append({
            "type": "confidence",
            "value": new_confidence,
            "reason": reason,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        entry.reconsolidation_count += 1
        return True

    def close_expired(self) -> list[LabileEntry]:
        now = time.time()
        closed: list[LabileEntry] = []
        expired_ids = [
            rid for rid, entry in self._labile.items()
            if now - entry.entered_at > self._window_sec
        ]
        for rid in expired_ids:
            closed.append(self._labile.pop(rid))
        return closed

    def pending_count(self) -> int:
        self.close_expired()
        return len(self._labile)

    def get_modifications(self, record_id: UUID) -> list[dict]:
        entry = self._labile.get(record_id)
        if entry is None:
            return []
        return entry.modifications

    def persist_valence(self, store: Any) -> tuple[int, int]:
        """Writes each staged valence delta through `store.raise_valence`,
        the single-column, kill-switch-honoring, [0.0, 1.0]-clamping write
        primitive. Returns ``(writes, saturated)``: `writes` counts records
        actually written (a no-op record_id, e.g. deleted since staging,
        does not count); `saturated` counts records whose stored valence was
        already at the 1.0 ceiling before this call, so `raise_valence`
        correctly declined the write -- distinct from a write that failed
        for any other reason. A per-record store error is swallowed so one
        bad row never costs the rest of the pool.

        Cross-process exclusivity on `store` is guaranteed upstream: opening
        a `MemoryStore` under `AccessMode.EXCLUSIVE` refuses a second
        process before any code in this loop runs. Within that single
        holder, this loop and `raise_valence`'s own internal read span two
        separate `store.get()` calls, so a second concurrent call on the
        same record_id inside the same process can still race a
        lost-increment; `raise_valence`'s monotonic-raise guard bounds that
        to a redundant same-value write, never a silent lower."""
        from iai_mcp.exceptions import StoreError

        writes = 0
        saturated = 0
        for record_id, entry in self._labile.items():
            deltas = [
                float(m["delta"])
                for m in entry.modifications
                if m.get("type") == "valence"
            ]
            if not deltas:
                continue
            try:
                current = store.get(record_id)
                if current is None:
                    continue
                already_saturated = float(current.valence) >= 1.0
                new_value = float(current.valence) + sum(deltas)
                if store.raise_valence(record_id, new_value):
                    writes += 1
                elif already_saturated:
                    saturated += 1
            except (OSError, ValueError, RuntimeError, StoreError) as exc:
                logger.debug(
                    "persist_valence per-record write failed for %s: %s",
                    record_id, exc,
                )
        return writes, saturated


def compute_stability_update(
    current_stability: float,
    was_recalled: bool,
    was_contradicted: bool,
) -> float:
    new = current_stability
    if was_recalled:
        new = min(1.0, new + STABILITY_BOOST_ON_RECALL)
    if was_contradicted:
        new = max(0.0, new - STABILITY_PENALTY_ON_CONTRADICTION)
    return new

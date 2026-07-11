from __future__ import annotations

"""In-RAM labile-entry tracking for retrieval-reconsolidation.

Retrieval-reconsolidation is already shipped end-to-end via
`store.reinforce_record(is_retrieval=True)` (sets `labile_until`, a SQL
column) followed by the nightly `step_reconsolidation` critic. This module's
`ReconsolidationBuffer` is intentionally not wired to that path: it would
duplicate the same labile-window concept in a second, divergent (in-RAM,
non-persistent) representation with no identified correctness gap. Kept as
dead code deliberately, not by oversight.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
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

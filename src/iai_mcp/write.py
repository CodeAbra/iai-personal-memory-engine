from __future__ import annotations

from uuid import UUID

import numpy as np

from iai_mcp.types import MemoryRecord

VIGILANCE_RHO = 0.95


def cosine(a: list[float], b: list[float]) -> float:
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


def apply_art_gate(
    existing_records: list[MemoryRecord],
    new_record: MemoryRecord,
    rho: float = VIGILANCE_RHO,
) -> tuple[str, UUID]:
    best_sim: float = -1.0
    best_id: UUID | None = None
    for rec in existing_records:
        if rec.never_merge:
            continue
        sim = cosine(new_record.embedding, rec.embedding)
        if sim > best_sim:
            best_sim = sim
            best_id = rec.id
    if best_id is not None and best_sim >= rho:
        return ("merge", best_id)
    return ("create", new_record.id)

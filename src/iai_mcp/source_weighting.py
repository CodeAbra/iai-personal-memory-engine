from __future__ import annotations

import math
import os
from typing import Any


DEFAULT_SOURCE_WEIGHT_FACTOR = 1.05
MAX_SOURCE_WEIGHT_FACTOR = 1.25


def source_weight_factor() -> float:
    """Return the bounded source boost; 1.0 disables source weighting."""
    raw = os.environ.get("IAI_MCP_SOURCE_WEIGHT_FACTOR")
    if raw is None:
        return DEFAULT_SOURCE_WEIGHT_FACTOR
    try:
        factor = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_SOURCE_WEIGHT_FACTOR
    if not math.isfinite(factor):
        return DEFAULT_SOURCE_WEIGHT_FACTOR
    return min(MAX_SOURCE_WEIGHT_FACTOR, max(1.0, factor))


def source_weighted_score(
    score: float,
    record: Any,
    factor: float | None = None,
) -> float:
    """Soft-boost curated documents and durable semantic digests."""
    if factor is None:
        factor = source_weight_factor()
    if factor == 1.0:
        return float(score)
    tags = tuple(str(tag) for tag in (getattr(record, "tags", None) or ()))
    durable = (
        any(tag.startswith("doc:") for tag in tags)
        or (
            getattr(record, "tier", "episodic") == "semantic"
            and "schema" not in tags
            and not any(tag.startswith("pattern:") for tag in tags)
        )
    )
    if not durable:
        return float(score)
    score = float(score)
    return score * factor if score >= 0.0 else score / factor

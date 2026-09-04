"""Process-cached tuned retrieval weight, read from `_hippo_meta`.

Cached as a per-store-instance attribute -- the tuned weight changes at most
nightly, so per-store-generation lifetime is correct; the recall hot path
never pays the AES decrypt. `invalidate(store)` forces a re-read after a
persist (the differential gate's and nightly re-tune's arm-point).
"""
from __future__ import annotations

from typing import Any

_CACHE_ATTR: str = "_retrieval_weight_cache"


def load(store: Any) -> dict[str, float]:
    cached = getattr(store, _CACHE_ATTR, None)
    if cached is not None:
        return cached
    from iai_mcp.lilli.profile.retrieval_tuning import (
        DEFAULT_W_COSINE,
        PROD_W_COSINE_MAX,
        PROD_W_COSINE_MIN,
        load_retrieval_weights_state,
    )
    try:
        weights = load_retrieval_weights_state(store)
    except Exception:  # noqa: BLE001 -- the recall hot path must never raise on this read
        weights = {"W_COSINE": DEFAULT_W_COSINE}
    # Clamp here, not upstream: load_retrieval_weights_state returns the raw
    # blob unclamped by contract, so an out-of-range value must never reach
    # the scorer unclamped.
    w = weights.get("W_COSINE", DEFAULT_W_COSINE)
    if not isinstance(w, (int, float)) or isinstance(w, bool):
        w = DEFAULT_W_COSINE
    weights = {"W_COSINE": max(PROD_W_COSINE_MIN, min(PROD_W_COSINE_MAX, float(w)))}
    try:
        setattr(store, _CACHE_ATTR, weights)
    except (AttributeError, TypeError):
        pass
    return weights


def invalidate(store: Any) -> None:
    try:
        if hasattr(store, _CACHE_ATTR):
            delattr(store, _CACHE_ATTR)
    except (AttributeError, TypeError):
        pass

"""Shared flag: while set, a recursive recall (claim_check's probe)
suppresses its own side effects; the read/rank/response path is unchanged."""
from __future__ import annotations

import contextvars

recall_suppressed: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "recall_suppressed", default=False
)

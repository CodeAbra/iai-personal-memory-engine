"""Shared normalization for optional model-attribution labels."""

from __future__ import annotations

MODEL_LABEL_MAX_CHARS = 80


def normalize_model(value: object) -> str | None:
    """Return a bounded printable model label, or ``None`` for unknown input."""
    if not isinstance(value, str):
        return None
    normalized = "".join(
        char for char in value if char.isalnum() or char in " ._:/+-"
    )
    normalized = " ".join(normalized.split())
    return normalized[:MODEL_LABEL_MAX_CHARS] or None

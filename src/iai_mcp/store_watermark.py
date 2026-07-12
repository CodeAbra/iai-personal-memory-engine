"""Store-advance watermark sidecar.

Writers stamp ``<store_root>/.max-created-at`` after a successful record
write; per-turn hooks read the sidecar instead of opening the store. Hooks
must never parse the store file itself — its on-disk engine format is not
part of any hook contract.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

WATERMARK_NAME = ".max-created-at"


def emit(store_root: "Path | str", ts_iso: str) -> None:
    """Monotone, atomic, fail-soft — must never raise into a write path."""
    try:
        if not ts_iso:
            return
        root = Path(store_root)
        prev = read(root)
        if prev is not None and prev >= ts_iso:
            return
        root.mkdir(parents=True, exist_ok=True)
        path = root / WATERMARK_NAME
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
        tmp.write_text(ts_iso, encoding="utf-8")
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001 -- sidecar is advisory, writes must not fail
        logger.debug("store watermark emit failed: %s", exc)


def read(store_root: "Path | str") -> "str | None":
    try:
        raw = (Path(store_root) / WATERMARK_NAME).read_text(encoding="utf-8").strip()
        return raw or None
    except OSError:
        return None

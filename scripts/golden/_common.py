"""Shared helper for emitting golden fixtures (raw little-endian .bin + JSON manifest)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def write_golden(
    path_stem: Path,
    arr_bytes: bytes,
    dtype: str,
    shape: list[int],
    byte_order: str,
    subsystem: str,
    source: str,
    generated_by: str,
) -> str:
    """Write a golden pair: `<stem>.bin` (raw bytes) and `<stem>.json` (manifest).

    The sha256 in the manifest is computed over the exact bytes written to
    `<stem>.bin`. Returns the hex digest.
    """
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)

    sha256 = hashlib.sha256(arr_bytes).hexdigest()

    bin_path = path_stem.with_suffix(".bin")
    bin_path.write_bytes(arr_bytes)

    manifest = {
        "name": path_stem.name,
        "subsystem": subsystem,
        "format": "raw_bytes",
        "payload": None,
        "dtype": dtype,
        "shape": shape,
        "byte_order": byte_order,
        "sha256": sha256,
        "source": source,
        "generated_by": generated_by,
    }

    json_path = path_stem.with_suffix(".json")
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return sha256

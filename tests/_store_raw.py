"""Driver-aware raw open and test-only helpers for at-rest assertions.

Returns a stdlib sqlite3 connection for a legacy-format backing file, or the
engine's crypto-blind raw DB-API connection for a native-format one. The
engine stores ciphertext BLOBs, so SELECT on encrypted columns returns the
same wire prefix the harnesses assert today.
"""
from __future__ import annotations

import sqlite3
import struct
from pathlib import Path


def open_store_raw(path: str | Path):
    """Open the backing store file for raw at-rest reads/writes.

    The backend is chosen from the on-disk header of ``path`` itself (mirroring
    production's ``_resolve_effective_driver``) rather than from the ambient
    environment, because this helper is always handed a concrete path to a file
    that already exists. On a legacy header, returns sqlite3.connect(path). On
    a native header, returns the engine raw connection for the file (the live
    registered connection when HippoDB is open, else a temporary reopen). The
    returned object supports execute()/commit()/close() and assignment of
    row_factory.

    Raises RuntimeError if the resolved driver is the native engine but no
    engine connection can be obtained for the path (file absent or genuine
    sqlite file) — fail loud rather than silently fall through to a broken
    sqlite3.connect.
    """
    from iai_mcp.hippo._db import _resolve_effective_driver  # noqa: PLC0415

    driver = _resolve_effective_driver(str(path))
    if driver != "lilli":
        return sqlite3.connect(str(path))

    from iai_mcp.lillibrain.connection import get_lilli_raw_conn  # noqa: PLC0415

    conn = get_lilli_raw_conn(str(path), allow_writer=True)
    if conn is None:
        raise RuntimeError(
            f"lilli driver active but no engine connection for {path!r}"
        )
    return conn


def simulate_daemon_reembed(
    store_root: "Path | str",
    *,
    text_fragment: str,
    embedding: "list[float]",
) -> None:
    """Test helper: UPDATE the embedding for records whose literal_surface matches
    text_fragment, simulating what the daemon's deferred re-embed pass would do.

    Used by two-phase direct-write tests that need to exercise the post-embed
    recall path without starting a real daemon. Kept here (not in production
    source) because it uses LIKE-without-LIMIT and exists solely for test
    scenarios.
    """
    root = Path(store_root)
    db_path = root / "hippo" / "brain.sqlite3"
    blob = struct.pack(f"<{len(embedding)}f", *embedding)
    from iai_mcp.hippo._raw_open import open_store_conn  # noqa: PLC0415
    _eng = open_store_conn(db_path, allow_writer=True)
    if _eng is not None:
        conn = _eng
    else:
        conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(
            "UPDATE records SET embedding = ?, embedding_pending = 0"
            " WHERE literal_surface LIKE ?",
            (blob, f"%{text_fragment}%"),
        )
        conn.commit()
    finally:
        conn.close()

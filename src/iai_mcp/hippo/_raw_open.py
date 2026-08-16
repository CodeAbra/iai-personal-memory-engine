"""Driver-aware raw open for production direct-access store paths.

Production analog of tests/_store_raw.open_store_raw, but with a deliberately
more conservative shape: returns a None sentinel on the default driver so each
production caller keeps its existing sqlite3.connect(...) line byte-for-byte
(the production sites carry heterogeneous stdlib kwargs — isolation_level=None,
timeout=2.0, uri=True — that differ per site; centralizing them would risk
silent kwarg drift). On the lilli driver: strips any file:...?mode=ro URI to a
bare filesystem path and returns the engine's crypto-blind raw connection via
get_lilli_raw_conn.

The engine stores ciphertext BLOBs; decryption is the caller's responsibility,
above this seam — no key crosses into the engine, and this factory takes no key
or key-provider parameter.
"""
from __future__ import annotations

import os
import re
from pathlib import Path


_URI_RE = re.compile(r"^file:(?P<path>[^?]*)")


def _bare_path(path: str) -> str:
    """Strip a sqlite file:...?mode=ro URI down to a bare filesystem path.

    The engine raw-conn factory performs Path(path).exists(); a URI string is
    not a valid filesystem path and that probe returns False, causing a
    spurious RuntimeError. Callers that pass a URI form must reduce it to a
    bare path before reaching the lilli branch.
    """
    m = _URI_RE.match(path)
    return m.group("path") if m else path


def open_store_conn(
    path: str | Path, *, read_only: bool = False, allow_writer: bool = False
):
    """Open the backing store for raw direct access, driver-aware.

    Returns None on the default driver: the caller keeps its existing
    sqlite3.connect(...) call verbatim — no kwargs are altered. Returns the
    engine raw connection on the lilli driver (live registered connection when
    HippoDB is open, else a temporary reopen for closed stores).

    read_only is enforced at the adapter level on the lilli driver: the
    returned adapter's execute() intercepts PRAGMA query_only=ON and rejects
    write SQL without mutating the shared writer connection's _query_only flag.
    This means a RO adapter on a live store cannot brick the writer's write path.
    The recall/export paths it serves are structurally SELECT-only.

    Raises RuntimeError if the lilli driver is active but get_lilli_raw_conn
    returns None (path absent or is a genuine sqlite3 file) — fail loud rather
    than silently hand back a broken or mis-typed connection.

    The writer branch (read_only=False) bypasses the read-only pool's
    generation fence, so a bare default call raises errors.DatabaseError; pass
    allow_writer=True to opt in explicitly — see
    iai_mcp.lillibrain.connection.get_lilli_raw_conn for the fence contract.

    Driver selection follows the on-disk file format (env only for a fresh or
    absent file), so a process without ``LILLI_STORAGE_DRIVER`` — e.g. the
    user-shell ``iai recall`` daemon-down raw path — still reads a native-engine
    store through the engine rather than returning the stdlib sentinel and
    handing the caller a lilli-format file to ``sqlite3.connect``.
    """
    from iai_mcp.hippo._db import _resolve_effective_driver  # noqa: PLC0415

    driver = _resolve_effective_driver(_bare_path(str(path)))
    if driver != "lilli":
        return None  # sentinel: caller keeps its existing sqlite3.connect verbatim

    from iai_mcp.lillibrain.connection import get_lilli_raw_conn  # noqa: PLC0415

    bare = _bare_path(str(path))
    conn = get_lilli_raw_conn(bare, read_only=read_only, allow_writer=allow_writer)
    if conn is None:
        raise RuntimeError(
            f"lilli driver active but no engine connection for {path!r}"
        )
    return conn

"""Spawn-context worker for the OPTIMIZE_HIPPO hnswlib index rebuild.

Receives the SQLite db path and build parameters from the parent over a Pipe,
reads all active embedding rows via its own read-only connection, builds a
fresh hnswlib index in the child's address space, saves it to the designated
temp path, and reports completion.  The OS reclaims the worker's address space
on exit so the numpy/hnswlib allocation never touches the daemon's resident set.

AES-key fence: the embedding column in the records table is stored as raw
float32 BLOB (not AES-encrypted).  The worker opens only a read-only SQLite
connection using the plain file path — no HippoDB, no crypto module, no key
material crosses the process boundary.

Top-level import surface: only `from __future__ import annotations`.  All
compute-side imports are inside the worker function body so the spawn-context
child's import does not trigger daemon-side modules at module-level.
"""

from __future__ import annotations


def _worker_entry(conn) -> None:  # noqa: ANN001
    """Build hnswlib index from SQLite and save to temp path.

    Protocol:
      Parent sends:  ("params", {"db_path": str, "tmp_path": str,
                                  "embed_dim": int, "hnsw_m": int,
                                  "ef_construction": int, "ef": int,
                                  "capacity": int})
      Worker sends:  ("done", {"rebuilt_count": int}) on success
                     ("error", str) on failure
    """
    from iai_mcp import _sqlite_stdlib
    import struct
    import sys

    from iai_mcp.hippo import _vecindex
    import numpy as np

    try:
        kind, payload = conn.recv()
        if kind != "params":
            conn.send(("error", f"expected 'params', got {kind!r}"))
            return
    except (BrokenPipeError, EOFError) as exc:
        return  # parent died before sending params — exit silently

    db_path: str = payload["db_path"]
    tmp_path: str = payload["tmp_path"]
    embed_dim: int = int(payload["embed_dim"])
    hnsw_m: int = int(payload["hnsw_m"])
    ef_construction: int = int(payload["ef_construction"])
    ef: int = int(payload["ef"])
    capacity: int = int(payload["capacity"])

    try:
        from iai_mcp.hippo._db import _resolve_effective_driver  # noqa: PLC0415

        driver = _resolve_effective_driver(db_path)
        if driver == "lilli":
            from iai_mcp.hippo._raw_open import _bare_path  # noqa: PLC0415
            from iai_mcp.lillibrain.connection import get_lilli_raw_conn
            ro_conn = get_lilli_raw_conn(_bare_path(db_path), read_only=True)
            if ro_conn is None:
                conn.send(("error", f"lilli driver active but no engine connection for {db_path!r}"))
                return
        else:
            ro_conn = _sqlite_stdlib.connect(
                f"file:{db_path}?mode=ro",
                uri=True,
                check_same_thread=False,
            )
            ro_conn.row_factory = _sqlite_stdlib.Row
        try:
            rows = ro_conn.execute(
                "SELECT vec_label, embedding FROM records"
                " WHERE tombstoned_at IS NULL"
                " AND COALESCE(embedding_pending, 0) = 0"
                " ORDER BY vec_label"
            ).fetchall()
        finally:
            ro_conn.close()

        n = len(rows)
        cap = max(capacity, n * 2)

        idx = _vecindex.Index(space="cosine", dim=embed_dim)
        idx.init_index(
            max_elements=cap,
            ef_construction=ef_construction,
            M=hnsw_m,
            allow_replace_deleted=True,
        )
        idx.set_ef(ef)
        idx.set_num_threads(1)

        if n > 0:
            vecs = np.stack([
                np.frombuffer(bytes(row["embedding"]), dtype=np.float32)
                for row in rows
            ])
            labels = np.array([int(row["vec_label"]) for row in rows], dtype=np.int64)
            idx.add_items(vecs, labels)

        idx.save_index(tmp_path)
        conn.send(("done", {"rebuilt_count": n}))

    except Exception as exc:  # noqa: BLE001
        try:
            conn.send(("error", f"{type(exc).__name__}: {exc!s}"[:500]))
        except (BrokenPipeError, OSError):
            pass
        sys.exit(1)
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass

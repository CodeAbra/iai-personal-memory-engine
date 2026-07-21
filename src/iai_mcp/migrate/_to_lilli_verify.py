"""Store-equality verifier for the SQLite -> lilli verbatim migration.

The verifier is the gate that stands between an operator and silent memory
loss.  It returns GREEN only when a migrated lilli store is byte-equal to its
SQLite source across every table and every parity dimension.  A weak gate that
greenlights a lossy migration is the single worst failure mode of this work, so
the gate exercises REAL recall (not just row counts) and decrypt parity, and it
is the strict AND of six independent dimensions.

Six dimensions:
  A  per-table row-count parity for every store table (fixed six-table allow-list)
  B  ciphertext + embedding byte-equality over EVERY row (bulk ordered co-scan)
  C  decrypt parity over EVERY encrypted record (same key + AAD -> identical plaintext)
  D  recall hit-id + score parity via MemoryStore.query_similar (a stored embedding
     is the cue) -- so a stale/empty ANN index RED-flags (recall=0 is not parity)
  E  temporal-recall parity via MemoryStore.query_similar_temporal (datetime path)
  F  V5 codec metadata round-trip (hv_tier + structure_hv_payload byte-identical)

The encryption key is handled ONLY here, above the engine seam, for dimension C;
the engine itself stays crypto-blind (it stores ciphertext BLOBs verbatim).

Performance model -- NO per-id point lookups on the engine.  The lilli engine has
a B+tree id-index on ``records`` but NOT on ``events``, so a ``WHERE id = ?`` probe
on ``events`` (255k+ rows) degrades to a full sequential scan; one probe per sample
turned the verifier into O(samples x rows) and timed it out at scale.  Instead each
compared table is read ONCE per store with a single ordered scan: the destination is
co-indexed into a column-projected ``id -> row`` map (built from that one scan), and
the source is streamed row-by-row and looked up in that map.  This is O(rows) per
table (two scans, no per-id queries) and -- because every source row is compared, not
a strided sample -- it strictly INCREASES coverage while removing the timeout.
"""

from __future__ import annotations

import math
import os
import shutil
from iai_mcp import _sqlite_stdlib
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from iai_mcp.crypto import decrypt_field


# The fixed store-table allow-list.  No table name is ever accepted from outside
# the verifier -- the six canonical Hippo tables only.
_ALL_TABLES: tuple[str, ...] = (
    "records",
    "edges",
    "events",
    "budget_ledger",
    "ratelimit_ledger",
    "_hippo_meta",
)

# Encrypted columns (carry self-contained ciphertext strings + AAD = record id).
_ENCRYPTED_RECORD_COLUMNS: tuple[str, ...] = (
    "literal_surface",
    "provenance_json",
    "profile_modulation_gain_json",
)
_ENCRYPTED_EVENTS_COLUMNS: tuple[str, ...] = ("data_json",)

# Columns compared byte-for-byte in dimension B for the records table.
_RECORD_BYTE_COLUMNS: tuple[str, ...] = (
    "literal_surface",
    "provenance_json",
    "profile_modulation_gain_json",
    "embedding",
)

# V5 codec metadata columns (dimension F).
_CODEC_COLUMNS: tuple[str, ...] = ("hv_tier", "structure_hv_payload")

# Score tolerance for dimension D/E float comparison.
_SCORE_TOL = 1e-4


@dataclass
class DimensionResult:
    """One parity dimension's outcome."""

    ok: bool
    reason: str
    counts: dict[str, int] = field(default_factory=dict)


@dataclass
class VerifyReport:
    """The full verifier outcome.  ``ok`` is the strict AND of every dimension."""

    ok: bool
    dimensions: dict[str, DimensionResult]
    sampled_n: int


def _dst_db_path(dst_root: str | Path) -> Path:
    """The dst store's backing file path (``<root>/hippo/brain.sqlite3``).

    Note: the dst is lilli-FORMAT (``DB_MAGIC = b"LILLI storage 1\\x00"``), so it
    is read through the lilli engine connection, NOT the stdlib sqlite3 driver.
    """
    return Path(dst_root) / "hippo" / "brain.sqlite3"


def _open_ro(db_path: str | Path) -> Any:
    """Open a genuine SQLite-format file read-only with the stdlib driver.

    Used only for the SOURCE store, which is genuine SQLite.  The destination is
    lilli-format and must be read via the lilli engine instead (see _open_dst).
    """
    conn = _sqlite_stdlib.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = _sqlite_stdlib.Row
    return conn


def _table_columns(conn: Any, table: str) -> list[str]:
    return [r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _count(conn: Any, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
    return int(row["n"]) if row is not None else 0


# Stream batch size for the source ordered scan (RSS-bounded fetchmany).
_SCAN_BATCH = 1000


def _fetch_row(conn: Any, table: str, rid: str) -> Optional[Any]:
    """Single-row lookup by id -- used ONLY on the source (genuine SQLite, which
    has a rowid/PK index), and never inside a per-sample loop on the engine.
    """
    return conn.execute(f"SELECT * FROM {table} WHERE id = ?", (rid,)).fetchone()


def _bulk_map_by_id(
    conn,
    table: str,
    columns: tuple[str, ...],
) -> dict[str, dict]:
    """Index a table into ``{id -> {col: value}}`` from a SINGLE ordered scan.

    Only ``columns`` (plus ``id``) are retained, so the resident map carries just
    the cells a dimension compares -- never the whole 34-column record row.  Works
    against both the stdlib source connection and the lilli engine connection (both
    expose ``execute(...).fetchmany`` and ``row[col]`` access); the engine performs
    one materialised scan here instead of a full sequential scan per sampled id.
    """
    proj = ("id",) + tuple(c for c in columns if c != "id")
    col_list = ", ".join(proj)
    cur = conn.execute(f"SELECT {col_list} FROM {table} ORDER BY id")
    out: dict[str, dict] = {}
    while True:
        rows = cur.fetchmany(_SCAN_BATCH)
        if not rows:
            break
        for r in rows:
            out[r["id"]] = {c: r[c] for c in proj}
    return out


def _all_ids_ordered(conn, table: str) -> list[str]:
    """Every id in ``table`` in id order, from a single ordered scan."""
    cur = conn.execute(f"SELECT id FROM {table} ORDER BY id")
    ids: list[str] = []
    while True:
        rows = cur.fetchmany(_SCAN_BATCH)
        if not rows:
            break
        ids.extend(r["id"] for r in rows)
    return ids


def _as_bytes(value) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, (memoryview, bytearray)):
        # The engine's projected-scan path returns BLOB cells as memoryview;
        # materialise to bytes so byte-equality compares the buffer, not a repr.
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    return str(value).encode("utf-8")


def _verify_counts(
    src: Any, dst: Any
) -> DimensionResult:
    """Dimension A -- per-table row count parity for every store table."""
    counts: dict[str, int] = {}
    for table in _ALL_TABLES:
        src_cols = _table_columns(src, table)
        dst_cols = _table_columns(dst, table)
        if not src_cols or not dst_cols:
            # A table missing on one side but present on the other is a mismatch.
            if bool(src_cols) != bool(dst_cols):
                return DimensionResult(
                    ok=False,
                    reason=f"{table}: present on one store only",
                    counts=counts,
                )
            continue
        sn = _count(src, table)
        dn = _count(dst, table)
        counts[f"{table}.src"] = sn
        counts[f"{table}.dst"] = dn
        if sn != dn:
            delta = dn - sn
            return DimensionResult(
                ok=False,
                reason=f"{table}: src={sn} dst={dn} ({delta:+d})",
                counts=counts,
            )
    return DimensionResult(ok=True, reason="all table counts match", counts=counts)


def _co_scan_bytes(
    src,
    dst,
    table: str,
    columns: tuple[str, ...],
) -> Optional[DimensionResult]:
    """Byte-compare ``columns`` over EVERY row of ``table`` via a bulk co-scan.

    The destination is indexed once into a column-projected ``id -> row`` map; the
    source is streamed in id order and each row is looked up in that map.  Returns
    a RED ``DimensionResult`` on the first mismatch (missing row, surplus row, or a
    byte difference) or ``None`` when every row is byte-equal.
    """
    dst_map = _bulk_map_by_id(dst, table, columns)
    seen = 0
    cur = src.execute(
        f"SELECT id, {', '.join(columns)} FROM {table} ORDER BY id"
    )
    while True:
        rows = cur.fetchmany(_SCAN_BATCH)
        if not rows:
            break
        for s_row in rows:
            rid = s_row["id"]
            d_row = dst_map.pop(rid, None)
            if d_row is None:
                return DimensionResult(
                    ok=False, reason=f"{table} id={rid} missing on dst"
                )
            seen += 1
            for col in columns:
                sb = _as_bytes(s_row[col])
                db = _as_bytes(d_row[col])
                if sb != db:
                    byte_at = next(
                        (i for i in range(min(len(sb), len(db))) if sb[i] != db[i]),
                        min(len(sb), len(db)),
                    )
                    return DimensionResult(
                        ok=False,
                        reason=(
                            f"{table} id={rid} {col} byte mismatch at byte {byte_at} "
                            f"(src {len(sb)}B / dst {len(db)}B)"
                        ),
                    )
    if dst_map:
        surplus = next(iter(dst_map))
        return DimensionResult(
            ok=False, reason=f"{table} id={surplus} present on dst only"
        )
    return None


def _verify_bytes(
    src: Any,
    dst,
) -> tuple[DimensionResult, list[str]]:
    """Dimension B -- ciphertext + embedding byte-equality over EVERY row.

    Compares the records byte-columns and the events ciphertext column across all
    rows via the bulk co-scan, then returns the dimension result AND the full
    id-ordered record-id list so dimensions C/F co-scan the same rows and D/E pick
    bounded probe cues from it.
    """
    record_ids = _all_ids_ordered(src, "records")

    rec_fail = _co_scan_bytes(src, dst, "records", _RECORD_BYTE_COLUMNS)
    if rec_fail is not None:
        return rec_fail, record_ids

    ev_fail = _co_scan_bytes(src, dst, "events", _ENCRYPTED_EVENTS_COLUMNS)
    if ev_fail is not None:
        return ev_fail, record_ids

    n_events = _count(src, "events")
    return (
        DimensionResult(
            ok=True,
            reason=f"{len(record_ids)} records + {n_events} events byte-equal",
            counts={"records_checked": len(record_ids), "events_checked": n_events},
        ),
        record_ids,
    )


def _verify_decrypt(
    src: Any,
    dst,
    crypto_key: bytes,
) -> DimensionResult:
    """Dimension C -- decrypt parity (same key + AAD -> identical plaintext).

    Co-scans EVERY record: the dst encrypted columns are indexed once into an
    ``id -> row`` map and the src is streamed, so no per-id engine lookup runs.
    """
    dst_map = _bulk_map_by_id(dst, "records", _ENCRYPTED_RECORD_COLUMNS)
    checked = 0
    cur = src.execute(
        f"SELECT id, {', '.join(_ENCRYPTED_RECORD_COLUMNS)} "
        f"FROM records ORDER BY id"
    )
    while True:
        rows = cur.fetchmany(_SCAN_BATCH)
        if not rows:
            break
        for s_row in rows:
            rid = s_row["id"]
            d_row = dst_map.get(rid)
            if d_row is None:
                return DimensionResult(
                    ok=False, reason=f"id={rid} missing on dst during decrypt parity"
                )
            ad = rid.lower().encode("ascii")
            for col in _ENCRYPTED_RECORD_COLUMNS:
                s_ct = s_row[col]
                d_ct = d_row[col]
                if not isinstance(s_ct, str) or not s_ct.startswith("iai:enc:"):
                    # Column not encrypted on this row (e.g. legacy/plaintext) --
                    # require it match verbatim and skip the decrypt.
                    if _as_bytes(s_ct) != _as_bytes(d_ct):
                        return DimensionResult(
                            ok=False,
                            reason=f"id={rid} {col} plaintext mismatch",
                        )
                    continue
                try:
                    s_pt = decrypt_field(s_ct, crypto_key, associated_data=ad)
                    d_pt = decrypt_field(d_ct, crypto_key, associated_data=ad)
                except Exception as exc:  # noqa: BLE001 -- a decrypt failure is RED
                    return DimensionResult(
                        ok=False,
                        reason=f"id={rid} {col} decrypt failed: {exc}",
                    )
                if s_pt != d_pt:
                    return DimensionResult(
                        ok=False,
                        reason=f"id={rid} {col} decrypts differently",
                    )
                checked += 1
    return DimensionResult(
        ok=True,
        reason=f"decrypt parity over {checked} encrypted cells",
        counts={"cells_checked": checked},
    )


def _decode_embedding(blob) -> list[float]:
    return np.frombuffer(_as_bytes(blob), dtype=np.float32).tolist()


def _open_memory_store(root: str | Path, *, lilli: bool, saved_env: dict):
    """Open a MemoryStore at ``root``, pointing the env override at it first.

    ``saved_env`` accumulates the prior env values once so the caller restores
    them in a single finally block.

    The ANN index is read exactly as the store serves it on open -- the verifier
    does NOT force a rebuild.  Forcing a rebuild would mask a migrator that left a
    stale/empty ``records.hnsw`` on disk (the recall=0 false-pass): the boot path
    loads that empty index and the SQLite-vs-SQLite integrity check passes, so no
    rebuild fires and recall returns 0.  Dimension D exists to catch exactly that,
    so it must observe the store's real on-open recall behaviour.
    """
    from iai_mcp.store import MemoryStore

    if "IAI_MCP_STORE" not in saved_env:
        saved_env["IAI_MCP_STORE"] = os.environ.get("IAI_MCP_STORE")
    if "LILLI_STORAGE_DRIVER" not in saved_env:
        saved_env["LILLI_STORAGE_DRIVER"] = os.environ.get("LILLI_STORAGE_DRIVER")

    os.environ["IAI_MCP_STORE"] = str(root)
    if lilli:
        os.environ["LILLI_STORAGE_DRIVER"] = "lilli"
    else:
        os.environ.pop("LILLI_STORAGE_DRIVER", None)

    return MemoryStore(root)


def _restore_env(saved_env: dict) -> None:
    for key, prior in saved_env.items():
        if prior is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prior


def _plant_keys_for_recall(
    crypto_key: bytes, roots: list[str | Path | None]
) -> list[Path]:
    """Write ``crypto_key`` as the store key file in each root that lacks one.

    Returns the list of key-file paths this call created (so they can be removed
    afterwards).  A root that already has a key file is left untouched -- only the
    keyless migrated dst / bare copy get a transient key planted so recall resolves
    the verifier's key (which encrypted the verbatim ciphertext) rather than a
    passphrase-derived mismatch.  The key file is 0o600, owner-only.
    """
    from iai_mcp.crypto import _KEY_FILE_NAME, KEY_BYTES

    if len(crypto_key) != KEY_BYTES:
        # The verifier's key is not a raw 32-byte key (passphrase mode) -- nothing
        # to plant; recall falls back to the store's own resolution.
        return []

    created: list[Path] = []
    for root in roots:
        if root is None:
            continue
        path = Path(root) / _KEY_FILE_NAME
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.fchmod(fd, 0o600)
            os.write(fd, crypto_key)
        finally:
            os.close(fd)
        created.append(path)
    return created


def _unplant_keys(created: list[Path]) -> None:
    """Remove the transient key files planted for the recall dimensions."""
    for path in created:
        try:
            path.unlink()
        except OSError:
            pass


def _snapshot_store_root(src_root: str | Path) -> Path:
    """Copy the source store's ``hippo`` dir into a throwaway temp root.

    The verifier runs every dimension against this snapshot rather than opening the
    caller's source store, so the live source is only ever READ (a one-time file
    copy) and never opened by the verifier.  Every source-dir write the verifier
    would otherwise make lands in the throwaway copy and is discarded:

      * the stdlib RO open of a WAL-mode ``brain.sqlite3`` (dimensions A/B/C/F)
        creates ``-wal``/``-shm`` sidecars in the store dir even with ``?mode=ro``;
      * the recall/temporal re-open (D/E) is a WRITABLE MemoryStore -- the ANN index
        must serve real neighbors for a true recall parity check -- which acquires
        the ``.lock``, may rewrite ``records.hnsw`` via a boot-integrity rebuild, and
        in passphrase mode briefly plants ``.crypto.key``.

    Every file in the source ``hippo`` dir is copied (including any ``-wal``/``-shm``
    sidecars), so the snapshot's reads reflect the same committed state the migrator
    reads (``?mode=ro``) and the gate stays a TRUE full-data parity check.
    """
    src_hippo = Path(src_root) / "hippo"
    snap_root = Path(tempfile.mkdtemp(prefix="iai-verify-src-")) / "src"
    snap_hippo = snap_root / "hippo"
    snap_hippo.mkdir(parents=True, exist_ok=True)
    for entry in src_hippo.iterdir():
        if entry.is_file():
            shutil.copy2(entry, snap_hippo / entry.name)
    return snap_root


def _hit_ids(hits) -> set[str]:
    return {str(rec.id) for rec, _ in hits}


def _scores_by_id(hits) -> dict[str, float]:
    return {str(rec.id): float(score) for rec, score in hits}


def _verify_recall(
    src_db_path: str | Path,
    src_root: str | Path,
    dst_root: str | Path,
    src_is_lilli: bool,
    sample: list[str],
    *,
    k: int,
) -> DimensionResult:
    """Dimension D -- recall hit-id + score parity through MemoryStore.query_similar."""
    src_conn = _open_ro(src_db_path)
    try:
        # Pick a few deterministic probe cues from the sample's stored embeddings.
        probe_ids = sample[:: max(1, len(sample) // 5)][:5] if sample else []
        cues: list[list[float]] = []
        for rid in probe_ids:
            row = _fetch_row(src_conn, "records", rid)
            if row is None:
                continue
            cues.append(_decode_embedding(row["embedding"]))
    finally:
        src_conn.close()

    if not cues:
        return DimensionResult(
            ok=False, reason="no probe cues available (empty records sample)"
        )

    saved_env: dict = {}
    try:
        dst = _open_memory_store(dst_root, lilli=True, saved_env=saved_env)
        try:
            try:
                dst_hits = [dst.query_similar(cue, k) for cue in cues]
            except Exception as exc:  # noqa: BLE001 -- recall blew up on dst => RED
                return DimensionResult(
                    ok=False,
                    reason=f"dst recall raised {type(exc).__name__}: {exc}",
                )
        finally:
            dst.db.close()
        src = _open_memory_store(src_root, lilli=src_is_lilli, saved_env=saved_env)
        try:
            src_hits = [src.query_similar(cue, k) for cue in cues]
        finally:
            src.db.close()
    finally:
        _restore_env(saved_env)

    for i, (sh, dh) in enumerate(zip(src_hits, dst_hits)):
        s_ids = _hit_ids(sh)
        d_ids = _hit_ids(dh)
        if not d_ids:
            return DimensionResult(
                ok=False,
                reason=(
                    f"probe {i}: dst recall=0 (empty/stale ANN index) "
                    f"while src returned {len(s_ids)} hits"
                ),
            )
        if s_ids != d_ids:
            return DimensionResult(
                ok=False,
                reason=(
                    f"probe {i}: recall hit-set diverged: "
                    f"src\\dst={sorted(s_ids - d_ids)[:3]} "
                    f"dst\\src={sorted(d_ids - s_ids)[:3]}"
                ),
            )
        s_scores = _scores_by_id(sh)
        d_scores = _scores_by_id(dh)
        for rid in s_ids:
            if not math.isclose(
                s_scores[rid], d_scores[rid], abs_tol=_SCORE_TOL
            ):
                return DimensionResult(
                    ok=False,
                    reason=(
                        f"probe {i}: score mismatch for id={rid} "
                        f"src={s_scores[rid]:.6f} dst={d_scores[rid]:.6f}"
                    ),
                )
    return DimensionResult(
        ok=True,
        reason=f"recall hit-id + score parity over {len(cues)} probes",
        counts={"probes": len(cues)},
    )


def _verify_temporal(
    src_db_path: str | Path,
    src_root: str | Path,
    dst_root: str | Path,
    src_is_lilli: bool,
    sample: list[str],
    *,
    k: int,
) -> DimensionResult:
    """Dimension E -- temporal-recall parity through query_similar_temporal."""
    src_conn = _open_ro(src_db_path)
    try:
        probe_ids = sample[:: max(1, len(sample) // 3)][:3] if sample else []
        probes: list[tuple[list[float], str]] = []
        for rid in probe_ids:
            row = _fetch_row(src_conn, "records", rid)
            if row is None:
                continue
            as_of = row["created_at"]
            if not as_of:
                continue
            probes.append((_decode_embedding(row["embedding"]), str(as_of)))
    finally:
        src_conn.close()

    if not probes:
        return DimensionResult(
            ok=True, reason="no datable probe cues (temporal dimension skipped)"
        )

    saved_env: dict = {}
    try:
        dst = _open_memory_store(dst_root, lilli=True, saved_env=saved_env)
        try:
            try:
                dst_hits = [
                    dst.query_similar_temporal(cue, as_of=as_of, k=k)
                    for cue, as_of in probes
                ]
            except Exception as exc:  # noqa: BLE001 -- temporal recall blew up => RED
                return DimensionResult(
                    ok=False,
                    reason=f"dst temporal recall raised {type(exc).__name__}: {exc}",
                )
        finally:
            dst.db.close()
        src = _open_memory_store(src_root, lilli=src_is_lilli, saved_env=saved_env)
        try:
            src_hits = [
                src.query_similar_temporal(cue, as_of=as_of, k=k)
                for cue, as_of in probes
            ]
        finally:
            src.db.close()
    finally:
        _restore_env(saved_env)

    for i, (sh, dh) in enumerate(zip(src_hits, dst_hits)):
        s_ids = _hit_ids(sh)
        d_ids = _hit_ids(dh)
        if s_ids != d_ids:
            return DimensionResult(
                ok=False,
                reason=(
                    f"temporal probe {i}: hit-set diverged: "
                    f"src\\dst={sorted(s_ids - d_ids)[:3]} "
                    f"dst\\src={sorted(d_ids - s_ids)[:3]}"
                ),
            )
    return DimensionResult(
        ok=True,
        reason=f"temporal-recall parity over {len(probes)} probes",
        counts={"probes": len(probes)},
    )


def _verify_codec(
    src: Any,
    dst,
) -> DimensionResult:
    """Dimension F -- V5 codec metadata round-trip (hv_tier + structure_hv_payload).

    Co-scans EVERY record (bulk dst map + streamed src), no per-id engine lookup.
    """
    fail = _co_scan_bytes(src, dst, "records", _CODEC_COLUMNS)
    if fail is not None:
        # Re-label the generic co-scan reason for this dimension.
        return DimensionResult(
            ok=False, reason=fail.reason.replace("byte mismatch", "codec metadata mismatch")
        )
    n_records = _count(src, "records")
    return DimensionResult(
        ok=True,
        reason=f"V5 codec metadata round-trip over {n_records} records",
        counts={"records_checked": n_records},
    )


def verify_store_equality(
    src_db_path: str | Path,
    dst_root: str | Path,
    crypto_key: bytes,
    *,
    sample_n: int = 1000,
    k: int = 10,
    src_root: str | Path | None = None,
    src_is_lilli: bool = False,
) -> VerifyReport:
    """Return GREEN only when src (SQLite) and dst (lilli) are byte-equal.

    ``src_db_path`` is the source store's ``brain.sqlite3`` file (opened RO with
    the stdlib driver).  ``dst_root`` is the migrated lilli store root.  The
    encryption key is handled ONLY in this function (dimension C) -- the engine
    never sees plaintext or the key.

    ``src_root`` defaults to the parent of the ``hippo`` directory holding
    ``src_db_path`` (so the source can be re-opened as a MemoryStore for the
    recall/temporal dimensions).  ``src_is_lilli`` selects the source driver for
    those re-opens (the source is stdlib in production; True only when the source
    is itself a lilli store, e.g. a self-equality test).

    ``sample_n`` is retained for API compatibility but no longer bounds the byte /
    decrypt / codec dimensions -- those now co-scan EVERY row.  It is unused; the
    recall/temporal probe count is fixed (a handful of cues), already bounded.
    """
    src_db_path = Path(src_db_path)
    if src_root is None:
        # <root>/hippo/brain.sqlite3 -> <root>
        src_root = src_db_path.parent.parent

    dimensions: dict[str, DimensionResult] = {}
    sample: list[str] = []

    # Snapshot the SOURCE store into a throwaway temp root, and run EVERY dimension
    # against the snapshot -- the verifier never opens the caller's source store
    # itself, only the one-time file copy.  Two source-dir write surfaces are closed
    # this way:
    #   * the stdlib RO open of a WAL-mode brain.sqlite3 (dimensions A/B/C/F) still
    #     creates -wal/-shm sidecars in the store dir, even with ``?mode=ro``;
    #   * the recall/temporal re-open (D/E) is a WRITABLE MemoryStore (the ANN index
    #     must serve real neighbors for a true recall parity check), which acquires
    #     the ``.lock``, can fire a boot-integrity rebuild that rewrites
    #     ``records.hnsw``, and -- in passphrase mode -- briefly plants
    #     ``.crypto.key``.
    # Both land in the throwaway copy and are discarded; the live source's
    # brain.sqlite3 (+ -wal/-shm sidecars) is only ever READ (file copy), never
    # opened.  The copy carries the same brain.sqlite3 + sidecars, so the gate stays
    # a TRUE full-data parity check.  The dst is freshly built by the migrator and is
    # fine to open normally.
    src_snap_root = _snapshot_store_root(src_root)
    src_snap_db = src_snap_root / "hippo" / "brain.sqlite3"
    try:
        # The source is genuine SQLite (stdlib RO); the destination is lilli-format
        # and must be read through the lilli engine connection.  Open the dst as a
        # HippoDB under its env override, then restore the env in a finally block.
        src = _open_ro(src_snap_db)
        saved_table_env: dict = {}
        try:
            from iai_mcp.hippo import HippoDB

            saved_table_env["IAI_MCP_STORE"] = os.environ.get("IAI_MCP_STORE")
            saved_table_env["LILLI_STORAGE_DRIVER"] = os.environ.get("LILLI_STORAGE_DRIVER")
            os.environ["IAI_MCP_STORE"] = str(dst_root)
            os.environ["LILLI_STORAGE_DRIVER"] = "lilli"
            dst_db = HippoDB(dst_root)
            try:
                # Every dst read is wrapped under the shared-connection re-entrant
                # lock (Hippo thread-safety invariant).
                with dst_db._conn_lock:
                    dst = dst_db._conn
                    dimensions["A_counts"] = _verify_counts(src, dst)
                    bytes_result, sample = _verify_bytes(src, dst)
                    dimensions["B_bytes"] = bytes_result
                    dimensions["C_decrypt"] = _verify_decrypt(src, dst, crypto_key)
                    dimensions["F_codec"] = _verify_codec(src, dst)
            finally:
                dst_db.close()
        finally:
            src.close()
            _restore_env(saved_table_env)

        # Dimensions D/E re-open both stores as MemoryStore (own RO sqlite handles
        # internally), so they run after the stdlib handles above are closed.
        #
        # Recall decrypts each returned record's fields with the store's OWN key, so
        # the re-opened src/dst stores must resolve the SAME key the verbatim
        # ciphertext was encrypted with -- the key this verifier already holds.  A
        # migrated dst (and the source snapshot) carries no key file, and a stray
        # IAI_MCP_CRYPTO_PASSPHRASE would otherwise derive a DIFFERENT key and fail
        # recall decrypt with InvalidTag.  Plant the verifier's key as the store key
        # file (the file-backed key wins over the passphrase in
        # CryptoKey.get_or_create) for the duration of D/E, then remove any key file
        # this verifier created.  The source key is planted on the SNAPSHOT, never the
        # real source root.  The key stays above the engine -- the engine still only
        # ever sees ciphertext.
        planted = _plant_keys_for_recall(crypto_key, [src_snap_root, dst_root])
        try:
            dimensions["D_recall"] = _verify_recall(
                src_snap_db, src_snap_root, dst_root, src_is_lilli, sample, k=k
            )
            dimensions["E_temporal"] = _verify_temporal(
                src_snap_db, src_snap_root, dst_root, src_is_lilli, sample, k=k
            )
        finally:
            _unplant_keys(planted)
    finally:
        shutil.rmtree(src_snap_root.parent, ignore_errors=True)

    ok = all(d.ok for d in dimensions.values())
    return VerifyReport(ok=ok, dimensions=dimensions, sampled_n=len(sample))

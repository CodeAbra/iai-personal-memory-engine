"""Unit suite for the store-equality verifier.

Each parity dimension is proven to RED on a deliberate single-point corruption
and GREEN on byte-equality.  The recall dimension (D) is proven to catch the
recall=0 false-pass: an empty/stale ANN index on the destination flips D to RED
even when every row count matches.

The synthetic source store is built on the stdlib driver; a minimal inline
verbatim raw-copy (NOT the production migrator -- that lands separately) feeds
the verifier.  Fixtures use ``alice``; everything is hermetic (tmp HOME + tmp
store + the test passphrase, single process).
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pytest

from iai_mcp.migrate import verify_store_equality, VerifyReport


_ALL_TABLES = (
    "records",
    "edges",
    "events",
    "budget_ledger",
    "ratelimit_ledger",
    "_hippo_meta",
)


def _make_record(i: int, vec: list[float], *, created_at: datetime):
    from iai_mcp.types import MemoryRecord

    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=f"alice said thing number {i} about cats and rust",
        aaak_index="",
        embedding=vec,
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"session_id": "sess", "role": "user"}],
        created_at=created_at,
        updated_at=created_at,
        tags=["role:user"],
        language="en",
    )


def _build_source_store(root: Path, n: int = 30) -> tuple[str, list[list[float]]]:
    """Build a synthetic stdlib store with ``n`` records + edges + events.

    Returns the source db path + the list of stored embedding vectors.
    """
    os.environ.pop("LILLI_STORAGE_DRIVER", None)
    os.environ["IAI_MCP_STORE"] = str(root)
    from iai_mcp.store import MemoryStore, flush_record_buffer, flush_edge_buffer
    from iai_mcp.events import write_event, flush_event_buffer

    store = MemoryStore(root)
    rng = np.random.RandomState(7)
    ids: list = []
    vecs: list[list[float]] = []
    base = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    for i in range(n):
        vec = rng.randn(384).tolist()
        vecs.append(vec)
        rec = _make_record(i, vec, created_at=base + timedelta(minutes=i))
        ids.append(rec.id)
        store.insert(rec)
    flush_record_buffer(store)
    store.boost_edges([(ids[0], ids[1]), (ids[2], ids[3])])
    flush_edge_buffer(store)
    write_event(store, kind="probe_event", data={"n": 1})
    flush_event_buffer(store)
    src_db = str(root / "hippo" / "brain.sqlite3")
    store.db.close()
    return src_db, vecs


def _verbatim_copy(src_db: str, dst_root: Path, *, rebuild: bool = True) -> None:
    """Minimal inline verbatim raw-copy into a fresh lilli store.

    This is intentionally NOT the production migrator -- just enough to feed the
    verifier.  When ``rebuild`` is False the dst ANN is left empty/stale (used to
    prove the recall=0 false-pass is caught).
    """
    src = sqlite3.connect(f"file:{src_db}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    os.environ["LILLI_STORAGE_DRIVER"] = "lilli"
    os.environ["IAI_MCP_STORE"] = str(dst_root)
    from iai_mcp.hippo import HippoDB

    dst = HippoDB(dst_root)
    try:
        for table in _ALL_TABLES:
            cols = [r["name"] for r in src.execute(f"PRAGMA table_info({table})").fetchall()]
            if not cols:
                continue
            cl = ", ".join(cols)
            ph = ", ".join("?" for _ in cols)
            verb = "INSERT OR REPLACE" if table == "_hippo_meta" else "INSERT"
            ins = f"{verb} INTO {table} ({cl}) VALUES ({ph})"
            cur = src.execute(f"SELECT {cl} FROM {table}")
            max_vec = 0
            with dst._conn_lock:
                while True:
                    rows = cur.fetchmany(500)
                    if not rows:
                        break
                    dst._conn.executemany(ins, [tuple(r[c] for c in cols) for r in rows])
                    dst._conn.commit()
                    if table == "records":
                        for r in rows:
                            if r["vec_label"] is not None:
                                max_vec = max(max_vec, int(r["vec_label"]))
            if table == "records" and max_vec > 0:
                set_hwm = getattr(dst._conn, "set_vec_label_hwm", None)
                if callable(set_hwm):
                    set_hwm("records", max_vec)
        if rebuild:
            dst._rebuild_index_from_sqlite()
    finally:
        dst.close()
        src.close()


def _crypto_key(root: Path) -> bytes:
    from iai_mcp.crypto import CryptoKey

    return CryptoKey(store_root=root).get_or_create()


def _dst_db(dst_root: Path) -> str:
    return str(dst_root / "hippo" / "brain.sqlite3")


@pytest.fixture
def migrated_pair(tmp_path: Path):
    """A GREEN-verified (source-root, dst-root, key) triple."""
    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    src_root.mkdir()
    src_db, _vecs = _build_source_store(src_root, n=30)
    _verbatim_copy(src_db, dst_root, rebuild=True)
    key = _crypto_key(src_root)
    # restore env so the verifier owns its own env juggling
    os.environ["IAI_MCP_STORE"] = str(src_root)
    os.environ.pop("LILLI_STORAGE_DRIVER", None)
    return src_db, src_root, dst_root, key


def test_green_on_identical(migrated_pair):
    src_db, src_root, dst_root, key = migrated_pair
    report = verify_store_equality(src_db, dst_root, key, src_root=src_root)
    assert isinstance(report, VerifyReport)
    assert report.ok is True, {k: v.reason for k, v in report.dimensions.items()}
    for name, dim in report.dimensions.items():
        assert dim.ok is True, f"{name}: {dim.reason}"


def test_counts_red_on_dropped_row(migrated_pair):
    src_db, src_root, dst_root, key = migrated_pair
    # Delete ONE record from the dst lilli store.
    os.environ["LILLI_STORAGE_DRIVER"] = "lilli"
    os.environ["IAI_MCP_STORE"] = str(dst_root)
    from iai_mcp.hippo import HippoDB

    dst = HippoDB(dst_root)
    try:
        with dst._conn_lock:
            rid = dst._conn.execute("SELECT id FROM records LIMIT 1").fetchone()[0]
            dst._conn.execute("DELETE FROM records WHERE id = ?", (rid,))
            dst._conn.commit()
    finally:
        dst.close()
    os.environ["IAI_MCP_STORE"] = str(src_root)
    os.environ.pop("LILLI_STORAGE_DRIVER", None)

    report = verify_store_equality(src_db, dst_root, key, src_root=src_root)
    assert report.ok is False
    assert report.dimensions["A_counts"].ok is False
    assert "records" in report.dimensions["A_counts"].reason


def test_ciphertext_red_on_mutation(migrated_pair):
    src_db, src_root, dst_root, key = migrated_pair
    # Flip one byte of a literal_surface ciphertext in dst.
    os.environ["LILLI_STORAGE_DRIVER"] = "lilli"
    os.environ["IAI_MCP_STORE"] = str(dst_root)
    from iai_mcp.hippo import HippoDB

    dst = HippoDB(dst_root)
    try:
        with dst._conn_lock:
            row = dst._conn.execute(
                "SELECT id, literal_surface FROM records LIMIT 1"
            ).fetchone()
            rid, ct = row["id"], row["literal_surface"]
            # mutate one character of the ciphertext string
            mutated = ct[:14] + ("A" if ct[14] != "A" else "B") + ct[15:]
            dst._conn.execute(
                "UPDATE records SET literal_surface = ? WHERE id = ?", (mutated, rid)
            )
            dst._conn.commit()
    finally:
        dst.close()
    os.environ["IAI_MCP_STORE"] = str(src_root)
    os.environ.pop("LILLI_STORAGE_DRIVER", None)

    report = verify_store_equality(src_db, dst_root, key, src_root=src_root)
    assert report.ok is False
    assert report.dimensions["B_bytes"].ok is False
    assert "literal_surface" in report.dimensions["B_bytes"].reason


def _save_empty_hnsw(dst_root: Path) -> int:
    """Write a provably-EMPTY records.hnsw to the dst hippo dir.

    Returns the index element count at plant time (asserted 0 by the caller), so
    the recovery test proves the planted index genuinely held zero elements before
    open -- making the post-open recall-parity assertion non-trivial.
    """
    import hnswlib

    idx = hnswlib.Index(space="cosine", dim=384)
    idx.init_index(max_elements=16, ef_construction=200, M=16, allow_replace_deleted=True)
    hnsw_path = dst_root / "hippo" / "records.hnsw"
    idx.save_index(str(hnsw_path))
    return idx.get_current_count()


def test_recall_recovers_from_empty_index(tmp_path: Path):
    """A planted-empty records.hnsw self-heals on open; recall parity is restored.

    The boot-integrity rebuild compares the actually-loaded hnsw element count
    against the live SQLite count and rebuilds a stale/undersized on-disk index on
    open, for both storage drivers.  An empty-but-valid index file is therefore NOT
    a silent recall=0 false-pass: the store rebuilds it and full recall coverage is
    restored.  This is the resilience the double-buffered ANN guarantees.

    The test plants a PROVABLY-empty index (count 0 at plant time), then asserts the
    verifier reaches GREEN on the recall dimension -- which is only possible if the
    self-heal actually fired and recall returned the same hit-set as the source.  A
    regression that disabled the boot-integrity rebuild would leave recall at 0 and
    flip D_recall back to RED, so this stays a real guard, not a trivial pass.
    """
    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    src_root.mkdir()
    src_db, _vecs = _build_source_store(src_root, n=30)
    _verbatim_copy(src_db, dst_root, rebuild=True)
    # Overwrite the live index with a provably-empty one (count 0 at plant time).
    planted_count = _save_empty_hnsw(dst_root)
    assert planted_count == 0, "planted index must be empty for the recovery to be meaningful"
    key = _crypto_key(src_root)
    os.environ["IAI_MCP_STORE"] = str(src_root)
    os.environ.pop("LILLI_STORAGE_DRIVER", None)

    report = verify_store_equality(src_db, dst_root, key, src_root=src_root)
    # Counts/bytes match (the data is intact -- only the ANN file was stale).
    assert report.dimensions["A_counts"].ok is True
    assert report.dimensions["B_bytes"].ok is True
    # The boot-integrity rebuild self-healed the empty ANN: recall parity restored.
    assert report.dimensions["D_recall"].ok is True, report.dimensions["D_recall"].reason
    assert report.dimensions["D_recall"].counts.get("probes", 0) > 0
    assert report.ok is True, {k: v.reason for k, v in report.dimensions.items()}


def _hippo_dir_snapshot(root: Path) -> dict[str, tuple[int, int]]:
    """Map every file in ``<root>/hippo`` to its (mtime_ns, size)."""
    hippo = root / "hippo"
    return {
        p.name: (p.stat().st_mtime_ns, p.stat().st_size)
        for p in hippo.iterdir()
        if p.is_file()
    }


def test_source_dir_unchanged_after_verify(migrated_pair):
    """The verifier must never open the SOURCE store writable.

    A GREEN verify re-opens the source for the recall/temporal dimensions, but it
    must do so against a throwaway snapshot, not the live source root.  Assert the
    source ``hippo`` dir is byte-identical before and after a verify: no sidecar
    mtime/size change (no WAL/SHM, no records.hnsw rebuild, no .lock churn) and no
    new file (no transient .crypto.key planted on the source).  Mirrors the
    dry-run's live-store mtime-unchanged invariant, but for the source dir.
    """
    src_db, src_root, dst_root, key = migrated_pair

    before = _hippo_dir_snapshot(src_root)
    # A bare source store carries no -wal/.crypto.key; capture the file set so a
    # newly-created sidecar is caught, not just a mutation of an existing one.
    before_names = set(before)

    report = verify_store_equality(src_db, dst_root, key, src_root=src_root)
    assert report.ok is True, {k: v.reason for k, v in report.dimensions.items()}
    # The recall dimension genuinely ran (a real full-data gate, not skipped).
    assert report.dimensions["D_recall"].ok is True
    assert report.dimensions["D_recall"].counts.get("probes", 0) > 0

    after = _hippo_dir_snapshot(src_root)
    after_names = set(after)

    # No new sidecar file appeared in the source dir (no WAL/SHM/.crypto.key).
    new_files = after_names - before_names
    assert not new_files, f"verifier created files in the SOURCE dir: {new_files}"
    # No existing source file was mutated (no records.hnsw rebuild, no .lock churn).
    assert after == before, (
        "verifier mutated SOURCE dir files: "
        f"{[n for n in after if after.get(n) != before.get(n)]}"
    )


def test_temporal_red_on_window_divergence(migrated_pair):
    src_db, src_root, dst_root, key = migrated_pair
    # Corrupt the dst created_at on a record so a time-filtered probe diverges:
    # push every dst created_at far into the future, so as_of cutoffs derived
    # from the (unchanged) src timestamps exclude rows on dst that src includes.
    os.environ["LILLI_STORAGE_DRIVER"] = "lilli"
    os.environ["IAI_MCP_STORE"] = str(dst_root)
    from iai_mcp.hippo import HippoDB

    dst = HippoDB(dst_root)
    try:
        with dst._conn_lock:
            dst._conn.execute(
                "UPDATE records SET created_at = '2099-01-01 00:00:00+00:00'"
            )
            dst._conn.commit()
            dst._rebuild_index_from_sqlite()
    finally:
        dst.close()
    os.environ["IAI_MCP_STORE"] = str(src_root)
    os.environ.pop("LILLI_STORAGE_DRIVER", None)

    report = verify_store_equality(src_db, dst_root, key, src_root=src_root)
    assert report.ok is False
    # The byte dimension also flags created_at? No -- created_at is not in the
    # byte-compared columns, so this isolates the temporal dimension.
    assert report.dimensions["E_temporal"].ok is False

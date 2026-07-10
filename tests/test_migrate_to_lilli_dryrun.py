"""Opt-in full-scale dry-run of the SQLite -> lilli migrator + verifier.

This harness proves the migrator and the store-equality verifier on the real
data at real scale -- but SAFELY, on a COPY.  It ``cp``s the live store into a
tmp directory (read-only copy), runs the full migrator on that copy into a
SEPARATE fresh tmp destination, and runs the full verifier.  It asserts all six
parity dimensions GREEN at full scale, peak RSS flat (jetsam-safe), and records
the wall-clock and RSS so the owner sees real timing before any cutover.

The live store is NEVER opened for write and is NEVER the migrator destination.
The harness asserts the live store's mtime is unchanged across the run.

It is OPT-IN: skipped unless ``IAI_MCP_DRYRUN_REAL_STORE`` is set AND the real
store exists, so the default suite and other machines skip it cleanly.
"""

from __future__ import annotations

import os
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


_MIN_CORPUS = 30000


def _real_store_db() -> Path | None:
    """Resolve the real store's db path WITHOUT triggering the hermeticity guard.

    ``_resolve_root`` raises during a pytest run if it would return the real home
    store, so resolve the real root directly from the operator's OS home (read
    from the passwd entry, not the monkeypatched ``$HOME``).  This is a read-only
    path lookup, never an open.
    """
    from iai_mcp.hippo import _operator_home

    real_root = _operator_home() / ".iai-mcp"
    db = real_root / "hippo" / "brain.sqlite3"
    return db if db.exists() else None


_DRYRUN_ENABLED = bool(os.environ.get("IAI_MCP_DRYRUN_REAL_STORE"))
_REAL_DB = _real_store_db()


@pytest.mark.skipif(
    not _DRYRUN_ENABLED or _REAL_DB is None,
    reason=(
        "dry-run is opt-in: set IAI_MCP_DRYRUN_REAL_STORE and have the real "
        "store present"
    ),
)
def test_dryrun_full_scale_roundtrip_on_copy(tmp_path: Path):
    real_db = _REAL_DB
    assert real_db is not None and real_db.exists()
    real_root = real_db.parent.parent

    # Record the live store's mtime + size BEFORE -- it must be untouched after.
    mtime_before = real_db.stat().st_mtime_ns
    size_before = real_db.stat().st_size
    print(f"\n[dry-run] real store: {size_before / 1e6:.1f} MB")

    # cp the live store into tmp (read-only copy is the migrator SOURCE).
    copy_root = tmp_path / "copy"
    (copy_root / "hippo").mkdir(parents=True)
    copy_db = copy_root / "hippo" / "brain.sqlite3"
    shutil.copy2(real_db, copy_db)
    # Copy the WAL sidecar too if present, for a consistent read.
    real_wal = real_db.with_name(real_db.name + "-wal")
    if real_wal.exists():
        shutil.copy2(real_wal, copy_db.with_name(copy_db.name + "-wal"))

    # A SEPARATE fresh tmp path is the migrator DESTINATION.
    dst_root = tmp_path / "dst"

    from iai_mcp.migrate import migrate_sqlite_to_lilli, verify_store_equality
    from iai_mcp.crypto import CryptoKey

    # The copy is encrypted with the REAL key; read it (read-only) so the
    # verifier's decrypt-parity uses the same key the store was encrypted with.
    key = CryptoKey(store_root=real_root).get_or_create()

    report = migrate_sqlite_to_lilli(str(copy_db), dst_root, batch=1000)
    print(
        f"[dry-run] migrated in {report.elapsed_sec:.1f}s; "
        f"peak RSS {report.peak_rss_mb:.0f} MB; "
        f"rows {report.rows_copied}; max vec_label {report.max_vec_label}"
    )

    rep = verify_store_equality(
        str(copy_db), dst_root, key, src_root=copy_root, sample_n=2000
    )
    for name, dim in rep.dimensions.items():
        status = "ok" if dim.ok else "FAIL"
        print(f"[dry-run]   [{status}] {name}: {dim.reason}")

    # All six dimensions GREEN at real scale.
    assert rep.ok is True, {k: v.reason for k, v in rep.dimensions.items()}

    # Peak RSS stays well under a generous jetsam-safe ceiling (measured ~120 MB).
    assert report.peak_rss_mb < 1500, (
        f"peak RSS {report.peak_rss_mb} MB exceeded the jetsam-safe ceiling"
    )

    # The live store was never opened for write -- mtime + size unchanged.
    assert real_db.stat().st_mtime_ns == mtime_before, "live store mtime changed!"
    assert real_db.stat().st_size == size_before, "live store size changed!"

    print(
        f"[dry-run] DONE: all 6 dimensions GREEN; "
        f"elapsed {report.elapsed_sec:.1f}s; peak RSS {report.peak_rss_mb:.0f} MB; "
        f"live store mtime unchanged"
    )


# ---------------------------------------------------------------------------
# >=30k cutover-on-copy: proves the scale failure mode (recall=0 at 30k on the
# engine) is gone on the Rust-backed store, with a NON-DEGENERATE corpus so the
# recall parity check is genuinely exercised (no single-cluster collapse).
# ---------------------------------------------------------------------------
def _make_record(i: int, vec: list[float], *, created_at: datetime):
    from iai_mcp.types import MemoryRecord

    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=f"alice noted distinct fact {i} about rust cats and stores",
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


def _spread_vectors(n: int, *, seed: int = 17):
    """Yield ``n`` NON-degenerate, L2-normalized 384-d vectors.

    Per-record gaussian variance spreads the corpus so recall exercises real ANN
    behaviour -- collinear/near-identical vectors would let dim-D pass trivially
    (every probe returning the same tiny cluster).  Fixed seed for reproducibility.
    """
    import numpy as np

    rng = np.random.RandomState(seed)
    for _ in range(n):
        v = rng.randn(384).astype("float64")
        norm = float(np.linalg.norm(v)) or 1.0
        yield (v / norm).tolist()


def _topup_store_to_min(root: Path, *, start_idx: int, target: int) -> int:
    """Append spread records to the COPY at ``root`` until it holds >= target.

    Returns the number of records added.  Operates only on the given (copied or
    synthetic) root -- NEVER the live store.
    """
    os.environ["IAI_MCP_STORE"] = str(root)
    os.environ.pop("LILLI_STORAGE_DRIVER", None)
    from iai_mcp.store import MemoryStore, flush_record_buffer

    store = MemoryStore(root)
    try:
        existing = store.db.open_table("records").count_rows()
        need = max(0, target - existing)
        base = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        for j, vec in enumerate(_spread_vectors(need, seed=start_idx + 1)):
            rec = _make_record(start_idx + j, vec, created_at=base + timedelta(seconds=j))
            store.insert(rec)
        flush_record_buffer(store)
    finally:
        store.db.close()
    return need


def _build_synthetic_source(root: Path, *, n: int) -> str:
    """Build a fresh stdlib source store of ``n`` spread records + edges + events."""
    os.environ["IAI_MCP_STORE"] = str(root)
    os.environ.pop("LILLI_STORAGE_DRIVER", None)
    from iai_mcp.store import MemoryStore, flush_record_buffer, flush_edge_buffer
    from iai_mcp.events import write_event, flush_event_buffer

    store = MemoryStore(root)
    ids: list = []
    base = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    for i, vec in enumerate(_spread_vectors(n, seed=17)):
        rec = _make_record(i, vec, created_at=base + timedelta(seconds=i))
        ids.append(rec.id)
        store.insert(rec)
    flush_record_buffer(store)
    store.boost_edges([(ids[0], ids[1]), (ids[2], ids[3])])
    flush_edge_buffer(store)
    write_event(store, kind="probe_event", data={"n": 1})
    flush_event_buffer(store)
    store.db.close()
    return str(root / "hippo" / "brain.sqlite3")


def _distinct_hit_count_through_lilli(
    dst_root: Path, *, k: int, crypto_key: bytes
) -> int:
    """Open the migrated dst under the Rust driver, probe once, count distinct hits.

    Returns the max distinct hit-id count across a few probes (the corpus must
    NOT collapse to a single cluster: at least one probe returns >= k ids).

    The dst holds ciphertext encrypted with ``crypto_key`` (the verifier's key);
    that key is planted transiently into ``dst_root`` (0o600, removed after) so
    ``query_similar`` decrypts the stored hits rather than resolving a mismatched
    machine-derived key.  Without it recall would raise InvalidTag on a real-copy
    source -- the verifier plants and removes the same key for its own D_recall
    probe, so this standalone probe must do the same.
    """
    from iai_mcp.crypto import _KEY_FILE_NAME, KEY_BYTES

    saved = {
        "IAI_MCP_STORE": os.environ.get("IAI_MCP_STORE"),
        "LILLI_STORAGE_DRIVER": os.environ.get("LILLI_STORAGE_DRIVER"),
    }
    os.environ["IAI_MCP_STORE"] = str(dst_root)
    os.environ["LILLI_STORAGE_DRIVER"] = "lilli"

    planted: Path | None = None
    key_path = Path(dst_root) / _KEY_FILE_NAME
    if len(crypto_key) == KEY_BYTES and not key_path.exists():
        key_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(key_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.fchmod(fd, 0o600)
            os.write(fd, crypto_key)
        finally:
            os.close(fd)
        planted = key_path
    try:
        from iai_mcp.store import MemoryStore

        store = MemoryStore(dst_root)
        try:
            best = 0
            for cue in _spread_vectors(3, seed=999):
                hits = store.query_similar(cue, k)
                best = max(best, len({str(rec.id) for rec, _ in hits}))
            return best
        finally:
            store.db.close()
    finally:
        if planted is not None:
            try:
                planted.unlink()
            except OSError:
                pass
        for key, prior in saved.items():
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior


@pytest.mark.skipif(
    not _DRYRUN_ENABLED,
    reason="cutover-on-copy is opt-in: set IAI_MCP_DRYRUN_REAL_STORE",
)
def test_dryrun_30k_correct_and_fast_on_copy(tmp_path: Path):
    from iai_mcp.crypto import CryptoKey
    from iai_mcp.migrate import migrate_sqlite_to_lilli, verify_store_equality

    k = 10
    copy_root = tmp_path / "src"
    (copy_root / "hippo").mkdir(parents=True)
    copy_db = copy_root / "hippo" / "brain.sqlite3"

    mtime_before = size_before = None
    provenance: str

    if _REAL_DB is not None:
        # Real read-only copy of the live store as the migrator SOURCE.
        real_db = _REAL_DB
        real_root = real_db.parent.parent
        mtime_before = real_db.stat().st_mtime_ns
        size_before = real_db.stat().st_size
        shutil.copy2(real_db, copy_db)
        real_wal = real_db.with_name(real_db.name + "-wal")
        if real_wal.exists():
            shutil.copy2(real_wal, copy_db.with_name(copy_db.name + "-wal"))
        # The copy is encrypted with the REAL key; read it so the verifier's
        # decrypt parity uses the same key the store was encrypted with.
        key = CryptoKey(store_root=real_root).get_or_create()

        os.environ["IAI_MCP_STORE"] = str(copy_root)
        os.environ.pop("LILLI_STORAGE_DRIVER", None)
        from iai_mcp.store import MemoryStore

        s = MemoryStore(copy_root)
        existing = s.db.open_table("records").count_rows()
        s.db.close()
        if existing >= _MIN_CORPUS:
            provenance = "real-copy"
        else:
            added = _topup_store_to_min(
                copy_root, start_idx=existing, target=_MIN_CORPUS
            )
            provenance = "real-copy+topup"
            print(f"[30k] topped up real copy by {added} spread records")
    else:
        # No real store present -- fully synthetic spread corpus.
        _build_synthetic_source(copy_root, n=_MIN_CORPUS)
        key = CryptoKey(store_root=copy_root).get_or_create()
        provenance = "synthetic"

    # Confirm the source reached the >= 30000 bar.
    os.environ["IAI_MCP_STORE"] = str(copy_root)
    os.environ.pop("LILLI_STORAGE_DRIVER", None)
    from iai_mcp.store import MemoryStore

    s = MemoryStore(copy_root)
    n_src = s.db.open_table("records").count_rows()
    s.db.close()
    assert n_src >= _MIN_CORPUS, f"source corpus {n_src} < {_MIN_CORPUS}"

    # Migrate into a SEPARATE fresh tmp dst through the Rust engine.
    dst_root = tmp_path / "dst"
    report = migrate_sqlite_to_lilli(str(copy_db), dst_root, batch=1000)
    print(
        f"[30k] provenance={provenance}; source records={n_src}; "
        f"migrated in {report.elapsed_sec:.1f}s; peak RSS {report.peak_rss_mb:.0f} MB"
    )

    # Six-dimension co-scan through the Rust-routed dst.
    os.environ["IAI_MCP_STORE"] = str(copy_root)
    os.environ.pop("LILLI_STORAGE_DRIVER", None)
    rep = verify_store_equality(
        str(copy_db), dst_root, key, src_root=copy_root, sample_n=2000
    )
    for name, dim in rep.dimensions.items():
        status = "ok" if dim.ok else "FAIL"
        print(f"[30k]   [{status}] {name}: {dim.reason}")
    assert rep.ok is True, {k_: v.reason for k_, v in rep.dimensions.items()}

    # Explicitly assert dim-D recall != 0 -- the exact scale failure surface.
    d = rep.dimensions["D_recall"]
    assert d.ok is True, f"D_recall RED (recall=0 surface): {d.reason}"

    # Non-degeneracy: at least one probe returns >= k DISTINCT hit ids, proving
    # recall is genuinely non-trivial (the corpus did not collapse to one cluster).
    distinct = _distinct_hit_count_through_lilli(dst_root, k=k, crypto_key=key)
    print(f"[30k] max distinct hit ids across probes: {distinct} (k={k})")
    assert distinct >= k, (
        f"recall is degenerate: best probe returned {distinct} distinct ids "
        f"(< k={k}); the corpus collapsed to a single cluster"
    )

    # Jetsam-safe peak RSS; sampled in-process by the migrator (no daemon restart).
    assert report.peak_rss_mb < 1500, (
        f"peak RSS {report.peak_rss_mb} MB exceeded the jetsam-safe ceiling"
    )

    # If a real source was copied, the live store must be untouched.
    if mtime_before is not None:
        assert _REAL_DB.stat().st_mtime_ns == mtime_before, "live store mtime changed!"
        assert _REAL_DB.stat().st_size == size_before, "live store size changed!"

    print(
        f"[30k] DONE: 6 dimensions GREEN incl. D_recall != 0; "
        f"{distinct} distinct hits; provenance={provenance}; "
        f"peak RSS {report.peak_rss_mb:.0f} MB"
    )

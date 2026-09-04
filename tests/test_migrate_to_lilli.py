"""Synthetic round-trip suite for the SQLite -> lilli verbatim migrator.

Builds a source store on the stdlib driver, migrates it to a fresh lilli store,
and asserts all six verifier dimensions GREEN plus vec_label/HWM correctness, RSS
flatness, and the same-path guard.  Parity assertions are driven through the
``verify_store_equality`` store-equality gate wherever a dimension maps cleanly.

Hermetic: tmp HOME + tmp store + the test passphrase, single process, no xdist.
Fixtures use ``alice``.
"""

from __future__ import annotations

import json
import os
import platform
import sqlite3
import subprocess
import sys
import textwrap
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pytest

from iai_mcp.migrate import (
    migrate_sqlite_to_lilli,
    MigrateReport,
    verify_store_equality,
)


def _ru_maxrss_mb(raw: int) -> float:
    """Normalize a getrusage ru_maxrss reading to MiB across platforms.

    macOS reports ru_maxrss in bytes; Linux reports it in KiB.
    """
    if platform.system() == "Darwin":
        return raw / (1024 * 1024)
    return raw / 1024


def _migrate_in_subprocess(
    src_db: str, dst_root: str, *, batch: int
) -> tuple[MigrateReport, float]:
    """Run one migration in a fresh, hermetic child and report ITS peak RSS.

    Returns the in-child ``MigrateReport`` plus the child's own high-water RSS
    in MiB taken from the child's ``getrusage(RUSAGE_SELF)``. Because the child
    starts clean, its resident set reflects only what the migration itself
    needs — the measurement is independent of how loaded the parent test runner
    already is, so the bound stays meaningful under a full-suite run.
    """
    worker = textwrap.dedent(
        """
        import json, resource, sys
        from iai_mcp.migrate import migrate_sqlite_to_lilli

        src_db, dst_root, batch = sys.argv[1], sys.argv[2], int(sys.argv[3])
        report = migrate_sqlite_to_lilli(src_db, dst_root, batch=batch)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(json.dumps({
            "rows_copied": report.rows_copied,
            "max_vec_label": report.max_vec_label,
            "elapsed_sec": report.elapsed_sec,
            "child_peak_rss_raw": peak,
        }))
        """
    )
    env = os.environ.copy()
    env["LILLI_STORAGE_DRIVER"] = "lilli"
    env.setdefault("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase-not-secret")
    proc = subprocess.run(
        [sys.executable, "-c", worker, src_db, dst_root, str(batch)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"migration subprocess failed (rc={proc.returncode}):\n{proc.stderr}"
    )
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    report = MigrateReport(
        rows_copied=payload["rows_copied"],
        max_vec_label=payload["max_vec_label"],
        elapsed_sec=payload["elapsed_sec"],
    )
    return report, _ru_maxrss_mb(int(payload["child_peak_rss_raw"]))


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


def _build_source_store(
    root: Path, *, monkeypatch: pytest.MonkeyPatch, n_records: int, n_events: int = 0
) -> str:
    """Build a synthetic stdlib source store; return its db path."""
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "stdlib")
    monkeypatch.setenv("IAI_MCP_STORE", str(root))
    from iai_mcp.store import MemoryStore, flush_record_buffer, flush_edge_buffer
    from iai_mcp.events import write_event, flush_event_buffer

    store = MemoryStore(root)
    rng = np.random.RandomState(11)
    ids: list = []
    base = datetime(2026, 5, 1, 9, 0, 0, tzinfo=timezone.utc)
    for i in range(n_records):
        vec = rng.randn(384).tolist()
        rec = _make_record(i, vec, created_at=base + timedelta(seconds=i))
        ids.append(rec.id)
        store.insert(rec)
    flush_record_buffer(store)
    if len(ids) >= 4:
        store.boost_edges([(ids[0], ids[1]), (ids[2], ids[3])])
        flush_edge_buffer(store)
    for j in range(n_events):
        write_event(store, kind="probe_event", data={"j": j})
    flush_event_buffer(store)
    src_db = str(root / "hippo" / "brain.sqlite3")
    store.db.close()
    return src_db


def _crypto_key(root: Path) -> bytes:
    from iai_mcp.crypto import CryptoKey

    return CryptoKey(store_root=root).get_or_create()


def _table_count_lilli(
    dst_root: Path, table: str, *, monkeypatch: pytest.MonkeyPatch
) -> int:
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    monkeypatch.setenv("IAI_MCP_STORE", str(dst_root))
    from iai_mcp.hippo import HippoDB

    dst = HippoDB(dst_root)
    try:
        with dst._conn_lock:
            return int(dst._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    finally:
        dst.close()


def _table_count_src(src_db: str, table: str) -> int:
    c = sqlite3.connect(f"file:{src_db}?mode=ro", uri=True)
    try:
        return int(c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    finally:
        c.close()


@pytest.fixture
def migrated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Migrate a 50-record synthetic store; return (src_db, src_root, dst_root, key, report)."""
    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    src_root.mkdir()
    src_db = _build_source_store(src_root, monkeypatch=monkeypatch, n_records=50)
    report = migrate_sqlite_to_lilli(src_db, dst_root, batch=20)
    key = _crypto_key(src_root)
    monkeypatch.setenv("IAI_MCP_STORE", str(src_root))
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "stdlib")
    return src_db, src_root, dst_root, key, report


def test_counts_match(migrated, monkeypatch: pytest.MonkeyPatch):
    src_db, src_root, dst_root, key, report = migrated
    assert isinstance(report, MigrateReport)
    for table in ("records", "edges", "events", "_hippo_meta"):
        assert _table_count_src(src_db, table) == _table_count_lilli(
            dst_root, table, monkeypatch=monkeypatch
        ), table


def test_ciphertext_roundtrip(migrated):
    src_db, src_root, dst_root, key, report = migrated
    rep = verify_store_equality(src_db, dst_root, key, src_root=src_root)
    assert rep.dimensions["B_bytes"].ok is True, rep.dimensions["B_bytes"].reason
    assert rep.dimensions["C_decrypt"].ok is True, rep.dimensions["C_decrypt"].reason


def test_embedding_bytes(migrated):
    src_db, src_root, dst_root, key, report = migrated
    # embedding is one of the byte-compared columns in dimension B.
    rep = verify_store_equality(src_db, dst_root, key, src_root=src_root)
    assert rep.dimensions["B_bytes"].ok is True


def test_recall_parity(migrated):
    src_db, src_root, dst_root, key, report = migrated
    rep = verify_store_equality(src_db, dst_root, key, src_root=src_root)
    assert rep.dimensions["D_recall"].ok is True, rep.dimensions["D_recall"].reason


def test_temporal_parity(migrated):
    src_db, src_root, dst_root, key, report = migrated
    rep = verify_store_equality(src_db, dst_root, key, src_root=src_root)
    assert rep.dimensions["E_temporal"].ok is True, rep.dimensions["E_temporal"].reason


def test_vec_label_hwm(migrated, monkeypatch: pytest.MonkeyPatch):
    src_db, src_root, dst_root, key, report = migrated
    src_max = _table_count_src(src_db, "records")  # 50 records, labels 1..50
    # report.max_vec_label preserved from the source.
    assert report.max_vec_label >= src_max

    # A fresh insert on the migrated lilli store must get a NEW label > max.
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    monkeypatch.setenv("IAI_MCP_STORE", str(dst_root))
    from iai_mcp.store import MemoryStore, flush_record_buffer

    store = MemoryStore(dst_root)
    try:
        before = int(
            store.db._conn.execute("SELECT MAX(vec_label) FROM records").fetchone()[0]
        )
        rec = _make_record(
            999, np.random.RandomState(99).randn(384).tolist(),
            created_at=datetime.now(timezone.utc),
        )
        store.insert(rec)
        flush_record_buffer(store)
        after = int(
            store.db._conn.execute(
                "SELECT vec_label FROM records WHERE id = ?", (str(rec.id),)
            ).fetchone()[0]
        )
    finally:
        store.db.close()

    assert after > before, f"new label {after} did not exceed prior max {before}"


def test_rss_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Larger batched copy: ~2000 records + ~10000 events.  Research measured
    # ~120 MB peak; assert well under a generous ceiling AND that doubling the
    # event count does not roughly double peak RSS (proves streaming).
    #
    # The migration runs in a FRESH child process and we bound the CHILD's own
    # peak RSS (getrusage RUSAGE_SELF), not the parent test runner's
    # process-global peak.  ``MigrateReport.peak_rss_mb`` samples
    # ``psutil.Process().memory_info().rss`` of whatever process it runs in, so
    # under a full-suite run it is inflated by everything the runner already
    # loaded (embedder, lilli, prior tests) and the bound would flake on load.
    # Measuring a clean child makes the bound load-independent while still
    # catching a real blow-up in the migration's own footprint.
    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    src_root.mkdir()
    src_db = _build_source_store(
        src_root, monkeypatch=monkeypatch, n_records=2000, n_events=10000
    )
    report, child_peak_mb = _migrate_in_subprocess(
        src_db, str(dst_root), batch=1000
    )
    assert report.rows_copied.get("records", 0) == 2000

    # Generous jetsam-safe ceiling on the migration's own resident footprint.
    # A streaming copy of 2000 records + 10000 events fits well under this; a
    # regression that buffers a whole table would blow past it.
    assert child_peak_mb < 800, (
        f"migration child peak RSS {child_peak_mb:.0f} MB too high"
    )

    # Flatness: a half-size copy must not have roughly half the peak RSS (peak
    # is dominated by the resident interpreter + engine, not the row count).
    # Both arms are measured in equally-clean children, so the ratio is robust
    # to ambient load.
    src_root2 = tmp_path / "src2"
    dst_root2 = tmp_path / "dst2"
    src_root2.mkdir()
    src_db2 = _build_source_store(
        src_root2, monkeypatch=monkeypatch, n_records=1000, n_events=5000
    )
    report2, child_peak_mb2 = _migrate_in_subprocess(
        src_db2, str(dst_root2), batch=1000
    )
    assert report2.rows_copied.get("records", 0) == 1000
    # If memory scaled with rows, peak would roughly halve.  Assert it does NOT.
    assert child_peak_mb2 > child_peak_mb * 0.5, (
        f"peak RSS scaled with row count (full={child_peak_mb:.0f} "
        f"half={child_peak_mb2:.0f}) -- not streaming"
    )


def test_same_path_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    src_root = tmp_path / "src"
    src_root.mkdir()
    src_db = _build_source_store(src_root, monkeypatch=monkeypatch, n_records=5)
    # dst_root whose brain.sqlite3 resolves to the SAME file as src_db.
    same_root = src_root  # <root>/hippo/brain.sqlite3 == src_db
    monkeypatch.setenv("IAI_MCP_STORE", str(src_root))
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "stdlib")
    with pytest.raises(ValueError, match="same store"):
        migrate_sqlite_to_lilli(src_db, same_root, batch=10)


def test_fresh_dest_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Migrator refuses a dest whose brain.sqlite3 already exists.

    Regression: LILLI_STORAGE_DRIVER=lilli is forced for the dest inside
    migrate_sqlite_to_lilli, but on-disk-magic detection now overrides the
    env for an EXISTING file. Without this guard, a pre-existing stdlib file
    at the dest would silently open as stdlib (its own on-disk format), and
    the migrator would report success while never writing a lilli-format
    store -- violating the docstring contract that dst_root must be FRESH.
    """
    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    src_root.mkdir()
    src_db = _build_source_store(src_root, monkeypatch=monkeypatch, n_records=5)

    # Pre-populate the dest with a genuine stdlib sqlite3 file at the exact
    # path the migrator would open (<dst_root>/hippo/brain.sqlite3).
    dst_hippo = dst_root / "hippo"
    dst_hippo.mkdir(parents=True)
    pre_existing = sqlite3.connect(str(dst_hippo / "brain.sqlite3"))
    pre_existing.execute("CREATE TABLE sentinel (x INTEGER)")
    pre_existing.commit()
    pre_existing.close()

    monkeypatch.setenv("IAI_MCP_STORE", str(src_root))
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "stdlib")
    with pytest.raises(ValueError, match="already exists"):
        migrate_sqlite_to_lilli(src_db, dst_root, batch=10)


def test_full_report_green(migrated):
    src_db, src_root, dst_root, key, report = migrated
    rep = verify_store_equality(src_db, dst_root, key, src_root=src_root)
    assert rep.ok is True, {k: v.reason for k, v in rep.dimensions.items()}

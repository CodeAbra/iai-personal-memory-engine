from __future__ import annotations

import json
import shutil
import socket
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from tests._store_raw import open_store_raw

import numpy as np
import pytest

lancedb = pytest.importorskip("lancedb")

from scripts.migrate_lance_to_hippo import (
    main,
)
from iai_mcp.types import EMBED_DIM


def _random_vec(seed: int) -> list[float]:
    return np.random.RandomState(seed).randn(EMBED_DIM).astype(np.float32).tolist()


def _seed_lancedb_store(store_root: Path, n: int = 3) -> list[dict]:
    import pyarrow as pa

    lance_root = store_root / "lancedb"
    lance_root.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(lance_root))

    records = []
    now_str = datetime.now(timezone.utc).isoformat()
    for i in range(n):
        rid = str(uuid.uuid4())
        vec = _random_vec(seed=5000 + i)
        records.append({
            "id": rid,
            "tier": "episodic",
            "literal_surface": f"migration test record {i}",
            "aaak_index": "",
            "embedding": vec,
            "structure_hv": b"",
            "community_id": None,
            "centrality": 0.0,
            "detail_level": 1,
            "pinned": False,
            "stability": 0.0,
            "difficulty": 0.0,
            "last_reviewed": None,
            "never_decay": False,
            "never_merge": False,
            "tombstoned_at": None,
            "schema_bypass": None,
            "labile_until": None,
            "provenance_json": "[]",
            "created_at": now_str,
            "updated_at": now_str,
            "tags_json": "[]",
            "language": "en",
            "s5_trust_score": 0.5,
            "profile_modulation_gain_json": "{}",
            "schema_version": 4,
            "wing": None,
            "room": None,
            "drawer": None,
            "valence": None,
        })

    schema = pa.schema([
        pa.field("id", pa.utf8()),
        pa.field("tier", pa.utf8()),
        pa.field("literal_surface", pa.utf8()),
        pa.field("aaak_index", pa.utf8()),
        pa.field("embedding", pa.list_(pa.float32(), EMBED_DIM)),
        pa.field("structure_hv", pa.binary()),
        pa.field("community_id", pa.utf8()),
        pa.field("centrality", pa.float64()),
        pa.field("detail_level", pa.int32()),
        pa.field("pinned", pa.bool_()),
        pa.field("stability", pa.float64()),
        pa.field("difficulty", pa.float64()),
        pa.field("last_reviewed", pa.utf8()),
        pa.field("never_decay", pa.bool_()),
        pa.field("never_merge", pa.bool_()),
        pa.field("tombstoned_at", pa.utf8()),
        pa.field("schema_bypass", pa.utf8()),
        pa.field("labile_until", pa.utf8()),
        pa.field("provenance_json", pa.utf8()),
        pa.field("created_at", pa.utf8()),
        pa.field("updated_at", pa.utf8()),
        pa.field("tags_json", pa.utf8()),
        pa.field("language", pa.utf8()),
        pa.field("s5_trust_score", pa.float64()),
        pa.field("profile_modulation_gain_json", pa.utf8()),
        pa.field("schema_version", pa.int32()),
        pa.field("wing", pa.utf8()),
        pa.field("room", pa.utf8()),
        pa.field("drawer", pa.utf8()),
        pa.field("valence", pa.float64()),
    ])

    import pyarrow as pa
    table = pa.Table.from_pylist(records, schema=schema)
    db.create_table("records", data=table, mode="overwrite")

    return records


def _make_fake_trash_fn(tmp_path: Path):
    def _fake_trash(path: Path, label: str) -> Path:
        dest_dir = tmp_path / "fake_trash"
        dest_dir.mkdir(exist_ok=True)
        dest = dest_dir / label
        shutil.move(str(path), str(dest))
        return dest
    return _fake_trash


def _run_main_with_args(monkeypatch, store_root: Path, *extra_args: str) -> None:
    monkeypatch.setattr(sys, "argv", [
        "migrate_lance_to_hippo.py",
        "--store", str(store_root),
        "--yes",
        *extra_args,
    ])


@pytest.fixture()
def _no_daemon_guard(monkeypatch):
    monkeypatch.setattr(
        "scripts.migrate_lance_to_hippo.pre_flight_daemon_alive",
        lambda *a, **k: (False, None),
    )


def test_migration_happy_path(tmp_path: Path, monkeypatch, _no_daemon_guard) -> None:
    _seed_lancedb_store(tmp_path, n=3)

    monkeypatch.setattr(
        "scripts.migrate_lance_to_hippo.move_to_trash",
        _make_fake_trash_fn(tmp_path),
    )
    _run_main_with_args(monkeypatch, tmp_path)
    main()

    assert (tmp_path / "hippo").exists(), "hippo/ should be created by migration"
    assert not (tmp_path / "lancedb").exists(), "lancedb/ should be trashed after migration"

    db_path = tmp_path / "hippo" / "brain.sqlite3"
    assert db_path.exists()
    conn = open_store_raw(db_path)
    rows = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    conn.close()
    assert rows == 3, f"expected 3 migrated records, got {rows}"


def test_migration_record_vector_byte_strict(tmp_path: Path, monkeypatch, _no_daemon_guard) -> None:
    records = _seed_lancedb_store(tmp_path, n=2)

    monkeypatch.setattr(
        "scripts.migrate_lance_to_hippo.move_to_trash",
        _make_fake_trash_fn(tmp_path),
    )
    _run_main_with_args(monkeypatch, tmp_path)
    main()

    db_path = tmp_path / "hippo" / "brain.sqlite3"
    conn = open_store_raw(db_path)
    conn.row_factory = sqlite3.Row

    for rec in records:
        row = conn.execute(
            "SELECT embedding FROM records WHERE id = ?", (rec["id"],)
        ).fetchone()
        assert row is not None, f"record {rec['id']} missing from hippo"
        expected_bytes = np.array(rec["embedding"], dtype=np.float32).tobytes()
        actual_bytes = bytes(row["embedding"])
        assert actual_bytes == expected_bytes, (
            f"Embedding bytes mismatch for {rec['id']}: "
            f"expected {expected_bytes[:16].hex()} got {actual_bytes[:16].hex()}"
        )
    conn.close()


def test_migration_hnsw_rebuilt_and_loadable(tmp_path: Path, monkeypatch, _no_daemon_guard) -> None:
    import hnswlib

    n = 4
    _seed_lancedb_store(tmp_path, n=n)

    monkeypatch.setattr(
        "scripts.migrate_lance_to_hippo.move_to_trash",
        _make_fake_trash_fn(tmp_path),
    )
    _run_main_with_args(monkeypatch, tmp_path)
    main()

    hnsw_path = tmp_path / "hippo" / "records.hnsw"
    assert hnsw_path.exists(), "records.hnsw must be present after migration"

    idx = hnswlib.Index(space="cosine", dim=EMBED_DIM)
    idx.load_index(str(hnsw_path))
    assert idx.get_current_count() == n, (
        f"hnsw index should contain {n} vectors, got {idx.get_current_count()}"
    )


def test_migration_rollback_restores_backup(tmp_path: Path, monkeypatch) -> None:
    _seed_lancedb_store(tmp_path, n=2)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    backup_path = tmp_path / f"lancedb.pre-migrate-{ts}"
    shutil.copytree(str(tmp_path / "lancedb"), str(backup_path))
    hippo_path = tmp_path / "hippo"
    hippo_path.mkdir()

    fail_json = tmp_path / f".migrate-FAILED-{ts}.json"
    fail_json.write_text(json.dumps({"ts": ts, "backup_path": str(backup_path)}))

    fake_trash_dir = tmp_path / "fake_trash"
    fake_trash_dir.mkdir()

    monkeypatch.setattr(
        "scripts.migrate_lance_to_hippo.move_to_trash",
        _make_fake_trash_fn(tmp_path),
    )
    monkeypatch.setattr(sys, "argv", [
        "migrate_lance_to_hippo.py",
        "--store", str(tmp_path),
        "--rollback",
        "--rollback-ts", ts,
    ])
    main()

    assert (tmp_path / "lancedb").exists(), "lancedb/ should be restored after rollback"
    assert not hippo_path.exists(), "hippo/ should be trashed after rollback"


def test_migration_refuses_when_daemon_socket_responds(tmp_path: Path, monkeypatch) -> None:
    import tempfile as _tempfile

    with _tempfile.TemporaryDirectory(prefix="iai_mig_") as short_tmp:
        store_root = Path(short_tmp)
        _seed_lancedb_store(store_root, n=1)

        sock_path = store_root / ".daemon.sock"
        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server_sock.bind(str(sock_path))
            server_sock.listen(1)

            monkeypatch.setattr(
                "scripts.migrate_lance_to_hippo.move_to_trash",
                _make_fake_trash_fn(tmp_path),
            )
            monkeypatch.setattr(sys, "argv", [
                "migrate_lance_to_hippo.py",
                "--store", str(store_root),
                "--yes",
            ])
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2, (
                f"expected exit(2) when daemon socket responds, got {exc_info.value.code}"
            )
        finally:
            server_sock.close()
            if sock_path.exists():
                sock_path.unlink()


def test_migration_fresh_run_fails_on_duplicates(tmp_path: Path, monkeypatch, _no_daemon_guard) -> None:
    _seed_lancedb_store(tmp_path, n=2)

    fake_trash_fn = _make_fake_trash_fn(tmp_path)
    monkeypatch.setattr("scripts.migrate_lance_to_hippo.move_to_trash", fake_trash_fn)

    import scripts.migrate_lance_to_hippo as _mig_mod

    _orig_stream_copy = _mig_mod.stream_copy_table

    def _fake_stream_copy(lance_db, hippo_conn, table_name, batch_size, dry_run=False):
        ins, dup = _orig_stream_copy(lance_db, hippo_conn, table_name, batch_size, dry_run)
        if table_name == "records" and ins > 0:
            return max(0, ins - 1), dup + 1
        return ins, dup

    monkeypatch.setattr("scripts.migrate_lance_to_hippo.stream_copy_table", _fake_stream_copy)
    _run_main_with_args(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 5, (
        f"expected exit(5) on fresh run with simulated duplicates, got {exc_info.value.code}"
    )


def test_migration_resume_flag_allows_duplicates(tmp_path: Path, monkeypatch, _no_daemon_guard) -> None:
    _seed_lancedb_store(tmp_path, n=2)

    fake_trash_fn = _make_fake_trash_fn(tmp_path)
    monkeypatch.setattr("scripts.migrate_lance_to_hippo.move_to_trash", fake_trash_fn)

    import scripts.migrate_lance_to_hippo as _mig_mod

    _orig_stream_copy = _mig_mod.stream_copy_table

    def _fake_stream_with_dup(lance_db, hippo_conn, table_name, batch_size, dry_run=False):
        ins, dup = _orig_stream_copy(lance_db, hippo_conn, table_name, batch_size, dry_run)
        if table_name == "records" and ins > 0:
            return max(0, ins - 1), dup + 1
        return ins, dup

    monkeypatch.setattr("scripts.migrate_lance_to_hippo.stream_copy_table", _fake_stream_with_dup)

    _run_main_with_args(monkeypatch, tmp_path, "--resume")
    main()

    assert (tmp_path / "hippo").exists()


def test_migration_failure_preserves_backup_and_writes_json(
    tmp_path: Path, monkeypatch, _no_daemon_guard
) -> None:
    _seed_lancedb_store(tmp_path, n=2)

    monkeypatch.setattr(
        "scripts.migrate_lance_to_hippo.verify_record_parity",
        lambda lance_db, hippo_conn: [{"table": "records", "id": "fake", "reason": "test"}],
    )
    monkeypatch.setattr(
        "scripts.migrate_lance_to_hippo.move_to_trash",
        _make_fake_trash_fn(tmp_path),
    )
    _run_main_with_args(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 4, (
        f"expected exit(4) on verification failure, got {exc_info.value.code}"
    )

    backups = list(tmp_path.glob("lancedb.pre-migrate-*"))
    assert backups, "backup should be preserved on verification failure"

    fail_jsons = list(tmp_path.glob(".migrate-FAILED-*.json"))
    assert fail_jsons, ".migrate-FAILED-*.json should be written on failure"
    payload = json.loads(fail_jsons[0].read_text())
    assert "mismatches" in payload
    assert payload["mismatches"]


def test_migration_dry_run_keeps_lancedb(tmp_path: Path, monkeypatch, _no_daemon_guard) -> None:
    _seed_lancedb_store(tmp_path, n=2)

    monkeypatch.setattr(
        "scripts.migrate_lance_to_hippo.move_to_trash",
        _make_fake_trash_fn(tmp_path),
    )
    _run_main_with_args(monkeypatch, tmp_path, "--dry-run")
    main()

    assert (tmp_path / "lancedb").exists(), "lancedb/ must be untouched after --dry-run"

    hippo_db_path = tmp_path / "hippo" / "brain.sqlite3"
    if hippo_db_path.exists():
        conn = open_store_raw(hippo_db_path)
        count = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        conn.close()
        assert count == 0, (
            f"dry-run must not insert records into hippo; found {count} rows"
        )

"""backup() must capture the actual database — the engine store under
hippo/ with its WAL — not a stale hardcoded file list; rebuildable
sidecars and transient litter stay out. restore() must hand back a byte-
identical database."""
from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from iai_mcp.backup import backup, restore


@pytest.fixture()
def synthetic_store(tmp_path, monkeypatch):
    store = tmp_path / "store"
    hippo = store / "hippo"
    hippo.mkdir(parents=True)
    (hippo / "brain.sqlite3").write_bytes(b"LILLI storage 1\x00" + b"D" * 4096)
    (hippo / "brain.sqlite3-wal").write_bytes(b"W" * 128)
    (hippo / "brain.sqlite3-shm").write_bytes(b"S" * 64)
    (hippo / ".max-created-at").write_text("2026-07-14T00:00:00+00:00")
    # Rebuildable + transient: must stay OUT of the archive.
    (hippo / "brain.sqlite3.ro").write_bytes(b"R" * 2048)
    (hippo / "records.hnsw").write_bytes(b"H" * 2048)
    (hippo / "records.colindex").write_bytes(b"C" * 2048)
    (hippo / ".lock").write_bytes(b"")
    (hippo / "tmpabc.hnsw.child.tmp").write_bytes(b"T" * 32)

    (store / "config.json").write_text("{}")
    bank = store / ".memory-bank" / "recent"
    bank.mkdir(parents=True)
    (bank / "window-2026-07-14.jsonl").write_text("enc-line\n")

    monkeypatch.setenv("IAI_MCP_STORE", str(store))
    return store


def test_backup_captures_the_database_and_skips_rebuildables(synthetic_store, tmp_path):
    archive = backup(tmp_path / "b.tar.gz")
    with tarfile.open(archive) as tar:
        names = set(tar.getnames())

    assert "hippo/brain.sqlite3" in names, "the database itself must be in the backup"
    assert "hippo/brain.sqlite3-wal" in names, "WAL travels with the DB for crash consistency"
    assert "hippo/brain.sqlite3-shm" in names
    assert "hippo/.max-created-at" in names
    assert any(n.startswith(".memory-bank") for n in names), "the bank layers are memory too"
    assert "config.json" in names

    for excluded in (
        "hippo/brain.sqlite3.ro",
        "hippo/records.hnsw",
        "hippo/records.colindex",
        "hippo/.lock",
        "hippo/tmpabc.hnsw.child.tmp",
    ):
        assert excluded not in names, f"{excluded} is rebuildable/transient and must stay out"


def test_restore_round_trips_the_database(synthetic_store, tmp_path):
    original = (synthetic_store / "hippo" / "brain.sqlite3").read_bytes()
    archive = backup(tmp_path / "b.tar.gz")

    target = tmp_path / "restored"
    restore(archive, target)

    restored = (target / "hippo" / "brain.sqlite3").read_bytes()
    assert restored == original, "restore must hand back a byte-identical database"
    assert (target / ".memory-bank" / "recent" / "window-2026-07-14.jsonl").exists()


def test_backup_warns_when_no_database_present(tmp_path, monkeypatch, caplog):
    empty = tmp_path / "empty-store"
    empty.mkdir()
    (empty / "config.json").write_text("{}")
    monkeypatch.setenv("IAI_MCP_STORE", str(empty))

    import logging

    with caplog.at_level(logging.WARNING, logger="iai_mcp.backup"):
        backup(tmp_path / "e.tar.gz")
    assert any("no database file found" in r.message for r in caplog.records), (
        "a database-less backup must be loudly flagged, never silent"
    )

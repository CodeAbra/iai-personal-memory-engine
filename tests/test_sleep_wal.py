from __future__ import annotations

import json

import pytest

from iai_mcp.sleep_wal import SleepWAL, TOMBSTONE_TTL_DAYS

@pytest.fixture
def wal(tmp_path):
    return SleepWAL(path=tmp_path / ".sleep-wal.jsonl")

class TestWALBasics:
    def test_begin_creates_pending_entry(self, wal):
        entry = wal.begin("tombstone", ["id-1", "id-2"])
        assert entry.status == "pending"
        assert entry.operation == "tombstone"
        assert entry.target_ids == ["id-1", "id-2"]
        assert wal.path.exists()

    def test_commit_marks_entry(self, wal):
        entry = wal.begin("edge_prune", ["e-1"])
        wal.commit(entry)
        assert entry.status == "committed"

    def test_rollback_marks_entry(self, wal):
        entry = wal.begin("optimize_drop", ["r-1"])
        wal.rollback(entry)
        assert entry.status == "rolled_back"

    def test_pending_entries_returns_uncommitted(self, wal):
        e1 = wal.begin("tombstone", ["a"])
        e2 = wal.begin("edge_prune", ["b"])
        wal.commit(e1)
        pending = wal.pending_entries()
        assert len(pending) == 1
        assert pending[0].id == e2.id

    def test_no_pending_after_all_committed(self, wal):
        e1 = wal.begin("tombstone", ["x"])
        e2 = wal.begin("edge_prune", ["y"])
        wal.commit(e1)
        wal.commit(e2)
        assert wal.pending_entries() == []

class TestWALCrashRecovery:
    def test_pending_survives_reopen(self, tmp_path):
        path = tmp_path / ".sleep-wal.jsonl"
        wal1 = SleepWAL(path=path)
        entry = wal1.begin("consolidate_merge", ["m-1", "m-2"])
        del wal1

        wal2 = SleepWAL(path=path)
        pending = wal2.pending_entries()
        assert len(pending) == 1
        assert pending[0].operation == "consolidate_merge"
        assert pending[0].target_ids == ["m-1", "m-2"]

    def test_committed_not_in_pending_after_reopen(self, tmp_path):
        path = tmp_path / ".sleep-wal.jsonl"
        wal1 = SleepWAL(path=path)
        entry = wal1.begin("tombstone", ["t-1"])
        wal1.commit(entry)
        del wal1

        wal2 = SleepWAL(path=path)
        assert wal2.pending_entries() == []

class TestWALDryRun:
    def test_dry_run_flag(self, tmp_path, monkeypatch):
        monkeypatch.setenv("IAI_MCP_ERASURE_DRY_RUN", "true")
        wal = SleepWAL(path=tmp_path / ".sleep-wal.jsonl")
        assert wal._dry_run is True

    def test_dry_run_still_writes_wal(self, tmp_path, monkeypatch):
        monkeypatch.setenv("IAI_MCP_ERASURE_DRY_RUN", "1")
        wal = SleepWAL(path=tmp_path / ".sleep-wal.jsonl")
        entry = wal.begin("tombstone", ["dry-1"])
        assert entry.status == "pending"
        assert wal.path.exists()

class TestWALCleanup:
    def test_cleanup_removes_old_committed(self, tmp_path):
        path = tmp_path / ".sleep-wal.jsonl"
        wal = SleepWAL(path=path)
        old_entry = {
            "id": "old-1",
            "operation": "tombstone",
            "target_ids": ["r-old"],
            "ts": "2020-01-01T00:00:00+00:00",
            "status": "committed",
            "metadata": {},
        }
        path.write_text(json.dumps(old_entry) + "\n")
        removed = wal.cleanup(max_age_hours=1)
        assert removed == 1

    def test_cleanup_keeps_pending(self, tmp_path):
        path = tmp_path / ".sleep-wal.jsonl"
        old_pending = {
            "id": "pending-1",
            "operation": "edge_prune",
            "target_ids": ["e-1"],
            "ts": "2020-01-01T00:00:00+00:00",
            "status": "pending",
            "metadata": {},
        }
        path.write_text(json.dumps(old_pending) + "\n")
        wal = SleepWAL(path=path)
        removed = wal.cleanup(max_age_hours=1)
        assert removed == 0
        assert len(wal.pending_entries()) == 1

class TestTombstoneTTL:
    def test_ttl_is_7_days(self):
        assert TOMBSTONE_TTL_DAYS == 7

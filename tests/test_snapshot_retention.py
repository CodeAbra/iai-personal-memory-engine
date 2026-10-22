"""Tests for the migration-snapshot retention and orphan-sweep command."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest

# Captured at collection time, before the autouse fixture below replaces the
# module attribute -- gives real-probe tests a handle on the genuine
# implementation to restore per-test.
from iai_mcp.migrate._snapshot_retention import _probe_store_lock as _REAL_PROBE_STORE_LOCK


@pytest.fixture(autouse=True)
def _unlocked_store(monkeypatch):
    import iai_mcp.migrate._snapshot_retention as retention_mod

    monkeypatch.setattr(retention_mod, "_probe_store_lock", lambda root: None)


def _make_snapshot(root: Path, name: str, *, mtime: float | None = None) -> Path:
    d = root / name
    d.mkdir()
    (d / "payload.bin").write_bytes(b"x" * 16)
    if mtime is not None:
        os.utime(d, (mtime, mtime))
    return d


def _make_live_store(root: Path) -> Path:
    d = root / "hippo"
    d.mkdir()
    (d / "hippo.db").write_bytes(b"live")
    return d


def test_dry_run_lists_and_deletes_nothing(tmp_path):
    from iai_mcp.migrate._snapshot_retention import prune_snapshots

    _make_live_store(tmp_path)
    _make_snapshot(tmp_path, "hippo-pre-cleanup-20260901T000000Z")
    _make_snapshot(tmp_path, "hippo-pre-idem-dedup-20260902T000000Z")
    _make_snapshot(tmp_path, "hippo-pre-v17-restart-20260827")

    before = sorted(p.name for p in tmp_path.iterdir())
    summary = prune_snapshots(tmp_path, keep=1)
    after = sorted(p.name for p in tmp_path.iterdir())

    assert before == after
    assert summary["mode"] == "dry-run"
    assert summary["snapshots_found"] == 3
    assert len(summary["kept"]) == 1
    assert len(summary["planned_prune"]) == 2


def test_snapshots_ordered_newest_first_by_mtime(tmp_path):
    from iai_mcp.migrate._snapshot_retention import list_snapshots

    now = 1_800_000_000.0
    oldest = _make_snapshot(tmp_path, "hippo-pre-v17-restart-20260827", mtime=now)
    middle = _make_snapshot(
        tmp_path, "hippo-pre-cleanup-20260901T000000Z", mtime=now + 100
    )
    newest = _make_snapshot(
        tmp_path, "hippo-pre-idem-dedup-20260902T000000Z", mtime=now + 200
    )

    result = list_snapshots(tmp_path)
    assert [s.path.name for s in result] == [newest.name, middle.name, oldest.name]


def test_always_keeps_at_least_one_when_keep_zero(tmp_path):
    from iai_mcp.migrate._snapshot_retention import prune_snapshots

    _make_live_store(tmp_path)
    _make_snapshot(tmp_path, "hippo-pre-cleanup-20260901T000000Z")
    _make_snapshot(tmp_path, "hippo-pre-idem-dedup-20260902T000000Z")

    summary = prune_snapshots(tmp_path, keep=0)
    assert len(summary["kept"]) == 1


def test_live_store_excluded_by_inode_identity(tmp_path):
    from iai_mcp.migrate._snapshot_retention import prune_snapshots

    live = _make_live_store(tmp_path)
    _make_snapshot(tmp_path, "hippo-pre-cleanup-20260901T000000Z")
    alias = tmp_path / "hippo-pre-alias"
    alias.symlink_to(live, target_is_directory=True)

    summary = prune_snapshots(tmp_path, keep=1)
    all_named = set(summary["kept"]) | set(summary["planned_prune"])
    assert "hippo-pre-alias" not in all_named


def test_prune_snapshots_registered(tmp_path):
    from iai_mcp.cli import _build_parser
    from iai_mcp.cli._maintenance import cmd_prune_snapshots

    parser = _build_parser()
    ns = parser.parse_args(
        ["prune-snapshots", "--store-path", str(tmp_path), "--keep", "2"]
    )
    assert ns.func is cmd_prune_snapshots
    assert ns.keep == 2


def test_dry_run_and_apply_are_mutually_exclusive():
    from iai_mcp.cli import _build_parser

    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["prune-snapshots", "--dry-run", "--apply"])


def test_prune_snapshots_default_keep_is_three():
    from iai_mcp.cli import _build_parser

    ns = _build_parser().parse_args(["prune-snapshots"])
    assert ns.keep == 3
    assert ns.apply is False


def test_apply_prunes_oldest_beyond_keep(tmp_path):
    from iai_mcp.migrate._snapshot_retention import prune_snapshots

    _make_live_store(tmp_path)
    now = 1_800_000_000.0
    s1 = _make_snapshot(tmp_path, "hippo-pre-cleanup-20260901T000000Z", mtime=now)
    s2 = _make_snapshot(
        tmp_path, "hippo-pre-idem-dedup-20260902T000000Z", mtime=now + 100
    )
    s3 = _make_snapshot(
        tmp_path, "hippo-pre-blob-quarantine-20260903T000000Z", mtime=now + 200
    )
    s4 = _make_snapshot(
        tmp_path, "hippo-pre-directive-sweep-20260904T000000Z", mtime=now + 300
    )

    summary = prune_snapshots(tmp_path, keep=2, apply=True)

    assert summary["mode"] == "apply"
    assert not s1.exists()
    assert not s2.exists()
    assert s3.exists()
    assert s4.exists()
    assert (tmp_path / "hippo").exists()
    assert sorted(summary["pruned"]) == sorted([s1.name, s2.name])


def test_apply_always_keeps_at_least_one(tmp_path):
    from iai_mcp.migrate._snapshot_retention import prune_snapshots

    _make_live_store(tmp_path)
    now = 1_800_000_000.0
    _make_snapshot(tmp_path, "hippo-pre-cleanup-20260901T000000Z", mtime=now)
    _make_snapshot(tmp_path, "hippo-pre-idem-dedup-20260902T000000Z", mtime=now + 100)
    s3 = _make_snapshot(
        tmp_path, "hippo-pre-blob-quarantine-20260903T000000Z", mtime=now + 200
    )

    prune_snapshots(tmp_path, keep=0, apply=True)

    remaining = sorted(p.name for p in tmp_path.glob("hippo-pre-*"))
    assert remaining == [s3.name]


def test_locked_store_skipped(tmp_path, monkeypatch):
    import iai_mcp.migrate._snapshot_retention as retention_mod
    from iai_mcp.cli._maintenance import cmd_prune_snapshots
    from iai_mcp.migrate._snapshot_retention import prune_snapshots

    monkeypatch.setattr(retention_mod, "_probe_store_lock", lambda root: "pid=999")

    _make_live_store(tmp_path)
    snap = _make_snapshot(tmp_path, "hippo-pre-cleanup-20260901T000000Z")

    before = sorted(p.name for p in tmp_path.iterdir())
    dry = prune_snapshots(tmp_path, keep=1, apply=False)
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    assert dry.get("locked") is True

    applied = prune_snapshots(tmp_path, keep=1, apply=True)
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    assert applied.get("locked") is True
    assert snap.exists()

    ns = argparse.Namespace(store_path=str(tmp_path), keep=1, apply=True)
    assert cmd_prune_snapshots(ns) == 1


def test_probe_store_lock_real_no_lock_prune_proceeds(tmp_path, monkeypatch):
    """Exercises the genuine _probe_store_lock body (no monkeypatch of the
    probe itself) against a real, closed store: prune must proceed."""
    import iai_mcp.migrate._snapshot_retention as retention_mod
    from iai_mcp.migrate._snapshot_retention import prune_snapshots
    from iai_mcp.store import MemoryStore

    monkeypatch.setattr(retention_mod, "_probe_store_lock", _REAL_PROBE_STORE_LOCK)

    store = MemoryStore(path=tmp_path)
    store.close()
    now = 1_800_000_000.0
    old = _make_snapshot(tmp_path, "hippo-pre-a-20260901T000000Z", mtime=now)
    new = _make_snapshot(tmp_path, "hippo-pre-b-20260902T000000Z", mtime=now + 100)

    summary = prune_snapshots(tmp_path, keep=1, apply=True)

    assert summary["locked"] is False
    assert not old.exists()
    assert new.exists()


def test_probe_store_lock_real_exclusive_lock_skips_prune(tmp_path, monkeypatch):
    """Exercises the genuine _probe_store_lock body against a store held
    EXCLUSIVE by a real second in-process MemoryStore: prune must skip.

    Mutation check: if _probe_store_lock's body were replaced with an
    unconditional ``return None``, this test fails (both snapshots would be
    pruned down to ``keep=1`` instead of the lock skipping the run outright).
    """
    import iai_mcp.migrate._snapshot_retention as retention_mod
    from iai_mcp.migrate._snapshot_retention import prune_snapshots
    from iai_mcp.store import MemoryStore

    monkeypatch.setattr(retention_mod, "_probe_store_lock", _REAL_PROBE_STORE_LOCK)

    holder = MemoryStore(path=tmp_path)
    now = 1_800_000_000.0
    old = _make_snapshot(tmp_path, "hippo-pre-a-20260901T000000Z", mtime=now)
    new = _make_snapshot(tmp_path, "hippo-pre-b-20260902T000000Z", mtime=now + 100)
    try:
        summary = prune_snapshots(tmp_path, keep=1, apply=True)
    finally:
        holder.close()

    assert summary["locked"] is True
    assert old.exists()
    assert new.exists()


def test_dry_run_against_nonexistent_store_creates_nothing(tmp_path, monkeypatch):
    """Exercises the genuine _probe_store_lock body (no monkeypatch of the
    probe itself) against a store-path that has never had a live store at
    all: dry-run must not materialize a hippo directory as a side effect of
    probing for a lock, and must still plan the snapshots it finds."""
    import iai_mcp.migrate._snapshot_retention as retention_mod
    from iai_mcp.migrate._snapshot_retention import prune_snapshots

    monkeypatch.setattr(retention_mod, "_probe_store_lock", _REAL_PROBE_STORE_LOCK)

    _make_snapshot(tmp_path, "hippo-pre-cleanup-20260901T000000Z")
    _make_snapshot(tmp_path, "hippo-pre-idem-dedup-20260902T000000Z")

    assert not (tmp_path / "hippo").exists()

    summary = prune_snapshots(tmp_path, keep=1, apply=False)

    assert not (tmp_path / "hippo").exists()
    assert summary["mode"] == "dry-run"
    assert summary["locked"] is False
    assert summary["snapshots_found"] == 2
    assert len(summary["planned_prune"]) == 1


def test_orphan_sweep_dry_run_and_apply(tmp_path):
    from iai_mcp.migrate._snapshot_retention import prune_snapshots

    live = _make_live_store(tmp_path)
    newest = _make_snapshot(tmp_path, "hippo-pre-cleanup-20260901T000000Z")
    (tmp_path / "lancedb").mkdir()
    (tmp_path / "lancedb" / "table.lance").write_bytes(b"x")
    (tmp_path / "lancedb-orphan-daemon-events").mkdir()
    (tmp_path / "lancedb-orphan-daemon-events" / "f").write_bytes(b"x")
    (tmp_path / "lance").mkdir()

    dry = prune_snapshots(tmp_path, keep=1, apply=False)
    assert sorted(dry["orphans_found"]) == sorted(
        ["lancedb", "lancedb-orphan-daemon-events", "lance"]
    )
    assert (tmp_path / "lancedb").exists()

    applied = prune_snapshots(tmp_path, keep=1, apply=True)
    assert sorted(applied["orphans_removed"]) == sorted(
        ["lancedb", "lancedb-orphan-daemon-events", "lance"]
    )
    assert not (tmp_path / "lancedb").exists()
    assert not (tmp_path / "lancedb-orphan-daemon-events").exists()
    assert not (tmp_path / "lance").exists()
    assert live.exists()
    assert newest.exists()


def test_refuses_snapshot_shaped_store_root(tmp_path):
    from iai_mcp.cli._maintenance import cmd_prune_snapshots
    from iai_mcp.migrate._snapshot_retention import prune_snapshots

    bad_root = tmp_path / "hippo-pre-cleanup-20260901T000000Z"
    bad_root.mkdir()

    with pytest.raises(ValueError):
        prune_snapshots(bad_root, keep=1, apply=False)
    with pytest.raises(ValueError):
        prune_snapshots(bad_root, keep=1, apply=True)

    ns = argparse.Namespace(store_path=str(bad_root), keep=1, apply=False)
    assert cmd_prune_snapshots(ns) == 2


def test_refuses_hippo_shaped_store_root(tmp_path):
    from iai_mcp.cli._maintenance import cmd_prune_snapshots
    from iai_mcp.migrate._snapshot_retention import prune_snapshots

    bad_root = tmp_path / "hippo"
    bad_root.mkdir()

    with pytest.raises(ValueError):
        prune_snapshots(bad_root, keep=1, apply=False)
    with pytest.raises(ValueError):
        prune_snapshots(bad_root, keep=1, apply=True)

    ns = argparse.Namespace(store_path=str(bad_root), keep=1, apply=False)
    assert cmd_prune_snapshots(ns) == 2


def test_dir_size_bytes_skips_files_that_vanish_mid_scan(tmp_path, monkeypatch):
    """A file that stats successfully during the is_file() check but raises
    on the explicit .stat() call right after (a vanish-mid-scan race) must
    not crash sizing -- dry-run size accounting is advisory."""
    from iai_mcp.migrate._snapshot_retention import prune_snapshots

    _make_live_store(tmp_path)
    snap = _make_snapshot(tmp_path, "hippo-pre-cleanup-20260901T000000Z")
    flaky_file = snap / "flaky.bin"
    flaky_file.write_bytes(b"x" * 8)

    real_stat = Path.stat
    calls: dict[str, int] = {}

    def _flaky_stat(self, *a, **kw):
        if self.name == "flaky.bin":
            calls[self.name] = calls.get(self.name, 0) + 1
            if calls[self.name] > 1:
                raise OSError("vanished mid-scan")
        return real_stat(self, *a, **kw)

    monkeypatch.setattr(Path, "stat", _flaky_stat)

    summary = prune_snapshots(tmp_path, keep=1)

    assert summary["mode"] == "dry-run"
    assert summary["snapshots_found"] == 1


def test_apply_continues_past_one_bad_dir(tmp_path, monkeypatch):
    import shutil as _shutil

    import iai_mcp.migrate._snapshot_retention as retention_mod
    from iai_mcp.migrate._snapshot_retention import prune_snapshots

    _make_live_store(tmp_path)
    now = 1_800_000_000.0
    bad = _make_snapshot(tmp_path, "hippo-pre-cleanup-20260901T000000Z", mtime=now)
    good = _make_snapshot(
        tmp_path, "hippo-pre-idem-dedup-20260902T000000Z", mtime=now + 100
    )
    _make_snapshot(
        tmp_path, "hippo-pre-blob-quarantine-20260903T000000Z", mtime=now + 200
    )

    real_rmtree = _shutil.rmtree

    def _flaky_rmtree(path, *a, **kw):
        if Path(path).name == bad.name:
            raise OSError("permission denied")
        return real_rmtree(path, *a, **kw)

    monkeypatch.setattr(retention_mod.shutil, "rmtree", _flaky_rmtree)

    summary = prune_snapshots(tmp_path, keep=1, apply=True)

    assert bad.exists()
    assert not good.exists()
    assert summary["errors"]
    assert good.name in summary["pruned"]

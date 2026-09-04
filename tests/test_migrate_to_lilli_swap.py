"""In-place swap of a verified native copy into the live store root.

Builds a legacy-format source store in the mode a default install actually
runs in -- a real crypto key FILE at the store root, no passphrase in the
environment, write-ahead-log sidecars present next to the backing file -- and
exercises ``swap_migrated_store`` against it: the happy path, dry-run's
no-write guarantee, every refusal, the symlinked-root case, and the marker
that turns an interrupted swap into a refusal at every daemon-start path.

Hermetic: tmp HOME + tmp store, single process, no xdist. Fixtures use
``alice``.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pytest

from iai_mcp import _sqlite_stdlib

# Connections deliberately left open by the fixture builder so their
# write-ahead-log sidecars are genuinely present when the builder returns --
# closing checkpoints and unlinks them (the literal subject of the defect
# this migration path exists to route around). Held here so the objects
# outlive the builder call; never explicitly closed (process teardown or GC
# at interpreter exit reclaims them, same as any other test-local handle).
_KEEP_ALIVE: list = []


def _make_record(i: int, vec: list[float], *, created_at: datetime):
    from iai_mcp.types import MemoryRecord

    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=f"alice recorded distinct fact number {i} about swapping stores",
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


_DB_MAGIC = b"SQLite format 3\x00"


def _build_legacy_source_with_file_key(
    root: Path, *, monkeypatch: pytest.MonkeyPatch, n: int = 8
) -> Path:
    """A legacy-format store at ``root``: sidecars present, a real crypto key
    FILE at the root, no passphrase in the environment -- the mode a default
    install actually runs in. Returns ``root``.
    """
    # Delete the passphrase BEFORE any key is derived, so the autouse
    # conftest fixture's value is gone before key derivation runs.
    monkeypatch.delenv("IAI_MCP_CRYPTO_PASSPHRASE", raising=False)

    from iai_mcp.crypto import CryptoKey

    # rotate() persists a FRESH key file when none exists; get_or_create()
    # only reads an existing key or derives one from a passphrase -- with
    # neither present it raises, by design.
    CryptoKey(store_root=root).rotate()

    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "stdlib")
    monkeypatch.setenv("IAI_MCP_STORE", str(root))

    from iai_mcp.store import MemoryStore, flush_record_buffer, flush_edge_buffer
    from iai_mcp.events import write_event, flush_event_buffer

    store = MemoryStore(root)
    rng = np.random.RandomState(13)
    ids: list = []
    base = datetime(2026, 6, 15, 8, 0, 0, tzinfo=timezone.utc)
    for i in range(n):
        vec = rng.randn(384).tolist()
        rec = _make_record(i, vec, created_at=base + timedelta(minutes=i))
        ids.append(rec.id)
        store.insert(rec)
    flush_record_buffer(store)
    if len(ids) >= 4:
        store.boost_edges([(ids[0], ids[1]), (ids[2], ids[3])])
        flush_edge_buffer(store)
    write_event(store, kind="probe_event", data={"n": 1})
    flush_event_buffer(store)

    db_path = root / "hippo" / "brain.sqlite3"
    with open(db_path, "rb") as fh:
        header = fh.read(len(_DB_MAGIC))
    assert header == _DB_MAGIC, f"expected SQLite header magic, got {header!r}"

    wal_path = root / "hippo" / "brain.sqlite3-wal"
    shm_path = root / "hippo" / "brain.sqlite3-shm"
    if not (wal_path.exists() and shm_path.exists()):
        # A prior code path already checkpointed them away -- re-open and
        # write so they are genuinely present, then keep THIS connection
        # alive too.
        raw = _sqlite_stdlib.connect(str(db_path))
        raw.execute("PRAGMA journal_mode=WAL")
        raw.execute(
            "INSERT INTO events (id, kind, severity, domain, ts, data_json, "
            "session_id, source_ids_json) VALUES (?, 'probe_event', '', '', "
            "?, '', 'system', '[]')",
            (str(uuid.uuid4()), datetime.now(timezone.utc).isoformat()),
        )
        raw.commit()
        _KEEP_ALIVE.append(raw)

    _KEEP_ALIVE.append(store)

    assert wal_path.exists(), "sidecar -wal must be present when the fixture returns"
    assert shm_path.exists(), "sidecar -shm must be present when the fixture returns"

    return root


def _read_store_mapping(root: Path, *, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Open ``root`` through the ordinary store entry point -- driver resolved
    from the on-disk header, never told which one to use -- and return
    ``{record id: literal-surface digest}``. Usable against a legacy root or
    a native root identically.
    """
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    monkeypatch.setenv("IAI_MCP_STORE", str(root))

    from iai_mcp.store import MemoryStore

    store = MemoryStore(root)
    try:
        return {
            str(r.id): hashlib.sha256(r.literal_surface.encode("utf-8")).hexdigest()
            for r in store.all_records()
        }
    finally:
        store.db.close()


def test_swap_migrated_store_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "live"
    root.mkdir()
    _build_legacy_source_with_file_key(root, monkeypatch=monkeypatch, n=8)
    src_db = root / "hippo" / "brain.sqlite3"

    before = _read_store_mapping(root, monkeypatch=monkeypatch)
    assert len(before) == 8

    from iai_mcp.migrate import swap_migrated_store

    summary = swap_migrated_store(str(src_db), apply=True)
    assert summary["swapped"] is True, summary

    from iai_mcp.hippo._db import _resolve_effective_driver

    assert _resolve_effective_driver(str(src_db)) == "lilli"

    after = _read_store_mapping(root, monkeypatch=monkeypatch)
    assert after == before


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _backup_dir(root: Path) -> Path:
    return root / f"hippo.sqlite-backup-{_today_utc()}"


def _dir_snapshot(root: Path) -> list[tuple[str, int]]:
    """(relative posix path, byte size) for every file under root, sorted --
    used to prove dry-run wrote nothing."""
    return sorted(
        (str(p.relative_to(root)), p.stat().st_size)
        for p in root.rglob("*")
        if p.is_file()
    )


def _read_legacy_hippo_dir_mapping(hippo_dir: Path, *, key: bytes) -> dict[str, str]:
    """Read a legacy hippo/ directory's records directly (no root wrapper) by
    decrypting ``literal_surface`` with ``key`` -- used for the dated backup,
    which holds the store's internals directly rather than nested under a
    second hippo/ subdirectory.
    """
    from iai_mcp.crypto import decrypt_field

    conn = _sqlite_stdlib.connect(
        f"file:{hippo_dir / 'brain.sqlite3'}?mode=ro", uri=True
    )
    conn.row_factory = _sqlite_stdlib.Row
    try:
        rows = conn.execute("SELECT id, literal_surface FROM records").fetchall()
    finally:
        conn.close()
    out: dict[str, str] = {}
    for row in rows:
        rid = row["id"]
        plaintext = decrypt_field(
            row["literal_surface"], key, associated_data=rid.lower().encode("ascii")
        )
        out[rid] = hashlib.sha256(plaintext.encode("utf-8")).hexdigest()
    return out


def _read_native_root_mapping_with_planted_key(
    root: Path, *, key: bytes, monkeypatch: pytest.MonkeyPatch
) -> dict[str, str]:
    """Read a native store root's mapping through a fresh open, planting
    ``key`` as its key file for the read when the root carries none of its
    own -- a migrated staging tree has no key file of its own (only the live
    root's key covers it, once the swap that never happened would have
    completed).
    """
    from iai_mcp.crypto import _KEY_FILE_NAME, write_key_material

    key_path = root / _KEY_FILE_NAME
    planted = False
    if not key_path.exists():
        write_key_material(key_path, key)
        planted = True
    try:
        return _read_store_mapping(root, monkeypatch=monkeypatch)
    finally:
        if planted:
            try:
                key_path.unlink()
            except OSError:
                pass


def _plant_marker(root: Path, *, pid, timestamp: str | None = None) -> Path:
    """Write a swap-in-progress marker by hand, in the exact shape the swap's
    own writer produces, with a caller-chosen pid (so a dead-pid case is
    reachable without waiting for a real process to die)."""
    import json as _json

    from iai_mcp.migrate._to_lilli_swap import MARKER_FILE_NAME

    marker = root / MARKER_FILE_NAME
    payload = {
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "pid": pid,
    }
    marker.write_text(_json.dumps(payload), encoding="utf-8")
    return marker


def test_dry_run_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "live"
    root.mkdir()
    _build_legacy_source_with_file_key(root, monkeypatch=monkeypatch, n=5)
    src_db = root / "hippo" / "brain.sqlite3"

    entries_before = sorted(p.name for p in root.iterdir())
    hippo_mtime_before = (root / "hippo").stat().st_mtime_ns

    from iai_mcp.migrate import swap_migrated_store

    summary = swap_migrated_store(str(src_db), apply=False, live_pid_probe=lambda: None)

    assert summary["mode"] == "dry-run"
    assert summary["swapped"] is False

    entries_after = sorted(p.name for p in root.iterdir())
    assert entries_after == entries_before
    assert (root / "hippo").stat().st_mtime_ns == hippo_mtime_before
    assert not (root / ".swap-in-progress").exists()


def test_apply_preserves_crypto_key_and_writes_readable_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "live"
    root.mkdir()
    _build_legacy_source_with_file_key(root, monkeypatch=monkeypatch, n=6)
    src_db = root / "hippo" / "brain.sqlite3"
    before = _read_store_mapping(root, monkeypatch=monkeypatch)

    # Sidecar existence is RECORDED, not asserted either way -- folding them
    # into the main file during the copy leg is lossless and permitted; a
    # change to the source's record set is not (that is what `before ==
    # backup_mapping` below actually proves).
    sidecars_before = (
        (root / "hippo" / "brain.sqlite3-wal").exists(),
        (root / "hippo" / "brain.sqlite3-shm").exists(),
    )

    key_path = root / ".crypto.key"
    key_bytes_before = key_path.read_bytes()
    key_mode_before = __import__("os").stat(key_path).st_mode
    key = key_bytes_before  # a 32-byte raw key file IS the key material

    from iai_mcp.migrate import swap_migrated_store

    summary = swap_migrated_store(str(src_db), apply=True, live_pid_probe=lambda: None)
    assert summary["swapped"] is True, summary

    key_files = list(root.rglob(".crypto.key"))
    assert len(key_files) == 1, key_files
    assert key_files[0] == key_path
    assert key_path.read_bytes() == key_bytes_before
    assert __import__("os").stat(key_path).st_mode == key_mode_before

    backup_dir = _backup_dir(root)
    assert backup_dir.is_dir()
    sidecars_after = (
        (backup_dir / "brain.sqlite3-wal").exists(),
        (backup_dir / "brain.sqlite3-shm").exists(),
    )
    print(f"sidecars before copy leg (wal, shm): {sidecars_before}")
    print(f"sidecars after copy leg, at the backup (wal, shm): {sidecars_after}")
    backup_mapping = _read_legacy_hippo_dir_mapping(backup_dir, key=key)
    assert backup_mapping == before


def test_verify_abort_leaves_live_store_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "live"
    root.mkdir()
    _build_legacy_source_with_file_key(root, monkeypatch=monkeypatch, n=5)
    src_db = root / "hippo" / "brain.sqlite3"
    live_hippo_dir = root / "hippo"
    mtime_before = live_hippo_dir.stat().st_mtime_ns

    import iai_mcp.migrate._to_lilli_verify as verify_mod
    from iai_mcp.migrate._to_lilli_verify import DimensionResult, VerifyReport

    def fake_verify(*args, **kwargs):
        return VerifyReport(
            ok=False,
            dimensions={
                "B_bytes": DimensionResult(ok=False, reason="simulated mismatch")
            },
            sampled_n=0,
        )

    monkeypatch.setattr(verify_mod, "verify_store_equality", fake_verify)

    from iai_mcp.migrate import swap_migrated_store

    summary = swap_migrated_store(str(src_db), apply=True, live_pid_probe=lambda: None)

    assert summary["swapped"] is False
    assert summary["verify_ok"] is False
    assert any("simulated mismatch" in b for b in summary["blockers"])
    assert live_hippo_dir.exists()
    assert live_hippo_dir.stat().st_mtime_ns == mtime_before
    assert not (root / ".swap-in-progress").exists()

    staging_dirs = [p for p in root.iterdir() if p.name.startswith(".migrate-staging-")]
    assert len(staging_dirs) == 1, "the staging tree is left for inspection, not removed"


def test_apply_on_already_native_store_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "live"
    root.mkdir()
    monkeypatch.delenv("IAI_MCP_CRYPTO_PASSPHRASE", raising=False)
    from iai_mcp.crypto import CryptoKey

    CryptoKey(store_root=root).rotate()
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    monkeypatch.setenv("IAI_MCP_STORE", str(root))
    from iai_mcp.store import MemoryStore

    store = MemoryStore(root)
    store.db.close()

    src_db = root / "hippo" / "brain.sqlite3"

    from iai_mcp.migrate import swap_migrated_store

    summary = swap_migrated_store(str(src_db), apply=True, live_pid_probe=lambda: None)

    assert summary["swapped"] is False
    assert summary["source_format"] == "lilli"
    assert summary["blockers"] == []
    entries = sorted(p.name for p in root.iterdir())
    assert not any(n.startswith("hippo.sqlite-backup-") for n in entries)
    assert not any(n.startswith(".migrate-staging-") for n in entries)


def test_apply_fsyncs_before_removing_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "live"
    root.mkdir()
    _build_legacy_source_with_file_key(root, monkeypatch=monkeypatch, n=6)
    src_db = root / "hippo" / "brain.sqlite3"

    import iai_mcp.migrate._to_lilli_swap as swap_mod

    order: list[str] = []
    orig_fsync = swap_mod._fsync_dir
    orig_remove = swap_mod._remove_marker

    def spy_fsync(path):
        order.append("fsync")
        return orig_fsync(path)

    def spy_remove(live_root):
        order.append("remove_marker")
        return orig_remove(live_root)

    monkeypatch.setattr(swap_mod, "_fsync_dir", spy_fsync)
    monkeypatch.setattr(swap_mod, "_remove_marker", spy_remove)

    from iai_mcp.migrate import swap_migrated_store

    summary = swap_migrated_store(str(src_db), apply=True, live_pid_probe=lambda: None)
    assert summary["swapped"] is True

    assert order == ["fsync", "remove_marker"]
    assert not (root / ".swap-in-progress").exists()


def test_refusal_helper_present_vs_absent_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import os as _os

    from iai_mcp.migrate._to_lilli_swap import refuse_if_marker_present

    root_absent = tmp_path / "no-marker"
    root_absent.mkdir()
    assert refuse_if_marker_present(root_absent) is None

    root_present = tmp_path / "has-marker"
    root_present.mkdir()
    marker = _plant_marker(root_present, pid=_os.getpid())

    reason = refuse_if_marker_present(root_present)
    assert reason is not None
    assert str(marker) in reason
    assert marker.exists()


def test_refusal_helper_truncated_marker_still_refuses(tmp_path: Path):
    from iai_mcp.migrate._to_lilli_swap import MARKER_FILE_NAME, refuse_if_marker_present

    root = tmp_path / "truncated"
    root.mkdir()
    marker = root / MARKER_FILE_NAME
    marker.write_bytes(b"{not valid json")

    before = marker.read_bytes()
    reason = refuse_if_marker_present(root)
    assert reason is not None
    assert marker.read_bytes() == before


def test_refusal_paths_never_delete_the_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import argparse
    import os as _os

    from iai_mcp.migrate._to_lilli_swap import refuse_if_marker_present

    root = tmp_path / "live"
    root.mkdir()
    marker = _plant_marker(root, pid=_os.getpid())
    before = marker.read_bytes()

    assert refuse_if_marker_present(root) is not None
    assert marker.read_bytes() == before

    monkeypatch.setenv("IAI_MCP_STORE", str(root))
    monkeypatch.setattr("iai_mcp.cli.subprocess.run", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("supervisor call must never run when the marker refuses first")
    ))

    from iai_mcp.cli._daemon import cmd_daemon_start

    rc = cmd_daemon_start(argparse.Namespace())
    assert rc != 0
    assert marker.read_bytes() == before


def test_refusal_helper_dead_pid_marker_says_not_alive(tmp_path: Path):
    from iai_mcp.migrate._to_lilli_swap import refuse_if_marker_present

    root = tmp_path / "live"
    root.mkdir()
    # A pid this large is not a real process on any platform this suite runs
    # on -- a deterministic dead pid without waiting for a real one to exit.
    _plant_marker(root, pid=999999)

    reason = refuse_if_marker_present(root)
    assert reason is not None
    assert "not alive" in reason


def test_symlinked_store_root_swaps_the_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    real_root = tmp_path / "real-store"
    real_root.mkdir()
    _build_legacy_source_with_file_key(real_root, monkeypatch=monkeypatch, n=6)
    before = _read_store_mapping(real_root, monkeypatch=monkeypatch)

    link_root = tmp_path / "store-link"
    link_root.symlink_to(real_root, target_is_directory=True)

    src_db = link_root / "hippo" / "brain.sqlite3"

    from iai_mcp.migrate import swap_migrated_store

    summary = swap_migrated_store(str(src_db), apply=True, live_pid_probe=lambda: None)
    assert summary["swapped"] is True, summary

    assert link_root.is_symlink()
    assert Path(__import__("os").readlink(link_root)) == real_root

    assert (real_root / "hippo").is_dir()
    assert _backup_dir(real_root).is_dir()

    # No stray directory beside the link -- everything landed under the
    # real target, resolved before any check ran.
    tmp_entries = sorted(p.name for p in tmp_path.iterdir())
    assert tmp_entries == sorted(["real-store", "store-link"])

    after = _read_store_mapping(real_root, monkeypatch=monkeypatch)
    assert after == before


def test_live_daemon_blocks_apply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "live"
    root.mkdir()
    _build_legacy_source_with_file_key(root, monkeypatch=monkeypatch, n=5)
    src_db = root / "hippo" / "brain.sqlite3"
    live_hippo_dir = root / "hippo"
    mtime_before = live_hippo_dir.stat().st_mtime_ns

    from iai_mcp.migrate import swap_migrated_store

    summary = swap_migrated_store(str(src_db), apply=True, live_pid_probe=lambda: 4242)

    assert summary["swapped"] is False
    assert summary["daemon_pid"] == 4242
    assert any("iai-mcp daemon stop" in b for b in summary["blockers"])
    assert any("iai-mcp daemon start" in b for b in summary["blockers"])
    assert live_hippo_dir.stat().st_mtime_ns == mtime_before


def test_existing_backup_dir_blocks_apply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "live"
    root.mkdir()
    _build_legacy_source_with_file_key(root, monkeypatch=monkeypatch, n=5)
    src_db = root / "hippo" / "brain.sqlite3"

    backup_dir = _backup_dir(root)
    backup_dir.mkdir()
    sentinel = backup_dir / "sentinel.txt"
    sentinel.write_text("do not clobber")

    from iai_mcp.migrate import swap_migrated_store

    summary = swap_migrated_store(str(src_db), apply=True, live_pid_probe=lambda: None)

    assert summary["swapped"] is False
    assert any("backup" in b for b in summary["blockers"])
    assert sentinel.read_text() == "do not clobber"


def test_cross_filesystem_staging_blocks_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "live"
    root.mkdir()
    _build_legacy_source_with_file_key(root, monkeypatch=monkeypatch, n=5)
    src_db = root / "hippo" / "brain.sqlite3"

    import iai_mcp.migrate._to_lilli_swap as swap_mod

    def fake_device_id(path: Path) -> int:
        return 1 if Path(path) == root else 2

    monkeypatch.setattr(swap_mod, "_device_id", fake_device_id)

    from iai_mcp.migrate import swap_migrated_store

    summary = swap_migrated_store(str(src_db), apply=True, live_pid_probe=lambda: None)

    assert summary["swapped"] is False
    assert any("filesystem" in b for b in summary["blockers"])


def test_wrong_parent_directory_raises(tmp_path: Path):
    bogus = tmp_path / "not_hippo" / "brain.sqlite3"

    from iai_mcp.migrate import swap_migrated_store

    with pytest.raises(ValueError):
        swap_migrated_store(str(bogus), apply=False)


def test_idempotent_second_apply_is_a_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "live"
    root.mkdir()
    _build_legacy_source_with_file_key(root, monkeypatch=monkeypatch, n=5)
    src_db = root / "hippo" / "brain.sqlite3"

    from iai_mcp.migrate import swap_migrated_store

    first = swap_migrated_store(str(src_db), apply=True, live_pid_probe=lambda: None)
    assert first["swapped"] is True

    entries_after_first = sorted(p.name for p in root.iterdir())

    second = swap_migrated_store(str(src_db), apply=True, live_pid_probe=lambda: None)
    assert second["swapped"] is False
    assert second["blockers"] == []

    entries_after_second = sorted(p.name for p in root.iterdir())
    assert entries_after_second == entries_after_first


def test_marker_survives_between_the_two_renames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "live"
    root.mkdir()
    _build_legacy_source_with_file_key(root, monkeypatch=monkeypatch, n=6)
    src_db = root / "hippo" / "brain.sqlite3"
    live_hippo_dir = root / "hippo"

    import iai_mcp.migrate._to_lilli_swap as swap_mod

    observed: dict = {}
    orig_rename = swap_mod._rename

    def spy_rename(src, dst):
        orig_rename(src, dst)
        if dst != live_hippo_dir:
            # The FIRST rename (live hippo/ -> backup) just completed: the
            # store subdirectory does not exist at the live root at all
            # right now -- the state the marker most has to survive.
            observed["hippo_absent"] = not live_hippo_dir.exists()
            observed["marker_present"] = (root / ".swap-in-progress").exists()
            observed["refusal"] = swap_mod.refuse_if_marker_present(root)

    monkeypatch.setattr(swap_mod, "_rename", spy_rename)

    from iai_mcp.migrate import swap_migrated_store

    summary = swap_migrated_store(str(src_db), apply=True, live_pid_probe=lambda: None)
    assert summary["swapped"] is True

    assert observed["hippo_absent"] is True
    assert observed["marker_present"] is True
    assert observed["refusal"] is not None


def test_swap_prewarm_persists_cache_at_live_root_in_file_key_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd
):
    """A swap in file-key mode (no passphrase, a real key FILE) actually
    persists the pre-warmed runtime graph cache -- at the LIVE root, where
    the swapped-in store looks for it, not stranded at the discarded staging
    root -- and never emits the runtime_graph_cache_encrypt_failed
    diagnostic. capfd (not capsys): the graph build spawns centrality/
    community child processes, whose stderr capsys does not see."""
    root = tmp_path / "live"
    root.mkdir()
    _build_legacy_source_with_file_key(root, monkeypatch=monkeypatch, n=8)
    src_db = root / "hippo" / "brain.sqlite3"

    from iai_mcp.migrate import swap_migrated_store

    summary = swap_migrated_store(str(src_db), apply=True, live_pid_probe=lambda: None)
    assert summary["swapped"] is True, summary

    live_cache = root / "runtime_graph_cache.json"
    assert live_cache.exists(), "the pre-warmed cache must land at the live root"

    staging_dirs = [
        p for p in root.iterdir() if p.name.startswith(".migrate-staging-")
    ]
    assert staging_dirs == [], (
        "no stray staging directory should survive a successful swap"
    )

    err = capfd.readouterr().err
    assert "runtime_graph_cache_encrypt_failed" not in err


def test_preswap_cache_at_live_root_survives_when_prewarm_produces_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A runtime graph cache already present at the live root before a swap
    is left alone when this run's pre-warm produces no fresh file (here, a
    forced build failure). The swap is content-preserving by construction
    (verify_store_equality proves the staged tree byte-identical to the
    source), so a cache that was valid before stays exactly as valid after,
    and the swap must not force an unrelated cold rebuild by deleting it."""
    root = tmp_path / "live"
    root.mkdir()
    _build_legacy_source_with_file_key(root, monkeypatch=monkeypatch, n=5)
    src_db = root / "hippo" / "brain.sqlite3"

    pre_existing_cache = root / "runtime_graph_cache.json"
    pre_existing_cache.write_text('{"sentinel": true}', encoding="utf-8")

    import iai_mcp.retrieve as _retrieve_mod

    def _raising_build(*args, **kwargs):
        raise RuntimeError("simulated pre-warm build failure")

    monkeypatch.setattr(_retrieve_mod, "build_runtime_graph", _raising_build)

    from iai_mcp.migrate import swap_migrated_store

    summary = swap_migrated_store(str(src_db), apply=True, live_pid_probe=lambda: None)
    assert summary["swapped"] is True, summary

    assert pre_existing_cache.exists()
    assert pre_existing_cache.read_text(encoding="utf-8") == '{"sentinel": true}'


def test_concurrent_apply_invocations_serialize_via_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A second apply invocation against the same live root, while another
    apply already holds the swap lock, refuses immediately -- without
    touching the live hippo/ directory or writing a backup -- instead of
    racing into the copy/verify/rename sequence."""
    import os as _os

    from iai_mcp import _flock

    root = tmp_path / "live"
    root.mkdir()
    _build_legacy_source_with_file_key(root, monkeypatch=monkeypatch, n=5)
    src_db = root / "hippo" / "brain.sqlite3"
    live_hippo_dir = root / "hippo"
    mtime_before = live_hippo_dir.stat().st_mtime_ns

    lock_path = root / ".swap.lock"
    held_fd = _os.open(str(lock_path), _os.O_CREAT | _os.O_RDWR, 0o600)
    _flock.flock(held_fd, _flock.LOCK_EX | _flock.LOCK_NB)
    try:
        from iai_mcp.migrate import swap_migrated_store

        summary = swap_migrated_store(
            str(src_db), apply=True, live_pid_probe=lambda: None
        )

        assert summary["swapped"] is False
        assert len(summary["blockers"]) == 1
        assert "already running" in summary["blockers"][0]
        assert live_hippo_dir.stat().st_mtime_ns == mtime_before
        assert not _backup_dir(root).exists()
        assert not (root / ".swap-in-progress").exists()
    finally:
        _flock.flock(held_fd, _flock.LOCK_UN)
        _os.close(held_fd)


def test_lock_open_failure_surfaces_as_a_blocker_not_an_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A pre-mutation I/O failure opening the swap lock file (permissions,
    ENOSPC, a read-only mount) is reported through the blockers list, not
    raised -- nothing was attempted yet, so there is no three-way state to
    point an operator at."""
    root = tmp_path / "live"
    root.mkdir()
    _build_legacy_source_with_file_key(root, monkeypatch=monkeypatch, n=5)
    src_db = root / "hippo" / "brain.sqlite3"
    live_hippo_dir = root / "hippo"
    mtime_before = live_hippo_dir.stat().st_mtime_ns

    import iai_mcp.migrate._to_lilli_swap as swap_mod

    lock_path = root / ".swap.lock"
    real_open = swap_mod.os.open

    def flaky_open(path, *args, **kwargs):
        if path == str(lock_path):
            raise OSError("simulated: permission denied opening the lock file")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(swap_mod.os, "open", flaky_open)

    from iai_mcp.migrate import swap_migrated_store

    summary = swap_migrated_store(str(src_db), apply=True, live_pid_probe=lambda: None)

    assert summary["swapped"] is False
    assert len(summary["blockers"]) == 1
    assert "swap lock file" in summary["blockers"][0]
    assert live_hippo_dir.stat().st_mtime_ns == mtime_before
    assert not _backup_dir(root).exists()
    staging_dirs = [
        p for p in root.iterdir() if p.name.startswith(".migrate-staging-")
    ]
    assert staging_dirs == []


def test_apply_returns_summary_even_if_the_final_lock_close_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A close() failure on the swap's own lock fd, after a fully
    successful swap, must not discard the completed summary -- os.close is
    guarded exactly like the flock(LOCK_UN) unlock beside it."""
    root = tmp_path / "live"
    root.mkdir()
    _build_legacy_source_with_file_key(root, monkeypatch=monkeypatch, n=5)
    src_db = root / "hippo" / "brain.sqlite3"

    import iai_mcp.migrate._to_lilli_swap as swap_mod

    lock_path = root / ".swap.lock"
    real_open = swap_mod.os.open
    real_close = swap_mod.os.close
    lock_fd_holder: list[int] = []

    def spy_open(path, *args, **kwargs):
        fd = real_open(path, *args, **kwargs)
        if path == str(lock_path):
            lock_fd_holder.append(fd)
        return fd

    def flaky_close(fd):
        if lock_fd_holder and fd == lock_fd_holder[-1]:
            raise OSError("simulated close failure on the lock fd")
        return real_close(fd)

    monkeypatch.setattr(swap_mod.os, "open", spy_open)
    monkeypatch.setattr(swap_mod.os, "close", flaky_close)

    from iai_mcp.migrate import swap_migrated_store

    summary = swap_migrated_store(str(src_db), apply=True, live_pid_probe=lambda: None)

    assert summary["swapped"] is True, summary
    assert lock_fd_holder, "the lock fd must actually have been tracked"


def test_second_rename_failure_leaves_three_way_state_and_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "live"
    root.mkdir()
    _build_legacy_source_with_file_key(root, monkeypatch=monkeypatch, n=6)
    src_db = root / "hippo" / "brain.sqlite3"
    before = _read_store_mapping(root, monkeypatch=monkeypatch)

    from iai_mcp.crypto import CryptoKey

    key = CryptoKey(store_root=root).get_or_create()

    import iai_mcp.migrate._to_lilli_swap as swap_mod

    live_hippo_dir = root / "hippo"
    orig_rename = swap_mod._rename

    def flaky_rename(src, dst):
        # Key on the DESTINATION argument, not on call count: the copy leg
        # and the verifier run first and may reach the same primitive via
        # their own atomic-write helpers, so a counter would fire on the
        # wrong call. Only the second swap rename targets the live path.
        if dst == live_hippo_dir:
            raise OSError("simulated failure on the second swap rename")
        return orig_rename(src, dst)

    monkeypatch.setattr(swap_mod, "_rename", flaky_rename)

    from iai_mcp.migrate import swap_migrated_store

    with pytest.raises(OSError):
        swap_migrated_store(str(src_db), apply=True, live_pid_probe=lambda: None)

    # Nothing at the live subdirectory path.
    assert not live_hippo_dir.exists()

    # The dated backup independently holds a complete, non-empty store.
    backup_dir = _backup_dir(root)
    assert backup_dir.is_dir()
    backup_mapping = _read_legacy_hippo_dir_mapping(backup_dir, key=key)
    assert backup_mapping, "backup mapping must not be empty"
    assert backup_mapping == before

    # The staging tree independently holds a complete, non-empty store.
    staging_dirs = [p for p in root.iterdir() if p.name.startswith(".migrate-staging-")]
    assert len(staging_dirs) == 1
    staging_mapping = _read_native_root_mapping_with_planted_key(
        staging_dirs[0], key=key, monkeypatch=monkeypatch
    )
    assert staging_mapping, "staging mapping must not be empty"
    assert staging_mapping == before

    # The marker is still present -- the removal never ran.
    marker = root / ".swap-in-progress"
    assert marker.exists()
    from iai_mcp.migrate._to_lilli_swap import refuse_if_marker_present

    assert refuse_if_marker_present(root) is not None

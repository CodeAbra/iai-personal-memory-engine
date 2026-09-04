"""The directive sweep retires phantom directives (records with no
explicit-declaration provenance stamp) while preserving explicitly-declared
ones, never touching literal_surface, and is idempotent. Every test builds
a tmp store and asserts it is NOT the real home before mutating anything.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp.cli import main as _cli_main
from iai_mcp.errors import DatabaseError
from iai_mcp.hippo import HippoTable
from iai_mcp.migrate import sweep_phantom_directives
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401, PLC0415
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _assert_not_real_home(store_path: Path) -> None:
    """Live-store guard: a sweep test must never resolve to the real home."""
    real_home = (Path.home() / ".iai-mcp").resolve()
    resolved = store_path.resolve()
    assert resolved != real_home, (
        f"REFUSING: test store path resolves to the real home ({real_home}); "
        "every sweep test must build a tmp store"
    )
    assert real_home not in resolved.parents or resolved != real_home


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    root.mkdir()
    _assert_not_real_home(root)
    return root


@pytest.fixture
def store(store_path: Path) -> MemoryStore:
    return MemoryStore(path=store_path)


def _directive_record(text: str, *, stamped: bool) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    provenance = [{"ts": now.isoformat(), "cue": "c", "session_id": "s1", "role": "user"}]
    if stamped:
        provenance[0]["directive_source"] = "explicit-command"
    rec = MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=provenance,
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )
    rec.directive = True
    return rec


def _seed_mixed_directives(store: MemoryStore) -> tuple[list, list]:
    """Insert 2 stamped (explicit) + 3 unstamped (phantom) live directives."""
    stamped = [
        _directive_record(f"explicit rule {i}: reply in English", stamped=True)
        for i in range(2)
    ]
    unstamped = [
        _directive_record(f"from now on rule {i}: use metric units", stamped=False)
        for i in range(3)
    ]
    for rec in stamped + unstamped:
        store.insert(rec)
    flush_record_buffer(store)
    return stamped, unstamped


def _live_directive_ids(store: MemoryStore) -> set:
    flush_record_buffer(store)
    return {
        rec.id
        for rec in store.iter_records(where="directive = 1 AND tombstoned_at IS NULL")
    }


# --- 1. DRY-RUN: reports counts, writes nothing -----------------------------


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_dry_run_reports_counts_and_writes_nothing(driver, store, store_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    stamped, unstamped = _seed_mixed_directives(store)

    cache_path = store_path / ".directives.cached.md"
    assert not cache_path.exists()

    result = sweep_phantom_directives(store, apply=False, store_path=store_path)

    assert result["mode"] == "dry-run"
    assert result["directives_found"] == 5
    assert result["unstamped"] == 3
    assert result["retired"] == 0
    assert result["snapshot_dir"] is None
    assert result["cache_refreshed"] is False

    # Nothing mutated: all 5 records are still live directives.
    assert _live_directive_ids(store) == {r.id for r in stamped + unstamped}
    assert not cache_path.exists()
    assert not any(store_path.glob("hippo-pre-directive-sweep-*"))


# --- 2. APPLY: snapshots, flips only unstamped, preserves stamped ----------


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_apply_flips_unstamped_preserves_stamped(driver, store, store_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    stamped, unstamped = _seed_mixed_directives(store)

    result = sweep_phantom_directives(store, apply=True, store_path=store_path)

    assert result["mode"] == "apply"
    assert result["directives_found"] == 5
    assert result["unstamped"] == 3
    assert result["retired"] == 3
    assert result["snapshot_dir"] is not None
    assert Path(result["snapshot_dir"]).is_dir()
    assert result["cache_refreshed"] is True

    remaining = _live_directive_ids(store)
    assert remaining == {r.id for r in stamped}
    for rec in unstamped:
        after = store.get(rec.id)
        assert after is not None
        assert after.directive is False


# --- 3. Cache is written UNDER the target store root, not the home default -


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_apply_writes_cache_under_target_store_root(driver, store, store_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    _seed_mixed_directives(store)

    real_home_cache = Path.home() / ".iai-mcp" / ".directives.cached.md"
    real_home_cache_existed_before = real_home_cache.exists()
    real_home_cache_mtime_before = (
        real_home_cache.stat().st_mtime if real_home_cache_existed_before else None
    )

    result = sweep_phantom_directives(store, apply=True, store_path=store_path)

    scoped_cache = store_path / ".directives.cached.md"
    assert scoped_cache.is_file(), result

    if real_home_cache_existed_before:
        assert real_home_cache.stat().st_mtime == real_home_cache_mtime_before
    else:
        assert not real_home_cache.exists()


# --- 4. literal_surface is byte-identical before and after -----------------


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_literal_surface_untouched(driver, store, store_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    stamped, unstamped = _seed_mixed_directives(store)
    before = {rec.id: rec.literal_surface for rec in stamped + unstamped}

    sweep_phantom_directives(store, apply=True, store_path=store_path)

    for rid, surface_before in before.items():
        after = store.get(rid)
        assert after is not None
        assert after.literal_surface == surface_before


# --- 5. Idempotent: a second --apply retires 0 ------------------------------


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_second_apply_retires_zero(driver, store, store_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    _seed_mixed_directives(store)

    first = sweep_phantom_directives(store, apply=True, store_path=store_path)
    assert first["retired"] == 3

    second = sweep_phantom_directives(store, apply=True, store_path=store_path)
    assert second["retired"] == 0
    assert second["unstamped"] == 0
    assert second["directives_found"] == 2


# --- 5b. one unreadable record must not abort the whole batch --------------


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_one_unreadable_record_is_skipped_and_reported_batch_still_completes(
    driver, store, store_path, monkeypatch
):
    """A single undecryptable/malformed directive record must be skipped and
    counted in `failed`/`errors`, not crash the sweep -- the sweep must still
    complete and retire the OTHER phantom directives."""
    _select_driver(driver, monkeypatch)
    stamped, unstamped = _seed_mixed_directives(store)
    bad_id = unstamped[0].id
    good_unstamped = unstamped[1:]

    orig_get = store.get

    def _flaky_get(rid, *args, **kwargs):
        if rid == bad_id:
            raise RuntimeError("simulated undecryptable directive record")
        return orig_get(rid, *args, **kwargs)

    monkeypatch.setattr(store, "get", _flaky_get)

    result = sweep_phantom_directives(store, apply=True, store_path=store_path)

    assert result["mode"] == "apply"
    # 5 live directives total, 1 unreadable -> 4 successfully classified.
    assert result["directives_found"] == 4, result
    # Of the 4 classified, the 2 remaining phantom (unstamped) ones are targets.
    assert result["unstamped"] == 2, result
    assert result["retired"] == 2, result
    assert result["failed"] == 1, result
    assert any(str(bad_id) in e for e in result["errors"]), result["errors"]
    assert result["snapshot_dir"] is not None
    assert result["cache_refreshed"] is True

    # The good phantom directives were actually retired.
    monkeypatch.setattr(store, "get", orig_get)
    for rec in good_unstamped:
        after = store.get(rec.id)
        assert after is not None
        assert after.directive is False

    # The stamped (explicit) directives are untouched.
    for rec in stamped:
        after = store.get(rec.id)
        assert after is not None
        assert after.directive is True

    # The unreadable record's directive flag is left alone (never classified
    # as a retire target, so never touched by the apply step).
    after_bad = store.get(bad_id)
    assert after_bad is not None
    assert after_bad.directive is True


# --- 5c. genuine at-rest corruption (not a mock) proves the same -----------


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_genuinely_corrupted_provenance_ciphertext_is_skipped_not_retired(
    driver, store, store_path, monkeypatch
):
    """Empirical proof (no mocking of store.get): tampering with a directive
    record's provenance_json ciphertext at rest -- breaking its AES-GCM auth
    tag -- makes store.get() raise for that record. The sweep must skip and
    report it, never silently retire it (a corrupt record falling through to
    `_is_explicitly_stamped(None) -> False -> retired` would silently
    phantom-clear an unreadable directive)."""
    _select_driver(driver, monkeypatch)
    stamped, unstamped = _seed_mixed_directives(store)
    bad_id = unstamped[0].id
    good_unstamped = unstamped[1:]

    from iai_mcp.crypto import CIPHERTEXT_PREFIX

    with store.db._conn_lock:
        row = store.db._conn.execute(
            "SELECT provenance_json FROM records WHERE id = ?", (str(bad_id),)
        ).fetchone()
        original = row[0]
        assert original.startswith(CIPHERTEXT_PREFIX), (
            "test setup requires provenance_json to be encrypted at rest; "
            f"got {original!r}"
        )
        # Flip one character deep in the ciphertext body (not the version
        # prefix) so is_encrypted() still recognizes it and attempts decrypt
        # -- the AES-GCM auth tag check must then fail loudly.
        tail = original[-1]
        flipped = "0" if tail != "0" else "1"
        corrupted = original[:-1] + flipped
        assert corrupted != original
        store.db._conn.execute(
            "UPDATE records SET provenance_json = ? WHERE id = ?",
            (corrupted, str(bad_id)),
        )
        store.db._conn.commit()

    result = sweep_phantom_directives(store, apply=True, store_path=store_path)

    assert result["mode"] == "apply"
    assert result["directives_found"] == 4, result
    assert result["unstamped"] == 2, result
    assert result["retired"] == 2, result
    assert result["failed"] == 1, result
    assert any(str(bad_id) in e for e in result["errors"]), result["errors"]

    # The good phantom directives were actually retired.
    for rec in good_unstamped:
        after = store.get(rec.id)
        assert after is not None
        assert after.directive is False

    # The stamped (explicit) directives are untouched.
    for rec in stamped:
        after = store.get(rec.id)
        assert after is not None
        assert after.directive is True

    # The corrupted record's directive flag is untouched -- store.get()
    # still raises on it (not silently phantom-cleared), and the read below
    # only checks the raw column, not a decrypted MemoryRecord.
    with pytest.raises(Exception):  # noqa: B017 -- HippoDecryptError, driver-specific
        store.get(bad_id)
    with store.db._conn_lock:
        raw_row = store.db._conn.execute(
            "SELECT directive FROM records WHERE id = ?", (str(bad_id),)
        ).fetchone()
    assert bool(raw_row[0]), "corrupted record's directive flag must be untouched"


# --- 5d. one failing update() must not abort the apply batch ---------------


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_one_failing_update_is_skipped_and_reported_batch_still_completes(
    driver, store, store_path, monkeypatch
):
    """A record that reads fine but whose apply-time update() raises the
    project's own storage-corruption signal (DatabaseError, not a subclass of
    OSError/ValueError/RuntimeError) must be skipped and counted in
    `failed`/`errors`, not crash the sweep -- the sweep must still complete,
    retire the OTHER phantom directives, produce a summary, and the CLI must
    exit non-zero."""
    _select_driver(driver, monkeypatch)
    stamped, unstamped = _seed_mixed_directives(store)
    bad_id = unstamped[0].id
    good_unstamped = unstamped[1:]

    orig_update = HippoTable.update

    def _flaky_update(self, where, values, *args, **kwargs):
        if str(bad_id) in where:
            raise DatabaseError("simulated corruption on directive-sweep update")
        return orig_update(self, where, values, *args, **kwargs)

    monkeypatch.setattr(HippoTable, "update", _flaky_update)

    result = sweep_phantom_directives(store, apply=True, store_path=store_path)

    assert result["mode"] == "apply"
    # All 5 records still read fine -- only the apply-time update fails.
    assert result["directives_found"] == 5, result
    assert result["unstamped"] == 3, result
    assert result["retired"] == 2, result
    assert result["failed"] == 1, result
    assert any(str(bad_id) in e for e in result["errors"]), result["errors"]
    assert result["snapshot_dir"] is not None
    assert result["cache_refreshed"] is True

    # The good phantom directives were actually retired.
    for rec in good_unstamped:
        after = store.get(rec.id)
        assert after is not None
        assert after.directive is False

    # The stamped (explicit) directives are untouched.
    for rec in stamped:
        after = store.get(rec.id)
        assert after is not None
        assert after.directive is True

    # The record whose update failed is left alone (never actually cleared).
    after_bad = store.get(bad_id)
    assert after_bad is not None
    assert after_bad.directive is True


# --- 5e. same failure, driven through real CLI dispatch, exits non-zero ----


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_cli_exits_nonzero_when_update_fails(driver, store, store_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    _, unstamped = _seed_mixed_directives(store)
    bad_id = unstamped[0].id

    orig_update = HippoTable.update

    def _flaky_update(self, where, values, *args, **kwargs):
        if str(bad_id) in where:
            raise DatabaseError("simulated corruption on directive-sweep update")
        return orig_update(self, where, values, *args, **kwargs)

    monkeypatch.setattr(HippoTable, "update", _flaky_update)

    rc = _cli_main(["directive-sweep", "--apply", "--store-path", str(store_path)])
    assert rc != 0

    store2 = MemoryStore(path=store_path)
    after_bad = store2.get(bad_id)
    assert after_bad is not None
    assert after_bad.directive is True


# --- 6. CLI-LEVEL: real argparse dispatch, dry-run then --apply ------------


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_cli_dispatch_dry_run_then_apply(driver, store, store_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    _seed_mixed_directives(store)

    rc_dry = _cli_main(["directive-sweep", "--dry-run", "--store-path", str(store_path)])
    assert rc_dry == 0

    rc_apply = _cli_main(["directive-sweep", "--apply", "--store-path", str(store_path)])
    assert rc_apply == 0

    store2 = MemoryStore(path=store_path)
    remaining = _live_directive_ids(store2)
    assert len(remaining) == 2

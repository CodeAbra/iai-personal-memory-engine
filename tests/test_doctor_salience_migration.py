from __future__ import annotations

import hashlib
import sqlite3

import pytest

from iai_mcp.migrate._salience_auto_migration import (
    MAX_ATTEMPTS,
    _ATTEMPTS_KEY,
    _DONE_MARKER_KEY,
    _LAST_ERROR_KEY,
    _NATIVE_KEY,
)

# Fresh SQLite file seeded directly (not through HippoDB), same rationale as
# test_doctor_hippo_rows.py: the on-disk header is always genuine SQLite, so
# the check's own open_store_conn -> _resolve_effective_driver resolution is
# exercised without a driver override.


def _build_store(db_path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE _hippo_meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(
        "CREATE TABLE records (id TEXT PRIMARY KEY, tombstoned_at TEXT,"
        " salience_level TEXT)"
    )
    conn.commit()
    conn.close()


def _insert_legacy_unflagged(db_path, n: int) -> None:
    conn = sqlite3.connect(str(db_path))
    for i in range(n):
        conn.execute(
            "INSERT INTO records (id, tombstoned_at, salience_level)"
            " VALUES (?, NULL, 'unflagged')",
            (f"alice-rec-{i}",),
        )
    conn.commit()
    conn.close()


def _set_meta(db_path, key: str, value: str) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute("INSERT OR REPLACE INTO _hippo_meta (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()


def test_ll_db_absent_pass(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))

    from iai_mcp.doctor import check_ll_salience_auto_migration

    result = check_ll_salience_auto_migration()
    assert result.status == "PASS"
    assert result.passed is True
    assert "absent" in result.detail


def test_ll_born_native_reports_not_needed(tmp_path, monkeypatch):
    """A store stamped native at birth never needs the migration -- this
    must be reported distinctly from 'pending', even though both start
    with the done marker unset."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    db_path = tmp_path / "hippo" / "brain.sqlite3"
    _build_store(db_path)
    _set_meta(db_path, _NATIVE_KEY, "1")

    from iai_mcp.doctor import check_ll_salience_auto_migration

    result = check_ll_salience_auto_migration()
    assert result.status == "PASS"
    assert result.passed is True
    assert "not needed" in result.detail


def test_ll_pending_no_marker_reports_legacy_count(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    db_path = tmp_path / "hippo" / "brain.sqlite3"
    _build_store(db_path)
    _insert_legacy_unflagged(db_path, 3)

    from iai_mcp.doctor import check_ll_salience_auto_migration

    result = check_ll_salience_auto_migration()
    assert result.status == "WARN"
    assert result.passed is True
    assert "pending" in result.detail
    assert "3" in result.detail


def test_ll_pending_zero_legacy_is_pass(tmp_path, monkeypatch):
    """No marker yet, but nothing left to promote — pending is benign."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    db_path = tmp_path / "hippo" / "brain.sqlite3"
    _build_store(db_path)

    from iai_mcp.doctor import check_ll_salience_auto_migration

    result = check_ll_salience_auto_migration()
    assert result.status == "PASS"
    assert result.passed is True
    assert "pending" in result.detail
    assert "0 legacy unflagged" in result.detail


def test_ll_done_marker_reports_remaining_count(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    db_path = tmp_path / "hippo" / "brain.sqlite3"
    _build_store(db_path)
    _insert_legacy_unflagged(db_path, 1)
    _set_meta(db_path, _DONE_MARKER_KEY, "1")

    from iai_mcp.doctor import check_ll_salience_auto_migration

    result = check_ll_salience_auto_migration()
    assert result.status == "PASS"
    assert result.passed is True
    assert "done" in result.detail
    assert "1 legacy unflagged remaining" in result.detail


def test_ll_gave_up_state_points_at_manual_escape_hatch(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    db_path = tmp_path / "hippo" / "brain.sqlite3"
    _build_store(db_path)
    _insert_legacy_unflagged(db_path, 2)
    _set_meta(db_path, _ATTEMPTS_KEY, str(MAX_ATTEMPTS))
    _set_meta(db_path, _LAST_ERROR_KEY, "RuntimeError: simulated failure")

    from iai_mcp.doctor import check_ll_salience_auto_migration

    result = check_ll_salience_auto_migration()
    assert result.status == "WARN"
    assert result.passed is True
    assert "failed after" in result.detail
    assert str(MAX_ATTEMPTS) in result.detail
    assert "simulated failure" in result.detail
    assert "salience-backfill --apply" in result.detail


def test_ll_legacy_count_bounded_reports_cap(tmp_path, monkeypatch):
    """A legacy backlog past the cap reports '≥cap', never the exact
    count -- proves the count is LIMIT-bounded, not an unbounded COUNT(*)."""
    import iai_mcp.doctor._storage_checks as _checks

    monkeypatch.setattr(_checks, "_LEGACY_UNFLAGGED_COUNT_CAP", 5)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    db_path = tmp_path / "hippo" / "brain.sqlite3"
    _build_store(db_path)
    _insert_legacy_unflagged(db_path, 8)

    from iai_mcp.doctor import check_ll_salience_auto_migration

    result = check_ll_salience_auto_migration()
    assert result.status == "WARN"
    assert "≥5" in result.detail
    assert "8" not in result.detail


def test_ll_legacy_count_under_cap_reports_exact(tmp_path, monkeypatch):
    import iai_mcp.doctor._storage_checks as _checks

    monkeypatch.setattr(_checks, "_LEGACY_UNFLAGGED_COUNT_CAP", 5)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    db_path = tmp_path / "hippo" / "brain.sqlite3"
    _build_store(db_path)
    _insert_legacy_unflagged(db_path, 3)

    from iai_mcp.doctor import check_ll_salience_auto_migration

    result = check_ll_salience_auto_migration()
    assert result.status == "WARN"
    assert "3 legacy unflagged" in result.detail
    assert "≥" not in result.detail


def test_ll_read_only_never_mutates_store(tmp_path, monkeypatch):
    """The check must not write — the file's own content stays unchanged."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    db_path = tmp_path / "hippo" / "brain.sqlite3"
    _build_store(db_path)
    _insert_legacy_unflagged(db_path, 5)
    before = db_path.read_bytes()

    from iai_mcp.doctor import check_ll_salience_auto_migration

    check_ll_salience_auto_migration()
    after = db_path.read_bytes()
    assert before == after


def test_ll_registered_in_all():
    from iai_mcp import doctor

    assert "check_ll_salience_auto_migration" in doctor.__all__
    assert callable(doctor.check_ll_salience_auto_migration)


def test_ll_full_diagnosis_run_succeeds(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.doctor import run_diagnosis

    results = run_diagnosis(fetch_update=False)
    names = [r.name for r in results]
    assert "(ll) salience auto-migration status" in names


# ---------------------------------------------------------------------------
# Native/lilli-driver coverage: open_store_conn's driver resolution is a
# driver-visible seam -- everything above exercises only the stdlib on-disk
# format. These build a real store through MemoryStore/HippoDB so the
# native engine read path is exercised too.
# ---------------------------------------------------------------------------


def _hash_store_tree(root) -> dict:
    """sha256 per file under ``root``, keyed by relative path -- proves
    byte-identity across a whole file set (db + any on-disk index
    artifacts), not just one file."""
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _require_lilli_driver(monkeypatch) -> None:
    try:
        import iai_mcp_native  # noqa: F401
    except ImportError:
        pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")


def test_ll_native_driver_born_native_reports_not_needed(tmp_path, monkeypatch):
    """A store born under a salience-aware binary is stamped native at
    construction (zero live records) -- proven against the real production
    stamping code path, not a synthetic marker row."""
    _require_lilli_driver(monkeypatch)

    from iai_mcp.store import MemoryStore

    store_root = tmp_path / "store"
    store = MemoryStore(path=store_root)
    store.close()

    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    from iai_mcp.doctor import check_ll_salience_auto_migration

    result = check_ll_salience_auto_migration()
    assert result.status == "PASS"
    assert result.passed is True
    assert "not needed" in result.detail


def test_ll_native_driver_pending_reports_legacy_count(tmp_path, monkeypatch):
    """A real lilli-format store carrying legacy unflagged content (native
    stamp stripped, as a genuine pre-producer store would be) reports
    'pending' with the matching count via the native engine read path."""
    _require_lilli_driver(monkeypatch)

    from datetime import datetime, timezone
    from uuid import uuid4

    from iai_mcp.store import MemoryStore, flush_record_buffer
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    def _emb(i: int) -> list[float]:
        vec = [0.0] * EMBED_DIM
        vec[i % EMBED_DIM] = 1.0
        return vec

    store_root = tmp_path / "store"
    store = MemoryStore(path=store_root)
    now = datetime.now(timezone.utc)
    texts = ["alice prefers tea over coffee", "alice asked about the roadmap"]
    for i, text in enumerate(texts):
        store.insert(MemoryRecord(
            id=uuid4(),
            tier="episodic",
            literal_surface=text,
            aaak_index="",
            embedding=_emb(i),
            community_id=None,
            centrality=0.0,
            detail_level=2,
            pinned=False,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=False,
            never_merge=False,
            provenance=[],
            created_at=now,
            updated_at=now,
            tags=[],
            language="en",
            directive=False,
            epistemic_status="unknown",
            salience_level="unflagged",
        ))
    flush_record_buffer(store)

    # Strip the birth-time native stamp -- a real legacy store never carries
    # it (it predates the producer that writes it).
    with store.db._conn_lock:
        store.db._conn.execute("DELETE FROM _hippo_meta WHERE key = ?", (_NATIVE_KEY,))
        store.db._conn.commit()
    store.close()

    before = _hash_store_tree(store_root)

    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    from iai_mcp.doctor import check_ll_salience_auto_migration

    result = check_ll_salience_auto_migration()
    assert result.status == "WARN"
    assert result.passed is True
    assert "pending" in result.detail
    assert f"{len(texts)} legacy unflagged" in result.detail

    after = _hash_store_tree(store_root)
    assert before == after, (
        "the read-only check must not mutate the native-format store's on-disk files"
    )

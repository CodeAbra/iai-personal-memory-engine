"""Regression: RoConnPool.borrow() self-heals against the read-only-snapshot
fence.

The engine's read-only-snapshot fence (crates/lillibrain/src/pager.rs:322-338)
correctly RAISES when a WAL auto-checkpoint invalidates an already-open RO
snapshot -- that safety mechanism itself is sound. The bug was that
``RoConnPool`` only refreshed a slot on a STALE GENERATION (bumped only by
row-count-changing writes such as insert/tombstone); ``reinforce_record`` /
``boost_edges`` (pure UPDATEs) never call ``mark_ro_pool_stale()``, so a slot
opened before a checkpoint fired stayed in the queue with its old generation
and was handed out again, raising an uncaught ``sqlite3.OperationalError`` on
the caller's very next ``execute()``.

The fix: a proactive ``SELECT 1`` probe inside ``borrow()`` forces the fence
check BEFORE handout. Only the narrowed fence message
("read-only snapshot invalidated") is retried (close + reopen); every other
``sqlite3.OperationalError`` propagates unchanged. The retry is bounded --
exhaustion raises so ``ro_conn()`` falls back to the writer path.

RoConnPool is lilli-only (never constructed on stdlib) -- every test here is
skipped under the stdlib driver.

Every store here lives under ``tmp_path`` — never a real or prod store.
"""

from __future__ import annotations

import os
import sqlite3
import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from iai_mcp.hippo._ro_pool import _RO_SNAPSHOT_FENCE_MARKER
from iai_mcp.store import MemoryStore

_LILLI = os.environ.get("LILLI_STORAGE_DRIVER", "").lower() == "lilli"

_EMBED_DIM = 384

_lilli_only = pytest.mark.skipif(
    not _LILLI,
    reason="RoConnPool is lilli-only; skipped under stdlib entirely "
    "(no stdlib-parity case applies -- ro_conn() never constructs a pool there)",
)


def _norm_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(_EMBED_DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def _embedding_blob(seed: int) -> bytes:
    return struct.pack(f"<{_EMBED_DIM}f", *_norm_vec(seed))


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path)


def _insert_record_direct(store: MemoryStore, seed: int) -> str:
    db = store.db
    rid = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()
    with db._conn_lock:
        db._conn.execute(
            "INSERT INTO records "
            "(id, tier, literal_surface, embedding, tombstoned_at, "
            " embedding_pending, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rid,
                "episodic",
                None,
                _embedding_blob(seed),
                None,
                0,
                now_iso,
                now_iso,
            ),
        )
        db._conn.commit()
    return rid


def _force_wal_checkpoint(store: MemoryStore) -> None:
    """Issue the real PRAGMA the maintenance path uses (maintenance.py:313),
    against the shared writer connection -- merges committed WAL frames into
    the main file and (per pager.rs:322-338) invalidates any already-open RO
    snapshot whose main-file length or checkpoint generation it moves past."""
    with store.db._conn_lock:
        store.db._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def _reinforce_style_update(store: MemoryStore, rid: str) -> None:
    """An in-place UPDATE of an EXISTING row -- mirrors what reinforce_record /
    boost_edges actually do (rewrite an already-resident page), as opposed to
    an INSERT (which only appends new pages beyond the header's cached
    db_size). The read-only snapshot fence (pager.rs:322-338) only fires on a
    genuine main-file-fallback read of a page whose content moved between
    generations; a page number beyond the RO snapshot's cached db_size is
    never walked at all (the header page pins db_size at open), so only an
    in-place rewrite of an EXISTING, already-checkpointed row can trigger it.
    """
    with store.db._conn_lock:
        store.db._conn.execute(
            "UPDATE records SET literal_surface = ? WHERE id = ?",
            (f"reinforced-{uuid.uuid4()}", rid),
        )
        store.db._conn.commit()


# ---------------------------------------------------------------------------
# Core repro: an already-borrowed-then-released slot, invalidated by a
# checkpoint the pool's generation counter never saw, self-heals on the next
# borrow() instead of raising into the caller.
# ---------------------------------------------------------------------------


@_lilli_only
def test_borrow_self_heals_after_checkpoint_invalidation(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    pool = store.db._ro_pool

    # Seed rows and checkpoint them into the main file FIRST -- the RO
    # snapshot's header page (and therefore its cached db_size) is captured
    # at open, so only an in-place rewrite of an already-checkpointed,
    # already-resident row (never a bare append) can trigger the fence.
    rid = _insert_record_direct(store, 0)
    _force_wal_checkpoint(store)

    # Open + release a slot (idle in the queue): a slot opened BEFORE the
    # invalidating checkpoint, still snapshotted at its open-time state.
    with pool.borrow() as conn:
        conn.execute("SELECT COUNT(*) FROM records").fetchone()

    # A reinforce/boost-STYLE in-place UPDATE (never calls mark_ro_pool_stale)
    # followed by a real WAL checkpoint genuinely invalidates the idle slot's
    # snapshot -- reproduces the exact gap.
    _reinforce_style_update(store, rid)
    _force_wal_checkpoint(store)

    gen_before = pool._current_generation()

    # dispatch #2: must NOT raise out of borrow()/the context even though the
    # pool's generation was never bumped by the checkpoint.
    with pool.borrow() as conn:
        row = conn.execute(
            "SELECT literal_surface FROM records WHERE id = ?", (rid,)
        ).fetchone()
    assert row is not None
    assert row[0] is not None and row[0].startswith("reinforced-"), (
        "the self-healed slot must serve the POST-checkpoint value, not a "
        "stale pre-checkpoint snapshot"
    )

    # dispatch #3: the original bug reproduced on TWO consecutive dispatches
    # (the single-retry fallback in core/__init__.py hit the same stale slot
    # on its own retry) -- prove the pool itself stays healthy across a second
    # borrow too, independent of any caller-side retry.
    with pool.borrow() as conn:
        row2 = conn.execute(
            "SELECT literal_surface FROM records WHERE id = ?", (rid,)
        ).fetchone()
    assert row2 is not None
    assert row2[0] is not None and row2[0].startswith("reinforced-")

    # The generation-refresh path (stale-generation branch) is untouched:
    # confirm the checkpoint alone did not bump the pool's own counter (that
    # is precisely the gap this proactive probe covers, not a generation fix).
    assert pool._current_generation() == gen_before

    store.close()


# ---------------------------------------------------------------------------
# Bounded-retry exhaustion: a slot that ALWAYS raises the fence must not spin
# forever -- borrow() raises so ro_conn() falls back to the writer path.
# ---------------------------------------------------------------------------


@_lilli_only
def test_bounded_retry_exhaustion_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _make_store(tmp_path)
    pool = store.db._ro_pool
    pool._size = 1  # deterministic: exactly one slot in play

    with pool.borrow():
        pass  # warm the single slot into the queue

    class _AlwaysFenced:
        def execute(self, sql, params=()):
            raise sqlite3.OperationalError(
                f"{_RO_SNAPSHOT_FENCE_MARKER}: simulated permanent invalidation"
            )

        def close(self):
            pass

    # Force the QUEUED slot itself to be the always-fenced connection, so the
    # very first probe inside borrow() already hits the fence (no real
    # checkpoint needed to exercise the bounded-retry ceiling itself).
    slot = pool._queue.get()
    slot.conn = _AlwaysFenced()
    pool._queue.put_nowait(slot)

    reopen_calls = {"n": 0}

    def _spy_open_slot():
        reopen_calls["n"] += 1
        return _AlwaysFenced()

    monkeypatch.setattr(pool, "_open_slot", _spy_open_slot)

    with pytest.raises(sqlite3.OperationalError) as excinfo:
        pool.borrow()

    assert _RO_SNAPSHOT_FENCE_MARKER in str(excinfo.value)
    # Every reopened replacement is ALSO always-fenced, so the retry loop
    # exhausts its bound and raises rather than spinning forever.
    from iai_mcp.hippo._ro_pool import _MAX_FENCE_REOPEN_ATTEMPTS

    assert reopen_calls["n"] == _MAX_FENCE_REOPEN_ATTEMPTS, (
        f"expected exactly {_MAX_FENCE_REOPEN_ATTEMPTS} reopen attempts before "
        f"raising, got {reopen_calls['n']} -- retry must be bounded, not spin "
        f"forever nor exit early"
    )

    store.close()


# ---------------------------------------------------------------------------
# Non-fence OperationalError must propagate unchanged -- never swallowed.
# ---------------------------------------------------------------------------


@_lilli_only
def test_non_fence_operational_error_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _make_store(tmp_path)
    pool = store.db._ro_pool
    pool._size = 1  # deterministic: exactly one slot in play

    with pool.borrow():
        pass  # warm the single slot into the queue

    class _GenuinelyBroken:
        def execute(self, sql, params=()):
            raise sqlite3.OperationalError("disk I/O error: simulated genuine failure")

        def close(self):
            pass

    def _spy_open_slot():
        return _GenuinelyBroken()

    monkeypatch.setattr(pool, "_open_slot", _spy_open_slot)

    # Force the queued slot itself to be the broken one so the very first
    # probe raises the non-fence error.
    slot = pool._queue.get()
    slot.conn = _GenuinelyBroken()
    pool._queue.put_nowait(slot)

    with pytest.raises(sqlite3.OperationalError) as excinfo:
        pool.borrow()

    assert _RO_SNAPSHOT_FENCE_MARKER not in str(excinfo.value)
    assert "disk I/O error" in str(excinfo.value), (
        "a non-fence OperationalError must propagate with its original "
        "message, not be swallowed or converted"
    )

    store.close()


# ---------------------------------------------------------------------------
# ro_conn() end-to-end: the checkpoint-invalidation repro through the actual
# caller-facing contextmanager, proving the defect no longer reproduces
# at the layer core.dispatch actually uses.
# ---------------------------------------------------------------------------


@_lilli_only
def test_ro_conn_end_to_end_survives_checkpoint_invalidation(tmp_path: Path) -> None:
    store = _make_store(tmp_path)

    # Seed + checkpoint FIRST (settles the header's cached db_size / resident
    # pages before any RO slot opens) -- see
    # test_borrow_self_heals_after_checkpoint_invalidation for why only an
    # in-place rewrite of an already-checkpointed row can trigger the fence.
    rid = _insert_record_direct(store, 0)
    _force_wal_checkpoint(store)

    with store.db.ro_conn() as conn:
        conn.execute(
            "SELECT literal_surface FROM records WHERE id = ?", (rid,)
        ).fetchone()

    _reinforce_style_update(store, rid)
    _force_wal_checkpoint(store)

    # Two consecutive dispatches through the exact ro_conn() contextmanager
    # caller code uses -- neither must raise.
    with store.db.ro_conn() as conn:
        row = conn.execute(
            "SELECT literal_surface FROM records WHERE id = ?", (rid,)
        ).fetchone()
    assert row is not None
    assert row[0] is not None and row[0].startswith("reinforced-")

    with store.db.ro_conn() as conn:
        row2 = conn.execute(
            "SELECT literal_surface FROM records WHERE id = ?", (rid,)
        ).fetchone()
    assert row2 is not None
    assert row2[0] is not None and row2[0].startswith("reinforced-")

    store.close()


# ---------------------------------------------------------------------------
# mark_stale / recycle / close are unaffected -- this fix adds a NEW path,
# it does not touch the existing generation-based refresh.
# ---------------------------------------------------------------------------


@_lilli_only
def test_existing_generation_refresh_still_works(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    pool = store.db._ro_pool

    rid = _insert_record_direct(store, 0)

    with pool.borrow() as conn:
        row = conn.execute(
            "SELECT id FROM records WHERE id = ?", (rid,)
        ).fetchone()
    assert row is not None

    # A row-count-changing write DOES bump the generation via the existing
    # mark_ro_pool_stale wiring (insert_pending_row / tombstone sites) -- this
    # fix must not have disturbed that path.
    store.db.mark_ro_pool_stale()
    gen_after = pool._current_generation()
    assert gen_after > 0

    with pool.borrow() as conn:
        conn.execute("SELECT COUNT(*) FROM records").fetchone()

    store.close()

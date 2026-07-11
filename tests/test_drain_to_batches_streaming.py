"""Streaming (lazy) guarantees for the record-batch drain path.

The drain path that backs ``iter_record_columns`` / ``to_batches`` must yield
each batch as it is fetched from the SQLite cursor, holding at most one batch in
memory at a time. A materializing variant (accumulate-all-then-return) would pull
the entire corpus into resident memory before the first batch is observable.

These tests pin two properties:

* correctness — the full row set comes out, in cursor order, with embeddings
  decoded exactly as before;
* laziness — consuming only the first batch does NOT drain the cursor (the
  underlying ``fetchmany`` is invoked once, not ``ceil(N / batch_size)`` times).

The laziness assertion is revert-proof: an accumulating drain fetches every row
before yielding anything, so ``fetchmany`` would already have been called for the
whole corpus by the time the first batch is pulled.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from iai_mcp.store import RECORDS_TABLE, MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _make_record(seed: int) -> MemoryRecord:
    rng = np.random.RandomState(seed)
    vec = rng.randn(EMBED_DIM).astype(np.float32).tolist()
    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=f"record seed {seed}",
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
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        language="en",
    )


class _CountingCursor:
    """Wraps a real sqlite cursor and tallies fetchmany invocations."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.fetchmany_calls = 0

    @property
    def description(self):
        return self._inner.description

    def fetchmany(self, size):
        self.fetchmany_calls += 1
        return self._inner.fetchmany(size)

    def fetchall(self):
        return self._inner.fetchall()

    def fetchone(self):
        return self._inner.fetchone()

    def close(self) -> None:
        self._inner.close()

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _SpyConnection:
    """Proxies a real sqlite connection, wrapping execute()'s cursor.

    ``sqlite3.Connection.execute`` is a read-only C attribute, so it cannot be
    monkeypatched in place. The drain path resolves the cursor through
    ``self._conn.execute(sql)`` where ``_conn`` is a plain Python attribute on
    the query object, so substituting this proxy lets us observe fetchmany.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.cursors: list[_CountingCursor] = []

    def execute(self, sql, *args, **kwargs):
        wrapped = _CountingCursor(self._inner.execute(sql, *args, **kwargs))
        self.cursors.append(wrapped)
        return wrapped

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _seed_store(tmp_path: Path, n: int) -> tuple[MemoryStore, list[MemoryRecord]]:
    store = MemoryStore(tmp_path, user_id="test")
    records = [_make_record(s) for s in range(n)]
    for rec in records:
        store.insert(rec)
    flush_record_buffer(store)
    return store, records


def test_drain_full_set_correct_and_ordered(tmp_path: Path) -> None:
    n = 500
    store, records = _seed_store(tmp_path, n)
    try:
        ids_inserted = {str(r.id) for r in records}

        out_rows = list(
            store.iter_record_columns(["id", "embedding"], batch_size=64)
        )

        # (a) full set, no loss, no duplication.
        assert len(out_rows) == n
        out_ids = [r["id"] for r in out_rows]
        assert set(out_ids) == ids_inserted
        assert len(set(out_ids)) == n

        # decode semantics preserved: each embedding is a plain float list of
        # EMBED_DIM, not raw bytes.
        for row in out_rows:
            emb = row["embedding"]
            assert isinstance(emb, list)
            assert len(emb) == EMBED_DIM
            assert all(isinstance(x, float) for x in emb)
    finally:
        store.close()


def test_drain_order_matches_cursor_order(tmp_path: Path) -> None:
    n = 300
    store, _records = _seed_store(tmp_path, n)
    try:
        tbl = store.db.open_table(RECORDS_TABLE)

        # Reference order: a single SELECT over the same table.
        ref_ids = [
            r["id"]
            for r in store.iter_record_columns(["id"], batch_size=1024)
        ]

        # Batched order must match the unbatched-equivalent order exactly.
        batched_ids = [
            r["id"]
            for r in store.iter_record_columns(["id"], batch_size=37)
        ]
        assert batched_ids == ref_ids

        # Batch boundaries respect batch_size (last batch may be short).
        batch_sizes = [
            batch.num_rows
            for batch in tbl.search().select(["id"]).to_batches(batch_size=37)
        ]
        assert sum(batch_sizes) == n
        assert all(s <= 37 for s in batch_sizes)
        assert all(s == 37 for s in batch_sizes[:-1])
    finally:
        store.close()


def test_drain_is_lazy_first_batch_does_not_drain_cursor(tmp_path: Path) -> None:
    n = 500
    batch_size = 50
    store, _records = _seed_store(tmp_path, n)
    try:
        tbl = store.db.open_table(RECORDS_TABLE)
        query = tbl.search().select(["id", "embedding"])

        # to_batches drains from a dedicated read-only snapshot connection (so a
        # full scan never holds the shared _conn_lock across the consumer). Wrap
        # that snapshot connection in the fetch-counting spy to observe laziness.
        spy_holder: list[_SpyConnection] = []
        _orig_open_snapshot = query._open_snapshot_conn

        def _spy_open_snapshot(db_path):
            spy = _SpyConnection(_orig_open_snapshot(db_path))
            spy_holder.append(spy)
            return spy

        query._open_snapshot_conn = _spy_open_snapshot  # type: ignore[assignment]

        gen = query.to_batches(batch_size=batch_size)

        first = next(gen)
        assert first.num_rows == batch_size

        # The snapshot connection is opened lazily, on the first pull.
        assert len(spy_holder) == 1
        spy = spy_holder[0]

        # A true generator has fetched exactly ONE chunk so far. A
        # materializing drain would have fetched the whole corpus
        # (ceil(500 / 50) + 1 == 11 calls) before yielding anything.
        assert len(spy.cursors) == 1
        assert spy.cursors[0].fetchmany_calls == 1, (
            "first batch must not drain the cursor; got "
            f"{spy.cursors[0].fetchmany_calls} fetchmany calls"
        )

        # Pulling a second batch advances the cursor by exactly one more
        # fetch — proving incremental, not pre-materialized, production.
        second = next(gen)
        assert second.num_rows == batch_size
        assert spy.cursors[0].fetchmany_calls == 2

        # Drain the rest and confirm the total still matches the corpus.
        remaining = sum(b.num_rows for b in gen)
        assert remaining == n - 2 * batch_size
        # Final empty fetchmany returns [] and stops the loop.
        assert spy.cursors[0].fetchmany_calls == (n // batch_size) + 1
    finally:
        store.close()

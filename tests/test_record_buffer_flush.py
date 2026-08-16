"""Record-buffer flush under IntegrityError: a genuinely poisoned batch must
not spin forever, and a non-duplicate integrity fault must never be assumed
durable.

The engine (both storage drivers) now raises `IntegrityError` on a duplicate
UNIQUE/PK via a plain `.add()`. `flush_record_buffer`'s batch write is
transactional, so ANY row's IntegrityError rolls the whole batch back; without
isolation the batch would stay buffered and the next flush would retry the
SAME poisoned batch forever. This suite proves: a confirmed-durable duplicate
is dropped with a loud log and the buffer no longer holds it; the other rows
in the same batch still land; a second flush is a no-op (no spin); and a
non-duplicate IntegrityError is hard-failed and unbuffered without ever being
treated as a safe drop.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture(autouse=True)
def _opt_out_of_buffer_autoflush(monkeypatch):
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")


def _make_record(literal_surface: str):
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=literal_surface,
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
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    )


def _clear_buffer(store) -> None:
    from iai_mcp import store as store_mod

    store_mod._record_buffer.pop(id(store), None)
    store_mod._record_last_flush_at.pop(id(store), None)
    store_mod._edge_buffer.pop(id(store), None)
    store_mod._edge_last_flush_at.pop(id(store), None)


def test_flush_drops_confirmed_durable_duplicate_and_lands_good_rows(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from iai_mcp import store as store_mod
    from iai_mcp.store import MemoryStore, flush_record_buffer

    with MemoryStore(path=tmp_path) as store:
        _clear_buffer(store)

        # alice's record already lands durably in records (via its own flush).
        existing = _make_record("alice-already-durable")
        store.insert(existing)
        assert flush_record_buffer(store) == 1

        # Buffer a batch: one genuinely new, distinct row, plus a row carrying
        # `existing`'s own id (the buffer is dict-shaped, same wire form
        # `.add` writes) -- simulating a duplicate re-entering the buffer.
        good = _make_record("alice-new-distinct")
        good_row = store._to_row(good)
        dup_row = dict(store._to_row(existing))
        store_mod._record_buffer[id(store)] = [dup_row, good_row]

        with caplog.at_level("WARNING"):
            flushed = flush_record_buffer(store)

        assert flushed == 1, "only the genuinely new row landed"
        assert not store_mod._record_buffer.get(id(store)), (
            "the buffer must be empty after the flush -- neither the "
            "dropped duplicate nor the landed row may retry"
        )
        assert any(
            "flush_record_buffer_dropped_durable_duplicate" in r.message
            for r in caplog.records
        ), "the drop must be logged loudly, not silently swallowed"

        recovered = {str(r.id): r for r in store.all_records()}
        assert str(good.id) in recovered
        assert str(existing.id) in recovered
        surfaces = [
            r.literal_surface
            for r in store.all_records()
            if r.id == existing.id
        ]
        assert len(surfaces) == 1, (
            "the duplicate must not have created a second physical row"
        )

        # A second flush is a no-op: proves no spin on the same poisoned batch.
        flushed2 = flush_record_buffer(store)
        assert flushed2 == 0
        assert not store_mod._record_buffer.get(id(store))


def _decode_recoverable_row(row: dict) -> dict:
    """Invert `_recoverable_row`: base64-wrapped bytes fields become bytes
    again, so a quarantined row reconstructs its exact stored content."""
    import base64

    out: dict = {}
    for k, v in row.items():
        if isinstance(v, dict) and set(v.keys()) == {"__bytes_b64__"}:
            out[k] = base64.b64decode(v["__bytes_b64__"])
        else:
            out[k] = v
    return out


def test_flush_quarantines_non_unique_integrity_error_recoverably(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A non-UNIQUE IntegrityError (e.g. NOT NULL, which the stdlib driver
    DOES route through IntegrityError) must never take the drop-if-durable
    path -- the row cannot be written as is, so it leaves the buffer to avoid
    an infinite retry, but its verbatim content is preserved to a recoverable
    dead-letter sink and the drop is surfaced loudly. It is NEVER silently
    dropped and NEVER assumed already durable.
    """
    import json

    from iai_mcp import store as store_mod
    from iai_mcp.errors import IntegrityError
    from iai_mcp.store import RECORDS_TABLE, MemoryStore, flush_record_buffer

    with MemoryStore(path=tmp_path) as store:
        _clear_buffer(store)

        good = _make_record("alice-good-row")
        bad = _make_record("alice-bad-verbatim-must-survive")
        good_row = store._to_row(good)
        bad_row = store._to_row(bad)
        # A non-empty bytes payload proves the base64 round-trip preserves raw
        # bytes exactly (never mangled through JSON).
        bad_row["structure_hv_payload"] = b"\x00\x01\x02\xffverbatim-bytes"
        store_mod._record_buffer[id(store)] = [bad_row, good_row]

        real_open_table = store.db.open_table
        real_table = real_open_table(RECORDS_TABLE)
        real_add = real_table.add

        def _add(rows):
            if bad_row in rows:
                raise IntegrityError("NOT NULL constraint failed: records.tier")
            return real_add(rows)

        class _Table:
            def add(self, rows):
                return _add(rows)

            def __getattr__(self, name):
                return getattr(real_table, name)

        store.db.open_table = lambda name: (
            _Table() if name == RECORDS_TABLE else real_open_table(name)
        )
        try:
            with caplog.at_level("ERROR"):
                flushed = flush_record_buffer(store)
        finally:
            store.db.open_table = real_open_table

        assert flushed == 1, "the good row in the same batch still lands"
        assert not store_mod._record_buffer.get(id(store)), (
            "the quarantined row must not stay buffered (no spin)"
        )
        assert any(
            "flush_record_buffer_integrity_quarantined" in r.message
            for r in caplog.records
        ), "a non-dup integrity fault must be logged loudly"

        recovered_ids = {str(r.id) for r in store.all_records()}
        assert str(good.id) in recovered_ids
        assert str(bad.id) not in recovered_ids, (
            "a quarantined row must never be treated as already durable"
        )

        # The verbatim content is preserved to a recoverable sink, not dropped.
        qdir = store.root / ".record-quarantine"
        qfiles = list(qdir.glob("*.jsonl"))
        assert qfiles, "the quarantined row must be written to a durable sink"
        # The sink holds plaintext record fields -> owner-only, never world-readable.
        import stat

        assert stat.S_IMODE(qdir.stat().st_mode) == 0o700, "quarantine dir must be 0700"
        for qf in qfiles:
            assert stat.S_IMODE(qf.stat().st_mode) == 0o600, "quarantine file must be 0600"
        lines = [
            json.loads(ln)
            for f in qfiles
            for ln in f.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        entry = next((e for e in lines if e.get("id") == str(bad.id)), None)
        assert entry is not None, "the bad row's content must be recoverable by id"
        assert "NOT NULL constraint failed" in entry["reason"]
        # Every field is preserved (nothing verbatim is lost), and the
        # verbatim-critical fields round-trip exactly -- the encrypted literal
        # surface, the embedding, and the raw bytes payload (via base64).
        decoded = _decode_recoverable_row(entry["row"])
        assert set(decoded.keys()) == set(bad_row.keys()), "no field may be lost"
        assert decoded["id"] == bad_row["id"]
        assert decoded["literal_surface"] == bad_row["literal_surface"]
        assert decoded["embedding"] == bad_row["embedding"]
        assert decoded["structure_hv"] == bad_row["structure_hv"]
        assert decoded["structure_hv_payload"] == bad_row["structure_hv_payload"]

        # The drop is observable beyond a log line: a loud telemetry event.
        from iai_mcp.events import query_events

        events = query_events(store, kind="record_quarantined")
        assert events, "a record_quarantined telemetry event must be emitted"
        assert any(str(bad.id) == (e.get("data") or {}).get("id") for e in events), (
            "the telemetry event must name the quarantined row"
        )


def test_flush_stops_per_row_retry_on_transient_fault_without_losing_content(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A transient (`OSError`/`RuntimeError`/`ValueError`) fault surfacing
    partway through the per-row IntegrityError-isolation retry must stop that
    retry immediately: the row that hit it, and every row after it in the
    batch, stay buffered for the next flush -- exactly like a transient
    fault on the whole-batch attempt. Only rows that already reached a
    terminal outcome (landed or hard-failed) before the transient fault are
    removed from the buffer.
    """
    from iai_mcp import store as store_mod
    from iai_mcp.errors import IntegrityError
    from iai_mcp.store import RECORDS_TABLE, MemoryStore, flush_record_buffer

    with MemoryStore(path=tmp_path) as store:
        _clear_buffer(store)

        bad = _make_record("alice-hard-fail-row")
        transient = _make_record("alice-transient-row")
        never_attempted = _make_record("alice-never-attempted-row")
        bad_row = store._to_row(bad)
        transient_row = store._to_row(transient)
        never_row = store._to_row(never_attempted)
        batch = [bad_row, transient_row, never_row]
        store_mod._record_buffer[id(store)] = list(batch)

        real_open_table = store.db.open_table
        real_table = real_open_table(RECORDS_TABLE)
        real_add = real_table.add
        calls: list[list[dict]] = []

        def _add(rows):
            calls.append(rows)
            if len(rows) == len(batch):
                # The whole-batch attempt: any row's IntegrityError rolls
                # the whole batch back, entering the per-row retry.
                raise IntegrityError("NOT NULL constraint failed: records.tier")
            if rows == [bad_row]:
                raise IntegrityError("NOT NULL constraint failed: records.tier")
            if rows == [transient_row]:
                raise OSError("disk full")
            return real_add(rows)

        class _Table:
            def add(self, rows):
                return _add(rows)

            def __getattr__(self, name):
                return getattr(real_table, name)

        store.db.open_table = lambda name: (
            _Table() if name == RECORDS_TABLE else real_open_table(name)
        )
        try:
            with caplog.at_level("WARNING"):
                flushed = flush_record_buffer(store)
        finally:
            store.db.open_table = real_open_table

        assert flushed == 0, "nothing landed: hard-fail + transient + unattempted"
        per_row_calls = [c for c in calls if len(c) == 1]
        assert not any(never_row in call_rows for call_rows in per_row_calls), (
            "the row after the transient fault must never be attempted"
        )
        remaining = store_mod._record_buffer.get(id(store), [])
        assert transient_row in remaining and never_row in remaining, (
            "the transient-faulting row and everything after it must stay "
            "buffered for the next flush"
        )
        assert bad_row not in remaining, (
            "the row that reached a terminal (hard-fail) outcome before the "
            "transient fault must not be re-buffered"
        )
        assert any(
            "flush_record_buffer_row_retry_transient" in r.message
            for r in caplog.records
        ), "the transient stop must be logged"

        recovered_ids = {str(r.id) for r in store.all_records()}
        assert str(bad.id) not in recovered_ids
        assert str(transient.id) not in recovered_ids
        assert str(never_attempted.id) not in recovered_ids

        # A later flush (transient fault cleared) lands the two buffered rows.
        store.db.open_table = real_open_table
        flushed2 = flush_record_buffer(store)
        assert flushed2 == 2
        recovered_ids2 = {str(r.id) for r in store.all_records()}
        assert str(transient.id) in recovered_ids2
        assert str(never_attempted.id) in recovered_ids2

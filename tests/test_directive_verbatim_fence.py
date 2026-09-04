"""literal_surface stays byte-identical across a directive flag set/clear.

Compares the raw stored column, not only the decrypted string -- AES-GCM
re-encryption of identical plaintext yields different ciphertext bytes, so
only the raw comparison actually proves the field was never rewritten.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.store import MemoryStore, RECORDS_TABLE
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _record(text: str) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.01] * EMBED_DIM,
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


def _raw_literal_surface(store: MemoryStore, record_id) -> str | None:
    with store.db._conn_lock:
        rows = store.db._conn.execute(
            f"SELECT literal_surface FROM {RECORDS_TABLE} WHERE id = ?",  # noqa: S608
            (str(record_id),),
        ).fetchall()
    return rows[0]["literal_surface"] if rows else None


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_verbatim_surface_untouched_by_flag_change(tmp_path, monkeypatch, driver):
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", driver)
    store = MemoryStore(path=tmp_path)

    rec = _record("the exact words the user wrote")
    rec.directive = True
    store.insert(rec)

    before_raw = _raw_literal_surface(store, rec.id)
    before_decoded = store.get(rec.id)
    assert before_decoded is not None
    assert before_decoded.directive is True
    assert before_decoded.literal_surface == rec.literal_surface

    tbl = store.db.open_table(RECORDS_TABLE)
    tbl.update(where=f"id = '{rec.id}'", values={"directive": False})

    after_raw = _raw_literal_surface(store, rec.id)
    after_decoded = store.get(rec.id)
    assert after_decoded is not None
    assert after_decoded.directive is False
    assert after_decoded.literal_surface == rec.literal_surface

    assert after_raw == before_raw

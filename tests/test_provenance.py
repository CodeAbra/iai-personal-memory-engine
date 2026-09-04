from __future__ import annotations

from uuid import UUID

from iai_mcp.core import dispatch
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM
from tests.test_store import _make

def test_recall_appends_provenance(tmp_path):
    store = MemoryStore(path=tmp_path)
    r = _make()
    store.insert(r)
    before = store.get(r.id).provenance
    dispatch(
        store,
        "memory_recall",
        {"cue": "test cue", "session_id": "s1", "cue_embedding": r.embedding},
    )
    after = store.get(r.id).provenance
    assert len(after) == len(before) + 1
    assert after[-1]["cue"] == "test cue"
    assert after[-1]["session_id"] == "s1"
    assert "ts" in after[-1]

def test_recall_appends_provenance_twice(tmp_path):
    store = MemoryStore(path=tmp_path)
    r = _make()
    store.insert(r)
    dispatch(store, "memory_recall", {"cue": "first", "session_id": "s1", "cue_embedding": r.embedding})
    dispatch(store, "memory_recall", {"cue": "second", "session_id": "s2", "cue_embedding": r.embedding})
    after = store.get(r.id).provenance
    assert len(after) == 2
    assert after[0]["cue"] == "first"
    assert after[1]["cue"] == "second"

def test_contradict_creates_linked_record_without_rewrite(tmp_path):
    store = MemoryStore(path=tmp_path)
    r = _make(text="Original fact")
    store.insert(r)
    original_text = store.get(r.id).literal_surface

    result = dispatch(
        store,
        "memory_contradict",
        {"id": str(r.id), "new_fact": "Contradicting fact", "cue_embedding": r.embedding},
    )
    assert result["edge_type"] == "contradicts"
    assert result["original_id"] == str(r.id)

    assert store.get(r.id).literal_surface == original_text

    new_rec = store.get(UUID(result["new_record_id"]))
    assert new_rec is not None
    assert new_rec.literal_surface == "Contradicting fact"
    assert "contradict" in new_rec.tags

def test_contradict_unknown_record_raises(tmp_path):
    import pytest
    store = MemoryStore(path=tmp_path)
    phantom_id = "11111111-2222-3333-4444-555555555555"
    with pytest.raises(ValueError):
        dispatch(
            store,
            "memory_contradict",
            {"id": phantom_id, "new_fact": "x", "cue_embedding": [0.0] * EMBED_DIM},
        )

def test_contradict_rejects_cyrillic_new_fact_without_raw_tag(tmp_path):
    import pytest

    store = MemoryStore(path=tmp_path)
    r = _make(text="Original fact")
    store.insert(r)

    with pytest.raises(ValueError, match="non-English"):
        dispatch(
            store,
            "memory_contradict",
            {
                "id": str(r.id),
                "new_fact": "Не правда, было не так",  # non-English fixture data: ru round-trip under test
                "cue_embedding": r.embedding,
            },
        )

def test_contradict_new_record_has_aaak_index_stamped(tmp_path):
    store = MemoryStore(path=tmp_path)
    r = _make(text="Original English fact")
    store.insert(r)

    result = dispatch(
        store,
        "memory_contradict",
        {
            "id": str(r.id),
            "new_fact": "English correction",
            "cue_embedding": r.embedding,
        },
    )
    new_rec = store.get(UUID(result["new_record_id"]))
    assert new_rec is not None
    assert new_rec.aaak_index != ""
    assert new_rec.aaak_index.startswith("W:")

def test_append_provenance_batch_basic(tmp_path):
    store = MemoryStore(path=tmp_path)
    r1 = _make(text="a")
    r2 = _make(text="b")
    r3 = _make(text="c")
    for r in (r1, r2, r3):
        store.insert(r)
    e1 = {"ts": "2026-04-17T00:00:00Z", "cue": "one", "session_id": "s1"}
    e2 = {"ts": "2026-04-17T00:00:01Z", "cue": "two", "session_id": "s1"}
    e3 = {"ts": "2026-04-17T00:00:02Z", "cue": "three", "session_id": "s1"}
    store.append_provenance_batch([(r1.id, e1), (r2.id, e2), (r3.id, e3)])
    got1 = store.get(r1.id).provenance
    got2 = store.get(r2.id).provenance
    got3 = store.get(r3.id).provenance
    assert got1[-1]["cue"] == "one"
    assert got2[-1]["cue"] == "two"
    assert got3[-1]["cue"] == "three"

def test_append_provenance_batch_multiple_entries_same_id(tmp_path):
    store = MemoryStore(path=tmp_path)
    r = _make()
    store.insert(r)
    before = len(store.get(r.id).provenance)
    e1 = {"ts": "t1", "cue": "first", "session_id": "s"}
    e2 = {"ts": "t2", "cue": "second", "session_id": "s"}
    store.append_provenance_batch([(r.id, e1), (r.id, e2)])
    after = store.get(r.id).provenance
    assert len(after) == before + 2
    assert after[-2]["cue"] == "first"
    assert after[-1]["cue"] == "second"

def test_append_provenance_batch_empty_list(tmp_path):
    store = MemoryStore(path=tmp_path)
    store.append_provenance_batch([])
    r = _make()
    store.insert(r)
    assert store.get(r.id) is not None

def test_append_provenance_batch_unknown_id(tmp_path):
    from uuid import uuid4
    store = MemoryStore(path=tmp_path)
    r = _make()
    store.insert(r)
    phantom = uuid4()
    e_known = {"ts": "t1", "cue": "known", "session_id": "s"}
    e_unknown = {"ts": "t0", "cue": "unknown", "session_id": "s"}
    store.append_provenance_batch([(phantom, e_unknown), (r.id, e_known)])
    after = store.get(r.id).provenance
    assert after[-1]["cue"] == "known"

def test_append_provenance_batch_equivalence_with_single(tmp_path):
    path_a = tmp_path / "a"
    path_b = tmp_path / "b"
    store_a = MemoryStore(path=path_a)
    store_b = MemoryStore(path=path_b)
    import copy
    base_records = [_make(text=f"r{i}") for i in range(3)]
    for r in base_records:
        store_a.insert(r)
        store_b.insert(copy.deepcopy(r))
    entries = [
        {"ts": "t1", "cue": "e1", "session_id": "sA"},
        {"ts": "t2", "cue": "e2", "session_id": "sB"},
        {"ts": "t3", "cue": "e3", "session_id": "sC"},
    ]
    for r, e in zip(base_records, entries):
        store_a.append_provenance(r.id, e)
    store_b.append_provenance_batch(list(zip([r.id for r in base_records], entries)))
    for r in base_records:
        prov_a = store_a.get(r.id).provenance
        prov_b = store_b.get(r.id).provenance
        assert prov_a == prov_b, f"provenance diverged for {r.id}: {prov_a} vs {prov_b}"

def test_append_provenance_batch_scan_count(tmp_path, monkeypatch):
    from iai_mcp.store import RECORDS_TABLE

    store = MemoryStore(path=tmp_path)
    recs = [_make(text=f"r{i}") for i in range(5)]
    for r in recs:
        store.insert(r)

    counter = [0]
    original_open_table = store.db.open_table

    def _wrap_to_pandas(obj):
        if not hasattr(obj, "to_pandas"):
            return obj
        original = obj.to_pandas

        def _counting_to_pandas(*a, **k):
            counter[0] += 1
            return original(*a, **k)

        try:
            obj.to_pandas = _counting_to_pandas  # type: ignore[assignment]
        except (AttributeError, TypeError):
            pass
        return obj

    def _counting_open_table(name, *args, **kwargs):
        tbl = original_open_table(name, *args, **kwargs)
        if name == RECORDS_TABLE:
            _wrap_to_pandas(tbl)
            if hasattr(tbl, "search"):
                original_search = tbl.search

                def _wrapped_search(*sa, **skw):
                    q = original_search(*sa, **skw)
                    return _wrap_to_pandas(q)

                try:
                    tbl.search = _wrapped_search  # type: ignore[assignment]
                except (AttributeError, TypeError):
                    pass
        return tbl

    store.db.open_table = _counting_open_table  # type: ignore[assignment]

    counter[0] = 0
    for r in recs:
        store.append_provenance(r.id, {"ts": "t", "cue": "c", "session_id": "s"})
    single_scans = counter[0]

    counter[0] = 0
    store.append_provenance_batch(
        [(r.id, {"ts": "t", "cue": "c2", "session_id": "s"}) for r in recs]
    )
    batch_scans = counter[0]

    assert single_scans == 5, f"append_provenance (single) did {single_scans} records-table scans; expected 5"
    # The batch path resolves the rows it needs with a single targeted
    # ``WHERE id IN (...)`` query rather than materializing the whole records
    # table, so it issues no full to_pandas scan at all (stronger than the
    # earlier single-scan N+1 collapse).
    assert batch_scans == 0, f"append_provenance_batch did {batch_scans} full scans; expected 0 (targeted IN-query)"

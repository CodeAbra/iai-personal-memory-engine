from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4



def _insert_raw_edge(
    store, src: str, dst: str, edge_type: str, weight: float, days_old: int,
) -> None:
    tbl = store.db.open_table("edges")
    old = datetime.now(timezone.utc) - timedelta(days=days_old)
    tbl.add([
        {
            "src": src,
            "dst": dst,
            "edge_type": edge_type,
            "weight": float(weight),
            "updated_at": old,
        }
    ])

def test_decay_edges_rejects_malformed_uuid(tmp_path):
    from iai_mcp.sleep import _decay_edges
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)

    clean_src_a = str(uuid4())
    clean_dst_a = str(uuid4())
    _insert_raw_edge(store, clean_src_a, clean_dst_a, "hebbian", weight=0.8, days_old=100)

    poisoned_src = "00000000-0000-0000-0000-000000000000' OR '1'='1"
    poisoned_dst = str(uuid4())
    _insert_raw_edge(store, poisoned_src, poisoned_dst, "hebbian", weight=0.5, days_old=100)

    clean_src_b = str(uuid4())
    clean_dst_b = str(uuid4())
    _insert_raw_edge(store, clean_src_b, clean_dst_b, "hebbian", weight=0.8, days_old=100)

    df_before = store.db.open_table("edges").to_pandas()
    assert len(df_before) == 3

    result = _decay_edges(store)

    assert result["decayed"] >= 2, (
        f"expected >= 2 clean rows decayed, got {result}"
    )

    df_after = store.db.open_table("edges").to_pandas()
    poisoned_row = df_after[df_after["src"] == poisoned_src]
    assert len(poisoned_row) == 1, "poisoned row must not be deleted"
    assert float(poisoned_row.iloc[0]["weight"]) == 0.5, (
        "poisoned row weight must not be decayed -- _uuid_literal rejected it"
    )

    row_a = df_after[df_after["src"] == clean_src_a]
    assert len(row_a) == 1
    assert float(row_a.iloc[0]["weight"]) < 0.8, (
        "clean row 0 should have been decayed"
    )

def test_decay_edges_imports_uuid_literal_at_module_scope():
    from iai_mcp import sleep as sleep_mod

    assert hasattr(sleep_mod, "_uuid_literal"), (
        "sleep.py must `from iai_mcp.store import _uuid_literal` at module scope"
    )

def test_decay_edges_single_poisoned_row_does_not_kill_sweep(tmp_path):
    from iai_mcp.sleep import _decay_edges
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)

    poisoned_src = "not-a-uuid-at-all"
    poisoned_dst = str(uuid4())
    _insert_raw_edge(store, poisoned_src, poisoned_dst, "hebbian", weight=0.02, days_old=500)

    clean_src = str(uuid4())
    clean_dst = str(uuid4())
    _insert_raw_edge(store, clean_src, clean_dst, "hebbian", weight=0.8, days_old=100)

    result = _decay_edges(store)

    assert (result["decayed"] + result["pruned"]) >= 1, (
        "sweep must continue past poisoned row"
    )

    df = store.db.open_table("edges").to_pandas()
    assert len(df[df["src"] == poisoned_src]) == 1

def test_migrate_imports_uuid_literal_at_module_scope():
    from iai_mcp import migrate as migrate_mod

    assert hasattr(migrate_mod, "_uuid_literal"), (
        "migrate.py must `from iai_mcp.store import _uuid_literal`"
    )

def test_migrate_delete_predicate_uses_uuid_literal(tmp_path, monkeypatch):
    import inspect
    from iai_mcp import migrate as migrate_mod

    src = inspect.getsource(migrate_mod)

    assert "_uuid_literal(record.id)" in src, (
        "migrate.py tbl.delete call must wrap record.id via _uuid_literal"
    )
    assert "id = '{record.id}'" not in src, (
        "migrate.py still contains raw f-string interpolation of record.id"
    )

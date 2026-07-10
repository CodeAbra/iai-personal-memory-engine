from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4



def _clear_edge_buffer(store) -> None:
    from iai_mcp import store as store_mod

    store_mod._edge_buffer.pop(id(store), None)
    store_mod._edge_last_flush_at.pop(id(store), None)


def _make_edge_row(
    src: str | None = None,
    dst: str | None = None,
    edge_type: str = "hebbian",
    weight: float = 0.5,
) -> dict:
    return {
        "src": src or str(uuid4()),
        "dst": dst or str(uuid4()),
        "edge_type": edge_type,
        "weight": weight,
        "updated_at": datetime.now(timezone.utc),
    }


def _table_row_count(store) -> int:
    from iai_mcp.store import EDGES_TABLE

    return len(store.db.open_table(EDGES_TABLE).to_pandas())


def test_flush_edge_buffer_all_distinct_land(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")

    from iai_mcp import store as store_mod
    from iai_mcp.store import MemoryStore, flush_edge_buffer

    with MemoryStore(path=tmp_path) as store:
        _clear_edge_buffer(store)

        n_before = _table_row_count(store)

        rows = [_make_edge_row() for _ in range(5)]
        for row in rows:
            store_mod._edge_buffer.setdefault(id(store), []).append(row)

        flushed = flush_edge_buffer(store)
        assert flushed == 5, f"expected 5 rows flushed; got {flushed}"

        n_after = _table_row_count(store)
        assert n_after == n_before + 5, (
            f"expected {n_before + 5} rows after flush of 5 distinct edges; got {n_after}"
        )


def test_flush_edge_buffer_partial_duplicate_no_exception(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")

    from iai_mcp import store as store_mod
    from iai_mcp.store import EDGES_TABLE, MemoryStore, flush_edge_buffer

    with MemoryStore(path=tmp_path) as store:
        _clear_edge_buffer(store)

        pre_existing = [_make_edge_row(weight=0.1) for _ in range(3)]
        store.db.open_table(EDGES_TABLE).add(pre_existing)

        n_after_seed = _table_row_count(store)
        assert n_after_seed == 3, f"seed failed: expected 3 rows, got {n_after_seed}"

        new_rows = [_make_edge_row(weight=0.9) for _ in range(2)]
        dup_rows = [
            {**row, "weight": 0.8, "updated_at": datetime.now(timezone.utc)}
            for row in pre_existing
        ]
        pending = dup_rows + new_rows

        for row in pending:
            store_mod._edge_buffer.setdefault(id(store), []).append(row)

        flushed = flush_edge_buffer(store)

        assert flushed == 5, f"expected 5 rows processed; got {flushed}"

        n_final = _table_row_count(store)
        assert n_final == 5, (
            f"expected 5 total rows after upsert (3 existing + 2 new); got {n_final}"
        )

        tbl = store.db.open_table(EDGES_TABLE)
        df = tbl.to_pandas()

        pre_srcs = {row["src"] for row in pre_existing}
        updated_rows = df[df["src"].isin(pre_srcs)]
        assert len(updated_rows) == 3, (
            f"expected 3 updated rows for pre-existing srcs; got {len(updated_rows)}"
        )
        for _, row in updated_rows.iterrows():
            assert abs(row["weight"] - 0.8) < 1e-9, (
                f"expected weight=0.8 (upsert latest wins) on existing edge; got {row['weight']}"
            )


def test_flush_edge_buffer_empty_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")

    from iai_mcp.store import MemoryStore, flush_edge_buffer

    with MemoryStore(path=tmp_path) as store:
        _clear_edge_buffer(store)

        n_before = _table_row_count(store)

        flushed = flush_edge_buffer(store)
        assert flushed == 0, f"expected 0 from empty flush; got {flushed}"

        n_after = _table_row_count(store)
        assert n_after == n_before, "empty flush changed row count"

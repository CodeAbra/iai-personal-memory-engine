from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4



def _insert_stale_edge(store, edge_type: str, weight: float, days_old: int):

    tbl = store.db.open_table("edges")
    old = datetime.now(timezone.utc) - timedelta(days=days_old)
    src_id, dst_id = str(uuid4()), str(uuid4())
    tbl.add([
        {
            "src": src_id,
            "dst": dst_id,
            "edge_type": edge_type,
            "weight": float(weight),
            "updated_at": old,
        }
    ])
    return src_id, dst_id


def test_decay_epsilon_default():
    from iai_mcp import sleep as sleep_mod

    assert sleep_mod.DECAY_EPSILON == 0.01


def test_decay_edges_preserves_fresh_hebbian_edges(tmp_path):
    from iai_mcp.sleep import _decay_edges
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    src, dst = _insert_stale_edge(store, "hebbian", weight=0.5, days_old=30)

    result = _decay_edges(store)
    assert result["decayed"] == 0
    assert result["pruned"] == 0

    df = store.db.open_table("edges").to_pandas()
    row = df[(df["src"] == src) & (df["dst"] == dst)]
    assert not row.empty
    assert float(row.iloc[0]["weight"]) == 0.5


def test_decay_edges_decays_stale_hebbian_edges(tmp_path):
    from iai_mcp.sleep import _decay_edges
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    src, dst = _insert_stale_edge(store, "hebbian", weight=0.8, days_old=100)

    result = _decay_edges(store)
    assert result["decayed"] >= 1

    df = store.db.open_table("edges").to_pandas()
    row = df[(df["src"] == src) & (df["dst"] == dst)]
    assert not row.empty
    assert float(row.iloc[0]["weight"]) < 0.8


def test_decay_edges_prunes_below_epsilon(tmp_path):
    from iai_mcp.sleep import _decay_edges
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    src, dst = _insert_stale_edge(store, "hebbian", weight=0.02, days_old=200)

    result = _decay_edges(store)
    assert result["pruned"] >= 1

    df = store.db.open_table("edges").to_pandas()
    gone = df[(df["src"] == src) & (df["dst"] == dst) & (df["edge_type"] == "hebbian")]
    assert gone.empty


def test_decay_edges_spares_contradicts(tmp_path):
    from iai_mcp.sleep import _decay_edges
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    src, dst = _insert_stale_edge(store, "contradicts", weight=0.5, days_old=1000)

    _decay_edges(store)

    df = store.db.open_table("edges").to_pandas()
    row = df[
        (df["src"] == src)
        & (df["dst"] == dst)
        & (df["edge_type"] == "contradicts")
    ]
    assert not row.empty
    assert float(row.iloc[0]["weight"]) == 0.5


def test_decay_edges_spares_invariant_anchor(tmp_path):
    from iai_mcp.sleep import _decay_edges
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    src, dst = _insert_stale_edge(store, "invariant_anchor", weight=0.001, days_old=5000)

    _decay_edges(store)
    df = store.db.open_table("edges").to_pandas()
    row = df[
        (df["src"] == src)
        & (df["dst"] == dst)
        & (df["edge_type"] == "invariant_anchor")
    ]
    assert not row.empty


def test_decay_edges_spares_consolidated_from(tmp_path):
    from iai_mcp.sleep import _decay_edges
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    src, dst = _insert_stale_edge(store, "consolidated_from", weight=0.01, days_old=2000)

    _decay_edges(store)
    df = store.db.open_table("edges").to_pandas()
    row = df[
        (df["src"] == src)
        & (df["dst"] == dst)
        & (df["edge_type"] == "consolidated_from")
    ]
    assert not row.empty


def test_decay_edges_custom_epsilon(tmp_path):
    from iai_mcp.sleep import _decay_edges
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    src, dst = _insert_stale_edge(store, "hebbian", weight=0.05, days_old=95)

    result_default = _decay_edges(store, epsilon=0.01)
    df = store.db.open_table("edges").to_pandas()
    remaining = df[(df["src"] == src) & (df["dst"] == dst) & (df["edge_type"] == "hebbian")]
    if not remaining.empty:
        store.db.open_table("edges").delete(
            f"src = '{src}' AND dst = '{dst}' AND edge_type = 'hebbian'"
        )

    src2, dst2 = _insert_stale_edge(store, "hebbian", weight=0.3, days_old=95)
    result_custom = _decay_edges(store, epsilon=0.5)
    df2 = store.db.open_table("edges").to_pandas()
    row = df2[(df2["src"] == src2) & (df2["dst"] == dst2) & (df2["edge_type"] == "hebbian")]
    assert row.empty
    assert result_custom["pruned"] >= 1

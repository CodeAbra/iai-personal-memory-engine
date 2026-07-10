from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4


from iai_mcp.store import EDGES_TABLE, MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _rec(vec=None, tags=None):
    vec = vec or [1.0] + [0.0] * (EMBED_DIM - 1)
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface="r",
        aaak_index="",
        embedding=vec,
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
        tags=list(tags or []),
        language="en",
    )


class _Hit:
    def __init__(self, rid: UUID, score: float):
        self.record_id = rid
        self.score = score


def test_curiosity_bridge_edge_on_fire(tmp_path):
    from iai_mcp.curiosity import fire_curiosity

    store = MemoryStore(path=tmp_path)
    recs = [_rec() for _ in range(3)]
    for r in recs:
        store.insert(r)
    hits = [_Hit(r.id, 0.5) for r in recs]

    q = fire_curiosity(
        store, hits, "ambiguous", entropy=0.85,
        session_id="s-bridge", turn=1,
    )
    assert q is not None

    edges = store.db.open_table(EDGES_TABLE).to_pandas()
    cb = edges[edges["edge_type"] == "curiosity_bridge"]
    assert len(cb) >= 3


def test_curiosity_bridge_edge_weight_proportional_entropy(tmp_path):
    from iai_mcp.curiosity import fire_curiosity

    store = MemoryStore(path=tmp_path)
    r1 = _rec()
    r2 = _rec()
    store.insert(r1)
    store.insert(r2)
    hits_low = [_Hit(r1.id, 0.5)]
    hits_high = [_Hit(r2.id, 0.5)]

    q1 = fire_curiosity(store, hits_low, "a", 0.75, session_id="s-a", turn=1)
    assert q1 is not None
    q2 = fire_curiosity(store, hits_high, "b", 0.95, session_id="s-b", turn=1)
    assert q2 is not None

    edges = store.db.open_table(EDGES_TABLE).to_pandas()
    cb = edges[edges["edge_type"] == "curiosity_bridge"]
    assert (cb["weight"] > 0).all()


def test_curiosity_bridge_edge_never_decays_in_sweep(tmp_path):
    from datetime import timedelta

    from iai_mcp.curiosity import fire_curiosity
    from iai_mcp.sleep import _decay_edges

    store = MemoryStore(path=tmp_path)
    r = _rec()
    store.insert(r)
    hits = [_Hit(r.id, 0.5)]
    fire_curiosity(store, hits, "c", 0.9, "s-never", turn=1)

    edges_tbl = store.db.open_table(EDGES_TABLE)
    ancient = datetime.now(timezone.utc) - timedelta(days=500)
    edges_tbl.update(
        where="edge_type = 'curiosity_bridge'",
        values={"updated_at": ancient, "weight": 0.0001},
    )
    _decay_edges(store)
    df = edges_tbl.to_pandas()
    cb = df[df["edge_type"] == "curiosity_bridge"]
    assert len(cb) >= 1

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.types import EMBED_DIM, MemoryRecord


def _anchor(s5_trust_score: float = 0.9) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface="User identity: alice",
        aaak_index="",
        embedding=[1.0] + [0.0] * (EMBED_DIM - 1),
        community_id=None,
        centrality=0.0,
        detail_level=5,
        pinned=True,
        stability=0.5,
        difficulty=0.3,
        last_reviewed=now,
        never_decay=True,
        never_merge=True,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=["identity"],
        language="en",
        s5_trust_score=s5_trust_score,
    )


class _FakeEmbedder:
    DIM = EMBED_DIM

    def embed(self, text):
        return [1.0] + [0.0] * (EMBED_DIM - 1)

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


@pytest.fixture(autouse=True)
def _patch_embedder(monkeypatch):
    from iai_mcp import embed as embed_mod

    monkeypatch.setattr(embed_mod, "Embedder", _FakeEmbedder)
    yield


def _reach_consensus(store, anchor_id):
    from iai_mcp.s5 import propose_invariant_update

    propose_invariant_update(store, anchor_id, "fact", "s1")
    propose_invariant_update(store, anchor_id, "fact", "s2")
    return propose_invariant_update(store, anchor_id, "fact", "s3")


def test_invariant_anchor_edge_on_s5_promotion(tmp_path):
    from iai_mcp.store import EDGES_TABLE, MemoryStore

    store = MemoryStore(path=tmp_path)
    anchor = _anchor()
    store.insert(anchor)
    verdict, new_id = _reach_consensus(store, anchor.id)
    assert verdict == "committed"
    assert new_id is not None

    df = store.db.open_table(EDGES_TABLE).to_pandas()
    ia = df[df["edge_type"] == "invariant_anchor"]
    assert len(ia) >= 1

    ids = {str(anchor.id), str(new_id)}
    found = any(
        {str(row["src"]), str(row["dst"])} == ids
        for _, row in ia.iterrows()
    )
    assert found


def test_invariant_anchor_edge_never_decays(tmp_path):
    from iai_mcp.sleep import _decay_edges
    from iai_mcp.store import EDGES_TABLE, MemoryStore

    store = MemoryStore(path=tmp_path)
    anchor = _anchor()
    store.insert(anchor)
    _reach_consensus(store, anchor.id)

    edges_tbl = store.db.open_table(EDGES_TABLE)
    df = edges_tbl.to_pandas()
    ia_rows = df[df["edge_type"] == "invariant_anchor"]
    assert not ia_rows.empty
    first = ia_rows.iloc[0]
    from datetime import timedelta as _td
    old_ts = datetime.now(timezone.utc) - _td(days=500)
    edges_tbl.update(
        where=(
            f"src = '{first['src']}' AND dst = '{first['dst']}' "
            f"AND edge_type = 'invariant_anchor'"
        ),
        values={"weight": 0.001, "updated_at": old_ts},
    )

    _decay_edges(store)

    df2 = store.db.open_table(EDGES_TABLE).to_pandas()
    survivors = df2[df2["edge_type"] == "invariant_anchor"]
    assert not survivors.empty


def test_invariant_anchor_edge_no_duplicate_within_cooldown(tmp_path):
    from iai_mcp.store import EDGES_TABLE, MemoryStore

    store = MemoryStore(path=tmp_path)
    anchor = _anchor()
    store.insert(anchor)
    _reach_consensus(store, anchor.id)

    df_after_first = store.db.open_table(EDGES_TABLE).to_pandas()
    ia_first = df_after_first[df_after_first["edge_type"] == "invariant_anchor"]
    count_first = len(ia_first)

    from iai_mcp.s5 import propose_invariant_update

    verdict, _ = propose_invariant_update(store, anchor.id, "another", "s4")
    assert verdict == "cooldown"

    df_after_second = store.db.open_table(EDGES_TABLE).to_pandas()
    ia_second = df_after_second[df_after_second["edge_type"] == "invariant_anchor"]
    assert len(ia_second) == count_first

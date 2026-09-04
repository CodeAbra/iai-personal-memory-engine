"""Membership-exact proof: the resident Rust rank index vs today's
LexicalIndex, over a corpus containing an `embedding_pending` row.

`build_runtime_graph` and `_rank_builder_graph_for` both exclude pending
rows at the SQL level (matching the vector/cosine scan's exclusion), while
`LexicalIndex` includes them. Feeding pending rows into the Rust index with
a zero vector and `pending=True` closes that membership gap. The
corpus-wide graph built here matches `build_runtime_graph`'s active-only
scope but skips its community-detection/centrality machinery, which is
irrelevant to index membership -- NOT a stand-in for `_rank_builder_graph_for`,
whose per-recall hydrated-candidate accumulation is a strict subset by
design and out of scope for this proof.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp.graph import MemoryGraph
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.store._rank_index import rank_index_for
from iai_mcp.types import MemoryRecord

_DIM = 16  # small synthetic dim -- keeps this a single-file, native-embedder-free test
_PENDING_TOKEN = "quirkalphamarker"


@pytest.fixture(autouse=True)
def _small_embed_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(_DIM))


@pytest.fixture(autouse=True)
def _crypto_passphrase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase-not-secret")


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(_keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p))
    monkeypatch.setattr(_keyring, "delete_password", lambda s, u: fake.pop((s, u), None))
    yield fake


def _unit_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _make_rec(rid: UUID, surface: str, embedding: list[float]) -> MemoryRecord:
    ts = datetime.now(timezone.utc)
    return MemoryRecord(
        id=rid,
        tier="episodic",
        literal_surface=surface,
        aaak_index="",
        embedding=embedding,
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
        created_at=ts,
        updated_at=ts,
        tags=["t"],
        language="en",
    )


def _build_corpus(store: MemoryStore) -> dict[str, UUID]:
    """Three ordinary records plus one `embedding_pending` record, all
    within the `tombstoned_at IS NULL` scope `LexicalIndex`'s build query
    covers -- the pending row's surface carries a distinctive token so its
    postings/BM25 inclusion is directly checkable."""
    ids = {"normal_1": uuid4(), "normal_2": uuid4(), "normal_3": uuid4()}
    store.insert(_make_rec(ids["normal_1"], "alpha filler record", _unit_vec(1)))
    store.insert(_make_rec(ids["normal_2"], "beta filler record", _unit_vec(2)))
    store.insert(_make_rec(ids["normal_3"], "gamma filler record", _unit_vec(3)))
    flush_record_buffer(store)

    pending_id = uuid4()
    now = datetime.now(timezone.utc).isoformat()
    store.insert_pending(
        record_id=str(pending_id),
        tier="episodic",
        literal_surface=f"a {_PENDING_TOKEN} pending row",
        tags_json="[]",
        provenance_json="[]",
        created_at=now,
        updated_at=now,
    )
    ids["pending"] = pending_id
    return ids


def _build_corpus_wide_graph(store: MemoryStore) -> MemoryGraph:
    """Every active, non-pending record -- the same
    `tombstoned_at IS NULL AND COALESCE(embedding_pending, 0) = 0` scope
    `retrieve.build_runtime_graph` streams, without its
    community-detection/centrality machinery."""
    graph = MemoryGraph()
    ids: list[UUID] = []
    for row in store.iter_record_columns(
        ["id"],
        batch_size=2048,
        where="tombstoned_at IS NULL AND COALESCE(embedding_pending, 0) = 0",
    ):
        try:
            ids.append(UUID(str(row["id"])))
        except (TypeError, ValueError):
            continue
    if not ids:
        return graph
    batch = store.get_batch(ids)
    for rid, rec in batch.items():
        payload = {
            "embedding": list(rec.embedding or []),
            "surface": rec.literal_surface or "",
            "centrality": float(rec.centrality or 0.0),
            "tier": rec.tier,
            "tags": list(rec.tags or []),
            "aaak_index": rec.aaak_index or "",
            "created_at": rec.created_at.isoformat() if rec.created_at else "",
            "stability": float(rec.stability or 0.0),
        }
        graph.add_node(rid, community_id=None, embedding=payload["embedding"])
        graph.set_node_payload(rid, payload)
    return graph


@pytest.fixture
def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "store")


def test_pending_row_in_lexical_out_of_cosine(_store: MemoryStore):
    ids = _build_corpus(_store)
    graph = _build_corpus_wide_graph(_store)
    handle = rank_index_for(_store, graph)

    generation, rust_ids, matrix, _degree, postings = handle.snapshot(
        graph, tokens=[_PENDING_TOKEN]
    )
    assert generation >= 0

    pending_int = ids["pending"].int
    assert pending_int in rust_ids, "the pending row must be resident in the Rust index"

    pending_flags = handle._index.pending()
    assert pending_flags[pending_int] is True, "the fed pending row must carry pending=True"
    for other_id, other_flag in pending_flags.items():
        if other_id != pending_int:
            assert other_flag is False, (
                f"a non-pending resident id {other_id} must not be marked pending"
            )

    slot = rust_ids.index(pending_int)
    assert np.allclose(matrix[slot], 0.0), (
        "a pending row must be resident with a zero vector, matching the "
        "zero embedding blob its own DB row already carries"
    )

    token_postings = postings.get(_PENDING_TOKEN, {})
    assert pending_int in token_postings, (
        "a pending row's surface tokens must participate in postings/BM25 "
        "candidate inclusion, matching today's LexicalIndex"
    )

    # Today's actual cosine-scan candidate source is the corpus-wide graph
    # (pipeline.py scores over graph._node_payload); it already excludes
    # pending rows by construction (build_runtime_graph's own SQL filter,
    # unrelated to this plan). No Rust cosine kernel exists yet at this
    # phase (no `cosine` token in lib.rs) for the resident `pending` flag
    # to filter against directly -- the flag + zero vector proven above are
    # the resident-state contract a future Rust-side cosine consumer
    # (265-05/06) will filter on to reproduce this same exclusion.
    assert str(ids["pending"]) not in graph._node_payload


def test_membership_matches_lexical_index(_store: MemoryStore):
    ids = _build_corpus(_store)
    graph = _build_corpus_wide_graph(_store)
    handle = rank_index_for(_store, graph)

    _generation, rust_ids, _matrix, _degree, _postings = handle.snapshot(graph, tokens=[])
    rust_membership = {str(UUID(int=i)) for i in rust_ids}

    _store.lexical_search("irrelevant probe query", k=1)
    idx = getattr(_store, "_lexical_idx", None)
    assert idx is not None, "lexical_search must build the LexicalIndex"
    lexical_membership = set(idx._doc_len.keys())

    assert rust_membership == lexical_membership, (
        "the resident Rust index's lexical membership must match today's "
        "LexicalIndex membership exactly over a corpus-wide build -- the "
        "precondition for a future LexicalIndex retirement to be "
        "membership-exact rather than a behavior change"
    )
    assert str(ids["pending"]) in rust_membership
    assert str(ids["pending"]) in lexical_membership
    for key in ("normal_1", "normal_2", "normal_3"):
        assert str(ids[key]) in rust_membership
        assert str(ids[key]) in lexical_membership

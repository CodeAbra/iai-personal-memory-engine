"""Focused tests for the consolidation-mint salience/entity signal.

Covers the mint-only edits in ``_create_semantic_summary``: entity tags feed
the AAAK entity field, a cluster at or above the floor is marked notable, and
the signal survives a write-time dedup fold into a near-duplicate survivor.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.aaak import parse_aaak_index
from iai_mcp.embed import embedder_for_store
from iai_mcp.sleep import CLUSTER_MIN_SIZE, _create_semantic_summary
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord
from tests.test_store import _make

_ENTITY_TOKEN = "nimbus-cluster"


def _select_driver(driver: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _member_embedding(i: int, count: int) -> list[float]:
    vec = [0.05] * EMBED_DIM
    span = EMBED_DIM // (count + 2)
    start = i * span
    for j in range(start, start + span):
        vec[j] = 0.9
    return vec


def _entity_cluster(store: MemoryStore) -> tuple[list[MemoryRecord], str]:
    texts = [
        f"alice deployed the `{_ENTITY_TOKEN}` service for the reporting pipeline",
        f"bob monitored `{_ENTITY_TOKEN}` metrics after the rollout finished",
        f"alice filed a ticket referencing `{_ENTITY_TOKEN}` latency spikes",
    ]
    members = [
        _make(text=t, vec=_member_embedding(i, len(texts)))
        for i, t in enumerate(texts)
    ]
    for m in members:
        store.insert(m)
    summary_text = "Cluster summary (3 records, lang=en): " + "; ".join(
        m.literal_surface for m in members
    )
    return members, summary_text


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_mint_carries_entity_tags_and_aaak_entity_field(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    members, summary_text = _entity_cluster(store)

    summary_id, folded = _create_semantic_summary(store, members, summary_text, "en")
    assert not folded

    summary = store.get(summary_id)
    assert summary is not None
    assert f"entity:{_ENTITY_TOKEN}" in summary.tags
    assert "semantic" in summary.tags and "cls_summary" in summary.tags

    parsed = parse_aaak_index(summary.aaak_index)
    assert parsed["entities"], "AAAK entity field must not be empty for a knowledge summary"
    assert _ENTITY_TOKEN in parsed["entities"]


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_mint_from_min_size_cluster_marked_notable(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    members, summary_text = _entity_cluster(store)
    assert len(members) >= CLUSTER_MIN_SIZE

    summary_id, folded = _create_semantic_summary(store, members, summary_text, "en")
    assert not folded

    summary = store.get(summary_id)
    assert summary is not None
    assert summary.salience_level == "notable"


def _reset_patsep_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "IAI_MCP_PATSEP_NEAR_DUP_THRESHOLD",
        "IAI_MCP_PATSEP_LINK_THRESHOLD",
        "IAI_MCP_PATSEP_LINK_INITIAL_WEIGHT",
        "IAI_MCP_PATSEP_TOP_K",
        "IAI_MCP_PATSEP_DRY_RUN",
    ):
        monkeypatch.delenv(var, raising=False)
    # dry-run defaults True under pytest; the SKIP id-rewrite effect these
    # fold tests assert is invisible otherwise.
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")


def _make_existing_survivor(
    store: MemoryStore,
    embedding: list[float],
    salience_level: str,
    extra_tags: list[str] | None = None,
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    survivor = MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface="pre-existing near-duplicate cluster summary",
        aaak_index="",
        embedding=embedding,
        community_id=None,
        centrality=0.0,
        detail_level=3,
        pinned=False,
        stability=0.5,
        difficulty=0.3,
        last_reviewed=now,
        never_decay=True,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=["semantic", "cls_summary"] + (extra_tags or []),
        language="en",
        salience_level=salience_level,
    )
    store.insert(survivor)
    return survivor


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_fold_survivor_raises_salience_and_unions_entity_tags(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    _reset_patsep_env(monkeypatch)
    store = MemoryStore(path=tmp_path)
    members, summary_text = _entity_cluster(store)
    emb = embedder_for_store(store).embed(summary_text)
    survivor = _make_existing_survivor(store, emb, salience_level="unflagged")

    summary_id, folded = _create_semantic_summary(store, members, summary_text, "en")
    assert folded, "the near-duplicate mint must dedup-fold into the existing survivor"
    assert summary_id == survivor.id

    row = store.get(survivor.id)
    assert row is not None
    assert row.salience_level == "notable"
    assert f"entity:{_ENTITY_TOKEN}" in row.tags

    parsed = parse_aaak_index(row.aaak_index)
    assert parsed["entities"], "fold must refresh the survivor's AAAK entity field"
    assert _ENTITY_TOKEN in parsed["entities"]


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_fold_survivor_regen_keeps_new_entity_within_cap(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    _reset_patsep_env(monkeypatch)
    store = MemoryStore(path=tmp_path)
    members, summary_text = _entity_cluster(store)
    emb = embedder_for_store(store).embed(summary_text)
    pre_existing_entities = [f"entity:capped-{i}" for i in range(16)]
    survivor = _make_existing_survivor(
        store, emb, salience_level="unflagged", extra_tags=pre_existing_entities,
    )

    summary_id, folded = _create_semantic_summary(store, members, summary_text, "en")
    assert folded, "the near-duplicate mint must dedup-fold into the existing survivor"
    assert summary_id == survivor.id

    row = store.get(survivor.id)
    assert row is not None
    parsed = parse_aaak_index(row.aaak_index)
    assert _ENTITY_TOKEN in parsed["entities"], (
        "newly-unioned entity must survive the 16-entity AAAK cap"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_fold_survivor_salience_never_lowered(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    _reset_patsep_env(monkeypatch)
    store = MemoryStore(path=tmp_path)
    members, summary_text = _entity_cluster(store)
    emb = embedder_for_store(store).embed(summary_text)
    survivor = _make_existing_survivor(store, emb, salience_level="critical")

    summary_id, folded = _create_semantic_summary(store, members, summary_text, "en")
    assert folded, "the near-duplicate mint must dedup-fold into the existing survivor"
    assert summary_id == survivor.id

    row = store.get(survivor.id)
    assert row is not None
    assert row.salience_level == "critical", "fold must never lower an existing higher salience mark"

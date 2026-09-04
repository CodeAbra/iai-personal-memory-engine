"""Shared FHRR entity-bind compose helper: multiplicity gating, the empty
fallback, and query-side recovery of a bound entity above a same-cluster
wrong-entity sample.
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from bench.fhrr_multiplicity_calibration import (
    FHRR_ENTITY_BIND_TAG,
    _compose_fhrr_entity_bind_payload,
)
from iai_mcp.lilli.core.projection import EMBED_DIM
from iai_mcp.lilli.crossmodal.embed_to_hv import from_embedding_fhrr
from iai_mcp.lilli.tiers import fhrr
from iai_mcp.types import MemoryRecord


def _select_driver(driver: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _distinct_embedding(i: int, total: int) -> list:
    vec = [0.1] * EMBED_DIM
    span = EMBED_DIM // (total + 2)
    start = i * span
    for j in range(start, start + span):
        vec[j] = 0.9
    return vec


def _seed_member(store, i: int, total: int, entities: list[str]) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    rec = MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=f"member note {i}",
        aaak_index="",
        embedding=_distinct_embedding(i, total),
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=True,
        never_merge=True,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[f"entity:{e}" for e in entities],
        language="en",
    )
    store.insert(rec)
    return rec


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_compose_low_multiplicity_binds_and_recovers_above_wrong_entity(
    driver, tmp_path, monkeypatch
) -> None:
    _select_driver(driver, monkeypatch)
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    total = 6
    # "hot" is shared by 3 members -- above the default MULTIPLICITY_CAP (2),
    # excluded entirely. "alpha"/"beta" are each unique to one member.
    rec0 = _seed_member(store, 0, total, ["hot"])
    rec1 = _seed_member(store, 1, total, ["hot"])
    rec2 = _seed_member(store, 2, total, ["hot"])
    rec3 = _seed_member(store, 3, total, ["alpha"])
    rec4 = _seed_member(store, 4, total, ["beta"])
    rec5 = _seed_member(store, 5, total, [])  # no extractable entity

    cluster = [rec0, rec1, rec2, rec3, rec4, rec5]
    batch = store.get_batch([r.id for r in cluster])
    cluster_recs = [batch[r.id] for r in cluster]

    payload, tags = _compose_fhrr_entity_bind_payload(cluster_recs)
    assert payload != b""
    assert len(payload) == 10000
    assert tags == [FHRR_ENTITY_BIND_TAG]

    retrieved_alpha = fhrr.unbind(payload, fhrr.role_hv("entity:alpha"))
    score_alpha_correct = fhrr.similarity(retrieved_alpha, from_embedding_fhrr(rec3.embedding))
    score_alpha_wrong = fhrr.similarity(retrieved_alpha, from_embedding_fhrr(rec4.embedding))
    assert score_alpha_correct > score_alpha_wrong, (
        "correct-entity retrieval score must exceed a same-cluster wrong-entity sample"
    )

    # "hot" exceeds the cap and must contribute zero bind terms -- its
    # retrieval score against its own members must not clear the bar the
    # surviving low-multiplicity entity does.
    retrieved_hot = fhrr.unbind(payload, fhrr.role_hv("entity:hot"))
    score_hot = fhrr.similarity(retrieved_hot, from_embedding_fhrr(rec0.embedding))
    assert score_hot < score_alpha_correct


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_compose_all_above_cap_or_no_entities_returns_empty_fallback(
    driver, tmp_path, monkeypatch
) -> None:
    _select_driver(driver, monkeypatch)
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    total = 4
    shared = [_seed_member(store, i, total, ["common"]) for i in range(3)]
    no_entity = _seed_member(store, 3, total, [])
    cluster = shared + [no_entity]
    batch = store.get_batch([r.id for r in cluster])
    cluster_recs = [batch[r.id] for r in cluster]

    payload, tags = _compose_fhrr_entity_bind_payload(cluster_recs)
    assert payload == b""
    assert tags == []


def test_cap_parameter_overrides_the_default() -> None:
    now = datetime.now(timezone.utc)

    def _rec(i: int, entity: str) -> MemoryRecord:
        return MemoryRecord(
            id=uuid4(),
            tier="episodic",
            literal_surface=f"member note {i}",
            aaak_index="",
            embedding=_distinct_embedding(i, 3),
            community_id=None,
            centrality=0.0,
            detail_level=2,
            pinned=False,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=True,
            never_merge=True,
            provenance=[],
            created_at=now,
            updated_at=now,
            tags=[f"entity:{entity}"],
            language="en",
        )

    rec0 = _rec(0, "pair")
    rec1 = _rec(1, "pair")  # multiplicity = 2

    payload_cap2, tags_cap2 = _compose_fhrr_entity_bind_payload([rec0, rec1], cap=2)
    assert payload_cap2 != b""
    assert tags_cap2 == [FHRR_ENTITY_BIND_TAG]

    payload_cap1, tags_cap1 = _compose_fhrr_entity_bind_payload([rec0, rec1], cap=1)
    assert payload_cap1 == b""
    assert tags_cap1 == []


def test_empty_cluster_returns_empty_fallback() -> None:
    payload, tags = _compose_fhrr_entity_bind_payload([])
    assert payload == b""
    assert tags == []


def test_tag_importable_from_bench_calibration_module() -> None:
    code = (
        "from bench.fhrr_multiplicity_calibration import FHRR_ENTITY_BIND_TAG\n"
        "assert FHRR_ENTITY_BIND_TAG == 'fhrr_entity_bind_v2'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert result.returncode == 0, result.stderr

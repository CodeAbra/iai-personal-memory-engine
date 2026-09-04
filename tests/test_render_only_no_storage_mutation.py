"""Structural byte-identity guard for the render-only cls_summary lever:
a full session-start render must never touch a cls_summary record's stored
literal_surface or embedding — the load-bearing proof that retrieval
(which reads only those two columns) cannot regress from this lever, by
construction rather than by statistical differential.

Drives the REAL build_runtime_graph / community-detection / centrality path
on a small in-process store so a cls_summary record genuinely reaches
rich_club membership as a star hub (maximal betweenness via its
consolidated_from edges to its members) — never a hand-injected
rich_club_ids list, which would make the guard vacuous.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp import runtime_graph_cache
from iai_mcp.retrieve import build_runtime_graph
from iai_mcp.session import RICH_CLUB_CLS_SUMMARY_CAP, _clean_surface, assemble_session_start
from iai_mcp.sleep import _create_semantic_summary
from iai_mcp.store import MemoryStore
from iai_mcp.types import CLS_SUMMARY_PREFIX_MARKER, CLS_SUMMARY_PREFIX_RE, EMBED_DIM, MemoryRecord

_MEMBER_COUNT = 5


def _select_driver(driver: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _distinct_embedding(i: int) -> list[float]:
    vec = [0.1] * EMBED_DIM
    span = EMBED_DIM // (_MEMBER_COUNT + 2)
    start = i * span
    for j in range(start, start + span):
        vec[j] = 0.9
    return vec


def _seed_member(store, i: int) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    rec = MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface=f"alice project note {i}: distinguishing detail about topic area {i}",
        aaak_index="",
        embedding=_distinct_embedding(i),
        community_id=None,
        centrality=0.0,
        detail_level=3,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=True,
        never_merge=True,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )
    store.insert(rec)
    return rec


def _seed_non_cls_summary(store, i: int) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    rec = MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface="alice unrelated content mixed into the same render pass",
        aaak_index="",
        embedding=_distinct_embedding(i),
        community_id=None,
        centrality=0.0,
        detail_level=3,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=True,
        never_merge=True,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )
    store.insert(rec)
    return rec


def _run_guard(store) -> None:
    members = [_seed_member(store, i) for i in range(_MEMBER_COUNT)]
    summary_text = (
        f"Cluster summary ({len(members)} records, lang=en): "
        + "; ".join(m.literal_surface[:80] for m in members)
    )
    summary_id, folded = _create_semantic_summary(store, members, summary_text, "en")
    assert not folded, "the cls_summary record dedup-folded into an existing survivor"
    non_cls = _seed_non_cls_summary(store, _MEMBER_COUNT)

    all_records = store.all_records()
    expected_count = len(members) + 2
    assert len(all_records) == expected_count, (
        f"expected {expected_count} distinct records, got {len(all_records)} "
        "— a dedup-fold or insert failure changed the construction"
    )
    assert len({r.id for r in all_records}) == expected_count, (
        "duplicate ids among the seeded records — the construction did not "
        "produce distinct rows"
    )
    assert non_cls.id in {r.id for r in all_records}

    batch = store.get_batch([summary_id])
    summary_rec = batch.get(summary_id)
    assert summary_rec is not None, "summary id does not round-trip from the store"
    pre_surface = summary_rec.literal_surface
    pre_embedding = list(summary_rec.embedding)

    runtime_graph_cache.invalidate(store)
    graph, assignment, rich_club = build_runtime_graph(store)

    centrality_map = {nid: graph.get_centrality(nid) for nid in graph.iter_nodes()}
    assert any(v != 0.0 for v in centrality_map.values()), (
        "resolved centrality is degenerate (all-zero) — a neutral degrade "
        "would seat an arbitrary node in rich_club, making this guard vacuous"
    )
    argmax_id = max(centrality_map, key=centrality_map.get)
    assert argmax_id == summary_id, (
        f"expected the cls_summary star hub ({summary_id}) to be the "
        f"centrality argmax, got {argmax_id} — the construction did not "
        "seat the hub"
    )
    assert summary_id in rich_club, "cls_summary hub not present in the computed rich_club"

    from iai_mcp.profile import default_state

    state = {**default_state(), "wake_depth": "standard"}
    payload = assemble_session_start(
        store, assignment, rich_club, session_id="structural-guard", profile_state=state
    )

    prefix_match = CLS_SUMMARY_PREFIX_RE.match(pre_surface)
    assert prefix_match is not None, "the mint-format surface does not match the full prefix pattern"
    expected_content = _clean_surface(pre_surface[prefix_match.end():])[:RICH_CLUB_CLS_SUMMARY_CAP]
    assert expected_content, "the matched full-pattern strip produced no content to render"

    summary_line = None
    for line in (payload.rich_club or "").splitlines():
        if ": " in line and line.split(": ", 1)[1] == expected_content:
            summary_line = line
            break
    assert summary_line is not None, (
        "no cls_summary record reached the render stripped+capped — the "
        "phase must not ship zero savings all-green"
    )
    content = summary_line.split(": ", 1)[1]
    assert not content.startswith(CLS_SUMMARY_PREFIX_MARKER)
    assert len(content) == RICH_CLUB_CLS_SUMMARY_CAP
    cleaned_pre = _clean_surface(pre_surface)
    assert len(content) < len(cleaned_pre[:60])

    post_batch = store.get_batch([summary_id])
    post_rec = post_batch.get(summary_id)
    assert post_rec is not None
    assert post_rec.literal_surface == pre_surface, (
        "literal_surface changed across a session-start render — the "
        "structural no-regression guarantee is broken"
    )
    assert list(post_rec.embedding) == pre_embedding, (
        "embedding changed across a session-start render — the structural "
        "no-regression guarantee is broken"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_render_only_leaves_cls_summary_storage_untouched(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    _run_guard(store)

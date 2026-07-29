"""Temporal binding at rank: explicit date mentions in the cue earn
matching-day records a bounded boost; a cue with no date mention leaves
recall byte-identical. Precision over recall in the parser — ambiguous
English month words must not fire.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from iai_mcp.temporal_cue import DateMention, matches_mentions, parse_date_mentions


class TestParser:
    @pytest.mark.parametrize(
        ("cue", "expected"),
        [
            ("what happened on 2026-07-04 exactly", [DateMention(2026, 7, 4)]),
            ("the 2026-07 retrospective", [DateMention(2026, 7, None)]),
            ("what did I decide on July 4", [DateMention(None, 7, 4)]),
            ("the 4th of July plans", [DateMention(None, 7, 4)]),
            ("meeting on July 4th, 2026", [DateMention(2026, 7, 4)]),
            ("back in July 2026", [DateMention(2026, 7, None)]),
            ("what we shipped in July", [DateMention(None, 7, None)]),
            ("notes from Jan 5", [DateMention(None, 1, 5)]),
            ("the December release", [DateMention(None, 12, None)]),
        ],
    )
    def test_explicit_forms_parse(self, cue, expected):
        assert parse_date_mentions(cue) == expected

    @pytest.mark.parametrize(
        "cue",
        [
            "may I ask what you did",
            "they march every morning",
            "the jan branch",
            "what is the plan",
            "remind me what we decided yesterday",
            "",
            "version 2026-99 of the spec",
        ],
    )
    def test_ambiguous_and_plain_cues_stay_silent(self, cue):
        assert parse_date_mentions(cue) == []

    def test_ambiguous_month_fires_with_day_attached(self):
        assert parse_date_mentions("the May 12 meeting") == [
            DateMention(None, 5, 12)
        ]
        assert parse_date_mentions("in March 2026") == [
            DateMention(2026, 3, None)
        ]


class TestMatching:
    _TS = datetime(2026, 7, 4, 15, 30, tzinfo=timezone.utc)

    def test_specified_components_must_all_match(self):
        assert matches_mentions(self._TS, [DateMention(2026, 7, 4)])
        assert matches_mentions(self._TS, [DateMention(None, 7, None)])
        assert not matches_mentions(self._TS, [DateMention(2026, 7, 5)])
        assert not matches_mentions(self._TS, [DateMention(2025, 7, 4)])

    def test_empty_mention_and_missing_ts_never_match(self):
        assert not matches_mentions(self._TS, [DateMention(None, None, None)])
        assert not matches_mentions(None, [DateMention(2026, 7, 4)])
        assert not matches_mentions(self._TS, [])


class TestRankTerm:
    def _setup(self, tmp_path):
        from uuid import uuid4
        from iai_mcp.community import CommunityAssignment
        from iai_mcp.graph import MemoryGraph
        from iai_mcp.store import MemoryStore, flush_record_buffer
        from iai_mcp.types import EMBED_DIM, MemoryRecord

        def one_hot(i):
            v = [0.0] * EMBED_DIM
            v[i] = 1.0
            return v

        def make(vec, text, created):
            return MemoryRecord(
                id=uuid4(), tier="episodic", literal_surface=text,
                aaak_index="", embedding=vec, community_id=None,
                centrality=0.0, detail_level=2, pinned=False, stability=0.0,
                difficulty=0.0, last_reviewed=None, never_decay=False,
                never_merge=False, provenance=[], created_at=created,
                updated_at=created, tags=[], language="en",
            )

        store = MemoryStore(path=tmp_path / "lancedb")
        on_day = make(
            one_hot(0), "picked the venue",
            datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc),
        )
        off_day = make(
            one_hot(0), "confirmed the venue",
            datetime(2026, 6, 20, 12, 0, tzinfo=timezone.utc),
        )
        for r in (on_day, off_day):
            store.insert(r)
        flush_record_buffer(store)

        graph = MemoryGraph()
        for r in (on_day, off_day):
            graph.add_node(r.id, community_id=None, embedding=list(r.embedding))
            graph.set_node_payload(r.id, {
                "embedding": list(r.embedding), "surface": r.literal_surface,
                "centrality": 0.0, "tier": r.tier, "tags": [], "language": "en",
            })
        cid = uuid4()
        assignment = CommunityAssignment(
            node_to_community={r.id: cid for r in (on_day, off_day)},
            community_centroids={cid: one_hot(0)},
            modularity=0.0, backend="flat", top_communities=[cid],
            mid_regions={cid: [on_day.id, off_day.id]},
        )

        class _E:
            DIM = EMBED_DIM

            def embed(self, text):
                return one_hot(0)

            def embed_batch(self, texts):
                return [one_hot(0) for _ in texts]

        return store, graph, assignment, _E(), on_day, off_day

    def test_date_cue_lifts_matching_day_record(self, tmp_path):
        from iai_mcp.pipeline import recall_for_benchmark

        store, graph, assignment, emb, on_day, off_day = self._setup(tmp_path)
        resp = recall_for_benchmark(
            store=store, graph=graph, assignment=assignment, rich_club=[],
            embedder=emb, cue="what venue did we pick on July 4",
            session_id="bench", k_hits=5,
        )
        ranks = {h.record_id: i for i, h in enumerate(resp.hits)}
        assert ranks[on_day.id] < ranks[off_day.id], (
            "the record created on the mentioned date must outrank its twin"
        )

    def test_plain_cue_leaves_order_to_base_scores(self, tmp_path):
        from iai_mcp.pipeline import recall_for_benchmark

        store, graph, assignment, emb, on_day, off_day = self._setup(tmp_path)
        resp = recall_for_benchmark(
            store=store, graph=graph, assignment=assignment, rich_club=[],
            embedder=emb, cue="what venue did we pick",
            session_id="bench", k_hits=5,
        )
        ranks = {h.record_id: i for i, h in enumerate(resp.hits)}
        # Identical cosine; the newer record wins on the age penalty alone.
        assert ranks[on_day.id] < ranks[off_day.id]

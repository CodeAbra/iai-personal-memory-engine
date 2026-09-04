"""Non-circularity gate for the ambient co-firing signal.

Proves, on planted data, that used_ids is not a restatement of the ranker's
own hit order -- the discipline this project applies to every new signal
before it is trusted anywhere downstream.
"""
from __future__ import annotations

import json
from pathlib import Path

from iai_mcp import cofire

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "cofire"
RANK_REVERSED_PATH = FIXTURES_DIR / "rank_reversed_planted.jsonl"
PARAPHRASE_ONLY_PATH = FIXTURES_DIR / "paraphrase_only.jsonl"

HIT_A = "a0a0a0a0-a0a0-a0a0-a0a0-a0a0a0a0a0a0"
HIT_B = "b0b0b0b0-b0b0-b0b0-b0b0-b0b0b0b0b0b0"
HIT_C = "c0c0c0c0-c0c0-c0c0-c0c0-c0c0c0c0c0c0"
HIT_D = "d0d0d0d0-d0d0-d0d0-d0d0-d0d0d0d0d0d0"
HIT_P = "p0p0p0p0-p0p0-p0p0-p0p0-p0p0p0p0p0p0"


def _load_fixture_objs(path: Path) -> "list[dict]":
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_used_ids_is_a_non_prefix_reverse_rank_subset_not_a_function_of_rank():
    objs = _load_fixture_objs(RANK_REVERSED_PATH)
    pairs = cofire.extract_recall_pairs(objs)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["hit_ids"] == [HIT_A, HIT_B, HIT_C, HIT_D]

    used_ids = cofire.compute_used_ids(
        pair["hit_ids"], pair["hit_surfaces"], pair["assistant_text"]
    )

    # Membership: a non-prefix subset. HA and HC are never used; HB and HD are.
    assert set(used_ids) == {HIT_B, HIT_D}
    for k in range(len(pair["hit_ids"]) + 1):
        assert used_ids != pair["hit_ids"][:k], (
            f"used_ids must not equal the rank-prefix at k={k}"
        )

    # Order: the assistant mentions HD before HB, the reverse of their rank
    # order -- used_ids must preserve first-mention order, not rank order.
    assert used_ids == [HIT_D, HIT_B]
    rank_order_subset = [hid for hid in pair["hit_ids"] if hid in set(used_ids)]
    assert rank_order_subset == [HIT_B, HIT_D]
    assert used_ids != rank_order_subset, (
        "used_ids must not equal the rank-ordered subset -- a signal that "
        "always agreed with rank order would be the self-confirmation trap "
        "reappearing under a new name"
    )


def test_paraphrase_without_verbatim_overlap_yields_no_used_id():
    """The v1 literal-overlap heuristic matches an exact substring of
    literal_surface, so a paraphrase with no verbatim overlap is a known
    false negative -- this is measured here, not hidden. Precision on real
    traffic (how often a genuine paraphrase should have counted as used) is
    unmeasured; this test only pins the current, honest v1 behavior: it
    never hallucinates usage from topical similarity alone.
    """
    objs = _load_fixture_objs(PARAPHRASE_ONLY_PATH)
    pairs = cofire.extract_recall_pairs(objs)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["hit_ids"] == [HIT_P]

    used_ids = cofire.compute_used_ids(
        pair["hit_ids"], pair["hit_surfaces"], pair["assistant_text"]
    )
    assert used_ids == [], (
        "a paraphrase with no verbatim overlap must not be counted as used "
        "-- this is the v1 heuristic's known limitation, not a hallucinated hit"
    )


def test_provenance_guard_only_retrieval_cofired_is_ever_written(tmp_path, monkeypatch):
    """After extract -> spool -> drain over a planted fixture, the new
    retrieval_cofired kind is written and the human-judgment
    retrieval_reinforced kind is never touched by this phase's paths.
    """
    from iai_mcp.events import flush_event_buffer, query_events
    from iai_mcp.store import MemoryStore

    monkeypatch.setenv("HOME", str(tmp_path))

    session_id = "cofire-provenance-session"
    objs = _load_fixture_objs(RANK_REVERSED_PATH)
    pairs = cofire.extract_recall_pairs(objs)
    assert len(pairs) == 1
    pair = pairs[0]
    used_ids = cofire.compute_used_ids(
        pair["hit_ids"], pair["hit_surfaces"], pair["assistant_text"]
    )
    assert used_ids

    cofire.write_cofire_spool(session_id, pair["hit_ids"], used_ids)

    store = MemoryStore(path=tmp_path / "store")
    counts = cofire.drain_cofire_spool(store)
    assert counts["events"] == 1
    flush_event_buffer(store)

    all_events = query_events(store)
    assert {e["kind"] for e in all_events} == {"retrieval_cofired"}, (
        "a fresh store drained from this phase's spool must carry exactly one "
        "event kind -- retrieval_cofired -- and nothing else, in particular "
        "never the human-judgment retrieval_reinforced stream written by "
        "retrieve.emit_retrieval_reinforced"
    )
    reinforced_events = query_events(store, kind="retrieval_reinforced")
    cofired_events = query_events(store, kind="retrieval_cofired")
    assert reinforced_events == []
    assert cofired_events, "the co-fire drain must produce retrieval_cofired events"
    assert cofired_events[0]["data"]["used_ids"] == used_ids

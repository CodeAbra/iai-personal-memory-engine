"""Ambient co-firing tracer: extract -> spool -> drain -> retrieval_cofired.

Proves the self-confirmation-trap escape end to end on a fixture transcript:
a returned-but-unused hit never reaches used_ids, used_ids order reflects
first-mention position in the assistant's own generated text (never rank),
and the whole chain stays isolated from the episodic spool and from
retrieval_reinforced -- on both storage drivers.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from iai_mcp import capture, cofire
from iai_mcp.events import flush_event_buffer, query_events
from iai_mcp.store import MemoryStore

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "cofire" / "list_hits_used_unused.jsonl"
FULL_RANK_ECHO_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "cofire" / "full_rank_echo.jsonl"
BACK_TO_BACK_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "cofire" / "back_to_back_recalls.jsonl"

HA_ID = "11111111-1111-1111-1111-111111111111"
HB_ID = "22222222-2222-2222-2222-222222222222"
HC_ID = "33333333-3333-3333-3333-333333333333"

SESSION_ID = "cofire-fixture-session"

_DRIVER_PARAMS = [
    pytest.param("stdlib", id="stdlib"),
    pytest.param("lilli", id="lilli"),
]


def _set_driver(monkeypatch: pytest.MonkeyPatch, driver: str) -> None:
    if driver == "stdlib":
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    else:
        pytest.importorskip("iai_mcp_native")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", driver)


def _load_fixture_objs(path: Path = FIXTURE_PATH) -> "list[dict]":
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_tracer_used_ids_first_mention_excludes_unused(tmp_path, monkeypatch, driver):
    _set_driver(monkeypatch, driver)
    monkeypatch.setattr(cofire, "_spool_root", lambda: tmp_path)
    monkeypatch.setattr(capture, "_spool_root", lambda: tmp_path)
    store = MemoryStore(path=tmp_path / "store")

    objs = _load_fixture_objs()
    pairs = cofire.extract_recall_pairs(objs)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["hit_ids"] == [HA_ID, HB_ID, HC_ID]

    used_ids = cofire.compute_used_ids(
        pair["hit_ids"], pair["hit_surfaces"], pair["assistant_text"]
    )
    assert used_ids == [HB_ID, HA_ID]
    assert HC_ID not in used_ids
    assert used_ids != [HA_ID, HB_ID]

    spool_path = cofire.write_cofire_spool(SESSION_ID, pair["hit_ids"], used_ids)
    assert spool_path.parent.name == ".cofire-spool"
    assert spool_path.parent != capture.deferred_captures_dir()

    counts = cofire.drain_cofire_spool(store)
    assert counts["events"] == 1
    flush_event_buffer(store)

    events = query_events(store, kind="retrieval_cofired")
    assert len(events) == 1
    data = events[0]["data"]
    assert data["session_id"] == SESSION_ID
    assert data["hit_ids"] == [HA_ID, HB_ID, HC_ID]
    assert data["used_ids"] == [HB_ID, HA_ID]

    # Provenance: this path never emits the human-judgment kind.
    assert query_events(store, kind="retrieval_reinforced") == []
    # Isolation: the co-fire spool never lands where the episodic drain looks.
    deferred_dir = capture.deferred_captures_dir()
    assert not deferred_dir.exists() or not any(deferred_dir.iterdir())


def test_is_full_rank_echo_predicate():
    assert cofire._is_full_rank_echo(["a", "b"], ["a", "b"]) is True
    assert cofire._is_full_rank_echo(["a", "b"], ["b", "a"]) is False
    assert cofire._is_full_rank_echo(["a", "b"], ["a"]) is False
    assert cofire._is_full_rank_echo([], []) is False


def test_full_rank_order_echo_suppressed_to_zero_pairs():
    objs = _load_fixture_objs(FULL_RANK_ECHO_FIXTURE_PATH)
    pairs = cofire.extract_recall_pairs(objs)
    assert len(pairs) == 1
    pair = pairs[0]
    assert len(pair["hit_ids"]) == 3

    used_ids = cofire.compute_used_ids(
        pair["hit_ids"], pair["hit_surfaces"], pair["assistant_text"]
    )
    assert used_ids == []


def test_selective_reordered_usage_still_produces_pairs():
    objs = _load_fixture_objs()
    pairs = cofire.extract_recall_pairs(objs)
    used_ids = cofire.compute_used_ids(
        pairs[0]["hit_ids"], pairs[0]["hit_surfaces"], pairs[0]["assistant_text"]
    )
    assert used_ids == [HB_ID, HA_ID]
    assert used_ids != []


def test_sibling_substring_not_admitted_without_own_evidence():
    hit_ids = ["id-a", "id-b"]
    surfaces = ["onboarding doc", "bob's onboarding doc lists three setup steps"]
    text = "Start with bob's onboarding doc lists three setup steps before anything else."
    used_ids = cofire.compute_used_ids(hit_ids, surfaces, text)
    assert used_ids == ["id-b"]
    assert "id-a" not in used_ids


def test_sibling_substring_counts_with_own_independent_evidence():
    hit_ids = ["id-a", "id-b"]
    surfaces = ["onboarding doc", "bob's onboarding doc lists three setup steps"]
    text = (
        "Start with bob's onboarding doc lists three setup steps before anything else. "
        "The onboarding doc was updated last week too."
    )
    used_ids = cofire.compute_used_ids(hit_ids, surfaces, text)
    assert used_ids == ["id-b", "id-a"]


def test_parse_list_hits_skips_missing_or_empty_record_id():
    content = [{
        "type": "text",
        "text": json.dumps({"hits": [
            {"record_id": "", "literal_surface": "empty id surface"},
            {"record_id": "keep-a", "literal_surface": "alpha surface text"},
            {"literal_surface": "missing id surface"},
            {"record_id": "keep-b", "literal_surface": "bravo surface text"},
            {"record_id": 42, "literal_surface": "non-string id surface"},
            {"record_id": "keep-c", "literal_surface": "charlie surface text"},
        ]}),
    }]
    hit_ids, hit_surfaces = cofire._parse_list_hits(content)
    assert hit_ids == ["keep-a", "keep-b", "keep-c"]
    assert hit_surfaces == ["alpha surface text", "bravo surface text", "charlie surface text"]

    # surviving pairing resolves against each hit's own surface, not a dropped one
    text = "Per bravo surface text, then alpha surface text -- charlie is not mentioned."
    used_ids = cofire.compute_used_ids(hit_ids, hit_surfaces, text)
    assert used_ids == ["keep-b", "keep-a"]
    assert "keep-c" not in used_ids


def test_compute_used_ids_never_admits_empty_hit_id():
    used_ids = cofire.compute_used_ids(
        ["", "real-id"],
        ["ghost surface", "real surface"],
        "ghost surface and real surface both appear",
    )
    assert "" not in used_ids
    assert used_ids == ["real-id"]


def test_back_to_back_recalls_do_not_cross_attribute_assistant_text():
    objs = _load_fixture_objs(BACK_TO_BACK_FIXTURE_PATH)
    pairs = cofire.extract_recall_pairs(objs)
    assert len(pairs) == 2

    first, second = pairs
    assert "first surface" in first["assistant_text"]
    assert "second surface" not in first["assistant_text"]
    assert "second surface" in second["assistant_text"]
    assert "first surface" not in second["assistant_text"]

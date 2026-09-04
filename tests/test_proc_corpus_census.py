"""Tracer + safety-guard tests for the read-only retrieval_reinforced census."""

from __future__ import annotations

import ast
import hashlib
import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

import bench.recall_accuracy_real as recall_accuracy_real
from bench.proc_corpus_census import (
    MIN_DISTINCT_SESSIONS,
    PAIR_COUNT_FLOOR,
    RANK_K,
    T_CUTOFF_RULE,
    bootstrap_lower_ci,
    derive_t_cutoff,
    distinct_session_distribution,
    events_sanity,
    inter_event_pairs_with_session,
    intra_event_pairs_with_session,
    load_assistant_records,
    load_reinforced_events,
    load_used_events,
    main,
    null_per_pair_deltas,
    ordered_pairs_inter_event,
    ordered_pairs_intra_event,
    pair_session_spread,
    pairs_meeting_session_floor,
    park_verdict,
    parse_tool_sequence,
    per_pair_deltas,
    rank_at_recall,
    repetition_count,
    run_census,
    split_odd_even,
    tool_bigram_spread,
    two_sided_aa_floor,
    used_hitids_repetition,
    value_metric,
)
from iai_mcp.lilli.profile.retrieval_tuning import RETRIEVAL_MIN_SAMPLES
from iai_mcp.events import flush_event_buffer, write_event
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import SCHEMA_VERSION_CURRENT, MemoryRecord

_CENSUS_SRC = Path(__file__).resolve().parent.parent / "bench" / "proc_corpus_census.py"

_FORBIDDEN_SUBSTRINGS = (
    ".literal_surface =",
    "UPDATE records",
    "UPDATE events",
    "merge_insert",
    "tbl.add(",
    ".write_event(",
    ".add([",
    "DELETE FROM",
    "INSERT INTO",
    "DROP TABLE",
)


def _fresh_store(tmp_path: Path) -> tuple[MemoryStore, Path]:
    home = tmp_path / "operator-home"
    store_root = home / ".iai-mcp"
    store = MemoryStore(path=store_root)
    return store, home


def _emit_reinforced(store: MemoryStore, session_id: str, ids: list[str]) -> None:
    write_event(
        store,
        kind="retrieval_reinforced",
        data={
            "session_id": session_id,
            "reinforced_ids": ids,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        severity="info",
        session_id=session_id,
        buffered=True,
    )


def test_tracer_intra_event_ordered_pair_counts(tmp_path):
    store, _home = _fresh_store(tmp_path)
    a, b, c = str(uuid4()), str(uuid4()), str(uuid4())
    _emit_reinforced(store, "sess-1", [a, b, c])
    _emit_reinforced(store, "sess-2", [a, b])
    flush_event_buffer(store)

    events = load_reinforced_events(store)
    pairs = ordered_pairs_intra_event(events)

    assert pairs[(a, b)] == 2
    assert pairs[(b, c)] == 1
    store.close()


def test_tracer_intra_event_loader_exhausts_full_table(tmp_path):
    store, _home = _fresh_store(tmp_path)
    for i in range(120):
        _emit_reinforced(store, f"sess-{i}", [str(uuid4()), str(uuid4())])
    flush_event_buffer(store)

    events = load_reinforced_events(store)
    assert len(events) == 120
    store.close()


def test_tracer_intra_event_loader_excludes_dash_session(tmp_path):
    store, _home = _fresh_store(tmp_path)
    a, b = str(uuid4()), str(uuid4())
    _emit_reinforced(store, "-", [a, b])
    _emit_reinforced(store, "sess-real", [a, b])
    flush_event_buffer(store)

    events = load_reinforced_events(store)
    assert len(events) == 1
    assert events[0]["session_id"] == "sess-real"
    store.close()


def test_tracer_intra_event_loader_sorts_ascending_by_ts(tmp_path):
    store, _home = _fresh_store(tmp_path)
    a, b = str(uuid4()), str(uuid4())
    _emit_reinforced(store, "sess-1", [a, b])
    _emit_reinforced(store, "sess-2", [b, a])
    flush_event_buffer(store)

    events = load_reinforced_events(store)
    assert len(events) == 2
    assert events[0]["ts"] <= events[1]["ts"]
    store.close()


def test_tracer_intra_event_pure_counter_takes_no_store(tmp_path):
    events = [
        {"data": {"reinforced_ids": ["x", "y", "z"]}},
        {"data": {"reinforced_ids": ["x", "y"]}},
    ]
    pairs = ordered_pairs_intra_event(events)
    assert pairs[("x", "y")] == 2
    assert pairs[("y", "z")] == 1


def test_tracer_intra_event_main_prints_real_number(tmp_path, monkeypatch, capsys):
    store, home = _fresh_store(tmp_path)
    a, b = str(uuid4()), str(uuid4())
    _emit_reinforced(store, "sess-1", [a, b])
    _emit_reinforced(store, "sess-2", [a, b])
    flush_event_buffer(store)
    store.close()

    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    monkeypatch.delenv("IAI_MCP_STORE", raising=False)
    monkeypatch.setattr(recall_accuracy_real, "_operator_home", lambda: home)
    monkeypatch.setattr("sys.argv", ["proc_corpus_census.py", "--driver", "stdlib"])

    main()

    out = capsys.readouterr().out
    assert '"count": 2' in out
    assert a in out
    assert b in out


def test_ordering_inter_event_pairs_last_to_first(tmp_path):
    store, _home = _fresh_store(tmp_path)
    a, b, c, d = str(uuid4()), str(uuid4()), str(uuid4()), str(uuid4())
    _emit_reinforced(store, "sess-1", [a, b])
    _emit_reinforced(store, "sess-1", [c, d])
    flush_event_buffer(store)

    events = load_reinforced_events(store)
    intra = ordered_pairs_intra_event(events)
    inter = ordered_pairs_inter_event(events)

    assert intra[(a, b)] == 1
    assert intra[(c, d)] == 1
    assert (b, c) not in intra
    assert inter[(b, c)] == 1
    assert (a, b) not in inter
    store.close()


def test_ordering_inter_event_never_crosses_sessions(tmp_path):
    store, _home = _fresh_store(tmp_path)
    a, b = str(uuid4()), str(uuid4())
    _emit_reinforced(store, "sess-1", [a])
    _emit_reinforced(store, "sess-2", [b])
    flush_event_buffer(store)

    events = load_reinforced_events(store)
    inter = ordered_pairs_inter_event(events)
    assert (a, b) not in inter
    assert sum(inter.values()) == 0
    store.close()


def test_pair_session_spread_distinct_sessions_intra(tmp_path):
    store, _home = _fresh_store(tmp_path)
    a, b = str(uuid4()), str(uuid4())
    _emit_reinforced(store, "sess-1", [a, b])
    _emit_reinforced(store, "sess-2", [a, b])
    _emit_reinforced(store, "sess-2", [a, b])
    flush_event_buffer(store)

    events = load_reinforced_events(store)
    spread = pair_session_spread(events, intra_event_pairs_with_session)
    assert spread[(a, b)] == {"sess-1", "sess-2"}
    store.close()


def test_pair_session_spread_distinct_sessions_inter(tmp_path):
    store, _home = _fresh_store(tmp_path)
    a, b = str(uuid4()), str(uuid4())
    _emit_reinforced(store, "sess-1", [a])
    _emit_reinforced(store, "sess-1", [b])
    flush_event_buffer(store)

    events = load_reinforced_events(store)
    spread = pair_session_spread(events, inter_event_pairs_with_session)
    assert spread[(a, b)] == {"sess-1"}
    store.close()


def test_distinct_session_distribution_buckets():
    spread = {
        ("a", "b"): {"s1"},
        ("c", "d"): {"s1", "s2"},
        ("e", "f"): {"s1", "s2", "s3"},
        ("g", "h"): {"s1", "s2", "s3", "s4"},
        ("i", "j"): {"s1", "s2", "s3", "s4", "s5"},
        ("k", "l"): {"s1", "s2", "s3", "s4", "s5", "s6"},
    }
    dist = distinct_session_distribution(spread)
    assert dist == {"1": 1, "2": 1, "3": 1, "4": 1, "5+": 2}


def test_ordering_two_orderings_never_collapsed_in_main(tmp_path, monkeypatch, capsys):
    store, home = _fresh_store(tmp_path)
    a, b, c, d = str(uuid4()), str(uuid4()), str(uuid4()), str(uuid4())
    _emit_reinforced(store, "sess-1", [a, b])
    _emit_reinforced(store, "sess-1", [c, d])
    flush_event_buffer(store)
    store.close()

    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    monkeypatch.delenv("IAI_MCP_STORE", raising=False)
    monkeypatch.setattr(recall_accuracy_real, "_operator_home", lambda: home)
    monkeypatch.setattr("sys.argv", ["proc_corpus_census.py", "--driver", "stdlib"])

    main()

    out_lines = capsys.readouterr().out.strip().splitlines()
    report = json.loads(out_lines[-2])
    assert report["a_intra_event"]["distinct_pairs"] == 2
    assert report["a_inter_event"]["distinct_pairs"] == 1
    assert report["a_intra_event"] != report["a_inter_event"]


def _insert_assistant_turn(
    store: MemoryStore, session_id: str, text: str, created_at: datetime | None = None
) -> None:
    now = created_at or datetime.now(timezone.utc)
    rec = MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.0] * store._embed_dim,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"ts": now.isoformat(), "cue": "test", "session_id": session_id, "role": "assistant"}],
        created_at=now,
        updated_at=now,
        tags=["capture", "role:assistant"],
        language="en",
        s5_trust_score=0.5,
        profile_modulation_gain={},
        schema_version=SCHEMA_VERSION_CURRENT,
    )
    store.insert(rec)


def test_context_only_hitids_never_reads_used_field(tmp_path):
    store, _home = _fresh_store(tmp_path)
    a, b = str(uuid4()), str(uuid4())
    write_event(
        store,
        kind="retrieval_used",
        data={"hit_ids": [a, b], "query": "some cue", "used": False, "budget_used": 10, "path": "baseline_recall"},
        severity="info",
        session_id="sess-1",
        buffered=True,
    )
    flush_event_buffer(store)

    events = load_used_events(store)
    pairs = used_hitids_repetition(events)
    assert pairs[tuple(sorted((a, b)))] == 1
    store.close()


def test_context_only_never_unioned_into_gating_pairs(tmp_path):
    store, _home = _fresh_store(tmp_path)
    a, b = str(uuid4()), str(uuid4())
    write_event(
        store,
        kind="retrieval_used",
        data={"hit_ids": [a, b], "query": "x", "used": True},
        severity="info",
        session_id="sess-1",
        buffered=True,
    )
    flush_event_buffer(store)

    reinforced_events = load_reinforced_events(store)
    intra_pairs = ordered_pairs_intra_event(reinforced_events)
    assert len(intra_pairs) == 0
    store.close()


def test_query_dropped_at_load(tmp_path):
    store, _home = _fresh_store(tmp_path)
    write_event(
        store,
        kind="retrieval_used",
        data={"hit_ids": [], "query": "sensitive cue text", "used": False},
        severity="info",
        session_id="sess-1",
        buffered=True,
    )
    flush_event_buffer(store)

    events = load_used_events(store)
    assert len(events) == 1
    assert "query" not in events[0]["data"]
    store.close()


def test_trailer_parse_real_trailer():
    surface = "did the thing\n[tools: memory_recall, memory_capture]"
    assert parse_tool_sequence(surface) == ["memory_recall", "memory_capture"]


def test_trailer_parse_overflow_suffix_stripped():
    names = ", ".join(f"tool{i}" for i in range(8))
    surface = f"turn text\n[tools: {names} +3]"
    assert parse_tool_sequence(surface) == [f"tool{i}" for i in range(8)]


def test_trailer_parse_no_trailer_returns_empty():
    assert parse_tool_sequence("just plain text, no trailer here") == []


def test_trailer_parse_decoy_mid_text_rejected():
    surface = "look at this [tools: fake, decoy] shape mid-sentence, not the real one"
    assert parse_tool_sequence(surface) == []


def test_bigram_created_at_datetime_shape():
    now = datetime.now(timezone.utc)
    records = [
        {
            "provenance": [{"session_id": "sess-1"}],
            "created_at": now,
            "literal_surface": "turn\n[tools: alpha, beta]",
        }
    ]
    spread = tool_bigram_spread(records)
    assert spread[("alpha", "beta")] == {"sess-1"}


def test_bigram_created_at_isoformat_string_shape():
    now_str = datetime.now(timezone.utc).isoformat()
    records = [
        {
            "provenance": [{"session_id": "sess-1"}],
            "created_at": now_str,
            "literal_surface": "turn\n[tools: alpha, beta]",
        }
    ]
    spread = tool_bigram_spread(records)
    assert spread[("alpha", "beta")] == {"sess-1"}


def test_bigram_spread_recurs_across_distinct_sessions(tmp_path):
    store, _home = _fresh_store(tmp_path)
    _insert_assistant_turn(store, "sess-1", "did stuff\n[tools: memory_recall, memory_capture]")
    _insert_assistant_turn(store, "sess-2", "did stuff again\n[tools: memory_recall, memory_capture]")
    flush_record_buffer(store)

    records = load_assistant_records(store)
    spread = tool_bigram_spread(records)
    assert spread[("memory_recall", "memory_capture")] == {"sess-1", "sess-2"}
    store.close()


def test_bigram_spread_dedupes_same_turn_captured_twice(tmp_path):
    store, _home = _fresh_store(tmp_path)
    fixed_ts = datetime.now(timezone.utc)
    _insert_assistant_turn(store, "sess-1", "did stuff\n[tools: alpha, beta]", created_at=fixed_ts)
    _insert_assistant_turn(store, "sess-1", "did stuff\n[tools: alpha, beta]", created_at=fixed_ts)
    flush_record_buffer(store)

    records = load_assistant_records(store)
    spread = tool_bigram_spread(records)
    assert spread[("alpha", "beta")] == {"sess-1"}
    store.close()


def test_load_assistant_records_matches_where_clause_count(tmp_path):
    store, _home = _fresh_store(tmp_path)
    _insert_assistant_turn(store, "sess-1", "did stuff\n[tools: alpha, beta]")
    _insert_assistant_turn(store, "sess-2", "did more stuff\n[tools: alpha, beta]")
    flush_record_buffer(store)

    records = load_assistant_records(store)
    assert len(records) == 2
    store.close()


def test_load_assistant_records_raises_on_truncated_read(tmp_path, monkeypatch):
    store, _home = _fresh_store(tmp_path)
    _insert_assistant_turn(store, "sess-1", "did stuff\n[tools: alpha, beta]")
    _insert_assistant_turn(store, "sess-2", "did more stuff\n[tools: alpha, beta]")
    flush_record_buffer(store)

    real_iter_records = store.iter_records

    def _truncated_iter_records(*args, **kwargs):
        for i, record in enumerate(real_iter_records(*args, **kwargs)):
            if i >= 1:
                return
            yield record

    monkeypatch.setattr(store, "iter_records", _truncated_iter_records)

    with pytest.raises(RuntimeError, match="truncated"):
        load_assistant_records(store)
    store.close()


def test_events_sanity_reports_count_and_oldest_ts(tmp_path):
    store, _home = _fresh_store(tmp_path)
    _emit_reinforced(store, "sess-1", [str(uuid4()), str(uuid4())])
    _emit_reinforced(store, "sess-2", [str(uuid4()), str(uuid4())])
    flush_event_buffer(store)

    sanity = events_sanity(store)
    assert sanity["events_count"] == 2
    assert sanity["oldest_event_ts"] is not None
    store.close()


def test_events_sanity_never_blocks_regardless_of_age(tmp_path):
    store, _home = _fresh_store(tmp_path)
    _emit_reinforced(store, "sess-1", [str(uuid4()), str(uuid4())])
    flush_event_buffer(store)

    sanity = events_sanity(store)
    assert isinstance(sanity, dict)
    assert "oldest_record_created_at" in sanity
    store.close()


def test_events_sanity_zero_events_reports_none_oldest(tmp_path):
    store, _home = _fresh_store(tmp_path)
    sanity = events_sanity(store)
    assert sanity["events_count"] == 0
    assert sanity["oldest_event_ts"] is None
    store.close()


_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _reinforced_row(session_id: str, ids: list[str], ts: datetime) -> dict:
    return {"data": {"reinforced_ids": ids}, "session_id": session_id, "ts": ts}


def _used_row(session_id: str, hit_ids: list[str], ts: datetime) -> dict:
    return {"data": {"hit_ids": hit_ids}, "session_id": session_id, "ts": ts}


def test_rank_k_and_t_cutoff_rule_are_committed_constants():
    assert RANK_K == 10
    assert T_CUTOFF_RULE == "median"


def test_derive_t_cutoff_odd_count_returns_middle_ts():
    events = [
        _reinforced_row("s1", ["a", "b"], _T0),
        _reinforced_row("s2", ["a", "b"], _T0 + timedelta(seconds=10)),
        _reinforced_row("s3", ["a", "b"], _T0 + timedelta(seconds=20)),
    ]
    assert derive_t_cutoff(events) == _T0 + timedelta(seconds=10)


def test_derive_t_cutoff_even_count_returns_midpoint():
    events = [
        _reinforced_row("s1", ["a", "b"], _T0),
        _reinforced_row("s2", ["a", "b"], _T0 + timedelta(seconds=10)),
        _reinforced_row("s3", ["a", "b"], _T0 + timedelta(seconds=20)),
        _reinforced_row("s4", ["a", "b"], _T0 + timedelta(seconds=30)),
    ]
    assert derive_t_cutoff(events) == _T0 + timedelta(seconds=15)


def test_repetition_count_restricted_to_history_before_cutoff():
    events = [
        _reinforced_row("s1", ["a", "b"], _T0),
        _reinforced_row("s2", ["a", "b"], _T0 + timedelta(seconds=10)),
        _reinforced_row("s3", ["a", "b"], _T0 + timedelta(seconds=100)),
    ]
    t_cutoff = _T0 + timedelta(seconds=50)
    counts = repetition_count(events, t_cutoff=t_cutoff)
    assert counts[("a", "b")] == 2


def test_rank_at_recall_returns_hit_ids_list_position():
    reinforced = [_reinforced_row("s1", ["a", "b"], _T0 + timedelta(seconds=100))]
    used = [_used_row("s1", ["x", "b", "y"], _T0 + timedelta(seconds=90))]
    assert rank_at_recall(reinforced, used, ("a", "b"), "s1") == 1


def test_rank_at_recall_none_when_b_absent_from_preceding_hits():
    reinforced = [_reinforced_row("s1", ["a", "b"], _T0 + timedelta(seconds=100))]
    used = [_used_row("s1", ["x", "y", "z"], _T0 + timedelta(seconds=90))]
    assert rank_at_recall(reinforced, used, ("a", "b"), "s1") is None


def test_rank_at_recall_none_when_no_preceding_event():
    reinforced = [_reinforced_row("s1", ["a", "b"], _T0 + timedelta(seconds=100))]
    used = [_used_row("s1", ["x", "b"], _T0 + timedelta(seconds=200))]
    assert rank_at_recall(reinforced, used, ("a", "b"), "s1") is None


def _planted_improving_rank_corpus() -> tuple[list[dict], list[dict]]:
    reinforced = [
        _reinforced_row("s-first", ["a", "b"], _T0),
        _reinforced_row("s-repeat", ["a", "b"], _T0 + timedelta(days=10)),
    ]
    used = [
        # First occurrence: b buried past RANK_K -> miss.
        _used_row("s-first", [f"filler-{i}" for i in range(RANK_K + 1)] + ["b"], _T0 - timedelta(seconds=1)),
        # Repeat occurrence: b at rank 0 -> hit.
        _used_row("s-repeat", ["b", "filler"], _T0 + timedelta(days=10) - timedelta(seconds=1)),
    ]
    return reinforced, used


def test_value_metric_positive_on_planted_improving_rank_data():
    reinforced, used = _planted_improving_rank_corpus()
    t_cutoff = _T0 + timedelta(days=1)
    assert value_metric(reinforced, used, t_cutoff=t_cutoff, k=RANK_K) > 0


def test_value_metric_is_exactly_mean_of_per_pair_deltas():
    reinforced, used = _planted_improving_rank_corpus()
    t_cutoff = _T0 + timedelta(days=1)
    deltas = per_pair_deltas(reinforced, used, t_cutoff=t_cutoff, k=RANK_K)
    assert value_metric(reinforced, used, t_cutoff=t_cutoff, k=RANK_K) == statistics.fmean(deltas.values())


def test_per_pair_deltas_excludes_repeat_session_inside_history_window():
    # FIRST and REPEAT both fall before t_cutoff -- no held-out observation
    # for this pair, so it must not appear in per_pair_deltas at all.
    reinforced = [
        _reinforced_row("s-first", ["a", "b"], _T0),
        _reinforced_row("s-repeat", ["a", "b"], _T0 + timedelta(hours=1)),
    ]
    used = [
        _used_row("s-first", ["b"], _T0 - timedelta(seconds=1)),
        _used_row("s-repeat", ["x"], _T0 + timedelta(hours=1) - timedelta(seconds=1)),
    ]
    t_cutoff = _T0 + timedelta(days=1)
    deltas = per_pair_deltas(reinforced, used, t_cutoff=t_cutoff, k=RANK_K)
    assert ("a", "b") not in deltas


def test_value_metric_zero_on_no_relationship_data():
    reinforced = [
        _reinforced_row("s-first", ["a", "b"], _T0),
        _reinforced_row("s-repeat", ["a", "b"], _T0 + timedelta(days=10)),
    ]
    used = [
        _used_row("s-first", ["b", "filler"], _T0 - timedelta(seconds=1)),
        _used_row("s-repeat", ["b", "filler"], _T0 + timedelta(days=10) - timedelta(seconds=1)),
    ]
    t_cutoff = _T0 + timedelta(days=1)
    assert value_metric(reinforced, used, t_cutoff=t_cutoff, k=RANK_K) == pytest.approx(0.0)


def test_none_rank_counts_as_miss_never_dropped_from_denominator():
    reinforced = [
        _reinforced_row("s-first", ["a", "b"], _T0),
        _reinforced_row("s-repeat", ["a", "b"], _T0 + timedelta(days=10)),
    ]
    used = [
        _used_row("s-first", ["b", "filler"], _T0 - timedelta(seconds=1)),
        # Repeat session's preceding recall exists but never surfaced b -- a
        # MISS, not an absent observation.
        _used_row("s-repeat", ["x", "y"], _T0 + timedelta(days=10) - timedelta(seconds=1)),
    ]
    t_cutoff = _T0 + timedelta(days=1)
    deltas = per_pair_deltas(reinforced, used, t_cutoff=t_cutoff, k=RANK_K)
    assert ("a", "b") in deltas
    assert deltas[("a", "b")] == pytest.approx(0.0 - 1.0)


def test_split_odd_even_disjoint_and_deterministic():
    session_ids = [f"sess-{i}" for i in range(10)]
    odd1, even1 = split_odd_even(session_ids)
    odd2, even2 = split_odd_even(session_ids)
    assert (odd1, even1) == (odd2, even2)
    assert set(odd1).isdisjoint(set(even1))
    assert set(odd1) | set(even1) == set(session_ids)


def test_two_sided_aa_floor_brackets_zero_on_balanced_null_deltas():
    deltas = {(f"a{i}", f"b{i}"): 0.5 for i in range(5)}
    deltas.update({(f"c{i}", f"d{i}"): -0.5 for i in range(5)})
    low, high = two_sided_aa_floor(deltas, iters=300, seed=0)
    assert low <= 0.0
    assert high >= 0.0


def test_two_sided_aa_floor_lower_bound_positive_on_planted_signed_effect():
    deltas = {(f"a{i}", f"b{i}"): 0.5 for i in range(5)}
    low, high = two_sided_aa_floor(deltas, iters=300, seed=0)
    assert low > 0.0


def test_bootstrap_lower_ci_consumes_per_pair_deltas_values_shape():
    reinforced, used = _planted_improving_rank_corpus()
    t_cutoff = _T0 + timedelta(days=1)
    deltas = per_pair_deltas(reinforced, used, t_cutoff=t_cutoff, k=RANK_K)
    result = bootstrap_lower_ci(deltas.values(), iters=50, seed=0)
    assert isinstance(result, float)


def test_null_per_pair_deltas_same_shape_as_per_pair_deltas():
    reinforced = [
        _reinforced_row("s1", ["a", "b"], _T0 + timedelta(days=2)),
        _reinforced_row("s2", ["a", "b"], _T0 + timedelta(days=3)),
        _reinforced_row("s3", ["a", "b"], _T0 + timedelta(days=4)),
        _reinforced_row("s4", ["a", "b"], _T0 + timedelta(days=5)),
    ]
    used = [
        _used_row("s1", ["b"], _T0 + timedelta(days=2) - timedelta(seconds=1)),
        _used_row("s2", ["b"], _T0 + timedelta(days=3) - timedelta(seconds=1)),
        _used_row("s3", ["b"], _T0 + timedelta(days=4) - timedelta(seconds=1)),
        _used_row("s4", ["b"], _T0 + timedelta(days=5) - timedelta(seconds=1)),
    ]
    t_cutoff = _T0 + timedelta(days=1)
    null_deltas = null_per_pair_deltas(reinforced, used, t_cutoff=t_cutoff, k=RANK_K)
    assert isinstance(null_deltas, dict)
    for pair, delta in null_deltas.items():
        assert isinstance(pair, tuple) and len(pair) == 2
        assert isinstance(delta, float)


def test_run_census_returns_every_verdict_doc_field(tmp_path):
    store, _home = _fresh_store(tmp_path)
    a, b = str(uuid4()), str(uuid4())
    _emit_reinforced(store, "sess-1", [a, b])
    _emit_reinforced(store, "sess-2", [a, b])
    write_event(
        store,
        kind="retrieval_used",
        data={"hit_ids": [b], "query": "x", "used": True},
        severity="info",
        session_id="sess-1",
        buffered=True,
    )
    flush_event_buffer(store)

    report = run_census(store)

    assert isinstance(report["value_metric"], float)
    assert isinstance(report["established_pairs"], int)
    assert isinstance(report["real_corpus_aa_floor"], tuple)
    assert len(report["real_corpus_aa_floor"]) == 2
    assert report["rank_k"] == RANK_K
    assert "a_intra_event" in report
    assert "a_inter_event" in report
    assert "b_context_only_not_a_gate" in report
    assert "c_tool_bigrams" in report
    store.close()


def test_run_census_value_metric_matches_standalone_value_metric_function(tmp_path):
    """report["value_metric"] must come from calling value_metric() itself,
    never a separately-maintained reimplementation of its formula.
    """
    store, _home = _fresh_store(tmp_path)
    a, b = str(uuid4()), str(uuid4())
    _emit_reinforced(store, "sess-1", [a, b])
    _emit_reinforced(store, "sess-2", [a, b])
    write_event(
        store,
        kind="retrieval_used",
        data={"hit_ids": [b], "query": "x", "used": True},
        severity="info",
        session_id="sess-1",
        buffered=True,
    )
    flush_event_buffer(store)

    report = run_census(store)
    reinforced_events = load_reinforced_events(store)
    used_events = load_used_events(store)
    t_cutoff = derive_t_cutoff(reinforced_events)
    expected = value_metric(reinforced_events, used_events, t_cutoff=t_cutoff, k=RANK_K)

    assert report["value_metric"] == expected
    store.close()


def test_main_wiring_prints_value_metric_and_real_corpus_aa_floor(
    tmp_path, monkeypatch, capsys
):
    store, home = _fresh_store(tmp_path)
    a, b = str(uuid4()), str(uuid4())
    _emit_reinforced(store, "sess-1", [a, b])
    _emit_reinforced(store, "sess-2", [a, b])
    flush_event_buffer(store)
    store.close()

    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    monkeypatch.delenv("IAI_MCP_STORE", raising=False)
    monkeypatch.setattr(recall_accuracy_real, "_operator_home", lambda: home)
    monkeypatch.setattr("sys.argv", ["proc_corpus_census.py", "--driver", "stdlib"])

    main()

    out_lines = capsys.readouterr().out.strip().splitlines()
    report = json.loads(out_lines[-2])
    assert "value_metric" in report
    assert "real_corpus_aa_floor" in report
    assert "rank_k" in report
    assert "t_cutoff" in report


def test_safety_static_scan_no_mutating_calls():
    src = _CENSUS_SRC.read_text(encoding="utf-8")
    for pattern in _FORBIDDEN_SUBSTRINGS:
        assert pattern not in src, f"forbidden mutating pattern found in census source: {pattern!r}"
    assert "query_events(" in src


def test_safety_static_scan_execute_calls_are_select_only():
    # The SQL string is assembled a few lines above each execute() call, not
    # inline in the call expression -- resolve the enclosing function's full
    # source (not just the call's own argument AST) to catch that shape.
    src = _CENSUS_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    execute_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "execute"
    ]
    assert execute_calls, "expected at least one read-only db._conn.execute(SELECT ...) call"
    for call in execute_calls:
        enclosing = min(
            (f for f in functions if f.lineno <= call.lineno <= (f.end_lineno or f.lineno)),
            key=lambda f: (f.end_lineno or f.lineno) - f.lineno,
        )
        func_src = (ast.get_source_segment(src, enclosing) or "").upper()
        assert "SELECT" in func_src, f"no SELECT literal found in {enclosing.name}"
        for banned in ("UPDATE ", "DELETE ", "INSERT ", "DROP ", "ALTER "):
            assert banned not in func_src, f"forbidden SQL keyword {banned!r} in {enclosing.name}"


def test_safety_static_scan_no_direct_operator_home_or_store_open():
    src = _CENSUS_SRC.read_text(encoding="utf-8")
    assert "_operator_home" not in src, (
        "census must never resolve the operator home itself -- only the "
        "reused open_eval_copy_store may do that, as the shutil.copy2 SOURCE"
    )
    assert "MemoryStore(" not in src, (
        "census must never open a store directly -- only through the "
        "reused open_eval_copy_store, which opens the COPY"
    )
    assert "shutil" not in src, (
        "census must not duplicate the copy logic -- it reuses "
        "open_eval_copy_store's existing shutil.copy2 path"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_read_only_safety_source_unchanged(tmp_path, monkeypatch, driver):
    if driver == "lilli":
        pytest.importorskip("iai_mcp_native")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    monkeypatch.delenv("IAI_MCP_STORE", raising=False)

    store, home = _fresh_store(tmp_path)
    a, b = str(uuid4()), str(uuid4())
    _emit_reinforced(store, "sess-1", [a, b])
    _emit_reinforced(store, "sess-2", [a, b])
    write_event(
        store,
        kind="retrieval_used",
        data={"hit_ids": [a, b], "query": "x", "used": True},
        severity="info",
        session_id="sess-1",
        buffered=True,
    )
    flush_event_buffer(store)
    _insert_assistant_turn(store, "sess-1", "did stuff\n[tools: memory_recall, memory_capture]")
    flush_record_buffer(store)
    store.close()

    db_path = home / ".iai-mcp" / "hippo" / "brain.sqlite3"
    mtime_before = db_path.stat().st_mtime
    size_before = db_path.stat().st_size
    hash_before = hashlib.sha256(db_path.read_bytes()).hexdigest()

    monkeypatch.setattr(recall_accuracy_real, "_operator_home", lambda: home)

    # Drives every reader run_census/main() exercise -- load_reinforced_events,
    # load_used_events, load_assistant_records (the decrypt path), and
    # events_sanity's raw SELECTs -- under the same byte-identity guard.
    with recall_accuracy_real.open_eval_copy_store(driver=driver) as copy_store:
        report = run_census(copy_store)

    assert report["a_intra_event"]["distinct_pairs"] >= 1
    assert report["c_tool_bigrams"]["distinct_bigrams"] >= 1
    assert "events_sanity" in report

    mtime_after = db_path.stat().st_mtime
    size_after = db_path.stat().st_size
    hash_after = hashlib.sha256(db_path.read_bytes()).hexdigest()
    assert mtime_after == mtime_before, "census run touched the source store mtime"
    assert size_after == size_before, "census run touched the source store size"
    assert hash_after == hash_before, "census run touched the source store bytes"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_both_drivers_agree_on_intra_event_count(tmp_path, monkeypatch, capsys, driver):
    if driver == "lilli":
        pytest.importorskip("iai_mcp_native")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    monkeypatch.delenv("IAI_MCP_STORE", raising=False)

    store, home = _fresh_store(tmp_path)
    a, b, c = str(uuid4()), str(uuid4()), str(uuid4())
    _emit_reinforced(store, "sess-1", [a, b, c])
    _emit_reinforced(store, "sess-2", [a, b])
    flush_event_buffer(store)
    store.close()

    monkeypatch.setattr(recall_accuracy_real, "_operator_home", lambda: home)
    monkeypatch.setattr("sys.argv", ["proc_corpus_census.py", "--driver", driver])

    main()

    out_lines = capsys.readouterr().out.strip().splitlines()
    payload = json.loads(out_lines[-1])
    assert payload["driver"] == driver
    assert payload["count"] == 2
    assert sorted(payload["top_pair"]) == sorted([a, b])


# ---------------------------------------------------------------------------
# Non-vacuity control -- SYNTHETIC data only. Proves the metric machinery
# (per_pair_deltas / value_metric / two_sided_aa_floor) can discriminate a
# planted signal from noise. The REAL non-vacuity floor -- whether real
# history clears its own real A/A floor -- comes from the live corpus run,
# filled in by the orchestrator, not from these synthetic fixtures.
# ---------------------------------------------------------------------------


def _planted_signal_corpus(
    n_pairs: int = PAIR_COUNT_FLOOR, n_repeat_sessions: int = 4
) -> tuple[list[dict], list[dict]]:
    """N established pairs, each with one HISTORY (miss) occurrence and
    several HELD-OUT (hit) occurrences -- a manufactured improving-rank
    trend, uniform across every pair.
    """
    reinforced: list[dict] = []
    used: list[dict] = []
    for i in range(n_pairs):
        a, b = f"sig-a{i}", f"sig-b{i}"
        first_session = f"sig-first-{i}"
        reinforced.append(_reinforced_row(first_session, [a, b], _T0))
        used.append(
            _used_row(
                first_session,
                [f"filler-{j}" for j in range(RANK_K + 1)] + [b],
                _T0 - timedelta(seconds=1),
            )
        )
        for r in range(n_repeat_sessions):
            session = f"sig-repeat-{i}-{r}"
            ts = _T0 + timedelta(days=10, hours=r)
            reinforced.append(_reinforced_row(session, [a, b], ts))
            used.append(_used_row(session, [b, "filler"], ts - timedelta(seconds=1)))
    return reinforced, used


def _iid_noise_corpus(
    n_pairs: int = 20, n_repeat_sessions: int = 6, seed: int = 0
) -> tuple[list[dict], list[dict]]:
    """N established pairs where every occurrence's hit/miss outcome is an
    independent 50/50 coin flip, uncorrelated with FIRST-vs-REPEAT status --
    no rank/repetition relationship planted anywhere.
    """
    import random as _random

    rng = _random.Random(seed)
    reinforced: list[dict] = []
    used: list[dict] = []

    def _observation(session: str, ts: datetime, b: str) -> None:
        if rng.random() < 0.5:
            used.append(_used_row(session, [b, "filler"], ts - timedelta(seconds=1)))
        else:
            used.append(
                _used_row(
                    session,
                    [f"filler-{j}" for j in range(RANK_K + 1)] + [b],
                    ts - timedelta(seconds=1),
                )
            )

    for i in range(n_pairs):
        a, b = f"noise-a{i}", f"noise-b{i}"
        first_session = f"noise-first-{i}"
        reinforced.append(_reinforced_row(first_session, [a, b], _T0))
        _observation(first_session, _T0, b)
        for r in range(n_repeat_sessions):
            session = f"noise-repeat-{i}-{r}"
            ts = _T0 + timedelta(days=10, hours=r)
            reinforced.append(_reinforced_row(session, [a, b], ts))
            _observation(session, ts, b)
    return reinforced, used


def test_non_vacuity_planted_improving_signal_clears_the_floor():
    reinforced, used = _planted_signal_corpus()
    t_cutoff = _T0 + timedelta(days=1)

    vm = value_metric(reinforced, used, t_cutoff=t_cutoff, k=RANK_K)
    null_deltas = null_per_pair_deltas(reinforced, used, t_cutoff=t_cutoff, k=RANK_K)
    low, high = two_sided_aa_floor(null_deltas, iters=300, seed=0)

    assert vm > high, "planted improving-rank signal did not clear its own A/A floor"


def test_non_vacuity_iid_noise_stays_within_the_floor():
    reinforced, used = _iid_noise_corpus()
    t_cutoff = _T0 + timedelta(days=1)

    vm = value_metric(reinforced, used, t_cutoff=t_cutoff, k=RANK_K)
    null_deltas = null_per_pair_deltas(reinforced, used, t_cutoff=t_cutoff, k=RANK_K)
    low, high = two_sided_aa_floor(null_deltas, iters=500, seed=0)

    assert low <= vm <= high, "i.i.d. noise with no planted relationship was flagged as a signal"


# ---------------------------------------------------------------------------
# park_verdict -- pre-committed thresholds + the scalar-vs-floor decision rule
# ---------------------------------------------------------------------------


def test_threshold_pair_count_floor_equals_retrieval_min_samples():
    assert PAIR_COUNT_FLOOR == RETRIEVAL_MIN_SAMPLES == 20


def test_threshold_min_distinct_sessions_is_three_and_assumed():
    assert MIN_DISTINCT_SESSIONS == 3


def test_threshold_pairs_meeting_session_floor_counts_at_and_above_min_sessions():
    distribution = {"1": 4, "2": 3, "3": 2, "4": 1, "5+": 5}
    assert pairs_meeting_session_floor(distribution, min_sessions=3) == 2 + 1 + 5
    assert pairs_meeting_session_floor(distribution, min_sessions=1) == 4 + 3 + 2 + 1 + 5


def test_park_verdict_both_signals_pass_proceeds():
    verdict = park_verdict(
        gating_pairs_over_floor=25,
        value_metric=0.2,
        established_pairs=200,
        real_aa_floor=(-0.05, 0.05),
        signalb_bigrams_over_floor=25,
    )
    assert verdict["milestone_verdict"] == "proceed"
    assert verdict["signal_b_verdict"] == "proceed"


def test_park_verdict_gating_thin_parks_milestone():
    verdict = park_verdict(
        gating_pairs_over_floor=5,
        value_metric=0.2,
        established_pairs=200,
        real_aa_floor=(-0.05, 0.05),
        signalb_bigrams_over_floor=25,
    )
    assert verdict["milestone_verdict"] == "PARK"
    assert verdict["gating_clears_floor"] is False


def test_park_verdict_value_metric_within_floor_parks_milestone():
    verdict = park_verdict(
        gating_pairs_over_floor=25,
        value_metric=0.02,
        established_pairs=200,
        real_aa_floor=(-0.05, 0.05),
        signalb_bigrams_over_floor=25,
    )
    assert verdict["milestone_verdict"] == "PARK"
    assert verdict["value_metric_clears_floor"] is False


def test_park_verdict_signal_b_thin_parks_signal_b_only_not_milestone():
    verdict = park_verdict(
        gating_pairs_over_floor=25,
        value_metric=0.2,
        established_pairs=200,
        real_aa_floor=(-0.05, 0.05),
        signalb_bigrams_over_floor=5,
    )
    assert verdict["milestone_verdict"] == "proceed"
    assert verdict["signal_b_verdict"] == "PARK"


def test_park_verdict_boundary_value_metric_exactly_at_floor_high_parks():
    verdict = park_verdict(
        gating_pairs_over_floor=25,
        value_metric=0.05,
        established_pairs=200,
        real_aa_floor=(-0.05, 0.05),
        signalb_bigrams_over_floor=25,
    )
    assert verdict["milestone_verdict"] == "PARK", "equal to the upper null bound does not clear it"


def test_park_verdict_boundary_value_metric_just_above_floor_high_proceeds():
    verdict = park_verdict(
        gating_pairs_over_floor=25,
        value_metric=0.05 + 1e-9,
        established_pairs=200,
        real_aa_floor=(-0.05, 0.05),
        signalb_bigrams_over_floor=25,
    )
    assert verdict["milestone_verdict"] == "proceed"


def test_park_verdict_boundary_value_metric_just_below_floor_high_parks():
    verdict = park_verdict(
        gating_pairs_over_floor=25,
        value_metric=0.05 - 1e-9,
        established_pairs=200,
        real_aa_floor=(-0.05, 0.05),
        signalb_bigrams_over_floor=25,
    )
    assert verdict["milestone_verdict"] == "PARK"


def test_park_verdict_zero_established_pairs_flags_underpowered():
    verdict = park_verdict(
        gating_pairs_over_floor=0,
        value_metric=0.0,
        established_pairs=0,
        real_aa_floor=(0.0, 0.0),
        signalb_bigrams_over_floor=0,
    )
    assert verdict["milestone_verdict"] == "PARK"
    assert verdict["underpowered"] is True


def test_park_verdict_nonzero_established_pairs_not_underpowered_even_when_parked():
    # A genuinely measured, flat value_metric PARKs on the metric arm --
    # underpowered must stay False so this is never confused with the
    # zero-established-pairs case above, which PARKs for a different reason.
    verdict = park_verdict(
        gating_pairs_over_floor=25,
        value_metric=0.0,
        established_pairs=200,
        real_aa_floor=(-0.05, 0.05),
        signalb_bigrams_over_floor=25,
    )
    assert verdict["milestone_verdict"] == "PARK"
    assert verdict["underpowered"] is False


def test_non_vacuity_park_verdict_planted_signal_proceeds():
    """End-to-end: the SAME planted-signal corpus feeds value_metric, its
    own real_aa_floor, AND its own gating count into park_verdict -- proves
    the full verdict pipeline (not just the metric arm) calls a planted
    improving-rank signal a proceed. SYNTHETIC data; the REAL verdict comes
    from the live-corpus run.
    """
    reinforced, used = _planted_signal_corpus(n_pairs=PAIR_COUNT_FLOOR)
    t_cutoff = _T0 + timedelta(days=1)

    vm = value_metric(reinforced, used, t_cutoff=t_cutoff, k=RANK_K)
    deltas = per_pair_deltas(reinforced, used, t_cutoff=t_cutoff, k=RANK_K)
    null_deltas = null_per_pair_deltas(reinforced, used, t_cutoff=t_cutoff, k=RANK_K)
    aa_floor = two_sided_aa_floor(null_deltas, iters=300, seed=0)
    intra_spread = pair_session_spread(reinforced, intra_event_pairs_with_session)
    gating = pairs_meeting_session_floor(
        distinct_session_distribution(intra_spread), min_sessions=MIN_DISTINCT_SESSIONS
    )

    verdict = park_verdict(
        gating_pairs_over_floor=gating,
        value_metric=vm,
        established_pairs=len(deltas),
        real_aa_floor=aa_floor,
        signalb_bigrams_over_floor=0,
    )
    assert verdict["milestone_verdict"] == "proceed"


def test_non_vacuity_park_verdict_iid_noise_parks_on_metric_arm():
    """End-to-end sibling of the planted-signal test above: the SAME
    i.i.d.-noise corpus's own gating count clears PAIR_COUNT_FLOOR (isolating
    the PARK to the metric arm, not a thin-gate artifact), while value_metric
    does not clear its own real_aa_floor -- park_verdict PARKs the milestone
    on synthetic noise with no planted relationship.
    """
    reinforced, used = _iid_noise_corpus()
    t_cutoff = _T0 + timedelta(days=1)

    vm = value_metric(reinforced, used, t_cutoff=t_cutoff, k=RANK_K)
    deltas = per_pair_deltas(reinforced, used, t_cutoff=t_cutoff, k=RANK_K)
    null_deltas = null_per_pair_deltas(reinforced, used, t_cutoff=t_cutoff, k=RANK_K)
    aa_floor = two_sided_aa_floor(null_deltas, iters=500, seed=0)
    intra_spread = pair_session_spread(reinforced, intra_event_pairs_with_session)
    gating = pairs_meeting_session_floor(
        distinct_session_distribution(intra_spread), min_sessions=MIN_DISTINCT_SESSIONS
    )
    assert gating >= PAIR_COUNT_FLOOR, "gating arm must clear its floor to isolate the metric arm"

    verdict = park_verdict(
        gating_pairs_over_floor=gating,
        value_metric=vm,
        established_pairs=len(deltas),
        real_aa_floor=aa_floor,
        signalb_bigrams_over_floor=0,
    )
    assert verdict["milestone_verdict"] == "PARK"
    assert verdict["value_metric_clears_floor"] is False


def test_park_verdict_run_census_fields_feed_it_without_re_derivation(tmp_path):
    store, _home = _fresh_store(tmp_path)
    a, b = str(uuid4()), str(uuid4())
    _emit_reinforced(store, "sess-1", [a, b])
    _emit_reinforced(store, "sess-2", [a, b])
    flush_event_buffer(store)

    report = run_census(store)
    assert "gating_pairs_over_floor" in report
    assert "signalb_bigrams_over_floor" in report
    assert "established_pairs" in report

    verdict = park_verdict(
        gating_pairs_over_floor=report["gating_pairs_over_floor"],
        value_metric=report["value_metric"],
        established_pairs=report["established_pairs"],
        real_aa_floor=report["real_corpus_aa_floor"],
        signalb_bigrams_over_floor=report["signalb_bigrams_over_floor"],
    )
    assert verdict["established_pairs"] == report["established_pairs"]
    assert verdict["gating_clears_floor"] == (
        report["gating_pairs_over_floor"] >= PAIR_COUNT_FLOOR
    )
    assert verdict["value_metric_clears_floor"] == (
        report["value_metric"] > report["real_corpus_aa_floor"][1]
    )
    store.close()


def test_park_verdict_wired_into_main_output(tmp_path, monkeypatch, capsys):
    store, home = _fresh_store(tmp_path)
    a, b = str(uuid4()), str(uuid4())
    _emit_reinforced(store, "sess-1", [a, b])
    _emit_reinforced(store, "sess-2", [a, b])
    flush_event_buffer(store)
    store.close()

    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    monkeypatch.delenv("IAI_MCP_STORE", raising=False)
    monkeypatch.setattr(recall_accuracy_real, "_operator_home", lambda: home)
    monkeypatch.setattr("sys.argv", ["proc_corpus_census.py", "--driver", "stdlib"])

    main()

    out_lines = capsys.readouterr().out.strip().splitlines()
    report = json.loads(out_lines[-2])
    assert "park_verdict" in report
    assert report["park_verdict"]["established_pairs"] == report["established_pairs"]
    assert report["park_verdict"]["underpowered"] == (report["established_pairs"] == 0)

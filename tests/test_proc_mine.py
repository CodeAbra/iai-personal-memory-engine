"""Tracer behavioral coverage for the Signal-A cofired-pair miner."""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp.events import flush_event_buffer, write_event
from iai_mcp.lilli.cycle import proc_mine
from iai_mcp.lilli.cycle.proc_mine import (
    COFIRE_MINE_SOURCE,
    MIN_DISTINCT_SESSIONS,
    PAIR_COUNT_FLOOR,
    load_cofired_events,
    mine_cofired_pairs,
)
from iai_mcp.store import MemoryStore

_BASE_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _fresh_store(tmp_path: Path) -> "tuple[MemoryStore, Path]":
    home = tmp_path / "operator-home"
    store_root = home / ".iai-mcp"
    store = MemoryStore(path=store_root)
    return store, home


def _emit_cofired(store: MemoryStore, session_id: str, hit_ids: list, used_ids: list) -> None:
    write_event(
        store,
        kind="retrieval_cofired",
        data={
            "session_id": session_id,
            "hit_ids": hit_ids,
            "used_ids": used_ids,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        severity="info",
        session_id=session_id,
        buffered=True,
    )


def _ts_seq():
    i = 0
    while True:
        yield _BASE_TS + timedelta(seconds=i)
        i += 1


def _make_event(ts_gen, session_id, used_ids, *, hit_ids=None):
    return {
        "session_id": session_id,
        "ts": next(ts_gen),
        "data": {
            "used_ids": list(used_ids),
            "hit_ids": list(hit_ids if hit_ids is not None else used_ids),
        },
    }


def _repeated_pair_ids(pair, repeats, filler_prefix):
    """(pair[0], pair[1], filler) blocks repeated `repeats` times -- the
    filler between blocks stops the reverse pair from ever forming."""
    a, b = pair
    ids = []
    for i in range(repeats):
        ids.extend([a, b, f"{filler_prefix}-{i}"])
    return ids


def _uneven_splits(total, n):
    base, rem = divmod(total, n)
    return [base + 1 if i < rem else base for i in range(n)]


def test_over_both_floors_pair_included():
    a, b = str(uuid4()), str(uuid4())
    ts_gen = _ts_seq()
    events = [
        _make_event(ts_gen, f"sess-{i}", _repeated_pair_ids((a, b), 10, f"f{i}"))
        for i in range(3)
    ]

    candidates = mine_cofired_pairs(events)
    by_pair = {c.pair: c for c in candidates}

    assert (a, b) in by_pair
    cand = by_pair[(a, b)]
    assert cand.count == 30
    assert cand.session_count == 3
    assert cand.sessions == {"sess-0", "sess-1", "sess-2"}
    assert cand.source == COFIRE_MINE_SOURCE
    assert cand.first_ts < cand.last_ts


def test_sub_count_floor_pair_excluded():
    a, b = str(uuid4()), str(uuid4())
    ts_gen = _ts_seq()
    splits = _uneven_splits(PAIR_COUNT_FLOOR - 1, MIN_DISTINCT_SESSIONS)
    events = [
        _make_event(ts_gen, f"sess-{i}", _repeated_pair_ids((a, b), n, f"f{i}"))
        for i, n in enumerate(splits)
    ]

    candidates = mine_cofired_pairs(events)
    assert (a, b) not in {c.pair for c in candidates}


def test_sub_session_floor_pair_excluded():
    a, b = str(uuid4()), str(uuid4())
    ts_gen = _ts_seq()
    events = [
        _make_event(ts_gen, "sess-only", _repeated_pair_ids((a, b), PAIR_COUNT_FLOOR + 10, "f")),
    ]

    candidates = mine_cofired_pairs(events)
    assert (a, b) not in {c.pair for c in candidates}


def test_inter_event_only_pair_never_gates():
    x, y = str(uuid4()), str(uuid4())
    ts_gen = _ts_seq()
    n_sessions = max(PAIR_COUNT_FLOOR, MIN_DISTINCT_SESSIONS)
    events = []
    for i in range(n_sessions):
        session_id = f"sess-{i}"
        events.append(_make_event(ts_gen, session_id, [f"lead-{i}", x]))
        events.append(_make_event(ts_gen, session_id, [y, f"tail-{i}"]))

    candidates = mine_cofired_pairs(events)
    assert (x, y) not in {c.pair for c in candidates}


def test_sub_floor_pair_excluded_has_teeth():
    a, b = str(uuid4()), str(uuid4())
    ts_gen = _ts_seq()
    planted_count = PAIR_COUNT_FLOOR - 1
    splits = _uneven_splits(planted_count, MIN_DISTINCT_SESSIONS)
    events = [
        _make_event(ts_gen, f"sess-{i}", _repeated_pair_ids((a, b), n, f"f{i}"))
        for i, n in enumerate(splits)
    ]

    default_candidates = mine_cofired_pairs(events)
    assert (a, b) not in {c.pair for c in default_candidates}

    control_candidates = mine_cofired_pairs(
        events, min_count=planted_count, min_distinct_sessions=MIN_DISTINCT_SESSIONS
    )
    control_by_pair = {c.pair: c for c in control_candidates}
    assert (a, b) in control_by_pair
    assert control_by_pair[(a, b)].count == planted_count


def test_sub_session_floor_pair_excluded_has_teeth():
    a, b = str(uuid4()), str(uuid4())
    ts_gen = _ts_seq()
    planted_count = PAIR_COUNT_FLOOR + 10
    events = [
        _make_event(ts_gen, "sess-only", _repeated_pair_ids((a, b), planted_count, "f")),
    ]

    default_candidates = mine_cofired_pairs(events)
    assert (a, b) not in {c.pair for c in default_candidates}

    control_candidates = mine_cofired_pairs(events, min_distinct_sessions=1)
    control_by_pair = {c.pair: c for c in control_candidates}
    assert (a, b) in control_by_pair
    assert control_by_pair[(a, b)].session_count == 1


def test_one_below_session_floor_pair_excluded_has_teeth():
    a, b = str(uuid4()), str(uuid4())
    ts_gen = _ts_seq()
    n_sessions = MIN_DISTINCT_SESSIONS - 1
    splits = _uneven_splits(PAIR_COUNT_FLOOR, n_sessions)
    events = [
        _make_event(ts_gen, f"sess-{i}", _repeated_pair_ids((a, b), n, f"f{i}"))
        for i, n in enumerate(splits)
    ]

    default_candidates = mine_cofired_pairs(events)
    assert (a, b) not in {c.pair for c in default_candidates}

    control_candidates = mine_cofired_pairs(events, min_distinct_sessions=n_sessions)
    control_by_pair = {c.pair: c for c in control_candidates}
    assert (a, b) in control_by_pair
    assert control_by_pair[(a, b)].session_count == n_sessions
    assert control_by_pair[(a, b)].count >= PAIR_COUNT_FLOOR


def test_boundary_count_and_source_discriminator():
    a, b = str(uuid4()), str(uuid4())
    ts_gen = _ts_seq()
    splits = _uneven_splits(PAIR_COUNT_FLOOR, MIN_DISTINCT_SESSIONS)

    events = []
    for i, n in enumerate(splits):
        session_id = f"sess-{i}"
        first_ids = _repeated_pair_ids((a, b), n, f"f{i}") + [a]
        second_ids = [b, f"tail-{i}"]
        events.append(_make_event(ts_gen, session_id, first_ids))
        events.append(_make_event(ts_gen, session_id, second_ids))

    candidates = mine_cofired_pairs(events)
    by_pair = {c.pair: c for c in candidates}

    assert (a, b) in by_pair
    cand = by_pair[(a, b)]
    assert cand.source == COFIRE_MINE_SOURCE
    assert cand.count == PAIR_COUNT_FLOOR
    assert cand.session_count == MIN_DISTINCT_SESSIONS
    assert cand.boundary_count == len(splits)

    x, y = str(uuid4()), str(uuid4())
    inter_only_ts_gen = _ts_seq()
    inter_only_n_sessions = max(PAIR_COUNT_FLOOR, MIN_DISTINCT_SESSIONS)
    inter_only_events = []
    for i in range(inter_only_n_sessions):
        session_id = f"inter-only-sess-{i}"
        inter_only_events.append(_make_event(inter_only_ts_gen, session_id, [f"lead-{i}", x]))
        inter_only_events.append(_make_event(inter_only_ts_gen, session_id, [y, f"tail-{i}"]))

    inter_only_candidates = mine_cofired_pairs(inter_only_events)
    assert (x, y) not in {c.pair for c in inter_only_candidates}


def test_shipped_defaults_boundary():
    a, b = str(uuid4()), str(uuid4())
    c, d = str(uuid4()), str(uuid4())
    ts_gen = _ts_seq()

    at_floor_splits = _uneven_splits(PAIR_COUNT_FLOOR, MIN_DISTINCT_SESSIONS)
    below_floor_splits = _uneven_splits(PAIR_COUNT_FLOOR - 1, MIN_DISTINCT_SESSIONS)

    events = []
    for i, n in enumerate(at_floor_splits):
        events.append(_make_event(ts_gen, f"at-sess-{i}", _repeated_pair_ids((a, b), n, f"atf{i}")))
    for i, n in enumerate(below_floor_splits):
        events.append(_make_event(ts_gen, f"below-sess-{i}", _repeated_pair_ids((c, d), n, f"belf{i}")))

    candidates = mine_cofired_pairs(events)
    pairs = {cand.pair for cand in candidates}

    assert (a, b) in pairs
    assert (c, d) not in pairs


def test_deterministic_order():
    a, b = str(uuid4()), str(uuid4())
    c, d = str(uuid4()), str(uuid4())
    ts_gen = _ts_seq()

    events = []
    for i in range(3):
        events.append(_make_event(ts_gen, f"sess-{i}", _repeated_pair_ids((a, b), 20, f"ab{i}")))
    for i in range(3):
        events.append(_make_event(ts_gen, f"sess-{i}", _repeated_pair_ids((c, d), 10, f"cd{i}")))

    first = mine_cofired_pairs(events)
    second = mine_cofired_pairs(events)

    assert first == second
    counts = [cand.count for cand in first]
    assert counts == sorted(counts, reverse=True)


def test_loader_round_trip(tmp_path):
    store, _home = _fresh_store(tmp_path)
    a, b, c = str(uuid4()), str(uuid4()), str(uuid4())
    hit_a, hit_b, hit_c = str(uuid4()), str(uuid4()), str(uuid4())

    _emit_cofired(store, "sess-first", [hit_a, hit_c], [a, c])
    _emit_cofired(store, "-", [hit_a], [a])
    _emit_cofired(store, "unknown", [hit_c], [c])
    _emit_cofired(store, "sess-second", [hit_b], [b])
    flush_event_buffer(store)

    events = load_cofired_events(store)

    assert len(events) == 2
    session_ids = [e["session_id"] for e in events]
    assert session_ids == ["sess-first", "sess-second"]
    for event in events:
        assert "hit_ids" not in event["data"]
    assert events[0]["data"]["used_ids"] == [a, c]
    assert events[1]["data"]["used_ids"] == [b]

    ts_values = [e["ts"] for e in events]
    assert ts_values == sorted(ts_values)
    store.close()


def test_loader_queries_only_retrieval_cofired(tmp_path):
    store, _home = _fresh_store(tmp_path)
    a, b = str(uuid4()), str(uuid4())

    _emit_cofired(store, "sess-real", [a], [a])
    write_event(
        store,
        kind="retrieval_used",
        data={"session_id": "sess-real", "hit_ids": [b], "query": "decoy"},
        severity="info",
        session_id="sess-real",
        buffered=True,
    )
    flush_event_buffer(store)

    events = load_cofired_events(store)

    assert len(events) == 1
    assert events[0]["data"]["used_ids"] == [a]
    store.close()


def test_output_is_hit_ids_blind():
    used_pair = (str(uuid4()), str(uuid4()))
    hit_pair_variant_a = (str(uuid4()), str(uuid4()))
    hit_pair_variant_b = (str(uuid4()), str(uuid4()))

    def _build(hit_pair):
        ts_gen = _ts_seq()
        events = []
        for i in range(MIN_DISTINCT_SESSIONS):
            session_id = f"sess-{i}"
            used_ids = _repeated_pair_ids(used_pair, PAIR_COUNT_FLOOR, f"u{i}")
            hit_ids = _repeated_pair_ids(hit_pair, PAIR_COUNT_FLOOR, f"h{i}")
            events.append(_make_event(ts_gen, session_id, used_ids, hit_ids=hit_ids))
        return events

    candidates_a = mine_cofired_pairs(_build(hit_pair_variant_a))
    candidates_b = mine_cofired_pairs(_build(hit_pair_variant_b))

    assert candidates_a == candidates_b

    pairs_a = {c.pair for c in candidates_a}
    assert used_pair in pairs_a
    assert hit_pair_variant_a not in pairs_a
    assert hit_pair_variant_b not in pairs_a


def test_mine_function_source_never_mentions_hit_ids():
    def _scan(src: str) -> bool:
        return "hit_ids" in src

    shipped_source = inspect.getsource(mine_cofired_pairs)
    assert "used_ids" in shipped_source
    assert _scan(shipped_source) is False

    def _decoy(events):
        return [event["data"]["hit_ids"] for event in events]

    decoy_source = inspect.getsource(_decoy)
    assert _scan(decoy_source) is True


def test_loader_strips_hit_ids(tmp_path):
    store, _home = _fresh_store(tmp_path)
    a, b = str(uuid4()), str(uuid4())
    hit_a, hit_b = str(uuid4()), str(uuid4())

    _emit_cofired(store, "sess-alpha", [hit_a], [a])
    _emit_cofired(store, "sess-beta", [hit_b], [b])
    flush_event_buffer(store)

    events = load_cofired_events(store)

    assert len(events) == 2
    for event in events:
        assert "hit_ids" not in event["data"]
    store.close()


def test_loader_exhausts_full_table(tmp_path, monkeypatch):
    store, _home = _fresh_store(tmp_path)
    total = 150
    for i in range(total):
        _emit_cofired(store, f"sess-{i}", [str(uuid4())], [str(uuid4())])
    flush_event_buffer(store)

    events = load_cofired_events(store)
    assert len(events) == total

    monkeypatch.setattr(proc_mine, "_EVENT_LOAD_LIMIT", 10)
    with pytest.raises(RuntimeError, match="load truncated"):
        load_cofired_events(store)
    store.close()


def test_loader_concurrent_append_does_not_false_raise(tmp_path, monkeypatch):
    store, _home = _fresh_store(tmp_path)
    for i in range(5):
        _emit_cofired(store, f"sess-{i}", [str(uuid4())], [str(uuid4())])
    flush_event_buffer(store)

    real_count = proc_mine._count_matching_events

    def _count_then_concurrent_write(*args, **kwargs):
        pre_write_count = real_count(*args, **kwargs)
        write_event(
            store,
            kind=COFIRE_MINE_SOURCE,
            data={
                "session_id": "sess-concurrent",
                "hit_ids": [],
                "used_ids": [str(uuid4())],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            severity="info",
            session_id="sess-concurrent",
            buffered=False,
        )
        return pre_write_count

    monkeypatch.setattr(proc_mine, "_count_matching_events", _count_then_concurrent_write)

    events = load_cofired_events(store)

    assert len(events) == 6
    assert {e["session_id"] for e in events} >= {"sess-0", "sess-concurrent"}
    store.close()


def test_none_data_degrades_instead_of_crashing():
    a, b = str(uuid4()), str(uuid4())
    ts_gen = _ts_seq()
    events = [_make_event(ts_gen, f"sess-{i}", _repeated_pair_ids((a, b), 25, f"f{i}")) for i in range(3)]
    events.append({"session_id": "sess-malformed", "ts": next(ts_gen), "data": None})

    candidates = mine_cofired_pairs(events)

    assert (a, b) in {c.pair for c in candidates}


def test_deterministic_on_fixed_fixture():
    a, b = str(uuid4()), str(uuid4())
    c, d = str(uuid4()), str(uuid4())
    ts_gen = _ts_seq()

    events = []
    for i in range(3):
        events.append(_make_event(ts_gen, f"sess-{i}", _repeated_pair_ids((a, b), 20, f"ab{i}")))
    for i in range(3):
        events.append(_make_event(ts_gen, f"sess-{i}", _repeated_pair_ids((c, d), 15, f"cd{i}")))

    first = mine_cofired_pairs(events)
    second = mine_cofired_pairs(events)

    assert first == second
    for cand_first, cand_second in zip(first, second):
        assert cand_first.pair == cand_second.pair
        assert cand_first.count == cand_second.count
        assert cand_first.session_count == cand_second.session_count
        assert cand_first.sessions == cand_second.sessions
        assert cand_first.first_ts == cand_second.first_ts
        assert cand_first.last_ts == cand_second.last_ts
        assert cand_first.boundary_count == cand_second.boundary_count

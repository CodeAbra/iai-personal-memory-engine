"""Tracer coverage for the Signal-B tool-call-sequence miner, driven through
step_proc_mine end to end (not the loader/miner functions called by hand --
the wiring into the sleep step is what a tracer plan proves)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from iai_mcp.events import flush_event_buffer, write_event
from iai_mcp.lifecycle_state import default_state, save_state
from iai_mcp.lilli.cycle.chunk import persist_proc_chunk
from iai_mcp.lilli.cycle.proc_mine import (
    COFIRE_MINE_SOURCE,
    MIN_DISTINCT_SESSIONS,
    PAIR_COUNT_FLOOR,
    CofirePairCandidate,
)
from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline
from iai_mcp.lilli.cycle.toolseq_mine import (
    TOOLSEQ_MINE_SOURCE,
    _ASSISTANT_WHERE,
    load_assistant_turns,
    mine_tool_ngrams,
    parse_tool_sequence,
)
from iai_mcp.store import RECORDS_TABLE, MemoryStore, flush_record_buffer
from iai_mcp.types import SCHEMA_VERSION_CURRENT, MemoryRecord

_BASE_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)

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


def _fresh_store(tmp_path: Path) -> "tuple[MemoryStore, Path]":
    home = tmp_path / "operator-home"
    store_root = home / ".iai-mcp"
    store = MemoryStore(path=store_root)
    return store, home


def _uneven_splits(total: int, n: int) -> list[int]:
    base, rem = divmod(total, n)
    return [base + 1 if i < rem else base for i in range(n)]


def _insert_assistant_turn(
    store: MemoryStore,
    session_id: str,
    tools: "tuple[str, str] | None",
    created_at: datetime,
) -> None:
    trailer = f"\n[tools: {', '.join(tools)}]" if tools else ""
    text = "the assistant finished a turn of ambient conversational work" + trailer
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
        provenance=[
            {
                "ts": created_at.isoformat(),
                "cue": "test",
                "session_id": session_id,
                "role": "assistant",
            }
        ],
        created_at=created_at,
        updated_at=created_at,
        tags=["capture", "role:assistant"],
        language="en",
        s5_trust_score=0.5,
        profile_modulation_gain={},
        schema_version=SCHEMA_VERSION_CURRENT,
    )
    store.insert(rec)


def _repeated_pair_ids(pair: "tuple[str, str]", repeats: int, filler_prefix: str) -> list[str]:
    a, b = pair
    ids: list[str] = []
    for i in range(repeats):
        ids.extend([a, b, f"{filler_prefix}-{i}"])
    return ids


def _ts_seq():
    i = 0
    while True:
        yield _BASE_TS + timedelta(seconds=i)
        i += 1


def _make_turn(ts: datetime, session_id: str, tools: "list[str] | None") -> dict:
    """Pure in-memory turn dict, the exact shape mine_tool_ngrams consumes --
    no store, mirrors test_proc_mine.py's _make_event for the pure-function
    gate/boundary tests."""
    trailer = f"\n[tools: {', '.join(tools)}]" if tools else ""
    return {
        "provenance": [{"session_id": session_id}],
        "created_at": ts,
        "literal_surface": "turn body" + trailer,
    }


def _repeated_pair_turns(
    pair: "tuple[str, str]", repeats: int, session_id: str, ts_gen,
) -> list[dict]:
    return [_make_turn(next(ts_gen), session_id, list(pair)) for _ in range(repeats)]


def _run_step(store: MemoryStore, tmp_path: Path, label: str) -> dict:
    lifecycle_path = tmp_path / f"lifecycle-{label}.json"
    save_state(default_state(), lifecycle_path)
    pipeline = SleepPipeline(store=store, lifecycle_state_path=lifecycle_path)
    done, payload = pipeline._step_proc_mine(interrupt_check=None)
    assert done is True
    return payload


# --- mint through step_proc_mine, RED variant, both drivers ----------------


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_tool_bigram_mints_procedural_chunk_through_step_proc_mine(
    tmp_path, monkeypatch, driver,
):
    _set_driver(monkeypatch, driver)
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")
    store, _home = _fresh_store(tmp_path)

    pair = ("Bash", "Agent")
    splits = _uneven_splits(PAIR_COUNT_FLOOR, MIN_DISTINCT_SESSIONS)
    ts = _BASE_TS
    for i, n in enumerate(splits):
        session_id = f"sess-{i}"
        for _j in range(n):
            _insert_assistant_turn(store, session_id, pair, ts)
            ts = ts + timedelta(seconds=1)

    # IAI_MCP_TEST_NO_AUTOFLUSH disables the conftest autoflush fixture
    # entirely, so the planted input turns need their own explicit flush to
    # become visible to load_assistant_turns' SELECT -- the opt-out is there
    # to expose persist_proc_chunk's OWN buffered-liveness window during the
    # step below, not to hide our fixture's own writes.
    flush_record_buffer(store)

    payload = _run_step(store, tmp_path, f"mint-{driver}")

    # Settle whatever the real buffered-visibility window still held so the
    # on-disk state is assertable -- the mint-vs-nothing decision already
    # happened under the real unflushed window above.
    flush_record_buffer(store)

    transitions = store.db.open_table("proc_transitions").to_pandas()
    b_rows = transitions[
        (transitions["src"] == pair[0])
        & (transitions["dst"] == pair[1])
        & (transitions["source"] == TOOLSEQ_MINE_SOURCE)
    ]
    assert len(b_rows) == 1
    chunk_id = b_rows.iloc[0]["chunk_id"]
    assert chunk_id

    records = store.db.open_table(RECORDS_TABLE).to_pandas()
    rec_rows = records[records["id"] == str(chunk_id)]
    assert len(rec_rows) == 1
    rec_row = rec_rows.iloc[0].to_dict()
    assert rec_row["tier"] == "procedural"
    tomb = rec_row.get("tombstoned_at")
    assert tomb is None or pd.isna(tomb)

    assert payload["chunks_persisted"] >= 1
    assert payload["candidates_gated"] >= 1
    store.close()


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_no_repeated_bigram_mints_zero_tool_sequence_chunks_red(
    tmp_path, monkeypatch, driver,
):
    """RED variant of the mint test above: an otherwise-identical store
    whose trailers never repeat the same bigram twice mints nothing for
    Signal B -- proving the mint above is caused by the planted repeats,
    not by inserting assistant turns at all."""
    _set_driver(monkeypatch, driver)
    store, _home = _fresh_store(tmp_path)

    splits = _uneven_splits(PAIR_COUNT_FLOOR, MIN_DISTINCT_SESSIONS)
    ts = _BASE_TS
    for i, n in enumerate(splits):
        session_id = f"sess-{i}"
        for j in range(n):
            unique_pair = (f"Tool{i}-{j}-A", f"Tool{i}-{j}-B")
            _insert_assistant_turn(store, session_id, unique_pair, ts)
            ts = ts + timedelta(seconds=1)

    payload = _run_step(store, tmp_path, f"red-{driver}")

    transitions = store.db.open_table("proc_transitions").to_pandas()
    b_rows = transitions[transitions["source"] == TOOLSEQ_MINE_SOURCE]
    assert len(b_rows) == 0
    assert payload["chunks_persisted"] == 0
    store.close()


def test_return_counts_reflect_both_producers_not_a_alone(tmp_path, monkeypatch):
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    store, _home = _fresh_store(tmp_path)

    a_id1, a_id2 = str(uuid4()), str(uuid4())
    a_splits = _uneven_splits(PAIR_COUNT_FLOOR, MIN_DISTINCT_SESSIONS)
    for i, n in enumerate(a_splits):
        session_id = f"a-sess-{i}"
        write_event(
            store,
            kind="retrieval_cofired",
            data={
                "session_id": session_id,
                "hit_ids": [],
                "used_ids": _repeated_pair_ids((a_id1, a_id2), n, f"af{i}"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            severity="info",
            session_id=session_id,
            buffered=True,
        )
    flush_event_buffer(store)

    b_pair = ("Bash", "Agent")
    b_splits = _uneven_splits(PAIR_COUNT_FLOOR, MIN_DISTINCT_SESSIONS)
    ts = _BASE_TS
    for i, n in enumerate(b_splits):
        session_id = f"b-sess-{i}"
        for _j in range(n):
            _insert_assistant_turn(store, session_id, b_pair, ts)
            ts = ts + timedelta(seconds=1)

    payload = _run_step(store, tmp_path, "both-producers")

    # Tight equalities, not >= floors: this fixture plants exactly one
    # gating Signal-A pair (a_id1, a_id2) and exactly one gating Signal-B
    # bigram (Bash, Agent) -- a payload of (2, 2) with Signal A contributing
    # both would pass a >= 2 assertion while still proving Signal B mined
    # nothing, which is exactly what this test's name claims to rule out.
    assert payload["candidates_gated"] == 2
    assert payload["chunks_persisted"] == 2

    sources = set(store.db.open_table("proc_transitions").to_pandas()["source"])
    assert {COFIRE_MINE_SOURCE, TOOLSEQ_MINE_SOURCE} <= sources
    store.close()


def test_tombstoned_assistant_turns_never_feed_the_miner(tmp_path, monkeypatch):
    """A retired assistant turn must not keep minting durable procedural
    state -- the census's WHERE clause this loader promotes has no
    tombstone filter because the census only reports on a read-only copy;
    this loader mints, so a tombstoned source record must be excluded."""
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    store, _home = _fresh_store(tmp_path)

    pair = ("Bash", "Agent")
    splits = _uneven_splits(PAIR_COUNT_FLOOR, MIN_DISTINCT_SESSIONS)
    ts = _BASE_TS
    for i, n in enumerate(splits):
        session_id = f"sess-{i}"
        for _j in range(n):
            _insert_assistant_turn(store, session_id, pair, ts)
            ts = ts + timedelta(seconds=1)

    tbl = store.db.open_table(RECORDS_TABLE)
    tbl.update(
        where="tags_json LIKE '%role:assistant%'",
        values={"tombstoned_at": datetime.now(timezone.utc).isoformat()},
    )

    payload = _run_step(store, tmp_path, "tombstoned-excluded")

    transitions = store.db.open_table("proc_transitions").to_pandas()
    b_rows = transitions[transitions["source"] == TOOLSEQ_MINE_SOURCE]
    assert len(b_rows) == 0
    assert payload["chunks_persisted"] == 0
    store.close()


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_unparseable_created_at_turn_excluded_signal_a_unaffected(
    tmp_path, monkeypatch, driver,
):
    """An assistant turn whose stored created_at is not a parseable
    timestamp must not crash step_proc_mine -- the turn is excluded from
    Signal-B's sequence grouping (it cannot be ordered), and Signal A's own
    candidates still mint in the same step."""
    _set_driver(monkeypatch, driver)
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")
    store, _home = _fresh_store(tmp_path)

    a_id1, a_id2 = str(uuid4()), str(uuid4())
    a_splits = _uneven_splits(PAIR_COUNT_FLOOR, MIN_DISTINCT_SESSIONS)
    for i, n in enumerate(a_splits):
        session_id = f"a-sess-{i}"
        write_event(
            store,
            kind="retrieval_cofired",
            data={
                "session_id": session_id,
                "hit_ids": [],
                "used_ids": _repeated_pair_ids((a_id1, a_id2), n, f"af{i}"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            severity="info",
            session_id=session_id,
            buffered=True,
        )
    flush_event_buffer(store)

    _insert_assistant_turn(store, "sess-corrupt", ("Bash", "Agent"), _BASE_TS)
    flush_record_buffer(store)

    tbl = store.db.open_table(RECORDS_TABLE)
    tbl.update(
        where="tags_json LIKE '%role:assistant%'",
        values={"created_at": "not-a-real-timestamp"},
    )

    payload = _run_step(store, tmp_path, f"unparseable-created-at-{driver}")

    assert payload["chunks_persisted"] >= 1

    transitions = store.db.open_table("proc_transitions").to_pandas()
    b_rows = transitions[transitions["source"] == TOOLSEQ_MINE_SOURCE]
    assert len(b_rows) == 0
    store.close()


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
@pytest.mark.parametrize(
    "malformed_provenance_json",
    [
        pytest.param('["not", "a", "dict"]', id="list-of-strings"),
        pytest.param('{"session_id": "sess-corrupt"}', id="bare-dict"),
    ],
)
def test_malformed_provenance_shape_turn_excluded_signal_a_unaffected(
    tmp_path, monkeypatch, driver, malformed_provenance_json,
):
    """An assistant turn whose stored provenance decodes to a shape that is
    not a list-of-dicts (a list of strings, or a bare top-level dict) must
    not crash step_proc_mine -- the turn is excluded from Signal-B's sequence
    grouping (its session_id cannot be read), and Signal A's own candidates
    still mint in the same step."""
    _set_driver(monkeypatch, driver)
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")
    store, _home = _fresh_store(tmp_path)

    a_id1, a_id2 = str(uuid4()), str(uuid4())
    a_splits = _uneven_splits(PAIR_COUNT_FLOOR, MIN_DISTINCT_SESSIONS)
    for i, n in enumerate(a_splits):
        session_id = f"a-sess-{i}"
        write_event(
            store,
            kind="retrieval_cofired",
            data={
                "session_id": session_id,
                "hit_ids": [],
                "used_ids": _repeated_pair_ids((a_id1, a_id2), n, f"af{i}"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            severity="info",
            session_id=session_id,
            buffered=True,
        )
    flush_event_buffer(store)

    _insert_assistant_turn(store, "sess-corrupt", ("Bash", "Agent"), _BASE_TS)
    flush_record_buffer(store)

    tbl = store.db.open_table(RECORDS_TABLE)
    rows = tbl.to_pandas()
    turn_id = rows[rows["tags_json"].str.contains("role:assistant")]["id"].iloc[0]
    # provenance_json is an encrypted column -- updates require an id-keyed
    # WHERE clause for AAD binding, unlike the plaintext created_at column.
    tbl.update(
        where=f"id = '{turn_id}'",
        values={"provenance_json": malformed_provenance_json},
    )

    payload = _run_step(store, tmp_path, f"malformed-provenance-{driver}")

    assert payload["chunks_persisted"] >= 1

    transitions = store.db.open_table("proc_transitions").to_pandas()
    b_rows = transitions[transitions["source"] == TOOLSEQ_MINE_SOURCE]
    assert len(b_rows) == 0
    store.close()


# --- count + distinct-session gate; boundary pass ---------------------------
# Pure-function tests against mine_tool_ngrams directly, mirroring
# test_proc_mine.py's own gate/boundary test shapes for mine_cofired_pairs.


def test_over_both_floors_bigram_included():
    a, b = "ToolA", "ToolB"
    ts_gen = _ts_seq()
    turns: list[dict] = []
    for i in range(3):
        turns += _repeated_pair_turns((a, b), 10, f"sess-{i}", ts_gen)

    candidates = mine_tool_ngrams(turns)
    by_pair = {c.pair: c for c in candidates}

    assert (a, b) in by_pair
    cand = by_pair[(a, b)]
    assert cand.count == 30
    assert cand.session_count == 3
    assert cand.sessions == {"sess-0", "sess-1", "sess-2"}
    assert cand.source == TOOLSEQ_MINE_SOURCE
    assert cand.first_ts < cand.last_ts


def test_sub_count_floor_bigram_excluded():
    a, b = "ToolA", "ToolB"
    ts_gen = _ts_seq()
    splits = _uneven_splits(PAIR_COUNT_FLOOR - 1, MIN_DISTINCT_SESSIONS)
    turns: list[dict] = []
    for i, n in enumerate(splits):
        turns += _repeated_pair_turns((a, b), n, f"sess-{i}", ts_gen)

    candidates = mine_tool_ngrams(turns)
    assert (a, b) not in {c.pair for c in candidates}


def test_sub_session_floor_bigram_excluded():
    a, b = "ToolA", "ToolB"
    ts_gen = _ts_seq()
    turns = _repeated_pair_turns((a, b), PAIR_COUNT_FLOOR + 10, "sess-only", ts_gen)

    candidates = mine_tool_ngrams(turns)
    assert (a, b) not in {c.pair for c in candidates}


def test_at_both_floors_bigram_mints_exactly_one():
    a, b = "ToolA", "ToolB"
    ts_gen = _ts_seq()
    splits = _uneven_splits(PAIR_COUNT_FLOOR, MIN_DISTINCT_SESSIONS)
    turns: list[dict] = []
    for i, n in enumerate(splits):
        turns += _repeated_pair_turns((a, b), n, f"sess-{i}", ts_gen)

    candidates = mine_tool_ngrams(turns)
    matches = [c for c in candidates if c.pair == (a, b)]
    assert len(matches) == 1
    assert matches[0].count == PAIR_COUNT_FLOOR
    assert matches[0].session_count == MIN_DISTINCT_SESSIONS


def test_boundary_only_bigram_never_mints():
    """A pair eligible ONLY via cross-turn boundary occurrences (never
    co-occurring inside a single turn) must never mint -- mirrors
    test_proc_mine.py::test_inter_event_only_pair_never_gates."""
    x, y = "ToolX", "ToolY"
    ts_gen = _ts_seq()
    n_sessions = max(PAIR_COUNT_FLOOR, MIN_DISTINCT_SESSIONS)
    turns: list[dict] = []
    for i in range(n_sessions):
        session_id = f"inter-only-sess-{i}"
        turns.append(_make_turn(next(ts_gen), session_id, [f"lead-{i}", x]))
        turns.append(_make_turn(next(ts_gen), session_id, [y, f"tail-{i}"]))

    candidates = mine_tool_ngrams(turns)
    assert (x, y) not in {c.pair for c in candidates}


def test_boundary_count_populated_on_eligible_pair_never_widens_gate():
    """The SAME pair minted via intra-turn occurrences additionally gains
    boundary_count from cross-turn adjacency, but count/session_count stay
    exactly at the intra-turn values -- mirrors
    test_proc_mine.py::test_boundary_count_and_source_discriminator."""
    a, b = "Bash", "Agent"
    ts_gen = _ts_seq()
    splits = _uneven_splits(PAIR_COUNT_FLOOR, MIN_DISTINCT_SESSIONS)
    turns: list[dict] = []
    for i, n in enumerate(splits):
        session_id = f"sess-{i}"
        turns += _repeated_pair_turns((a, b), n, session_id, ts_gen)
        # One boundary occurrence per session: last tool of this turn ("a")
        # -> first tool of the next turn ("b") forms an extra (a, b) pair
        # that must land in boundary_count, never in count.
        turns.append(_make_turn(next(ts_gen), session_id, [f"lead-{i}", a]))
        turns.append(_make_turn(next(ts_gen), session_id, [b, f"tail-{i}"]))

    candidates = mine_tool_ngrams(turns)
    by_pair = {c.pair: c for c in candidates}

    assert (a, b) in by_pair
    cand = by_pair[(a, b)]
    assert cand.source == TOOLSEQ_MINE_SOURCE
    assert cand.count == PAIR_COUNT_FLOOR
    assert cand.session_count == MIN_DISTINCT_SESSIONS
    assert cand.boundary_count == len(splits)


def test_deterministic_gate_order():
    a, b = "ToolA", "ToolB"
    c, d = "ToolC", "ToolD"
    ts_gen = _ts_seq()

    turns: list[dict] = []
    for i in range(3):
        turns += _repeated_pair_turns((a, b), 20, f"sess-{i}", ts_gen)
    for i in range(3):
        turns += _repeated_pair_turns((c, d), 10, f"sess-{i}", ts_gen)

    first = mine_tool_ngrams(turns)
    second = mine_tool_ngrams(turns)

    assert first == second
    counts = [cand.count for cand in first]
    assert counts == sorted(counts, reverse=True)


# --- source non-collision, both-driver parity -------------------------------


def test_source_discriminator_no_collision_with_signal_a(tmp_path):
    store, _home = _fresh_store(tmp_path)

    a_candidate = CofirePairCandidate(
        pair=("shared-src", "shared-dst"),
        source=COFIRE_MINE_SOURCE,
        count=PAIR_COUNT_FLOOR,
        session_count=MIN_DISTINCT_SESSIONS,
        sessions=frozenset({"s1", "s2", "s3"}),
        first_ts=_BASE_TS,
        last_ts=_BASE_TS + timedelta(minutes=5),
    )
    b_candidate = CofirePairCandidate(
        pair=("shared-src", "shared-dst"),
        source=TOOLSEQ_MINE_SOURCE,
        count=PAIR_COUNT_FLOOR,
        session_count=MIN_DISTINCT_SESSIONS,
        sessions=frozenset({"s1", "s2", "s3"}),
        first_ts=_BASE_TS,
        last_ts=_BASE_TS + timedelta(minutes=5),
    )

    a_id = persist_proc_chunk(store, a_candidate)
    b_id = persist_proc_chunk(store, b_candidate)

    assert a_id is not None
    assert b_id is not None
    assert a_id != b_id

    rows = store.db.open_table("proc_transitions").to_pandas()
    matching = rows[(rows["src"] == "shared-src") & (rows["dst"] == "shared-dst")]
    assert len(matching) == 2
    assert set(matching["source"]) == {COFIRE_MINE_SOURCE, TOOLSEQ_MINE_SOURCE}
    store.close()


# --- provenance-grouping edge cases ------------------------------------------


def test_missing_trailer_returns_empty_contributes_no_bigram():
    assert parse_tool_sequence("just plain text, no trailer here at all") == []
    turn = _make_turn(_BASE_TS, "sess-1", None)
    assert mine_tool_ngrams([turn]) == []


def test_single_tool_turn_yields_no_bigram():
    turn = _make_turn(_BASE_TS, "sess-1", ["OnlyOneTool"])
    assert parse_tool_sequence(turn["literal_surface"]) == ["OnlyOneTool"]
    assert mine_tool_ngrams([turn]) == []


def test_mid_text_decoy_rejected_by_trailing_anchor():
    surface = "look at this [tools: fake, decoy] shape mid-sentence, not the real one"
    assert parse_tool_sequence(surface) == []
    turn = {
        "provenance": [{"session_id": "sess-1"}],
        "created_at": _BASE_TS,
        "literal_surface": surface,
    }
    assert mine_tool_ngrams([turn]) == []


def test_overflow_trailer_strips_suffix_keeps_eight_named_tools():
    """Grounded in capture.py::_tools_trailer's real overflow shape: up to
    8 named tools, ` +N` for the count dropped past 8."""
    names = [f"tool{i}" for i in range(8)]
    surface = f"turn text\n[tools: {', '.join(names)} +5]"
    assert parse_tool_sequence(surface) == names

    turn = {
        "provenance": [{"session_id": "sess-1"}],
        "created_at": _BASE_TS,
        "literal_surface": surface,
    }
    # The 8-name prefix still yields real bigrams; overflow strip does not
    # block mining, it only stops the miner from asserting completeness.
    candidates = mine_tool_ngrams([turn], min_count=1, min_distinct_sessions=1)
    pairs = {c.pair for c in candidates}
    assert (names[0], names[1]) in pairs
    assert (names[6], names[7]) in pairs


def test_double_captured_turn_deduped_counts_once_through_step_proc_mine(
    tmp_path, monkeypatch,
):
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")
    store, _home = _fresh_store(tmp_path)

    pair = ("Bash", "Agent")
    splits = _uneven_splits(PAIR_COUNT_FLOOR, MIN_DISTINCT_SESSIONS)
    ts = _BASE_TS
    for i, n in enumerate(splits):
        session_id = f"sess-{i}"
        for _j in range(n):
            # Same (session_id, created_at) captured twice -- a retry/replay
            # scenario. Must count as ONE turn, never two.
            _insert_assistant_turn(store, session_id, pair, ts)
            _insert_assistant_turn(store, session_id, pair, ts)
            ts = ts + timedelta(seconds=1)

    flush_record_buffer(store)
    payload = _run_step(store, tmp_path, "double-capture")
    flush_record_buffer(store)

    transitions = store.db.open_table("proc_transitions").to_pandas()
    b_rows = transitions[
        (transitions["src"] == pair[0])
        & (transitions["dst"] == pair[1])
        & (transitions["source"] == TOOLSEQ_MINE_SOURCE)
    ]
    assert len(b_rows) == 1
    assert int(b_rows.iloc[0]["count"]) == PAIR_COUNT_FLOOR
    assert payload["chunks_persisted"] >= 1
    store.close()


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_cross_session_discrimination_below_then_at_session_floor(
    tmp_path, monkeypatch, driver,
):
    """The same bigram, at a count that already clears PAIR_COUNT_FLOOR,
    does not mint while its distinct-session spread is below
    MIN_DISTINCT_SESSIONS, and mints exactly once once the spread reaches
    it -- both drivers, column-read tombstone-lens, never get()==None."""
    _set_driver(monkeypatch, driver)
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")
    store, _home = _fresh_store(tmp_path)

    pair = ("Read", "Write")
    below_sessions = MIN_DISTINCT_SESSIONS - 1
    below_splits = _uneven_splits(PAIR_COUNT_FLOOR, below_sessions)
    ts = _BASE_TS
    for i, n in enumerate(below_splits):
        session_id = f"below-sess-{i}"
        for _j in range(n):
            _insert_assistant_turn(store, session_id, pair, ts)
            ts = ts + timedelta(seconds=1)
    flush_record_buffer(store)

    payload_below = _run_step(store, tmp_path, f"cross-session-below-{driver}")
    flush_record_buffer(store)

    transitions = store.db.open_table("proc_transitions").to_pandas()
    b_rows = transitions[
        (transitions["src"] == pair[0])
        & (transitions["dst"] == pair[1])
        & (transitions["source"] == TOOLSEQ_MINE_SOURCE)
    ]
    assert len(b_rows) == 0
    assert payload_below["chunks_persisted"] == 0

    # One more session's worth clears MIN_DISTINCT_SESSIONS -- the count
    # floor was already cleared above, isolating this to the session gate.
    session_id = f"below-sess-{below_sessions}"
    for _j in range(PAIR_COUNT_FLOOR):
        _insert_assistant_turn(store, session_id, pair, ts)
        ts = ts + timedelta(seconds=1)
    flush_record_buffer(store)

    payload_at = _run_step(store, tmp_path, f"cross-session-at-{driver}")
    flush_record_buffer(store)

    transitions = store.db.open_table("proc_transitions").to_pandas()
    b_rows = transitions[
        (transitions["src"] == pair[0])
        & (transitions["dst"] == pair[1])
        & (transitions["source"] == TOOLSEQ_MINE_SOURCE)
    ]
    assert len(b_rows) == 1
    assert payload_at["chunks_persisted"] >= 1

    chunk_id = b_rows.iloc[0]["chunk_id"]
    records = store.db.open_table(RECORDS_TABLE).to_pandas()
    rec_rows = records[records["id"] == str(chunk_id)]
    assert len(rec_rows) == 1
    tomb = rec_rows.iloc[0].to_dict().get("tombstoned_at")
    assert tomb is None or pd.isna(tomb)
    store.close()


# --- keyset-paginated load_assistant_turns hardening ------------------------
# load_assistant_turns reads through store.iter_record_columns (keyset
# pagination over the primary key) rather than store.iter_records. These
# tests prove the paginated stream matches iter_records' output shape and
# that pagination survives a multi-page corpus.


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_load_assistant_turns_paginated_matches_full_materialize_equivalence(
    tmp_path, monkeypatch, driver,
):
    _set_driver(monkeypatch, driver)
    store, _home = _fresh_store(tmp_path)

    pair = ("Bash", "Agent")
    ts = _BASE_TS
    for i in range(12):
        session_id = f"sess-{i % 4}"
        _insert_assistant_turn(store, session_id, pair, ts)
        ts = ts + timedelta(seconds=1)
    # An untrailered assistant turn -- proves parity holds for that shape too.
    _insert_assistant_turn(store, "sess-decoy", None, ts)
    flush_record_buffer(store)

    paginated = load_assistant_turns(store)

    reference: list[dict] = []
    for record in store.iter_records(where=_ASSISTANT_WHERE):
        reference.append(
            {
                "provenance": record.provenance,
                "created_at": record.created_at,
                "literal_surface": record.literal_surface,
            }
        )

    def _shape(turns: list[dict]) -> set:
        out = set()
        for t in turns:
            provenance = t.get("provenance") or []
            session_id = provenance[0].get("session_id", "-") if provenance else "-"
            out.add((session_id, t["created_at"].isoformat(), t["literal_surface"]))
        return out

    assert len(paginated) == len(reference) == 13
    assert _shape(paginated) == _shape(reference)
    store.close()


def test_load_assistant_turns_paginated_exhausts_multi_page_corpus(tmp_path, monkeypatch):
    """iter_record_columns's default batch_size is 1024 -- plant enough
    turns to force at least two keyset pages and prove no page boundary
    drops or duplicates a row."""
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    store, _home = _fresh_store(tmp_path)

    total = 1100
    ts = _BASE_TS
    for i in range(total):
        _insert_assistant_turn(store, f"sess-{i % 5}", ("Bash", "Agent"), ts)
        ts = ts + timedelta(seconds=1)
    flush_record_buffer(store)

    turns = load_assistant_turns(store)
    assert len(turns) == total
    store.close()

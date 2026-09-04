"""Live-hook wiring for the ambient co-firing tracer.

Targets the parser at the shape real traffic actually uses (str_hits_json,
149x in this repo's corpus; the list_hits_used_unused shape occurs 0x) and
proves the trigger stays additively inert on the sensitive per-turn capture
path: byte-identical episodic output and turnstate cursor with the trigger
off vs on.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from iai_mcp import cofire
from iai_mcp.cofire import HANDLED_SHAPES, classify_tool_result_shape


class _FrozenDateTime(datetime):
    """Pins the episodic header's wall-clock `deferred_at` field so two
    otherwise-identical capture calls a few milliseconds apart produce
    byte-identical output -- real time drift is not a co-fire behavior."""

    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 1, 1, tzinfo=tz or timezone.utc)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "cofire"
STR_HITS_JSON_PATH = FIXTURES_DIR / "str_hits_json.jsonl"
STRING_ERROR_PATH = FIXTURES_DIR / "string_error.jsonl"
STRADDLE_PATH = FIXTURES_DIR / "straddle_two_window.jsonl"
OBSERVED_SHAPES_PATH = FIXTURES_DIR / "observed_shapes.json"

HIT_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
HIT_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
HIT_C = "cccccccc-cccc-cccc-cccc-cccccccccccc"

HIT_D = "dddddddd-dddd-dddd-dddd-dddddddddddd"
HIT_E = "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"
HIT_F = "ffffffff-ffff-ffff-ffff-ffffffffffff"

STRADDLE_SESSION = "cofire-straddle-session"


def _build_args(session_id: str, transcript_path: Path, max_turns: int = 200) -> argparse.Namespace:
    return argparse.Namespace(
        session_id=session_id,
        transcript_path=str(transcript_path),
        max_turns_per_call=max_turns,
    )


def _read_turnstate(tmp_home: Path, session_id: str) -> dict:
    p = tmp_home / ".iai-mcp" / ".capture-state" / f"{session_id}.turnstate.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _cofire_spool_path(tmp_home: Path, session_id: str) -> Path:
    return tmp_home / ".iai-mcp" / ".cofire-spool" / f"{session_id}.cofire.jsonl"


def _load_fixture_objs(path: Path) -> "list[dict]":
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_str_hits_json_is_the_primary_extraction_path():
    objs = _load_fixture_objs(STR_HITS_JSON_PATH)
    pairs = cofire.extract_recall_pairs(objs)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["hit_ids"] == [HIT_A, HIT_B, HIT_C]

    used_ids = cofire.compute_used_ids(
        pair["hit_ids"], pair["hit_surfaces"], pair["assistant_text"]
    )
    assert used_ids, "str_hits_json fixture must yield non-empty used_ids"
    assert used_ids == [HIT_B, HIT_A]
    assert used_ids != pair["hit_ids"], "first-mention order must not equal rank order"
    assert HIT_C not in used_ids, "returned-but-unused hit must be excluded"


def test_string_error_shape_classified_and_skipped():
    objs = _load_fixture_objs(STRING_ERROR_PATH)
    for obj in objs:
        content = obj.get("message", {}).get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                assert classify_tool_result_shape(block["content"]) == "str_error"
    pairs = cofire.extract_recall_pairs(objs)
    assert pairs == []


def test_unknown_shape_is_a_logged_no_op_never_raises():
    assert classify_tool_result_shape(12345) not in HANDLED_SHAPES
    assert classify_tool_result_shape(None) not in HANDLED_SHAPES
    # Never raises, no hits.
    objs = [{
        "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_x", "content": {"weird": "shape"}},
        ]},
    }]
    pairs = cofire.extract_recall_pairs(objs)
    assert pairs == []


def test_malformed_str_hits_json_degrades_to_no_hits_never_raises():
    objs = [
        {"message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_bad", "name": "mcp__iai-mcp__memory_recall", "input": {}},
        ]}},
        {"message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_bad", "content": "{not valid json"},
        ]}},
    ]
    pairs = cofire.extract_recall_pairs(objs)
    assert pairs == []


def test_handled_shapes_covers_every_observed_shape():
    observed = json.loads(OBSERVED_SHAPES_PATH.read_text(encoding="utf-8"))
    unhandled = set(observed) - HANDLED_SHAPES
    assert not unhandled, f"observed shape(s) not dispositioned: {sorted(unhandled)}"


def test_handled_shapes_str_hits_json_genuinely_extracted_not_fail_safe_masked():
    # A newly-observed shape that adds str_hits_json to HANDLED_SHAPES without
    # real extraction would still pass the coverage gate above -- this pins
    # non-vacuous coverage on the dominant real-traffic shape.
    objs = _load_fixture_objs(STR_HITS_JSON_PATH)
    pairs = cofire.extract_recall_pairs(objs)
    assert pairs and pairs[0]["hit_ids"], "str_hits_json must be genuinely extracted"


def test_cofire_off_kill_switch_skips_the_block_entirely(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_COFIRE_OFF", "1")
    from iai_mcp.cli import cmd_capture_turn_deferred

    transcript = tmp_path / "t.jsonl"
    transcript.write_text(STR_HITS_JSON_PATH.read_text(encoding="utf-8"))

    rc = cmd_capture_turn_deferred(_build_args("cofire-off-session", transcript))
    assert rc == 0

    snap = _read_turnstate(tmp_path, "cofire-off-session")
    assert "pending_recall" not in snap
    assert not (tmp_path / ".iai-mcp" / ".cofire-spool").exists()


def test_cofire_on_resolves_a_same_window_recall_into_the_spool(tmp_path, monkeypatch):
    monkeypatch.delenv("IAI_MCP_COFIRE_OFF", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.cli import cmd_capture_turn_deferred

    transcript = tmp_path / "t.jsonl"
    transcript.write_text(STR_HITS_JSON_PATH.read_text(encoding="utf-8"))

    rc = cmd_capture_turn_deferred(_build_args("cofire-on-session", transcript))
    assert rc == 0

    spool_path = _cofire_spool_path(tmp_path, "cofire-on-session")
    assert spool_path.exists(), "same-window recall must resolve into the spool immediately"
    record = json.loads(spool_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["hit_ids"] == [HIT_A, HIT_B, HIT_C]
    assert record["used_ids"] == [HIT_B, HIT_A]

    snap = _read_turnstate(tmp_path, "cofire-on-session")
    assert snap.get("pending_recall") in (None, {}), "resolved-in-window recall leaves nothing pending"


def test_carry_forward_across_hook_windows(tmp_path, monkeypatch):
    monkeypatch.delenv("IAI_MCP_COFIRE_OFF", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.cli import cmd_capture_turn_deferred

    all_lines = STRADDLE_PATH.read_text(encoding="utf-8").splitlines(keepends=True)
    transcript = tmp_path / "t.jsonl"

    # Window 1: tool_use + tool_result only -- the assistant text response has
    # not arrived yet, so the recall must carry forward, not resolve or drop.
    transcript.write_text("".join(all_lines[:2]))
    rc1 = cmd_capture_turn_deferred(_build_args(STRADDLE_SESSION, transcript))
    assert rc1 == 0

    spool_path = _cofire_spool_path(tmp_path, STRADDLE_SESSION)
    assert not spool_path.exists(), "window 1 alone must not resolve the recall"

    snap1 = _read_turnstate(tmp_path, STRADDLE_SESSION)
    pending = snap1.get("pending_recall")
    assert pending, "an unresolved recall must be carried into turnstate.json"
    assert pending["hit_ids"] == [HIT_D, HIT_E]

    # Window 2: the assistant's text response arrives, closed by the next
    # user turn (the one whose UserPromptSubmit fired this hook call).
    with transcript.open("a", encoding="utf-8") as fh:
        fh.write("".join(all_lines[2:]))
    rc2 = cmd_capture_turn_deferred(_build_args(STRADDLE_SESSION, transcript))
    assert rc2 == 0

    assert spool_path.exists(), "the carried recall must resolve once its text arrives"
    record = json.loads(spool_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["hit_ids"] == [HIT_D, HIT_E]
    assert record["used_ids"] == [HIT_E, HIT_D], "first-mention order carries through the straddle"

    snap2 = _read_turnstate(tmp_path, STRADDLE_SESSION)
    assert snap2.get("pending_recall") in (None, {})

    from iai_mcp.store import MemoryStore
    from iai_mcp.events import flush_event_buffer, query_events

    store = MemoryStore(path=tmp_path / "store")
    counts = cofire.drain_cofire_spool(store)
    assert counts["events"] == 1
    flush_event_buffer(store)
    events = query_events(store, kind="retrieval_cofired")
    assert len(events) == 1
    assert events[0]["data"]["used_ids"] == [HIT_E, HIT_D]


def test_drain_capture_backlog_also_drains_the_cofire_spool(tmp_path, monkeypatch):
    monkeypatch.delenv("IAI_MCP_COFIRE_OFF", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp import capture
    from iai_mcp.cli import cmd_capture_turn_deferred
    from iai_mcp.store import MemoryStore

    transcript = tmp_path / "t.jsonl"
    transcript.write_text(STR_HITS_JSON_PATH.read_text(encoding="utf-8"))
    cmd_capture_turn_deferred(_build_args("cofire-drain-session", transcript))

    spool_path = _cofire_spool_path(tmp_path, "cofire-drain-session")
    assert spool_path.exists()

    store = MemoryStore(path=tmp_path / "store")
    counts = capture.drain_capture_backlog(store)
    assert any(k.startswith("cofire_") for k in counts), f"expected a cofire_* key, got {counts}"
    assert not spool_path.exists(), "backlog drain must consume the cofire spool file"


def _run_capture_turn_deferred(
    monkeypatch: pytest.MonkeyPatch, home: Path, session_id: str, cofire_off: bool,
) -> "tuple[int, Path]":
    from iai_mcp import capture as _capture_mod
    from iai_mcp.cli import cmd_capture_turn_deferred

    monkeypatch.setattr(_capture_mod, "datetime", _FrozenDateTime)
    monkeypatch.setenv("HOME", str(home))
    if cofire_off:
        monkeypatch.setenv("IAI_MCP_COFIRE_OFF", "1")
    else:
        monkeypatch.delenv("IAI_MCP_COFIRE_OFF", raising=False)

    transcript = home / "t.jsonl"
    transcript.write_text(STR_HITS_JSON_PATH.read_text(encoding="utf-8"))
    rc = cmd_capture_turn_deferred(_build_args(session_id, transcript))
    return rc, transcript


def test_off_vs_on_byte_identical_episodic_output_and_turnstate(tmp_path, monkeypatch):
    session_id = "cofire-byte-identity-session"
    off_root = tmp_path / "off"
    on_root = tmp_path / "on"
    off_root.mkdir()
    on_root.mkdir()

    rc_off, _ = _run_capture_turn_deferred(monkeypatch, off_root, session_id, cofire_off=True)
    rc_on, _ = _run_capture_turn_deferred(monkeypatch, on_root, session_id, cofire_off=False)
    assert rc_off == 0
    assert rc_on == 0

    live_off = off_root / ".iai-mcp" / ".deferred-captures" / f"{session_id}.live.jsonl"
    live_on = on_root / ".iai-mcp" / ".deferred-captures" / f"{session_id}.live.jsonl"
    episodic_off = live_off.read_bytes()
    episodic_on = live_on.read_bytes()
    assert episodic_off, "non-vacuity: the OFF run must actually write episodic output"
    assert episodic_off == episodic_on, "episodic .deferred-captures output must be byte-identical OFF vs ON"
    assert b"used_ids" not in episodic_on
    assert b"retrieval_cofired" not in episodic_on

    snap_off = _read_turnstate(off_root, session_id)
    snap_on = _read_turnstate(on_root, session_id)
    for key in ("offset", "pending", "fp"):
        assert key in snap_off, f"non-vacuity: {key} must be present in the OFF turnstate snapshot"
        assert key in snap_on, f"non-vacuity: {key} must be present in the ON turnstate snapshot"
        assert snap_off[key] == snap_on[key], f"turnstate.{key} diverged between OFF and ON"
    assert "pending_recall" not in snap_off, "the OFF run must never write the additive key"
    # This fixture resolves within one window -- nothing straddles, so ON also
    # carries nothing forward; pending_recall stays absent/empty either way.
    assert snap_on.get("pending_recall") in (None, {})

    off_spool_dir = off_root / ".iai-mcp" / ".cofire-spool"
    on_spool_path = on_root / ".iai-mcp" / ".cofire-spool" / f"{session_id}.cofire.jsonl"
    assert not off_spool_dir.exists(), "the OFF run must never create the co-fire spool"
    assert on_spool_path.exists(), "non-vacuity: the ON run must actually populate the co-fire spool"


def test_partial_batch_failure_does_not_resurrect_a_spooled_pending_recall(tmp_path, monkeypatch):
    """A carried recall resolved and durably spooled in this call must never
    be restored into turnstate.json by a later pair's spool-write failure
    within the same batch -- restoring it fires a second, mis-attributed
    retrieval_cofired event against an unrelated next-window transcript."""
    monkeypatch.delenv("IAI_MCP_COFIRE_OFF", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.cli import cmd_capture_turn_deferred
    import iai_mcp.cofire as cofire_mod

    session_id = "cofire-cr01-session"
    transcript = tmp_path / "t.jsonl"

    straddle_lines = STRADDLE_PATH.read_text(encoding="utf-8").splitlines(keepends=True)

    # Window 1: tool_use + tool_result only -- no text yet, so D/E carry
    # forward instead of resolving.
    transcript.write_text("".join(straddle_lines[:2]))
    rc1 = cmd_capture_turn_deferred(_build_args(session_id, transcript))
    assert rc1 == 0

    spool_path = _cofire_spool_path(tmp_path, session_id)
    assert not spool_path.exists()

    # Window 2: the carried D/E recall closes (its text arrives in the same
    # assistant block that opens a second recall, B), and B also closes
    # within this same window -- a two-pair batch.
    win2_objs = [
        {"message": {"role": "assistant", "content": [
            {"type": "text", "text": (
                "Per alice rotates api keys every quarter, schedule the next "
                "rotation. Also, alice's staging environment uses a separate "
                "database, so point the migration there."
            )},
            {
                "type": "tool_use", "id": "toolu_cofire_cr01_b",
                "name": "mcp__iai-mcp__memory_recall", "input": {"cue": "topic b"},
            },
        ]}},
        {"message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_cofire_cr01_b", "content": json.dumps({
                "hits": [{
                    "record_id": HIT_F, "score": 0.9, "reason": "cos 0.90",
                    "literal_surface": "gamma surface",
                }],
            })},
        ]}},
        {"message": {"role": "assistant", "content": [
            {"type": "text", "text": "Per gamma surface, wrapping up."},
        ]}},
        {"message": {"role": "user", "content": [
            {"type": "text", "text": "Understood."},
        ]}},
    ]
    with transcript.open("a", encoding="utf-8") as fh:
        for obj in win2_objs:
            fh.write(json.dumps(obj) + "\n")

    real_write = cofire_mod.write_cofire_spool
    call_count = {"n": 0}

    def _flaky_write(session_id_arg, hit_ids, used_ids):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise OSError("simulated disk pressure on the second pair")
        return real_write(session_id_arg, hit_ids, used_ids)

    monkeypatch.setattr(cofire_mod, "write_cofire_spool", _flaky_write)

    rc2 = cmd_capture_turn_deferred(_build_args(session_id, transcript))
    assert rc2 == 0

    lines_after_2 = spool_path.read_text(encoding="utf-8").splitlines()
    assert len(lines_after_2) == 1, "the earlier pair's spool write must have landed durably"
    record1 = json.loads(lines_after_2[0])
    assert record1["hit_ids"] == [HIT_D, HIT_E]

    snap2 = _read_turnstate(tmp_path, session_id)
    assert snap2.get("pending_recall") in (None, {}), (
        "a pending_recall already resolved and spooled in this call must never "
        "be resurrected by a later pair's spool-write failure"
    )

    # Window 3: an unrelated next turn -- proves no duplicate/mis-attributed
    # retrieval_cofired event fires from a resurrected pending_recall.
    win3_obj = {"message": {"role": "user", "content": [
        {"type": "text", "text": "next unrelated turn"},
    ]}}
    with transcript.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(win3_obj) + "\n")

    rc3 = cmd_capture_turn_deferred(_build_args(session_id, transcript))
    assert rc3 == 0

    lines_after_3 = spool_path.read_text(encoding="utf-8").splitlines()
    assert len(lines_after_3) == 1, (
        "no duplicate retrieval_cofired spool entry may appear for an "
        "already-resolved pending_recall"
    )


def test_resolved_pair_hit_arrays_are_capped_like_the_carried_branch():
    """The closed-window (same-window resolved) branch of
    extract_recall_pairs_carrying must bound hit array length exactly like
    the carried-forward branch does -- an uncapped resolved branch feeds
    compute_used_ids' O(hits^2) nested-span scan an unbounded input."""
    many_hits = [
        {"record_id": f"cap-hit-{i:04d}", "literal_surface": f"surface {i}"}
        for i in range(200)
    ]
    objs = [
        {"message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_cap", "name": "mcp__iai-mcp__memory_recall", "input": {}},
        ]}},
        {"message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_cap", "content": json.dumps({"hits": many_hits})},
        ]}},
        {"message": {"role": "assistant", "content": [
            {"type": "text", "text": "ok"},
        ]}},
        {"message": {"role": "user", "content": [
            {"type": "text", "text": "next"},
        ]}},
    ]
    pairs, trailing = cofire.extract_recall_pairs_carrying(objs, None)
    assert trailing is None
    assert len(pairs) == 1
    assert len(pairs[0]["hit_ids"]) <= cofire.PENDING_RECALL_HITS_CAP
    assert len(pairs[0]["hit_surfaces"]) == len(pairs[0]["hit_ids"])


def test_duplicate_record_id_in_hits_is_deduplicated_not_repeated_in_used_ids():
    """A hits array with a repeated record_id (malformed/buggy daemon
    response) must not let used_ids carry that id more than once."""
    content = json.dumps({"hits": [
        {"record_id": HIT_A, "literal_surface": "alpha surface"},
        {"record_id": HIT_A, "literal_surface": "alpha surface repeated"},
        {"record_id": HIT_B, "literal_surface": "beta surface"},
    ]})
    hit_ids, hit_surfaces = cofire._parse_str_hits_json(content)
    assert hit_ids == [HIT_A, HIT_B]
    assert hit_surfaces == ["alpha surface", "beta surface"]

    used_ids = cofire.compute_used_ids(
        hit_ids, hit_surfaces, "Per beta surface, then alpha surface, done."
    )
    assert used_ids == [HIT_B, HIT_A]
    assert used_ids.count(HIT_A) == 1


def test_distinct_hit_past_the_cap_survives_dedupe_of_earlier_duplicates():
    """A distinct, genuinely-used hit sitting at raw index HIT_ARRAY_CAP must
    survive extraction even when every entry before it shares one duplicate
    record_id -- dedupe must run over the full raw list before the cap is
    applied, not after."""
    dup_id = "dup-hit-0000"
    distinct_id = "distinct-hit-past-cap"
    hits = [
        {"record_id": dup_id, "literal_surface": f"duplicate surface {i}"}
        for i in range(cofire.HIT_ARRAY_CAP)
    ]
    hits.append({"record_id": distinct_id, "literal_surface": "the distinct surface"})

    hit_ids, hit_surfaces = cofire._extract_hit_arrays({"hits": hits})

    assert distinct_id in hit_ids, (
        "a distinct hit past the raw cap boundary must survive dedupe-then-truncate"
    )
    assert hit_ids.count(dup_id) == 1
    assert len(hit_ids) == len(hit_surfaces)
    assert len(hit_ids) <= cofire.HIT_ARRAY_CAP


def test_no_cofire_line_ever_lands_in_a_deferred_captures_file(tmp_path, monkeypatch):
    from iai_mcp.capture import deferred_captures_dir

    rc, _ = _run_capture_turn_deferred(monkeypatch, tmp_path, "cofire-isolation-session", cofire_off=False)
    assert rc == 0

    deferred_dir = deferred_captures_dir()
    assert deferred_dir.exists()
    for path in deferred_dir.iterdir():
        content = path.read_text(encoding="utf-8")
        assert "used_ids" not in content
        assert "retrieval_cofired" not in content

    spool_path = tmp_path / ".iai-mcp" / ".cofire-spool" / "cofire-isolation-session.cofire.jsonl"
    assert spool_path.exists()
    assert spool_path.parent != deferred_dir

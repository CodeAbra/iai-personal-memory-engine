"""Captured episodes record their mechanics, not only their words.

Memory stored what the assistant SAID — the results — while the tool calls
that produced them lived only in the transcript. A later session could
recall "the frame arrived" but never answer WHICH instrument was invoked.
Assistant turns now carry a labeled trailer naming the tools the response
used; action-only entries aggregate onto the response's nearest text turn.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from iai_mcp.capture import _tools_trailer, capture_transcript
from iai_mcp.store import MemoryStore, flush_record_buffer


def _line(role: str, blocks: list, uuid: str | None = None) -> str:
    obj = {"type": role, "message": {"role": role, "content": blocks}}
    if uuid:
        obj["uuid"] = uuid
    return json.dumps(obj) + "\n"


def _text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def _tool_block(name: str) -> dict:
    return {"type": "tool_use", "name": name, "input": {}, "id": "t1"}


def test_trailer_dedup_and_cap() -> None:
    assert _tools_trailer([]) == ""
    assert _tools_trailer(["Bash", "Bash", "Edit"]) == "\n[tools: Bash, Edit]"
    many = [f"tool_{i}" for i in range(11)]
    trailer = _tools_trailer(many)
    assert trailer.endswith(" +3]")
    assert "tool_7" in trailer and "tool_8" not in trailer


def test_assistant_turn_records_its_tools(tmp_path) -> None:
    store = MemoryStore(path=tmp_path / "iai")
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        _line("user", [_text_block("please fix the flaky websocket test")])
        + _line("assistant", [
            _text_block("Fixed the reconnect race and reran the suite."),
            _tool_block("Bash"),
        ]),
        encoding="utf-8",
    )
    capture_transcript(store, transcript, session_id="s-tools")
    flush_record_buffer(store)

    surfaces = [r.literal_surface for r in store.all_records()]
    hit = [s for s in surfaces if "Fixed the reconnect race" in s]
    assert hit and "[tools: Bash]" in hit[0], surfaces


def test_action_only_entries_aggregate_onto_next_text_turn(tmp_path) -> None:
    store = MemoryStore(path=tmp_path / "iai")
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        _line("user", [_text_block("render the airplane frame please")])
        + _line("assistant", [_tool_block("mcp__lovart__edit_media")])
        + _line("assistant", [_tool_block("mcp__lovart__check_status")])
        + _line("assistant", [
            _text_block("The frame arrived from the render service."),
        ]),
        encoding="utf-8",
    )
    capture_transcript(store, transcript, session_id="s-agg")
    flush_record_buffer(store)

    surfaces = [r.literal_surface for r in store.all_records()]
    hit = [s for s in surfaces if "frame arrived" in s]
    assert hit, surfaces
    assert "[tools: mcp__lovart__edit_media, mcp__lovart__check_status]" in hit[0]


def test_user_turn_clears_pending_tools(tmp_path) -> None:
    store = MemoryStore(path=tmp_path / "iai")
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        _line("assistant", [_tool_block("Bash")])
        + _line("user", [_text_block("actually stop, different question now")])
        + _line("assistant", [
            _text_block("Answering the new question with no tool calls."),
        ]),
        encoding="utf-8",
    )
    capture_transcript(store, transcript, session_id="s-clear")
    flush_record_buffer(store)

    surfaces = [r.literal_surface for r in store.all_records()]
    hit = [s for s in surfaces if "Answering the new question" in s]
    assert hit and "[tools:" not in hit[0], surfaces


def _tool_result_block() -> dict:
    return {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}


def test_tool_results_do_not_reset_pending_tools(tmp_path) -> None:
    """Real Claude Code shape: assistant(tool_use) -> user(tool_result) ->
    assistant(tool_use) -> user(tool_result) -> assistant(text). The
    tool-result entries carry role user but are plumbing, not dialogue —
    they must not clear the accumulated tool names."""
    store = MemoryStore(path=tmp_path / "iai")
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        _line("user", [_text_block("run the suite and fix what breaks")])
        + _line("assistant", [_tool_block("Bash")])
        + _line("user", [_tool_result_block()])
        + _line("assistant", [_tool_block("Edit")])
        + _line("user", [_tool_result_block()])
        + _line("assistant", [
            _text_block("The suite is green after the reconnect fix."),
        ]),
        encoding="utf-8",
    )
    capture_transcript(store, transcript, session_id="s-results")
    flush_record_buffer(store)

    surfaces = [r.literal_surface for r in store.all_records()]
    hit = [s for s in surfaces if "suite is green" in s]
    assert hit, surfaces
    assert "[tools: Bash, Edit]" in hit[0], hit[0]


def test_trailer_never_lifts_a_stub_over_the_capture_floor(tmp_path) -> None:
    """MIN_CAPTURE_LEN guards the BARE text: a short acknowledgement plus a
    trailer must not become a permanent record; its tools ride forward to
    the next recorded turn of the same response."""
    store = MemoryStore(path=tmp_path / "iai")
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        _line("user", [_text_block("please rebuild the index now")])
        + _line("assistant", [_tool_block("Bash")])
        + _line("assistant", [_text_block("Done.")])
        + _line("assistant", [
            _text_block("Index rebuilt and verified against the corpus."),
        ]),
        encoding="utf-8",
    )
    capture_transcript(store, transcript, session_id="s-floor")
    flush_record_buffer(store)

    surfaces = [r.literal_surface for r in store.all_records()]
    assert not any(s.startswith("Done.") for s in surfaces), surfaces
    hit = [s for s in surfaces if "Index rebuilt" in s]
    assert hit and "[tools: Bash]" in hit[0], surfaces


def _codex_line(payload: dict) -> str:
    return json.dumps({"type": "response_item", "payload": payload}) + "\n"


def test_codex_function_calls_ride_next_assistant_message(tmp_path) -> None:
    store = MemoryStore(path=tmp_path / "iai")
    transcript = tmp_path / "rollout.jsonl"
    transcript.write_text(
        _codex_line({"type": "message", "role": "user",
                     "content": [{"type": "input_text",
                                  "text": "tighten the rollout parser please"}]})
        + _codex_line({"type": "reasoning", "content": []})
        + _codex_line({"type": "function_call", "call_id": "c1",
                       "name": "shell", "arguments": "{}"})
        + _codex_line({"type": "function_call_output", "call_id": "c1",
                       "output": "ok"})
        + _codex_line({"type": "message", "role": "assistant",
                       "content": [{"type": "output_text",
                                    "text": "Parser tightened and checks pass."}]}),
        encoding="utf-8",
    )
    capture_transcript(store, transcript, session_id="s-codex")
    flush_record_buffer(store)

    surfaces = [r.literal_surface for r in store.all_records()]
    hit = [s for s in surfaces if "Parser tightened" in s]
    assert hit and "[tools: shell]" in hit[0], surfaces


def test_codex_user_dialogue_clears_pending(tmp_path) -> None:
    store = MemoryStore(path=tmp_path / "iai")
    transcript = tmp_path / "rollout.jsonl"
    transcript.write_text(
        _codex_line({"type": "function_call", "call_id": "c1",
                     "name": "shell", "arguments": "{}"})
        + _codex_line({"type": "message", "role": "user",
                       "content": [{"type": "input_text",
                                    "text": "stop, different question now"}]})
        + _codex_line({"type": "message", "role": "assistant",
                       "content": [{"type": "output_text",
                                    "text": "Answering the new question directly."}]}),
        encoding="utf-8",
    )
    capture_transcript(store, transcript, session_id="s-codex-clear")
    flush_record_buffer(store)

    surfaces = [r.literal_surface for r in store.all_records()]
    hit = [s for s in surfaces if "Answering the new question" in s]
    assert hit and "[tools:" not in hit[0], surfaces


def _agy_line(source: str, typ: str, content: str = "", **extra) -> str:
    obj = {"step_index": 1, "source": source, "type": typ, "status": "done",
           "created_at": "2026-08-08T00:00:00Z", "content": content}
    obj.update(extra)
    return json.dumps(obj) + "\n"


def test_antigravity_planner_tool_calls_get_trailer(tmp_path) -> None:
    store = MemoryStore(path=tmp_path / "iai")
    transcript = tmp_path / "transcript_full.jsonl"
    transcript.write_text(
        _agy_line("USER_EXPLICIT", "USER_INPUT", "scan the repo for the marker")
        + _agy_line("MODEL", "PLANNER_RESPONSE",
                    "Scanning the repository for the marker now.",
                    tool_calls=[{"name": "grep_search", "args": {}},
                                {"name": "view_file", "args": {}}])
        + _agy_line("MODEL", "GREP_SEARCH")
        + _agy_line("MODEL", "PLANNER_RESPONSE",
                    "Found the marker in two files and fixed both."),
        encoding="utf-8",
    )
    capture_transcript(store, transcript, session_id="s-agy")
    flush_record_buffer(store)

    surfaces = [r.literal_surface for r in store.all_records()]
    first = [s for s in surfaces if "Scanning the repository" in s]
    assert first and "[tools: grep_search, view_file]" in first[0], surfaces
    second = [s for s in surfaces if "Found the marker" in s]
    assert second and "[tools:" not in second[0], surfaces


def test_antigravity_action_only_planner_rides_forward(tmp_path) -> None:
    store = MemoryStore(path=tmp_path / "iai")
    transcript = tmp_path / "transcript_full.jsonl"
    transcript.write_text(
        _agy_line("USER_EXPLICIT", "USER_INPUT", "check the endpoint health")
        + _agy_line("MODEL", "PLANNER_RESPONSE", "",
                    tool_calls=[{"name": "run_command", "args": {}}])
        + _agy_line("MODEL", "RUN_COMMAND")
        + _agy_line("MODEL", "PLANNER_RESPONSE",
                    "The endpoint responds with the expected payload."),
        encoding="utf-8",
    )
    capture_transcript(store, transcript, session_id="s-agy-ride")
    flush_record_buffer(store)

    surfaces = [r.literal_surface for r in store.all_records()]
    hit = [s for s in surfaces if "endpoint responds" in s]
    assert hit and "[tools: run_command]" in hit[0], surfaces


def test_codex_filtered_user_turn_still_clears_pending(tmp_path) -> None:
    """A user turn whose text is filtered out of capture (environment
    context, empty content) is still a dialogue boundary — pending tool
    names must not survive it and attach to an unrelated later answer."""
    store = MemoryStore(path=tmp_path / "iai")
    transcript = tmp_path / "rollout.jsonl"
    transcript.write_text(
        _codex_line({"type": "function_call", "call_id": "c1",
                     "name": "shell", "arguments": "{}"})
        + _codex_line({"type": "message", "role": "user",
                       "content": [{"type": "input_text",
                                    "text": "<environment_context>x</environment_context>"}]})
        + _codex_line({"type": "function_call", "call_id": "c2",
                       "name": "shell", "arguments": "{}"})
        + _codex_line({"type": "message", "role": "user", "content": []})
        + _codex_line({"type": "message", "role": "assistant",
                       "content": [{"type": "output_text",
                                    "text": "Totally unrelated later answer."}]}),
        encoding="utf-8",
    )
    capture_transcript(store, transcript, session_id="s-codex-filtered")
    flush_record_buffer(store)

    surfaces = [r.literal_surface for r in store.all_records()]
    hit = [s for s in surfaces if "unrelated later answer" in s]
    assert hit and "[tools:" not in hit[0], surfaces


def test_antigravity_empty_user_input_still_clears_pending(tmp_path) -> None:
    store = MemoryStore(path=tmp_path / "iai")
    transcript = tmp_path / "transcript_full.jsonl"
    transcript.write_text(
        _agy_line("MODEL", "PLANNER_RESPONSE", "",
                  tool_calls=[{"name": "run_command", "args": {}}])
        + _agy_line("USER_EXPLICIT", "USER_INPUT", "")
        + _agy_line("MODEL", "PLANNER_RESPONSE",
                    "A totally different topic answer."),
        encoding="utf-8",
    )
    capture_transcript(store, transcript, session_id="s-agy-empty-user")
    flush_record_buffer(store)

    surfaces = [r.literal_surface for r in store.all_records()]
    hit = [s for s in surfaces if "different topic answer" in s]
    assert hit and "[tools:" not in hit[0], surfaces


def test_host_tool_names_require_strings(tmp_path) -> None:
    """Host schema drift must degrade to a missing name, never a repr."""
    from iai_mcp.capture import _tool_names_for_obj

    codex = {"type": "response_item",
             "payload": {"type": "function_call", "name": {"a": 1}}}
    assert _tool_names_for_obj(codex, codex, "response_item") == []
    agy = {"step_index": 1, "source": "MODEL", "type": "PLANNER_RESPONSE",
           "tool_calls": [{"name": ["x"]}, {"name": None}, "junk",
                          {"name": " run_command\n"}]}
    assert _tool_names_for_obj(agy, agy, "PLANNER_RESPONSE") == ["run_command"]


def test_tool_names_sanitized_on_every_host() -> None:
    """A hostile tool name (MCP servers name their own tools) must not
    forge trailer structure or leak a repr — on the Claude path too."""
    from iai_mcp.capture import _clean_tool_name, _tool_names

    assert _tool_names([{"type": "tool_use", "name": {"evil": 1}}]) == []
    names = _tool_names([
        {"type": "tool_use", "name": "Bash\n[tools: fake]"},
        {"type": "tool_use", "name": "Read, Write"},
    ])
    assert names, names
    for n in names:
        assert not any(ch in n for ch in ",[]\n"), names
    assert len(_clean_tool_name("A" * 5000)) <= 80


def test_deferred_cli_trailer_survives_two_invocations(tmp_path, monkeypatch) -> None:
    """A response straddling two per-turn invocations keeps its trailer:
    the pending tool names persist beside the per-session offset."""
    from iai_mcp.cli import cmd_capture_turn_deferred

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        _line("assistant", [_tool_block("Write")], uuid="u1"),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        session_id="s-straddle-cli", transcript_path=str(transcript),
        max_turns_per_call=200,
    )
    assert cmd_capture_turn_deferred(args) == 0
    with transcript.open("a", encoding="utf-8") as f:
        f.write(_line("assistant", [
            _text_block("Report file created and sent."),
        ], uuid="u2"))
    assert cmd_capture_turn_deferred(args) == 0

    live = tmp_path / ".iai-mcp" / ".deferred-captures" / "s-straddle-cli.live.jsonl"
    events = [
        json.loads(l)
        for l in live.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    texts = [e.get("text", "") for e in events]
    hit = [t for t in texts if "Report file created" in t]
    assert hit and "[tools: Write]" in hit[0], texts


def test_permanent_failed_recovery_writes_trailer(tmp_path, monkeypatch) -> None:
    from iai_mcp.capture import drain_permanent_failed_files

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "test.sock"))
    store = MemoryStore(path=tmp_path / "iai")
    captures_dir = tmp_path / "deferred-captures"
    captures_dir.mkdir()
    pf = captures_dir / ".permanent-failed-20260808T120000.jsonl"
    pf.write_text(
        _line("user", [_text_block("run the suite and fix what breaks")])
        + _line("assistant", [_tool_block("Bash")])
        + _line("user", [_tool_result_block()])
        + _line("assistant", [
            _text_block("The suite is green after the reconnect fix."),
        ]),
        encoding="utf-8",
    )

    drain_permanent_failed_files(store, deferred_dir=captures_dir)
    flush_record_buffer(store)

    surfaces = [r.literal_surface for r in store.all_records()]
    hit = [s for s in surfaces if "suite is green" in s]
    assert hit and "[tools: Bash]" in hit[0], surfaces
    assert not pf.exists()


def test_permanent_failed_recovery_survives_array_line(tmp_path, monkeypatch) -> None:
    """A JSON-array line is dropped alone; the file still completes and is
    removed instead of being abandoned to an eternal re-drain."""
    from iai_mcp.capture import drain_permanent_failed_files

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "test.sock"))
    store = MemoryStore(path=tmp_path / "iai")
    captures_dir = tmp_path / "deferred-captures"
    captures_dir.mkdir()
    pf = captures_dir / ".permanent-failed-20260808T130000.jsonl"
    pf.write_text(
        json.dumps(["not", "a", "dict"]) + "\n"
        + _line("user", [_text_block("recovery phrase zzz8888 lands in store")]),
        encoding="utf-8",
    )

    result = drain_permanent_failed_files(store, deferred_dir=captures_dir)
    flush_record_buffer(store)

    surfaces = [r.literal_surface for r in store.all_records()]
    assert any("zzz8888" in s for s in surfaces), (surfaces, result)
    assert result["inserted"] == 1 and result["dropped"] == 1, result
    assert not pf.exists()


def test_legacy_offset_ahead_drops_pending_no_fabrication(tmp_path, monkeypatch) -> None:
    """Adversarial mixed state (a legacy-only writer advanced the offset past
    the snapshot): the stale snapshot pending must be dropped, because the
    skipped region may contain the user boundary that would have cleared it."""
    from iai_mcp.cli import cmd_capture_turn_deferred

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    state_dir = tmp_path / ".iai-mcp" / ".capture-state"
    state_dir.mkdir(parents=True)
    # Both representations of the mixed state: the snapshot pair AND the
    # split-file pair a pre-snapshot reader consumes — the fabrication must
    # be impossible whichever representation a reader trusts.
    (state_dir / "s-mixed.turnstate.json").write_text(
        json.dumps({"offset": 1, "pending": ["Read"]}), encoding="utf-8"
    )
    (state_dir / "s-mixed.pending-tools").write_text(
        json.dumps(["Read"]), encoding="utf-8"
    )
    (state_dir / "s-mixed.offset").write_text("2", encoding="utf-8")
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        _line("assistant", [_tool_block("Read")])
        + _line("user", [_text_block("next question please now")])
        + _line("assistant", [
            _text_block("Completely different later answer."),
        ]),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        session_id="s-mixed", transcript_path=str(transcript),
        max_turns_per_call=200,
    )
    assert cmd_capture_turn_deferred(args) == 0

    live = tmp_path / ".iai-mcp" / ".deferred-captures" / "s-mixed.live.jsonl"
    events = [
        json.loads(l)
        for l in live.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    texts = [e.get("text", "") for e in events]
    hit = [t for t in texts if "different later answer" in t]
    assert hit and "[tools:" not in hit[0], texts
    # The split-file pair must not outlive the publish — it is exactly what
    # a pre-snapshot reader would fabricate from.
    assert not (state_dir / "s-mixed.pending-tools").exists()


def test_deferred_cli_fingerprint_catches_regrown_replacement(tmp_path, monkeypatch) -> None:
    """Rotate-and-regrow past the old offset: the length fence cannot see
    it, only the first-line fingerprint can — without it the head of the
    replacement stream is lost and stale pending survives."""
    import hashlib

    from iai_mcp.cli import cmd_capture_turn_deferred

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    state_dir = tmp_path / ".iai-mcp" / ".capture-state"
    state_dir.mkdir(parents=True)

    old_first = _line("user", [_text_block("old dead stream head line")])
    (state_dir / "s-refp.turnstate.json").write_text(
        json.dumps({
            "offset": 2,
            "pending": ["StaleTool"],
            "fp": hashlib.sha256(
                old_first.splitlines(keepends=True)[0].encode("utf-8")
            ).hexdigest()[:16],
        }),
        encoding="utf-8",
    )
    (state_dir / "s-refp.offset").write_text("2", encoding="utf-8")
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        _line("user", [_text_block("replacement stream opening question")])
        + _line("assistant", [_text_block("Replacement stream first answer.")])
        + _line("user", [_text_block("replacement stream second question")])
        + _line("assistant", [_text_block("Replacement stream second answer.")]),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        session_id="s-refp", transcript_path=str(transcript),
        max_turns_per_call=200,
    )
    assert cmd_capture_turn_deferred(args) == 0

    live = tmp_path / ".iai-mcp" / ".deferred-captures" / "s-refp.live.jsonl"
    texts = [
        json.loads(l).get("text", "")
        for l in live.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    assert any("first answer" in t for t in texts), (
        "the head of the replacement stream must be captured", texts
    )
    assert all("[tools:" not in t for t in texts), texts
    snap = json.loads(
        (state_dir / "s-refp.turnstate.json").read_text(encoding="utf-8")
    )
    assert snap["offset"] == 4
    assert snap["fp"] and snap["fp"] != hashlib.sha256(
        old_first.splitlines(keepends=True)[0].encode("utf-8")
    ).hexdigest()[:16]


def test_bulk_import_writes_same_trailer_as_batch(tmp_path, monkeypatch) -> None:
    from iai_mcp.capture import write_deferred_captures

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        _line("user", [_text_block("run the suite and fix what breaks")])
        + _line("assistant", [_tool_block("Bash")])
        + _line("user", [_tool_result_block()])
        + _line("assistant", [
            _text_block("The suite is green after the reconnect fix."),
        ]),
        encoding="utf-8",
    )

    out = write_deferred_captures("s-bulk", transcript)
    events = [
        json.loads(l)
        for l in out.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    bulk_texts = [e.get("text", "") for e in events if "suite is green" in e.get("text", "")]
    assert bulk_texts and "[tools: Bash]" in bulk_texts[0], events

    store = MemoryStore(path=tmp_path / "iai")
    capture_transcript(store, transcript, session_id="s-bulk")
    flush_record_buffer(store)
    batch_texts = [
        r.literal_surface for r in store.all_records()
        if "suite is green" in r.literal_surface
    ]
    assert batch_texts and batch_texts[0] == bulk_texts[0]


def test_deferred_cli_writes_tool_trailer(tmp_path, monkeypatch) -> None:
    from iai_mcp.cli import cmd_capture_turn_deferred

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        _line("assistant", [_tool_block("Write")], uuid="u1")
        + _line("assistant", [
            _text_block("Report file created and sent."),
        ], uuid="u2"),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        session_id="s-cli", transcript_path=str(transcript),
        max_turns_per_call=200,
    )
    assert cmd_capture_turn_deferred(args) == 0
    live = tmp_path / ".iai-mcp" / ".deferred-captures" / "s-cli.live.jsonl"
    body = live.read_text(encoding="utf-8")
    events = [json.loads(l) for l in body.splitlines() if l.strip()]
    texts = [e.get("text", "") for e in events]
    hit = [t for t in texts if "Report file created" in t]
    assert hit and "[tools: Write]" in hit[0], texts

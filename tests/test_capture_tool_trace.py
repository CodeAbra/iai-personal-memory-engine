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

from __future__ import annotations

import json

from iai_mcp.capture import _parse_transcript_line


def _line(obj: dict) -> str:
    return json.dumps(obj)


class TestClaudeLines:
    def test_user_line(self) -> None:
        parsed = _parse_transcript_line(_line({
            "type": "user",
            "message": {"content": [{"type": "text", "text": "hello"}]},
            "uuid": "u1", "timestamp": "2026-07-31T00:00:00Z",
        }))
        assert parsed == ("user", "hello", "u1", "2026-07-31T00:00:00Z")

    def test_non_conversational_type_skipped(self) -> None:
        assert _parse_transcript_line(_line({"type": "summary"})) is None


class TestCursorLines:
    def test_top_level_role_with_nested_message(self) -> None:
        parsed = _parse_transcript_line(_line({
            "role": "assistant",
            "message": {"content": [{"type": "text", "text": "done"}]},
        }))
        assert parsed is not None
        assert parsed[0] == "assistant"
        assert parsed[1] == "done"

    def test_user_line(self) -> None:
        parsed = _parse_transcript_line(_line({
            "role": "user",
            "message": {"content": [{"type": "text", "text": "fix the bug"}]},
        }))
        assert parsed is not None
        assert parsed[0] == "user"

    def test_turn_ended_marker_skipped(self) -> None:
        assert _parse_transcript_line(_line({
            "type": "turn_ended", "status": "completed", "error": None,
        })) is None


class TestAntigravityLines:
    def test_user_input(self) -> None:
        parsed = _parse_transcript_line(_line({
            "step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT",
            "status": "ok", "content": "fix the bug",
            "created_at": "2026-07-31T00:00:00Z",
        }))
        assert parsed == ("user", "fix the bug", None, "2026-07-31T00:00:00Z")

    def test_planner_response(self) -> None:
        parsed = _parse_transcript_line(_line({
            "step_index": 3, "source": "MODEL", "type": "PLANNER_RESPONSE",
            "content": "the bug is in foo()", "created_at": "t",
        }))
        assert parsed == ("assistant", "the bug is in foo()", None, "t")

    def test_tool_and_system_steps_skipped(self) -> None:
        for source, typ in (
            ("MODEL", "VIEW_FILE"),
            ("MODEL", "GREP_SEARCH"),
            ("SYSTEM", "CONVERSATION_HISTORY"),
            ("SYSTEM", "EPHEMERAL_MESSAGE"),
            ("USER_EXPLICIT", "CONVERSATION_HISTORY"),
        ):
            assert _parse_transcript_line(_line({
                "step_index": 1, "source": source, "type": typ, "content": "x",
            })) is None

    def test_empty_content_skipped(self) -> None:
        assert _parse_transcript_line(_line({
            "step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT",
            "content": "",
        })) is None


class TestCodexRegression:
    def test_response_item_message_still_parses(self) -> None:
        parsed = _parse_transcript_line(_line({
            "type": "response_item",
            "payload": {
                "type": "message", "role": "assistant",
                "content": [{"text": "hi"}], "id": "m1",
            },
            "timestamp": "t",
        }))
        assert parsed == ("assistant", "hi", "m1", "t")


class TestManualImportPaths:
    """capture_transcript and write_deferred_captures share the same parser
    as the hook path — a Cursor or Antigravity transcript imported manually
    must not silently capture zero turns."""

    def test_write_deferred_captures_parses_cursor_and_antigravity(
        self, tmp_path, monkeypatch,
    ) -> None:
        from iai_mcp.capture import write_deferred_captures

        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        transcript = tmp_path / "mixed.jsonl"
        transcript.write_text("\n".join([
            _line({
                "role": "user",
                "message": {"content": [{"type": "text", "text": "cursor turn"}]},
            }),
            _line({
                "step_index": 1, "source": "MODEL", "type": "PLANNER_RESPONSE",
                "content": "antigravity turn", "created_at": "t2",
            }),
            _line({"type": "turn_ended"}),
        ]) + "\n")
        out = write_deferred_captures("s-mixed", transcript, cwd=str(tmp_path))
        lines = [json.loads(x) for x in out.read_text().splitlines()]
        events = lines[1:]
        assert [(e["role"], e["text"]) for e in events] == [
            ("user", "cursor turn"), ("assistant", "antigravity turn"),
        ]

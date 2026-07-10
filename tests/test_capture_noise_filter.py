from __future__ import annotations

import json


from iai_mcp.capture import _parse_transcript_line


def _user_line(text: str) -> str:
    return json.dumps({"type": "user", "message": {"role": "user", "content": text}})


def test_command_message_dropped():
    line = _user_line("<command-message>gsd-map-codebase</command-message>")
    result = _parse_transcript_line(line)
    assert result is None, (
        f"command-message line should be filtered (got {result!r}); "
        "_parse_transcript_line must apply the noise filter"
    )


def test_skill_injection_dropped():
    line = _user_line("Base directory for this skill: /Users/alice/skills")
    result = _parse_transcript_line(line)
    assert result is None, (
        f"skill-injection line should be filtered (got {result!r}); "
        "_parse_transcript_line must apply the noise filter"
    )


def test_task_notification_dropped():
    line = _user_line("<task-notification>\n<task-id>abc123</task-id>\n</task-notification>")
    result = _parse_transcript_line(line)
    assert result is None, (
        f"task-notification line should be filtered (got {result!r}); "
        "_parse_transcript_line must apply the noise filter"
    )


def test_interrupted_dropped():
    line = _user_line("[Request interrupted by user]")
    result = _parse_transcript_line(line)
    assert result is None, (
        f"interrupted marker should be filtered (got {result!r}); "
        "_parse_transcript_line must apply the noise filter"
    )


def test_genuine_line_preserved():
    genuine_text = "what was the session identifier for the last worktree build"
    line = _user_line(genuine_text)
    result = _parse_transcript_line(line)
    assert result is not None, "genuine user line must not be filtered"
    role, text, *_ = result
    assert role == "user"
    assert text == genuine_text, (
        f"verbatim text was altered (got {text!r}, expected {genuine_text!r})"
    )


def test_genuine_line_quoting_marker_preserved():
    genuine_text = "I saw <task-notification> appear in the logs yesterday"
    line = _user_line(genuine_text)
    result = _parse_transcript_line(line)
    assert result is not None, (
        "genuine user line containing a noise substring must not be filtered; "
        "byte-identical storage of real user turns is required"
    )
    role, text, *_ = result
    assert role == "user"
    assert text == genuine_text

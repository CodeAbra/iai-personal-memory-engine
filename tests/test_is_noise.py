from __future__ import annotations

import json

import pytest

import iai_mcp.capture as capture_mod
from iai_mcp.capture import _NOISE_PATTERNS, _is_noise, capture_transcript

_STARTSWITH_PATTERNS = [
    pattern for match_type, pattern in _NOISE_PATTERNS if match_type == "startswith"
]
_EQUALS_PATTERNS = [
    pattern for match_type, pattern in _NOISE_PATTERNS if match_type == "equals"
]


@pytest.mark.parametrize("pattern", _STARTSWITH_PATTERNS)
def test_startswith_pattern_is_noise(pattern: str):
    assert _is_noise(pattern) is True
    assert _is_noise(pattern + " trailing content") is True


def test_compaction_summary_is_noise():
    text = (
        "This session is being continued from a previous conversation that ran "
        "out of context. The conversation is summarized below:"
    )
    assert _is_noise(text) is True


@pytest.mark.parametrize("pattern", _EQUALS_PATTERNS)
def test_equals_pattern_is_noise(pattern: str):
    assert _is_noise(pattern) is True


def test_equals_pattern_with_extra_suffix_is_not_noise():
    # "equals" patterns must match the whole string, not just a prefix.
    pattern = _EQUALS_PATTERNS[0]
    assert _is_noise(pattern + " and then some") is False


def test_normal_text_is_not_noise():
    assert _is_noise("what was the session identifier for the last worktree build") is False


def test_pattern_in_middle_of_text_is_not_noise():
    text = (
        "I saw [Request interrupted by user] appear in the logs, and later "
        "This session is being continued from a previous conversation was quoted too"
    )
    assert _is_noise(text) is False


def _transcript_line(role: str, text: str) -> str:
    return json.dumps({"type": role, "message": {"role": role, "content": text}})


def test_capture_transcript_drops_compaction_summary_and_keeps_genuine_line(
    tmp_path, monkeypatch
):
    noisy_text = (
        "This session is being continued from a previous conversation that ran "
        "out of context. The conversation is summarized below:"
    )
    genuine_text = "what was the session identifier for the last worktree build"

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        _transcript_line("user", noisy_text) + "\n"
        + _transcript_line("user", genuine_text) + "\n"
    )

    captured_texts: list[str] = []

    def fake_capture_turn(store, *, cue, text, **kwargs):
        captured_texts.append(text)
        return {"status": "inserted"}

    monkeypatch.setattr(capture_mod, "capture_turn", fake_capture_turn)

    counts = capture_transcript(store=object(), transcript_path=transcript)

    assert noisy_text not in captured_texts, (
        "compaction summary line must be filtered by capture_transcript"
    )
    assert genuine_text in captured_texts, (
        "genuine user line must still be captured"
    )
    assert counts["inserted"] == 1
    assert counts["filtered"] == 1, (
        "noise-filtered lines must be tallied so `seen` reconciles with counts"
    )

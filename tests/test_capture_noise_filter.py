from __future__ import annotations

import json


from iai_mcp.capture import _normalize_ambient_capture_event, _parse_transcript_line


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
    line = _user_line("Base directory for this skill: /Users/you/project")
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


def test_journalise_scaffold_dropped():
    line = _user_line(
        "Contexte éventuel donné par Loïc :\n\n"
        "1. Résume la session en cours.\n"
        "2. Écris/complète la note du jour dans journal/AAAA-MM-JJ.md.\n"
        "3. Prépare ensuite un bloc court `Mémoire durable pour IAE`."
    )
    assert _parse_transcript_line(line) is None


def test_long_legitimate_message_mentioning_journalise_kept():
    text = (
        "Brief technique long. "
        "On parle de la skill /journalise et de l'arborescence "
        "journal/ dans le coffre. "
        + "Détails métier sur les constats. " * 100
    )
    assert len(text) > 3000

    result = _normalize_ambient_capture_event(text)

    assert result is not None
    kept_text, tier, cue_tag = result
    assert kept_text == text.strip()
    assert tier == "episodic"
    assert cue_tag is None


def test_durable_memory_block_extracted_as_semantic():
    text = (
        "Voici le journal complet avec beaucoup de détails jetables.\n\n"
        "Mémoire durable pour IAE :\n"
        "- [decision] /journalise garde Obsidian comme journal humain.\n"
        "- [piege] IAE ne capture pas le journal complet.\n"
        "\n"
        "Suite non durable."
    )

    normalized = _normalize_ambient_capture_event(text)

    assert normalized is not None
    kept_text, tier, cue_tag = normalized
    assert tier == "semantic"
    assert cue_tag == "memoire-durable-iae"
    assert kept_text == (
        "Mémoire durable pour IAE :\n"
        "- [decision] /journalise garde Obsidian comme journal humain.\n"
        "- [piege] IAE ne capture pas le journal complet."
    )


def test_durable_memory_nothing_to_capture_dropped():
    normalized = _normalize_ambient_capture_event(
        "Mémoire durable pour IAE : rien à capturer."
    )
    assert normalized is None


def test_durable_memory_block_caps_at_eight_bullets():
    bullets = "\n".join(f"- [etat-courant] point {i}" for i in range(10))
    normalized = _normalize_ambient_capture_event(
        f"Memoire durable IAE :\n{bullets}"
    )

    assert normalized is not None
    kept_text, tier, _cue_tag = normalized
    assert tier == "semantic"
    assert kept_text.count("\n- ") == 8
    assert "point 8" not in kept_text

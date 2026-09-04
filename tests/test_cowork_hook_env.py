"""Runs the real packaged hook scripts as subprocesses under a redirected
CLAUDE_CONFIG_DIR and asserts on the observable Claude-side path-resolution
outcome (which transcript file the script actually read, or the log line
it wrote) -- never on an internal script variable. Scripts are resolved
through importlib.resources against the installed package, the same way
the plugin materializer resolves them, never by a repository-relative
path."""
from __future__ import annotations

import importlib.resources as _res
import json
import subprocess
from pathlib import Path

_TURN_CAPTURE = "iai-mcp-turn-capture.sh"
_SESSION_CAPTURE = "iai-mcp-session-capture.sh"
_SESSION_RECALL = "iai-mcp-session-recall.sh"
_PER_TURN_RECALL = "iai-mcp-per-turn-recall.sh"

_SHARED_MARKER = "FROM_SHARED_HOME_TRANSCRIPT"
_ALT_MARKER = "FROM_ALT_CONFIG_DIR_TRANSCRIPT"


def _script_path(name: str) -> Path:
    src = _res.files("iai_mcp") / "_deploy" / "hooks" / name
    assert src.exists(), f"hook script missing from installed package: {src}"
    return Path(str(src))


def _base_env(home: Path, *, plugin_root: str = None) -> dict:
    env = {"HOME": str(home), "PATH": "/usr/bin:/bin"}
    if plugin_root is not None:
        env["CLAUDE_PLUGIN_ROOT"] = plugin_root
    return env


def _field_lines(text: str) -> list:
    """Lines the script actually appended to its log, excluding the bare
    "---" separator line, which carries no key=value fields."""
    return [ln for ln in text.splitlines() if ln.strip() and ln.strip() != "---"]


def _write_jsonl(path: Path, lines: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj) + "\n")


def _turn_line(marker: str) -> dict:
    return {"type": "user", "message": {"role": "user", "content": f"turn text {marker}"}}


def _run(
    script_name: str, home: Path, payload: dict, *, config_dir=None, plugin_root=None,
) -> subprocess.CompletedProcess:
    env = _base_env(home, plugin_root=plugin_root)
    if config_dir is not None:
        env["CLAUDE_CONFIG_DIR"] = config_dir
    return subprocess.run(
        ["bash", str(_script_path(script_name))],
        input=json.dumps(payload), capture_output=True, text=True, env=env, timeout=20,
    )


def _live_text(home: Path, session_id: str) -> str:
    live = home / ".iai-mcp" / ".deferred-captures" / f"{session_id}.live.jsonl"
    return live.read_text(encoding="utf-8") if live.exists() else ""


def _capture_log_text(home: Path) -> str:
    logs_dir = home / ".iai-mcp" / "logs"
    if not logs_dir.is_dir():
        return ""
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(logs_dir.glob("capture-*.log")))


def _turn_capture_log_text(home: Path) -> str:
    logs_dir = home / ".iai-mcp" / "logs"
    if not logs_dir.is_dir():
        return ""
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(logs_dir.glob("turn-capture-*.log")))


def _recall_log_text(home: Path) -> str:
    logs_dir = home / ".iai-mcp" / "logs"
    if not logs_dir.is_dir():
        return ""
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(logs_dir.glob("recall-*.log")))


def _foresight_ledger_text(home: Path) -> str:
    ledger = home / ".iai-mcp" / "logs" / "foresight-served.jsonl"
    return ledger.read_text(encoding="utf-8") if ledger.exists() else ""


def _run_per_turn_recall(home: Path, payload: dict, *, plugin_root=None) -> subprocess.CompletedProcess:
    env = _base_env(home, plugin_root=plugin_root)
    env["IAI_MCP_PER_TURN_SOCKET_ACCEL"] = "0"
    return subprocess.run(
        ["bash", str(_script_path(_PER_TURN_RECALL))],
        input=json.dumps(payload), capture_output=True, text=True, env=env, timeout=20,
    )


# --- turn-capture.sh: observable outcome is which transcript's lines
# reached the live spool. ---


def test_turn_capture_config_dir_unset_uses_shared_home(tmp_path):
    home = tmp_path / "home"
    sid = "sess-unset"
    _write_jsonl(
        home / ".claude" / "projects" / "proj" / f"{sid}.jsonl",
        [_turn_line(_SHARED_MARKER)],
    )
    proc = _run(
        _TURN_CAPTURE, home,
        {"session_id": sid, "transcript_path": str(home / "nonexistent-stdin.jsonl"), "prompt_id": ""},
    )
    assert proc.returncode == 0
    assert _SHARED_MARKER in _live_text(home, sid)


def test_turn_capture_config_dir_equal_to_shared_home_is_identical(tmp_path):
    home = tmp_path / "home"
    sid = "sess-equal"
    _write_jsonl(
        home / ".claude" / "projects" / "proj" / f"{sid}.jsonl",
        [_turn_line(_SHARED_MARKER)],
    )
    proc = _run(
        _TURN_CAPTURE, home,
        {"session_id": sid, "transcript_path": str(home / "nonexistent-stdin.jsonl"), "prompt_id": ""},
        config_dir=str(home / ".claude"),
    )
    assert proc.returncode == 0
    assert _SHARED_MARKER in _live_text(home, sid)


def test_turn_capture_config_dir_redirect_uses_alt_transcript(tmp_path):
    home = tmp_path / "home"
    alt = tmp_path / "alt-config"
    sid = "sess-redirect"
    _write_jsonl(
        alt / "projects" / "proj" / f"{sid}.jsonl",
        [_turn_line(_ALT_MARKER)],
    )
    (alt).mkdir(parents=True, exist_ok=True)
    proc = _run(
        _TURN_CAPTURE, home,
        {"session_id": sid, "transcript_path": str(home / "nonexistent-stdin.jsonl"), "prompt_id": ""},
        config_dir=str(alt),
    )
    assert proc.returncode == 0
    live = _live_text(home, sid)
    assert _ALT_MARKER in live
    assert _SHARED_MARKER not in live


def test_turn_capture_config_dir_with_space_uses_alt_transcript(tmp_path):
    # A real Cowork agent-session config dir lives under "Application
    # Support" (a space in the path); the charset guard must accept it.
    home = tmp_path / "home"
    alt = tmp_path / "Application Support" / "alt-config"
    sid = "sess-space"
    _write_jsonl(alt / "projects" / "proj" / f"{sid}.jsonl", [_turn_line(_ALT_MARKER)])
    proc = _run(
        _TURN_CAPTURE, home,
        {"session_id": sid, "transcript_path": str(home / "nonexistent-stdin.jsonl"), "prompt_id": ""},
        config_dir=str(alt),
    )
    assert proc.returncode == 0
    live = _live_text(home, sid)
    assert _ALT_MARKER in live
    assert "config-dir-refused" not in _turn_capture_log_text(home)


def test_turn_capture_config_dir_nonexistent_falls_back_and_exits_zero(tmp_path):
    home = tmp_path / "home"
    sid = "sess-missing-dir"
    _write_jsonl(
        home / ".claude" / "projects" / "proj" / f"{sid}.jsonl",
        [_turn_line(_SHARED_MARKER)],
    )
    proc = _run(
        _TURN_CAPTURE, home,
        {"session_id": sid, "transcript_path": str(home / "nonexistent-stdin.jsonl"), "prompt_id": ""},
        config_dir=str(home / "does-not-exist"),
    )
    assert proc.returncode == 0
    assert _SHARED_MARKER in _live_text(home, sid)
    assert "config-dir-refused" in _turn_capture_log_text(home)


def test_turn_capture_config_dir_traversal_refused_and_logged(tmp_path):
    home = tmp_path / "home"
    sid = "sess-traversal"
    _write_jsonl(
        home / ".claude" / "projects" / "proj" / f"{sid}.jsonl",
        [_turn_line(_SHARED_MARKER)],
    )
    hostile = str(home / ".claude" / ".." / ".claude")
    proc = _run(
        _TURN_CAPTURE, home,
        {"session_id": sid, "transcript_path": str(home / "nonexistent-stdin.jsonl"), "prompt_id": ""},
        config_dir=hostile,
    )
    assert proc.returncode == 0
    assert _SHARED_MARKER in _live_text(home, sid)
    assert "config-dir-refused" in _turn_capture_log_text(home)


def test_turn_capture_config_dir_metacharacter_refused_and_logged(tmp_path):
    home = tmp_path / "home"
    sid = "sess-metachar"
    _write_jsonl(
        home / ".claude" / "projects" / "proj" / f"{sid}.jsonl",
        [_turn_line(_SHARED_MARKER)],
    )
    proc = _run(
        _TURN_CAPTURE, home,
        {"session_id": sid, "transcript_path": str(home / "nonexistent-stdin.jsonl"), "prompt_id": ""},
        config_dir="$(touch " + str(tmp_path / "pwned") + ")",
    )
    assert proc.returncode == 0
    assert not (tmp_path / "pwned").exists()
    assert _SHARED_MARKER in _live_text(home, sid)
    assert "config-dir-refused" in _turn_capture_log_text(home)


def test_turn_capture_iai_side_paths_never_move_with_config_dir(tmp_path):
    home = tmp_path / "home"
    alt = tmp_path / "alt-config"
    sid = "sess-iai-side"
    _write_jsonl(alt / "projects" / "proj" / f"{sid}.jsonl", [_turn_line(_ALT_MARKER)])
    proc = _run(
        _TURN_CAPTURE, home,
        {"session_id": sid, "transcript_path": str(home / "nonexistent-stdin.jsonl"), "prompt_id": ""},
        config_dir=str(alt),
    )
    assert proc.returncode == 0
    assert (home / ".iai-mcp" / ".deferred-captures" / f"{sid}.live.jsonl").exists()
    assert not (alt / ".iai-mcp").exists()


# --- session-capture.sh: observable outcome is the transcript= field on
# its header log line (the CLI itself is not installed in this harness, so
# the run degrades to "iai-mcp CLI not found" after the log line fires). ---


def test_session_capture_config_dir_unset_uses_shared_home(tmp_path):
    home = tmp_path / "home"
    sid = "sc-unset"
    (home / ".claude" / "projects" / "proj").mkdir(parents=True)
    (home / ".claude" / "projects" / "proj" / f"{sid}.jsonl").write_text("{}\n")
    proc = _run(_SESSION_CAPTURE, home, {"session_id": sid, "cwd": "/tmp"})
    assert proc.returncode == 0
    log = _capture_log_text(home)
    assert f"session={sid}" in log
    assert "/.claude/projects/proj/" in log


def test_session_capture_config_dir_redirect_uses_alt_projects(tmp_path):
    home = tmp_path / "home"
    alt = tmp_path / "alt-config"
    sid = "sc-redirect"
    (alt / "projects" / "proj").mkdir(parents=True)
    (alt / "projects" / "proj" / f"{sid}.jsonl").write_text("{}\n")
    proc = _run(_SESSION_CAPTURE, home, {"session_id": sid, "cwd": "/tmp"}, config_dir=str(alt))
    assert proc.returncode == 0
    log = _capture_log_text(home)
    assert f"session={sid}" in log
    assert str(alt) in log
    assert str(home / ".claude") not in log


def test_session_capture_config_dir_with_space_uses_alt_projects(tmp_path):
    # A real Cowork agent-session config dir lives under "Application
    # Support" (a space in the path); the charset guard must accept it.
    home = tmp_path / "home"
    alt = tmp_path / "Application Support" / "alt-config"
    sid = "sc-space"
    (alt / "projects" / "proj").mkdir(parents=True)
    (alt / "projects" / "proj" / f"{sid}.jsonl").write_text("{}\n")
    proc = _run(_SESSION_CAPTURE, home, {"session_id": sid, "cwd": "/tmp"}, config_dir=str(alt))
    assert proc.returncode == 0
    log = _capture_log_text(home)
    assert f"session={sid}" in log
    assert str(alt) in log
    assert "config-dir-refused" not in log


def test_session_capture_config_dir_traversal_refused_and_logged(tmp_path):
    home = tmp_path / "home"
    sid = "sc-traversal"
    (home / ".claude" / "projects" / "proj").mkdir(parents=True)
    (home / ".claude" / "projects" / "proj" / f"{sid}.jsonl").write_text("{}\n")
    hostile = str(home / ".claude" / ".." / ".claude")
    proc = _run(_SESSION_CAPTURE, home, {"session_id": sid, "cwd": "/tmp"}, config_dir=hostile)
    assert proc.returncode == 0
    log = _capture_log_text(home)
    assert "/.claude/projects/proj/" in log
    assert "config-dir-refused" in log


# --- Channel discriminator: every line a script appends to its daily log
# carries channel=plugin when CLAUDE_PLUGIN_ROOT is set, channel=settings
# otherwise, on every path including the early-exit and refusal lines, and
# without disturbing any pre-existing field. ---


def test_turn_capture_channel_is_plugin_when_plugin_root_set(tmp_path):
    home = tmp_path / "home"
    sid = "tc-chan-plugin"
    _write_jsonl(home / ".claude" / "projects" / "proj" / f"{sid}.jsonl", [_turn_line(_SHARED_MARKER)])
    proc = _run(
        _TURN_CAPTURE, home,
        {"session_id": sid, "transcript_path": str(home / "nope.jsonl"), "prompt_id": ""},
        plugin_root=str(tmp_path / "plugin-root"),
    )
    assert proc.returncode == 0
    for line in _field_lines(_turn_capture_log_text(home)):
        assert "channel=plugin" in line, line


def test_turn_capture_channel_is_settings_when_plugin_root_unset(tmp_path):
    home = tmp_path / "home"
    sid = "tc-chan-settings"
    _write_jsonl(home / ".claude" / "projects" / "proj" / f"{sid}.jsonl", [_turn_line(_SHARED_MARKER)])
    proc = _run(
        _TURN_CAPTURE, home,
        {"session_id": sid, "transcript_path": str(home / "nope.jsonl"), "prompt_id": ""},
    )
    assert proc.returncode == 0
    for line in _field_lines(_turn_capture_log_text(home)):
        assert "channel=settings" in line, line


def test_turn_capture_channel_present_on_invalid_session_id_early_exit(tmp_path):
    home = tmp_path / "home"
    proc = _run(_TURN_CAPTURE, home, {"session_id": "bad id!", "transcript_path": "/nope", "prompt_id": ""})
    assert proc.returncode == 0
    lines = _field_lines(_turn_capture_log_text(home))
    assert len(lines) == 1
    assert "skipped: invalid session_id" in lines[0]
    assert "channel=settings" in lines[0]


def test_turn_capture_every_appended_line_carries_channel(tmp_path):
    # The traversal-refusal scenario appends TWO lines in one run (the
    # refusal, then the final session/rc line) -- both must carry it.
    home = tmp_path / "home"
    sid = "tc-chan-all"
    _write_jsonl(home / ".claude" / "projects" / "proj" / f"{sid}.jsonl", [_turn_line(_SHARED_MARKER)])
    hostile = str(home / ".claude" / ".." / ".claude")
    proc = _run(
        _TURN_CAPTURE, home,
        {"session_id": sid, "transcript_path": str(home / "nope.jsonl"), "prompt_id": ""},
        config_dir=hostile,
    )
    assert proc.returncode == 0
    lines = _field_lines(_turn_capture_log_text(home))
    assert len(lines) == 2
    assert all("channel=" in ln for ln in lines)


def test_turn_capture_grammar_stability(tmp_path):
    # Frozen field tokens captured from the pre-channel-discriminator output
    # of this same script (session=, rc= on the normal line; the two
    # skip-reason strings on the early-exit lines). The channel field must
    # join them, never replace or reorder them.
    home = tmp_path / "home"
    sid = "tc-grammar"
    _write_jsonl(home / ".claude" / "projects" / "proj" / f"{sid}.jsonl", [_turn_line(_SHARED_MARKER)])
    proc = _run(
        _TURN_CAPTURE, home,
        {"session_id": sid, "transcript_path": str(home / "nope.jsonl"), "prompt_id": ""},
    )
    assert proc.returncode == 0
    normal_line = _field_lines(_turn_capture_log_text(home))[0]
    assert f"session={sid}" in normal_line
    assert "rc=" in normal_line
    assert "channel=" in normal_line
    assert normal_line.index(f"session={sid}") < normal_line.index("rc=")

    home2 = tmp_path / "home2"
    proc2 = _run(_TURN_CAPTURE, home2, {"transcript_path": ""})
    missing_line = _field_lines(_turn_capture_log_text(home2))[0]
    assert "skipped: missing session_id or transcript_path" in missing_line
    assert "channel=" in missing_line

    home3 = tmp_path / "home3"
    proc3 = _run(_TURN_CAPTURE, home3, {"session_id": "bad id!", "transcript_path": "/nope"})
    invalid_line = _field_lines(_turn_capture_log_text(home3))[0]
    assert "skipped: invalid session_id" in invalid_line
    assert "channel=" in invalid_line


def test_session_capture_channel_is_plugin_when_plugin_root_set(tmp_path):
    home = tmp_path / "home"
    sid = "sc-chan-plugin"
    (home / ".claude" / "projects" / "proj").mkdir(parents=True)
    (home / ".claude" / "projects" / "proj" / f"{sid}.jsonl").write_text("{}\n")
    proc = _run(
        _SESSION_CAPTURE, home, {"session_id": sid, "cwd": "/tmp"},
        plugin_root=str(tmp_path / "plugin-root"),
    )
    assert proc.returncode == 0
    for line in _field_lines(_capture_log_text(home)):
        assert "channel=plugin" in line, line


def test_session_capture_channel_is_settings_when_plugin_root_unset(tmp_path):
    home = tmp_path / "home"
    sid = "sc-chan-settings"
    (home / ".claude" / "projects" / "proj").mkdir(parents=True)
    (home / ".claude" / "projects" / "proj" / f"{sid}.jsonl").write_text("{}\n")
    proc = _run(_SESSION_CAPTURE, home, {"session_id": sid, "cwd": "/tmp"})
    assert proc.returncode == 0
    for line in _field_lines(_capture_log_text(home)):
        assert "channel=settings" in line, line


def test_session_capture_every_appended_line_carries_channel(tmp_path):
    home = tmp_path / "home"
    sid = "sc-chan-all"
    (home / ".claude" / "projects" / "proj").mkdir(parents=True)
    (home / ".claude" / "projects" / "proj" / f"{sid}.jsonl").write_text("{}\n")
    hostile = str(home / ".claude" / ".." / ".claude")
    proc = _run(_SESSION_CAPTURE, home, {"session_id": sid, "cwd": "/tmp"}, config_dir=hostile)
    assert proc.returncode == 0
    lines = _field_lines(_capture_log_text(home))
    assert len(lines) == 3
    assert all("channel=" in ln for ln in lines)


def test_session_capture_grammar_stability(tmp_path):
    home = tmp_path / "home"
    sid = "sc-grammar"
    (home / ".claude" / "projects" / "proj").mkdir(parents=True)
    (home / ".claude" / "projects" / "proj" / f"{sid}.jsonl").write_text("{}\n")
    proc = _run(_SESSION_CAPTURE, home, {"session_id": sid, "cwd": "/tmp"})
    assert proc.returncode == 0
    lines = _field_lines(_capture_log_text(home))
    header, no_cli = lines[0], lines[1]
    assert f"session={sid}" in header
    assert "cwd=/tmp" in header
    assert "transcript=" in header
    assert "channel=" in header
    assert header.index(f"session={sid}") < header.index("cwd=/tmp") < header.index("transcript=")
    assert "skipped: iai-mcp CLI not found" in no_cli
    assert "channel=" in no_cli


def test_session_recall_channel_is_plugin_when_plugin_root_set(tmp_path):
    home = tmp_path / "home"
    proc = _run(
        _SESSION_RECALL, home, {"session_id": "sr-chan-plugin", "source": "startup"},
        plugin_root=str(tmp_path / "plugin-root"),
    )
    assert proc.returncode == 0
    for line in _field_lines(_recall_log_text(home)):
        assert "channel=plugin" in line, line


def test_session_recall_channel_is_settings_when_plugin_root_unset(tmp_path):
    home = tmp_path / "home"
    proc = _run(_SESSION_RECALL, home, {"session_id": "sr-chan-settings", "source": "startup"})
    assert proc.returncode == 0
    for line in _field_lines(_recall_log_text(home)):
        assert "channel=settings" in line, line


def test_session_recall_grammar_stability(tmp_path):
    home = tmp_path / "home"
    sid = "sr-grammar"
    proc = _run(_SESSION_RECALL, home, {"session_id": sid, "source": "startup"})
    assert proc.returncode == 0
    lines = _field_lines(_recall_log_text(home))
    header = lines[0]
    assert f"session={sid}" in header
    assert "source=startup" in header
    assert "channel=" in header
    assert header.index(f"session={sid}") < header.index("source=startup")
    assert any("cache-miss absent" in ln and "channel=" in ln for ln in lines)
    assert any("skipped: iai-mcp CLI not found" in ln and "channel=" in ln for ln in lines)


def test_per_turn_recall_ledger_carries_channel(tmp_path):
    home = tmp_path / "home"
    (home / ".iai-mcp").mkdir(parents=True)
    (home / ".iai-mcp" / ".next-turn-pack.cached.md").write_text("anticipated content here")
    proc = _run_per_turn_recall(home, {"session_id": "pt-chan-settings", "prompt": "hi"})
    assert proc.returncode == 0
    ledger = _foresight_ledger_text(home)
    assert '"channel":"settings"' in ledger


def test_per_turn_recall_ledger_channel_is_plugin_when_plugin_root_set(tmp_path):
    home = tmp_path / "home"
    (home / ".iai-mcp").mkdir(parents=True)
    (home / ".iai-mcp" / ".next-turn-pack.cached.md").write_text("anticipated content here")
    proc = _run_per_turn_recall(
        home, {"session_id": "pt-chan-plugin", "prompt": "hi"}, plugin_root=str(tmp_path / "plugin-root"),
    )
    assert proc.returncode == 0
    ledger = _foresight_ledger_text(home)
    assert '"channel":"plugin"' in ledger


def test_channel_env_var_value_never_written_to_a_log(tmp_path):
    # CLAUDE_PLUGIN_ROOT names an absolute path on the user's machine; only
    # the fixed "plugin"/"settings" literal belongs in a log line.
    home = tmp_path / "home"
    sid = "sc-no-leak"
    (home / ".claude" / "projects" / "proj").mkdir(parents=True)
    (home / ".claude" / "projects" / "proj" / f"{sid}.jsonl").write_text("{}\n")
    secret_root = str(tmp_path / "very-specific-plugin-root-path-marker")
    proc = _run(_SESSION_CAPTURE, home, {"session_id": sid, "cwd": "/tmp"}, plugin_root=secret_root)
    assert proc.returncode == 0
    assert secret_root not in _capture_log_text(home)

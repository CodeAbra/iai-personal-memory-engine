from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "lint_agent_prompts.py"


def _load_lint_module(tmp_path: Path):
    spec = importlib.util.spec_from_file_location(
        f"lint_agent_prompts_{tmp_path.name}", SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def lint_in_tmp(tmp_path, monkeypatch):
    agents_dir = tmp_path / ".claude" / "agents"
    agents_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    mod = _load_lint_module(tmp_path)
    return mod, agents_dir


def _write_clean_agent(path: Path, name: str = "sample") -> None:
    path.write_text(
        f"---\n"
        f"name: {name}\n"
        f"description: A clean agent.\n"
        f"tools: Read, Write, Bash\n"
        f"---\n\n"
        f"Body here.\n\n"
        f"**MCP tools — MANDATORY:** Use context-mode MCP "
        f"(`ctx_search`, `ctx_batch_execute`) for code search and "
        f"mempalace MCP for memory. Do NOT use Grep or Glob.\n",
        encoding="utf-8",
    )


def test_no_agents_dir_returns_zero(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mod = _load_lint_module(tmp_path)
    assert mod.main([]) == 0


def test_clean_agents_pass(lint_in_tmp):
    mod, agents_dir = lint_in_tmp
    _write_clean_agent(agents_dir / "a.md", name="a")
    _write_clean_agent(agents_dir / "b.md", name="b")
    assert mod.main([]) == 0


def test_grep_in_tools_line_fails(lint_in_tmp):
    mod, agents_dir = lint_in_tmp
    (agents_dir / "bad.md").write_text(
        "---\n"
        "name: bad\n"
        "description: violates the rule.\n"
        "tools: Read, Grep, Bash\n"
        "---\n\n"
        "Body.\n",
        encoding="utf-8",
    )
    assert mod.main([]) == 1


def test_glob_in_tools_line_fails(lint_in_tmp):
    mod, agents_dir = lint_in_tmp
    (agents_dir / "bad.md").write_text(
        "---\n"
        "name: bad\n"
        "description: violates the rule.\n"
        "tools: Read, Bash, Glob\n"
        "---\n\n"
        "Body.\n",
        encoding="utf-8",
    )
    assert mod.main([]) == 1


def test_inline_grep_call_in_body_fails(lint_in_tmp):
    mod, agents_dir = lint_in_tmp
    (agents_dir / "bad.md").write_text(
        "---\n"
        "name: bad\n"
        "description: example uses inline Grep.\n"
        "tools: Read\n"
        "---\n\n"
        "Then do `Grep(\"class.*Foo\")`.\n",
        encoding="utf-8",
    )
    assert mod.main([]) == 1


def test_inline_glob_call_in_body_fails(lint_in_tmp):
    mod, agents_dir = lint_in_tmp
    (agents_dir / "bad.md").write_text(
        "---\n"
        "name: bad\n"
        "description: example uses inline Glob.\n"
        "tools: Read\n"
        "---\n\n"
        "Then `Glob(\"src/**/*.py\")`.\n",
        encoding="utf-8",
    )
    assert mod.main([]) == 1


def test_directive_text_is_not_a_violation(lint_in_tmp):
    mod, agents_dir = lint_in_tmp
    (agents_dir / "ok.md").write_text(
        "---\n"
        "name: ok\n"
        "description: directive present in body.\n"
        "tools: Read\n"
        "---\n\n"
        "**MCP tools — MANDATORY:** Use context-mode MCP "
        "(`ctx_search`, `ctx_batch_execute`) for code search and "
        "mempalace MCP for memory. Do NOT use Grep or Glob.\n",
        encoding="utf-8",
    )
    assert mod.main([]) == 0


def test_word_boundary_avoids_false_positives(lint_in_tmp):
    mod, agents_dir = lint_in_tmp
    (agents_dir / "ok.md").write_text(
        "---\n"
        "name: ok\n"
        "description: tokens like GreppyBar or GlobetrotterTool must not match.\n"
        "tools: Read, GreppyBar, GlobetrotterTool, Bash\n"
        "---\n\n"
        "Body mentions GreppyBar and GlobetrotterTool but no inline calls.\n",
        encoding="utf-8",
    )
    assert mod.main([]) == 0

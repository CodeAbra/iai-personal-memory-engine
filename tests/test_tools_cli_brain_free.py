"""Guard: the shelled `iai_mcp.tools_cli` path the tools MCP server invokes
never loads a brain module (Hippo, lilli, the brain native extension).

This is the Pitfall-1 guarantee the server's code_search, session_search,
and stats backing paths all rely on.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

BRAIN_MODULES = (
    "iai_mcp.hippo",
    "iai_mcp.lilli",
    "iai_mcp.store",
    "iai_mcp.cli",
    "iai_mcp_native",
)


def _run_probe(probe: str, env: dict, stdin: "str | None" = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(REPO),
        env=env,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_import_tools_cli_loads_no_brain_module(tmp_path: Path) -> None:
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["HOME"] = str(tmp_path)

    probe = (
        "import sys, json;"
        "import iai_mcp.tools_cli;"
        f"brain = [m for m in {BRAIN_MODULES!r} if m in sys.modules];"
        "print(json.dumps({'brain_modules': brain}))"
    )
    proc = _run_probe(probe, env)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["brain_modules"] == [], (
        f"importing iai_mcp.tools_cli must load no brain module, found: {result['brain_modules']}"
    )


def test_dispatch_search_loads_no_brain_module(tmp_path: Path) -> None:
    import os

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "only.py").write_text("def marker_token_xyz():\n    pass\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["HOME"] = str(tmp_path)

    index_proc = subprocess.run(
        [sys.executable, "-m", "iai_mcp.tools_cli", "index", str(corpus)],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert index_proc.returncode == 0, index_proc.stderr

    probe = (
        "import sys, json;"
        "import iai_mcp.tools_cli as tools_cli;"
        "rc = tools_cli.main(['search', 'marker_token_xyz']);"
        f"brain = [m for m in {BRAIN_MODULES!r} if m in sys.modules];"
        "print(json.dumps({'rc': rc, 'brain_modules': brain}))"
    )
    proc = _run_probe(probe, env)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["rc"] == 0
    assert result["brain_modules"] == [], (
        f"dispatching search must load no brain module, found: {result['brain_modules']}"
    )


def test_dispatch_search_json_loads_no_brain_module(tmp_path: Path) -> None:
    import os

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "only.py").write_text("def marker_token_xyz():\n    pass\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["HOME"] = str(tmp_path)

    index_proc = subprocess.run(
        [sys.executable, "-m", "iai_mcp.tools_cli", "index", str(corpus)],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert index_proc.returncode == 0, index_proc.stderr

    probe = (
        "import sys, json;"
        "import iai_mcp.tools_cli as tools_cli;"
        "rc = tools_cli.main(['search', 'marker_token_xyz', '--json']);"
        f"brain = [m for m in {BRAIN_MODULES!r} if m in sys.modules];"
        "print(json.dumps({'rc': rc, 'brain_modules': brain}))"
    )
    proc = _run_probe(probe, env)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["rc"] == 0
    assert result["brain_modules"] == [], (
        f"dispatching search --json must load no brain module, found: {result['brain_modules']}"
    )


def test_dispatch_session_search_loads_no_brain_module(tmp_path: Path) -> None:
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["HOME"] = str(tmp_path)

    capture_proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from iai_mcp.tools.session import capture_event;"
            "capture_event({'tool_name': 'Edit', 'file_path': 'src/guard.py', "
            "'session_id': 'sess-guard', 'source_hook': 'PostToolUse'})",
        ],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert capture_proc.returncode == 0, capture_proc.stderr

    probe = (
        "import sys, json;"
        "import iai_mcp.tools_cli as tools_cli;"
        "rc = tools_cli.main(['session-search', 'sess-guard']);"
        f"brain = [m for m in {BRAIN_MODULES!r} if m in sys.modules];"
        "print(json.dumps({'rc': rc, 'brain_modules': brain}))"
    )
    proc = _run_probe(probe, env)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["rc"] == 0
    assert result["brain_modules"] == [], (
        f"dispatching session-search must load no brain module, found: {result['brain_modules']}"
    )


def test_dispatch_capture_loads_no_brain_module(tmp_path: Path) -> None:
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["HOME"] = str(tmp_path)

    payload = (
        '{"tool_name": "Edit", "tool_input": {"file_path": "src/guard.py"}, '
        '"session_id": "sess-guard-capture"}'
    )

    probe = (
        "import sys, json;"
        "import iai_mcp.tools_cli as tools_cli;"
        "rc = tools_cli.main(['capture', '--source-hook', 'PostToolUse']);"
        f"brain = [m for m in {BRAIN_MODULES!r} if m in sys.modules];"
        "print(json.dumps({'rc': rc, 'brain_modules': brain}))"
    )
    proc = _run_probe(probe, env, stdin=payload)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["rc"] == 0
    assert result["brain_modules"] == [], (
        f"dispatching capture must load no brain module, found: {result['brain_modules']}"
    )


def test_dispatch_stats_loads_no_brain_module(tmp_path: Path) -> None:
    import os

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "only.py").write_text("def marker_token_xyz():\n    pass\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["HOME"] = str(tmp_path)

    index_proc = subprocess.run(
        [sys.executable, "-m", "iai_mcp.tools_cli", "index", str(corpus)],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert index_proc.returncode == 0, index_proc.stderr

    probe = (
        "import sys, json;"
        "import iai_mcp.tools_cli as tools_cli;"
        "rc = tools_cli.main(['stats']);"
        f"brain = [m for m in {BRAIN_MODULES!r} if m in sys.modules];"
        "print(json.dumps({'rc': rc, 'brain_modules': brain}))"
    )
    proc = _run_probe(probe, env)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["rc"] == 0
    assert result["brain_modules"] == [], (
        f"dispatching stats must load no brain module, found: {result['brain_modules']}"
    )


def test_index_admits_planning_tree_and_stays_bounded(tmp_path: Path) -> None:
    """A fresh index makes a token living only under `.planning/` findable
    via search-literal, while the superseded-milestone archive, a junk
    build directory, and a non-allowlisted dot-directory stay invisible --
    and the walker's own file count, the store's scanned count, and an
    in-process walker call over the same corpus all agree on one set."""
    import os

    corpus = tmp_path / "corpus"
    planning = corpus / ".planning"
    planning.mkdir(parents=True)
    (planning / "roadmap.md").write_text("planning_only_marker_alpha", encoding="utf-8")
    archive = planning / "milestones"
    archive.mkdir()
    (archive / "old.md").write_text("milestone_archive_marker_beta", encoding="utf-8")
    src_dir = corpus / "src"
    src_dir.mkdir()
    (src_dir / "real.md").write_text("source_real_marker_gamma", encoding="utf-8")
    junk_dir = corpus / "node_modules"
    junk_dir.mkdir()
    (junk_dir / "pkg.md").write_text("node_modules_junk_marker_delta", encoding="utf-8")
    other_hidden = corpus / ".other_hidden"
    other_hidden.mkdir()
    (other_hidden / "h.md").write_text("other_hidden_marker_epsilon", encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["HOME"] = str(tmp_path)

    index_proc = subprocess.run(
        [sys.executable, "-m", "iai_mcp.tools_cli", "index", str(corpus)],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert index_proc.returncode == 0, index_proc.stderr
    match = re.search(r"indexed (\d+) files", index_proc.stdout)
    assert match, index_proc.stdout
    n_indexed = int(match.group(1))

    def _search_literal(token: str) -> dict:
        proc = subprocess.run(
            [sys.executable, "-m", "iai_mcp.tools_cli", "search-literal", token, "--json"],
            cwd=str(REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout.strip().splitlines()[-1])

    planning_payload = _search_literal("planning_only_marker_alpha")
    assert planning_payload["status"] == "hits", planning_payload

    source_payload = _search_literal("source_real_marker_gamma")
    assert source_payload["status"] == "hits", source_payload

    archive_payload = _search_literal("milestone_archive_marker_beta")
    assert archive_payload["status"] == "absent", archive_payload

    junk_payload = _search_literal("node_modules_junk_marker_delta")
    assert junk_payload["status"] == "absent", junk_payload

    hidden_payload = _search_literal("other_hidden_marker_epsilon")
    assert hidden_payload["status"] == "absent", hidden_payload

    files_scanned = source_payload["files_scanned"]

    from iai_mcp.study import TOOLS_INDEX_DOT_DIR_ALLOWLIST, iter_study_files

    walker_count = len(
        iter_study_files(
            corpus, dot_dir_allowlist=TOOLS_INDEX_DOT_DIR_ALLOWLIST, max_files=sys.maxsize,
        )
    )
    assert n_indexed == files_scanned == walker_count, (n_indexed, files_scanned, walker_count)

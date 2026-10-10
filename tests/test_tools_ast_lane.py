"""End-to-end coverage for the Python AST index lane: its own on-disk
segment/manifest store, the `index_all` 4th-step publish, and
go-to-definition -- the `goto-def` `code_search` lane and its
`iai-tools search-goto-def` CLI counterpart, both rendering off the SAME
`goto_definition` glue body.

`code_search`/`iai-tools` resolve their default store under
`Path.home()`, bound at IMPORT time -- an in-process `store_root=`
override cannot reach that path. Every test here runs the real CLI (or
the built MCP server) as a subprocess with `HOME` pointed at a fresh
temp directory, mirroring tests/test_tools_xref_reverse_query.py's
hermetic pattern.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
WRAPPER = REPO / "mcp-wrapper"


# --------------------------------------------------------------------------
# CLI subprocess helpers (mirrors test_tools_xref_reverse_query.py)
# --------------------------------------------------------------------------


def _cli_env(home_dir: Path) -> dict:
    env = os.environ.copy()
    env["HOME"] = str(home_dir)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run_cli(home_dir: Path, *args: str) -> subprocess.CompletedProcess:
    env = _cli_env(home_dir)
    return subprocess.run(
        [sys.executable, "-m", "iai_mcp.tools_cli", *args],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _run_cli_json(home_dir: Path, *args: str) -> dict:
    proc = _run_cli(home_dir, *args, "--json")
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def _index_all_lanes(home_dir: Path, corpus: Path) -> None:
    proc = _run_cli(home_dir, "index", str(corpus))
    assert proc.returncode == 0, proc.stderr


def _new_home_and_corpus(tmp_path: Path) -> "tuple[Path, Path]":
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    return home_dir, corpus


# --------------------------------------------------------------------------
# MCP server subprocess helpers (mirrors test_tools_xref_reverse_query.py)
# --------------------------------------------------------------------------


def _wrapper_dist() -> Path:
    return WRAPPER / "dist" / "tools-index.js"


@pytest.fixture(scope="module")
def built_wrapper() -> Path:
    if not (WRAPPER / "node_modules").exists():
        subprocess.run(["npm", "install"], cwd=WRAPPER, check=True)
    subprocess.run(["npm", "run", "build"], cwd=WRAPPER, check=True)
    dist = _wrapper_dist()
    assert dist.exists(), "npm run build should have produced dist/tools-index.js"
    return dist


def _spawn_server(home_dir: Path, nonexistent_sock: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["IAI_MCP_PYTHON"] = sys.executable
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["HOME"] = str(home_dir)
    env["IAI_TOOLS_SOCKET_PATH"] = str(nonexistent_sock)
    return subprocess.Popen(
        ["node", str(_wrapper_dist())],
        cwd=str(REPO),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _send_rpc(proc: subprocess.Popen, method: str, params: dict, rpc_id: int) -> None:
    req = {"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params}
    assert proc.stdin is not None
    proc.stdin.write((json.dumps(req) + "\n").encode())
    proc.stdin.flush()


def _send_notification(proc: subprocess.Popen, method: str) -> None:
    note = {"jsonrpc": "2.0", "method": method}
    assert proc.stdin is not None
    proc.stdin.write((json.dumps(note) + "\n").encode())
    proc.stdin.flush()


def _read_response_with_id(proc: subprocess.Popen, rpc_id: int, timeout_sec: float) -> dict:
    assert proc.stdout is not None
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            stderr_tail = b""
            try:
                stderr_tail = proc.stderr.read() if proc.stderr else b""
            except Exception:
                pass
            raise RuntimeError(
                f"server closed stdout before responding to id={rpc_id}; "
                f"stderr tail: {stderr_tail.decode(errors='replace')[-2000:]}"
            )
        line_str = line.decode(errors="replace").strip()
        if not line_str:
            continue
        try:
            msg = json.loads(line_str)
        except json.JSONDecodeError:
            continue
        if isinstance(msg, dict) and msg.get("id") == rpc_id:
            return msg
    raise TimeoutError(f"server did not respond to id={rpc_id} within {timeout_sec}s")


def _initialize(proc: subprocess.Popen) -> None:
    _send_rpc(
        proc,
        "initialize",
        {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "iai-tools-ast-lane-test", "version": "0.1.0"},
        },
        1,
    )
    init_resp = _read_response_with_id(proc, 1, timeout_sec=5.0)
    assert "result" in init_resp, f"initialize failed: {init_resp}"
    _send_notification(proc, "notifications/initialized")


def _call_tool(proc: subprocess.Popen, name: str, arguments: dict, rpc_id: int) -> dict:
    _send_rpc(proc, "tools/call", {"name": name, "arguments": arguments}, rpc_id)
    resp = _read_response_with_id(proc, rpc_id, timeout_sec=20.0)
    assert "result" in resp, f"tools/call {name} error: {resp}"
    result = resp["result"]
    assert not result.get("isError"), f"{name} returned an error: {result}"
    return result.get("structuredContent") or json.loads(result["content"][0]["text"])


def _terminate(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


# --------------------------------------------------------------------------
# Same-file go-to-def, resolved, round-tripped across a second, separate
# CLI process (proves persistence, not an in-memory shortcut).
# --------------------------------------------------------------------------


def test_goto_def_resolves_a_same_file_definition_end_to_end(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    target_file = corpus / "target_file.py"
    target_file.write_text(
        "def target_symbol():\n    return 1\nresult = target_symbol()\n"
    )
    _index_all_lanes(home_dir, corpus)

    payload = _run_cli_json(
        home_dir, "search-goto-def", "target_symbol", "--from-path", str(target_file)
    )
    assert payload["query"] == "target_symbol"
    assert payload["lane"] == "goto-def"
    assert payload["status"] == "resolved"
    assert len(payload["hits"]) == 1
    assert payload["hits"][0]["path"] == str(target_file.resolve())
    assert payload["hits"][0]["line"] == 1


def test_goto_def_never_resolves_a_case_mismatched_query_end_to_end(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    target_file = corpus / "target_file.py"
    target_file.write_text("def Foo():\n    return 1\nfoo()\n")
    _index_all_lanes(home_dir, corpus)

    mismatched = _run_cli_json(
        home_dir, "search-goto-def", "foo", "--from-path", str(target_file)
    )
    assert mismatched["status"] == "unresolved", (
        "a query for \"foo\" must never resolve against a def named \"Foo\" -- Python "
        f"identifiers are case-sensitive: {mismatched}"
    )
    assert mismatched["hits"] == []

    exact = _run_cli_json(
        home_dir, "search-goto-def", "Foo", "--from-path", str(target_file)
    )
    assert exact["status"] == "resolved"
    assert exact["hits"][0]["line"] == 1


def test_empty_symbol_is_rejected(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    target_file = corpus / "alpha.py"
    target_file.write_text("def zebracornrejectguard():\n    return 1\n")
    _index_all_lanes(home_dir, corpus)

    proc = _run_cli(home_dir, "search-goto-def", "", "--from-path", str(target_file))
    assert proc.returncode != 0
    assert "must not be empty" in proc.stderr
    assert proc.stdout == ""


def test_mcp_code_search_goto_def_lane_resolves_a_same_file_definition(
    built_wrapper: Path, tmp_path: Path
) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    target_file = corpus / "target_file.py"
    target_file.write_text(
        "def target_symbol():\n    return 1\nresult = target_symbol()\n"
    )

    nonexistent_sock = tmp_path / "iai-tools-goto-def-lane.sock"
    proc = _spawn_server(home_dir, nonexistent_sock)
    try:
        _initialize(proc)
        index_structured = _call_tool(proc, "code_index", {"path": str(corpus)}, 2)
        assert index_structured["indexed_files"] == 1

        result = _call_tool(
            proc,
            "code_search",
            {"query": "target_symbol", "lane": "goto-def", "from_path": str(target_file)},
            3,
        )
        assert result["status"] == "resolved"
        assert result["hits"][0]["path"] == str(target_file.resolve())
        assert result["hits"][0]["line"] == 1
    finally:
        _terminate(proc)


def test_mcp_code_search_from_path_rejected_outside_goto_def_lane(
    built_wrapper: Path, tmp_path: Path
) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    (corpus / "alpha.py").write_text("def zebracornfrompathreject():\n    return 1\n")

    nonexistent_sock = tmp_path / "iai-tools-frompath-reject.sock"
    proc = _spawn_server(home_dir, nonexistent_sock)
    try:
        _initialize(proc)
        _call_tool(proc, "code_index", {"path": str(corpus)}, 2)

        _send_rpc(
            proc,
            "tools/call",
            {
                "name": "code_search",
                "arguments": {"query": "zebracornfrompathreject", "lane": "whole", "from_path": "x"},
            },
            3,
        )
        resp = _read_response_with_id(proc, 3, timeout_sec=20.0)
        assert "result" in resp
        assert resp["result"].get("isError") is True
    finally:
        _terminate(proc)


def test_mcp_code_search_goto_def_lane_requires_from_path(
    built_wrapper: Path, tmp_path: Path
) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    (corpus / "alpha.py").write_text("def zebracornmissingfrompath():\n    return 1\n")

    nonexistent_sock = tmp_path / "iai-tools-goto-def-missing-frompath.sock"
    proc = _spawn_server(home_dir, nonexistent_sock)
    try:
        _initialize(proc)
        _call_tool(proc, "code_index", {"path": str(corpus)}, 2)

        _send_rpc(
            proc,
            "tools/call",
            {"name": "code_search", "arguments": {"query": "zebracornmissingfrompath", "lane": "goto-def"}},
            3,
        )
        resp = _read_response_with_id(proc, 3, timeout_sec=20.0)
        assert "result" in resp
        assert resp["result"].get("isError") is True
    finally:
        _terminate(proc)


# --------------------------------------------------------------------------
# The AST lane as a full 4th lockstep lane: coexistence, cross-file-via-
# import resolution, broken-file degradation, and the honest status axis.
# --------------------------------------------------------------------------


def test_index_all_builds_a_4th_ast_store_alongside_whole_symbol_xref(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    (corpus / "alpha.py").write_text("def zebracorncoexist():\n    return 1\n")
    _index_all_lanes(home_dir, corpus)

    tools_root = home_dir / ".iai-mcp" / "tools"
    assert tools_root.exists()
    assert (tools_root / "symbols").exists()
    assert (tools_root / "xref").exists()
    assert (tools_root / "ast").exists(), "the AST lane must publish its own store root alongside the other three"
    assert any((tools_root / "ast").iterdir()), "the AST store root must contain at least a manifest file"


def test_goto_def_resolves_cross_file_via_an_already_resolved_import(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    def_file = corpus / "def_file.py"
    def_file.write_text("def cross_file_target():\n    return 1\n")
    importer_file = corpus / "importer_file.py"
    importer_file.write_text(
        "from def_file import cross_file_target\nresult = cross_file_target()\n"
    )
    _index_all_lanes(home_dir, corpus)

    payload = _run_cli_json(
        home_dir, "search-goto-def", "cross_file_target", "--from-path", str(importer_file)
    )
    assert payload["status"] == "resolved"
    assert payload["hits"][0]["path"] == str(def_file.resolve())
    assert payload["hits"][0]["line"] == 1

    # Own-file definition still wins first.
    own_file_payload = _run_cli_json(
        home_dir, "search-goto-def", "cross_file_target", "--from-path", str(def_file)
    )
    assert own_file_payload["status"] == "resolved"
    assert own_file_payload["hits"][0]["path"] == str(def_file.resolve())


def test_a_syntax_broken_file_degrades_without_aborting_the_batch(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    broken_file = corpus / "broken.py"
    broken_file.write_text("def broken(:\n")
    healthy_file = corpus / "healthy.py"
    healthy_file.write_text("def zebracornhealthy():\n    return 1\n")

    proc = _run_cli(home_dir, "index", str(corpus))
    assert proc.returncode == 0, proc.stderr
    assert "indexed 2 files" in proc.stdout

    healthy_payload = _run_cli_json(
        home_dir, "search-goto-def", "zebracornhealthy", "--from-path", str(healthy_file)
    )
    assert healthy_payload["status"] == "resolved"

    broken_payload = _run_cli_json(
        home_dir, "search-goto-def", "anything", "--from-path", str(broken_file)
    )
    assert broken_payload["status"] == "parse_error"


def test_goto_def_status_unsupported_language_for_a_non_python_target(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    (corpus / "alpha.py").write_text("def zebracornunsupported():\n    return 1\n")
    rust_file = corpus / "other.rs"
    rust_file.write_text("fn main() {}\n")
    _index_all_lanes(home_dir, corpus)

    payload = _run_cli_json(
        home_dir, "search-goto-def", "zebracornunsupported", "--from-path", str(rust_file)
    )
    assert payload["status"] == "unsupported_language"
    assert payload["hits"] == []


def test_goto_def_status_not_indexed_for_a_python_file_never_indexed(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    (corpus / "alpha.py").write_text("def zebracornnotindexed():\n    return 1\n")
    _index_all_lanes(home_dir, corpus)

    never_indexed = corpus / "never_indexed.py"
    never_indexed.write_text("def something():\n    return 1\n")

    payload = _run_cli_json(
        home_dir, "search-goto-def", "zebracornnotindexed", "--from-path", str(never_indexed)
    )
    assert payload["status"] == "not_indexed"
    assert payload["hits"] == []


def test_goto_def_status_unresolved_for_a_name_never_referenced(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    target_file = corpus / "alpha.py"
    target_file.write_text("x = 1\n")
    _index_all_lanes(home_dir, corpus)

    payload = _run_cli_json(
        home_dir, "search-goto-def", "a_name_never_referenced_zzz", "--from-path", str(target_file)
    )
    assert payload["status"] == "unresolved"
    assert payload["hits"] == []


def test_index_all_skips_ast_parsing_for_the_command_output_store(tmp_path: Path) -> None:
    sys.path.insert(0, str(REPO / "src"))
    from iai_mcp.tools import index_all, is_command_output_store_root

    command_output_root = tmp_path / "tools-output"
    assert is_command_output_store_root(command_output_root)
    py_file = tmp_path / "captured_output.py"
    py_file.write_text("def zebracorncommandoutput():\n    return 1\n")

    ast_root = tmp_path / "tools-output-ast"
    index_all([str(py_file)], store_root=command_output_root, ast_store_root=ast_root)

    assert not ast_root.exists() or not any(ast_root.iterdir())
    assert not (ast_root / "manifest.laman").exists(), (
        "command output must never be AST-parsed into the source AST store"
    )


def test_ast_lane_converges_after_a_crash_between_the_rust_and_ast_publish(tmp_path: Path) -> None:
    # Explicit store roots throughout -- never the `default_*_store_root()`
    # accessors, whose `_DEFAULT_IAI_ROOT` binds to the REAL `Path.home()`
    # at module-import time and cannot be redirected by mutating
    # `os.environ["HOME"]` from inside an already-running process.
    sys.path.insert(0, str(REPO / "src"))
    from iai_mcp.tools import goto_definition, index_all
    import iai_mcp_tools_core as _native

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    target_file = corpus / "target_file.py"
    target_file.write_text("def zebracornconverge():\n    return 1\n")

    root = tmp_path / "tools"
    symbol_root = tmp_path / "tools-symbols"
    xref_root = tmp_path / "tools-xref"
    ast_root = tmp_path / "tools-ast"
    root.mkdir(parents=True, exist_ok=True)
    symbol_root.mkdir(parents=True, exist_ok=True)
    xref_root.mkdir(parents=True, exist_ok=True)

    # Simulates a crash AFTER the Rust 3-lane index_all returns but BEFORE
    # the Python-side AST step runs -- the AST lane is left one batch
    # behind, but loadable (there is simply no AST store yet). A go-to-def
    # query against it must report not_indexed, never raise.
    _native.index_all(str(root), str(symbol_root), str(xref_root), [str(target_file)])
    assert not ast_root.exists() or not any(ast_root.iterdir())
    result = goto_definition("zebracornconverge", from_path=str(target_file), ast_store_root=ast_root)
    assert result["status"] == "not_indexed"

    # The next full index_all call, given the SAME explicit roots,
    # converges the AST lane.
    index_all(
        [str(target_file)],
        store_root=root,
        symbol_store_root=symbol_root,
        xref_store_root=xref_root,
        ast_store_root=ast_root,
    )
    converged = goto_definition("zebracornconverge", from_path=str(target_file), ast_store_root=ast_root)
    assert converged["status"] == "resolved"


# --------------------------------------------------------------------------
# stale-path pre-gate: a re-index re-parses only a changed file; a cold
# index parses every file; the stale set comes solely from the AST
# manifest's own file_index content hashes.
# --------------------------------------------------------------------------


def test_stale_paths_gate_reindex_reparses_only_the_changed_file(tmp_path: Path) -> None:
    sys.path.insert(0, str(REPO / "src"))
    import iai_mcp_tools_core as _native
    from iai_mcp.tools import index_all

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    a = corpus / "a.py"
    a.write_text("def a_def():\n    return 1\n")
    b = corpus / "b.py"
    b.write_text("def b_def():\n    return 2\n")

    root = tmp_path / "tools"
    symbol_root = tmp_path / "tools-symbols"
    xref_root = tmp_path / "tools-xref"
    ast_root = tmp_path / "tools-ast"

    index_all(
        [str(a), str(b)],
        store_root=root,
        symbol_store_root=symbol_root,
        xref_store_root=xref_root,
        ast_store_root=ast_root,
    )
    n_files_before, gen_before = _native.ast_manifest_stats(str(ast_root))
    assert n_files_before == 2

    # Change only `a.py`; `b.py` is byte-identical.
    a.write_text("def a_def():\n    return 999\n")

    stale = _native.stale_paths(str(ast_root), [str(a.resolve()), str(b.resolve())])
    assert stale == [str(a.resolve())], (
        f"only the changed file must be reported stale, unchanged siblings must never re-parse, got {stale}"
    )

    index_all(
        [str(a), str(b)],
        store_root=root,
        symbol_store_root=symbol_root,
        xref_store_root=xref_root,
        ast_store_root=ast_root,
    )
    n_files_after, gen_after = _native.ast_manifest_stats(str(ast_root))
    assert n_files_after == 2
    assert gen_after > gen_before, "re-indexing the genuinely changed file must bump generation"


def test_cold_index_of_a_fresh_ast_store_treats_every_file_as_stale(tmp_path: Path) -> None:
    sys.path.insert(0, str(REPO / "src"))
    import iai_mcp_tools_core as _native

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    a = corpus / "a.py"
    a.write_text("x = 1\n")
    b = corpus / "b.py"
    b.write_text("y = 2\n")
    ast_root = tmp_path / "tools-ast"

    stale = _native.stale_paths(str(ast_root), [str(a.resolve()), str(b.resolve())])
    assert set(stale) == {str(a.resolve()), str(b.resolve())}, (
        "a fresh AST store's empty file_index must report every path stale"
    )


# --------------------------------------------------------------------------
# index_all's AST step degrades one bad file rather than aborting the
# whole batch -- both a re-read failure (file removed/truncated between
# the earlier decodability check and here) and a native publish failure.
# --------------------------------------------------------------------------


def test_index_all_degrades_a_file_that_disappears_before_the_ast_reread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sys.path.insert(0, str(REPO / "src"))
    from pathlib import Path as _Path

    from iai_mcp.tools import goto_definition, index_all

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    good = corpus / "good.py"
    good.write_text("def good_def():\n    return 1\n")
    victim = corpus / "victim.py"
    victim.write_text("def victim_def():\n    return 1\n")

    root = tmp_path / "tools"
    symbol_root = tmp_path / "tools-symbols"
    xref_root = tmp_path / "tools-xref"
    ast_root = tmp_path / "tools-ast"

    real_read_text = _Path.read_text

    def _flaky_read_text(self: _Path, *args, **kwargs):
        if self.name == "victim.py":
            raise OSError("simulated disappearance between decode check and AST re-read")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(_Path, "read_text", _flaky_read_text)

    count = index_all(
        [str(good), str(victim)],
        store_root=root,
        symbol_store_root=symbol_root,
        xref_store_root=xref_root,
        ast_store_root=ast_root,
    )
    assert count == 2, "both files stay counted as decodable -- the AST step's own failure is separate"

    victim_result = goto_definition("victim_def", from_path=str(victim), ast_store_root=ast_root)
    assert victim_result["status"] == "parse_error", (
        "a file whose AST re-read fails must degrade like a parse failure, never abort the batch"
    )

    good_result = goto_definition("good_def", from_path=str(good), ast_store_root=ast_root)
    assert good_result["status"] == "resolved", (
        "a sibling file must still index and resolve normally despite the victim's read failure"
    )


def test_index_all_degrades_a_file_whose_ast_reread_hits_invalid_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sys.path.insert(0, str(REPO / "src"))
    from pathlib import Path as _Path

    from iai_mcp.tools import goto_definition, index_all

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    good = corpus / "good.py"
    good.write_text("def good_def():\n    return 1\n")
    victim = corpus / "victim.py"
    victim.write_text("def victim_def():\n    return 1\n")

    root = tmp_path / "tools"
    symbol_root = tmp_path / "tools-symbols"
    xref_root = tmp_path / "tools-xref"
    ast_root = tmp_path / "tools-ast"

    real_read_text = _Path.read_text

    def _truncated_mid_write_read_text(self: _Path, *args, **kwargs):
        if self.name == "victim.py":
            raise UnicodeDecodeError(
                "utf-8", b"\xc3\x28", 0, 1, "simulated truncated multi-byte sequence"
            )
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(_Path, "read_text", _truncated_mid_write_read_text)

    count = index_all(
        [str(good), str(victim)],
        store_root=root,
        symbol_store_root=symbol_root,
        xref_store_root=xref_root,
        ast_store_root=ast_root,
    )
    assert count == 2, "both files stay counted as decodable -- the AST step's own failure is separate"

    victim_result = goto_definition("victim_def", from_path=str(victim), ast_store_root=ast_root)
    assert victim_result["status"] == "parse_error", (
        "a file whose AST re-read hits an invalid utf-8 sequence mid-write must degrade like a "
        "parse failure, never abort the batch"
    )

    good_result = goto_definition("good_def", from_path=str(good), ast_store_root=ast_root)
    assert good_result["status"] == "resolved", (
        "a sibling file must still index and resolve normally despite the victim's decode failure"
    )


def test_index_all_survives_a_native_ast_publish_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sys.path.insert(0, str(REPO / "src"))
    import iai_mcp_tools_core as _native
    from iai_mcp.tools import index_all

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    only = corpus / "only.py"
    only.write_text("def only_def():\n    return 1\n")

    root = tmp_path / "tools"
    symbol_root = tmp_path / "tools-symbols"
    xref_root = tmp_path / "tools-xref"
    ast_root = tmp_path / "tools-ast"

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated native ast publish failure")

    monkeypatch.setattr(_native, "index_ast_facts", _boom)

    count = index_all(
        [str(only)],
        store_root=root,
        symbol_store_root=symbol_root,
        xref_store_root=xref_root,
        ast_store_root=ast_root,
    )
    assert count == 1, (
        "a native AST publish failure must not propagate out of index_all -- the whole-file/"
        "symbol/xref lanes it runs after already succeeded"
    )


# --------------------------------------------------------------------------
# Source hygiene: this plan's own touched source must carry no
# planning-process identifier (phase/plan/task/wave/decision code, internal
# doc ref, "Constitutional" framing) in any comment/string/name. Native
# grep/rg/find are hard-blocked -- implemented with plain Python string/regex
# matching instead. Filters out this guard's own defining lines rather than
# gating a raw match count at zero, so the pattern's own definition never
# self-trips the scan.
# --------------------------------------------------------------------------

_PLANNING_MARKER_RE = re.compile(
    r"\bCODE-\d+\b"
    r"|\bPhase\s+\d+\b"
    r"|\bPlan\s+\d+\b"
    r"|\bTask\s+\d+\b"
    r"|\bWave\s+\d+\b"
    r"|\bD-AST-[A-Z]+\b"
    r"|\.planning/"
    r"|\bConstitutional\b",
    re.IGNORECASE,
)

_TOUCHED_SOURCE_FILES = [
    REPO / "rust" / "iai_mcp_tools_core" / "src" / "ast_segment.rs",
    REPO / "rust" / "iai_mcp_tools_core" / "src" / "ast_index.rs",
    REPO / "rust" / "iai_mcp_tools_core" / "src" / "lib.rs",
    REPO / "src" / "iai_mcp" / "tools" / "ast_facts.py",
    REPO / "src" / "iai_mcp" / "tools" / "__init__.py",
    REPO / "src" / "iai_mcp" / "tools_cli.py",
    REPO / "mcp-wrapper" / "src" / "tools-index.ts",
]


def test_touched_source_carries_no_planning_process_identifiers() -> None:
    # Scoped to the production surfaces this plan modified -- NOT this test
    # module itself, whose own meta-commentary about the check necessarily
    # names the very substrings the check looks for.
    offenders: list[str] = []
    for path in _TOUCHED_SOURCE_FILES:
        text = path.read_text()
        for i, line in enumerate(text.splitlines(), 1):
            if _PLANNING_MARKER_RE.search(line):
                offenders.append(f"{path}:{i}: {line.strip()}")
    assert not offenders, "planning-process identifier(s) leaked into source:\n" + "\n".join(offenders)

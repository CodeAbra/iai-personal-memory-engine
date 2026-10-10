"""End-to-end coverage for the find-references surface (the reverse of
go-to-definition): the `references` `code_search` lane and its
`iai-tools search-references` CLI counterpart, both rendering off the
SAME `find_references` glue body. Also carries the two new lanes'
shared honest-status regression coverage, the tool-count/schema
non-regression lock, and this phase's own source-hygiene guard.

`code_search`/`iai-tools` resolve their default store under
`Path.home()`, bound at IMPORT time -- an in-process `store_root=`
override cannot reach that path. Every test here runs the real CLI (or
the built MCP server) as a subprocess with `HOME` pointed at a fresh
temp directory, mirroring tests/test_tools_xref_reverse_query.py's and
tests/test_tools_ast_lane.py's hermetic pattern.
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
# CLI subprocess helpers (mirrors test_tools_ast_lane.py)
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
# MCP server subprocess helpers (mirrors test_tools_ast_lane.py)
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
            "clientInfo": {"name": "iai-tools-find-refs-test", "version": "0.1.0"},
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


def _list_tools(proc: subprocess.Popen, rpc_id: int) -> dict:
    _send_rpc(proc, "tools/list", {}, rpc_id)
    resp = _read_response_with_id(proc, rpc_id, timeout_sec=10.0)
    assert "result" in resp, f"tools/list error: {resp}"
    return resp["result"]


def _terminate(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


# --------------------------------------------------------------------------
# resolved/unresolved rows, star-import, dynamic-import, end to end.
# --------------------------------------------------------------------------


def test_references_returns_resolved_and_unresolved_rows_end_to_end(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    def_file = corpus / "def_file.py"
    def_file.write_text("def target_symbol():\n    return 1\n")
    importer_file = corpus / "importer_file.py"
    importer_file.write_text("from def_file import target_symbol\ntarget_symbol()\n")
    bare_file = corpus / "bare_file.py"
    bare_file.write_text("target_symbol()\n")
    _index_all_lanes(home_dir, corpus)

    payload = _run_cli_json(
        home_dir, "search-references", "target_symbol", "--def-path", str(def_file)
    )
    assert payload["query"] == "target_symbol"
    assert payload["lane"] == "references"
    assert payload["status"] == "hits"
    assert len(payload["hits"]) == 2, f"expected exactly two reference rows: {payload['hits']}"

    by_path = {Path(hit["path"]).name: hit["status"] for hit in payload["hits"]}
    assert by_path["importer_file.py"] == "resolved"
    assert by_path["bare_file.py"] == "unresolved"


def test_star_import_reference_is_tagged_unresolved(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    def_file = corpus / "def_file.py"
    def_file.write_text("def target_symbol():\n    return 1\n")
    star_file = corpus / "star_file.py"
    star_file.write_text("from def_file import *\ntarget_symbol()\n")
    _index_all_lanes(home_dir, corpus)

    payload = _run_cli_json(
        home_dir, "search-references", "target_symbol", "--def-path", str(def_file)
    )
    assert payload["status"] == "hits"
    star_hits = [hit for hit in payload["hits"] if hit["path"].endswith("star_file.py")]
    assert len(star_hits) == 1, f"expected exactly one row for the wildcard-import caller: {payload['hits']}"
    assert star_hits[0]["status"] == "unresolved", (
        "a wildcard import's exports cannot be enumerated -- its call site must stay unresolved, "
        f"never resolved: {payload['hits']}"
    )


def test_dynamic_import_produces_no_reference_row(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    def_file = corpus / "def_file.py"
    def_file.write_text("def target_symbol():\n    return 1\n")
    dynamic_file = corpus / "dynamic_file.py"
    dynamic_file.write_text(
        "import importlib\n"
        "importlib.import_module('target_symbol')\n"
        "__import__('target_symbol')\n"
    )
    _index_all_lanes(home_dir, corpus)

    payload = _run_cli_json(
        home_dir, "search-references", "target_symbol", "--def-path", str(def_file)
    )
    assert payload["status"] == "no_references", (
        "the AST pass has no import-shaped node naming the dynamically-imported "
        f"string, so it must produce zero rows, never a false positive: {payload}"
    )
    assert payload["hits"] == []
    assert not any(hit["path"].endswith("dynamic_file.py") for hit in payload["hits"])


def test_references_never_matches_a_case_mismatched_query_end_to_end(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    def_file = corpus / "def_file.py"
    def_file.write_text("def Foo():\n    return 1\nfoo()\n")
    _index_all_lanes(home_dir, corpus)

    mismatched = _run_cli_json(
        home_dir, "search-references", "foo", "--def-path", str(def_file)
    )
    assert mismatched["status"] == "no_references", (
        "querying \"foo\" must not match a def named \"Foo\" -- Python identifiers are "
        f"case-sensitive: {mismatched}"
    )
    assert mismatched["hits"] == []

    exact = _run_cli_json(
        home_dir, "search-references", "Foo", "--def-path", str(def_file)
    )
    assert exact["status"] == "no_references", (
        "the exact-case query \"Foo\" has no matching call site (the only call is "
        f"lowercase \"foo\"): {exact}"
    )


def test_empty_symbol_is_rejected(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    def_file = corpus / "alpha.py"
    def_file.write_text("def zebracornrejectguard():\n    return 1\n")
    _index_all_lanes(home_dir, corpus)

    proc = _run_cli(home_dir, "search-references", "", "--def-path", str(def_file))
    assert proc.returncode != 0
    assert "must not be empty" in proc.stderr
    assert proc.stdout == ""


def test_def_path_is_required(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    def_file = corpus / "alpha.py"
    def_file.write_text("def zebracornrequiredguard():\n    return 1\n")
    _index_all_lanes(home_dir, corpus)

    proc = _run_cli(home_dir, "search-references", "zebracornrequiredguard")
    assert proc.returncode != 0
    assert proc.stdout == ""


def test_mcp_code_search_references_lane_resolves_end_to_end(
    built_wrapper: Path, tmp_path: Path
) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    def_file = corpus / "def_file.py"
    def_file.write_text("def target_symbol():\n    return 1\n")
    importer_file = corpus / "importer_file.py"
    importer_file.write_text("from def_file import target_symbol\ntarget_symbol()\n")

    nonexistent_sock = tmp_path / "iai-tools-references-lane.sock"
    proc = _spawn_server(home_dir, nonexistent_sock)
    try:
        _initialize(proc)
        index_structured = _call_tool(proc, "code_index", {"path": str(corpus)}, 2)
        assert index_structured["indexed_files"] == 2

        result = _call_tool(
            proc,
            "code_search",
            {"query": "target_symbol", "lane": "references", "def_path": str(def_file)},
            3,
        )
        assert result["status"] == "hits"
        assert result["hits"][0]["path"].endswith("importer_file.py")
        assert result["hits"][0]["status"] == "resolved"
    finally:
        _terminate(proc)


def test_mcp_code_search_def_path_rejected_outside_importers_and_references_lanes(
    built_wrapper: Path, tmp_path: Path
) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    (corpus / "alpha.py").write_text("def zebracorndefpathreject2():\n    return 1\n")

    nonexistent_sock = tmp_path / "iai-tools-references-defpath-reject.sock"
    proc = _spawn_server(home_dir, nonexistent_sock)
    try:
        _initialize(proc)
        _call_tool(proc, "code_index", {"path": str(corpus)}, 2)

        _send_rpc(
            proc,
            "tools/call",
            {
                "name": "code_search",
                "arguments": {"query": "zebracorndefpathreject2", "lane": "whole", "def_path": "x"},
            },
            3,
        )
        resp = _read_response_with_id(proc, 3, timeout_sec=20.0)
        assert "result" in resp
        assert resp["result"].get("isError") is True
    finally:
        _terminate(proc)


def test_mcp_code_search_references_lane_requires_def_path(
    built_wrapper: Path, tmp_path: Path
) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    (corpus / "alpha.py").write_text("def zebracornmissingdefpath():\n    return 1\n")

    nonexistent_sock = tmp_path / "iai-tools-references-missing-defpath.sock"
    proc = _spawn_server(home_dir, nonexistent_sock)
    try:
        _initialize(proc)
        _call_tool(proc, "code_index", {"path": str(corpus)}, 2)

        _send_rpc(
            proc,
            "tools/call",
            {"name": "code_search", "arguments": {"query": "zebracornmissingdefpath", "lane": "references"}},
            3,
        )
        resp = _read_response_with_id(proc, 3, timeout_sec=20.0)
        assert "result" in resp
        assert resp["result"].get("isError") is True
    finally:
        _terminate(proc)


# --------------------------------------------------------------------------
# the honest nothing-found status axis, shared by BOTH new lanes.
# --------------------------------------------------------------------------


def test_references_status_unsupported_language_for_a_non_python_def_path(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    (corpus / "alpha.py").write_text("def zebracornrefsunsupported():\n    return 1\n")
    rust_file = corpus / "other.rs"
    rust_file.write_text("fn main() {}\n")
    _index_all_lanes(home_dir, corpus)

    payload = _run_cli_json(
        home_dir, "search-references", "zebracornrefsunsupported", "--def-path", str(rust_file)
    )
    assert payload["status"] == "unsupported_language"
    assert payload["hits"] == []


def test_references_status_not_indexed_for_a_python_def_path_never_indexed(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    (corpus / "alpha.py").write_text("def zebracornrefsnotindexed():\n    return 1\n")
    _index_all_lanes(home_dir, corpus)

    never_indexed = corpus / "never_indexed.py"
    never_indexed.write_text("def something():\n    return 1\n")

    payload = _run_cli_json(
        home_dir, "search-references", "zebracornrefsnotindexed", "--def-path", str(never_indexed)
    )
    assert payload["status"] == "not_indexed"
    assert payload["hits"] == []


def test_references_status_parse_error_for_a_parse_failed_def_path(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    broken_file = corpus / "broken.py"
    broken_file.write_text("def broken(:\n")
    _index_all_lanes(home_dir, corpus)

    payload = _run_cli_json(
        home_dir, "search-references", "anything", "--def-path", str(broken_file)
    )
    assert payload["status"] == "parse_error"
    assert payload["hits"] == []


def test_references_status_no_references_for_an_indexed_parsed_but_never_called_symbol(
    tmp_path: Path,
) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    def_file = corpus / "def_file.py"
    def_file.write_text("def zebracornneverreferenced():\n    return 1\n")
    _index_all_lanes(home_dir, corpus)

    payload = _run_cli_json(
        home_dir, "search-references", "zebracornneverreferenced", "--def-path", str(def_file)
    )
    assert payload["status"] == "no_references"
    assert payload["hits"] == []


def test_goto_def_and_references_agree_on_the_nothing_found_vocabulary(tmp_path: Path) -> None:
    # Both new lanes must report the EXACT SAME status string for each of the
    # three honestly-scoped nothing-found cases, over the SAME fixtures.
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    (corpus / "alpha.py").write_text("def zebracornsharedvocab():\n    return 1\n")
    rust_file = corpus / "other.rs"
    rust_file.write_text("fn main() {}\n")
    broken_file = corpus / "broken.py"
    broken_file.write_text("def broken(:\n")
    _index_all_lanes(home_dir, corpus)
    never_indexed = corpus / "never_indexed.py"
    never_indexed.write_text("def something():\n    return 1\n")

    goto_unsupported = _run_cli_json(
        home_dir, "search-goto-def", "zebracornsharedvocab", "--from-path", str(rust_file)
    )
    refs_unsupported = _run_cli_json(
        home_dir, "search-references", "zebracornsharedvocab", "--def-path", str(rust_file)
    )
    assert goto_unsupported["status"] == refs_unsupported["status"] == "unsupported_language"

    goto_not_indexed = _run_cli_json(
        home_dir, "search-goto-def", "zebracornsharedvocab", "--from-path", str(never_indexed)
    )
    refs_not_indexed = _run_cli_json(
        home_dir, "search-references", "zebracornsharedvocab", "--def-path", str(never_indexed)
    )
    assert goto_not_indexed["status"] == refs_not_indexed["status"] == "not_indexed"

    goto_parse_error = _run_cli_json(
        home_dir, "search-goto-def", "anything", "--from-path", str(broken_file)
    )
    refs_parse_error = _run_cli_json(
        home_dir, "search-references", "anything", "--def-path", str(broken_file)
    )
    assert goto_parse_error["status"] == refs_parse_error["status"] == "parse_error"


def test_no_call_hierarchy_surface_in_cli() -> None:
    sys.path.insert(0, str(REPO / "src"))
    from iai_mcp.tools_cli import _build_parser

    parser = _build_parser()
    texts = [parser.format_help()]
    subparsers_action = next(
        action for action in parser._subparsers._group_actions if hasattr(action, "choices")
    )
    for cmd_name, subparser in subparsers_action.choices.items():
        texts.append(cmd_name)
        texts.append(subparser.format_help())
    combined = "\n".join(texts).lower()
    for forbidden in ("call-hierarchy", "call hierarchy", "callhierarchy"):
        assert forbidden not in combined, f"call-hierarchy surface leaked into the CLI: {forbidden!r}"


def test_no_call_hierarchy_surface_in_mcp_schema(built_wrapper: Path, tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    nonexistent_sock = tmp_path / "iai-tools-no-call-hierarchy.sock"
    proc = _spawn_server(home_dir, nonexistent_sock)
    try:
        _initialize(proc)
        result = _list_tools(proc, 2)
        combined = json.dumps(result).lower()
        for forbidden in ("call-hierarchy", "call hierarchy", "callhierarchy"):
            assert forbidden not in combined, f"call-hierarchy surface leaked into the MCP schema: {forbidden!r}"
    finally:
        _terminate(proc)


# --------------------------------------------------------------------------
# non-regression lock -- tool count, existing lanes untouched, the two new
# lanes' OWN response-shape guard (kept OUT of
# test_tools_absence_contract.py's shared {hits, unknown} loop).
# --------------------------------------------------------------------------


EXPECTED_TOOL_NAMES = {
    "code_search",
    "code_index",
    "execute",
    "session_search",
    "stats",
    "command_insight",
    "batch_execute",
}


def test_mcp_tool_set_stays_at_exactly_seven(built_wrapper: Path, tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    nonexistent_sock = tmp_path / "iai-tools-toolcount.sock"
    proc = _spawn_server(home_dir, nonexistent_sock)
    try:
        _initialize(proc)
        result = _list_tools(proc, 2)
        names = {t["name"] for t in result["tools"]}
        assert names == EXPECTED_TOOL_NAMES, (
            f"the MCP tool set must stay pinned at exactly the 7 expected names, got {sorted(names)}"
        )
        assert len(result["tools"]) == 7
    finally:
        _terminate(proc)


def test_code_search_still_accepts_the_pre_existing_whole_lane(
    built_wrapper: Path, tmp_path: Path
) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    (corpus / "alpha.py").write_text("def zebracornwholelanestillworks():\n    return 1\n")

    nonexistent_sock = tmp_path / "iai-tools-whole-lane-still-works.sock"
    proc = _spawn_server(home_dir, nonexistent_sock)
    try:
        _initialize(proc)
        _call_tool(proc, "code_index", {"path": str(corpus)}, 2)
        result = _call_tool(
            proc, "code_search", {"query": "zebracornwholelanestillworks", "lane": "whole"}, 3
        )
        assert result["status"] == "hits"
        assert any(hit["path"].endswith("alpha.py") for hit in result["hits"])
    finally:
        _terminate(proc)


_GOTO_DEF_LANE_ALLOWED_KEYS = {"query", "lane", "from_path", "hits", "status"}
_GOTO_DEF_LANE_REQUIRED_KEYS = {"query", "lane", "from_path", "hits", "status"}
_GOTO_DEF_LANE_STATUSES = {"resolved", "unresolved", "unsupported_language", "not_indexed", "parse_error"}

_REFERENCES_LANE_ALLOWED_KEYS = {"query", "lane", "def_path", "hits", "status", "truncated"}
_REFERENCES_LANE_REQUIRED_KEYS = {"query", "lane", "def_path", "hits", "status"}
_REFERENCES_LANE_STATUSES = {"hits", "no_references", "unsupported_language", "not_indexed", "parse_error"}


def _assert_goto_def_lane_response_shape(payload: dict) -> None:
    extra_keys = set(payload.keys()) - _GOTO_DEF_LANE_ALLOWED_KEYS
    assert not extra_keys, (
        f"goto-def lane response carries an unexpected key {extra_keys}, "
        f"outside the locked allowlist {sorted(_GOTO_DEF_LANE_ALLOWED_KEYS)}: {payload}"
    )
    missing_keys = _GOTO_DEF_LANE_REQUIRED_KEYS - set(payload.keys())
    assert not missing_keys, f"goto-def lane response is missing required key(s) {missing_keys}: {payload}"
    assert payload["status"] in _GOTO_DEF_LANE_STATUSES, (
        f"goto-def lane status must stay within {_GOTO_DEF_LANE_STATUSES}, got {payload['status']!r}"
    )


def _assert_references_lane_response_shape(payload: dict) -> None:
    extra_keys = set(payload.keys()) - _REFERENCES_LANE_ALLOWED_KEYS
    assert not extra_keys, (
        f"references lane response carries an unexpected key {extra_keys}, "
        f"outside the locked allowlist {sorted(_REFERENCES_LANE_ALLOWED_KEYS)}: {payload}"
    )
    missing_keys = _REFERENCES_LANE_REQUIRED_KEYS - set(payload.keys())
    assert not missing_keys, f"references lane response is missing required key(s) {missing_keys}: {payload}"
    assert payload["status"] in _REFERENCES_LANE_STATUSES, (
        f"references lane status must stay within {_REFERENCES_LANE_STATUSES}, got {payload['status']!r}"
    )
    for hit in payload["hits"]:
        assert set(hit.keys()) == {"path", "status"}, f"references lane hit has an unexpected shape: {hit}"
        assert hit["status"] in ("resolved", "unresolved"), f"references lane row status must be resolved/unresolved: {hit}"


def test_goto_def_lane_own_response_shape_guard(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    target_file = corpus / "target_file.py"
    target_file.write_text("def zebracorngotoshapeguard():\n    return 1\n")
    _index_all_lanes(home_dir, corpus)

    resolved = _run_cli_json(
        home_dir, "search-goto-def", "zebracorngotoshapeguard", "--from-path", str(target_file)
    )
    _assert_goto_def_lane_response_shape(resolved)

    unresolved = _run_cli_json(
        home_dir, "search-goto-def", "never_defined_zzz", "--from-path", str(target_file)
    )
    _assert_goto_def_lane_response_shape(unresolved)


def test_references_lane_own_response_shape_guard(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    def_file = corpus / "def_file.py"
    def_file.write_text("def zebracornrefsshapeguard():\n    return 1\n")
    importer_file = corpus / "importer_file.py"
    importer_file.write_text("from def_file import zebracornrefsshapeguard\nzebracornrefsshapeguard()\n")
    _index_all_lanes(home_dir, corpus)

    hits_payload = _run_cli_json(
        home_dir, "search-references", "zebracornrefsshapeguard", "--def-path", str(def_file)
    )
    _assert_references_lane_response_shape(hits_payload)

    no_refs_payload = _run_cli_json(
        home_dir, "search-references", "never_called_zzz", "--def-path", str(def_file)
    )
    _assert_references_lane_response_shape(no_refs_payload)


def test_shared_absence_contract_stays_green_for_pre_existing_lanes() -> None:
    # The shared _TEXT_LANES/_LANE_KEY_ALLOWLISTS loop in
    # test_tools_absence_contract.py must never gain a key for either new
    # lane -- both carry their OWN guard, entirely in this file.
    contract_module = REPO / "tests" / "test_tools_absence_contract.py"
    text = contract_module.read_text()
    assert '"goto-def"' not in text and "'goto-def'" not in text, (
        "the goto-def lane must never be folded into the shared absence contract"
    )
    assert '"references"' not in text and "'references'" not in text, (
        "the references lane must never be folded into the shared absence contract"
    )


# --------------------------------------------------------------------------
# Source hygiene: this phase's own touched production source (Plans 01+02)
# must carry no planning-process identifier (phase/plan/task/wave/decision
# code, internal doc ref, "Constitutional" framing) in any comment/string/
# name. Native grep/rg/find are hard-blocked -- implemented with plain
# Python string/regex matching instead. Filters out this guard's own
# defining lines rather than gating a raw match count at zero, so the
# pattern's own definition never self-trips the scan.
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
    REPO / "rust" / "iai_mcp_tools_core" / "src" / "ast_compaction.rs",
    REPO / "rust" / "iai_mcp_tools_core" / "src" / "lib.rs",
    REPO / "src" / "iai_mcp" / "tools" / "ast_facts.py",
    REPO / "src" / "iai_mcp" / "tools" / "__init__.py",
    REPO / "src" / "iai_mcp" / "tools_cli.py",
    REPO / "mcp-wrapper" / "src" / "tools-index.ts",
]


def test_touched_source_carries_no_planning_process_identifiers() -> None:
    # Scoped to the production surfaces this phase modified (Plans 01+02) --
    # NOT this test module itself, whose own meta-commentary about the check
    # necessarily names the very substrings the check looks for.
    offenders: list[str] = []
    for path in _TOUCHED_SOURCE_FILES:
        text = path.read_text()
        for i, line in enumerate(text.splitlines(), 1):
            if _PLANNING_MARKER_RE.search(line):
                offenders.append(f"{path}:{i}: {line.strip()}")
    assert not offenders, "planning-process identifier(s) leaked into source:\n" + "\n".join(offenders)

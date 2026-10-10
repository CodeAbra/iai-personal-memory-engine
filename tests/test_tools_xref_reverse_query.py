"""End-to-end coverage for the reverse xref query surface ("what files
import/call symbol X"): the `importers` code_search lane and its
`iai-tools search-importers` CLI counterpart, both rendering off the
SAME `importers_for_symbol` glue body.

`code_search`/`iai-tools` resolve their default store under
`Path.home()`, bound at IMPORT time -- an in-process `store_root=`
override cannot reach that path. Every test here runs the real CLI (or
the built MCP server) as a subprocess with `HOME` pointed at a fresh
temp directory, mirroring tests/test_tools_literal_scan.py's hermetic
pattern. The importers lane never embeds, so no Hugging Face cache
passthrough is needed here.
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
# CLI subprocess helpers (mirrors test_tools_literal_scan.py)
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
# MCP server subprocess helpers (mirrors test_tools_literal_scan.py)
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
            "clientInfo": {"name": "iai-tools-reverse-query-test", "version": "0.1.0"},
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
# Present importer, end to end (CLI --json)
# --------------------------------------------------------------------------


def test_present_importer_is_found_end_to_end(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    (corpus / "def_file.py").write_text("def helperzebracornreverse():\n    return 1\n")
    (corpus / "importer_file.py").write_text("from def_file import helperzebracornreverse\n")
    _index_all_lanes(home_dir, corpus)

    payload = _run_cli_json(home_dir, "search-importers", "helperzebracornreverse")
    assert payload["status"] == "hits"
    assert payload["lane"] == "importers"
    assert len(payload["hits"]) == 1
    assert payload["hits"][0]["path"].endswith("importer_file.py")
    assert payload["hits"][0]["kind"] == "import"
    assert payload["scope"]["n_files"] == 2
    assert "unresolved" not in payload
    assert "truncated" not in payload


def test_multiple_importers_are_all_returned(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    (corpus / "def_file.py").write_text("def sharedzebracornreversetarget():\n    return 1\n")
    (corpus / "importer_one.py").write_text("from def_file import sharedzebracornreversetarget\n")
    (corpus / "importer_two.py").write_text("from def_file import sharedzebracornreversetarget\n")
    _index_all_lanes(home_dir, corpus)

    payload = _run_cli_json(home_dir, "search-importers", "sharedzebracornreversetarget")
    assert payload["status"] == "hits"
    paths = {hit["path"] for hit in payload["hits"]}
    assert len(payload["hits"]) == 2
    assert any(p.endswith("importer_one.py") for p in paths)
    assert any(p.endswith("importer_two.py") for p in paths)


def test_empty_symbol_is_rejected(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    (corpus / "alpha.py").write_text("def zebracornrejectguard():\n    return 1\n")
    _index_all_lanes(home_dir, corpus)

    proc = _run_cli(home_dir, "search-importers", "")
    assert proc.returncode != 0
    assert "must not be empty" in proc.stderr
    assert proc.stdout == ""

    proc_json = _run_cli(home_dir, "search-importers", "", "--json")
    assert proc_json.returncode != 0
    assert "must not be empty" in proc_json.stderr
    assert proc_json.stdout == ""


# --------------------------------------------------------------------------
# SC1xSC2 interaction, envelope honesty, def_path, row-count bound
# --------------------------------------------------------------------------


def test_import_and_call_of_same_symbol_yield_two_hits_with_distinct_kind(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    (corpus / "def_file.py").write_text("def zebracornbothshapes():\n    return 1\n")
    (corpus / "importer_file.py").write_text(
        "from def_file import zebracornbothshapes\nresult = zebracornbothshapes()\n"
    )
    _index_all_lanes(home_dir, corpus)

    payload = _run_cli_json(home_dir, "search-importers", "zebracornbothshapes")
    assert payload["status"] == "hits"
    assert len(payload["hits"]) == 2, (
        "a file that both imports AND bare-calls the same symbol must appear "
        f"TWICE with distinct kind, never deduplicated: {payload['hits']}"
    )
    kinds = sorted(hit["kind"] for hit in payload["hits"])
    assert kinds == ["call", "import"]
    assert all(hit["path"].endswith("importer_file.py") for hit in payload["hits"])


def test_no_edge_is_distinguished_from_edge_exists_but_unresolved(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    (corpus / "alpha.py").write_text(
        "from ghost_module_never_indexed import zebracornunresolvedref\n"
    )
    _index_all_lanes(home_dir, corpus)

    no_edge_payload = _run_cli_json(home_dir, "search-importers", "a_symbol_no_reference_at_all_zzz")
    assert no_edge_payload["status"] == "no_importers"
    assert "unresolved" not in no_edge_payload

    unresolved_payload = _run_cli_json(home_dir, "search-importers", "zebracornunresolvedref")
    assert unresolved_payload["status"] == "no_importers"
    assert unresolved_payload["hits"] == []
    assert len(unresolved_payload["unresolved"]) == 1
    assert unresolved_payload["unresolved"][0]["path"].endswith("alpha.py")


def test_def_path_scopes_hits_but_not_unresolved(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    def_a = corpus / "def_a.py"
    def_a.write_text("def zebracorndefpathtarget():\n    return 1\n")
    (corpus / "importer_a.py").write_text("from def_a import zebracorndefpathtarget\n")
    (corpus / "importer_ghost.py").write_text(
        "from ghost_module_never_indexed import zebracorndefpathtarget\n"
    )
    _index_all_lanes(home_dir, corpus)

    payload = _run_cli_json(
        home_dir, "search-importers", "zebracorndefpathtarget", "--def-path", str(def_a.resolve())
    )
    assert payload["status"] == "hits"
    assert payload["def_path"] == str(def_a.resolve())
    assert len(payload["hits"]) == 1
    assert payload["hits"][0]["path"].endswith("importer_a.py")
    assert len(payload["unresolved"]) == 1, (
        "an unresolved reference must still surface even when def_path is set, "
        "since it never resolved to any file and cannot be scoped away"
    )
    assert payload["unresolved"][0]["path"].endswith("importer_ghost.py")


def test_hits_are_bounded_by_k_with_truncation_disclosed(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    (corpus / "def_file.py").write_text("def zebracorntruncationtarget():\n    return 1\n")
    (corpus / "importer_one.py").write_text("from def_file import zebracorntruncationtarget\n")
    (corpus / "importer_two.py").write_text("from def_file import zebracorntruncationtarget\n")
    _index_all_lanes(home_dir, corpus)

    proc = _run_cli(home_dir, "search-importers", "zebracorntruncationtarget", "--k", "1", "--json")
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "hits"
    assert len(payload["hits"]) == 1, "hits must be bounded to k, never returning the unbounded native row set"
    assert payload["truncated"] is True


def test_k_zero_or_negative_is_rejected_not_silently_reported_as_no_importers(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    (corpus / "def_file.py").write_text("def zebracornkzeroguard():\n    return 1\n")
    (corpus / "importer_file.py").write_text("from def_file import zebracornkzeroguard\n")
    _index_all_lanes(home_dir, corpus)

    for bad_k in ("0", "-1"):
        proc = _run_cli(home_dir, "search-importers", "zebracornkzeroguard", "--k", bad_k, "--json")
        assert proc.returncode != 0, (
            f"--k {bad_k} on a symbol with a real resolved importer must error cleanly, "
            f"never silently report no_importers: stdout={proc.stdout!r}"
        )
        assert "--k must be >= 1" in proc.stderr
        assert proc.stdout == "", f"a rejected --k must never emit a JSON payload: stdout={proc.stdout!r}"


def test_def_path_matches_an_equivalent_non_canonical_form(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    def_a = corpus / "def_a.py"
    def_a.write_text("def zebracornnoncanonicaltarget():\n    return 1\n")
    (corpus / "importer_a.py").write_text("from def_a import zebracornnoncanonicaltarget\n")
    _index_all_lanes(home_dir, corpus)

    non_canonical_def_path = str(corpus) + "/./def_a.py"
    assert non_canonical_def_path != str(def_a.resolve()), (
        "fixture must exercise a genuinely non-canonical spelling, not an already-resolved one"
    )

    payload = _run_cli_json(
        home_dir, "search-importers", "zebracornnoncanonicaltarget", "--def-path", non_canonical_def_path
    )
    assert payload["status"] == "hits"
    assert len(payload["hits"]) == 1
    assert payload["hits"][0]["path"].endswith("importer_a.py")


# --------------------------------------------------------------------------
# The importers lane's OWN response-shape guard -- a DISTINCT allowed/
# required split from test_tools_absence_contract.py's shared {hits, unknown}
# loop (def_path/unresolved/truncated are allowed but not required, since
# each is omitted when absent/empty/false). Deliberately NOT folded into
# that shared loop -- see the literal lane's identical precedent.
# --------------------------------------------------------------------------

_IMPORTERS_LANE_ALLOWED_KEYS = {
    "query",
    "lane",
    "def_path",
    "hits",
    "unresolved",
    "truncated",
    "scope",
    "status",
}
_IMPORTERS_LANE_REQUIRED_KEYS = {
    "query",
    "lane",
    "hits",
    "scope",
    "status",
}


def _assert_importers_lane_response_shape(payload: dict) -> None:
    extra_keys = set(payload.keys()) - _IMPORTERS_LANE_ALLOWED_KEYS
    assert not extra_keys, (
        f"importers lane response carries an unexpected key {extra_keys}, "
        f"outside the locked allowlist {sorted(_IMPORTERS_LANE_ALLOWED_KEYS)}: {payload}"
    )
    missing_keys = _IMPORTERS_LANE_REQUIRED_KEYS - set(payload.keys())
    assert not missing_keys, (
        f"importers lane response is missing required key(s) {missing_keys}: {payload}"
    )
    assert payload["status"] in ("hits", "no_importers"), (
        f"importers lane status must stay within {{'hits', 'no_importers'}}, got {payload['status']!r}"
    )


def test_importers_lane_response_shape_guard(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    (corpus / "def_file.py").write_text("def zebracornshapeguard():\n    return 1\n")
    (corpus / "importer_file.py").write_text("from def_file import zebracornshapeguard\n")
    _index_all_lanes(home_dir, corpus)

    hit_payload = _run_cli_json(home_dir, "search-importers", "zebracornshapeguard")
    miss_payload = _run_cli_json(home_dir, "search-importers", "a_symbol_never_referenced_zzz")
    _assert_importers_lane_response_shape(hit_payload)
    _assert_importers_lane_response_shape(miss_payload)
    assert hit_payload["status"] == "hits"
    assert miss_payload["status"] == "no_importers"


# --------------------------------------------------------------------------
# MCP structuredContent
# --------------------------------------------------------------------------


def test_mcp_code_search_importers_lane_present_and_no_importers(
    built_wrapper: Path, tmp_path: Path
) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    (corpus / "def_file.py").write_text("def zebracornmcppresent():\n    return 1\n")
    (corpus / "importer_file.py").write_text("from def_file import zebracornmcppresent\n")

    nonexistent_sock = tmp_path / "iai-tools-importers-lane.sock"
    proc = _spawn_server(home_dir, nonexistent_sock)
    try:
        _initialize(proc)
        index_structured = _call_tool(proc, "code_index", {"path": str(corpus)}, 2)
        assert index_structured["indexed_files"] == 2

        hit_result = _call_tool(
            proc,
            "code_search",
            {"query": "zebracornmcppresent", "lane": "importers", "k": 5},
            3,
        )
        assert hit_result["status"] == "hits"
        assert hit_result["hits"][0]["path"].endswith("importer_file.py")
        assert hit_result["hits"][0]["kind"] == "import"
        assert "scope" in hit_result
        _assert_importers_lane_response_shape(hit_result)

        miss_result = _call_tool(
            proc,
            "code_search",
            {"query": "a_symbol_never_referenced_over_mcp_zzz", "lane": "importers", "k": 5},
            4,
        )
        assert miss_result["status"] == "no_importers"
        assert "scope" in miss_result
        _assert_importers_lane_response_shape(miss_result)
    finally:
        _terminate(proc)


def test_mcp_code_search_importers_lane_def_path_scopes_hits(
    built_wrapper: Path, tmp_path: Path
) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    def_a = corpus / "def_a.py"
    def_a.write_text("def zebracornmcpdefpath():\n    return 1\n")
    (corpus / "importer_a.py").write_text("from def_a import zebracornmcpdefpath\n")
    (corpus / "importer_ghost.py").write_text(
        "from ghost_module_never_indexed import zebracornmcpdefpath\n"
    )

    nonexistent_sock = tmp_path / "iai-tools-importers-defpath.sock"
    proc = _spawn_server(home_dir, nonexistent_sock)
    try:
        _initialize(proc)
        index_structured = _call_tool(proc, "code_index", {"path": str(corpus)}, 2)
        assert index_structured["indexed_files"] == 3

        result = _call_tool(
            proc,
            "code_search",
            {
                "query": "zebracornmcpdefpath",
                "lane": "importers",
                "def_path": str(def_a.resolve()),
                "k": 5,
            },
            3,
        )
        assert result["status"] == "hits"
        assert len(result["hits"]) == 1
        assert result["hits"][0]["path"].endswith("importer_a.py")
        assert len(result["unresolved"]) == 1
        assert result["unresolved"][0]["path"].endswith("importer_ghost.py")
        assert result["def_path"] == str(def_a.resolve())
    finally:
        _terminate(proc)


def test_mcp_code_search_def_path_rejected_outside_importers_lane(
    built_wrapper: Path, tmp_path: Path
) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    (corpus / "alpha.py").write_text("def zebracorndefpathreject():\n    return 1\n")

    nonexistent_sock = tmp_path / "iai-tools-defpath-reject.sock"
    proc = _spawn_server(home_dir, nonexistent_sock)
    try:
        _initialize(proc)
        _call_tool(proc, "code_index", {"path": str(corpus)}, 2)

        _send_rpc(
            proc,
            "tools/call",
            {
                "name": "code_search",
                "arguments": {"query": "zebracorndefpathreject", "lane": "whole", "def_path": "x"},
            },
            3,
        )
        resp = _read_response_with_id(proc, 3, timeout_sec=20.0)
        assert "result" in resp
        assert resp["result"].get("isError") is True
    finally:
        _terminate(proc)


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
    r"\bIMAP-\d+\b"
    r"|\bPhase\s+\d+\b"
    r"|\bPlan\s+\d+\b"
    r"|\bTask\s+\d+\b"
    r"|\bWave\s+\d+\b"
    r"|\.planning/"
    r"|\bConstitutional\b",
    re.IGNORECASE,
)

_TOUCHED_SOURCE_FILES = [
    REPO / "rust" / "iai_mcp_tools_core" / "src" / "xref_index.rs",
    REPO / "rust" / "iai_mcp_tools_core" / "src" / "xref_segment.rs",
    REPO / "rust" / "iai_mcp_tools_core" / "src" / "xref_compaction.rs",
    REPO / "rust" / "iai_mcp_tools_core" / "src" / "lib.rs",
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

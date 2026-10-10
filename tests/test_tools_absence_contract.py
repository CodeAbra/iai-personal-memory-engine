"""End-to-end coverage for the search response's bounded scope/status
envelope: every JSON search response carries a server-computed, live
`scope` (n_files, generation) and a `status` ("hits"/"unknown") so an
empty result is never mistaken for a proven absence of the searched text.

`code_search`/`iai-tools` resolve their default store under
`Path.home()`, bound at IMPORT time -- an in-process `store_root=`
override cannot reach that path. Every test here runs the real CLI (or
the built MCP server) as a subprocess with `HOME` pointed at a fresh
temp directory, so the default-store resolution lands under an isolated,
disposable root rather than the operator's real store.
"""

from __future__ import annotations

import json
import os
import pwd
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
WRAPPER = REPO / "mcp-wrapper"


def _hf_cache_env() -> dict:
    """Points the vector lane's embed-worker subprocess at the REAL,
    already-populated Hugging Face model cache (via the OS user's actual
    home directory, bypassing any sandboxed `HOME` override) instead of
    the tmp `HOME` this module otherwise uses for store isolation -- a
    sandboxed `HOME` with no override here would trigger a network
    download of the model weights on first use, which hermetic tests
    must never do.
    """
    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    real_cache = real_home / ".cache" / "huggingface"
    return {
        "HF_HOME": str(real_cache),
        "HF_HUB_CACHE": str(real_cache / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(real_cache / "hub"),
    }


def _hf_cache_available() -> bool:
    """True when the operator's real Hugging Face cache already holds the
    embedder's model weights -- mirrors `HAS_BGE_SMALL_CACHE` in
    tests/test_embedder_numeric_parity.py. Vector-lane tests here spawn a
    real embed-worker subprocess pointed at `_hf_cache_env()`'s real
    cache; with no cache present that subprocess falls back to a live
    network download instead of a clean skip.
    """
    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    real_cache = real_home / ".cache" / "huggingface"
    return real_cache.exists() and any(real_cache.rglob("*bge-small-en*"))


HAS_BGE_SMALL_CACHE = _hf_cache_available()

requires_hf_cache = pytest.mark.skipif(
    not HAS_BGE_SMALL_CACHE,
    reason="real Hugging Face model cache not present; vector-lane tests require a pre-warmed cache",
)


# --------------------------------------------------------------------------
# CLI subprocess helpers
# --------------------------------------------------------------------------


def _cli_env(home_dir: Path) -> dict:
    env = os.environ.copy()
    env["HOME"] = str(home_dir)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run_cli(home_dir: Path, *args: str, extra_env: "dict | None" = None) -> subprocess.CompletedProcess:
    env = _cli_env(home_dir)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-m", "iai_mcp.tools_cli", *args],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _run_cli_json(home_dir: Path, *args: str, extra_env: "dict | None" = None) -> dict:
    proc = _run_cli(home_dir, *args, "--json", extra_env=extra_env)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def _index_whole_file(home_dir: Path, corpus: Path) -> None:
    proc = _run_cli(home_dir, "index", str(corpus))
    assert proc.returncode == 0, proc.stderr


def _index_into_store(home_dir: Path, files: "list[Path]", store_kind: str) -> None:
    """Indexes `files` into a specific lane's OWN store under the tmp HOME
    via a small inline script -- lets a test build lane-specific N without
    touching the whole-file lane's own store (which `index` always
    populates in lockstep with symbol+xref)."""
    if store_kind == "output":
        code = (
            "from iai_mcp.tools import index_paths, default_command_output_store_root;"
            "import sys;"
            "index_paths(sys.argv[1:], store_root=default_command_output_store_root())"
        )
    elif store_kind == "symbol":
        code = (
            "from iai_mcp.tools import index_symbols, default_symbol_store_root;"
            "import sys;"
            "index_symbols(sys.argv[1:], store_root=default_symbol_store_root())"
        )
    else:
        raise ValueError(store_kind)
    env = _cli_env(home_dir)
    proc = subprocess.run(
        [sys.executable, "-c", code, *[str(f) for f in files]],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr


# --------------------------------------------------------------------------
# MCP server subprocess helpers (mirrors tests/test_tools_mcp_server_list_no_daemon.py)
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


def _spawn_server(
    built_wrapper: Path,
    home_dir: Path,
    nonexistent_sock: Path,
    extra_env: "dict | None" = None,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["IAI_MCP_PYTHON"] = sys.executable
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["HOME"] = str(home_dir)
    env["IAI_TOOLS_SOCKET_PATH"] = str(nonexistent_sock)
    if extra_env:
        env.update(extra_env)
    return subprocess.Popen(
        ["node", str(built_wrapper)],
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
            "clientInfo": {"name": "iai-tools-absence-contract-test", "version": "0.1.0"},
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
# Whole-file lane, CLI --json
# --------------------------------------------------------------------------


def test_whole_file_scope_matches_indexed_file_count_and_is_query_independent(
    tmp_path: Path,
) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "alpha.py").write_text("def alphascopemarker():\n    return 'alpha content'\n")
    (corpus / "beta.py").write_text("def betascopemarker():\n    return 'beta content'\n")

    _index_whole_file(home_dir, corpus)

    hit_payload = _run_cli_json(home_dir, "search", "alphascopemarker")
    assert hit_payload["scope"]["n_files"] == 2
    assert hit_payload["scope"]["generation"] >= 1
    assert hit_payload["status"] == "hits"

    miss_payload = _run_cli_json(home_dir, "search", "a_string_never_indexed_anywhere_zzz")
    assert miss_payload["status"] == "unknown"
    assert miss_payload["scope"] == hit_payload["scope"], (
        "n_files/generation must be byte-identical across two different "
        "queries against the same store -- server-computed, never query-derived"
    )


def test_reindexing_a_changed_file_never_inflates_n_files(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    target = corpus / "gamma.py"
    target.write_text("def gammascopemarkerold():\n    return 1\n")

    _index_whole_file(home_dir, corpus)
    payload_before = _run_cli_json(home_dir, "search", "gammascopemarkerold")
    n_before = payload_before["scope"]["n_files"]
    assert n_before == 1

    target.write_text("def gammascopemarkernew():\n    return 2\n")
    _index_whole_file(home_dir, corpus)
    payload_after = _run_cli_json(home_dir, "search", "gammascopemarkernew")

    assert payload_after["scope"]["n_files"] == n_before, (
        "a tombstoning re-index of the same path must not inflate n_files"
    )
    assert payload_after["scope"]["generation"] > payload_before["scope"]["generation"]

    old_payload = _run_cli_json(home_dir, "search", "gammascopemarkerold")
    assert old_payload["status"] == "unknown"


def test_store_output_reports_the_output_stores_own_distinct_nonzero_scope(
    tmp_path: Path,
) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    code_corpus = tmp_path / "code-corpus"
    code_corpus.mkdir()
    (code_corpus / "one.py").write_text("def onestorescopemarker():\n    return 1\n")

    _index_whole_file(home_dir, code_corpus)

    # A DIFFERENT, non-empty file count in the output store. Fixture content
    # deliberately avoids the command-output secret-gate's trigger alphabet
    # (token/secret/password/api_key/private key/bearer/...) -- a refused
    # file would silently index zero and report (0, 0), a vacuous pass that
    # would not prove the readback follows the resolved store_root.
    output_dir = tmp_path / "output-artifacts"
    output_dir.mkdir()
    out_files = []
    for i in range(3):
        p = output_dir / f"run-{i}.log"
        p.write_text(f"outputstorescopemarker{i} plain captured run output\n")
        out_files.append(p)
    _index_into_store(home_dir, out_files, "output")

    code_payload = _run_cli_json(home_dir, "search", "onestorescopemarker")
    output_payload = _run_cli_json(
        home_dir, "search", "outputstorescopemarker0", "--store", "output"
    )

    assert code_payload["scope"]["n_files"] > 0
    assert output_payload["scope"]["n_files"] > 0
    assert output_payload["scope"]["n_files"] != code_payload["scope"]["n_files"], (
        "the output store must report its OWN scope, distinct from the code "
        "store -- an empty output store would report (0, 0) and pass "
        "trivially without proving the readback threads the resolved store_root"
    )
    assert output_payload["scope"]["n_files"] == 3
    assert code_payload["scope"]["n_files"] == 1


# --------------------------------------------------------------------------
# Whole-file lane, MCP structuredContent
# --------------------------------------------------------------------------


def test_mcp_code_search_structured_content_carries_scope_and_status(
    built_wrapper: Path, tmp_path: Path
) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "marker.py").write_text("def deltamcpscopemarker():\n    return 'delta content'\n")

    nonexistent_sock = tmp_path / "iai-tools-this-socket-does-not-exist.sock"
    proc = _spawn_server(built_wrapper, home_dir, nonexistent_sock)
    try:
        _initialize(proc)
        index_structured = _call_tool(proc, "code_index", {"path": str(corpus)}, 2)
        assert index_structured["indexed_files"] == 1

        result = _call_tool(
            proc, "code_search", {"query": "deltamcpscopemarker", "k": 10}, 3
        )
        assert result["scope"]["n_files"] == 1
        assert result["scope"]["generation"] >= 1
        assert result["status"] == "hits"

        miss_result = _call_tool(
            proc, "code_search", {"query": "never_indexed_anywhere_qqq", "k": 10}, 4
        )
        assert miss_result["status"] == "unknown"
        assert miss_result["scope"] == result["scope"]
    finally:
        _terminate(proc)


# --------------------------------------------------------------------------
# Symbol/xref/fused/vector lanes, CLI --json
# --------------------------------------------------------------------------


def test_symbol_scope_is_read_from_its_own_manifest_independent_of_whole_file(
    tmp_path: Path,
) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "alpha.py").write_text("def alphasymbolscopemarker():\n    return 1\n")
    _index_whole_file(home_dir, corpus)

    # One MORE file indexed into the symbol store ONLY, so symbol N
    # diverges from the whole-file N -- proving the symbol readback comes
    # from its own manifest, not a shared/whole-file one.
    extra_dir = tmp_path / "extra"
    extra_dir.mkdir()
    extra_file = extra_dir / "extra.py"
    extra_file.write_text("def extrasymbolonlymarker():\n    return 2\n")
    _index_into_store(home_dir, [extra_file], "symbol")

    whole_payload = _run_cli_json(home_dir, "search", "alphasymbolscopemarker")
    symbol_payload = _run_cli_json(home_dir, "search-symbol", "alphasymbolscopemarker")

    assert whole_payload["scope"]["n_files"] == 1
    assert symbol_payload["scope"]["n_files"] == 2
    assert symbol_payload["scope"]["n_files"] != whole_payload["scope"]["n_files"]
    assert symbol_payload["status"] == "hits"


def test_xref_and_fused_scope_equal_the_whole_file_lanes_own_scope(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "alpha.py").write_text("def xrefandfusedscopemarker():\n    return 1\n")
    _index_whole_file(home_dir, corpus)

    whole_payload = _run_cli_json(home_dir, "search", "xrefandfusedscopemarker")
    xref_payload = _run_cli_json(home_dir, "search-xref", "xrefandfusedscopemarker")
    fused_payload = _run_cli_json(home_dir, "search-fused", "xrefandfusedscopemarker")

    assert xref_payload["scope"] == whole_payload["scope"], (
        "the xref lane has no generation of its own -- it inherits the "
        "whole-file lane's N/G exactly"
    )
    assert xref_payload["status"] == "hits"

    assert fused_payload["scope"] == whole_payload["scope"], (
        "the whole-file lane is the only byte-bearing corpus among fused's "
        "inputs, so its N/G is fused's scanned scope"
    )
    assert fused_payload["status"] == "hits"
    assert "vector_lane" in fused_payload
    assert "reason" in fused_payload


def test_symbol_xref_fused_zero_hit_reports_unknown_status(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "alpha.py").write_text("def zetaexistingmarker():\n    return 1\n")
    _index_whole_file(home_dir, corpus)

    miss_query = "a_string_never_indexed_anywhere_zzz"

    symbol_payload = _run_cli_json(home_dir, "search-symbol", miss_query)
    assert symbol_payload["status"] == "unknown"
    assert symbol_payload["scope"]["n_files"] == 1

    xref_payload = _run_cli_json(home_dir, "search-xref", miss_query)
    assert xref_payload["status"] == "unknown"
    assert xref_payload["scope"]["n_files"] == 1

    if not HAS_BGE_SMALL_CACHE:
        pytest.skip(
            "real Hugging Face model cache not present; a fused-lane miss "
            "spawns the vector-embed subprocess and needs a warm cache"
        )
    fused_payload = _run_cli_json(
        home_dir, "search-fused", miss_query, extra_env=_hf_cache_env()
    )
    assert fused_payload["status"] == "unknown"
    assert fused_payload["scope"]["n_files"] == 1


@requires_hf_cache
def test_vector_lane_carries_status_and_omits_the_scope_key(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()

    # No vector store is ever built here -- search_vector on a missing
    # store returns an empty result by contract, which must still carry
    # status ("unknown"), proving an empty embedding-recall result cannot
    # read as proof-of-absence any more than a BM25 zero can.
    payload = _run_cli_json(
        home_dir, "search-vector", "anything at all whatsoever", extra_env=_hf_cache_env()
    )
    assert payload["status"] == "unknown"
    assert "scope" not in payload, "the vector lane has no manifest/generation -- no scope key"


# --------------------------------------------------------------------------
# Vector/fused lanes, MCP structuredContent
# --------------------------------------------------------------------------


@requires_hf_cache
def test_mcp_code_search_vector_lane_has_status_but_no_scope_key(
    built_wrapper: Path, tmp_path: Path
) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    nonexistent_sock = tmp_path / "iai-tools-vector-lane.sock"

    proc = _spawn_server(built_wrapper, home_dir, nonexistent_sock, extra_env=_hf_cache_env())
    try:
        _initialize(proc)
        result = _call_tool(
            proc, "code_search", {"query": "anything at all whatsoever", "lane": "vector", "k": 5}, 2
        )
        assert "status" in result
        assert "scope" not in result
    finally:
        _terminate(proc)


@requires_hf_cache
def test_mcp_code_search_fused_lane_carries_scope_status_vector_lane_and_reason_together(
    built_wrapper: Path, tmp_path: Path
) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "marker.py").write_text("def fusedmcpscopemarker():\n    return 1\n")

    nonexistent_sock = tmp_path / "iai-tools-fused-lane.sock"
    proc = _spawn_server(built_wrapper, home_dir, nonexistent_sock, extra_env=_hf_cache_env())
    try:
        _initialize(proc)
        index_structured = _call_tool(proc, "code_index", {"path": str(corpus)}, 2)
        assert index_structured["indexed_files"] == 1

        result = _call_tool(
            proc, "code_search", {"query": "fusedmcpscopemarker", "lane": "fused", "k": 5}, 3
        )
        assert "scope" in result
        assert "status" in result
        assert "vector_lane" in result
        assert "reason" in result
    finally:
        _terminate(proc)


# --------------------------------------------------------------------------
# Contract-invariant regression guard -- locks the response shape future
# callers report absence through. Positive enum + key allowlist, never a
# negative word search, so an unexpected value or a stray new key is
# caught automatically.
# --------------------------------------------------------------------------

_LANE_KEY_ALLOWLISTS = {
    "whole": {"query", "lane", "hits", "scope", "status"},
    "symbol": {"query", "lane", "hits", "scope", "status"},
    "xref": {"query", "lane", "hits", "scope", "status"},
    "fused": {"query", "lane", "hits", "scope", "status", "vector_lane", "reason"},
    "vector": {"query", "lane", "hits", "status"},
}

_LANE_SUBCOMMANDS = {
    "whole": "search",
    "symbol": "search-symbol",
    "xref": "search-xref",
    "fused": "search-fused",
    "vector": "search-vector",
}

_TEXT_LANES = ("whole", "symbol", "xref", "fused")


def _assert_locked_response_shape(lane: str, payload: dict) -> None:
    allowed = _LANE_KEY_ALLOWLISTS[lane]
    extra_keys = set(payload.keys()) - allowed
    assert not extra_keys, (
        f"lane {lane!r} response carries an unexpected key {extra_keys}, "
        f"outside the locked allowlist {sorted(allowed)}: {payload}"
    )
    missing_keys = allowed - set(payload.keys())
    assert not missing_keys, (
        f"lane {lane!r} response is missing required key(s) {missing_keys}, "
        f"a silent drop of a locked-contract key: {payload}"
    )
    assert payload["status"] in ("hits", "unknown"), (
        f"lane {lane!r} status must stay within the locked enum "
        f"{{'hits', 'unknown'}}, got {payload['status']!r}"
    )
    if lane == "vector":
        assert "scope" not in payload, (
            f"lane {lane!r} must never carry a scope key -- the vector "
            f"store has no manifest/generation: {payload}"
        )
    else:
        n_files = payload["scope"]["n_files"]
        generation = payload["scope"]["generation"]
        assert isinstance(n_files, int) and not isinstance(n_files, bool) and n_files >= 0
        assert isinstance(generation, int) and not isinstance(generation, bool) and generation >= 0


def test_every_lane_response_shape_stays_within_the_locked_absence_contract(
    tmp_path: Path,
) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "alpha.py").write_text("def contractguardhitmarker():\n    return 1\n")
    _index_whole_file(home_dir, corpus)

    hit_query = "contractguardhitmarker"
    miss_query = "a_string_never_indexed_anywhere_for_the_contract_guard_zzz"

    for lane in _TEXT_LANES:
        if lane == "fused" and not HAS_BGE_SMALL_CACHE:
            # a fused-lane miss spawns the vector-embed subprocess; skip
            # this iteration cleanly without a warm cache, keep the rest
            continue
        subcommand = _LANE_SUBCOMMANDS[lane]
        extra_env = _hf_cache_env() if lane == "fused" else None
        hit_payload = _run_cli_json(home_dir, subcommand, hit_query, extra_env=extra_env)
        miss_payload = _run_cli_json(home_dir, subcommand, miss_query, extra_env=extra_env)

        _assert_locked_response_shape(lane, hit_payload)
        _assert_locked_response_shape(lane, miss_payload)

        assert hit_payload["status"] == "hits", f"lane {lane!r}: {hit_payload}"
        assert miss_payload["status"] == "unknown", (
            f"lane {lane!r}: an empty result must always report status "
            f"unknown, got {miss_payload}"
        )
        if "scope" in hit_payload:
            assert hit_payload["scope"] == miss_payload["scope"], (
                f"lane {lane!r}: n_files/generation must stay unchanged "
                f"across two distinct queries against the same store"
            )

    # Vector lane: no vector store is built here, so every query is a
    # miss -- the invariant this guard needs from it (empty -> unknown,
    # no scope key, status within the locked enum) does not require a hit.
    if not HAS_BGE_SMALL_CACHE:
        pytest.skip("real Hugging Face model cache not present; vector-lane tests require a pre-warmed cache")
    vector_payload = _run_cli_json(
        home_dir, "search-vector", miss_query, extra_env=_hf_cache_env()
    )
    _assert_locked_response_shape("vector", vector_payload)
    assert vector_payload["status"] == "unknown"
    assert "scope" not in vector_payload

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
WRAPPER = REPO / "mcp-wrapper"


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
) -> subprocess.Popen:
    env = os.environ.copy()
    env["IAI_MCP_PYTHON"] = sys.executable
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["HOME"] = str(home_dir)
    env["IAI_TOOLS_SOCKET_PATH"] = str(nonexistent_sock)
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


def _read_response_with_id(
    proc: subprocess.Popen,
    rpc_id: int,
    timeout_sec: float,
) -> dict:
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
    raise TimeoutError(
        f"server did not respond to id={rpc_id} within {timeout_sec}s"
    )


EXPECTED_TOOL_NAMES = {
    "code_search", "code_index", "execute", "session_search", "stats", "command_insight",
    "batch_execute",
}


def _capture_session_event(home_dir: Path, session_id: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["HOME"] = str(home_dir)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from iai_mcp.tools.session import capture_event;"
            "capture_event({'tool_name': 'Edit', 'file_path': 'src/guard.py', "
            f"'session_id': {session_id!r}, 'source_hook': 'PostToolUse'}})",
        ],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr


def _call_tool(proc: subprocess.Popen, name: str, arguments: dict, rpc_id: int) -> dict:
    _send_rpc(proc, "tools/call", {"name": name, "arguments": arguments}, rpc_id)
    resp = _read_response_with_id(proc, rpc_id, timeout_sec=15.0)
    assert "result" in resp, f"tools/call {name} error: {resp}"
    result = resp["result"]
    assert not result.get("isError"), f"{name} returned an error: {result}"
    return result.get("structuredContent") or json.loads(result["content"][0]["text"])


def test_all_tools_list_and_read_tools_no_daemon(
    built_wrapper: Path,
    tmp_path: Path,
) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "marker.py").write_text(
        "def zebracorn_finder():\n    return 'zebracorn habitat lookup'\n"
    )

    session_id = "sess-tools-mcp-e2e"
    _capture_session_event(home_dir, session_id)

    tools_store = home_dir / ".iai-mcp" / "tools"
    nonexistent_sock = tmp_path / "iai-tools-this-socket-does-not-exist.sock"
    assert not nonexistent_sock.exists()

    proc = _spawn_server(built_wrapper, home_dir, nonexistent_sock)
    try:
        _send_rpc(
            proc,
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "iai-tools-no-daemon-test", "version": "0.1.0"},
            },
            1,
        )
        init_resp = _read_response_with_id(proc, 1, timeout_sec=5.0)
        assert "result" in init_resp, f"initialize failed: {init_resp}"

        _send_notification(proc, "notifications/initialized")

        _send_rpc(proc, "tools/list", {}, 2)
        list_resp = _read_response_with_id(proc, 2, timeout_sec=5.0)
        assert "result" in list_resp, f"tools/list error: {list_resp}"
        names = {t["name"] for t in list_resp["result"]["tools"]}
        assert names == EXPECTED_TOOL_NAMES, (
            f"tools/list returned {names}, expected exactly {EXPECTED_TOOL_NAMES}"
        )
        assert "fetch_index" not in names

        index_structured = _call_tool(proc, "code_index", {"path": str(corpus)}, 3)
        assert index_structured["indexed_files"] == 1, index_structured
        assert index_structured["path"] == str(corpus)

        search_structured = _call_tool(
            proc, "code_search", {"query": "zebracorn", "k": 10}, 4
        )
        hits = search_structured["hits"]
        assert hits, "expected at least one ranked hit"
        assert any("marker.py" in hit["path"] for hit in hits), (
            f"expected marker.py among hits, got: {hits}"
        )
        assert all(
            "line_start" in hit and "line_end" in hit and "snippet" in hit
            for hit in hits
        ), f"expected every hit to carry a bounded snippet, got: {hits}"

        symbol_structured = _call_tool(
            proc,
            "code_search",
            {"query": "zebracorn_finder", "k": 10, "lane": "symbol"},
            7,
        )
        symbol_hits = symbol_structured["hits"]
        assert symbol_hits, "expected at least one symbol-lane hit"
        assert any("marker.py" in hit["path"] for hit in symbol_hits), (
            f"expected marker.py among symbol-lane hits, got: {symbol_hits}"
        )

        cli_env = os.environ.copy()
        cli_env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + cli_env.get("PYTHONPATH", "")
        cli_env["HOME"] = str(home_dir)
        cli_symbol = subprocess.run(
            [sys.executable, "-m", "iai_mcp.tools_cli", "search-symbol", "zebracorn_finder", "--json"],
            cwd=str(REPO),
            env=cli_env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert cli_symbol.returncode == 0, cli_symbol.stderr
        cli_symbol_hits = json.loads(cli_symbol.stdout)["hits"]
        assert [h["path"] for h in cli_symbol_hits] == [h["path"] for h in symbol_hits], (
            "MCP symbol-lane hit paths must match the CLI's own search-symbol --json output"
        )

        fused_structured = _call_tool(
            proc,
            "code_search",
            {"query": "zebracorn_finder", "k": 10, "lane": "fused"},
            8,
        )
        assert "vector_lane" in fused_structured, (
            f"expected a fused-lane response to carry vector_lane, got: {fused_structured}"
        )
        assert fused_structured["vector_lane"] in ("not_attempted", "ok", "degraded")

        _send_rpc(
            proc,
            "tools/call",
            {"name": "code_search", "arguments": {"query": "zebracorn", "lane": "not_a_real_lane"}},
            9,
        )
        bad_lane_resp = _read_response_with_id(proc, 9, timeout_sec=5.0)
        assert bad_lane_resp["result"]["isError"], (
            f"an unknown lane must be rejected before any CLI spawn, got: {bad_lane_resp}"
        )

        session_structured = _call_tool(
            proc, "session_search", {"session_id": session_id}, 5
        )
        assert session_structured["session_id"] == session_id
        events = session_structured["events"]
        assert events, "expected at least one captured event"

        stats_structured = _call_tool(proc, "stats", {}, 6)
        assert stats_structured["code_index"]["whole_file"]["segment_count"] >= 1
        assert stats_structured["sessions"]["session_count"] >= 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    assert not nonexistent_sock.exists(), (
        "the executor socket must never be created by the four read-tool calls"
    )

    segments_after = list(tools_store.glob("*"))
    assert segments_after, "the code-index segment must land under the isolated temp store"

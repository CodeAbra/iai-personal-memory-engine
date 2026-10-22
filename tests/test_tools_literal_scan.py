"""End-to-end coverage for the `literal` code_search lane: a byte-for-byte
exact-substring scan over the currently-indexed file set that can PROVE a
literal's absence, unlike the tokenizing lanes' honest "unknown".

`code_search`/`iai-tools` resolve their default store under
`Path.home()`, bound at IMPORT time -- an in-process `store_root=`
override cannot reach that path. Every test here runs the real CLI (or
the built MCP server) as a subprocess with `HOME` pointed at a fresh
temp directory, mirroring tests/test_tools_absence_contract.py's
hermetic pattern. The literal lane never embeds, so no Hugging Face
cache passthrough is needed here.
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
# CLI subprocess helpers (mirrors test_tools_absence_contract.py)
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


def _index_whole_file(home_dir: Path, corpus: Path) -> None:
    proc = _run_cli(home_dir, "index", str(corpus))
    assert proc.returncode == 0, proc.stderr


def _new_home_and_corpus(tmp_path: Path) -> "tuple[Path, Path]":
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    return home_dir, corpus


# --------------------------------------------------------------------------
# MCP server subprocess helpers (mirrors test_tools_absence_contract.py)
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
            "clientInfo": {"name": "iai-tools-literal-scan-test", "version": "0.1.0"},
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
# Present literal, end to end (CLI --json)
# --------------------------------------------------------------------------


def test_present_literal_is_found_end_to_end(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    (corpus / "alpha.py").write_text(
        "def unrelated():\n    return zebracornliteralpresent\n"
    )
    _index_whole_file(home_dir, corpus)

    payload = _run_cli_json(home_dir, "search-literal", "zebracornliteralpresent")
    assert payload["status"] == "hits"
    assert payload["lane"] == "literal"
    assert len(payload["hits"]) == 1
    assert payload["hits"][0]["path"].endswith("alpha.py")
    assert payload["hits"][0]["line"] == 2
    assert payload["files_scanned"] == 1
    assert payload["scope"]["n_files"] == 1
    assert payload["scope"]["generation"] >= 1
    assert "files_unreadable" not in payload
    assert "files_truncated" not in payload


def test_case_variant_literal_returns_zero_hits(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    (corpus / "alpha.py").write_text("CaseSensitiveLiteralMarker = 1\n")
    _index_whole_file(home_dir, corpus)

    payload = _run_cli_json(home_dir, "search-literal", "casesensitiveliteralmarker")
    assert payload["status"] == "absent"
    assert payload["hits"] == []


def test_search_literal_accepts_and_ignores_k(tmp_path: Path) -> None:
    # The generic runCodeSearch handler always appends --k for every lane;
    # search-literal must accept it without erroring, never using it.
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    (corpus / "alpha.py").write_text("ignoredkliteralmarker\n")
    _index_whole_file(home_dir, corpus)

    proc = _run_cli(home_dir, "search-literal", "ignoredkliteralmarker", "--k", "3", "--json")
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "hits"


def test_plain_text_mode_present_prints_line_and_path(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    (corpus / "alpha.py").write_text("plaintextliteralmarker\n")
    _index_whole_file(home_dir, corpus)

    proc = _run_cli(home_dir, "search-literal", "plaintextliteralmarker")
    assert proc.returncode == 0, proc.stderr
    assert "alpha.py" in proc.stdout
    assert proc.stdout.startswith("1\t")


def test_plain_text_mode_absent_prints_files_scanned_keyed_claim(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    (corpus / "alpha.py").write_text("present content only\n")
    _index_whole_file(home_dir, corpus)

    proc = _run_cli(home_dir, "search-literal", "a_literal_never_indexed_anywhere_zzz")
    assert proc.returncode == 0, proc.stderr
    assert "0 of 1 scanned" in proc.stdout
    assert "of 1 indexed" in proc.stdout


def test_empty_literal_is_rejected_not_a_zero_bucket_response(tmp_path: Path) -> None:
    # An empty needle would otherwise short-circuit the scan loop entirely,
    # leaving files_scanned == 0 while n_files > 0 -- indistinguishable from
    # a genuine "scanned everything, found nothing" absence proof. Must be
    # a clean rejection, not a zero-bucket JSON or plain-text response.
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    (corpus / "alpha.py").write_text("anything at all\n")
    _index_whole_file(home_dir, corpus)

    proc = _run_cli(home_dir, "search-literal", "")
    assert proc.returncode != 0
    assert "must not be empty" in proc.stderr
    assert proc.stdout == ""

    proc_json = _run_cli(home_dir, "search-literal", "", "--json")
    assert proc_json.returncode != 0
    assert "must not be empty" in proc_json.stderr
    assert proc_json.stdout == ""


# --------------------------------------------------------------------------
# Proof of absence: files_scanned-keyed denominator, deleted-file and
# truncated-tail exclusion, bucket-sum invariant, read-cap fence
# --------------------------------------------------------------------------


def test_absent_literal_is_proven_absent_over_files_scanned(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    (corpus / "alpha.py").write_text("present content only\n")
    (corpus / "beta.py").write_text("more present content\n")
    _index_whole_file(home_dir, corpus)

    payload = _run_cli_json(home_dir, "search-literal", "a_literal_never_indexed_anywhere_zzz")
    assert payload["status"] == "absent"
    assert payload["hits"] == []
    assert payload["files_scanned"] == 2
    assert payload["scope"]["n_files"] == 2
    assert payload["scope"]["generation"] >= 1
    assert "files_unreadable" not in payload
    assert "files_truncated" not in payload


def test_deleted_file_is_excluded_from_scanned_and_surfaced_as_unreadable(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    kept = corpus / "alpha.py"
    kept.write_text("keptfileliteralmarker\n")
    deleted = corpus / "beta.py"
    deleted.write_text("deletedfilecontent\n")
    _index_whole_file(home_dir, corpus)
    deleted.unlink()

    payload = _run_cli_json(home_dir, "search-literal", "a_literal_never_indexed_anywhere_zzz")
    assert payload["status"] == "absent"
    assert payload["scope"]["n_files"] == 2
    assert payload["files_scanned"] == 1
    assert payload["files_unreadable"] == 1
    assert "files_truncated" not in payload


def test_truncated_tail_file_is_excluded_from_scanned_and_surfaced_as_truncated(
    tmp_path: Path,
) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    target = corpus / "alpha.py"
    target.write_text("small content, indexed first\n")
    _index_whole_file(home_dir, corpus)

    read_cap = 4 * 1024 * 1024
    with target.open("ab") as f:
        written = 0
        filler = b"x" * 8192
        while written < read_cap:
            f.write(filler)
            written += len(filler)
        f.write(b"unreadtailliteralforintegration\n")

    payload = _run_cli_json(home_dir, "search-literal", "unreadtailliteralforintegration")
    assert payload["status"] == "absent", "the literal lives past the read cap -- must never be a hit"
    assert payload["hits"] == []
    assert payload["files_scanned"] == 0
    assert payload["files_truncated"] == 1
    assert "files_unreadable" not in payload

    proc = _run_cli(home_dir, "search-literal", "unreadtailliteralforintegration")
    assert "0 of 0 scanned" in proc.stdout
    assert "1 truncated" in proc.stdout


def test_bucket_counts_sum_to_n_files(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    a = corpus / "alpha.py"
    a.write_text("aaa content\n")
    b = corpus / "beta.py"
    b.write_text("bbb content\n")
    c = corpus / "gamma.py"
    c.write_text("ccc content\n")
    _index_whole_file(home_dir, corpus)
    b.unlink()

    payload = _run_cli_json(home_dir, "search-literal", "never_indexed_anywhere_for_bucket_sum_zzz")
    n_files = payload["scope"]["n_files"]
    files_scanned = payload["files_scanned"]
    files_unreadable = payload.get("files_unreadable", 0)
    files_truncated = payload.get("files_truncated", 0)
    assert files_scanned + files_unreadable + files_truncated == n_files


def test_read_cap_covers_the_admission_cap_live(tmp_path: Path) -> None:
    # The ONLY guard that fires if someone raises the admission cap past
    # the read cap -- both values imported live, never mirrored constants.
    import iai_mcp_tools_core as _native
    from iai_mcp.study import STUDY_DIR_MAX_FILE_BYTES

    assert _native.literal_scan_read_cap_bytes() >= STUDY_DIR_MAX_FILE_BYTES


def test_literal_at_admission_cap_boundary_is_found(tmp_path: Path) -> None:
    # A file just UNDER the admission cap (so it is actually indexed) with
    # the unique literal in its LAST bytes (well under the read cap) must
    # still be found -- nothing is truncated out of the scan at the real
    # boundary the corpus can actually produce.
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    target = corpus / "nearcap.py"
    admission_cap = 2 * 1024 * 1024
    literal = "nearadmissioncapboundaryliteral"
    filler_len = admission_cap - len(literal) - 4096
    with target.open("wb") as f:
        f.write(b"y" * filler_len)
        f.write(literal.encode())
        f.write(b"\n")

    _index_whole_file(home_dir, corpus)

    payload = _run_cli_json(home_dir, "search-literal", literal)
    assert payload["status"] == "hits"
    assert len(payload["hits"]) == 1
    assert payload["files_scanned"] == 1
    assert "files_truncated" not in payload


# --------------------------------------------------------------------------
# The literal lane's OWN response-shape guard -- a DISTINCT allowed/required
# split from test_tools_absence_contract.py's shared {hits, unknown} loop
# (files_unreadable/files_truncated are allowed but not required, since
# each is omitted when zero). Deliberately NOT folded into that shared
# loop -- the literal lane's third status value must never leak into the
# four unrelated lanes' assertions there.
# --------------------------------------------------------------------------

_LITERAL_LANE_ALLOWED_KEYS = {
    "query",
    "lane",
    "hits",
    "scope",
    "files_scanned",
    "status",
    "files_unreadable",
    "files_truncated",
}
_LITERAL_LANE_REQUIRED_KEYS = {
    "query",
    "lane",
    "hits",
    "scope",
    "files_scanned",
    "status",
}


def _assert_literal_lane_response_shape(payload: dict) -> None:
    extra_keys = set(payload.keys()) - _LITERAL_LANE_ALLOWED_KEYS
    assert not extra_keys, (
        f"literal lane response carries an unexpected key {extra_keys}, "
        f"outside the locked allowlist {sorted(_LITERAL_LANE_ALLOWED_KEYS)}: {payload}"
    )
    missing_keys = _LITERAL_LANE_REQUIRED_KEYS - set(payload.keys())
    assert not missing_keys, (
        f"literal lane response is missing required key(s) {missing_keys}: {payload}"
    )
    assert payload["status"] in ("hits", "absent"), (
        f"literal lane status must stay within {{'hits', 'absent'}}, got {payload['status']!r}"
    )


def test_literal_lane_response_shape_guard(tmp_path: Path) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    (corpus / "alpha.py").write_text("shapeguardliteralmarker\n")
    _index_whole_file(home_dir, corpus)

    hit_payload = _run_cli_json(home_dir, "search-literal", "shapeguardliteralmarker")
    miss_payload = _run_cli_json(
        home_dir, "search-literal", "a_literal_never_indexed_for_shape_guard_zzz"
    )
    _assert_literal_lane_response_shape(hit_payload)
    _assert_literal_lane_response_shape(miss_payload)
    assert hit_payload["status"] == "hits"
    assert miss_payload["status"] == "absent"


# --------------------------------------------------------------------------
# MCP structuredContent
# --------------------------------------------------------------------------


def test_mcp_code_search_literal_lane_present_and_absent(
    built_wrapper: Path, tmp_path: Path
) -> None:
    home_dir, corpus = _new_home_and_corpus(tmp_path)
    (corpus / "marker.py").write_text("mcpliteralpresentmarker\n")

    nonexistent_sock = tmp_path / "iai-tools-literal-lane.sock"
    proc = _spawn_server(home_dir, nonexistent_sock)
    try:
        _initialize(proc)
        index_structured = _call_tool(proc, "code_index", {"path": str(corpus)}, 2)
        assert index_structured["indexed_files"] == 1

        hit_result = _call_tool(
            proc,
            "code_search",
            {"query": "mcpliteralpresentmarker", "lane": "literal", "k": 5},
            3,
        )
        assert hit_result["status"] == "hits"
        assert hit_result["hits"][0]["path"].endswith("marker.py")
        _assert_literal_lane_response_shape(hit_result)

        miss_result = _call_tool(
            proc,
            "code_search",
            {"query": "a_literal_never_indexed_over_mcp_zzz", "lane": "literal", "k": 5},
            4,
        )
        assert miss_result["status"] == "absent"
        assert "files_scanned" in miss_result
        _assert_literal_lane_response_shape(miss_result)
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
    r"\bLSCAN-\d+\b"
    r"|\bPhase\s+\d+\b"
    r"|\bPlan\s+\d+\b"
    r"|\bTask\s+\d+\b"
    r"|\bWave\s+\d+\b"
    r"|\.planning/"
    r"|\bConstitutional\b",
    re.IGNORECASE,
)

_TOUCHED_SOURCE_FILES = [
    REPO / "rust" / "iai_mcp_tools_core" / "src" / "exact_scan.rs",
    REPO / "rust" / "iai_mcp_tools_core" / "src" / "snippet.rs",
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

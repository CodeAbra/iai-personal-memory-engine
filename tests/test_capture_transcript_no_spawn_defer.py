from __future__ import annotations

import json
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX subprocess + AF_UNIX",
)


def _isolated_env(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    sock_dir = Path(tempfile.mkdtemp(prefix="iai-sock-"))
    sock_path = sock_dir / "d.sock"

    iai_dir = tmp_path / ".iai-mcp"
    iai_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    env["IAI_DAEMON_SOCKET_PATH"] = str(sock_path)
    env["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"
    env["IAI_MCP_CRYPTO_PASSPHRASE"] = "test-no-spawn-defer-pass"
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")

    return env, iai_dir / ".deferred-captures", sock_path


def _cleanup_socket(sock_path: Path) -> None:
    try:
        sock_path.unlink()
    except FileNotFoundError:
        pass
    try:
        sock_path.parent.rmdir()
    except OSError:
        pass


def _make_transcript(tmp_path: Path) -> Path:
    turns = [
        {"type": "user", "message": {"role": "user", "content": "hello phase 7 5"}},
        {"type": "assistant", "message": {"role": "assistant", "content": "ack always defer"}},
        {"type": "user", "message": {"role": "user", "content": "third defer turn"}},
    ]
    transcript_path = tmp_path / "transcript.jsonl"
    transcript_path.write_text("\n".join(json.dumps(t) for t in turns) + "\n")
    return transcript_path


def _run_no_spawn(env: dict[str, str], transcript_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "iai_mcp.cli",
            "capture-transcript",
            "--no-spawn",
            "--session-id",
            "test-phase75",
            str(transcript_path),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )


def _bind_listener(sock_path: Path) -> socket.socket:
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    if sock_path.exists():
        sock_path.unlink()
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(str(sock_path))
    s.listen(1)
    return s


def test_no_spawn_reachable_defers_not_inserts(tmp_path):
    env, deferred_dir, sock_path = _isolated_env(tmp_path)
    transcript = _make_transcript(tmp_path)

    listener = _bind_listener(sock_path)
    try:
        proc = _run_no_spawn(env, transcript)
    finally:
        listener.close()
        _cleanup_socket(sock_path)

    assert proc.returncode == 0, f"stderr={proc.stderr!r} stdout={proc.stdout!r}"

    payload = json.loads(proc.stdout.strip())
    assert payload.get("status") == "deferred", (
        f"reachable case must defer; got {payload!r}"
    )
    assert "path" in payload, payload
    assert "inserted" not in payload, (
        f"inline-ingest path must not run under --no-spawn; got {payload!r}"
    )

    assert "Loading weights" not in proc.stderr, (
        f"embedder cold-loaded on reachable --no-spawn path:\n"
        f"{proc.stderr}"
    )
    assert "sentence_transformers" not in proc.stderr, (
        f"sentence_transformers output on --no-spawn path (should be impossible):\n"
        f"{proc.stderr}"
    )

    files = sorted(deferred_dir.glob("*.jsonl"))
    assert len(files) == 1, f"expected 1 deferred file, got {files}"
    header = json.loads(files[0].read_text().splitlines()[0])
    assert header["version"] == 1
    assert header["session_id"] == "test-phase75"


def test_no_spawn_unreachable_still_defers(tmp_path):
    env, deferred_dir, sock_path = _isolated_env(tmp_path)
    transcript = _make_transcript(tmp_path)

    assert not sock_path.exists()

    try:
        proc = _run_no_spawn(env, transcript)
    finally:
        _cleanup_socket(sock_path)

    assert proc.returncode == 0, f"stderr={proc.stderr!r} stdout={proc.stdout!r}"
    payload = json.loads(proc.stdout.strip())
    assert payload.get("status") == "deferred", payload
    assert "inserted" not in payload, payload

    assert "Loading weights" not in proc.stderr, proc.stderr
    assert "sentence_transformers" not in proc.stderr, proc.stderr

    files = sorted(deferred_dir.glob("*.jsonl"))
    assert len(files) == 1, f"expected 1 deferred file, got {files}"


def test_no_spawn_zero_embedder_imports_in_fresh_process(tmp_path):
    env, deferred_dir, _sock_path = _isolated_env(tmp_path)
    transcript = _make_transcript(tmp_path)

    driver = (
        "import sys, json\n"
        "from iai_mcp.cli import main\n"
        "rc = main([\n"
        "  'capture-transcript', '--no-spawn',\n"
        "  '--session-id', 'test-phase75-fresh',\n"
        f"  {str(transcript)!r},\n"
        "])\n"
        "loaded = sorted(\n"
        "  k for k in sys.modules\n"
        "  if k == 'iai_mcp.embed' or k.startswith('iai_mcp.embed.')\n"
        "  or k == 'sentence_transformers' or k.startswith('sentence_transformers.')\n"
        "  or k == 'torch' or k.startswith('torch.')\n"
        "  or k == 'transformers' or k.startswith('transformers.')\n"
        ")\n"
        "print('IAIMCP75_DUMP=' + json.dumps({'rc': rc, 'loaded': loaded}))\n"
    )

    proc = subprocess.run(
        [sys.executable, "-c", driver],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert proc.returncode == 0, f"driver failed: stderr={proc.stderr!r}"

    dump_lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("IAIMCP75_DUMP=")]
    assert len(dump_lines) == 1, f"expected 1 dump line, got {dump_lines!r}"
    dump = json.loads(dump_lines[0][len("IAIMCP75_DUMP=") :])

    assert dump["rc"] == 0, f"main() returned {dump['rc']}"

    loaded = set(dump["loaded"])
    forbidden = {m for m in loaded if (
        m == "iai_mcp.embed" or m.startswith("iai_mcp.embed.")
        or m == "sentence_transformers" or m.startswith("sentence_transformers.")
    )}
    assert not forbidden, (
        f"--no-spawn must not import embedder/ML deps; loaded: {sorted(forbidden)}"
    )

    assert any(deferred_dir.glob("*.jsonl"))


def test_no_spawn_branch_has_no_inline_imports():
    cli_src = (REPO / "src" / "iai_mcp" / "cli" / "_capture.py").read_text()

    fn_match = re.search(
        r"^def cmd_capture_transcript\(.*?\n(.*?)^def ",
        cli_src,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert fn_match, "could not locate cmd_capture_transcript in cli.py"
    fn_body = fn_match.group(1)

    no_spawn_match = re.search(
        r"^    if no_spawn:\n(.*?)^    # Default path",
        fn_body,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert no_spawn_match, (
        "could not isolate `if no_spawn:` block; layout drifted"
    )
    no_spawn_block = no_spawn_match.group(1)

    assert "write_deferred_captures" in no_spawn_block, (
        "no_spawn branch must call write_deferred_captures"
    )

    assert "from iai_mcp.capture import capture_transcript" not in no_spawn_block, (
        "Regression: capture_transcript reintroduced into "
        "--no-spawn branch (would trigger embedder cold-load on every "
        "Stop-hook fire)"
    )
    assert "from iai_mcp.store import MemoryStore" not in no_spawn_block, (
        "Regression: MemoryStore reintroduced into --no-spawn "
        "branch"
    )

    assert "_try_short_timeout_connect" not in no_spawn_block, (
        "Socket probe must be gone from --no-spawn branch (the "
        "probe was the gate that selected the inline path)"
    )

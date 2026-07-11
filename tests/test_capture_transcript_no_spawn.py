from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

REPO = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX subprocess + AF_UNIX",
)


def _count_iai_mcp_processes() -> dict[str, int]:
    counts = {"core": 0, "daemon": 0}
    for p in psutil.process_iter(["cmdline"]):
        try:
            cl = p.info.get("cmdline") or []
            if not cl:
                continue
            joined = " ".join(c or "" for c in cl)
            if "iai_mcp.core" in joined:
                counts["core"] += 1
            if "iai_mcp.daemon" in joined:
                counts["daemon"] += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return counts


def _isolated_env(tmp_path: Path) -> tuple[dict[str, str], Path]:
    sock_dir = Path(f"/tmp/iai-no-spawn-{os.getpid()}-{id(tmp_path)}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"

    iai_dir = tmp_path / ".iai-mcp"
    iai_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    env["IAI_DAEMON_SOCKET_PATH"] = str(sock_path)
    env["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"
    env["IAI_MCP_CRYPTO_PASSPHRASE"] = "test-no-spawn-pass"
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")

    return env, iai_dir / ".deferred-captures"


def _make_transcript(tmp_path: Path) -> Path:
    turns = [
        {"type": "user", "message": {"role": "user", "content": "hello world"}},
        {"type": "assistant", "message": {"role": "assistant", "content": "hi back at you"}},
        {"type": "user", "message": {"role": "user", "content": "third turn here"}},
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
            "test-r3",
            str(transcript_path),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )


def test_no_spawn_writes_deferred_when_daemon_down(tmp_path):
    env, deferred_dir = _isolated_env(tmp_path)
    transcript = _make_transcript(tmp_path)

    proc = _run_no_spawn(env, transcript)

    assert proc.returncode == 0, f"stderr={proc.stderr!r} stdout={proc.stdout!r}"
    payload = json.loads(proc.stdout.strip())
    assert payload.get("status") == "deferred", payload

    files = sorted(deferred_dir.glob("*.jsonl"))
    assert len(files) == 1, f"expected 1 deferral file, got {files}"

    out_path = files[0]
    lines = out_path.read_text().splitlines()
    assert len(lines) >= 2, f"expected header + ≥1 event, got {lines}"

    header = json.loads(lines[0])
    assert header["version"] == 1, header
    assert header["session_id"] == "test-r3", header
    assert "deferred_at" in header
    assert "cwd" in header

    for line in lines[1:]:
        ev = json.loads(line)
        assert "text" in ev and ev["text"], ev
        assert ev["tier"] == "episodic", ev
        assert ev["role"] in {"user", "assistant"}, ev


def test_no_spawn_completes_in_under_2s(tmp_path):
    env, _ = _isolated_env(tmp_path)
    transcript = _make_transcript(tmp_path)

    t0 = time.time()
    proc = _run_no_spawn(env, transcript)
    duration = time.time() - t0

    assert proc.returncode == 0, f"stderr={proc.stderr!r}"
    assert duration < 2.0, (
        f"--no-spawn took {duration:.3f}s; R3 budget is <2.0s. "
        f"Hook would block session teardown."
    )


def test_no_spawn_does_not_spawn_daemon(tmp_path):
    env, _ = _isolated_env(tmp_path)
    transcript = _make_transcript(tmp_path)

    before = _count_iai_mcp_processes()
    proc = _run_no_spawn(env, transcript)
    time.sleep(0.5)
    after = _count_iai_mcp_processes()

    assert proc.returncode == 0, f"stderr={proc.stderr!r}"

    delta_daemon = after["daemon"] - before["daemon"]
    delta_core = after["core"] - before["core"]
    assert delta_daemon <= 0, (
        f"--no-spawn spawned {delta_daemon} new daemon(s); R3 violated. "
        f"before={before} after={after}"
    )
    assert delta_core <= 0, (
        f"--no-spawn spawned {delta_core} new core(s); R3 violated. "
        f"before={before} after={after}"
    )


def test_no_spawn_flag_default_false(tmp_path):
    env, deferred_dir = _isolated_env(tmp_path)

    help_proc = subprocess.run(
        [sys.executable, "-m", "iai_mcp.cli", "capture-transcript", "--help"],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert help_proc.returncode == 0, help_proc.stderr
    assert "--no-spawn" in help_proc.stdout, help_proc.stdout

    default_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "iai_mcp.cli",
            "capture-transcript",
            str(tmp_path / "no-such-file.jsonl"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert default_proc.returncode == 0, default_proc.stderr

    payload = json.loads(default_proc.stdout.strip())
    assert "errors" in payload or "inserted" in payload, payload

    if deferred_dir.exists():
        assert not list(deferred_dir.glob("*.jsonl")), (
            f"default capture-transcript must not write deferred files; got "
            f"{list(deferred_dir.glob('*.jsonl'))}"
        )


def test_deferred_jsonl_format_v1_header(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    transcript = _make_transcript(tmp_path)

    from iai_mcp.capture import write_deferred_captures

    out_path = write_deferred_captures(
        session_id="unit-e",
        transcript_path=transcript,
        cwd="/some/cwd",
    )

    assert out_path.exists()
    assert out_path.parent == tmp_path / ".iai-mcp" / ".deferred-captures"
    assert out_path.name.startswith("unit-e-"), out_path.name
    assert out_path.suffix == ".jsonl", out_path.name

    lines = out_path.read_text().splitlines()
    assert len(lines) == 4, lines

    header = json.loads(lines[0])
    assert header["version"] == 1
    assert header["session_id"] == "unit-e"
    assert header["cwd"] == "/some/cwd"
    assert "deferred_at" in header

    for ln in lines[1:]:
        ev = json.loads(ln)
        assert set(ev.keys()) >= {"text", "cue", "tier", "role", "ts"}, ev.keys()
        assert ev["tier"] == "episodic"
        assert ev["role"] in {"user", "assistant"}
        assert ev["text"] in {"hello world", "hi back at you", "third turn here"}


def test_deferred_jsonl_handles_missing_transcript(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    from iai_mcp.capture import write_deferred_captures

    out_path = write_deferred_captures(
        session_id="unit-f",
        transcript_path=tmp_path / "does-not-exist.jsonl",
    )

    assert out_path.exists()
    lines = out_path.read_text().splitlines()
    assert len(lines) == 1, f"expected header-only, got {lines}"

    header = json.loads(lines[0])
    assert header["version"] == 1
    assert header["session_id"] == "unit-f"
    assert isinstance(header.get("cwd"), str) and header["cwd"], header

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.skipif(platform.system() == "Windows", reason="POSIX bash + AF_UNIX"),
]

REPO = Path(__file__).resolve().parent.parent
HOOK = REPO / "src" / "iai_mcp" / "_deploy" / "hooks" / "iai-mcp-turn-capture.sh"


def _skip_guards():
    if not HOOK.exists():
        pytest.skip(f"hook script missing at {HOOK}")
    if not shutil.which("bash"):
        pytest.skip("bash not on PATH")


def _seed_db_and_watermark(home: Path, sid: str, past_ts: str, old_watermark_ts: str) -> None:
    # The gate reads the writer-stamped store-advance sidecar, never the store
    # file itself — the sidecar is engine-format-agnostic by design.
    hippo_dir = home / ".iai-mcp" / "hippo"
    hippo_dir.mkdir(parents=True, exist_ok=True)
    (hippo_dir / ".max-created-at").write_text(past_ts)

    state_dir = home / ".iai-mcp" / ".capture-state"
    state_dir.mkdir(parents=True, exist_ok=True)
    wm_path = state_dir / f"{sid}.watermark"
    wm_path.write_text(old_watermark_ts)


def _write_transcript(home: Path, sid: str) -> Path:
    transcript = home / f"{sid}.jsonl"
    transcript.write_text(
        json.dumps({
            "type": "user",
            "message": {"role": "user", "content": "hello gate test"},
        }) + "\n"
    )
    return transcript


def _run_hook(home: Path, sid: str, transcript: Path, extra_env: dict) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["HOME"] = str(home)
    env.update(extra_env)
    env.pop("IAI_DAEMON_SOCKET_PATH", None)
    env.update(extra_env)

    stdin_data = json.dumps({
        "session_id": sid,
        "transcript_path": str(transcript),
        "cwd": str(home),
    })
    return subprocess.run(
        ["bash", str(HOOK)],
        input=stdin_data,
        capture_output=True,
        text=True,
        env=env,
        timeout=15,
    )


def test_folded_gate_emits_full_oversized_brief():
    _skip_guards()

    home = Path(tempfile.mkdtemp(dir="/tmp"))
    try:
        sid = "gate5a-" + uuid.uuid4().hex[:8]
        past_ts = "2026-01-01T00:00:00+00:00"
        old_wm = "2025-12-31T23:59:59+00:00"

        _seed_db_and_watermark(home, sid, past_ts, old_wm)
        transcript = _write_transcript(home, sid)

        big_surface = "## Memory refreshed\n\nALICE-SURFACE-TOKEN " + ("x" * 20000)
        future_ts = "2026-06-01T00:00:00+00:00"
        reply_obj = {"result": {"rendered": big_surface, "new_max_ts": future_ts}}
        reply_frame = (json.dumps(reply_obj) + "\n").encode("utf-8")

        sock_path = str(home / "fake.sock")
        accept_done = threading.Event()

        def _listener():
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv.bind(sock_path)
            srv.listen(1)
            srv.settimeout(0.1)
            try:
                while not accept_done.is_set():
                    try:
                        conn, _ = srv.accept()
                    except socket.timeout:
                        continue
                    except Exception:
                        break
                    try:
                        conn.settimeout(1.0)
                        buf = b""
                        while b"\n" not in buf:
                            chunk = conn.recv(4096)
                            if not chunk:
                                break
                            buf += chunk
                        conn.sendall(reply_frame)
                    except Exception:
                        pass
                    finally:
                        try:
                            conn.close()
                        except Exception:
                            pass
                    break
            finally:
                try:
                    srv.close()
                except Exception:
                    pass

        t = threading.Thread(target=_listener, daemon=True)
        t.start()
        time.sleep(0.05)

        result = _run_hook(home, sid, transcript, {"IAI_DAEMON_SOCKET_PATH": sock_path})
        accept_done.set()

        assert result.returncode == 0, f"Hook rc={result.returncode}\nstderr: {result.stderr}"

        stdout = result.stdout
        assert stdout, "Hook stdout was empty — gate did not emit additionalContext"
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            pytest.fail(
                f"Hook stdout is not valid JSON (likely truncated reply): {exc}\n"
                f"stdout length={len(stdout)}, first 200 chars: {stdout[:200]}"
            )

        ac = payload.get("hookSpecificOutput", {}).get("additionalContext", "")
        assert "ALICE-SURFACE-TOKEN" in ac, (
            "additionalContext missing the expected surface token"
        )
        assert len(ac) > 16000, (
            f"additionalContext truncated: got {len(ac)} chars, expected > 16000"
        )
    finally:
        shutil.rmtree(str(home), ignore_errors=True)


def test_folded_gate_custom_store_does_not_touch_default_socket():
    _skip_guards()

    home = Path(tempfile.mkdtemp(dir="/tmp"))
    try:
        sid = "gate5b-" + uuid.uuid4().hex[:8]
        past_ts = "2026-01-01T00:00:00+00:00"
        old_wm = "2025-12-31T23:59:59+00:00"

        _seed_db_and_watermark(home, sid, past_ts, old_wm)
        transcript = _write_transcript(home, sid)

        custom_store = home / "custom_store"
        custom_store.mkdir()

        default_sock_dir = home / ".iai-mcp"
        default_sock_dir.mkdir(parents=True, exist_ok=True)
        default_sock_path = str(default_sock_dir / ".daemon.sock")

        accept_count = [0]
        stop_listener = threading.Event()

        def _listener():
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv.bind(default_sock_path)
            srv.listen(1)
            srv.settimeout(0.1)
            try:
                while not stop_listener.is_set():
                    try:
                        conn, _ = srv.accept()
                        accept_count[0] += 1
                        conn.close()
                    except socket.timeout:
                        continue
                    except Exception:
                        break
            finally:
                try:
                    srv.close()
                except Exception:
                    pass

        t = threading.Thread(target=_listener, daemon=True)
        t.start()
        time.sleep(0.05)

        env = os.environ.copy()
        env["HOME"] = str(home)
        env["IAI_MCP_STORE"] = str(custom_store)
        env.pop("IAI_DAEMON_SOCKET_PATH", None)

        stdin_data = json.dumps({
            "session_id": sid,
            "transcript_path": str(transcript),
            "cwd": str(home),
        })
        result = subprocess.run(
            ["bash", str(HOOK)],
            input=stdin_data,
            capture_output=True,
            text=True,
            env=env,
            timeout=15,
        )

        stop_listener.set()

        assert result.returncode == 0, f"Hook rc={result.returncode}"
        assert accept_count[0] == 0, (
            f"Default socket was contacted {accept_count[0]} time(s); "
            "the custom-store guard should have skipped the RPC entirely"
        )
        assert "additionalContext" not in result.stdout, (
            "Hook emitted additionalContext despite custom-store guard"
        )
    finally:
        shutil.rmtree(str(home), ignore_errors=True)


def test_folded_gate_failure_does_not_abort_capture():
    _skip_guards()

    home = Path(tempfile.mkdtemp(dir="/tmp"))
    try:
        sid = "gate5c-" + uuid.uuid4().hex[:8]
        past_ts = "2026-01-01T00:00:00+00:00"
        old_wm = "2025-12-31T23:59:59+00:00"

        _seed_db_and_watermark(home, sid, past_ts, old_wm)
        transcript = _write_transcript(home, sid)

        dead_sock = str(home / "dead_nonexistent.sock")

        result = _run_hook(
            home, sid, transcript,
            {"IAI_DAEMON_SOCKET_PATH": dead_sock},
        )

        assert result.returncode == 0, (
            f"Hook rc={result.returncode} — gate failure must not make the hook non-zero"
        )

        live_file = home / ".iai-mcp" / ".deferred-captures" / f"{sid}.live.jsonl"
        assert live_file.exists(), "live.jsonl not created — capture failed"
        lines = [ln for ln in live_file.read_text().splitlines() if ln.strip()]
        events = []
        for ln in lines:
            try:
                obj = json.loads(ln)
                if "text" in obj:
                    events.append(obj)
            except Exception:
                pass
        assert len(events) >= 1, "No capture events in live.jsonl"
        assert any("gate test" in ev.get("text", "") for ev in events), (
            "Expected transcript text not found in captured events"
        )

        offset_file = home / ".iai-mcp" / ".capture-state" / f"{sid}.offset"
        assert offset_file.exists(), "offset file not written"
        offset_val = int(offset_file.read_text().strip())
        assert offset_val > 0, f"offset not advanced: {offset_val}"

        assert "additionalContext" not in result.stdout, (
            "Gate emitted additionalContext despite dead socket — unexpected"
        )
    finally:
        shutil.rmtree(str(home), ignore_errors=True)

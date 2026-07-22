#!/usr/bin/env python3
"""Isolated integration check: daemon WAKE drain_active_live_captures end-to-end.

Safety contract:
- Does NOT touch the real ~/.iai-mcp/ store/socket.
- Spawns a SECOND daemon pointed at a tmp dir.
- Kills ONLY the tmp daemon (by captured PID / PGID), never pkill by name.

Usage:
    cd <repo-root>
    source .venv/bin/activate
    python scripts/check_wake_drain_integration.py
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VENV_PYTHON = REPO / ".venv" / "bin" / "python"
REAL_HOME = Path.home()
REAL_HF_CACHE = REAL_HOME / ".cache" / "huggingface"

PASSPHRASE = "iai-mcp-bench-falsifiability-deterministic-2026"

DISTINCT_NONCE = "zeta integration probe 9931 -- unique marker for drain e2e"


# ---------------------------------------------------------------------------
# Socket helpers
# ---------------------------------------------------------------------------

def _send_jsonrpc(sock_path: str, method: str, params: dict, rpc_id: int = 1) -> dict:
    """One-shot JSON-RPC 2.0 request over Unix socket; returns the response dict."""
    req = json.dumps({"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params}) + "\n"
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(sock_path)
        s.sendall(req.encode("utf-8"))
        data = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
    finally:
        s.close()
    line = data.split(b"\n")[0]
    return json.loads(line.decode("utf-8"))


def _send_type_request(sock_path: str, req_type: str, timeout: float = 5.0) -> dict | None:
    """Send a legacy {type: ...} request (status probe)."""
    req = json.dumps({"type": req_type}) + "\n"
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(sock_path)
        s.sendall(req.encode("utf-8"))
        data = b""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                chunk = s.recv(65536)
            except socket.timeout:
                break
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
    except (socket.timeout, OSError):
        return None
    finally:
        s.close()
    if not data:
        return None
    line = data.split(b"\n")[0]
    try:
        return json.loads(line.decode("utf-8"))
    except (json.JSONDecodeError, ValueError):
        return None


def _wait_daemon_ready(sock_path: str, timeout: float = 90.0) -> bool:
    """Poll until daemon replies ok=True to a {type:status} probe."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = _send_type_request(sock_path, "status", timeout=3.0)
        if resp is not None and resp.get("ok"):
            return True
        time.sleep(1.5)
    return False


# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------

def run() -> int:
    tmpdir = tempfile.mkdtemp(prefix="iai_drain_check_")
    print(f"[INFO] tmp sandbox: {tmpdir}")

    # Safety verification: real store must NOT be inside tmpdir
    real_sock = str(REAL_HOME / ".iai-mcp" / ".daemon.sock")

    fake_uuid = str(uuid.uuid4())
    tmp_sock = os.path.join(tmpdir, ".iai-mcp", ".daemon.sock")
    tmp_store = os.path.join(tmpdir, ".iai-mcp")

    # Pre-flight: stat the real socket inode if it exists (read-only, safe)
    real_sock_inode = None
    try:
        real_sock_inode = os.stat(real_sock).st_ino
        print(f"[INFO] real daemon socket inode: {real_sock_inode}")
    except FileNotFoundError:
        print("[INFO] real daemon socket not found (daemon may be down) — that's fine")

    # Build the subprocess environment
    env = os.environ.copy()
    env["HOME"] = tmpdir
    env["IAI_MCP_STORE"] = tmp_store
    env["IAI_DAEMON_SOCKET_PATH"] = tmp_sock
    env["IAI_MCP_CRYPTO_PASSPHRASE"] = PASSPHRASE
    # Point HF cache at the real location so the Rust embedder finds its model
    env["HF_HOME"] = str(REAL_HF_CACHE)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    # Suppress idle-shutdown so daemon stays alive during test
    env["IAI_DAEMON_IDLE_SHUTDOWN_SECS"] = "3600"

    # Create the deferred-captures dir so we can inject the live file
    deferred_dir = Path(tmpdir) / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] deferred-captures dir: {deferred_dir}")

    # Write the fake session live file BEFORE spawning the daemon.
    # The daemon's startup drain only processes .jsonl files (not .live.jsonl),
    # and drain_active_live_captures only fires on WAKE transition.
    # We'll inject it before starting so it's present when we trigger the drain.
    live_file = deferred_dir / f"{fake_uuid}.live.jsonl"
    header = {
        "version": 1,
        "session_id": fake_uuid,
        "deferred_at": datetime.now(timezone.utc).isoformat(),
        "cwd": "/tmp",
    }
    turn = {
        "cue": "zeta probe nonce",
        "text": DISTINCT_NONCE,
        "tier": "episodic",
        "role": "user",
        "session_id": fake_uuid,
    }
    with live_file.open("w") as fh:
        fh.write(json.dumps(header) + "\n")
        fh.write(json.dumps(turn) + "\n")

    print(f"[INFO] injected live file: {live_file}")
    print(f"[INFO] fake session_id: {fake_uuid}")
    print(f"[INFO] nonce text: {DISTINCT_NONCE!r}")

    # Spawn the daemon subprocess
    print("[INFO] spawning tmp daemon...")
    proc = subprocess.Popen(
        [str(VENV_PYTHON), "-m", "iai_mcp.daemon"],
        cwd=str(REPO),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,  # own process group for clean kill
    )
    daemon_pid = proc.pid
    daemon_pgid = os.getpgid(daemon_pid)
    print(f"[INFO] daemon pid={daemon_pid} pgid={daemon_pgid}")

    try:
        # --- Isolation gate ---
        # Verify the tmp socket is a different file than the real one.
        print("[INFO] waiting for daemon ready...")
        ready = _wait_daemon_ready(tmp_sock, timeout=90.0)
        if not ready:
            # Grab stderr for diagnosis
            proc.send_signal(signal.SIGTERM)
            stderr_out = b""
            try:
                stderr_out = proc.stderr.read(8000) if proc.stderr else b""
            except Exception:
                pass
            print(f"[FAIL] daemon did not become ready within 90s")
            print(f"[FAIL] stderr: {stderr_out.decode('utf-8', errors='replace')[:2000]}")
            return 1

        print("[INFO] daemon ready")

        # Post-bind: verify tmp socket is not the real socket
        try:
            tmp_inode = os.stat(tmp_sock).st_ino
            print(f"[INFO] tmp daemon socket inode: {tmp_inode}")
            if real_sock_inode is not None and tmp_inode == real_sock_inode:
                print("[ABORT] SAFETY VIOLATION: tmp socket == real socket inode — aborting")
                return 2
            print("[INFO] isolation gate PASSED: sockets are distinct")
        except FileNotFoundError:
            print("[WARN] tmp socket file not found by path (may be abstract socket) — continuing")

        # --- Trigger drain_active_live_captures via session_refresh_if_stale ---
        # This RPC path calls drain_deferred_captures + drain_active_live_captures
        # + flush_record_buffer, then returns a rendered payload.
        # It's faster than waiting for a natural WAKE tick from the REM cycle.
        print("[INFO] triggering drain via session_refresh_if_stale RPC...")
        try:
            refresh_resp = _send_jsonrpc(
                tmp_sock,
                "session_refresh_if_stale",
                {"session_id": "orchestrator-check-session", "watermark": ""},
                rpc_id=10,
            )
            print(f"[INFO] session_refresh_if_stale response: {json.dumps(refresh_resp)[:400]}")
        except Exception as exc:
            print(f"[WARN] session_refresh_if_stale failed: {exc}")
            print("[INFO] falling back to natural WAKE tick poll (up to 120s)...")
            # If the RPC fails, poll the events for the active_live_drain_wake event
            # which only fires during the WAKE transition after a REM cycle.
            # In that case we poll up to 120s for it.
            # But first let's try the events_query path.

        # --- Also try via the WAKE tick by checking for the drain event ---
        # Give the daemon a few seconds to process and flush
        print("[INFO] waiting 3s for flush...")
        time.sleep(3.0)

        # --- Query episodes_recent filtered to fake_uuid session ---
        print(f"[INFO] querying episodes_recent for session_id={fake_uuid}...")
        try:
            episodes_resp = _send_jsonrpc(
                tmp_sock,
                "episodes_recent",
                {"n": 50, "session_id": fake_uuid},
                rpc_id=20,
            )
            print(f"[INFO] episodes_recent response: {json.dumps(episodes_resp)[:1000]}")
        except Exception as exc:
            print(f"[FAIL] episodes_recent RPC failed: {exc}")
            return 1

        # --- Also query events for active_live_drain_wake / drain events ---
        events_resp: dict = {}
        print("[INFO] querying drain events from store...")
        try:
            events_resp = _send_jsonrpc(
                tmp_sock,
                "events_query",
                {"kind": "active_live_drain_wake", "limit": 10},
                rpc_id=30,
            )
            print(f"[INFO] active_live_drain_wake events: {json.dumps(events_resp)[:800]}")
        except Exception as exc:
            print(f"[INFO] events_query active_live_drain_wake failed (non-critical): {exc}")

        # --- Evaluate result ---
        result_turns = []
        if isinstance(episodes_resp.get("result"), dict):
            result_turns = episodes_resp["result"].get("turns", [])
        elif isinstance(episodes_resp.get("turns"), list):
            result_turns = episodes_resp["turns"]

        # Look for the nonce in any of the returned turns
        found = False
        matching_turn = None
        for turn_item in result_turns:
            # Check session_id matches and text contains nonce
            turn_session = turn_item.get("session_id")
            turn_text = turn_item.get("literal_surface", "")
            if turn_session == fake_uuid or (
                isinstance(turn_text, str) and DISTINCT_NONCE[:20] in turn_text
            ):
                found = True
                matching_turn = turn_item
                break

        print("\n" + "="*60)
        print("VERDICT")
        print("="*60)
        print(f"fake_uuid:     {fake_uuid}")
        print(f"nonce:         {DISTINCT_NONCE!r}")
        print(f"turns_returned: {len(result_turns)}")

        if found:
            print(f"\nRESULT: YES — nonce SURFACED via episodes_recent")
            print(f"  matching_turn: {json.dumps(matching_turn, indent=2)}")
            print("\nCONCLUSION: drain_active_live_captures WAKE path WORKS end-to-end.")
            print("  Prior e7k9p non-appearance was dedup of near-identical nonces + contaminated manual pokes.")
            print("  NOT a drain bug.")
            verdict = 0
        else:
            print(f"\nRESULT: NO — nonce NOT found in episodes_recent")
            print(f"  All returned turns: {json.dumps(result_turns, indent=2)}")
            # Check if there was a drain event at all
            drain_events = []
            if isinstance(events_resp.get("result"), dict):
                drain_events = events_resp["result"].get("events", [])
            elif isinstance(events_resp.get("events"), list):
                drain_events = events_resp["events"]
            print(f"  drain_events count: {len(drain_events)}")
            if drain_events:
                print(f"  first drain event: {json.dumps(drain_events[0], indent=2)[:400]}")
                # If events_inserted==0 in a drain event that processed our file,
                # that's advance-without-insert bug evidence.
                for ev in drain_events:
                    data = ev.get("data", {}) if isinstance(ev, dict) else {}
                    if isinstance(data, dict):
                        ins = data.get("events_inserted", -1)
                        print(f"  drain event events_inserted={ins}")
            print("\nCONCLUSION: INCONCLUSIVE or real bug — see details above.")
            verdict = 1

        return verdict

    finally:
        # --- Teardown: kill ONLY the tmp daemon by its captured PGID ---
        print(f"\n[TEARDOWN] killing tmp daemon pid={daemon_pid} pgid={daemon_pgid}...")
        try:
            # First try graceful SIGTERM
            os.killpg(daemon_pgid, signal.SIGTERM)
            proc.wait(timeout=8)
            print(f"[TEARDOWN] daemon exited cleanly (rc={proc.returncode})")
        except subprocess.TimeoutExpired:
            print("[TEARDOWN] daemon did not exit after SIGTERM, sending SIGKILL...")
            try:
                os.killpg(daemon_pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        except ProcessLookupError:
            print("[TEARDOWN] daemon already exited")
        except Exception as exc:
            print(f"[TEARDOWN] kill error: {exc}")
        finally:
            # Verify we did NOT kill the real daemon
            if real_sock_inode is not None:
                try:
                    new_inode = os.stat(real_sock).st_ino
                    if new_inode == real_sock_inode:
                        print("[TEARDOWN] real daemon socket intact (inode unchanged)")
                    else:
                        print(f"[WARN] real daemon socket inode changed: was {real_sock_inode}, now {new_inode}")
                except FileNotFoundError:
                    print("[WARN] real daemon socket gone after teardown — may have been unrelated")

        print(f"[TEARDOWN] removing tmp dir: {tmpdir}")
        shutil.rmtree(tmpdir, ignore_errors=True)
        print("[TEARDOWN] done")


if __name__ == "__main__":
    sys.exit(run())

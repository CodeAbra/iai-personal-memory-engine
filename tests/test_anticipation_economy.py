"""The anticipation economy is countable, end to end.

Proving "the agent no longer searches" needs three instruments: every
agent-initiated search leaves a `recall_dispatched` event with its payload
weight; every pack build leaves a `foresight_pack_built` event with its token
estimate; every pack actually DELIVERED leaves a line in the serve ledger the
hook appends. The dashboard's /api/economy folds the three into the numbers
shown to the user (packs served, tokens injected, searches remaining, saved
tokens as a conservative lower bound).
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from iai_mcp import foresight
from iai_mcp.brainview import make_server
from iai_mcp.capture import capture_turn
from iai_mcp.events import query_events, write_event
from iai_mcp.store import MemoryStore, flush_record_buffer


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401, PLC0415
        except ImportError:
            pytest.skip("iai_mcp_native not built")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


@pytest.fixture(autouse=True)
def _foresight_env(monkeypatch):
    monkeypatch.setenv(foresight.FORESIGHT_MIN_COS_ENV, "0.45")
    monkeypatch.setenv(foresight.FORESIGHT_GOAL_WEIGHT_ENV, "0")
    from iai_mcp import working_tier

    working_tier._reset()
    yield
    working_tier._reset()


def _turn(store, text, session):
    result = capture_turn(
        store, cue="", text=text, tier="episodic",
        session_id=session, role="user", live_turn=True,
    )
    flush_record_buffer(store)
    return result


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_pack_build_leaves_a_countable_event(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    _turn(store, "The hippocampus consolidates memories during sleep.", "past")
    _turn(store, "How does hippocampal sleep consolidation work?", "today")

    from iai_mcp.events import flush_event_buffer
    flush_event_buffer(store)
    events = query_events(store, kind="foresight_pack_built", limit=10)
    assert events, f"[{driver}] pack build must be countable"
    data = events[0].get("data") or {}
    assert int(data.get("items") or 0) >= 1
    assert int(data.get("tokens_est") or 0) > 0


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_agent_search_leaves_a_countable_event(driver, tmp_path, monkeypatch):
    from iai_mcp import core as _core

    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    r = _turn(store, "Ashby's law: only variety can absorb variety.", "past")
    rec = store.get(__import__("uuid").UUID(r["record_id"]))

    _core.dispatch(store, "memory_recall", {
        "cue": "variety and control",
        "session_id": "probe",
        "cue_embedding": list(rec.embedding),
    })

    from iai_mcp.events import flush_event_buffer
    flush_event_buffer(store)
    events = query_events(store, kind="recall_dispatched", limit=10)
    assert events, f"[{driver}] an agent search must be countable"
    data = events[0].get("data") or {}
    assert int(data.get("resp_chars") or 0) > 0


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_economy_endpoint_folds_all_three_sources(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    for _ in range(3):
        write_event(store, "foresight_pack_built",
                    {"items": 2, "tokens_est": 100}, severity="info", buffered=False)
    for _ in range(2):
        write_event(store, "recall_dispatched",
                    {"hits": 5, "resp_chars": 4000}, severity="info", buffered=False)
    ledger = Path(store.root) / "logs" / "foresight-served.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        '{"ts":"2026-07-06T07:00:00Z","bytes":800}\n'
        '{"ts":"2026-07-06T07:01:00Z","bytes":1200}\n',
        encoding="utf-8",
    )

    server = make_server(store, port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/economy", timeout=10
        ) as resp:
            eco = json.loads(resp.read())
        assert eco["packs_built"] == 3
        assert eco["packs_served"] == 2
        assert eco["tokens_injected_est"] == 500  # (800+1200)/4
        assert eco["agent_searches"] == 2
        assert eco["avg_search_tokens_est"] == 1000  # 4000/4
        # 2 served * (220 overhead + (1000 observed - 250 avg pack) premium)
        assert eco["tokens_saved_lower_est"] == 1940
    finally:
        server.shutdown()


def test_hook_appends_serve_ledger(tmp_path):
    import os
    import subprocess

    hook = (
        Path(__file__).resolve().parents[1]
        / "src/iai_mcp/_deploy/hooks/iai-mcp-per-turn-recall.sh"
    )
    root = tmp_path / "iai-root"
    root.mkdir()
    pack = tmp_path / "pack.md"
    pack.write_text("- [episodic] served words\n", encoding="utf-8")
    env = dict(os.environ)
    env["IAI_MCP_STORE"] = str(root)
    env.pop("IAI_MCP_ROOT", None)
    env["IAI_MCP_FORESIGHT_PACK"] = str(pack)
    env["IAI_MCP_WORKING_TIER_CACHE"] = str(tmp_path / "absent.md")
    env.pop("IAI_MCP_PER_TURN_SOCKET_ACCEL", None)

    for _ in range(2):
        proc = subprocess.run(
            [str(hook)], input='{"prompt":"x"}', capture_output=True,
            text=True, env=env, timeout=10,
        )
        assert proc.returncode == 0
        assert "<iai-mcp-foresight>" in proc.stdout

    ledger = root / "logs" / "foresight-served.jsonl"
    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2, "one ledger line per delivered pack"
    row = json.loads(lines[0])
    assert row["bytes"] > 0 and row["ts"]


def _serve_one_recall_reply(sock_path: str, reply_obj: dict):
    reply_frame = (json.dumps(reply_obj) + "\n").encode("utf-8")
    accept_done = threading.Event()

    def _listener():
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(sock_path)
        srv.listen(1)
        srv.settimeout(2.0)
        try:
            conn, _ = srv.accept()
            try:
                conn.settimeout(2.0)
                buf = b""
                while b"\n" not in buf:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
                conn.sendall(reply_frame)
            finally:
                conn.close()
        except OSError:
            pass
        finally:
            srv.close()
            accept_done.set()

    threading.Thread(target=_listener, daemon=True).start()
    time.sleep(0.05)
    return accept_done


def test_socket_recall_fires_by_default_without_the_env_var():
    """The durable template default is ON: an unset IAI_MCP_PER_TURN_SOCKET_ACCEL
    must still fire the live socket recall when the daemon socket is present."""
    import shutil
    import tempfile

    hook = (
        Path(__file__).resolve().parents[1]
        / "src/iai_mcp/_deploy/hooks/iai-mcp-per-turn-recall.sh"
    )
    # AF_UNIX sun_path is capped near 104 bytes; pytest's own tmp_path nests
    # deep enough to overflow it, so use a short-lived /tmp dir instead.
    root = Path(tempfile.mkdtemp(dir="/tmp", prefix="iai-recall-"))
    try:
        sock_path = str(root / ".daemon.sock")
        reply = {
            "jsonrpc": "2.0", "id": 1,
            "result": {"hits": [
                {"record_id": "r1", "literal_surface": "alice prefers dark mode",
                 "valid_to": None},
            ]},
        }
        done = _serve_one_recall_reply(sock_path, reply)

        env = dict(os.environ)
        env["IAI_MCP_STORE"] = str(root)
        env.pop("IAI_MCP_ROOT", None)
        env.pop("IAI_MCP_PER_TURN_SOCKET_ACCEL", None)  # unset -> exercise the durable default
        env["IAI_MCP_FORESIGHT_PACK"] = str(root / "absent-pack.md")
        env["IAI_MCP_WORKING_TIER_CACHE"] = str(root / "absent-wt.md")

        proc = subprocess.run(
            [str(hook)], input='{"prompt":"remind me about alice"}',
            capture_output=True, text=True, env=env, timeout=10,
        )
        done.wait(timeout=2)
        assert proc.returncode == 0
        assert "<iai-mcp-recall>" in proc.stdout
        assert "alice prefers dark mode" in proc.stdout
    finally:
        shutil.rmtree(str(root), ignore_errors=True)


def test_socket_recall_still_disableable_via_explicit_zero():
    """Setting the env var to '0' still opts a caller out — the flip changes
    the default, not the ability to turn the accelerator off."""
    import shutil
    import tempfile

    hook = (
        Path(__file__).resolve().parents[1]
        / "src/iai_mcp/_deploy/hooks/iai-mcp-per-turn-recall.sh"
    )
    root = Path(tempfile.mkdtemp(dir="/tmp", prefix="iai-recall-"))
    try:
        sock_path = str(root / ".daemon.sock")
        reply = {
            "jsonrpc": "2.0", "id": 1,
            "result": {"hits": [
                {"record_id": "r1", "literal_surface": "alice prefers dark mode",
                 "valid_to": None},
            ]},
        }
        done = _serve_one_recall_reply(sock_path, reply)

        env = dict(os.environ)
        env["IAI_MCP_STORE"] = str(root)
        env.pop("IAI_MCP_ROOT", None)
        env["IAI_MCP_PER_TURN_SOCKET_ACCEL"] = "0"
        env["IAI_MCP_FORESIGHT_PACK"] = str(root / "absent-pack.md")
        env["IAI_MCP_WORKING_TIER_CACHE"] = str(root / "absent-wt.md")

        proc = subprocess.run(
            [str(hook)], input='{"prompt":"remind me about alice"}',
            capture_output=True, text=True, env=env, timeout=10,
        )
        assert proc.returncode == 0
        assert "<iai-mcp-recall>" not in proc.stdout
        done.set()
    finally:
        shutil.rmtree(str(root), ignore_errors=True)


def test_savings_floor_without_observed_searches():
    """Foresight working perfectly means ZERO observed searches — the floor
    must still credit the avoided call overhead, not collapse to 0."""
    from iai_mcp.brainview import SEARCH_CALL_OVERHEAD_TOK, _savings_lower_est

    assert _savings_lower_est(25, 33324, 0) == 25 * SEARCH_CALL_OVERHEAD_TOK
    assert _savings_lower_est(0, 0, 0) == 0
    # observed premium adds on top of the overhead floor
    assert _savings_lower_est(2, 2000, 1000) == 2 * SEARCH_CALL_OVERHEAD_TOK + 2 * (1000 - 250)


def test_deployed_per_turn_hook_and_render_helper_stay_byte_identical_to_template(
    tmp_path, monkeypatch,
):
    """An old hook running next to a stale or missing render helper is a
    live-failure class: the dated-staleness signal silently disappears.
    Prove the installer keeps both files byte-identical to the tracked
    template, using a throwaway HOME -- never ~/.claude."""
    import argparse

    repo_root = Path(__file__).resolve().parents[1]
    hook_template = repo_root / "src/iai_mcp/_deploy/hooks/iai-mcp-per-turn-recall.sh"
    render_template = repo_root / "src/iai_mcp/_deploy/hooks/_recall_render.py"

    scratch_home = tmp_path / "scratch-home"
    scratch_home.mkdir()
    monkeypatch.setenv("HOME", str(scratch_home))

    from iai_mcp.cli._capture import cmd_capture_hooks_install

    rc = cmd_capture_hooks_install(argparse.Namespace(target="claude"))
    assert rc == 0

    deployed_hook = scratch_home / ".claude" / "hooks" / "iai-mcp-per-turn-recall.sh"
    deployed_render = scratch_home / ".claude" / "hooks" / "_recall_render.py"
    assert deployed_hook.read_bytes() == hook_template.read_bytes(), (
        "deployed per-turn hook diverged from the tracked template"
    )
    assert deployed_render.read_bytes() == render_template.read_bytes(), (
        "deployed render helper diverged from the tracked template"
    )


def test_per_turn_recall_block_stays_within_existing_budget():
    """The dated-staleness render must not blow the pre-existing 3-hit /
    [:400] per-turn injection budget. Invariant #3 bounds SessionStart;
    per-turn injection has its own, smaller bound this guards so the
    render change cannot grow it unbounded."""
    import sys

    hooks_dir = Path(__file__).resolve().parents[1] / "src/iai_mcp/_deploy/hooks"
    if str(hooks_dir) not in sys.path:
        sys.path.insert(0, str(hooks_dir))
    from _recall_render import render_recall_block

    hits = [
        {"record_id": f"r{i}", "literal_surface": "x" * 500,
         "valid_to": "2000-01-01T00:00:00+00:00"}
        for i in range(5)
    ]
    result = {
        "hits": hits,
        "anti_hits": [
            {"record_id": "a1", "literal_surface": "y" * 500,
             "valid_to": "2000-01-01T00:00:00+00:00"},
        ],
    }
    block = render_recall_block(result)
    assert block.count("\n- ") <= 3, "must stay within the existing 3-hit budget"
    assert len(block.encode("utf-8")) < 2048, (
        f"per-turn recall block grew unbounded: {len(block)} bytes"
    )

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pytest


@pytest.fixture
def iai_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-freshness-trigger-passphrase")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))

    import keyring.core

    keyring.core._keyring_backend = None
    yield tmp_path
    keyring.core._keyring_backend = None


def _open_store(home: Path):
    from iai_mcp.store import MemoryStore

    return MemoryStore(path=home / ".iai-mcp")


def _insert_record(store, text: str):
    from iai_mcp.capture import capture_turn
    from iai_mcp.store import flush_record_buffer

    result = capture_turn(store, text=text, cue="", tier="episodic", role="user")
    flush_record_buffer(store)
    return result


def _write_drainable_deferred(home: Path, session_id: str, text: str) -> Path:
    deferred_dir = home / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    suffix = int(time.time())
    out = deferred_dir / f"{session_id}-{suffix}.jsonl"
    header = {
        "version": 1,
        "deferred_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "cwd": "/tmp",
    }
    event = {
        "text": text,
        "cue": f"session {session_id} deferred cue",
        "tier": "episodic",
        "role": "user",
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    out.write_text(
        json.dumps(header, ensure_ascii=False) + "\n"
        + json.dumps(event, ensure_ascii=False) + "\n"
    )
    return out


def test_watermark_round_trip(iai_home):
    from iai_mcp.cli import read_watermark, write_watermark

    ts = "2026-05-29T10:00:00+00:00"
    write_watermark("test-session", ts)
    result = read_watermark("test-session")
    assert result is not None
    dt_stored = datetime.fromisoformat(result.replace("Z", "+00:00"))
    dt_orig = datetime.fromisoformat(ts)
    assert abs((dt_stored - dt_orig).total_seconds()) < 1


def test_read_watermark_absent(iai_home):
    from iai_mcp.cli import read_watermark

    assert read_watermark("nonexistent-session-xyz") is None


def test_baseline_on_first(iai_home, monkeypatch):
    from iai_mcp import cli

    store = _open_store(iai_home)
    _insert_record(store, "alice wrote the tokenizer module")

    rpc_calls: list = []

    def fake_rpc(method, params, **_kw):
        rpc_calls.append((method, params))
        return None

    monkeypatch.setattr(cli, "_send_jsonrpc_request", fake_rpc)

    captured = StringIO()
    monkeypatch.setattr(sys, "stdout", captured)

    import argparse

    args = argparse.Namespace(session_id="baseline-session")
    rc = cli.cmd_session_refresh_if_stale(args)

    assert rc == 0
    assert rpc_calls == [], "RPC must not be called on the first (baseline) prompt"
    assert captured.getvalue() == "", "No additionalContext on first prompt"

    from iai_mcp.cli import read_watermark

    wm = read_watermark("baseline-session")
    assert wm is not None


def test_no_trigger_when_not_newer(iai_home, monkeypatch):
    from iai_mcp import cli
    from iai_mcp.cli import write_watermark

    store = _open_store(iai_home)
    _insert_record(store, "alice refactored the parser")

    from iai_mcp.session import max_record_created_at

    current_max = max_record_created_at(store)
    assert current_max is not None
    write_watermark("same-session", current_max)

    rpc_calls: list = []

    def fake_rpc(method, params, **_kw):
        rpc_calls.append((method, params))
        return None

    monkeypatch.setattr(cli, "_send_jsonrpc_request", fake_rpc)

    captured = StringIO()
    monkeypatch.setattr(sys, "stdout", captured)

    import argparse

    args = argparse.Namespace(session_id="same-session")
    rc = cli.cmd_session_refresh_if_stale(args)

    assert rc == 0
    assert rpc_calls == [], "RPC must not fire when nothing new exists"
    assert captured.getvalue() == ""


def test_trigger_when_newer(iai_home, monkeypatch):
    from iai_mcp import cli
    from iai_mcp.cli import read_watermark, write_watermark
    from iai_mcp.session import max_record_created_at

    store = _open_store(iai_home)

    _insert_record(store, "alice shipped the tokenizer module for the compiler pipeline")
    old_max = max_record_created_at(store)
    assert old_max is not None
    write_watermark("trigger-session", old_max)

    time.sleep(0.2)
    r2 = _insert_record(store, "chlorophyll absorbs red and blue light to drive the light reactions")
    assert r2.get("status") == "inserted", f"Second insert status: {r2}"
    new_max = max_record_created_at(store)
    assert new_max is not None
    assert new_max > old_max

    new_max_ts_returned = new_max

    rpc_calls: list = []

    def fake_rpc(method, params, **_kw):
        rpc_calls.append((method, params))
        return {"result": {"rendered": "## Memory refreshed\n\nalice shipped the parser refactor", "new_max_ts": new_max_ts_returned}}

    monkeypatch.setattr(cli, "_send_jsonrpc_request", fake_rpc)

    captured = StringIO()
    monkeypatch.setattr(sys, "stdout", captured)

    import argparse

    args = argparse.Namespace(session_id="trigger-session")
    rc = cli.cmd_session_refresh_if_stale(args)

    assert rc == 0
    assert len(rpc_calls) == 1, "RPC must be called exactly once"
    method, params = rpc_calls[0]
    assert method == "session_refresh_if_stale"
    from iai_mcp.cli import _utc_iso
    assert _utc_iso(params["watermark"]) == _utc_iso(old_max)
    assert params["session_id"] == "trigger-session"

    out = captured.getvalue()
    assert out != "", "additionalContext JSON must be emitted"
    payload = json.loads(out)
    assert "hookSpecificOutput" in payload
    assert payload["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "alice shipped the parser refactor" in payload["hookSpecificOutput"]["additionalContext"]

    wm_after = read_watermark("trigger-session")
    assert wm_after is not None
    from iai_mcp.cli import _utc_iso
    assert _utc_iso(wm_after) == _utc_iso(new_max_ts_returned)


def test_daemon_down(iai_home, monkeypatch):
    from iai_mcp import cli
    from iai_mcp.cli import read_watermark, write_watermark
    from iai_mcp.session import max_record_created_at

    store = _open_store(iai_home)
    _insert_record(store, "alice added the event bus")
    old_max = max_record_created_at(store)
    assert old_max is not None

    old_wm = "2020-01-01T00:00:00+00:00"
    write_watermark("daemon-down-session", old_wm)

    monkeypatch.setattr(cli, "_send_jsonrpc_request", lambda *_a, **_kw: None)

    captured = StringIO()
    monkeypatch.setattr(sys, "stdout", captured)

    import argparse

    args = argparse.Namespace(session_id="daemon-down-session")
    rc = cli.cmd_session_refresh_if_stale(args)

    assert rc == 0
    assert captured.getvalue() == "", "No output when daemon is down"

    wm_after = read_watermark("daemon-down-session")
    assert wm_after is not None
    from iai_mcp.cli import _utc_iso
    assert _utc_iso(wm_after) == _utc_iso(old_wm)


def test_utc_normalization(iai_home, monkeypatch):
    from iai_mcp import cli
    from iai_mcp.cli import _utc_iso, write_watermark

    ts_z = "2026-05-29T12:00:00Z"
    ts_offset = "2026-05-29T12:00:00+00:00"
    assert _utc_iso(ts_z) == _utc_iso(ts_offset), "Z and +00:00 must normalize identically"

    ts_positive = "2026-05-29T14:00:00+02:00"
    assert _utc_iso(ts_positive) == _utc_iso(ts_z)

    from iai_mcp.session import max_record_created_at

    store = _open_store(iai_home)
    _insert_record(store, "alice checked in the event log")
    current_max = max_record_created_at(store)
    assert current_max is not None

    if current_max.endswith("+00:00"):
        wm_alt = current_max.replace("+00:00", "Z")
    else:
        wm_alt = current_max.replace("Z", "+00:00") if current_max.endswith("Z") else current_max

    write_watermark("tz-session", wm_alt)

    rpc_calls: list = []
    monkeypatch.setattr(cli, "_send_jsonrpc_request", lambda *a, **kw: rpc_calls.append(a) or None)

    import argparse

    captured = StringIO()
    monkeypatch.setattr(sys, "stdout", captured)
    args = argparse.Namespace(session_id="tz-session")
    cli.cmd_session_refresh_if_stale(args)

    assert rpc_calls == [], "Same instant in different TZ format must NOT trigger"


def test_sc3_end_to_end(iai_home, monkeypatch):
    from iai_mcp import cli
    from iai_mcp.cli import write_watermark
    from iai_mcp.session import max_record_created_at

    store = _open_store(iai_home)

    _insert_record(store, "alice set up the project scaffolding")
    baseline_max = max_record_created_at(store)
    assert baseline_max is not None
    write_watermark("sc3-session", baseline_max)

    time.sleep(0.2)

    r2 = _insert_record(store, "mitochondria produce ATP through oxidative phosphorylation in the inner membrane")
    assert r2.get("status") == "inserted", f"OOB insert must be a new record: {r2}"

    new_max = max_record_created_at(store)
    assert new_max is not None
    assert new_max > baseline_max, "OOB record must have a newer created_at"

    def in_process_rpc(method: str, params: dict, **_kw) -> dict:
        from iai_mcp.core import dispatch as core_dispatch
        result = core_dispatch(store, method, params)
        return {"result": result}

    monkeypatch.setattr(cli, "_send_jsonrpc_request", in_process_rpc)

    captured = StringIO()
    monkeypatch.setattr(sys, "stdout", captured)

    import argparse

    args = argparse.Namespace(session_id="sc3-session")
    rc = cli.cmd_session_refresh_if_stale(args)

    assert rc == 0
    out = captured.getvalue()
    assert out != "", "additionalContext must be emitted on SC3 trigger"

    payload = json.loads(out)
    assert "hookSpecificOutput" in payload
    ctx = payload["hookSpecificOutput"]["additionalContext"]
    assert ctx, "additionalContext must not be empty"
    assert "mitochondria" in ctx or "oxidative phosphorylation" in ctx, (
        f"OOB record text not found in additionalContext:\n{ctx[:500]}"
    )


def test_hook_shape_regression():
    hook_path = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "_deploy" / "hooks" / "iai-mcp-turn-capture.sh"
    assert hook_path.exists(), f"Hook not found at {hook_path}"

    content = hook_path.read_text()

    assert ".live.jsonl" in content, (
        "Hook must still write to .live.jsonl (per-turn capture must not be removed)"
    )

    assert "session_refresh_if_stale" in content, (
        "Hook must contain the inlined gate using the session_refresh_if_stale RPC method"
    )
    assert "additionalContext" in content, (
        "Hook must emit additionalContext JSON to stdout when the gate triggers"
    )

    assert "MemoryStore(" not in content, (
        "Hook must NOT open a MemoryStore — all store mutations are daemon-owned (Pitfall 7)"
    )
    assert "drain_deferred_captures" not in content, (
        "Hook must NOT call drain_deferred_captures directly — call via daemon RPC (Pitfall 7)"
    )


def _write_live_file(home: Path, session_id: str, texts: list) -> Path:
    deferred_dir = home / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    live = deferred_dir / f"{session_id}.live.jsonl"
    header = {
        "version": 1,
        "deferred_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "cwd": "/tmp",
    }
    with live.open("w") as fh:
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        for text in texts:
            ev = {
                "text": text,
                "cue": f"session {session_id} live turn",
                "tier": "episodic",
                "role": "user",
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
    return live


def test_drain_active_live_b_still_open(iai_home):
    from iai_mcp.capture import drain_active_live_captures

    store = _open_store(iai_home)

    b_session = "session-b-live"
    live_file = _write_live_file(
        iai_home,
        b_session,
        ["alice completed the live-file parser feature"],
    )

    counts = drain_active_live_captures(store, exclude_session_id="session-a-refresh")

    assert counts["events_inserted"] >= 1, f"Expected at least 1 insert, got {counts}"

    assert live_file.exists(), ".live.jsonl must NOT be deleted during active drain"

    offset_path = iai_home / ".iai-mcp" / ".capture-state" / f"{b_session}.drain-offset"
    assert offset_path.exists(), ".drain-offset sidecar must be written"
    offset_val = int(offset_path.read_text().strip())
    assert offset_val >= 1, "drain-offset must reflect the number of events drained"


def test_drain_active_live_idempotency(iai_home):
    from iai_mcp.capture import drain_active_live_captures
    from iai_mcp.session import max_record_created_at
    from iai_mcp.store import flush_record_buffer

    store = _open_store(iai_home)
    b_session = "session-b-idem"
    _write_live_file(iai_home, b_session, ["alice drafted the idempotency contract"])

    c1 = drain_active_live_captures(store, exclude_session_id="session-a")
    flush_record_buffer(store)
    assert c1["events_inserted"] >= 1

    record_count_after_first = store.db.open_table("records").count_rows()
    assert record_count_after_first >= 1, "First drain must land rows in SQLite"
    max_after_first = max_record_created_at(store)

    c2 = drain_active_live_captures(store, exclude_session_id="session-a")
    flush_record_buffer(store)
    assert c2["events_inserted"] == 0, "Second drain must insert nothing (offset honored)"
    assert c2["events_reinforced"] == 0, "Second drain must reinforce nothing either"

    assert store.db.open_table("records").count_rows() == record_count_after_first
    assert max_record_created_at(store) == max_after_first


def test_drain_active_live_no_self_drain(iai_home):
    from iai_mcp.capture import drain_active_live_captures

    store = _open_store(iai_home)
    a_session = "session-a-self"
    _write_live_file(iai_home, a_session, ["alice wrote a self-referential turn"])

    counts = drain_active_live_captures(store, exclude_session_id=a_session)

    assert counts["events_inserted"] == 0, "Own .live.jsonl must NOT be drained"

    live_file = iai_home / ".iai-mcp" / ".deferred-captures" / f"{a_session}.live.jsonl"
    assert live_file.exists(), ".live.jsonl must not be deleted"


def test_drain_active_live_no_double_insert(iai_home):
    from iai_mcp.capture import drain_active_live_captures, drain_deferred_captures
    from iai_mcp.store import flush_record_buffer

    store = _open_store(iai_home)
    b_session = "session-b-nodup"

    live_file = _write_live_file(
        iai_home,
        b_session,
        ["alice finalized the dedup contract logic"],
    )
    counts_live = drain_active_live_captures(store, exclude_session_id="session-a")
    flush_record_buffer(store)
    assert counts_live["events_inserted"] >= 1

    record_count_after_live = store.db.open_table("records").count_rows()
    assert record_count_after_live >= 1, (
        "Live drain did not land any rows in SQLite — test would be vacuous"
    )

    epoch = int(time.time())
    ended_file = live_file.parent / f"{b_session}.live-{epoch}.jsonl"
    live_file.rename(ended_file)

    counts_norm = drain_deferred_captures(store)
    flush_record_buffer(store)

    assert counts_norm.get("events_reinforced", 0) >= 1, (
        f"Expected at least one reinforcement from cos>=0.95 dedup, got: {counts_norm}"
    )
    assert counts_norm.get("events_inserted", 0) == 0, (
        f"Normal drain must not insert duplicate records: {counts_norm}"
    )

    record_count_after_normal = store.db.open_table("records").count_rows()
    assert record_count_after_normal == record_count_after_live, (
        f"Normal drain after live-drain must not add records: "
        f"{record_count_after_live} -> {record_count_after_normal}"
    )


def test_sc3_b_still_open_surfaces_via_refresh(iai_home, monkeypatch):
    from iai_mcp import cli
    from iai_mcp.cli import write_watermark
    from iai_mcp.session import max_record_created_at

    store = _open_store(iai_home)

    _insert_record(store, "alice seeded the store for cross-session test")
    baseline_max = max_record_created_at(store)
    assert baseline_max is not None
    write_watermark("session-a-trigger", baseline_max)

    b_session = "session-b-open"
    _write_live_file(
        iai_home,
        b_session,
        ["alice shipped the live-file cross-session feature in session B"],
    )

    time.sleep(0.2)
    r_trigger = _insert_record(store, "photosynthesis converts carbon dioxide and water into glucose using sunlight")
    assert r_trigger.get("status") == "inserted", f"Trigger insert must be a new record: {r_trigger}"
    new_store_max = max_record_created_at(store)
    assert new_store_max is not None
    assert new_store_max > baseline_max

    def in_process_rpc(method: str, params: dict, **_kw) -> dict:
        from iai_mcp.core import dispatch as core_dispatch
        result = core_dispatch(store, method, params)
        return {"result": result}

    monkeypatch.setattr(cli, "_send_jsonrpc_request", in_process_rpc)

    captured = StringIO()
    monkeypatch.setattr(sys, "stdout", captured)

    import argparse
    args = argparse.Namespace(session_id="session-a-trigger")
    rc = cli.cmd_session_refresh_if_stale(args)

    assert rc == 0
    out = captured.getvalue()
    assert out != "", "additionalContext must be emitted when B's live turn is drained"

    payload = json.loads(out)
    ctx = payload["hookSpecificOutput"]["additionalContext"]
    assert ctx, "additionalContext must not be empty"

    assert "alice shipped the live-file cross-session feature in session B" in ctx or \
           "live-file cross-session" in ctx or \
           "session B" in ctx, (
        f"B's live-file turn not found in brief:\n{ctx[:600]}"
    )

    live_b = iai_home / ".iai-mcp" / ".deferred-captures" / f"{b_session}.live.jsonl"
    assert live_b.exists(), "B's .live.jsonl must survive the active drain"

    offset_b = iai_home / ".iai-mcp" / ".capture-state" / f"{b_session}.drain-offset"
    assert offset_b.exists(), "drain-offset for B must be recorded after live drain"


def test_live_growth_only_trips_gate(iai_home, monkeypatch):
    from iai_mcp import cli
    from iai_mcp.cli import (
        read_live_fingerprint,
        write_live_fingerprint,
        write_watermark,
    )
    from iai_mcp.session import max_record_created_at

    store = _open_store(iai_home)
    _insert_record(store, "alice seeded the store for live-growth gate test")
    baseline_max = max_record_created_at(store)
    assert baseline_max is not None

    write_watermark("session-a-lg", baseline_max)

    write_live_fingerprint("session-a-lg", 0)

    b_session = "session-b-lg"
    _write_live_file(
        iai_home,
        b_session,
        ["alice completed the live-growth gate feature for cross-session continuity"],
    )

    rpc_calls: list = []

    def fake_rpc(method, params, **_kw):
        rpc_calls.append((method, params))
        return {
            "result": {
                "rendered": "## Memory refreshed\n\nalice live-growth gate fired",
                "new_max_ts": baseline_max,
            }
        }

    monkeypatch.setattr(cli, "_send_jsonrpc_request", fake_rpc)

    captured = __import__("io").StringIO()
    monkeypatch.setattr(__import__("sys"), "stdout", captured)

    import argparse

    args = argparse.Namespace(session_id="session-a-lg")
    rc = cli.cmd_session_refresh_if_stale(args)

    assert rc == 0
    assert len(rpc_calls) == 1, "Gate must trip (live growth) and send RPC"
    assert captured.getvalue() != "", "additionalContext must be emitted on live-growth trigger"

    fp_after = read_live_fingerprint("session-a-lg")
    assert fp_after is not None
    live_size_now = cli.get_other_sessions_live_size("session-a-lg")
    assert fp_after == live_size_now, "Fingerprint must advance to current live size after refresh"


def test_live_growth_idempotent(iai_home, monkeypatch):
    from iai_mcp import cli
    from iai_mcp.cli import write_live_fingerprint, write_watermark
    from iai_mcp.session import max_record_created_at

    store = _open_store(iai_home)
    _insert_record(store, "alice seeded the store for live-growth idempotency test")
    baseline_max = max_record_created_at(store)
    assert baseline_max is not None

    b_session = "session-b-idem-lg"
    _write_live_file(
        iai_home,
        b_session,
        ["alice wrote the idempotency contract for live-growth gating"],
    )

    write_watermark("session-a-idem-lg", baseline_max)
    current_live_size = cli.get_other_sessions_live_size("session-a-idem-lg")
    write_live_fingerprint("session-a-idem-lg", current_live_size)

    rpc_calls: list = []
    monkeypatch.setattr(
        cli,
        "_send_jsonrpc_request",
        lambda *a, **kw: rpc_calls.append(a) or {"result": {"rendered": "x", "new_max_ts": baseline_max}},
    )

    import argparse
    captured = __import__("io").StringIO()
    monkeypatch.setattr(__import__("sys"), "stdout", captured)

    args = argparse.Namespace(session_id="session-a-idem-lg")
    rc = cli.cmd_session_refresh_if_stale(args)

    assert rc == 0
    assert rpc_calls == [], (
        "Gate must NOT trip when live size is unchanged and store MAX is unchanged"
    )
    assert captured.getvalue() == "", "No additionalContext on idempotent check"


def test_no_self_trigger_own_live(iai_home, monkeypatch):
    from iai_mcp import cli
    from iai_mcp.cli import write_live_fingerprint, write_watermark
    from iai_mcp.session import max_record_created_at

    store = _open_store(iai_home)
    _insert_record(store, "alice seeded the store for self-trigger exclusion test")
    baseline_max = max_record_created_at(store)
    assert baseline_max is not None

    session_a = "session-a-self-lg"
    write_watermark(session_a, baseline_max)
    write_live_fingerprint(session_a, 0)

    own_live = (
        iai_home / ".iai-mcp" / ".deferred-captures" / f"{session_a}.live.jsonl"
    )
    own_live.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "version": 1,
        "deferred_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "session_id": session_a,
        "cwd": "/tmp",
    }
    import json as _json
    with own_live.open("w") as fh:
        fh.write(_json.dumps(header) + "\n")
        ev = {"text": "alice wrote her own turn", "cue": "", "tier": "episodic", "role": "user", "ts": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()}
        fh.write(_json.dumps(ev) + "\n")

    rpc_calls: list = []
    monkeypatch.setattr(
        cli,
        "_send_jsonrpc_request",
        lambda *a, **kw: rpc_calls.append(a) or None,
    )

    import argparse
    captured = __import__("io").StringIO()
    monkeypatch.setattr(__import__("sys"), "stdout", captured)

    args = argparse.Namespace(session_id=session_a)
    rc = cli.cmd_session_refresh_if_stale(args)

    assert rc == 0
    assert rpc_calls == [], (
        "Own session's .live.jsonl growth must NOT trip the gate (no self-trigger)"
    )
    assert captured.getvalue() == ""


def test_first_prompt_live_fingerprint_baseline(iai_home, monkeypatch):
    from iai_mcp import cli
    from iai_mcp.cli import read_live_fingerprint, write_watermark
    from iai_mcp.session import max_record_created_at

    store = _open_store(iai_home)
    _insert_record(store, "alice seeded the store for first-prompt fingerprint test")
    baseline_max = max_record_created_at(store)
    assert baseline_max is not None

    write_watermark("session-a-fp-first", baseline_max)

    b_session = "session-b-fp-first"
    _write_live_file(
        iai_home,
        b_session,
        ["alice wrote a pre-existing B turn before A's first fingerprint check"],
    )

    rpc_calls: list = []
    monkeypatch.setattr(
        cli,
        "_send_jsonrpc_request",
        lambda *a, **kw: rpc_calls.append(a) or None,
    )

    import argparse
    captured = __import__("io").StringIO()
    monkeypatch.setattr(__import__("sys"), "stdout", captured)

    args = argparse.Namespace(session_id="session-a-fp-first")
    rc = cli.cmd_session_refresh_if_stale(args)

    assert rc == 0
    assert rpc_calls == [], (
        "First look at live files (no fingerprint sidecar) must NOT trigger — "
        "it sets the baseline instead"
    )
    assert captured.getvalue() == ""

    fp = read_live_fingerprint("session-a-fp-first")
    assert fp is not None, "Fingerprint sidecar must be written on first look"
    expected = cli.get_other_sessions_live_size("session-a-fp-first")
    assert fp == expected, "Fingerprint must equal current live size"

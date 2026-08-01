from __future__ import annotations

import json
import sys

from iai_mcp.capture import capture_turn, drain_permanent_failed_files
from tests._helpers import make_tmp_store


def test_tem_import_guard(tmp_path, monkeypatch):
    store = make_tmp_store(tmp_path)

    original = sys.modules.get("iai_mcp.tem", _SENTINEL := object())
    sys.modules["iai_mcp.tem"] = None  # type: ignore[assignment]
    try:
        result = capture_turn(
            store,
            cue="tem guard test cue",
            text="tem import guard regression test text turnmark59 unique content",
            tier="episodic",
            session_id="sess-tem-guard",
            role="user",
        )
        assert result["status"] == "inserted", (
            f"REQ-3: store.insert must survive ImportError from iai_mcp.tem "
            f"when the guard is present; got status={result['status']!r} "
            f"reason={result.get('reason')!r}. "
            "The tem import must be wrapped in try/except ImportError: pass."
        )
    finally:
        if original is _SENTINEL:
            sys.modules.pop("iai_mcp.tem", None)
        else:
            sys.modules["iai_mcp.tem"] = original  # type: ignore[assignment]


def test_capture_turn_inserts_in_clean_env(tmp_path):
    store = make_tmp_store(tmp_path)
    result = capture_turn(
        store,
        cue="clean env smoke test",
        text="clean env smoke test phrase turnmark59 unique content abc123",
        tier="episodic",
        session_id="sess-clean-env",
        role="user",
    )
    assert result["status"] == "inserted", (
        f"capture_turn failed in clean env: {result!r}"
    )


def test_drain_permanent_failed_reingests(tmp_path, monkeypatch):
    store = make_tmp_store(tmp_path)

    captures_dir = tmp_path / "deferred-captures"
    captures_dir.mkdir()

    session_id = "sess-recovery-test"
    genuine_text = "recovery test phrase zzz7777 should land in store"
    event = {
        "type": "user",
        "message": {"role": "user", "content": genuine_text},
        "session_id": session_id,
    }
    pf_file = captures_dir / f".permanent-failed-20260530T120000.jsonl"
    pf_file.write_text(json.dumps(event) + "\n", encoding="utf-8")

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "hippo"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "test.sock"))

    drained = drain_permanent_failed_files(store, deferred_dir=captures_dir)

    records = store.all_records()
    matching = [r for r in records if genuine_text in r.literal_surface]
    assert matching, (
        f"REQ-3: permanent-failed re-drain must ingest genuine turn into store; "
        f"'{genuine_text}' not found. drain result: {drained!r}"
    )

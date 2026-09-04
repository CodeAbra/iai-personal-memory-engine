from __future__ import annotations

import asyncio
import json
import platform
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX paths + UNIX socket semantics",
)


@pytest.fixture
def iai_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-forced-sleep-drain-passphrase")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))
    import keyring.core
    keyring.core._keyring_backend = None
    yield tmp_path
    keyring.core._keyring_backend = None


def _open_store():
    from iai_mcp.store import MemoryStore
    return MemoryStore()


def _write_live_file(
    deferred_dir: Path,
    session_id: str,
    events: list[dict],
) -> Path:
    deferred_dir.mkdir(parents=True, exist_ok=True)
    path = deferred_dir / f"{session_id}.live.jsonl"
    header = {
        "version": 1,
        "deferred_at": "2026-05-31T04:45:00.000000+00:00",
        "session_id": session_id,
        "cwd": "/tmp/test",
    }
    lines = [json.dumps(header, ensure_ascii=False)]
    for ev in events:
        lines.append(json.dumps(ev, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_forced_sleep_edge_drains_live_spool_into_records(driver, iai_home, monkeypatch):
    """A forced WAKE->SLEEP collapse (force_rem / user_sleep_request
    dispatching FORCE_SLEEP twice in one lifecycle tick) must drain a live
    spool turn into records, on both storage drivers."""
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")

    if driver == "lilli":
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)

    from iai_mcp.capture import drain_capture_backlog
    from iai_mcp.daemon import _maybe_drain_on_lifecycle_edge
    from iai_mcp.events import write_event
    from iai_mcp.lifecycle_state import LifecycleState

    session = f"forced-sleep-drain-session-{driver}"
    nonce = f"forced sleep edge drain nonce marker for {driver} driver test content"

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    _write_live_file(
        deferred_dir,
        session,
        [
            {
                "text": nonce,
                "cue": f"session {session} turn",
                "tier": "episodic",
                "role": "user",
                "ts": "2026-05-31T04:45:43.000000+00:00",
            },
        ],
    )

    store = _open_store()
    try:
        fired = asyncio.run(
            _maybe_drain_on_lifecycle_edge(
                LifecycleState.WAKE,
                LifecycleState.SLEEP,
                store,
                drain_fn=drain_capture_backlog,
                write_event_fn=write_event,
                queue_drain_fn=lambda store, *, write_event_fn, phase=None: None,
            )
        )
        assert fired is True

        turns = store.recent_user_turns(50, session_id=session)
        assert len(turns) >= 1, (
            f"[driver={driver}] recent_user_turns returned {len(turns)} turns; "
            "expected >= 1 after forced-sleep-edge drain"
        )
        texts = [t.literal_surface for t in turns]
        assert any(nonce in (t or "") for t in texts), (
            f"[driver={driver}] nonce not found in recent_user_turns; got: {texts!r}"
        )
    finally:
        store.close()

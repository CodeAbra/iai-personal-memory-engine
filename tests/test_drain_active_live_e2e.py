from __future__ import annotations

import json
import platform
from pathlib import Path

import pytest

from iai_mcp.store import MemoryStore


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX paths + UNIX socket semantics",
)


@pytest.fixture
def iai_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-live-drain-e2e-passphrase")
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


def test_drain_active_live_nonce_surfaces(iai_home):
    from iai_mcp.capture import drain_active_live_captures

    B_SESSION = "7173b585-291f-43e4-96b7-80f3a45e9e14"
    NONCE = "e7k9p cross-session ambient nonce live drain e2e"
    A_SESSION = "aaaabbbb-cccc-dddd-eeee-ffffffffffff"

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    _write_live_file(
        deferred_dir,
        B_SESSION,
        [
            {
                "text": NONCE,
                "cue": f"session {B_SESSION} turn",
                "tier": "episodic",
                "role": "user",
                "ts": "2026-05-31T04:45:43.000000+00:00",
            },
            {
                "text": "ok",
                "cue": f"session {B_SESSION} turn",
                "tier": "episodic",
                "role": "user",
                "ts": "2026-05-31T04:45:44.000000+00:00",
            },
        ],
    )

    store = _open_store()

    from iai_mcp.capture import drain_deferred_captures
    ended_counts = drain_deferred_captures(store)
    assert ended_counts["events_inserted"] == 0, (
        f"drain_deferred must skip .live.jsonl: {ended_counts}"
    )

    counts = drain_active_live_captures(store, exclude_session_id=A_SESSION)
    assert counts["events_inserted"] == 1, (
        f"Expected 1 inserted (nonce), got: {counts}"
    )
    assert counts["events_skipped"] == 1, (
        f"Expected 1 skipped (too-short 'ok'), got: {counts}"
    )

    turns = store.recent_user_turns(50, session_id=B_SESSION)
    assert len(turns) >= 1, (
        f"recent_user_turns(session_id={B_SESSION!r}) returned {len(turns)} turns; "
        "expected >= 1 after drain"
    )
    texts = [t.literal_surface for t in turns]
    assert any(NONCE in (t or "") for t in texts), (
        f"Nonce not found in recent_user_turns for session {B_SESSION!r}; "
        f"got: {texts!r}"
    )

    for t in turns:
        if NONCE in (t.literal_surface or ""):
            prov = (t.provenance or [{}])[0]
            assert prov.get("session_id") == B_SESSION, (
                f"provenance[0].session_id must be {B_SESSION!r}, "
                f"got {prov.get('session_id')!r}"
            )


def test_drain_active_excludes_own_session(iai_home):
    from iai_mcp.capture import drain_active_live_captures

    OWN_SESSION = "ownown00-0000-0000-0000-000000000000"
    NONCE = "own session excluded from live drain test marker xyz"

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    _write_live_file(
        deferred_dir,
        OWN_SESSION,
        [
            {
                "text": NONCE,
                "cue": "own session turn",
                "tier": "episodic",
                "role": "user",
                "ts": "2026-05-31T04:45:50.000000+00:00",
            }
        ],
    )

    store = _open_store()
    counts = drain_active_live_captures(store, exclude_session_id=OWN_SESSION)
    assert counts["events_inserted"] == 0, (
        f"Own session's live file must not be drained (exclude_session_id): {counts}"
    )
    turns = store.recent_user_turns(50, session_id=OWN_SESSION)
    assert len(turns) == 0, (
        f"No turns for own session expected; got {len(turns)}"
    )


def test_drain_active_idempotent_with_offset(iai_home):
    from iai_mcp.capture import drain_active_live_captures

    B_SESSION = "bbbbbbb0-0000-0000-0000-000000000000"
    NONCE = "idempotent drain test nonce for offset sidecar live file path"
    A_SESSION = "aaaaaaa0-0000-0000-0000-000000000000"

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    _write_live_file(
        deferred_dir,
        B_SESSION,
        [
            {
                "text": NONCE,
                "cue": "test turn",
                "tier": "episodic",
                "role": "user",
                "ts": "2026-05-31T04:46:00.000000+00:00",
            }
        ],
    )

    store = _open_store()
    first = drain_active_live_captures(store, exclude_session_id=A_SESSION)
    assert first["events_inserted"] == 1, f"First drain: {first}"

    second = drain_active_live_captures(store, exclude_session_id=A_SESSION)
    assert second["events_inserted"] == 0, (
        f"Second drain must be idempotent (offset at EOF): {second}"
    )

    turns = store.recent_user_turns(50, session_id=B_SESSION)
    assert len(turns) == 1, f"Expected 1 turn after two drains; got {len(turns)}"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_drain_active_live_promotion_both_drivers(driver, iai_home, monkeypatch):
    """The .live.jsonl -> drain -> Hippo promotion path surfaces content
    identically on both storage drivers.

    Closes the "zero dual-driver coverage" gap for the promotion path:
    verified this session that no existing test in this cluster exercises
    LILLI_STORAGE_DRIVER=lilli.
    """
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")

    from iai_mcp.capture import drain_active_live_captures

    if driver == "lilli":
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)

    session = f"dual-driver-promo-session-{driver}"
    nonce = f"dual driver promotion nonce marker for {driver} driver test content"
    exclude_session = f"exclude-session-{driver}"

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
        counts = drain_active_live_captures(store, exclude_session_id=exclude_session)
        assert counts["events_inserted"] == 1, (
            f"[driver={driver}] expected 1 inserted, got: {counts}"
        )

        turns = store.recent_user_turns(50, session_id=session)
        assert len(turns) >= 1, (
            f"[driver={driver}] recent_user_turns returned {len(turns)} turns; "
            "expected >= 1 after drain"
        )
        texts = [t.literal_surface for t in turns]
        assert any(nonce in (t or "") for t in texts), (
            f"[driver={driver}] nonce not found in recent_user_turns; got: {texts!r}"
        )
    finally:
        store.close()

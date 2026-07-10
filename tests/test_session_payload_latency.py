from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

@pytest.fixture
def iai_home_latency(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "test.sock"))
    yield tmp_path

def _make_large_live_file(deferred_dir: Path, session_id: str, n_events: int = 550) -> Path:
    path = deferred_dir / f"{session_id}.live.jsonl"
    header = {
        "version": 1,
        "deferred_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "cwd": "/tmp/latency-test",
    }
    lines = [json.dumps(header, ensure_ascii=False)]
    base = datetime(2026, 5, 31, 8, 0, 0, tzinfo=timezone.utc)
    for i in range(n_events):
        ev = {
            "text": f"session {session_id} event {i} content for latency test",
            "role": "user",
            "tier": "episodic",
            "ts": (base + timedelta(seconds=i)).isoformat(),
        }
        lines.append(json.dumps(ev))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

@pytest.mark.perf
def test_session_payload_assembly_latency_with_live_files(iai_home_latency, monkeypatch):
    from iai_mcp.community import CommunityAssignment
    from iai_mcp.session import assemble_session_start
    from iai_mcp.store import MemoryStore
    import iai_mcp.capture as _cap_mod

    store = MemoryStore(path=iai_home_latency)

    deferred_dir = iai_home_latency / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    for i in range(20):
        _make_large_live_file(deferred_dir, f"lat-session-{i:02d}", n_events=550)

    call_count = [0]
    original = _cap_mod.read_pending_live_events

    def spy(*args, **kwargs):
        call_count[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_cap_mod, "read_pending_live_events", spy)

    assignment = CommunityAssignment()
    rich_club: list = []
    assemble_session_start(
        store, assignment, rich_club,
        profile_state={"wake_depth": "standard"},
    )
    call_count[0] = 0

    N = 30
    durations: list[float] = []
    for _ in range(N):
        t0 = time.monotonic()
        assemble_session_start(
            store, assignment, rich_club,
            profile_state={"wake_depth": "standard"},
        )
        durations.append(time.monotonic() - t0)

    durations.sort()
    p95_ms = durations[int(N * 0.95)] * 1000
    p95_idx = int(N * 0.95)
    p95_ms = durations[p95_idx] * 1000

    assert call_count[0] >= 1, (
        f"read_pending_live_events must be called during standard assembly; "
        f"got call_count={call_count[0]}.  The live-merge may have been silently "
        f"skipped, producing a false-green latency result."
    )

    assert p95_ms < 100.0, (
        f"p95 assembly latency {p95_ms:.1f} ms >= 100 ms with 20 large live files. "
        f"The bounded deque read must keep this well under the 100 ms budget."
    )

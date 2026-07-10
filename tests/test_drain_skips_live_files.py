from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX paths + glob semantics",
)


@pytest.fixture
def iai_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-skip-live-pass")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp" / "lancedb"))
    import keyring.core
    keyring.core._keyring_backend = None
    yield tmp_path
    keyring.core._keyring_backend = None


def _open_store():
    from iai_mcp.store import MemoryStore
    return MemoryStore()


_DISTINCT_TEXTS = [
    "apples are red and grow on trees in orchards across the world",
    "quantum chromodynamics describes the strong nuclear force precisely",
    "hummingbirds beat their wings about eighty times per second in flight",
    "the great barrier reef stretches over two thousand kilometres of coast",
    "ada lovelace wrote the first algorithm intended for a mechanical engine",
    "tectonic plates drift several centimetres per year across the mantle",
    "the periodic table organises elements by their atomic number and mass",
    "chess endgames with two kings and a rook have a known forced win line",
    "monarch butterflies migrate thousands of miles between mexico and canada",
    "the speed of light in vacuum is the universal upper bound of causality",
]


def _write_jsonl(deferred_dir: Path, fname: str, session_id: str, n_events: int) -> Path:
    deferred_dir.mkdir(parents=True, exist_ok=True)
    path = deferred_dir / fname
    header = {
        "version": 1,
        "deferred_at": "2026-05-12T00:00:00Z",
        "session_id": session_id,
        "cwd": "/tmp",
    }
    lines = [json.dumps(header)]
    for i in range(n_events):
        text = _DISTINCT_TEXTS[i % len(_DISTINCT_TEXTS)]
        lines.append(json.dumps({
            "text": f"[{session_id}-{i}] {text}",
            "cue": f"[{session_id}-{i}] {text}",
            "tier": "episodic",
            "role": "user",
            "ts": "2026-05-12T00:00:00Z",
        }))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_drain_skips_exact_dot_live_files(iai_home):
    from iai_mcp.capture import drain_deferred_captures

    deferred = iai_home / ".iai-mcp" / ".deferred-captures"
    live = _write_jsonl(deferred, "abc-123.live.jsonl", "abc-123", 5)
    finalized = _write_jsonl(deferred, "def-456-1700000000.jsonl", "def-456", 3)

    store = _open_store()
    counts = drain_deferred_captures(store)

    assert counts["events_inserted"] == 3, counts
    assert counts["files_drained"] == 1, counts
    assert live.exists(), "active-writer .live.jsonl must remain on disk"
    assert not finalized.exists(), "finalized file must be unlinked"


def test_drain_processes_renamed_live_dash_epoch(iai_home):
    from iai_mcp.capture import drain_deferred_captures

    deferred = iai_home / ".iai-mcp" / ".deferred-captures"
    renamed = _write_jsonl(deferred, "ghi-789.live-1700000001.jsonl", "ghi-789", 4)

    store = _open_store()
    counts = drain_deferred_captures(store)

    assert counts["events_inserted"] == 4, counts
    assert counts["files_drained"] == 1, counts
    assert not renamed.exists()


def test_drain_processes_finalized_after_rename(iai_home):
    from iai_mcp.capture import drain_deferred_captures

    deferred = iai_home / ".iai-mcp" / ".deferred-captures"
    live = _write_jsonl(deferred, "jkl-001.live.jsonl", "jkl-001", 2)
    epoch = int(time.time())
    renamed = live.with_name(f"jkl-001.live-{epoch}.jsonl")
    live.rename(renamed)
    assert not live.exists()

    store = _open_store()
    counts = drain_deferred_captures(store)

    assert counts["events_inserted"] == 2, counts
    assert not renamed.exists()
    assert not live.exists()


def test_skip_predicate_exactness(iai_home):
    from iai_mcp.capture import drain_deferred_captures

    deferred = iai_home / ".iai-mcp" / ".deferred-captures"
    weird_live = _write_jsonl(deferred, "weird.live.jsonl", "weird", 3)
    weird_dash = _write_jsonl(deferred, "weird.live-9.jsonl", "weird", 2)

    store = _open_store()
    counts = drain_deferred_captures(store)

    assert counts["events_inserted"] == 2, counts
    assert counts["files_drained"] == 1, counts
    assert weird_live.exists(), "weird.live.jsonl must be skipped (exact suffix)"
    assert not weird_dash.exists(), "weird.live-9.jsonl must be processed"

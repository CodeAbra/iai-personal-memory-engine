"""Working-tier snapshot cache + the daemon-independent per-turn hook.

The per-turn reader (a UserPromptSubmit shell hook) must surface the active
task WITHOUT a daemon socket and WITHOUT importing iai_mcp — it reads only
the snapshot cache this module emits on every feed. Recall/capture never
depend on the file; its absence merely yields no injected context.
"""
from __future__ import annotations

import os
import stat
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp import working_tier
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord

_HOOK = (
    Path(__file__).resolve().parents[1]
    / "src/iai_mcp/_deploy/hooks/iai-mcp-per-turn-recall.sh"
)


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401, PLC0415
        except ImportError:
            pytest.skip("iai_mcp_native not built")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _make(text: str) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"session_id": "s-cache"}],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    )


@pytest.fixture(autouse=True)
def _fresh_singleton():
    working_tier._reset()
    yield
    working_tier._reset()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_capture_emits_snapshot_cache(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    cache = tmp_path / "wt-cache.md"
    monkeypatch.setenv(working_tier.WORKING_TIER_CACHE_ENV, str(cache))

    store = MemoryStore(path=tmp_path)
    store.insert(_make("study the hebbian weave design"))

    assert cache.is_file(), f"[{driver}] snapshot cache not emitted on capture"
    text = cache.read_text(encoding="utf-8")
    assert "study the hebbian weave design" in text, "goal must be verbatim"
    mode = stat.S_IMODE(cache.stat().st_mode)
    assert mode == 0o600, f"snapshot must be private, got {oct(mode)}"


def test_close_task_removes_snapshot(tmp_path, monkeypatch):
    cache = tmp_path / "wt-cache.md"
    monkeypatch.setenv(working_tier.WORKING_TIER_CACHE_ENV, str(cache))

    working_tier.update_from_record(_make("short-lived task"))
    assert cache.is_file()
    working_tier.close_task(store=None)
    assert not cache.exists(), "a closed task must not stay readable as active"


def _run_hook(cache_path: Path, extra_env: dict | None = None) -> str:
    env = dict(os.environ)
    env["IAI_MCP_WORKING_TIER_CACHE"] = str(cache_path)
    env.pop("IAI_MCP_PER_TURN_SOCKET_ACCEL", None)
    env.update(extra_env or {})
    proc = subprocess.run(
        [str(_HOOK)],
        input='{"prompt": "what am I working on"}',
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    assert proc.returncode == 0, f"hook must always exit 0: {proc.stderr}"
    return proc.stdout


def test_hook_surfaces_fresh_snapshot_without_daemon(tmp_path):
    cache = tmp_path / "wt.md"
    cache.write_text("goal: finish the teach flow\n", encoding="utf-8")
    out = _run_hook(cache)
    assert "<iai-mcp-working-tier>" in out
    assert "finish the teach flow" in out


def test_hook_silent_when_cache_absent(tmp_path):
    out = _run_hook(tmp_path / "missing.md")
    assert out.strip() == ""


def test_hook_ignores_stale_snapshot(tmp_path):
    cache = tmp_path / "wt.md"
    cache.write_text("goal: yesterday's task\n", encoding="utf-8")
    old = time.time() - 3 * 3600
    os.utime(cache, (old, old))
    out = _run_hook(cache)
    assert "yesterday" not in out, "stale snapshot must not be injected"


def test_hook_is_fast_and_import_free(tmp_path):
    cache = tmp_path / "wt.md"
    cache.write_text("goal: latency check\n", encoding="utf-8")
    t0 = time.monotonic()
    _run_hook(cache)
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0, f"cache-only hook must be fast, took {elapsed:.2f}s"
    text = _HOOK.read_text(encoding="utf-8")
    assert "import iai_mcp" not in text, "hook must never import iai_mcp"
    assert "working_tier import" not in text

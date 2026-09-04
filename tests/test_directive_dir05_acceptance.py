"""Literal cross-session acceptance: a directive set in one session replays
verbatim in a fresh session with zero overlap, across both lanes."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from iai_mcp.capture import capture_turn
from iai_mcp.community import CommunityAssignment
from iai_mcp.session import assemble_session_start, format_payload_as_markdown
from iai_mcp.store import MemoryStore

_HOOK = (
    Path(__file__).resolve().parents[1]
    / "src/iai_mcp/_deploy/hooks/iai-mcp-per-turn-recall.sh"
)

DIRECTIVE_TEXT = "from now on reply in English"
SESSION_B_MESSAGE = "stop for now"


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _run_per_turn_hook(store_root: Path, prompt: str) -> str:
    env = dict(os.environ)
    env["IAI_MCP_STORE"] = str(store_root)
    env.pop("IAI_MCP_WORKING_TIER_CACHE", None)
    env.pop("IAI_MCP_FORESIGHT_PACK", None)
    env.pop("IAI_MCP_PER_TURN_SOCKET_ACCEL", None)
    proc = subprocess.run(
        [str(_HOOK)],
        input=f'{{"prompt": "{prompt}"}}',
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    assert proc.returncode == 0, f"hook must always exit 0: {proc.stderr}"
    return proc.stdout


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_cross_session_directive_replay_has_zero_overlap(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)

    store = MemoryStore(path=tmp_path / "lancedb")

    hook_root = tmp_path / "hook-root"
    hook_root.mkdir()
    cache_path = hook_root / ".directives.cached.md"

    import iai_mcp.directive_cache as directive_cache_mod
    _real_write = directive_cache_mod.write_directives_cache

    def _write_to_isolated_cache(passed_store, **_kwargs):
        _real_write(passed_store, cache_path=cache_path)

    monkeypatch.setattr(directive_cache_mod, "write_directives_cache", _write_to_isolated_cache)

    result = capture_turn(
        store=store, cue="reply preference", text=DIRECTIVE_TEXT,
        directive=True, session_id="session-a", role="user",
    )
    assert result["status"] == "inserted", result

    # No time.sleep anywhere in this test: these assertions run immediately
    # after capture, proving the synchronous cache path, not a sweep.
    assert cache_path.exists()
    assert DIRECTIVE_TEXT in cache_path.read_text(encoding="utf-8")

    payload = assemble_session_start(
        store, CommunityAssignment(), [],
        session_id="session-b", profile_state={"wake_depth": "standard"},
    )
    assert DIRECTIVE_TEXT in payload.directives

    rendered = format_payload_as_markdown(payload)
    assert "## Standing orders (always active)" in rendered
    assert DIRECTIVE_TEXT in rendered

    hook_out = _run_per_turn_hook(hook_root, SESSION_B_MESSAGE)
    assert "<iai-mcp-directives>" in hook_out
    assert "</iai-mcp-directives>" in hook_out
    assert DIRECTIVE_TEXT in hook_out

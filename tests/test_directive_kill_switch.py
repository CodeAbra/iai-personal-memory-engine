"""IAI_MCP_DIRECTIVES_OFF kill switch disables the standing-orders tier's two
injection lanes -- session-start render and the per-turn hook -- while
leaving the default (unset) path byte-identical to current behavior.

Capture-side contract (post-deletion): there is no classifier write path left
for the switch to gate -- the auto-classify branch is deleted outright, not
env-gated. A directive-shaped text with no explicit caller value and no
marker opt-in never auto-sets directive, regardless of this switch.

Mirrors the IAI_MCP_COFIRE_OFF idiom (src/iai_mcp/cli/_capture.py): default
unset/absent = feature ON, value "1" = OFF.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from iai_mcp.capture import capture_turn
from iai_mcp.directive_cache import write_directives_cache
from iai_mcp.session import render_directive_segment
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord

_DIRECTIVE_TEXT = "from now on reply in English"

_HOOK = (
    Path(__file__).resolve().parents[1]
    / "src/iai_mcp/_deploy/hooks/iai-mcp-per-turn-recall.sh"
)


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401, PLC0415
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "lancedb")


def _directive_record(text: str) -> MemoryRecord:
    rec = MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    )
    rec.directive = True
    return rec


def _run_hook(store_root: Path, *, directives_off: bool) -> str:
    env = dict(os.environ)
    env["IAI_MCP_STORE"] = str(store_root)
    env.pop("IAI_MCP_WORKING_TIER_CACHE", None)
    env.pop("IAI_MCP_FORESIGHT_PACK", None)
    env.pop("IAI_MCP_PER_TURN_SOCKET_ACCEL", None)
    if directives_off:
        env["IAI_MCP_DIRECTIVES_OFF"] = "1"
    else:
        env.pop("IAI_MCP_DIRECTIVES_OFF", None)
    proc = subprocess.run(
        [str(_HOOK)],
        input='{"prompt": "unrelated turn text"}',
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    assert proc.returncode == 0, f"hook must always exit 0: {proc.stderr}"
    return proc.stdout


# --- CAPTURE: no classifier write path remains, independent of the switch -


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_kill_switch_on_capture_never_auto_classifies_directive(driver, store, monkeypatch):
    _select_driver(driver, monkeypatch)
    monkeypatch.setenv("IAI_MCP_DIRECTIVES_OFF", "1")

    result = capture_turn(
        store=store, cue="c", text=_DIRECTIVE_TEXT, session_id="s1", role="user",
    )

    assert result["status"] == "inserted", result
    rec = store.get(UUID(result["record_id"]))
    assert rec is not None
    assert rec.directive is False


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_kill_switch_off_default_capture_never_auto_classifies_directive(
    driver, store, monkeypatch
):
    """Post-deletion contract: the retired classifier's write path is gone,
    so directive-shaped text is NOT auto-flagged even with the switch unset
    -- there is no auto-classify branch left for the switch to gate."""
    _select_driver(driver, monkeypatch)
    monkeypatch.delenv("IAI_MCP_DIRECTIVES_OFF", raising=False)

    result = capture_turn(
        store=store, cue="c", text=_DIRECTIVE_TEXT, session_id="s1", role="user",
    )

    assert result["status"] == "inserted", result
    rec = store.get(UUID(result["record_id"]))
    assert rec is not None
    assert rec.directive is False


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_kill_switch_on_does_not_override_explicit_directive_flag(driver, store, monkeypatch):
    """The switch has nothing capture-side left to gate; an explicit
    caller-supplied directive=True/False must pass through untouched."""
    _select_driver(driver, monkeypatch)
    monkeypatch.setenv("IAI_MCP_DIRECTIVES_OFF", "1")

    result = capture_turn(
        store=store, cue="c", text="plain non-directive alice update",
        directive=True, session_id="s1", role="user",
    )

    assert result["status"] == "inserted", result
    rec = store.get(UUID(result["record_id"]))
    assert rec is not None
    assert rec.directive is True


# --- SESSION-START INJECTION: render_directive_segment returns empty ------


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_kill_switch_on_render_directive_segment_returns_empty(driver, store, monkeypatch):
    _select_driver(driver, monkeypatch)
    store.insert(_directive_record(_DIRECTIVE_TEXT))

    monkeypatch.setenv("IAI_MCP_DIRECTIVES_OFF", "1")
    assert render_directive_segment(store) == ""


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_kill_switch_off_default_render_directive_segment_unchanged(driver, store, monkeypatch):
    _select_driver(driver, monkeypatch)
    store.insert(_directive_record(_DIRECTIVE_TEXT))

    monkeypatch.delenv("IAI_MCP_DIRECTIVES_OFF", raising=False)
    rendered = render_directive_segment(store)
    assert _DIRECTIVE_TEXT in rendered


# --- write_directives_cache writes empty content when switch is on --------


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_kill_switch_on_write_directives_cache_yields_empty_content(
    driver, store, tmp_path, monkeypatch
):
    _select_driver(driver, monkeypatch)
    store.insert(_directive_record(_DIRECTIVE_TEXT))
    flush_record_buffer(store)

    monkeypatch.setenv("IAI_MCP_DIRECTIVES_OFF", "1")
    cache_path = tmp_path / ".directives.cached.md"
    write_directives_cache(store, cache_path=cache_path)

    assert cache_path.is_file()
    assert cache_path.read_text(encoding="utf-8") == ""


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_kill_switch_off_default_write_directives_cache_unchanged(
    driver, store, tmp_path, monkeypatch
):
    _select_driver(driver, monkeypatch)
    store.insert(_directive_record(_DIRECTIVE_TEXT))
    flush_record_buffer(store)

    monkeypatch.delenv("IAI_MCP_DIRECTIVES_OFF", raising=False)
    cache_path = tmp_path / ".directives.cached.md"
    write_directives_cache(store, cache_path=cache_path)

    assert cache_path.is_file()
    assert _DIRECTIVE_TEXT in cache_path.read_text(encoding="utf-8")


# --- PER-TURN HOOK: emit_directives returns early when switch is on -------


def test_kill_switch_on_hook_emits_nothing(tmp_path):
    store_root = tmp_path / "store"
    store_root.mkdir()
    cache = store_root / ".directives.cached.md"
    cache.write_text("- always reply in English\n", encoding="utf-8")

    out = _run_hook(store_root, directives_off=True)

    assert "<iai-mcp-directives>" not in out
    assert "always reply in English" not in out


def test_kill_switch_off_default_hook_still_emits(tmp_path):
    store_root = tmp_path / "store"
    store_root.mkdir()
    cache = store_root / ".directives.cached.md"
    cache.write_text("- always reply in English\n", encoding="utf-8")

    out = _run_hook(store_root, directives_off=False)

    assert "<iai-mcp-directives>" in out
    assert "always reply in English" in out

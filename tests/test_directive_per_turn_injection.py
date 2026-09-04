"""Per-turn directive block emission, no staleness gate, safe when the
cache file is absent."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

_HOOK = (
    Path(__file__).resolve().parents[1]
    / "src/iai_mcp/_deploy/hooks/iai-mcp-per-turn-recall.sh"
)


def _run_hook(store_root: Path) -> str:
    env = dict(os.environ)
    env["IAI_MCP_STORE"] = str(store_root)
    env.pop("IAI_MCP_WORKING_TIER_CACHE", None)
    env.pop("IAI_MCP_FORESIGHT_PACK", None)
    env.pop("IAI_MCP_PER_TURN_SOCKET_ACCEL", None)
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


def test_per_turn_hook_emits_directive_block(tmp_path):
    store_root = tmp_path / "store"
    store_root.mkdir()
    cache = store_root / ".directives.cached.md"
    cache.write_text("- always reply in English\n", encoding="utf-8")

    out = _run_hook(store_root)

    assert "<iai-mcp-directives>" in out
    assert "</iai-mcp-directives>" in out
    assert "always reply in English" in out


def test_per_turn_hook_silent_when_directive_cache_absent(tmp_path):
    store_root = tmp_path / "store"
    store_root.mkdir()

    out = _run_hook(store_root)

    assert "<iai-mcp-directives>" not in out


def test_per_turn_hook_directive_block_has_no_staleness_gate(tmp_path):
    store_root = tmp_path / "store"
    store_root.mkdir()
    cache = store_root / ".directives.cached.md"
    cache.write_text("- never touch production credentials\n", encoding="utf-8")
    old = time.time() - 30 * 24 * 3600
    os.utime(cache, (old, old))

    out = _run_hook(store_root)

    assert "<iai-mcp-directives>" in out
    assert "never touch production credentials" in out


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_directive_cache_writer_produces_content_the_hook_can_render(
    driver, tmp_path, monkeypatch
):
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)

    from datetime import datetime, timezone
    from uuid import uuid4

    from iai_mcp.directive_cache import write_directives_cache
    from iai_mcp.store import MemoryStore
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    store = MemoryStore(path=tmp_path)
    store.insert(
        MemoryRecord(
            id=uuid4(),
            tier="episodic",
            literal_surface="always answer in plain terse language",
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
            provenance=[{"session_id": "s-directive-cache"}],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tags=[],
            language="en",
            directive=True,
        )
    )

    cache_path = tmp_path / ".directives.cached.md"
    write_directives_cache(store, cache_path=cache_path)

    assert cache_path.is_file()
    content = cache_path.read_text(encoding="utf-8")
    assert "always answer in plain terse language" in content

    store_root = tmp_path / "hook-root"
    store_root.mkdir()
    (store_root / ".directives.cached.md").write_text(content, encoding="utf-8")
    out = _run_hook(store_root)
    assert "<iai-mcp-directives>" in out
    assert "always answer in plain terse language" in out

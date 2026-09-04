"""Before/after content differential proving boot warm-up resides the
priming cache — never a vacuous hasattr check, since ``load()`` setattrs
the memo unconditionally even on a miss.
"""
from __future__ import annotations

from pathlib import Path

from iai_mcp import prime_cache
from iai_mcp.daemon._boot_warmup import warm_dispatch_surface
from iai_mcp.store import MemoryStore


def _fresh_store(tmp_path: Path) -> "tuple[MemoryStore, Path]":
    home = tmp_path / "operator-home"
    store_root = home / ".iai-mcp"
    store = MemoryStore(path=store_root)
    return store, home


def test_boot_warmup_loads_prime_cache(tmp_path: Path) -> None:
    store, home = _fresh_store(tmp_path)
    blob = {
        "seed_to_chunks": {"A": ["c1", "c2"]},
        "chunk_members": {"c1": ["A", "B"], "c2": ["A", "C"]},
    }
    assert prime_cache.save(store, blob) is True

    # Fresh store instance -- no inherited process memo.
    warm_store = MemoryStore(path=home / ".iai-mcp")
    assert getattr(warm_store, "_prime_cache", None) is None

    summary = warm_dispatch_surface(warm_store)

    assert isinstance(summary, dict)
    assert "error" not in summary

    resident = getattr(warm_store, "_prime_cache", None)
    assert resident is not None
    assert resident["seed_to_chunks"] == {"A": ["c1", "c2"]}
    assert resident["chunk_members"] == {"c1": ["A", "B"], "c2": ["A", "C"]}

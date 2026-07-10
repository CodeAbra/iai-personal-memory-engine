from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make

@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch, tmp_path: Path):
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))
    yield

def _norm_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    from iai_mcp.types import EMBED_DIM
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()

def _build_thick_store(tmp_path: Path):
    from iai_mcp.store import MemoryStore, flush_record_buffer

    store_root = tmp_path / "store"
    store = MemoryStore(str(store_root))

    for i in range(15):
        store.insert(_make(text=f"User record {i}", vec=_norm_vec(i + 100)))
    flush_record_buffer(store)
    return store, store_root

def _warm_store_via_rebuild(store) -> None:
    from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline
    pipeline = SleepPipeline(store)
    done, payload = pipeline._step_recall_index_rebuild(None)
    assert done is True, f"Rebuild step must return done=True; got done={done}, payload={payload}"

def test_drowsy_rewake_cold_then_rebuild_ready(monkeypatch, tmp_path):
    from iai_mcp import runtime_graph_cache

    store, _ = _build_thick_store(tmp_path)

    _warm_store_via_rebuild(store)

    _, _, _, src, _ = runtime_graph_cache.load_recall_structural(store)
    assert src in ("overlay", "normal"), (
        f"Warm baseline must be overlay/normal before invalidation; got {src!r}. "
        "Fixture may be too thin — add more records/edges."
    )

    runtime_graph_cache.invalidate(store)

    _, _, _, src_cold, _ = runtime_graph_cache.load_recall_structural(store)
    assert src_cold in ("cold_degrade", "last_good"), (
        f"After invalidate the cache must be cold; got {src_cold!r}."
    )

    rebuild_ready_event = getattr(runtime_graph_cache, "rebuild_ready", None)

    gate = threading.Event()
    real_rebuild = runtime_graph_cache._rebuild_and_save_rgc  # type: ignore[attr-defined]

    def _gated_rebuild(s):
        gate.wait(timeout=10)
        return real_rebuild(s)

    monkeypatch.setattr(runtime_graph_cache, "_rebuild_and_save_rgc", _gated_rebuild)

    rebuild_ready_event.clear()  # type: ignore[union-attr]

    import iai_mcp.daemon as _daemon_mod
    _daemon_mod._kick_drowsy_rgc_rebuild(store)  # type: ignore[attr-defined]

    assert not rebuild_ready_event.is_set(), (  # type: ignore[union-attr]
        "rebuild_ready must NOT be set while the worker is blocked on the gate "
        "(the kick must be non-blocking — flag-not-gate design)."
    )

    gate.set()
    assert rebuild_ready_event.wait(timeout=10), (  # type: ignore[union-attr]
        "rebuild_ready must be set after the background rebuild completes."
    )

    _, _, _, src_after, _ = runtime_graph_cache.load_recall_structural(store)
    assert src_after in ("overlay", "normal"), (
        f"After DROWSY-edge rebuild, structural_source must be overlay/normal; "
        f"got {src_after!r}."
    )

def test_wake_hook_rebuilds_cold_cache(monkeypatch, tmp_path):
    from iai_mcp import runtime_graph_cache

    store, _ = _build_thick_store(tmp_path)

    _warm_store_via_rebuild(store)

    runtime_graph_cache.invalidate(store)

    _, _, _, src_cold, _ = runtime_graph_cache.load_recall_structural(store)
    assert src_cold in ("cold_degrade", "last_good"), (
        f"After invalidate, cache must be cold; got {src_cold!r}."
    )

    import iai_mcp.daemon as _daemon_mod

    _daemon_mod._wake_hook_rebuild_if_cold(store)  # type: ignore[attr-defined]

    _, _, _, src_after, _ = runtime_graph_cache.load_recall_structural(store)
    assert src_after in ("overlay", "normal"), (
        f"After wake-hook rebuild-if-cold, structural_source must be overlay/normal; "
        f"got {src_after!r}."
    )

def test_wake_hook_skips_when_warm(monkeypatch, tmp_path):
    from iai_mcp import runtime_graph_cache

    store, _ = _build_thick_store(tmp_path)

    _warm_store_via_rebuild(store)

    _, _, _, src_warm, _ = runtime_graph_cache.load_recall_structural(store)
    assert src_warm in ("overlay", "normal"), (
        f"Warm baseline must be overlay/normal; got {src_warm!r}."
    )

    gen_before = runtime_graph_cache.get_current_generation()

    import iai_mcp.daemon as _daemon_mod

    _daemon_mod._wake_hook_rebuild_if_cold(store)  # type: ignore[attr-defined]

    _, _, _, src_after, _ = runtime_graph_cache.load_recall_structural(store)
    assert src_after in ("overlay", "normal"), (
        f"Cache must remain warm after helper called on warm cache; got {src_after!r}."
    )
    gen_after = runtime_graph_cache.get_current_generation()
    assert gen_after == gen_before, (
        f"Helper must NOT advance the generation when cache is already warm; "
        f"before={gen_before}, after={gen_after}."
    )

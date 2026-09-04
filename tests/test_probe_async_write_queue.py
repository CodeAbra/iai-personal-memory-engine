from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make

from iai_mcp.embed import Embedder
from iai_mcp.pipeline import recall_for_response
from iai_mcp.retrieve import build_runtime_graph
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM

N_RECORDS = 20
CLOSE_BOUND_SECONDS = 15.0


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _random_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _build_store(tmp_path: Path, n: int) -> MemoryStore:
    store = MemoryStore(str(tmp_path / "store"))
    for i in range(n):
        rec = _make(text=f"Async write queue fixture record {i}", vec=_random_vec(8_000 + i))
        store.insert(rec)
    return store


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_enable_async_writes_makes_both_deferred_write_queues_live(tmp_path, monkeypatch, driver):
    _select_driver(driver, monkeypatch)
    store = _build_store(tmp_path, N_RECORDS)

    assert store._reinforce_queue is None
    assert store._provenance_queue is None

    asyncio.run(store.enable_async_writes())

    assert store._reinforce_queue is not None
    assert store._provenance_queue is not None

    store.close()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_recall_completes_and_close_drains_within_bounded_time(tmp_path, monkeypatch, driver):
    _select_driver(driver, monkeypatch)
    store = _build_store(tmp_path, N_RECORDS)
    asyncio.run(store.enable_async_writes())
    assert store._reinforce_queue is not None
    assert store._provenance_queue is not None

    graph, assignment, rich_club = build_runtime_graph(store)
    embedder = Embedder()

    response = recall_for_response(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue="what did we discuss about the async write queue",
        session_id="async-write-queue-test", budget_tokens=1500, mode="concept",
    )
    assert response is not None

    t0 = time.perf_counter()
    store.close()
    close_ms = (time.perf_counter() - t0) * 1000.0

    assert close_ms < CLOSE_BOUND_SECONDS * 1000.0, close_ms

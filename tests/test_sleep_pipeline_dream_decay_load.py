from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import iai_mcp.sleep as sleep_module
import iai_mcp.user_model as user_model_module
from iai_mcp.lifecycle_event_log import LifecycleEventLog
from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline
from iai_mcp.user_model import UserModel

@pytest.fixture
def state_path(tmp_path: Path) -> Path:
    return tmp_path / "lifecycle_state.json"

@pytest.fixture
def event_log(tmp_path: Path) -> LifecycleEventLog:
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return LifecycleEventLog(log_dir=log_dir)

@pytest.fixture
def pipeline(state_path: Path, event_log: LifecycleEventLog) -> SleepPipeline:
    return SleepPipeline(
        store=None,
        lifecycle_state_path=state_path,
        event_log=event_log,
        quarantine_ttl_hours=24.0,
    )

def test_dream_decay_step_does_not_raise_attribute_error_on_user_model_load(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_decay(store: Any, plasticity_gain: float = 1.0, **kwargs: Any) -> dict[str, int]:
        return {"decayed": 0, "pruned": 0}

    monkeypatch.setattr(sleep_module, "_decay_edges", _fake_decay)

    completed, payload = pipeline._step_dream_decay(interrupt_check=None)

    assert completed is True
    assert isinstance(payload, dict)
    assert payload.get("decayed") == 0
    assert payload.get("pruned") == 0

def test_dream_decay_step_reads_plasticity_gain_from_user_model(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, float] = {}

    def _fake_decay(store: Any, plasticity_gain: float = 1.0, **kwargs: Any) -> dict[str, int]:
        captured["plasticity_gain"] = float(plasticity_gain)
        return {"decayed": 0, "pruned": 0}

    def _fake_load() -> UserModel:
        return UserModel(plasticity_gain=0.5)

    monkeypatch.setattr(sleep_module, "_decay_edges", _fake_decay)
    monkeypatch.setattr(user_model_module, "load", _fake_load)

    completed, _payload = pipeline._step_dream_decay(interrupt_check=None)

    assert completed is True
    assert captured.get("plasticity_gain") == pytest.approx(0.5)


class TestDecayChunkedTransactions:
    @pytest.fixture(autouse=True)
    def _isolate_store(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai-mcp-store"))

    def _store_with_stale_edges(self, tmp_path: Path, n: int):
        import uuid
        from datetime import datetime, timedelta, timezone

        from iai_mcp.store import EDGES_TABLE, MemoryStore

        store = MemoryStore(path=str(tmp_path / "iai-mcp-store"), user_id="alice")
        stale = (
            datetime.now(timezone.utc) - timedelta(days=200)
        ).isoformat()
        rows = [
            {
                "src": str(uuid.uuid4()),
                "dst": str(uuid.uuid4()),
                "edge_type": "hebbian",
                "weight": 0.9,
                "updated_at": stale,
            }
            for _ in range(n)
        ]
        tbl = store.db.open_table(EDGES_TABLE)
        tbl.merge_insert(["src", "dst", "edge_type"]).when_matched_update_all().execute(rows)
        return store

    def test_full_run_processes_every_stale_edge(self, tmp_path: Path) -> None:
        store = self._store_with_stale_edges(tmp_path, 50)
        result = sleep_module._decay_edges(store)
        assert result["interrupted"] is False
        assert result["decayed"] + result["pruned"] == 50

    def test_defer_between_chunks_resumes_idempotently(self, tmp_path: Path) -> None:
        from iai_mcp.store import EDGES_TABLE

        n = sleep_module.DECAY_CHUNK_EDGES + 100
        store = self._store_with_stale_edges(tmp_path, n)

        # First run yields after chunk 0: its edges are committed and carry a
        # fresh updated_at; nothing else is touched.
        first = sleep_module._decay_edges(
            store, should_defer=lambda chunk_no: chunk_no >= 1,
        )
        assert first["interrupted"] is True
        applied_first = first["decayed"] + first["pruned"]
        assert applied_first == sleep_module.DECAY_CHUNK_EDGES

        # Second run resumes for free — the grace window skips the already-
        # applied edges — and completes exactly the remainder.
        second = sleep_module._decay_edges(store)
        assert second["interrupted"] is False
        applied_second = second["decayed"] + second["pruned"]
        assert applied_first + applied_second == n

        remaining = len(store.db.open_table(EDGES_TABLE).to_pandas())
        assert remaining == n - first["pruned"] - second["pruned"]

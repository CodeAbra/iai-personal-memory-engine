from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from iai_mcp import daemon as daemon_mod
from iai_mcp import daemon_state as state_mod


def test_persist_serialises_a_snapshot_not_the_live_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Regression: save_state used to run in a worker thread over the *live* `state`
    # dict while the event loop kept mutating it -> "dictionary changed size during
    # iteration", which crashed the scheduler tick and froze consolidation. _persist
    # must hand the writer an isolated deepcopy taken synchronously in the loop.
    state_path = tmp_path / ".daemon-state.json"
    monkeypatch.setattr(state_mod, "STATE_PATH", state_path, raising=True)

    captured: dict = {}

    def spy_save(passed: dict) -> None:
        captured["obj"] = passed
        state_mod.save_state(passed)

    monkeypatch.setattr(daemon_mod, "save_state", spy_save, raising=True)

    state: dict = {"fsm_state": "WAKE", "nested": {"a": 1}}
    asyncio.run(daemon_mod._persist(state))

    # The writer received a copy, not the live dict (deep, not shallow)...
    assert captured["obj"] is not state
    assert captured["obj"]["nested"] is not state["nested"]

    # ...so mutating the live state after the snapshot cannot corrupt an in-flight
    # serialise, and what landed on disk is the pre-mutation snapshot.
    state["nested"]["b"] = 2
    assert "b" not in captured["obj"]["nested"]
    assert json.loads(state_path.read_text()) == {"fsm_state": "WAKE", "nested": {"a": 1}}

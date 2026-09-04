"""Contracts for the durable running-agent registry in daemon_state.json.

Registry writes must go through ``update_state`` — the same flock-protected
load-mutate-save path every other daemon-state writer uses — never a
whole-dict ``save_state`` on a long-held dict, which is the split-brain class
this module was already hardened against.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from iai_mcp import model_attribution
from iai_mcp.daemon_state import (
    MAX_RUNNING_AGENTS,
    RUNNING_AGENT_ID_MAX_CHARS,
    RUNNING_AGENT_TTL_HOURS,
    complete_running_agent,
    load_state,
    normalize_model,
    prune_stale_agents,
    register_running_agent,
    save_state,
)


@pytest.fixture()
def state_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    return tmp_path


def test_register_persists_pending_entry_readable_after_reload(state_root: Path) -> None:
    register_running_agent(
        agent_id="a1",
        role="research",
        expected_artifact="RESEARCH.md",
        agent_model="claude-sonnet-5",
    )

    reloaded = load_state()
    entry = reloaded["running_agents"]["a1"]
    assert entry["status"] == "pending"
    assert entry["role"] == "research"
    assert entry["expected_artifact"] == "RESEARCH.md"
    assert entry["model"] == normalize_model("claude-sonnet-5")
    assert entry["completed_at"] is None
    spawned_at = datetime.fromisoformat(entry["spawned_at"])
    assert spawned_at.tzinfo is not None


def test_register_with_no_model_stores_none_without_crash(state_root: Path) -> None:
    register_running_agent(
        agent_id="a1", role="research", expected_artifact="RESEARCH.md",
    )

    entry = load_state()["running_agents"]["a1"]
    assert entry["model"] is None


def test_complete_stamps_status_and_completed_at_server_side(state_root: Path) -> None:
    register_running_agent(
        agent_id="a1", role="research", expected_artifact="RESEARCH.md",
    )

    complete_running_agent("a1")

    entry = load_state()["running_agents"]["a1"]
    assert entry["status"] == "complete"
    completed_at = datetime.fromisoformat(entry["completed_at"])
    assert completed_at.tzinfo is not None


def test_complete_missing_id_is_a_safe_no_op(state_root: Path) -> None:
    save_state({})

    complete_running_agent("missing")

    assert load_state() == {}


def test_role_and_artifact_newlines_and_control_chars_are_stripped(state_root: Path) -> None:
    register_running_agent(
        agent_id="a1",
        role="research\nplan\x07",
        expected_artifact="RESEARCH.md\r\ninjected line",
    )

    entry = load_state()["running_agents"]["a1"]
    assert "\n" not in entry["role"]
    assert "\r" not in entry["role"]
    assert "\x07" not in entry["role"]
    assert "\n" not in entry["expected_artifact"]
    assert "\r" not in entry["expected_artifact"]
    assert entry["role"] == "researchplan"
    assert entry["expected_artifact"] == "RESEARCH.mdinjected line"


def test_role_and_artifact_are_length_capped(state_root: Path) -> None:
    register_running_agent(
        agent_id="a1",
        role="r" * 500,
        expected_artifact="e" * 500,
    )

    entry = load_state()["running_agents"]["a1"]
    assert len(entry["role"]) <= 80
    assert len(entry["expected_artifact"]) <= 200


def test_agent_id_charset_is_sanitized_before_becoming_a_key(state_root: Path) -> None:
    register_running_agent(
        agent_id="a1/../etc\npasswd!!",
        role="research",
        expected_artifact="RESEARCH.md",
    )

    agents = load_state()["running_agents"]
    assert len(agents) == 1
    key = next(iter(agents))
    assert key == "a1etcpasswd"
    assert "/" not in key
    assert "\n" not in key
    assert "!" not in key
    assert "." not in key


def test_agent_id_is_length_capped_before_becoming_a_key(state_root: Path) -> None:
    register_running_agent(
        agent_id="a" * 500, role="research", expected_artifact="RESEARCH.md",
    )

    agents = load_state()["running_agents"]
    key = next(iter(agents))
    assert key == "a" * RUNNING_AGENT_ID_MAX_CHARS
    assert len(key) == RUNNING_AGENT_ID_MAX_CHARS


def test_agent_id_sanitizing_to_empty_is_a_safe_no_op(state_root: Path) -> None:
    register_running_agent(agent_id="/../\n", role="research", expected_artifact="R.md")

    assert load_state() == {}


def test_complete_uses_the_same_bound_key_as_register(state_root: Path) -> None:
    raw_id = "a1/../ x"
    register_running_agent(agent_id=raw_id, role="research", expected_artifact="RESEARCH.md")

    complete_running_agent(raw_id)

    agents = load_state()["running_agents"]
    assert len(agents) == 1
    entry = next(iter(agents.values()))
    assert entry["status"] == "complete", (
        "completing with the SAME raw id the caller registered with must "
        "resolve to the same sanitized key, not miss"
    )


def test_two_sequential_registers_both_persist(state_root: Path) -> None:
    register_running_agent(
        agent_id="a1", role="research", expected_artifact="RESEARCH.md",
    )
    register_running_agent(
        agent_id="a2", role="implement", expected_artifact="PLAN.md",
    )

    final = load_state()["running_agents"]
    assert "a1" in final
    assert "a2" in final


def test_prune_drops_pending_and_complete_entries_past_ttl(state_root: Path) -> None:
    now = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
    stale = (now - timedelta(hours=RUNNING_AGENT_TTL_HOURS + 1)).isoformat()
    fresh = (now - timedelta(hours=1)).isoformat()
    state = {
        "running_agents": {
            "stale-pending": {
                "role": "r", "spawned_at": stale, "expected_artifact": "e",
                "status": "pending", "model": None, "completed_at": None,
            },
            "stale-complete": {
                "role": "r", "spawned_at": stale, "expected_artifact": "e",
                "status": "complete", "model": None, "completed_at": stale,
            },
            "fresh-pending": {
                "role": "r", "spawned_at": fresh, "expected_artifact": "e",
                "status": "pending", "model": None, "completed_at": None,
            },
        }
    }

    removed = prune_stale_agents(state, now=now)

    assert removed == 2
    assert set(state["running_agents"].keys()) == {"fresh-pending"}


def test_prune_caps_max_entries_keeps_newest(state_root: Path) -> None:
    now = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
    agents = {
        f"agent-{i}": {
            "role": "r",
            "spawned_at": (now - timedelta(minutes=i)).isoformat(),
            "expected_artifact": "e",
            "status": "pending",
            "model": None,
            "completed_at": None,
        }
        for i in range(MAX_RUNNING_AGENTS + 5)
    }
    state = {"running_agents": agents}

    removed = prune_stale_agents(state, now=now)

    assert removed == 5
    kept = state["running_agents"]
    assert len(kept) == MAX_RUNNING_AGENTS
    assert set(kept.keys()) == {f"agent-{i}" for i in range(MAX_RUNNING_AGENTS)}


def test_prune_handles_empty_and_missing_registry() -> None:
    assert prune_stale_agents({}) == 0
    assert prune_stale_agents({"running_agents": {}}) == 0
    assert prune_stale_agents({"running_agents": None}) == 0


def test_registry_model_label_reuses_normalize_model_symbol() -> None:
    from iai_mcp import daemon_state

    assert daemon_state.normalize_model is model_attribution.normalize_model

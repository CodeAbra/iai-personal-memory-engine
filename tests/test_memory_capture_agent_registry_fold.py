"""Coverage for folding agent spawn/completion into the memory_capture write path.

memory_capture's dispatch handler gains optional agent_id / agent_role /
agent_expected_artifact / agent_complete_id / agent_model params on the same
seam next_action/focus already use -- a manual convention, not a new tool.
register/complete run AFTER the ordinary capture_turn and, unlike
next_action/focus, never reach working_tier._persist_entry's own continuity
refresh -- so the dispatch calls session.write_continuity_cache(store)
itself, right after each register/complete.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iai_mcp import daemon_state
from iai_mcp.core import dispatch
from iai_mcp.store import MemoryStore


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryStore:
    # daemon_state.json is store-independent (resolved off IAI_MCP_STORE, not
    # store.root) -- pin it into the same tmp tree so the registry and the
    # continuity cache both land under a hermetic path.
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    s = MemoryStore(path=tmp_path / "lancedb")
    yield s
    s.close()


def _continuity_text(s: MemoryStore) -> str:
    path = s.root / ".session-continuity.cached.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _agent_block(text: str) -> str:
    if "<iai-mcp-agent-registry>" not in text:
        return ""
    return text.split("<iai-mcp-agent-registry>\n", 1)[1].split(
        "</iai-mcp-agent-registry>", 1,
    )[0]


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_spawn_registers_agent_and_refreshes_continuity_agent_block(
    driver, store, monkeypatch,
) -> None:
    _select_driver(driver, monkeypatch)

    result = dispatch(
        store, "memory_capture",
        {
            "text": "spawning a research agent for the recall regression",
            "cue": "c",
            "session_id": "sess-spawn",
            "role": "user",
            "agent_id": "a1",
            "agent_role": "research",
            "agent_expected_artifact": "RESEARCH.md",
            "agent_model": "claude-sonnet-5",
        },
    )
    assert result["status"] == "inserted", result

    entry = daemon_state.load_state()["running_agents"]["a1"]
    assert entry["status"] == "pending"
    assert entry["role"] == "research"
    assert entry["expected_artifact"] == "RESEARCH.md"
    assert entry["model"] == daemon_state.normalize_model("claude-sonnet-5")
    assert entry["completed_at"] is None

    agent_block = _agent_block(_continuity_text(store))
    assert "a1" in agent_block
    assert "research" in agent_block


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_completion_marks_agent_and_removes_it_from_the_continuity_agent_block(
    driver, store, monkeypatch,
) -> None:
    _select_driver(driver, monkeypatch)

    dispatch(
        store, "memory_capture",
        {
            "text": "spawning a research agent for the recall regression",
            "cue": "c",
            "session_id": "sess-complete",
            "role": "user",
            "agent_id": "a1",
            "agent_role": "research",
            "agent_expected_artifact": "RESEARCH.md",
        },
    )
    assert "a1" in _agent_block(_continuity_text(store))

    result = dispatch(
        store, "memory_capture",
        {
            "text": "the research agent finished and handed back the artifact",
            "cue": "c",
            "session_id": "sess-complete",
            "role": "user",
            "agent_complete_id": "a1",
        },
    )
    assert result["status"] == "inserted", result

    entry = daemon_state.load_state()["running_agents"]["a1"]
    assert entry["status"] == "complete"
    assert entry["completed_at"] is not None

    assert "a1" not in _agent_block(_continuity_text(store))


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_ordinary_capture_without_agent_params_leaves_registry_untouched(
    driver, store, monkeypatch,
) -> None:
    _select_driver(driver, monkeypatch)

    calls: list[dict] = []
    original_register = daemon_state.register_running_agent
    original_complete = daemon_state.complete_running_agent
    monkeypatch.setattr(
        daemon_state, "register_running_agent",
        lambda **kw: (calls.append(kw), original_register(**kw))[1],
    )
    monkeypatch.setattr(
        daemon_state, "complete_running_agent",
        lambda agent_id: (calls.append({"agent_id": agent_id}), original_complete(agent_id))[1],
    )

    result = dispatch(
        store, "memory_capture",
        {
            "text": "an ordinary capture with no agent fields set",
            "cue": "c",
            "session_id": "sess-plain",
            "role": "user",
        },
    )
    assert result["status"] == "inserted", result
    assert calls == [], "register/complete must never fire when no agent params are present"
    assert daemon_state.load_state().get("running_agents", {}) == {}


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_partial_spawn_missing_expected_artifact_is_a_noop(
    driver, store, monkeypatch,
) -> None:
    _select_driver(driver, monkeypatch)

    result = dispatch(
        store, "memory_capture",
        {
            "text": "a spawn attempt missing the expected artifact field",
            "cue": "c",
            "session_id": "sess-partial",
            "role": "user",
            "agent_id": "a1",
            "agent_role": "research",
        },
    )
    assert result["status"] == "inserted", result
    assert daemon_state.load_state().get("running_agents", {}) == {}

from __future__ import annotations


from iai_mcp.capture import capture_turn
from iai_mcp.core import dispatch
from tests._helpers import make_tmp_store


def test_cross_session_recency_query(tmp_path):
    store = make_tmp_store(tmp_path)

    b_session_id = "b-session-111"
    distinctive_phrase = "distinctive phrase bxyz turnmark59 cross session marker"

    result_b = capture_turn(
        store,
        cue="session b distinctive phrase",
        text=distinctive_phrase,
        tier="episodic",
        session_id=b_session_id,
        role="user",
    )
    assert result_b["status"] == "inserted", f"session B insert failed: {result_b}"

    a_session_id = "a-session-222"
    for i in range(2):
        capture_turn(
            store,
            cue=f"session a turn {i}",
            text=f"session a turn {i} normal content for e2e test turnmark59",
            tier="episodic",
            session_id=a_session_id,
            role="user",
        )

    global_result = dispatch(store, "episodes_recent", {"n": 5})
    assert "turns" in global_result, (
        f"episodes_recent global query missing 'turns': {global_result!r}"
    )
    global_surfaces = [t.get("literal_surface", "") for t in global_result["turns"]]
    assert any(distinctive_phrase in s for s in global_surfaces), (
        f"session B's phrase not found in global query result; "
        f"surfaces: {global_surfaces!r}"
    )

    filtered_result = dispatch(
        store,
        "episodes_recent",
        {"n": 5, "session_id": b_session_id},
    )
    assert "turns" in filtered_result, (
        f"episodes_recent session-filtered query missing 'turns': {filtered_result!r}"
    )
    filtered_turns = filtered_result["turns"]
    assert filtered_turns, "session-filtered query returned no turns"
    top = filtered_turns[0]
    assert top.get("literal_surface") == distinctive_phrase, (
        f"top session-filtered turn must be B's phrase; "
        f"got {top.get('literal_surface')!r}"
    )
    assert top.get("session_id") == b_session_id, (
        f"session_id on filtered top turn must be {b_session_id!r}; "
        f"got {top.get('session_id')!r}"
    )


def test_pending_live_visible_across_sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "test.sock"))

    store = make_tmp_store(tmp_path)

    a_session_id = "cross-live-a-session"
    live_text = "cross session pending live turn text long enough for capture test"

    from iai_mcp.capture import write_deferred_event
    write_deferred_event(a_session_id, "user", live_text)

    result = dispatch(
        store,
        "episodes_recent",
        {"n": 10, "session_id": a_session_id},
    )
    assert "turns" in result, f"episodes_recent missing 'turns': {result!r}"
    turns = result["turns"]
    assert len(turns) >= 1, f"expected >= 1 pending turn; got {len(turns)}"
    surfaces = [t.get("literal_surface", "") for t in turns]
    assert any(live_text in s for s in surfaces), (
        f"pending live turn must be returned without drain; "
        f"got surfaces: {surfaces!r}"
    )
    assert all(t["record_id"] != "None" for t in turns), (
        f"pending turn record_id must not be literal 'None'"
    )

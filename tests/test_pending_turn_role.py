from __future__ import annotations

import datetime


from iai_mcp.store import _PendingTurn

_TS = datetime.datetime(2026, 5, 31, 12, 0, 0, tzinfo=datetime.timezone.utc)

def _make(role: str | None = None, **kwargs) -> _PendingTurn:
    base = dict(
        text="some text",
        session_id="sess-test-01",
        ts=_TS,
        idem_tag="idem:abc123",
        source_uuid=None,
    )
    if role is not None:
        base["role"] = role
    base.update(kwargs)
    return _PendingTurn(**base)

def test_assistant_pending_turn_tags():
    pt = _make(role="assistant")
    assert "role:assistant" in pt.tags, f"Expected 'role:assistant' in tags, got {pt.tags}"

def test_assistant_pending_turn_provenance():
    pt = _make(role="assistant")
    prov = pt.provenance
    assert prov, "provenance is empty"
    assert prov[0]["role"] == "assistant", (
        f"Expected provenance role 'assistant', got {prov[0].get('role')}"
    )

def test_default_role_is_user_in_tags():
    pt = _make()
    assert "role:user" in pt.tags, f"Expected 'role:user' in tags, got {pt.tags}"

def test_default_role_is_user_in_provenance():
    pt = _make()
    prov = pt.provenance
    assert prov, "provenance is empty"
    assert prov[0]["role"] == "user", (
        f"Expected provenance role 'user', got {prov[0].get('role')}"
    )

def test_explicit_user_role_in_tags():
    pt = _make(role="user")
    assert "role:user" in pt.tags

def test_explicit_user_role_in_provenance():
    pt = _make(role="user")
    assert pt.provenance[0]["role"] == "user"

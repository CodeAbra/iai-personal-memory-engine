from __future__ import annotations



def test_hash_component_deterministic():
    from iai_mcp.delta import hash_component

    a = hash_component("hello world")
    b = hash_component("hello world")
    c = hash_component("hello world!")
    assert a == b
    assert a != c


def test_hash_component_returns_hex_string():
    from iai_mcp.delta import hash_component

    h = hash_component("test")
    assert isinstance(h, str)
    assert len(h) == 16
    int(h, 16)


def test_build_delta_first_session_returns_full_payload():
    from iai_mcp.delta import build_delta

    payload = {
        "l0": "identity",
        "l1": "critical facts",
        "l2": ["community a", "community b"],
        "rich_club": "hubs",
    }
    delta, new_hashes = build_delta({}, payload)
    assert "l0" in delta
    assert "l1" in delta
    assert "l2" in delta
    assert "rich_club" in delta
    for k in ("l0", "l1", "l2", "rich_club"):
        assert k in new_hashes


def test_build_delta_unchanged_is_empty():
    from iai_mcp.delta import build_delta

    payload = {
        "l0": "identity",
        "l1": "critical facts",
        "l2": ["community a"],
        "rich_club": "hubs",
    }
    _first, hashes = build_delta({}, payload)
    delta2, _hashes2 = build_delta(hashes, payload)
    assert delta2 == {}


def test_build_delta_partial_change():
    from iai_mcp.delta import build_delta

    payload_a = {
        "l0": "identity",
        "l1": "critical facts",
        "l2": ["community a"],
        "rich_club": "hubs",
    }
    _first, hashes = build_delta({}, payload_a)
    payload_b = dict(payload_a)
    payload_b["l2"] = ["community a", "community b"]
    delta, new_hashes = build_delta(hashes, payload_b)
    assert "l2" in delta
    assert "l0" not in delta
    assert "l1" not in delta
    assert "rich_club" not in delta


def test_apply_delta_reconstructs():
    from iai_mcp.delta import apply_delta, build_delta

    base = {"l0": "a", "l1": "b", "l2": ["x"], "rich_club": "c"}
    _first, hashes = build_delta({}, base)
    new = {"l0": "z", "l1": "b", "l2": ["x"], "rich_club": "c"}
    delta, _ = build_delta(hashes, new)
    reconstructed = apply_delta(base, delta)
    assert reconstructed == new


def test_delta_on_hash_miss_returns_full_component():
    from iai_mcp.delta import build_delta

    stale = {"l0": "deadbeef00000000", "l1": "cafebabe00000000"}
    payload = {"l0": "new", "l1": "facts", "l2": [], "rich_club": ""}
    delta, _ = build_delta(stale, payload)
    assert "l0" in delta
    assert delta["l0"] == "new"
    assert "l1" in delta

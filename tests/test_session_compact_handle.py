from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp.handle import (
    COMPACT_HANDLE_RE,
    COMPACT_HANDLE_TOKEN_BUDGET,
    decode_compact_handle,
    encode_compact_handle,
    _reset_cache_for_tests,
)
from iai_mcp.session import _approx_tokens

@pytest.fixture(autouse=True)
def _fresh_handle_cache():
    _reset_cache_for_tests()
    yield
    _reset_cache_for_tests()

@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(
        _keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p)
    )
    monkeypatch.setattr(
        _keyring, "delete_password", lambda s, u: fake.pop((s, u), None)
    )
    yield fake

@pytest.fixture
def _fresh_store(tmp_path: Path):
    from iai_mcp.store import MemoryStore

    return MemoryStore(path=tmp_path / "lancedb")

def _assemble_with_wake_depth(store, wake_depth):
    from iai_mcp import retrieve
    from iai_mcp.session import assemble_session_start

    _graph, assignment, rc = retrieve.build_runtime_graph(store)
    return assemble_session_start(
        store,
        assignment,
        rc,
        session_id=str(uuid4()),
        profile_state={"wake_depth": wake_depth},
    )

def test_minimal_payload_carries_non_empty_compact_handle(_fresh_store):
    payload = _assemble_with_wake_depth(_fresh_store, "minimal")
    assert payload.wake_depth == "minimal"
    assert payload.compact_handle != ""
    assert COMPACT_HANDLE_RE.fullmatch(payload.compact_handle)

def test_encode_is_deterministic():
    a = encode_compact_handle("abcdef01", "12345678", "general", 3)
    b = encode_compact_handle("abcdef01", "12345678", "general", 3)
    assert a == b
    assert COMPACT_HANDLE_RE.fullmatch(a)

def test_decode_round_trips_for_lru_hit():
    handle = encode_compact_handle("feedface", "cafebabe", "security", 7)
    parts = decode_compact_handle(handle)
    assert parts is not None
    assert parts[0] == "feedface"
    assert parts[1] == "cafebabe"
    assert parts[2] == "security"
    assert parts[3] == 7

def test_decode_cold_cache_returns_none():
    fake = "<iai:" + ("a" * 16) + ">"
    assert decode_compact_handle(fake) is None

@pytest.mark.parametrize(
    "malformed",
    [
        "",
        "abcdef0123456789",
        "<iai:ABCDEF0123456789>",
        "<iai:xyz>",
        "<iai:" + ("a" * 15) + ">",
        "<iai:" + ("a" * 17) + ">",
        "<id:abcdef01>",
        None,
        12345,
    ],
)
def test_decode_rejects_malformed(malformed):
    assert decode_compact_handle(malformed) is None

def test_standard_and_deep_populate_compact_handle_for_back_compat(_fresh_store):
    for depth in ("standard", "deep"):
        payload = _assemble_with_wake_depth(_fresh_store, depth)
        assert payload.wake_depth == depth
        assert payload.compact_handle != "", f"compact_handle missing at wake_depth={depth}"
        assert COMPACT_HANDLE_RE.fullmatch(payload.compact_handle)

def test_minimal_payload_cached_tokens_within_budget(_fresh_store):
    payload = _assemble_with_wake_depth(_fresh_store, "minimal")
    assert payload.total_cached_tokens <= COMPACT_HANDLE_TOKEN_BUDGET, (
        f"cached={payload.total_cached_tokens} exceeds budget "
        f"{COMPACT_HANDLE_TOKEN_BUDGET}"
    )
    assert _approx_tokens(payload.compact_handle) <= COMPACT_HANDLE_TOKEN_BUDGET

def test_compact_handle_is_hex_only_no_knob_leak():
    import re

    knob_names = [
        "wake_depth",
        "autistic_mode",
        "hebbian_rate",
        "camouflaging_relaxation",
        "response_formality",
    ]
    handle = encode_compact_handle("abcdef01", "12345678", "general", 0)
    for name in knob_names:
        assert name not in handle, f"knob {name!r} leaked into {handle!r}"
    body = handle[5:-1]
    assert re.fullmatch(r"[0-9a-f]{16}", body)

def test_minimal_cached_count_charges_only_compact_handle(_fresh_store):
    payload = _assemble_with_wake_depth(_fresh_store, "minimal")
    assert payload.brain_handle.startswith("<sess:")
    assert payload.topic_cluster_hint.startswith("<topic:")
    assert payload.total_cached_tokens == _approx_tokens(payload.compact_handle)
    assert payload.total_cached_tokens <= COMPACT_HANDLE_TOKEN_BUDGET

def test_resolve_compact_handle_rebuilds_legacy_triple():
    from iai_mcp.session import _resolve_compact_handle_to_pointers

    handle = encode_compact_handle("abcdef01", "12345678", "general", 4)
    triple = _resolve_compact_handle_to_pointers(handle)
    assert triple is not None
    identity_pointer, brain_handle, topic_cluster_hint = triple
    assert identity_pointer == "<id:abcdef01>"
    assert brain_handle == "<sess:12345678 pend:4>"
    assert topic_cluster_hint == "<topic:general>"

def test_resolve_compact_handle_returns_none_for_unknown():
    from iai_mcp.session import _resolve_compact_handle_to_pointers

    fake = "<iai:" + ("b" * 16) + ">"
    assert _resolve_compact_handle_to_pointers(fake) is None

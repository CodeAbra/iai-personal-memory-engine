from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

@pytest.fixture(autouse=True)
def _reset_session_refresh_debounce():
    from iai_mcp.core import _reset_session_refresh_debounce as _reset
    _reset()
    yield
    _reset()

@pytest.fixture
def iai_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-session-refresh-passphrase")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp" / "lancedb"))

    import keyring.core
    keyring.core._keyring_backend = None
    yield tmp_path
    keyring.core._keyring_backend = None

def _open_store():
    from iai_mcp.store import MemoryStore
    return MemoryStore()

def _insert_record(store, text: str):
    from iai_mcp.capture import capture_turn
    return capture_turn(store, text=text, cue="test cue", tier="episodic", role="user")

def _write_drainable_deferred(home: Path, session_id: str, text: str) -> Path:
    deferred_dir = home / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    suffix = int(time.time())
    out = deferred_dir / f"{session_id}-{suffix}.jsonl"
    header = {
        "version": 1,
        "deferred_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "cwd": "/tmp",
    }
    event = {
        "text": text,
        "cue": f"session {session_id} deferred cue",
        "tier": "episodic",
        "role": "user",
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    out.write_text(
        json.dumps(header, ensure_ascii=False) + "\n"
        + json.dumps(event, ensure_ascii=False) + "\n"
    )
    return out

def test_max_created_at_empty_store(iai_home):
    from iai_mcp.session import max_record_created_at
    store = _open_store()
    assert max_record_created_at(store) is None

def test_max_created_at_after_insert(iai_home):
    from iai_mcp.session import max_record_created_at
    store = _open_store()
    _insert_record(store, "alice said: first record inserted for max_created_at test")
    result = max_record_created_at(store)
    assert result is not None
    assert isinstance(result, str)
    _insert_record(store, "alice said: second record, must be at least 12 chars")
    result2 = max_record_created_at(store)
    assert result2 is not None
    assert result2 >= result

def test_not_stale_returns_empty(iai_home):
    from iai_mcp.session import max_record_created_at
    from iai_mcp.core import dispatch
    store = _open_store()
    _insert_record(store, "alice said: record present before watermark is set")

    current_max = max_record_created_at(store)
    assert current_max is not None

    result = dispatch(store, "session_refresh_if_stale", {
        "watermark": current_max,
        "session_id": "test-session-no-op",
    })
    assert result["rendered"] == ""

def _seed_dense_fixture(store) -> None:
    """Populate a corpus dense enough that the full session-start brief
    clears DELTA_MAX_TOKENS, so the token-ceiling assertions measure the
    real per-turn payload rather than an emptily-passing brief."""
    from datetime import datetime, timezone
    from uuid import uuid4
    from iai_mcp.core import _seed_l0_identity
    from iai_mcp.store import flush_record_buffer
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    _seed_l0_identity(store)

    now = datetime.now(timezone.utc)
    for i in range(10):
        rec = MemoryRecord(
            id=uuid4(),
            tier="semantic",
            literal_surface=(
                f"Pinned reference fact number {i}: durable project detail padded "
                f"with enough distinct wording to clear the per-line token floor "
                f"for segment {i}."
            ),
            aaak_index="",
            embedding=[0.1] * EMBED_DIM,
            community_id=None,
            centrality=0.5,
            detail_level=5,
            pinned=True,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=True,
            never_merge=False,
            provenance=[],
            created_at=now,
            updated_at=now,
            tags=[],
            language="en",
        )
        store.insert(rec)

    for i in range(5):
        _insert_record(
            store,
            f"alice recent work note {i}: long enough conversational turn text "
            f"to fill the recent-thread segment line with distinct content for "
            f"turn number {i}",
        )

    flush_record_buffer(store)

def test_stale_returns_nonempty(iai_home):
    from iai_mcp.core import dispatch
    from iai_mcp.session import (
        DELTA_MAX_TOKENS,
        _approx_tokens,
        _compose_session_start_payload,
        format_payload_as_markdown,
    )
    store = _open_store()

    _seed_dense_fixture(store)

    old_watermark = "2000-01-01T00:00:00+00:00"

    result = dispatch(store, "session_refresh_if_stale", {
        "watermark": old_watermark,
        "session_id": "test-session-stale",
    })
    assert result["rendered"] != "", "vacuous render"
    assert result["new_max_ts"] > old_watermark, "vacuous render"

    from iai_mcp import retrieve
    _graph, assignment, rc = retrieve.build_runtime_graph(store)
    full_payload = _compose_session_start_payload(
        store, assignment, rc, session_id="x", profile_state={"wake_depth": "standard"},
    )
    full = format_payload_as_markdown(full_payload)
    assert _approx_tokens(full) > DELTA_MAX_TOKENS, "fixture too sparse"

    assert _approx_tokens(result["rendered"]) <= DELTA_MAX_TOKENS, "per-turn render exceeds token ceiling"

def test_per_turn_delta_token_ceiling(iai_home):
    from uuid import uuid4
    from iai_mcp.core import dispatch
    from iai_mcp.session import (
        DELTA_MAX_TOKENS,
        _approx_tokens,
        _compose_session_start_payload,
        format_payload_as_markdown,
        max_record_created_at,
    )
    store = _open_store()

    _seed_dense_fixture(store)

    watermark = max_record_created_at(store)
    assert watermark is not None

    marker = f"deltamarker{uuid4().hex}"
    first = _insert_record(
        store,
        f"{marker} distinct fresh delta content for the store-advance probe, entry one"
    )
    second = _insert_record(
        store,
        "alice unrelated distinct fresh delta content for the store-advance probe, entry two"
    )
    assert first["status"] == "inserted"
    assert second["status"] == "inserted"

    result = dispatch(store, "session_refresh_if_stale", {
        "watermark": watermark,
        "session_id": "delta-ceiling-test",
    })

    assert _approx_tokens(result["rendered"]) <= DELTA_MAX_TOKENS, "per-turn render exceeds token ceiling"

    rendered = result["rendered"]
    assert "## Key memories" not in rendered, "static brief header present"
    assert "## Topic communities" not in rendered, "static brief header present"
    assert "## Critical facts" not in rendered, "static brief header present"
    assert "## Identity" not in rendered, "static brief header present"
    assert "ages like" not in rendered, "static brief header present"

    assert rendered != "", "vacuous render"
    assert result["new_max_ts"] > watermark, "vacuous render"
    assert marker in rendered, "vacuous render"

    from iai_mcp import retrieve
    _graph, assignment, rc = retrieve.build_runtime_graph(store)
    full_payload = _compose_session_start_payload(
        store, assignment, rc, session_id="x", profile_state={"wake_depth": "standard"},
    )
    full = format_payload_as_markdown(full_payload)
    assert _approx_tokens(full) > DELTA_MAX_TOKENS, "fixture too sparse"

def test_emit_free(iai_home):
    from iai_mcp.events import flush_event_buffer, query_events
    from iai_mcp.core import dispatch
    store = _open_store()

    for i in range(3):
        _insert_record(
            store,
            f"alice emit-free test record {i} with enough length to pass min_capture_len"
        )

    flush_event_buffer(store)
    before = len(query_events(store, kind="session_started"))

    dispatch(store, "session_refresh_if_stale", {
        "watermark": "2000-01-01T00:00:00+00:00",
        "session_id": "test-session-emit-free",
    })

    flush_event_buffer(store)
    after = len(query_events(store, kind="session_started"))
    assert after == before, (
        f"session_refresh_if_stale emitted {after - before} session_started event(s); "
        "expected 0 (emit-free path required)"
    )

def test_sc4_drain_before_read(iai_home):
    from iai_mcp.session import max_record_created_at
    from iai_mcp.core import dispatch

    store = _open_store()
    old_watermark = max_record_created_at(store) or "2000-01-01T00:00:00+00:00"

    unique_text = "alice SC4 deferred unique content for drain-before-read assertion test"
    _write_drainable_deferred(iai_home, "sc4-test-session", unique_text)

    assert not any(
        "SC4 deferred unique content" in (r.literal_surface or "")
        for r in store.all_records()
    ), "Entry must be absent from store before drain runs"

    result = dispatch(store, "session_refresh_if_stale", {
        "watermark": old_watermark,
        "session_id": "sc4-test-session",
    })

    assert any(
        "SC4 deferred unique content" in (r.literal_surface or "")
        for r in store.all_records()
    ), "Entry must be present in store after drain-before-read RPC"

    post_max = max_record_created_at(store)
    assert post_max is not None, "Store should have records after drain"
    assert post_max > old_watermark, (
        f"MAX(created_at) should have advanced: pre={old_watermark}, post={post_max}"
    )
    assert result["new_max_ts"] == post_max

def test_sc5_global_store_cross_cwd(iai_home):
    import os

    dir_a = iai_home / "project_a"
    dir_b = iai_home / "project_b"
    dir_a.mkdir(parents=True, exist_ok=True)
    dir_b.mkdir(parents=True, exist_ok=True)

    original_cwd = os.getcwd()
    try:
        os.chdir(str(dir_a))
        store_a = _open_store()
        result = _insert_record(
            store_a,
            "alice project-b work: unique cross-cwd test record for SC5 global store assertion"
        )
        assert result.get("status") in ("inserted", "reinforced"), (
            f"Insert failed: {result}"
        )

        os.chdir(str(dir_b))
        store_b = _open_store()
        all_records = store_b.all_records()
        texts = [r.literal_surface for r in all_records]
        found = any(
            "SC5 global store" in t or "cross-cwd test record" in t
            for t in texts
        )
        assert found, (
            "SC5 FAIL: record inserted under dir_a not found under dir_b. "
            f"Store root must be HOME-based, not cwd-based. "
            f"Records found: {[t[:60] for t in texts]}"
        )
    finally:
        os.chdir(original_cwd)

def test_debounce_suppresses_and_accumulates(iai_home):
    from iai_mcp.core import _reset_session_refresh_debounce, dispatch
    from iai_mcp.session import max_record_created_at

    store = _open_store()
    _insert_record(store, "alice seed record placed before the debounce probe watermark")
    watermark = max_record_created_at(store)
    assert watermark is not None

    session_id = "debounce-probe-session"

    _insert_record(
        store,
        "alice batch one distinct content for the debounce accumulation probe"
    )
    first = dispatch(store, "session_refresh_if_stale", {
        "watermark": watermark,
        "session_id": session_id,
    })
    assert first["rendered"] != "", "first fire must render"
    assert "batch one" in first["rendered"]

    second = dispatch(store, "session_refresh_if_stale", {
        "watermark": watermark,
        "session_id": session_id,
    })
    assert second["rendered"] == "", "immediate refire must be debounced"

    _insert_record(
        store,
        "alice batch two distinct content for the debounce accumulation probe"
    )
    third = dispatch(store, "session_refresh_if_stale", {
        "watermark": watermark,
        "session_id": session_id,
    })
    assert third["rendered"] == "", (
        "refire after a fresh store advance must still be debounced, "
        "not merely re-suppressed by the staleness no-op"
    )

    _reset_session_refresh_debounce()

    fourth = dispatch(store, "session_refresh_if_stale", {
        "watermark": watermark,
        "session_id": session_id,
    })
    assert "batch one" in fourth["rendered"], "batch one lost across the suppressed window"
    assert "batch two" in fourth["rendered"], "batch two lost across the suppressed window"

def test_delta_overflow_marker_and_ceiling(iai_home):
    from iai_mcp.core import dispatch
    from iai_mcp.session import DELTA_MAX_TOKENS, K_DELTA, _approx_tokens, max_record_created_at

    store = _open_store()
    _insert_record(store, "alice seed record placed before the overflow probe watermark")
    watermark = max_record_created_at(store)
    assert watermark is not None

    for i in range(K_DELTA + 6):
        _insert_record(
            store,
            f"alice overflow probe distinct fresh record number {i} padded with enough "
            f"unique wording to avoid dedup collisions across the burst"
        )

    result = dispatch(store, "session_refresh_if_stale", {
        "watermark": watermark,
        "session_id": "overflow-probe-session",
    })

    rendered = result["rendered"]
    assert rendered != "", "vacuous render"
    assert "elided" in rendered, "truncation marker missing on overflow burst"
    assert _approx_tokens(rendered) <= DELTA_MAX_TOKENS, "overflow render exceeds token ceiling"

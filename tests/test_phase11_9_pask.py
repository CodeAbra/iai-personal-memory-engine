from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from iai_mcp.core import dispatch
from iai_mcp.daemon import PaskConfig, _load_pask_config
from iai_mcp.events import query_events
from iai_mcp.pask_teachback import verify_hit_set
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord

@pytest.fixture(autouse=True)
def _isolate_iai_pask(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai-mcp-store"))
    monkeypatch.setenv("IAI_MCP_KEYRING_BYPASS", "true")
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL", raising=False)
    for var in (
        "IAI_MCP_PASK_ENABLED",
        "IAI_MCP_PASK_DRY_RUN",
    ):
        monkeypatch.delenv(var, raising=False)

def _make_record(
    *,
    embed_dim: int,
    literal_surface: str = "alice prefers tea over coffee",
    community_id: uuid.UUID | None = None,
    created_at: datetime | None = None,
    embedding: list[float] | None = None,
) -> MemoryRecord:
    now = created_at if created_at is not None else datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=literal_surface,
        aaak_index="",
        embedding=embedding if embedding is not None else [0.01] * embed_dim,
        community_id=community_id,
        centrality=0.5,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        language="en",
        tags=["t"],
    )

def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(
        path=str(tmp_path / "iai-mcp-store"),
        user_id="alice",
        read_consistency_interval=timedelta(seconds=0),
    )

def _seed_orthogonal_records(
    store: MemoryStore, labels: list[str],
) -> list[uuid.UUID]:
    embed_dim = store._embed_dim
    ids: list[uuid.UUID] = []
    for i, surface in enumerate(labels):
        emb = [0.0] * embed_dim
        emb[i % embed_dim] = 1.0
        rec = _make_record(
            embed_dim=embed_dim,
            literal_surface=surface,
            embedding=emb,
        )
        store.insert(rec)
        ids.append(rec.id)
    return ids

def test_verify_hit_set_empty_input_no_contradictions(tmp_path: Path) -> None:
    store = _make_store(tmp_path)

    result = verify_hit_set(store, [])

    assert isinstance(result, dict)
    assert result["has_contradictions"] is False, (
        f"empty input must report has_contradictions=False; got {result!r}"
    )
    assert result["contradiction_pairs"] == [], (
        f"empty input must report empty contradiction_pairs; got {result!r}"
    )
    assert result["hit_count"] == 0, (
        f"hit_count must echo len(hit_record_ids); got {result!r}"
    )
    assert isinstance(result["teachback_summary"], str)
    assert "0" in result["teachback_summary"], (
        f"teachback_summary must mention hit_count=0; "
        f"got {result['teachback_summary']!r}"
    )

def test_verify_hit_set_detects_contradicts_edge(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    ids = _seed_orthogonal_records(
        store, ["alice topic alpha", "bob topic beta"],
    )
    rec_a, rec_b = ids[0], ids[1]

    store.boost_edges(
        [(rec_a, rec_b)], delta=1.0, edge_type="contradicts",
    )

    result = verify_hit_set(store, [rec_a, rec_b])

    assert result["has_contradictions"] is True, (
        f"a contradicts edge between two hit ids must surface; "
        f"got {result!r}"
    )
    assert result["hit_count"] == 2
    assert len(result["contradiction_pairs"]) == 1, (
        f"exactly one contradicts edge was seeded; "
        f"got {result['contradiction_pairs']!r}"
    )

    surfaced = result["contradiction_pairs"][0]
    expected_sorted = tuple(sorted([str(rec_a), str(rec_b)]))
    assert tuple(sorted(surfaced)) == expected_sorted, (
        f"surfaced contradicts pair must match the seeded UUIDs "
        f"(order-independent); got {surfaced!r}, expected sorted "
        f"{expected_sorted!r}"
    )

    assert "WARNING" in result["teachback_summary"], (
        f"teachback_summary must carry WARNING marker on a contradicts "
        f"hit; got {result['teachback_summary']!r}"
    )

def test_verify_hit_set_no_edges_returns_consistent(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    ids = _seed_orthogonal_records(
        store,
        ["alice topic alpha", "bob topic beta", "alice and bob agree on tea"],
    )

    store.boost_edges(
        [(ids[0], ids[1])], delta=1.0, edge_type="hebbian",
    )

    result = verify_hit_set(store, ids)

    assert result["has_contradictions"] is False, (
        f"no contradicts edges must report has_contradictions=False "
        f"(a hebbian edge must NOT bleed into the filter); got {result!r}"
    )
    assert result["contradiction_pairs"] == []
    assert result["hit_count"] == 3
    assert "consistent" in result["teachback_summary"].lower(), (
        f"clean-pass summary must mention 'consistent'; "
        f"got {result['teachback_summary']!r}"
    )

def test_memory_recall_payload_contains_teachback(tmp_path: Path) -> None:
    store = _make_store(tmp_path)

    response = dispatch(
        store,
        "memory_recall",
        {"cue": "alice prefers tea", "session_id": "test-session"},
    )

    assert isinstance(response, dict)
    assert "pask_teachback" in response, (
        f"memory_recall response must carry 'pask_teachback' key under "
        f"PASK_ENABLED=true (default); got keys {sorted(response.keys())!r}"
    )
    teachback = response["pask_teachback"]
    assert isinstance(teachback, dict)
    assert teachback["has_contradictions"] is False
    assert teachback["contradiction_pairs"] == []
    assert teachback["hit_count"] == 0
    assert isinstance(teachback["teachback_summary"], str)

def test_memory_recall_emits_pask_event_with_dry_run_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_PASK_DRY_RUN", "true")

    store = _make_store(tmp_path)

    response = dispatch(
        store,
        "memory_recall",
        {"cue": "alice prefers tea", "session_id": "test-session"},
    )

    assert "pask_teachback" in response

    events = query_events(
        store, kind="pask_teachback_pass", limit=10,
    )
    assert len(events) >= 1, (
        f"memory_recall must emit at least one pask_teachback_pass "
        f"event under PASK_ENABLED=true; got {len(events)}"
    )
    body = events[0]["data"]
    assert "hit_count" in body
    assert "has_contradictions" in body
    assert "contradiction_count" in body
    assert "dry_run_mode" in body
    assert body["dry_run_mode"] is True, (
        f"dry_run_mode must echo IAI_MCP_PASK_DRY_RUN=true; got {body!r}"
    )

def test_pask_disabled_skips_response_key_and_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_PASK_ENABLED", "false")

    store = _make_store(tmp_path)

    response = dispatch(
        store,
        "memory_recall",
        {"cue": "alice prefers tea", "session_id": "test-session"},
    )

    assert "pask_teachback" not in response, (
        f"PASK_ENABLED=false must suppress the pask_teachback response "
        f"key; got keys {sorted(response.keys())!r}"
    )

    events = query_events(
        store, kind="pask_teachback_pass", limit=10,
    )
    assert len(events) == 0, (
        f"PASK_ENABLED=false must suppress ALL pask_teachback_pass "
        f"event emits; got {len(events)}"
    )

@pytest.mark.parametrize(
    "env_var, bad_value",
    [
        ("IAI_MCP_PASK_ENABLED", "not-a-bool"),
        ("IAI_MCP_PASK_ENABLED", "maybe"),
        ("IAI_MCP_PASK_ENABLED", "perhaps"),
        ("IAI_MCP_PASK_DRY_RUN", "bogus"),
        ("IAI_MCP_PASK_DRY_RUN", "maybe"),
        ("IAI_MCP_PASK_DRY_RUN", "kinda"),
    ],
)
def test_env_var_fail_loud_parametrized(
    monkeypatch: pytest.MonkeyPatch, env_var: str, bad_value: str,
) -> None:
    monkeypatch.setenv(env_var, bad_value)

    with pytest.raises(ValueError) as excinfo:
        _load_pask_config()

    assert env_var in str(excinfo.value), (
        f"ValueError must name the offending env var {env_var!r}; "
        f"got {excinfo.value!r}"
    )

def test_pask_config_defaults_under_pytest() -> None:
    cfg = _load_pask_config()
    assert isinstance(cfg, PaskConfig)
    assert cfg.enabled is True
    assert cfg.dry_run is True

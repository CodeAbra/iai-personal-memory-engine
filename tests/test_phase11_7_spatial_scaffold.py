from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from iai_mcp.daemon import SpatialConfig, _load_spatial_config
from iai_mcp.events import query_events
from iai_mcp.spatial_tagger import SpatialTagger
from iai_mcp.store import RECORDS_TABLE, MemoryStore
from iai_mcp.types import MemoryRecord

@pytest.fixture(autouse=True)
def _isolate_iai_spatial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai-mcp-store"))
    monkeypatch.setenv("IAI_MCP_KEYRING_BYPASS", "true")
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL", raising=False)
    for var in (
        "IAI_MCP_SPATIAL_AUTO_TAG",
        "IAI_MCP_SPATIAL_DEFAULT_WING",
        "IAI_MCP_SPATIAL_DRY_RUN",
    ):
        monkeypatch.delenv(var, raising=False)

def _make_record(
    *,
    embed_dim: int,
    provenance: list[dict] | None = None,
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface="alice prefers tea over coffee",
        aaak_index="",
        embedding=[0.01] * embed_dim,
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=provenance if provenance is not None else [],
        created_at=now,
        updated_at=now,
        language="en",
        tags=["t"],
    )

def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(str(tmp_path / "store"), user_id="alice")

def test_R1_fresh_store_has_wing_room_drawer_columns(tmp_path: Path) -> None:
    s = _make_store(tmp_path)
    names = s.db.open_table(RECORDS_TABLE).schema.names
    assert "wing" in names, names
    assert "room" in names, names
    assert "drawer" in names, names

def test_R2_spatial_tagger_path_heuristic_correctness() -> None:
    wing, room, drawer = SpatialTagger.tag(
        None, "/Users/alice/Desktop/IAI-MCP/src/iai_mcp/store.py",
    )
    assert wing == "IAI-MCP", wing
    assert room == "iai_mcp", room
    assert drawer == "store", drawer

    wing2, room2, drawer2 = SpatialTagger.tag(
        None, "/Users/alice/Documents/notes/today.md",
    )
    assert wing2 == "Documents", wing2
    assert room2 == "notes", room2
    assert drawer2 == "today", drawer2

    wing3, room3, drawer3 = SpatialTagger.tag(
        None, "/var/log/syslog", default_wing="general",
    )
    assert wing3 == "general", wing3
    assert room3 == "log", room3
    assert drawer3 == "syslog", drawer3

    for empty in (None, "", "   "):
        w, r, d = SpatialTagger.tag(None, empty, default_wing="general")
        assert w is None, (empty, w)
        assert r is None, (empty, r)
        assert d is None, (empty, d)

def test_R3_insert_path_auto_tag_populates_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_SPATIAL_AUTO_TAG", "true")
    monkeypatch.setenv("IAI_MCP_SPATIAL_DRY_RUN", "false")

    s = _make_store(tmp_path)
    rec = _make_record(
        embed_dim=s._embed_dim,
        provenance=[{
            "source_path": "/Users/alice/Desktop/IAI-MCP/src/iai_mcp/store.py",
            "ts": datetime.now(timezone.utc).isoformat(),
            "cue": "alice testing R3 positive path",
            "session_id": "s1",
        }],
    )
    s.insert(rec)

    assert getattr(rec, "wing", None) == "IAI-MCP"
    assert getattr(rec, "room", None) == "iai_mcp"
    assert getattr(rec, "drawer", None) == "store"

    df = s.db.open_table(RECORDS_TABLE).to_pandas()
    row = df[df["id"] == str(rec.id)].iloc[0]
    assert row["wing"] == "IAI-MCP", row["wing"]
    assert row["room"] == "iai_mcp", row["room"]
    assert row["drawer"] == "store", row["drawer"]

def test_R3_insert_path_skip_when_auto_tag_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_SPATIAL_AUTO_TAG", "false")

    s = _make_store(tmp_path)
    rec = _make_record(
        embed_dim=s._embed_dim,
        provenance=[{
            "source_path": "/Users/alice/Desktop/IAI-MCP/src/iai_mcp/store.py",
            "ts": datetime.now(timezone.utc).isoformat(),
            "cue": "alice testing R3 negative path",
            "session_id": "s1",
        }],
    )
    s.insert(rec)

    assert getattr(rec, "wing", None) is None
    assert getattr(rec, "room", None) is None
    assert getattr(rec, "drawer", None) is None

    df = s.db.open_table(RECORDS_TABLE).to_pandas()
    row = df[df["id"] == str(rec.id)].iloc[0]
    assert row["wing"] is None or (
        isinstance(row["wing"], float) and row["wing"] != row["wing"]
    ), row["wing"]
    assert row["room"] is None or (
        isinstance(row["room"], float) and row["room"] != row["room"]
    ), row["room"]
    assert row["drawer"] is None or (
        isinstance(row["drawer"], float) and row["drawer"] != row["drawer"]
    ), row["drawer"]

    events = query_events(s, kind="spatial_tag_pass")
    assert events == [], events

def test_R5_dry_run_no_mutation_but_event_emitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_SPATIAL_AUTO_TAG", "true")
    monkeypatch.setenv("IAI_MCP_SPATIAL_DRY_RUN", "true")

    s = _make_store(tmp_path)
    rec = _make_record(
        embed_dim=s._embed_dim,
        provenance=[{
            "source_path": "/Users/alice/Desktop/IAI-MCP/src/iai_mcp/store.py",
            "ts": datetime.now(timezone.utc).isoformat(),
            "cue": "alice testing R5 dry-run",
            "session_id": "s1",
        }],
    )
    s.insert(rec)

    assert getattr(rec, "wing", None) is None
    assert getattr(rec, "room", None) is None
    assert getattr(rec, "drawer", None) is None

    df = s.db.open_table(RECORDS_TABLE).to_pandas()
    row = df[df["id"] == str(rec.id)].iloc[0]
    assert row["wing"] is None or (
        isinstance(row["wing"], float) and row["wing"] != row["wing"]
    ), row["wing"]
    assert row["room"] is None or (
        isinstance(row["room"], float) and row["room"] != row["room"]
    ), row["room"]
    assert row["drawer"] is None or (
        isinstance(row["drawer"], float) and row["drawer"] != row["drawer"]
    ), row["drawer"]

    events = query_events(s, kind="spatial_tag_pass")
    assert len(events) == 1, f"expected exactly 1 event, got {len(events)}"
    body = events[0]["data"]
    assert body["wing"] == "IAI-MCP", body
    assert body["room"] == "iai_mcp", body
    assert body["drawer"] == "store", body
    assert body["dry_run_mode"] is True, body
    assert body["record_id"] == str(rec.id), body
    assert body["source_path"] == (
        "/Users/alice/Desktop/IAI-MCP/src/iai_mcp/store.py"
    ), body

@pytest.mark.parametrize(
    "env_var, bad_value",
    [
        ("IAI_MCP_SPATIAL_AUTO_TAG", "banana"),
        ("IAI_MCP_SPATIAL_AUTO_TAG", "maybe"),
        ("IAI_MCP_SPATIAL_DRY_RUN", "banana"),
        ("IAI_MCP_SPATIAL_DRY_RUN", "maybe"),
    ],
)
def test_R4_invalid_env_var_raises_ValueError_naming_var(
    monkeypatch: pytest.MonkeyPatch, env_var: str, bad_value: str,
) -> None:
    monkeypatch.setenv(env_var, bad_value)
    with pytest.raises(ValueError, match=env_var):
        _load_spatial_config()

def test_R4_defaults_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "IAI_MCP_SPATIAL_AUTO_TAG",
        "IAI_MCP_SPATIAL_DEFAULT_WING",
        "IAI_MCP_SPATIAL_DRY_RUN",
    ):
        monkeypatch.delenv(var, raising=False)
    cfg = _load_spatial_config()
    assert isinstance(cfg, SpatialConfig)
    assert cfg.auto_tag is False
    assert cfg.default_wing == "general"
    assert cfg.dry_run is True

if __name__ == "__main__":  # pragma: no cover -- direct-run convenience
    raise SystemExit(pytest.main([__file__, "-v"]))

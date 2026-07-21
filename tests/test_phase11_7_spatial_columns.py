from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from iai_mcp.store import RECORDS_TABLE, MemoryStore
from iai_mcp.types import MemoryRecord

@pytest.fixture(autouse=True)
def _isolate_iai_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai-mcp-store"))
    monkeypatch.setenv("IAI_MCP_KEYRING_BYPASS", "true")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-pp")
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL", raising=False)
    monkeypatch.delenv("IAI_MCP_SPATIAL_AUTO_TAG", raising=False)
    monkeypatch.delenv("IAI_MCP_SPATIAL_DEFAULT_WING", raising=False)
    monkeypatch.delenv("IAI_MCP_SPATIAL_DRY_RUN", raising=False)

def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(str(tmp_path / "store"), user_id="alice")

def _make_record(*, embed_dim: int) -> MemoryRecord:
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
        provenance=[],
        created_at=now,
        updated_at=now,
        language="en",
        tags=["t"],
    )

def test_fresh_store_has_spatial_columns(tmp_path: Path) -> None:
    s = _make_store(tmp_path)
    names = s.db.open_table(RECORDS_TABLE).schema.names
    assert "wing" in names, names
    assert "room" in names, names
    assert "drawer" in names, names

def test_preexisting_store_migrates_spatial_columns(tmp_path: Path) -> None:
    s = _make_store(tmp_path)
    tbl = s.db.open_table(RECORDS_TABLE)
    tbl.drop_columns(["wing", "room", "drawer"])
    names_pre = s.db.open_table(RECORDS_TABLE).schema.names
    assert "wing" not in names_pre
    assert "room" not in names_pre
    assert "drawer" not in names_pre
    s2 = MemoryStore(str(tmp_path / "store"), user_id="alice")
    names_post = s2.db.open_table(RECORDS_TABLE).schema.names
    assert "wing" in names_post
    assert "room" in names_post
    assert "drawer" in names_post
    s3 = MemoryStore(str(tmp_path / "store"), user_id="alice")
    names_post2 = s3.db.open_table(RECORDS_TABLE).schema.names
    assert "wing" in names_post2
    assert "room" in names_post2
    assert "drawer" in names_post2

def test_half_migrated_store_adds_only_missing(tmp_path: Path) -> None:
    s = _make_store(tmp_path)
    tbl = s.db.open_table(RECORDS_TABLE)
    tbl.drop_columns(["room", "drawer"])
    names_pre = s.db.open_table(RECORDS_TABLE).schema.names
    assert "wing" in names_pre
    assert "room" not in names_pre
    assert "drawer" not in names_pre
    s2 = MemoryStore(str(tmp_path / "store"), user_id="alice")
    names_post = s2.db.open_table(RECORDS_TABLE).schema.names
    assert "wing" in names_post
    assert "room" in names_post
    assert "drawer" in names_post

def test_legacy_row_reads_as_null_spatial(tmp_path: Path) -> None:
    s = _make_store(tmp_path)
    tbl = s.db.open_table(RECORDS_TABLE)
    tbl.drop_columns(["wing", "room", "drawer"])
    s2 = MemoryStore(str(tmp_path / "store"), user_id="alice")
    rec = _make_record(embed_dim=s2._embed_dim)
    s2.insert(rec)
    tbl2 = s2.db.open_table(RECORDS_TABLE)
    df = tbl2.to_pandas()
    assert len(df) >= 1
    assert "wing" in df.columns
    assert "room" in df.columns
    assert "drawer" in df.columns
    assert df["wing"].isna().all()
    assert df["room"].isna().all()
    assert df["drawer"].isna().all()

def test_add_columns_failure_raises_runtimeerror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _make_store(tmp_path)
    tbl = s.db.open_table(RECORDS_TABLE)
    tbl.drop_columns(["wing", "room", "drawer"])

    from iai_mcp.hippo import HippoTable

    original = HippoTable.add_columns

    def _raise(self: HippoTable, *args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated add_columns failure")

    monkeypatch.setattr(HippoTable, "add_columns", _raise)
    with pytest.raises(RuntimeError) as exc:
        MemoryStore(str(tmp_path / "store"), user_id="alice")
    msg = str(exc.value)
    assert any(name in msg for name in ("wing", "room", "drawer")), msg
    monkeypatch.setattr(HippoTable, "add_columns", original)

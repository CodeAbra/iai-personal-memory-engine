from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp.store import (
    EDGES_TABLE,
    MemoryStore,
)
from iai_mcp.types import MemoryRecord


EMBED_DIM = 4
REFERENCE_EMBEDDING: list[float] = [1.0, 0.0, 0.0, 0.0]
LINK_INITIAL_WEIGHT_DEFAULT: float = 0.10


def _make_embedding_at_cosine(
    cos_target: float, embed_dim: int = EMBED_DIM,
) -> list[float]:
    if not (-1.0 <= cos_target <= 1.0):
        raise ValueError(f"cos_target out of range: {cos_target}")
    residual = math.sqrt(max(0.0, 1.0 - cos_target * cos_target))
    return [cos_target, residual] + [0.0] * (embed_dim - 2)


def _make_record(
    *,
    embedding: list[float],
    tier: str = "episodic",
    literal_surface: str = "alice prefers tea over coffee",
    **overrides,
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    base = dict(
        id=uuid4(),
        tier=tier,
        literal_surface=literal_surface,
        aaak_index="",
        embedding=list(embedding),
        community_id=None,
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
    )
    base.update(overrides)
    return MemoryRecord(**base)


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(
        path=str(tmp_path / "iai-mcp"),
        user_id="alice",
        read_consistency_interval=timedelta(seconds=0),
    )


@pytest.fixture(autouse=True)
def _reset_patsep_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in (
        "IAI_MCP_PATSEP_NEAR_DUP_THRESHOLD",
        "IAI_MCP_PATSEP_LINK_THRESHOLD",
        "IAI_MCP_PATSEP_LINK_INITIAL_WEIGHT",
        "IAI_MCP_PATSEP_TOP_K",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(EMBED_DIM))
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL", raising=False)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai-mcp"))


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    import keyring as _keyring

    fake: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(
        _keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p)
    )
    monkeypatch.setattr(
        _keyring, "delete_password", lambda s, u: fake.pop((s, u), None)
    )


@pytest.fixture
def fresh_store(tmp_path: Path) -> MemoryStore:
    return _make_store(tmp_path)


def _self_loop_ids(store: MemoryStore) -> set[str]:
    edges_tbl = store.db.open_table(EDGES_TABLE)
    edges_df = edges_tbl.to_pandas()
    if len(edges_df) == 0:
        return set()
    mask = (
        (edges_df["edge_type"] == "hebbian")
        & (edges_df["src"] == edges_df["dst"])
    )
    return set(edges_df.loc[mask, "src"].astype(str).tolist())


def test_fresh_insert_writes_self_loop(fresh_store: MemoryStore) -> None:
    record_ids: set[str] = set()
    for i in range(5):
        emb = [0.0] * EMBED_DIM
        emb[i % EMBED_DIM] = 1.0
        if i >= EMBED_DIM:
            emb[i % EMBED_DIM] = 0.5
            emb[(i + 1) % EMBED_DIM] = math.sqrt(1 - 0.25)
        rec = _make_record(embedding=emb, literal_surface=f"record-{i}")
        record_ids.add(str(rec.id))
        fresh_store.insert(rec)

    self_loops = _self_loop_ids(fresh_store)
    assert self_loops == record_ids, (
        f"fresh-INSERT path must write a (rid, rid) hebbian self-loop "
        f"per record. records={record_ids}, self_loops={self_loops}, "
        f"missing={record_ids - self_loops}"
    )


def test_fresh_insert_self_loop_weight_matches_link_initial_weight(
    fresh_store: MemoryStore,
) -> None:
    rec = _make_record(embedding=REFERENCE_EMBEDDING)
    fresh_store.insert(rec)

    edges_tbl = fresh_store.db.open_table(EDGES_TABLE)
    edges_df = edges_tbl.to_pandas()
    self_loops = edges_df[
        (edges_df["edge_type"] == "hebbian")
        & (edges_df["src"] == str(rec.id))
        & (edges_df["dst"] == str(rec.id))
    ]
    assert len(self_loops) == 1, (
        f"exactly one self-loop expected after one fresh insert; "
        f"got {len(self_loops)}: {edges_df.to_dict('records')}"
    )
    weight = float(self_loops["weight"].iloc[0])
    assert abs(weight - LINK_INITIAL_WEIGHT_DEFAULT) < 1e-6, (
        f"fresh-INSERT self-loop weight must be link_initial_weight "
        f"(~{LINK_INITIAL_WEIGHT_DEFAULT}); got {weight}"
    )


def test_symmetry_post_migration_all_or_none(
    fresh_store: MemoryStore,
) -> None:
    from iai_mcp.maintenance import symmetrize_self_loops

    record_ids: list[str] = []
    for i in range(6):
        emb = [0.0] * EMBED_DIM
        emb[i % EMBED_DIM] = 1.0
        if i >= EMBED_DIM:
            emb[i % EMBED_DIM] = 0.5
            emb[(i + 1) % EMBED_DIM] = math.sqrt(1 - 0.25)
        rec = _make_record(embedding=emb, literal_surface=f"record-{i}")
        record_ids.append(str(rec.id))
        fresh_store.insert(rec)

    edges_tbl = fresh_store.db.open_table(EDGES_TABLE)
    missing_ids = record_ids[:4]
    for rid in missing_ids:
        edges_tbl.delete(
            f"edge_type == 'hebbian' AND src == '{rid}' AND dst == '{rid}'"
        )

    pre_self_loops = _self_loop_ids(fresh_store)
    assert len(pre_self_loops) == 2, (
        f"setup invariant: 6 records - 4 deleted = 2 self-loops; "
        f"got {len(pre_self_loops)}: {pre_self_loops}"
    )

    result = symmetrize_self_loops(fresh_store, dry_run=False)

    assert result["dry_run"] is False, result
    assert result["records_total"] == 6, result
    assert result["self_loops_present"] == 2, result
    assert result["self_loops_pending"] == 4, result
    assert result["self_loops_inserted"] == 4, result

    post_self_loops = _self_loop_ids(fresh_store)
    assert post_self_loops == set(record_ids), (
        f"post-migration symmetry violated. records={set(record_ids)}, "
        f"self_loops={post_self_loops}, missing="
        f"{set(record_ids) - post_self_loops}"
    )


def test_symmetry_dry_run_reports_without_writing(
    fresh_store: MemoryStore,
) -> None:
    from iai_mcp.maintenance import symmetrize_self_loops

    record_ids: list[str] = []
    for i in range(6):
        emb = [0.0] * EMBED_DIM
        emb[i % EMBED_DIM] = 1.0
        if i >= EMBED_DIM:
            emb[i % EMBED_DIM] = 0.5
            emb[(i + 1) % EMBED_DIM] = math.sqrt(1 - 0.25)
        rec = _make_record(embedding=emb, literal_surface=f"record-{i}")
        record_ids.append(str(rec.id))
        fresh_store.insert(rec)

    edges_tbl = fresh_store.db.open_table(EDGES_TABLE)
    for rid in record_ids[:4]:
        edges_tbl.delete(
            f"edge_type == 'hebbian' AND src == '{rid}' AND dst == '{rid}'"
        )

    pre_self_loops_count = len(_self_loop_ids(fresh_store))
    assert pre_self_loops_count == 2

    result = symmetrize_self_loops(fresh_store, dry_run=True)

    assert result["dry_run"] is True, result
    assert result["records_total"] == 6, result
    assert result["self_loops_present"] == 2, result
    assert result["self_loops_pending"] == 4, result
    assert result["self_loops_inserted"] == 0, result

    post_self_loops_count = len(_self_loop_ids(fresh_store))
    assert post_self_loops_count == pre_self_loops_count, (
        f"dry_run must NOT mutate the edges table. "
        f"pre={pre_self_loops_count}, post={post_self_loops_count}"
    )


def test_dedup_smoke_with_self_loop_guard(
    fresh_store: MemoryStore,
) -> None:
    rec_a = _make_record(embedding=REFERENCE_EMBEDDING)
    A_id = rec_a.id
    fresh_store.insert(rec_a)

    for _ in range(10):
        rec_dup = _make_record(embedding=REFERENCE_EMBEDDING)
        fresh_store.insert(rec_dup)

    edges_tbl = fresh_store.db.open_table(EDGES_TABLE)
    edges_df = edges_tbl.to_pandas()
    self_loops = edges_df[
        (edges_df["edge_type"] == "hebbian")
        & (edges_df["src"] == str(A_id))
        & (edges_df["dst"] == str(A_id))
    ]
    assert len(self_loops) >= 1, (
        f"dedup-touched record must have a self-loop; got "
        f"{edges_df.to_dict('records')}"
    )
    weight = float(self_loops["weight"].iloc[0])
    assert weight >= 0.9, (
        f"dedup-target self-loop weight floor violated. "
        f"expected >= 0.9 (typical ~1.1); got {weight}"
    )


def test_cli_subcommand_dry_run_smoke(
    fresh_store: MemoryStore,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    from iai_mcp.cli import cmd_maintenance_symmetrize_self_loops

    record_ids: list[str] = []
    for i in range(6):
        emb = [0.0] * EMBED_DIM
        emb[i % EMBED_DIM] = 1.0
        if i >= EMBED_DIM:
            emb[i % EMBED_DIM] = 0.5
            emb[(i + 1) % EMBED_DIM] = math.sqrt(1 - 0.25)
        rec = _make_record(embedding=emb, literal_surface=f"record-{i}")
        record_ids.append(str(rec.id))
        fresh_store.insert(rec)

    edges_tbl = fresh_store.db.open_table(EDGES_TABLE)
    for rid in record_ids[:4]:
        edges_tbl.delete(
            f"edge_type == 'hebbian' AND src == '{rid}' AND dst == '{rid}'"
        )

    store_root = str(tmp_path / "iai-mcp")
    args = argparse.Namespace(
        store_path=store_root,
        dry_run=True,
        apply=False,
        yes=False,
    )

    exit_code = cmd_maintenance_symmetrize_self_loops(args)
    assert exit_code == 0, f"dry-run should return 0; got {exit_code}"

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["dry_run"] is True, payload
    assert payload["records_total"] == 6, payload
    assert payload["self_loops_present"] == 2, payload
    assert payload["self_loops_pending"] == 4, payload
    assert payload["self_loops_inserted"] == 0, payload

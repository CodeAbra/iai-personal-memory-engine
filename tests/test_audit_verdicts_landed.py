from __future__ import annotations

import inspect
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

EMBED_DIM = 4
REFERENCE_EMBEDDING: list[float] = [1.0, 0.0, 0.0, 0.0]

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
    literal_surface: str = "alice prefers tea over coffee",
):
    from iai_mcp.types import MemoryRecord
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
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

@pytest.fixture(autouse=True)
def _reset_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in (
        "IAI_MCP_PATSEP_NEAR_DUP_THRESHOLD",
        "IAI_MCP_PATSEP_LINK_THRESHOLD",
        "IAI_MCP_PATSEP_LINK_INITIAL_WEIGHT",
        "IAI_MCP_PATSEP_TOP_K",
        "IAI_MCP_PATSEP_DRY_RUN",
        "IAI_MCP_ORTHO_ENABLED",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(EMBED_DIM))
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL", raising=False)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai-mcp"))

def _make_store(tmp_path: Path):
    from iai_mcp.store import MemoryStore
    return MemoryStore(
        path=str(tmp_path / "iai-mcp"),
        user_id="alice",
        read_consistency_interval=timedelta(seconds=0),
    )

def test_pask_conversation_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        import iai_mcp.pask_conversation  # noqa: F401

def test_organizational_closure_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        import iai_mcp.organizational_closure  # noqa: F401

def test_recursive_autopoiesis_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        import iai_mcp.recursive_autopoiesis  # noqa: F401

def test_pattern_separation_still_present_and_importable() -> None:
    from iai_mcp.pattern_separation import (  # noqa: F401
        OrthogonalizationResult,
        detect_hubness,
        orthogonalize_for_routing,
    )

def test_pattern_separation_gate_default_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iai_mcp import pattern_separation as ps

    monkeypatch.delenv("IAI_MCP_ORTHO_ENABLED", raising=False)
    store = _make_store(tmp_path)

    e0 = REFERENCE_EMBEDDING
    e1 = _make_embedding_at_cosine(0.5)
    store.insert(_make_record(embedding=e0))
    store.insert(_make_record(embedding=e1))

    def _no_call(*args, **kwargs):
        raise AssertionError(
            "orthogonalize_for_routing called with flag OFF"
        )
    monkeypatch.setattr(ps, "orthogonalize_for_routing", _no_call)

    rec = _make_record(embedding=_make_embedding_at_cosine(0.3))
    action, _payload = store.pattern_separation_gate(rec)
    assert action is not None

def test_pattern_separation_gate_flag_on_invokes_orthogonalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iai_mcp.store as _store_mod
    from iai_mcp import pattern_separation as ps

    monkeypatch.setenv("IAI_MCP_ORTHO_ENABLED", "1")
    store = _make_store(tmp_path)

    store.insert(_make_record(embedding=REFERENCE_EMBEDDING))
    store.insert(_make_record(embedding=_make_embedding_at_cosine(0.6)))

    calls: list[tuple[list[float], list[list[float]]]] = []
    original = ps.orthogonalize_for_routing

    def _spy(vec, neighbor_vecs, strength=0.3):
        calls.append((list(vec), [list(nv) for nv in neighbor_vecs]))
        return original(vec, neighbor_vecs, strength=strength)

    monkeypatch.setattr(
        "iai_mcp.pattern_separation.orthogonalize_for_routing", _spy,
    )
    assert _store_mod is not None

    rec = _make_record(embedding=_make_embedding_at_cosine(0.4))
    store.pattern_separation_gate(rec)

    assert len(calls) >= 1, (
        "orthogonalize_for_routing was not called with flag ON"
    )
    captured_vec, captured_neighbors = calls[-1]
    assert captured_vec == list(rec.embedding)
    assert len(captured_neighbors) >= 1
    for nv in captured_neighbors:
        assert isinstance(nv, list)
        assert len(nv) == EMBED_DIM

def test_pattern_separation_gate_preserves_embedding_invariant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_ORTHO_ENABLED", "1")
    store = _make_store(tmp_path)

    store.insert(_make_record(embedding=REFERENCE_EMBEDDING))
    store.insert(_make_record(embedding=_make_embedding_at_cosine(0.7)))

    rec = _make_record(embedding=_make_embedding_at_cosine(0.5))
    pre_snapshot = list(rec.embedding)
    store.pattern_separation_gate(rec)
    post_snapshot = list(rec.embedding)

    assert pre_snapshot == post_snapshot, (
        "R4 EMBEDDING INVARIANT violated: gate mutated record.embedding "
        f"pre={pre_snapshot[:3]}... post={post_snapshot[:3]}..."
    )

def test_pattern_separation_gate_static_no_record_embedding_assign() -> None:
    from iai_mcp.store import MemoryStore
    src = inspect.getsource(
        MemoryStore._pattern_separation_gate_with_hits,
    )
    assert "record.embedding =" not in src, (
        "R4 EMBEDDING INVARIANT static-check failure: literal "
        "`record.embedding =` assignment found inside "
        "_pattern_separation_gate_with_hits source."
    )

def test_isolate_keyring_removed_from_memory_footprint() -> None:
    import bench.memory_footprint as mf

    assert not hasattr(mf, "_isolate_keyring_in_memory"), (
        "DEF-25-E regression: `_isolate_keyring_in_memory` still defined "
        "in bench.memory_footprint"
    )
    sig = inspect.signature(mf.run_memory_footprint)
    assert "isolate_keyring" not in sig.parameters, (
        "DEF-25-E regression: `isolate_keyring` parameter still on "
        "run_memory_footprint signature"
    )

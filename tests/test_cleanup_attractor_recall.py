"""SC-2 exercise for the cleanup-attractor with rejection threshold (plan 04,
Task 2).

``_cleanup_if_confident`` (``lilli/ops/cleanup.py``) wraps the existing
``cleanup()`` primitive with a Hamming-distance rejection threshold
(``max_hamming_frac=0.15``): snap a noisy structural HV to its nearest
codebook entry only when the match is close enough, otherwise return the
noisy HV unchanged (HIPPEA: an ambiguous match is a prediction error, not
something to smooth over — research Pattern 3).

``_recall_core`` calls it, bounded to a <=200-entry shortlist derived from
``reachable_indices``, inside the structural-similarity block guarded by
``structural_weight > 0``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
import pytest

_DIM = 16


def _make_record(text="x", **overrides):
    from iai_mcp.types import MemoryRecord

    base = dict(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * _DIM,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    )
    base.update(overrides)
    return MemoryRecord(**base)


def _flip_bits(hv: bytes, frac: float, seed: int = 0) -> bytes:
    """Return hv with `frac` of its bits flipped (deterministic, seeded)."""
    arr = np.frombuffer(hv, dtype=np.uint8).copy()
    unpacked = np.unpackbits(arr)
    n_bits = len(unpacked)
    n_flip = int(n_bits * frac)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_bits, size=n_flip, replace=False)
    unpacked[idx] ^= 1
    return np.packbits(unpacked).tobytes()


@pytest.fixture(autouse=True)
def _small_embed_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(_DIM))


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(_keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p))
    monkeypatch.setattr(_keyring, "delete_password", lambda s, u: fake.pop((s, u), None))
    yield fake


@pytest.fixture(autouse=True)
def _crypto_passphrase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase-not-secret")


def _select_driver(driver: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _build_scene(tmp_path, driver, monkeypatch, *, target_structure_hv: bytes, other_structure_hv: bytes):
    """Build a minimal store + graph + assignment with a target + one other
    record, both carrying a structure_hv, wired for a direct
    recall_for_response() call (bypasses dispatch's global profile_state so
    structural_weight can be set explicitly per the plan's Task-2 wiring)."""
    from iai_mcp.community import CommunityAssignment
    from iai_mcp.graph import MemoryGraph
    from iai_mcp.store import MemoryStore

    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path / f"store-{driver}")

    target = _make_record(text="the structural target", structure_hv=target_structure_hv)
    other = _make_record(text="an unrelated record", structure_hv=other_structure_hv)
    store.insert(target)
    store.insert(other)

    graph = MemoryGraph()
    for r in (target, other):
        graph.add_node(r.id, community_id=r.id, embedding=r.embedding)
    assignment = CommunityAssignment(
        node_to_community={r.id: r.id for r in (target, other)},
        community_centroids={r.id: list(r.embedding) for r in (target, other)},
        modularity=0.5,
        top_communities=[target.id, other.id],
        mid_regions={target.id: [target.id], other.id: [other.id]},
    )
    return store, graph, assignment, target, other


class _StaticEmbedder:
    DIM = _DIM
    DEFAULT_DIM = _DIM
    DEFAULT_MODEL_KEY = "test"

    def embed(self, _text: str) -> list[float]:
        return [0.1] * self.DIM


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_close_noisy_hv_is_snapped(tmp_path, driver, monkeypatch):
    """When structural_weight > 0 and the target's structure_hv differs from
    the nearest codebook entry by LESS than 15% of bit-width, the cleanup
    snap is applied (verified via a spy on cleanup() itself)."""
    from iai_mcp import tem
    from iai_mcp.pipeline import recall_for_response
    import iai_mcp.lilli.ops.cleanup as _cleanup_mod

    codebook_hv = tem.pack_pairs([("TOPIC", tem.filler_hv("alpha"))])
    noisy_close = _flip_bits(codebook_hv, frac=0.10, seed=1)

    store, graph, assignment, target, other = _build_scene(
        tmp_path, driver, monkeypatch,
        target_structure_hv=noisy_close, other_structure_hv=codebook_hv,
    )

    spy = MagicMock(wraps=_cleanup_mod.cleanup)
    monkeypatch.setattr(_cleanup_mod, "cleanup", spy)

    resp = recall_for_response(
        store=store, graph=graph, assignment=assignment, rich_club=[],
        embedder=_StaticEmbedder(), cue="alpha", session_id="-",
        budget_tokens=5000, profile_state={"structural_weight": 0.9},
    )

    assert len(resp.hits) >= 1
    assert spy.call_count >= 1, (
        "cleanup() was never called with structural_weight > 0 and a close noisy HV"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_far_noisy_hv_is_rejected(tmp_path, driver, monkeypatch):
    """When the difference exceeds 15% of bit-width, _cleanup_if_confident
    must return the ORIGINAL noisy HV unchanged (rejection path)."""
    from iai_mcp import tem
    from iai_mcp.lilli.ops.cleanup import _cleanup_if_confident
    from iai_mcp.lilli.core import similarity as sim

    codebook_hv = tem.pack_pairs([("TOPIC", tem.filler_hv("alpha"))])
    noisy_far = _flip_bits(codebook_hv, frac=0.30, seed=2)

    result = _cleanup_if_confident(noisy_far, [codebook_hv], max_hamming_frac=0.15)

    assert result == noisy_far, "far noisy HV was snapped despite exceeding the rejection threshold"
    assert sim.hamming(noisy_far, codebook_hv) > 0.15


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_structural_weight_zero_never_calls_cleanup(tmp_path, driver, monkeypatch):
    """When structural_weight == 0 (the default for most callers), cleanup()
    must NEVER be called — branch pruning verified."""
    from iai_mcp import tem
    from iai_mcp.pipeline import recall_for_response
    import iai_mcp.lilli.ops.cleanup as _cleanup_mod

    codebook_hv = tem.pack_pairs([("TOPIC", tem.filler_hv("alpha"))])
    noisy_close = _flip_bits(codebook_hv, frac=0.10, seed=3)

    store, graph, assignment, target, other = _build_scene(
        tmp_path, driver, monkeypatch,
        target_structure_hv=noisy_close, other_structure_hv=codebook_hv,
    )

    spy = MagicMock(wraps=_cleanup_mod.cleanup)
    monkeypatch.setattr(_cleanup_mod, "cleanup", spy)

    resp = recall_for_response(
        store=store, graph=graph, assignment=assignment, rich_club=[],
        embedder=_StaticEmbedder(), cue="alpha", session_id="-",
        budget_tokens=5000, profile_state={"structural_weight": 0.0},
    )

    assert len(resp.hits) >= 1
    assert spy.call_count == 0, (
        "cleanup() was called even though structural_weight == 0"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_shortlist_codebook_bounded_to_200(tmp_path, driver, monkeypatch):
    """The codebook passed into _cleanup_if_confident must never exceed 200
    entries, regardless of corpus size."""
    from iai_mcp import tem
    from iai_mcp.community import CommunityAssignment
    from iai_mcp.graph import MemoryGraph
    from iai_mcp.pipeline import recall_for_response
    from iai_mcp.store import MemoryStore
    import iai_mcp.lilli.ops.cleanup as _cleanup_mod

    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path / f"store-{driver}")

    codebook_hv = tem.pack_pairs([("TOPIC", tem.filler_hv("alpha"))])
    recs = []
    for i in range(230):
        rec = _make_record(text=f"row-{i}", structure_hv=codebook_hv)
        store.insert(rec)
        recs.append(rec)

    graph = MemoryGraph()
    for r in recs:
        graph.add_node(r.id, community_id=r.id, embedding=r.embedding)
    assignment = CommunityAssignment(
        node_to_community={r.id: r.id for r in recs},
        community_centroids={r.id: list(r.embedding) for r in recs},
        modularity=0.5,
        top_communities=[r.id for r in recs],
        mid_regions={r.id: [r.id] for r in recs},
    )

    seen_codebook_sizes: list[int] = []
    _orig_cleanup_if_confident = _cleanup_mod._cleanup_if_confident

    def _spy_cleanup_if_confident(noisy_hv, codebook, **kwargs):
        seen_codebook_sizes.append(len(codebook))
        return _orig_cleanup_if_confident(noisy_hv, codebook, **kwargs)

    monkeypatch.setattr(
        "iai_mcp.pipeline._cleanup_if_confident", _spy_cleanup_if_confident, raising=False,
    )
    # The call site imports _cleanup_if_confident locally inside _recall_core;
    # patch the source module so the local import picks up the spy.
    monkeypatch.setattr(_cleanup_mod, "_cleanup_if_confident", _spy_cleanup_if_confident)

    recall_for_response(
        store=store, graph=graph, assignment=assignment, rich_club=[],
        embedder=_StaticEmbedder(), cue="alpha", session_id="-",
        budget_tokens=5000, profile_state={"structural_weight": 0.9},
    )

    assert seen_codebook_sizes, "_cleanup_if_confident was never called"
    for size in seen_codebook_sizes:
        assert size <= 200, f"codebook size {size} exceeded the bounded-shortlist cap of 200"

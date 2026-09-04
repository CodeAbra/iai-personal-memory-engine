"""FHRR D=10000 mint-side encoder: format, determinism, fail-loud input
validation, projection byte-identity, and geometry sanity of the phase
quantization against real cosine similarity.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
from scipy.stats import spearmanr

from iai_mcp.lilli.core.projection import EMBED_DIM, P, P_SHA256_HASH
from iai_mcp.lilli.crossmodal.embed_to_hv import from_embedding_fhrr
from iai_mcp.lilli.tiers import fhrr

_HARDCODED_P_SHA256 = "df97cc72a960567da17edbba16107881340349bd47b69f9b58d3091d96eb4e4e"


def _unit_vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    return v / float(np.linalg.norm(v))


def test_output_is_10000_bytes_uint8() -> None:
    emb = _unit_vec(1).tolist()
    payload = from_embedding_fhrr(emb)
    assert isinstance(payload, bytes)
    assert len(payload) == 10000
    arr = np.frombuffer(payload, dtype=np.uint8)
    assert arr.shape == (10000,)


def test_deterministic_same_input_same_output() -> None:
    emb = _unit_vec(2).tolist()
    a = from_embedding_fhrr(emb)
    b = from_embedding_fhrr(emb)
    assert a == b


def test_wrong_shape_raises() -> None:
    with pytest.raises(ValueError, match="length-384"):
        from_embedding_fhrr([0.1] * 10)


def test_non_finite_raises() -> None:
    emb = _unit_vec(3).tolist()
    emb[0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        from_embedding_fhrr(emb)


def test_zero_norm_raises() -> None:
    with pytest.raises(ValueError, match="zero-norm"):
        from_embedding_fhrr([0.0] * EMBED_DIM)


def test_projection_byte_identity_fence() -> None:
    """Independent hardcoded literal, NOT a comparison against P_SHA256_HASH
    itself — the literal is the load-bearing half that catches a joint edit
    of P and the constant together."""
    actual = hashlib.sha256(P.tobytes()).hexdigest()
    assert actual == _HARDCODED_P_SHA256
    assert P_SHA256_HASH == _HARDCODED_P_SHA256


def test_geometry_sanity_tracks_cosine_similarity() -> None:
    """Interpolate between two random unit vectors at controlled cosines and
    confirm the FHRR similarity of their encoded payloads tracks the real
    cosine similarity via Spearman rank correlation (not strict monotonicity,
    which flakes on a stochastic phase estimate)."""
    rng = np.random.default_rng(42)
    v0 = rng.standard_normal(EMBED_DIM).astype(np.float32)
    v0 = v0 / float(np.linalg.norm(v0))
    v1 = rng.standard_normal(EMBED_DIM).astype(np.float32)
    v1 = v1 / float(np.linalg.norm(v1))

    cos_sims: list[float] = []
    fhrr_sims: list[float] = []
    for t in np.linspace(0.0, 1.0, 15):
        v = (1.0 - t) * v0 + t * v1
        norm = float(np.linalg.norm(v))
        if norm <= 0.0:
            continue
        v_unit = v / norm
        cos_sim = float(np.dot(v0, v_unit))
        payload_a = from_embedding_fhrr(v0.tolist())
        payload_b = from_embedding_fhrr(v_unit.tolist())
        fhrr_sim = fhrr.similarity(payload_a, payload_b)
        cos_sims.append(cos_sim)
        fhrr_sims.append(fhrr_sim)

    rho, _ = spearmanr(cos_sims, fhrr_sims)
    assert rho > 0.9, (
        f"FHRR similarity rank correlation with cosine similarity too weak: "
        f"rho={rho}, cos_sims={cos_sims}, fhrr_sims={fhrr_sims}"
    )


def test_real_embedder_output_has_finite_positive_norm(tmp_path) -> None:
    """Informational A1 check: real embedder_for_store outputs are finite and
    have a positive L2 norm, documenting (without depending on) the
    unit-norm-ish assumption the encoder's normalize contract absorbs."""
    from iai_mcp.embed import embedder_for_store
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    embedder = embedder_for_store(store)
    for text in ("alice project note one", "distinct topic area two", "third sample sentence"):
        emb = embedder.embed(text)
        arr = np.asarray(emb, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        assert np.all(np.isfinite(arr))
        assert norm > 0.0


# --- shared mint site wiring: round-trip, fail-closed codec, fail-loud ---

_MEMBER_COUNT = 5


def _select_driver(driver: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _distinct_embedding(i: int) -> list:
    vec = [0.1] * EMBED_DIM
    span = EMBED_DIM // (_MEMBER_COUNT + 2)
    start = i * span
    for j in range(start, start + span):
        vec[j] = 0.9
    return vec


def _seed_member(store, i: int):
    from datetime import datetime, timezone
    from uuid import uuid4

    from iai_mcp.types import MemoryRecord

    now = datetime.now(timezone.utc)
    rec = MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=f"alice project note {i}: distinguishing detail about topic area {i}",
        aaak_index="",
        embedding=_distinct_embedding(i),
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=True,
        never_merge=True,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )
    store.insert(rec)
    return rec


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_mint_site_lands_bsc_default_hv_tier(driver, tmp_path, monkeypatch) -> None:
    _select_driver(driver, monkeypatch)
    from iai_mcp.sleep import _create_semantic_summary
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    members = [_seed_member(store, i) for i in range(_MEMBER_COUNT)]
    summary_text = (
        f"Cluster summary ({len(members)} records, lang=en): "
        + "; ".join(m.literal_surface[:80] for m in members)
    )
    summary_id, folded = _create_semantic_summary(store, members, summary_text, "en")
    assert not folded, "the summary dedup-folded into an existing survivor"

    batch = store.get_batch([summary_id])
    stored = batch.get(summary_id)
    assert stored is not None
    assert stored.hv_tier == "bsc"
    assert stored.structure_hv_payload == b""


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_fail_closed_codec_bogus_hv_tier_emits_telemetry(driver, tmp_path, monkeypatch) -> None:
    _select_driver(driver, monkeypatch)
    from iai_mcp import events
    from iai_mcp.store import MemoryStore, flush_record_buffer
    from iai_mcp.types import MemoryRecord
    from datetime import datetime, timezone
    from uuid import uuid4

    store = MemoryStore(path=tmp_path)
    now = datetime.now(timezone.utc)
    valid_payload = from_embedding_fhrr(_unit_vec(9).tolist())
    rec_id = uuid4()
    rec = MemoryRecord(
        id=rec_id,
        tier="semantic",
        literal_surface="valid fhrr-tier record before row-level corruption",
        aaak_index="",
        embedding=_unit_vec(9).tolist(),
        community_id=None,
        centrality=0.0,
        detail_level=3,
        pinned=False,
        stability=0.5,
        difficulty=0.3,
        last_reviewed=None,
        never_decay=True,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
        hv_tier="fhrr",
        structure_hv_payload=valid_payload,
    )
    store.insert(rec)
    flush_record_buffer(store)

    # Row-level UPDATE bypasses MemoryRecord.__post_init__'s HV_TIER_ENUM
    # validation — the DDL column is TEXT NOT NULL DEFAULT 'bsc' with no CHECK,
    # so a bogus enum string is schema-legal at the row level.
    with store.db._conn_lock:
        store.db._conn.execute(
            "UPDATE records SET hv_tier = ? WHERE id = ?",
            ("bogus_enum_value", str(rec_id)),
        )

    batch = store.get_batch([rec_id])
    degraded = batch.get(rec_id)
    assert degraded is not None
    assert degraded.hv_tier == "bsc"
    assert degraded.structure_hv_payload == b""

    emitted = events.query_events(store, kind=events.TELEMETRY_CODEC_MARKER_MISSING, limit=20)
    matching = [e for e in emitted if e["data"].get("record_id") == str(rec_id)]
    assert matching, (
        f"expected a {events.TELEMETRY_CODEC_MARKER_MISSING} event referencing "
        f"record {rec_id}; got {emitted}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_fail_loud_at_mint_boundary_on_malformed_embedding(driver, tmp_path, monkeypatch) -> None:
    _select_driver(driver, monkeypatch)
    import iai_mcp.embed as embed_mod
    from iai_mcp.sleep import _process_cluster_summaries
    from iai_mcp.store import MemoryStore, flush_edge_buffer

    store = MemoryStore(path=tmp_path)
    members = [_seed_member(store, i) for i in range(_MEMBER_COUNT)]
    pairs = [
        (members[i].id, members[(i + 1) % len(members)].id)
        for i in range(len(members))
    ]
    store.boost_edges(pairs, delta=0.3, edge_type="hebbian")
    flush_edge_buffer(store)

    class _BadEmbedder:
        def embed(self, text: str) -> list:
            return [0.1] * (EMBED_DIM - 1)  # wrong length -> store.insert raises

    monkeypatch.setattr(
        embed_mod, "embedder_for_store", lambda store, **kwargs: _BadEmbedder()
    )

    with pytest.raises(ValueError):
        _process_cluster_summaries(store)

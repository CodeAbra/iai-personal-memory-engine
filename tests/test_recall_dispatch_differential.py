"""Dispatch-level byte differential: the byte-parity gate never enters
`core.dispatch` (it calls `recall_for_response` directly), so this is the
only coverage that proves a resident-cache hit reproduces a fresh fetch's
served score through the real dispatch path.

The target is reachable ONLY through a hop1 edge (a small `K_CANDIDATES`
keeps it outside the ANN top-k) so every recall in this file actually
resolves it through `_resident_split`/`_resident_hit_views` -- a target
always found via ANN would never exercise the code this gate exists to
cover (see `test_rank_index_live_path.py`'s identical hop-reachable shape).

`IAI_MCP_W_DEGREE=0` neutralizes the degree term: this store never builds
the corpus-wide warm bundle before the first recall (`_incident_edges_warm`
degrades to a `store.incident_edges` scan), so the FIRST recall computes
the hop target's degree from that cold scan while later recalls read a
freshly-cached warm-graph degree -- a real but pre-existing input
difference upstream of `_rank_index.py`/`core.py`'s hydrate block, not a
residency-path regression, and it would otherwise perturb the very score
this gate compares for a reason unrelated to what it tests. The remaining
score delta across recalls is float32 ULP-level jitter (compared with
`pytest.approx`), not something byte-parity's frozen-graph capture ever
has to tolerate -- this gate exists precisely because `core.dispatch`'s
live hydrate path does.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

import iai_mcp.pipeline as _pipeline_mod
from iai_mcp import retrieve
from iai_mcp.store import MemoryStore, flush_edge_buffer, flush_record_buffer
from iai_mcp.types import MemoryRecord
from tests._helpers import stub_embedder_for_store

_DIM = 16  # small synthetic dim; avoids loading the Rust embedder
_FALLBACK_MARKER = "rank_index_snapshot_failed"


@pytest.fixture(autouse=True)
def _small_embed_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(_DIM))


@pytest.fixture(autouse=True)
def _crypto_passphrase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase-not-secret")


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(_keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p))
    monkeypatch.setattr(_keyring, "delete_password", lambda s, u: fake.pop((s, u), None))
    yield fake


@pytest.fixture(autouse=True)
def _isolate_recall_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_EXACT_AUTHORITY_OFF", "1")
    monkeypatch.setenv("IAI_MCP_W_DEGREE", "0")
    monkeypatch.setattr(_pipeline_mod, "K_CANDIDATES", 3)


def _select_driver(driver: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built -- lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


class _StubEmbedder:
    """Deterministic stand-in embedder -- a fixed cue vector regardless of text."""

    def __init__(self, vec: list[float]) -> None:
        self._vec = vec

    def embed(self, _text: str) -> list[float]:
        return list(self._vec)


def _seeded_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _orthogonal_vec(base: list[float], seed: int) -> list[float]:
    """A unit vector with near-zero cosine to `base` -- far enough from the
    cue that a small ANN k never surfaces it, so it is reachable ONLY via a
    graph edge."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(_DIM).astype(np.float32)
    b = np.asarray(base, dtype=np.float32)
    noise -= float(np.dot(noise, b)) * b
    norm = float(np.linalg.norm(noise))
    if norm < 1e-6:
        noise = np.roll(b, 1) - b
        norm = float(np.linalg.norm(noise)) or 1.0
    return (noise / norm).tolist()


def _make_rec(
    rid: UUID, surface: str, embedding: list[float],
    valence: float = 0.0, salience_level: str = "unflagged",
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=rid,
        tier="episodic",
        literal_surface=surface,
        aaak_index="",
        embedding=embedding,
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
        created_at=now,
        updated_at=now,
        tags=["t"],
        language="en",
        valence=valence,
        salience_level=salience_level,
    )


def _seed_hop_reachable_corpus(store: MemoryStore, cue_vec: list[float]) -> MemoryRecord:
    """A seed near the cue plus two near-cue fillers (so ANN k=3 fills on
    cue-adjacent records only) and a hop target, reachable ONLY through a
    hebbian edge from the seed, carrying the elevated valence/salience_level
    this gate proves survive the resident-cache round trip."""
    seed = _make_rec(uuid4(), "seed near the cue", cue_vec)
    near_1 = _make_rec(uuid4(), "near-cue filler one", cue_vec)
    near_2 = _make_rec(uuid4(), "near-cue filler two", cue_vec)
    hop_target = _make_rec(
        uuid4(), "hop target reachable only by edge", _orthogonal_vec(cue_vec, 20),
        valence=0.8, salience_level="critical",
    )
    store.insert(seed)
    store.insert(near_1)
    store.insert(near_2)
    store.insert(hop_target)
    flush_record_buffer(store)
    store.boost_edges([(seed.id, hop_target.id)], edge_type="hebbian", delta=[5.0])
    flush_edge_buffer(store)
    return hop_target


def _dispatch_recall(store: MemoryStore, cue_vec: list[float], session_id: str) -> dict:
    from iai_mcp import core as _core

    store._build_exact_index_sync()
    _pipeline_mod._last_recall_latency_ms = 0.0
    return _core.dispatch(store, "memory_recall", {
        "cue": "dispatch differential probe",
        "session_id": session_id,
        "budget_tokens": 2000,
        "cue_embedding": cue_vec,
    })


def _target_hit(resp: dict, target_id: UUID) -> "tuple | None":
    """The one hit this gate compares: the resident-candidate target's own
    (id, score, reason)."""
    for h in resp.get("hits", []):
        if h["record_id"] == str(target_id):
            return (h["record_id"], h["score"], h["reason"])
    return None


_DEG_NORM_RE = re.compile(r"deg_norm [0-9.]+\*")


def _normalize_reason(reason: str) -> str:
    """Strips the raw `deg_norm` value the served reason prints even when
    its weight is 0: with no warm bundle built before the first recall,
    `_incident_edges_warm` degrades to a cold `store.incident_edges` scan
    on that call only, so the hop target's computed degree (not its scoring
    weight) differs between the first, cold recall and every later,
    warm-bundle-backed one -- an input difference upstream of the resident
    cache this gate is not testing. `IAI_MCP_W_DEGREE=0` already zeroes its
    contribution to the score; this only keeps the volatile raw value from
    also breaking reason-string equality."""
    return _DEG_NORM_RE.sub("deg_norm N*", reason)


@pytest.fixture
def _store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def _make(driver: str) -> MemoryStore:
        _select_driver(driver, monkeypatch)
        return MemoryStore(path=tmp_path / f"store-{driver}")
    return _make


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_hop_target_reproduces_across_dispatches_rust_path(
    driver, _store, monkeypatch, caplog,
):
    """Three `core.dispatch` recalls of the SAME cue: recall 1 fetches the
    hop target fresh (a resident-cache MISS), an in-place write (unchanged
    content) evicts it, recall 2 re-fetches it fresh again (another MISS),
    recall 3 finally serves it from the resident cache (a HIT). The served
    hit must reproduce across all three, and the Rust scorer must stay
    engaged (no fallback marker)."""
    store = _store(driver)
    cue_vec = _seeded_vec(11)
    target = _seed_hop_reachable_corpus(store, cue_vec)
    stub_embedder_for_store(monkeypatch, _StubEmbedder(cue_vec))
    retrieve.build_runtime_graph(store)  # registers the write-time sync hook

    caplog.set_level(logging.DEBUG, logger="iai_mcp.core")

    from iai_mcp.store._rank_index import rank_index_for
    handle = rank_index_for(store)

    caplog.clear()
    resp1 = _dispatch_recall(store, cue_vec, session_id="dispatch-diff-rust")
    assert handle.resident.get(target.id) is not None, (
        "fixture self-check: the hop target must be resident after recall 1"
    )
    fallback1 = [r for r in caplog.records if _FALLBACK_MARKER in r.getMessage()]
    assert not fallback1, f"recall 1 unexpectedly tripped the fallback: {[r.getMessage() for r in fallback1]}"

    store.update(target)  # unchanged content -- evicts the resident entry
    assert handle.resident.get(target.id) is None, "fixture self-check: the write must evict the resident entry"

    caplog.clear()
    resp2 = _dispatch_recall(store, cue_vec, session_id="dispatch-diff-rust")
    assert handle.resident.get(target.id) is not None, (
        "fixture self-check: recall 2's miss must re-populate the resident entry"
    )
    fallback2 = [r for r in caplog.records if _FALLBACK_MARKER in r.getMessage()]
    assert not fallback2, f"recall 2 tripped the fallback: {[r.getMessage() for r in fallback2]}"

    caplog.clear()
    resp3 = _dispatch_recall(store, cue_vec, session_id="dispatch-diff-rust")
    fallback3 = [r for r in caplog.records if _FALLBACK_MARKER in r.getMessage()]
    assert not fallback3, f"recall 3 tripped the fallback: {[r.getMessage() for r in fallback3]}"

    hit1, hit2, hit3 = (
        _target_hit(resp1, target.id), _target_hit(resp2, target.id), _target_hit(resp3, target.id),
    )
    assert hit1 is not None, "fixture self-check: the hop target must be a served hit on recall 1"
    assert hit1[0] == hit2[0] == hit3[0]
    r1n, r2n, r3n = _normalize_reason(hit1[2]), _normalize_reason(hit2[2]), _normalize_reason(hit3[2])
    assert r1n == r2n == r3n, (
        f"the served reason must be identical (deg_norm value excluded) across all three recalls: "
        f"{hit1[2]!r} / {hit2[2]!r} / {hit3[2]!r}"
    )
    assert hit1[1] == pytest.approx(hit2[1], rel=1e-5) == pytest.approx(hit3[1], rel=1e-5), (
        f"the served score must reproduce across all three recalls: {hit1[1]} / {hit2[1]} / {hit3[1]}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_hop_target_reproduces_across_dispatches_python_fallback_path(
    driver, _store, monkeypatch,
):
    """Same three-dispatch shape, forced onto the Python-fallback scorer
    (`IAI_MCP_RECALL_RUST_SCORER_OFF=1`): the resident-cache HIT on recall
    3 must reproduce recall 1/2's fresh-fetch score and reason -- including
    the `xval` valence-multiplier term -- proving a resident hit on the
    fallback path carries its real valence and salience_level (not 0.0)."""
    monkeypatch.setenv("IAI_MCP_RECALL_RUST_SCORER_OFF", "1")
    store = _store(driver)
    cue_vec = _seeded_vec(12)
    target = _seed_hop_reachable_corpus(store, cue_vec)
    stub_embedder_for_store(monkeypatch, _StubEmbedder(cue_vec))
    retrieve.build_runtime_graph(store)  # registers the write-time sync hook

    from iai_mcp.store._rank_index import rank_index_for
    handle = rank_index_for(store)

    resp1 = _dispatch_recall(store, cue_vec, session_id="dispatch-diff-fallback")
    assert handle.resident.get(target.id) is not None

    store.update(target)  # unchanged content -- evicts the resident entry
    assert handle.resident.get(target.id) is None

    resp2 = _dispatch_recall(store, cue_vec, session_id="dispatch-diff-fallback")
    assert handle.resident.get(target.id) is not None
    resp3 = _dispatch_recall(store, cue_vec, session_id="dispatch-diff-fallback")

    hit1, hit2, hit3 = (
        _target_hit(resp1, target.id), _target_hit(resp2, target.id), _target_hit(resp3, target.id),
    )
    assert hit1 is not None, "fixture self-check: the hop target must be a served hit on recall 1"
    for hit, label in ((hit1, "recall 1"), (hit2, "recall 2"), (hit3, "recall 3")):
        assert "xval 1.80" in hit[2], (
            f"{label} must carry the real valence multiplier (valence=0.8) in its "
            f"served reason, not silently drop it: {hit[2]!r}"
        )
        assert "xsalience" in hit[2], f"{label} must carry the salience boost: {hit[2]!r}"

    r1n, r2n, r3n = _normalize_reason(hit1[2]), _normalize_reason(hit2[2]), _normalize_reason(hit3[2])
    assert r1n == r2n == r3n, (
        f"the served reason must be identical (deg_norm value excluded) across all three recalls: "
        f"{hit1[2]!r} / {hit2[2]!r} / {hit3[2]!r}"
    )
    assert hit1[1] == pytest.approx(hit2[1], rel=1e-5) == pytest.approx(hit3[1], rel=1e-5), (
        f"the served score must reproduce across all three recalls: {hit1[1]} / {hit2[1]} / {hit3[1]}"
    )

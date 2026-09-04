"""Rust-path recall-read wiring: the tuned W_COSINE reaches the FORCED-Rust
production scorer (`use_rust_scorer=True`), byte-identical to the frozen
pre-change baseline when untuned, with the differential-gate perturb hook
still taking precedence over a present tuned weight.

Baseline data below is a committed constant derived from a fixed synthetic
3-record corpus captured on a build from unmodified source before the
`effective_w_cosine` FFI threading edit -- not a runtime read of a log path,
so this test carries no dev-process path string.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid5

import pytest

from iai_mcp import core, retrieval_weight_cache
from iai_mcp.lilli.profile.retrieval_tuning import (
    PROD_W_COSINE_MAX,
    PROD_W_COSINE_MIN,
    save_retrieval_weights_state,
)
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord
from tests._helpers import stub_embedder_for_store

_DRIVER_PARAMS = [
    pytest.param("stdlib", id="stdlib"),
    pytest.param("lilli", id="lilli"),
]

_ID_NAMESPACE = uuid5(NAMESPACE_URL, "iai-mcp:275-05b-rust-baseline-fixture")

# Far-future timestamp: _age_penalty returns 0.0 whenever `now - created_at`
# is negative, so the age term stays deterministically 0.0 regardless of
# wall-clock drift between the baseline capture and any later test run.
_FIXED_CREATED_AT = datetime(2100, 1, 1, tzinfo=timezone.utc)

_REC_A_TEXT = "fixed corpus record alpha"
_REC_B_TEXT = "fixed corpus record beta"
_REC_C_TEXT = "fixed corpus record gamma close to alpha"
_CUE_TEXT = "byte identity fence probe cue"

# Captured on a build from UNMODIFIED source, before the effective_w_cosine
# threading edit -- the sole net against a rank-core regression.
_BASELINE_SCORE_HEX = {
    _REC_A_TEXT: "0x1.0cccccccccccdp+0",
    _REC_B_TEXT: "0x1.999999999999ap-5",
    _REC_C_TEXT: "0x1.e66667999999ap-1",
}


def _set_driver(monkeypatch: pytest.MonkeyPatch, driver: str) -> None:
    if driver == "stdlib":
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    else:
        pytest.importorskip("iai_mcp_native")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", driver)


def _unit(i: int) -> list[float]:
    v = [0.0] * EMBED_DIM
    v[i] = 1.0
    return v


def _mix(i: int, j: int, cos_to_i: float) -> list[float]:
    v = [0.0] * EMBED_DIM
    v[i] = cos_to_i
    v[j] = math.sqrt(1.0 - cos_to_i * cos_to_i)
    return v


def _fixed_id(text: str):
    return uuid5(_ID_NAMESPACE, text)


class _MappedEmbedder:
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = dict(mapping)

    def embed(self, text: str) -> list[float]:
        vec = self._mapping.get(text)
        if vec is not None:
            return list(vec)
        fallback = [0.0] * EMBED_DIM
        fallback[EMBED_DIM - 1] = 1.0
        return fallback


def _rec(text: str, emb: list[float]) -> MemoryRecord:
    return MemoryRecord(
        id=_fixed_id(text), tier="episodic", literal_surface=text, aaak_index="",
        embedding=emb, community_id=None, centrality=0.0, detail_level=2,
        pinned=False, stability=0.0, difficulty=0.0, last_reviewed=None,
        never_decay=False, never_merge=False, provenance=[],
        created_at=_FIXED_CREATED_AT, updated_at=_FIXED_CREATED_AT, tags=[], language="en",
    )


def _corpus() -> dict[str, list[float]]:
    return {
        _REC_A_TEXT: _unit(0),
        _REC_B_TEXT: _unit(1),
        _REC_C_TEXT: _mix(0, 1, 0.9),
    }


def _setup_store(tmp_path, monkeypatch: pytest.MonkeyPatch, driver: str) -> MemoryStore:
    _set_driver(monkeypatch, driver)
    mapping = _corpus()
    stub_embedder_for_store(monkeypatch, _MappedEmbedder(mapping))
    store = MemoryStore(path=tmp_path)
    for text, vec in mapping.items():
        store.insert(_rec(text, vec))
    flush_record_buffer(store)
    return store


def _dispatch_forced_rust(monkeypatch: pytest.MonkeyPatch, store: MemoryStore) -> dict:
    # The production dispatch boundary (core/__init__.py:1244..) always
    # requests use_rust_scorer=True -- no env toggle needed here, unlike the
    # non-Rust wiring test in plan 05a. The exact-cosine authority merge is
    # an orthogonal recall feature (a separate exact-vector-index top-k that
    # can unconditionally override a hit's served score once its matrix is
    # warm) -- disabled here so this test isolates the rank-core FFI
    # threading this plan concerns, not that feature's own behavior.
    monkeypatch.setenv("IAI_MCP_EXACT_AUTHORITY_OFF", "1")
    resp = core.dispatch(store, "memory_recall", {
        "cue": _CUE_TEXT, "session_id": "retrieval-weight-rust-path-test",
        "budget_tokens": 10000, "cue_embedding": _unit(0),
    })
    assert "error" not in resp, f"recall dispatch errored: {resp.get('error')}"
    assert resp["hits"], "expected hits for the fixed synthetic corpus probe"
    return resp


def _hits_by_text(resp: dict) -> dict[str, dict]:
    by_id = {str(_fixed_id(text)): text for text in _corpus()}
    out: dict[str, dict] = {}
    for h in resp["hits"]:
        text = by_id.get(h["record_id"])
        if text is not None:
            out[text] = h
    return out


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_tuned_weight_moves_forced_rust_score(tmp_path, monkeypatch, driver) -> None:
    store = _setup_store(tmp_path, monkeypatch, driver)

    resp_untuned = _dispatch_forced_rust(monkeypatch, store)
    untuned_score = _hits_by_text(resp_untuned)[_REC_A_TEXT]["score"]

    tuned = PROD_W_COSINE_MIN
    save_retrieval_weights_state(store, {"W_COSINE": tuned})
    retrieval_weight_cache.invalidate(store)

    resp_tuned = _dispatch_forced_rust(monkeypatch, store)
    tuned_score = _hits_by_text(resp_tuned)[_REC_A_TEXT]["score"]

    assert tuned_score != untuned_score, (
        "a persisted clamp-edge W_COSINE must move the forced-Rust base_s -- "
        f"untuned={untuned_score!r} tuned={tuned_score!r}"
    )
    # rec_a has cosine=1.0, so moving W_COSINE from the untuned default (1.0)
    # to `tuned` must shift the served score by exactly (1.0 - tuned) * 1.0 --
    # every other additive term (aaak/degree/age/stability-lift) is identical
    # across both dispatches on the same unchanged corpus.
    assert tuned_score == pytest.approx(untuned_score - (1.0 - tuned), abs=1e-9), (
        f"forced-Rust score did not move by exactly the tuned weight delta: "
        f"untuned={untuned_score!r} tuned={tuned_score!r} expected_delta={1.0 - tuned!r}"
    )

    # Pins the _reason_w_cosine coefficient in the Rust-path reason string --
    # the observability surface plan 07 reads -- so a regression there is
    # caught here, not first noticed downstream.
    tuned_reason = _hits_by_text(resp_tuned)[_REC_A_TEXT]["reason"]
    assert f"cos 1.000*{tuned:g}" in tuned_reason, (
        f"Rust-path reason string does not carry the tuned coefficient: {tuned_reason!r}"
    )


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_byte_identity_fence_untuned_env_unset(tmp_path, monkeypatch, driver) -> None:
    monkeypatch.delenv("IAI_MCP_RANK_PERTURB_W_COSINE", raising=False)
    store = _setup_store(tmp_path, monkeypatch, driver)
    # No save_retrieval_weights_state call: the store carries no weight blob.

    resp = _dispatch_forced_rust(monkeypatch, store)
    by_text = _hits_by_text(resp)

    for text, expected_hex in _BASELINE_SCORE_HEX.items():
        got_hex = float(by_text[text]["score"]).hex()
        assert got_hex == expected_hex, (
            f"forced-Rust score bits for {text!r} drifted from the unmodified-source "
            f"baseline: expected {expected_hex}, got {got_hex}"
        )


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_perturb_env_wins_over_present_tuned_weight(tmp_path, monkeypatch, driver) -> None:
    store = _setup_store(tmp_path, monkeypatch, driver)
    save_retrieval_weights_state(store, {"W_COSINE": PROD_W_COSINE_MIN})
    retrieval_weight_cache.invalidate(store)

    perturb_value = PROD_W_COSINE_MAX
    assert perturb_value != PROD_W_COSINE_MIN
    monkeypatch.setenv("IAI_MCP_RANK_PERTURB_W_COSINE", str(perturb_value))

    resp = _dispatch_forced_rust(monkeypatch, store)
    score = _hits_by_text(resp)[_REC_A_TEXT]["score"]

    # rec_a's cosine is 1.0, and the 0.05 stability lift is unconditional --
    # perturb_value * 1.0 + 0.05 is the exact expected score iff the env
    # override reached fused_score in place of the tuned weight.
    assert score == pytest.approx(perturb_value + 0.05, abs=1e-9), (
        f"IAI_MCP_RANK_PERTURB_W_COSINE must win over a present tuned weight: "
        f"env={perturb_value} tuned={PROD_W_COSINE_MIN} got_score={score!r}"
    )

"""Pure aggregate-window bounded observe/apply for the retrieval rank
weight, plus its dedicated own-key encrypted persistence -- disjoint from
the sealed 10-knob profile_state blob.
"""
from __future__ import annotations

import pytest

from iai_mcp.lilli.profile.retrieval_tuning import (
    DEFAULT_W_COSINE,
    MAX_WEIGHT_DELTA,
    PROD_W_COSINE_MAX,
    PROD_W_COSINE_MIN,
    RETRIEVAL_MIN_SAMPLES,
    RETRIEVAL_WEIGHTS_BLOB_AAD,
    RETRIEVAL_WEIGHTS_META_KEY,
    apply_retrieval_weight,
    load_retrieval_weights_state,
    observe_retrieval_weight,
    save_retrieval_weights_state,
)


_DRIVER_PARAMS = [
    pytest.param("stdlib", id="stdlib"),
    pytest.param("lilli", id="lilli"),
]


def _set_driver(monkeypatch: pytest.MonkeyPatch, driver: str) -> None:
    if driver == "stdlib":
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    else:
        pytest.importorskip("iai_mcp_native")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", driver)


def _window(n: int, use_rate: float) -> list[dict]:
    """n synthetic session rows, each with 10 hit_ids and a `use_rate`
    fraction of them also reinforced."""
    rows = []
    for i in range(n):
        hit_ids = [f"h{i}-{j}" for j in range(10)]
        reinforced_count = round(use_rate * 10)
        rows.append({"hit_ids": hit_ids, "reinforced_ids": hit_ids[:reinforced_count]})
    return rows


# ---------------------------------------------------------------------------
# observe_retrieval_weight
# ---------------------------------------------------------------------------


def test_observe_empty_window_yields_zero_samples():
    observed_use_rate, n, signal = observe_retrieval_weight([])
    assert n == 0
    assert observed_use_rate == 0.0
    assert isinstance(signal, dict)


def test_observe_aggregates_to_one_mean_use_rate():
    rows = _window(30, 0.7)
    observed_use_rate, n, _signal = observe_retrieval_weight(rows)
    assert n == 30
    assert abs(observed_use_rate - 0.7) < 1e-6


def test_observe_signal_free_rows_do_not_count_toward_n():
    # A window padded with empty-hit_ids rows must not inflate n past the
    # min_samples gate on rows that carry no rate.
    rated = _window(5, 1.0)
    padding = [{"hit_ids": [], "reinforced_ids": []} for _ in range(20)]
    observed_use_rate, n, signal = observe_retrieval_weight(rated + padding)
    assert n == 5
    assert signal["rated_rows"] == 5
    assert abs(observed_use_rate - 1.0) < 1e-6


def test_apply_below_min_samples_after_signal_free_padding_does_not_move():
    rated = _window(5, 1.0)
    padding = [{"hit_ids": [], "reinforced_ids": []} for _ in range(20)]
    observed_use_rate, n, _signal = observe_retrieval_weight(rated + padding)
    current = DEFAULT_W_COSINE
    new_w = apply_retrieval_weight(current, observed_use_rate, n)
    assert new_w == current


# ---------------------------------------------------------------------------
# apply_retrieval_weight -- min_samples gate
# ---------------------------------------------------------------------------


def test_apply_below_min_samples_does_not_move():
    current = DEFAULT_W_COSINE
    new_w = apply_retrieval_weight(current, observed_use_rate=1.0, n=RETRIEVAL_MIN_SAMPLES - 1)
    assert new_w == current


# ---------------------------------------------------------------------------
# apply_retrieval_weight -- saturation resistance (aggregate, not per-event fold)
# ---------------------------------------------------------------------------


def test_apply_80_sample_all_positive_window_moves_at_most_one_step():
    current = DEFAULT_W_COSINE
    new_w = apply_retrieval_weight(current, observed_use_rate=1.0, n=80)
    assert abs(new_w - current) <= MAX_WEIGHT_DELTA + 1e-9
    assert new_w <= PROD_W_COSINE_MAX


def test_apply_never_exceeds_narrow_production_band():
    # Repeated all-positive nightly applications must never walk past the
    # narrow clamp, regardless of how many nights accumulate.
    w = DEFAULT_W_COSINE
    for _ in range(200):
        w = apply_retrieval_weight(w, observed_use_rate=1.0, n=80)
        assert PROD_W_COSINE_MIN <= w <= PROD_W_COSINE_MAX


def test_apply_never_below_narrow_production_band():
    w = DEFAULT_W_COSINE
    for _ in range(200):
        w = apply_retrieval_weight(w, observed_use_rate=0.0, n=80)
        assert PROD_W_COSINE_MIN <= w <= PROD_W_COSINE_MAX


# ---------------------------------------------------------------------------
# apply_retrieval_weight -- bounded per-night delta
# ---------------------------------------------------------------------------


def test_apply_single_call_bounded_by_max_weight_delta():
    current = DEFAULT_W_COSINE
    new_w = apply_retrieval_weight(current, observed_use_rate=1.0, n=RETRIEVAL_MIN_SAMPLES)
    assert abs(new_w - current) <= MAX_WEIGHT_DELTA + 1e-9


# ---------------------------------------------------------------------------
# apply_retrieval_weight -- fixed-point idempotence
# ---------------------------------------------------------------------------


def test_apply_at_fixed_point_use_rate_does_not_drift():
    current = DEFAULT_W_COSINE
    new_w = apply_retrieval_weight(current, observed_use_rate=0.5, n=RETRIEVAL_MIN_SAMPLES)
    assert new_w == pytest.approx(current)


# ---------------------------------------------------------------------------
# Dedicated own-key persistence, disjoint from the sealed profile_state blob
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_save_load_round_trip(tmp_path, monkeypatch, driver):
    from iai_mcp.store import MemoryStore

    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path)

    saved = save_retrieval_weights_state(store, {"W_COSINE": 1.3})
    assert saved is True

    loaded = load_retrieval_weights_state(store)
    assert loaded == {"W_COSINE": 1.3}


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_own_key_isolation_from_sealed_profile_knobs(tmp_path, monkeypatch, driver):
    from iai_mcp.lilli.profile.knobs import default_state
    from iai_mcp.lilli.profile.persistence import load_profile_state, save_profile_state
    from iai_mcp.store import MemoryStore

    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path)

    save_profile_state(store, knobs=default_state(), posterior={}, pins={})
    save_retrieval_weights_state(store, {"W_COSINE": 1.7})

    blob = load_profile_state(store)
    assert blob is not None
    assert "W_COSINE" not in blob["knobs"], (
        "the tuned retrieval weight must never appear in the sealed profile knobs dict"
    )

    weights = load_retrieval_weights_state(store)
    assert weights["W_COSINE"] == 1.7


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_absent_key_returns_defaults_without_raising(tmp_path, monkeypatch, driver):
    from iai_mcp.store import MemoryStore

    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path)

    loaded = load_retrieval_weights_state(store)
    assert loaded == {"W_COSINE": DEFAULT_W_COSINE}


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_undecryptable_prior_blob_preserved_as_orphan(tmp_path, monkeypatch, driver):
    from iai_mcp.store import MemoryStore

    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path)

    db = store.db
    corrupt_ciphertext = "iai:enc:v1:not-actually-valid-base64==="
    with db._conn_lock:
        db._conn.execute(
            "INSERT OR IGNORE INTO _hippo_meta (key, value) VALUES (?, ?)",
            (RETRIEVAL_WEIGHTS_META_KEY, corrupt_ciphertext),
        )
        db._conn.commit()

    # A subsequent legitimate save must not silently discard the corrupt prior blob.
    save_retrieval_weights_state(store, {"W_COSINE": 1.1})

    with db._conn_lock:
        orphan_row = db._conn.execute(
            "SELECT value FROM _hippo_meta WHERE key = ?",
            ("retrieval_weights_state.orphan",),
        ).fetchone()
    assert orphan_row is not None
    assert orphan_row["value"] == corrupt_ciphertext

    loaded = load_retrieval_weights_state(store)
    assert loaded == {"W_COSINE": 1.1}

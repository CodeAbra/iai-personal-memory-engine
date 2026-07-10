from __future__ import annotations

import json
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent

PARITY_JSON_PATH = (
    _REPO_ROOT
    / "bench"
    / "results"
    / "v7.1"
    / "iteration-0"
    / "contradiction_longitudinal_custom_leiden.json"
)

V7_0_BASELINE_PER_SEED: dict[str, float] = {
    "13": 1.0,
    "42": 1.0,
    "137": 1.0,
}

PARITY_TOLERANCE = 0.02

V7_0_CROSS_SEED_MEAN_BASELINE = (
    sum(V7_0_BASELINE_PER_SEED.values()) / len(V7_0_BASELINE_PER_SEED)
)


@pytest.fixture(scope="module")
def parity_data() -> dict:
    if not PARITY_JSON_PATH.exists():
        pytest.skip(
            f"Parity JSON not yet produced. Run "
            f"`PYTHONPATH=src python bench/contradiction_longitudinal_claude.py "
            f"--scale honest --seeds 13 42 137 "
            f"--output-dir bench/results/v7.1/iteration-0/` then the post-process "
            f"pipeline (analyze_efe_ab.py + write contradiction_longitudinal_"
            f"custom_leiden.json). Missing: {PARITY_JSON_PATH}"
        )
    return json.loads(PARITY_JSON_PATH.read_text())


@pytest.mark.slow
def test_parity_json_exists() -> None:
    assert PARITY_JSON_PATH.exists(), (
        f"Parity JSON missing: {PARITY_JSON_PATH}. "
        f"Bench must be run before parity enforcement is meaningful."
    )


def test_parity_json_schema_matches_baseline(parity_data: dict) -> None:
    required_keys = {
        "per_seed",
        "cross_seed_mean_rescue",
        "seeds",
        "n_recalls",
        "backend",
    }
    missing = required_keys - set(parity_data.keys())
    assert not missing, (
        f"Parity JSON missing required top-level keys: {sorted(missing)}. "
        f"Got: {sorted(parity_data.keys())}"
    )

    per_seed = parity_data["per_seed"]
    for seed_key in V7_0_BASELINE_PER_SEED:
        assert seed_key in per_seed, (
            f"per_seed missing seed {seed_key!r}. Got: {sorted(per_seed.keys())}"
        )
        seed_block = per_seed[seed_key]
        for required_field in ("efe_real_rescue", "efe_shadow_rescue", "delta"):
            assert required_field in seed_block, (
                f"per_seed[{seed_key!r}] missing {required_field!r}. "
                f"Got: {sorted(seed_block.keys())}"
            )


def test_backend_label_is_leiden_custom(parity_data: dict) -> None:
    backend = parity_data.get("backend")
    assert backend == "leiden-custom", (
        f"Parity bench must run the custom_leiden backend, got: {backend!r}. "
        f"Either community.detect_communities was not wired correctly "
        f"(parity regression) or the post-process injected the wrong "
        f"label."
    )


@pytest.mark.parametrize("seed_key", sorted(V7_0_BASELINE_PER_SEED.keys()))
def test_per_seed_rescue_within_002_of_baseline(
    parity_data: dict, seed_key: str
) -> None:
    baseline = V7_0_BASELINE_PER_SEED[seed_key]
    seed_block = parity_data["per_seed"][seed_key]
    measured = float(seed_block["efe_shadow_rescue"])
    delta = abs(measured - baseline)
    assert delta <= PARITY_TOLERANCE, (
        f"PARITY_GATE_FAILED seed={seed_key}: "
        f"efe_shadow_rescue={measured:.4f} vs v7.0 baseline={baseline:.4f} "
        f"(|delta|={delta:.4f} > tolerance={PARITY_TOLERANCE:.4f}). "
        f"GPL removal BLOCKED on parity gate green."
    )


def test_cross_seed_mean_rescue_within_002(parity_data: dict) -> None:
    measured = float(parity_data["cross_seed_mean_rescue"])
    delta = abs(measured - V7_0_CROSS_SEED_MEAN_BASELINE)
    assert delta <= PARITY_TOLERANCE, (
        f"PARITY_GATE_FAILED cross_seed: "
        f"mean Rescue@10={measured:.4f} vs v7.0 baseline mean="
        f"{V7_0_CROSS_SEED_MEAN_BASELINE:.4f} "
        f"(|delta|={delta:.4f} > tolerance={PARITY_TOLERANCE:.4f}). "
        f"GPL removal BLOCKED on parity gate green."
    )


def test_seeds_match_required_three(parity_data: dict) -> None:
    seeds_field = parity_data.get("seeds", [])
    seeds_str = {str(s) for s in seeds_field}
    expected = set(V7_0_BASELINE_PER_SEED.keys())
    assert seeds_str == expected, (
        f"Seed list mismatch: got {sorted(seeds_str)}, expected "
        f"{sorted(expected)}. Re-run bench with --seeds 13 42 137 for "
        f"fair v7.0 vs v7.1 comparison."
    )


def test_n_recalls_at_least_3000(parity_data: dict) -> None:
    n_recalls = int(parity_data.get("n_recalls", 0))
    assert n_recalls >= 3000, (
        f"Parity bench n_recalls={n_recalls} too small for honest-scale "
        f"comparison. v7.0 baseline was 3000 attributable recalls. "
        f"Re-run with --scale honest."
    )

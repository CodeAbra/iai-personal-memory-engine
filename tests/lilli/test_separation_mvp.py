from __future__ import annotations

import pytest

from iai_mcp.lilli.core.similarity import hamming
from iai_mcp.lilli.ops.separation import (
    SIMILARITY_THRESHOLD_DEFAULT,
    pattern_separate,
)
from iai_mcp.lilli.tiers import bsc

def _max_sim(hv: bytes, background: list[bytes]) -> float:
    if not background:
        return 0.0
    return max(1.0 - hamming(hv, b) for b in background)

def _distinct_hvs(n: int) -> list[bytes]:
    roles = list(bsc.BSC_ROLE_VOCABULARY[:n])
    return [bsc.role_hv(r) for r in roles]

def test_empty_background_returns_target_unchanged() -> None:
    target = bsc.role_hv("WHEN")
    result = pattern_separate(target, [])
    assert result == target, "Empty background must return target unchanged"

@pytest.mark.parametrize("n_background", [1, 5, 50])
def test_returns_bytes_of_same_length(n_background: int) -> None:
    background = [bsc.filler_hv(f"bg-{i}") for i in range(n_background)]
    target = bsc.role_hv("WHEN")
    result = pattern_separate(target, background)
    assert len(result) == len(target), (
        f"Output length {len(result)} != input length {len(target)} "
        f"for background size {n_background}"
    )

def test_deterministic_output() -> None:
    target = bsc.role_hv("PROJECT")
    background = _distinct_hvs(3)
    result_a = pattern_separate(target, background, similarity_threshold=0.5)
    result_b = pattern_separate(target, background, similarity_threshold=0.5)
    assert result_a == result_b, "pattern_separate must be deterministic"

def test_dissimilar_target_early_accepted() -> None:
    target = bsc.role_hv("WHEN")
    background = [bsc.role_hv("OBJECT")]

    sim_score = 1.0 - hamming(target, background[0])
    assert sim_score < SIMILARITY_THRESHOLD_DEFAULT, (
        f"Precondition failed: role HVs have sim={sim_score:.3f} >= threshold; "
        "test needs independent seeds"
    )

    result = pattern_separate(target, background)
    assert result == target, (
        "Dissimilar target should be returned unchanged (early accept on attempt 0)"
    )

def test_identical_target_to_background_perturbed() -> None:
    target = bsc.role_hv("WHEN")
    background = [target]
    result = pattern_separate(target, background, similarity_threshold=0.99)
    assert result != target, (
        "Target identical to background must be perturbed by at least one bit flip"
    )

def test_rejection_sampling_decorrelates() -> None:
    import numpy as np

    base = bsc.role_hv("WHEN")
    bits = np.unpackbits(np.frombuffer(base, dtype=np.uint8))
    bits[0] ^= 1
    bits[1] ^= 1
    target = np.packbits(bits).tobytes()

    background = [base]

    input_max_sim = _max_sim(target, background)
    assert input_max_sim > 0.95, f"Precondition: input sim={input_max_sim:.4f} must be > 0.95"

    result = pattern_separate(target, background)
    output_max_sim = _max_sim(result, background)

    assert output_max_sim < input_max_sim, (
        f"Rejection sampling must reduce max-similarity: "
        f"input={input_max_sim:.4f}, output={output_max_sim:.4f}"
    )

def test_bounded_retries_returns_best_on_exhaustion() -> None:
    target = bsc.role_hv("WHEN")
    background = [target] * 5

    result = pattern_separate(target, background, max_retries=1, similarity_threshold=0.999)
    assert isinstance(result, bytes), "Result must be bytes"
    assert len(result) == len(target), f"Result length {len(result)} != {len(target)}"

def test_progressive_flip_count_increases_separation() -> None:
    target = bsc.role_hv("WHEN")
    background = [target] * 3

    result_1 = pattern_separate(target, background, max_retries=1, similarity_threshold=0.999)
    result_3 = pattern_separate(target, background, max_retries=3, similarity_threshold=0.999)

    ham_1 = hamming(target, result_1)
    ham_3 = hamming(target, result_3)

    assert ham_3 >= ham_1, (
        f"More retries should explore larger perturbations: "
        f"ham(max_retries=1)={ham_1:.4f}, ham(max_retries=3)={ham_3:.4f}"
    )

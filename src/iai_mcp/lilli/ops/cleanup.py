from __future__ import annotations

from iai_mcp.lilli.core import similarity as sim


def cleanup(noisy_hv: bytes, codebook: list[bytes]) -> bytes:
    if not codebook:
        raise ValueError("codebook must not be empty")

    for idx, entry in enumerate(codebook):
        if len(entry) != len(noisy_hv):
            raise ValueError(
                f"codebook entry {idx} length {len(entry)} does not match "
                f"noisy_hv length {len(noisy_hv)}"
            )

    best_dist, best_entry = min(
        (sim.hamming(noisy_hv, entry), i)
        for i, entry in enumerate(codebook)
    )
    return codebook[best_entry]


def _cleanup_if_confident(
    noisy_hv: bytes,
    codebook: list[bytes],
    *,
    max_hamming_frac: float = 0.15,
) -> bytes:
    """Snap `noisy_hv` to its nearest codebook entry only within a rejection
    threshold (HIPPEA: an ambiguous match is a prediction error, not something
    to smooth over — research 184 Pattern 3). `max_hamming_frac=0.15` is the
    researched default; above it, the input is returned unchanged.
    """
    if not codebook:
        return noisy_hv
    cleaned = cleanup(noisy_hv, codebook)
    dist_frac = sim.hamming(noisy_hv, cleaned)
    if dist_frac > max_hamming_frac:
        return noisy_hv
    return cleaned

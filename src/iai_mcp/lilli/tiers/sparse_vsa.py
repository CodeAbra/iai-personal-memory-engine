from __future__ import annotations

import struct
from collections import Counter
from typing import Sequence

import numpy as np

from iai_mcp.lilli.core import hd_backend as backend
from iai_mcp.lilli.core.seed import seed_from_str


LILLI_SPARSE_DIM: int = 2048
"""Hypervector dimension.  All active-bit indices live in [0, LILLI_SPARSE_DIM)."""

SPARSE_K: int = 20
"""Number of active bits per HV (~1 % sparsity at D=2048)."""

SPARSE_ROLE_SEED_PREFIX: str = "lilli-sparse-role"
SPARSE_FILLER_SEED_PREFIX: str = "lilli-sparse-filler"
SPARSE_PAD_SEED_PREFIX: str = "sparse-pad"

TIER_INFO: dict = {
    "backend": "sparse_vsa",
    "D": 2048,
    "bytes_per_hv": 40,
    "use_case": "procedural",
}
"""Tier metadata dictionary (read-only; consumed by tier_info() dispatcher)."""

_SENTINEL: int = 0xFFFF
"""Padding sentinel value used when pack_indices receives fewer than SPARSE_K
indices.  Stripped on unpack."""


def random_indices(seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    indices = rng.choice(LILLI_SPARSE_DIM, size=SPARSE_K, replace=False)
    return sorted(int(x) for x in indices)


def role_hv(role: str) -> list[int]:
    seed = seed_from_str(SPARSE_ROLE_SEED_PREFIX, role)
    return random_indices(seed)


def filler_hv(value: str) -> list[int]:
    seed = seed_from_str(SPARSE_FILLER_SEED_PREFIX, value)
    return random_indices(seed)


def pack_indices(indices: Sequence[int]) -> bytes:
    idx_list = list(indices)[:SPARSE_K]
    while len(idx_list) < SPARSE_K:
        idx_list.append(_SENTINEL)
    return struct.pack(f"<{SPARSE_K}H", *idx_list)


def unpack_indices(packed: bytes) -> list[int]:
    values = struct.unpack(f"<{SPARSE_K}H", packed)
    return [v for v in values if v != _SENTINEL]


_SPLITMIX64_MASK: int = 0xFFFF_FFFF_FFFF_FFFF
_SPLITMIX64_GAMMA: int = 0x9E37_79B9_7F4A_7C15
_SPLITMIX64_MIX1: int = 0xBF58_476D_1CE4_E5B9
_SPLITMIX64_MIX2: int = 0x94D0_49BB_1331_11EB


def _pad_to_k(indices: list[int], seed_a: list[int], seed_b: list[int]) -> list[int]:
    existing = set(indices)
    if len(existing) >= SPARSE_K:
        return sorted(existing)[:SPARSE_K]

    # Backfill seed is derived from the sorted operand lists via the shared
    # SHA256 head, so the pad path is process-stable (independent of any
    # per-process hash randomisation) and reproducible across implementations.
    operand_repr = f"{sorted(seed_a)}|{sorted(seed_b)}"
    state = seed_from_str(SPARSE_PAD_SEED_PREFIX, operand_repr)

    while len(existing) < SPARSE_K:
        # SplitMix64 step over the seed stream proposes candidate indices.
        state = (state + _SPLITMIX64_GAMMA) & _SPLITMIX64_MASK
        z = state
        z = ((z ^ (z >> 30)) * _SPLITMIX64_MIX1) & _SPLITMIX64_MASK
        z = ((z ^ (z >> 27)) * _SPLITMIX64_MIX2) & _SPLITMIX64_MASK
        z ^= z >> 31
        existing.add(z % LILLI_SPARSE_DIM)
    return sorted(existing)[:SPARSE_K]


def bind(a: list[int], b: list[int]) -> list[int]:
    if backend.use_rust():
        return list(backend.native().sparse_bind(list(a), list(b)))
    merged = sorted(set(a) | set(b))
    if len(merged) >= SPARSE_K:
        return merged[:SPARSE_K]
    return _pad_to_k(merged, a, b)


def unbind(bound: list[int], key: list[int]) -> list[int]:
    if backend.use_rust():
        return list(backend.native().sparse_unbind(list(bound), list(key)))
    remainder = sorted(set(bound) - set(key))
    if len(remainder) >= SPARSE_K:
        return remainder[:SPARSE_K]
    return _pad_to_k(remainder, bound, key)


def bundle(hvs: list[list[int]]) -> list[int]:
    if backend.use_rust():
        return list(backend.native().sparse_bundle([list(hv) for hv in hvs]))
    if not hvs:
        return []
    counts: Counter = Counter()
    for hv in hvs:
        for idx in hv:
            counts[idx] += 1
    ranked = sorted(counts.keys(), key=lambda i: (-counts[i], i))
    return sorted(ranked[:SPARSE_K])


def permute(hv: list[int], shift: int) -> list[int]:
    if backend.use_rust():
        return list(backend.native().sparse_permute(list(hv), shift))
    return sorted((i + shift) % LILLI_SPARSE_DIM for i in hv)


def similarity(a: list[int], b: list[int]) -> float:
    if backend.use_rust():
        return float(backend.native().sparse_similarity(list(a), list(b)))
    sa, sb = set(a), set(b)
    union_size = len(sa | sb)
    if union_size == 0:
        return 0.0
    return len(sa & sb) / union_size

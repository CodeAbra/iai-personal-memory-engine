from __future__ import annotations

import math

import numpy as np

from iai_mcp.lilli.core import hd_backend as backend
from iai_mcp.lilli.core.seed import seed_from_str


LILLI_FHRR_DIM: int = 10000
"""Hypervector dimensionality for the FHRR semantic tier."""

FHRR_PHASE_BINS: int = 256
"""Number of phase quantisation bins (uint8 = 256 bins covering [0, 2pi))."""

FHRR_ROLE_SEED_PREFIX: str = "lilli-fhrr-role"
"""Namespace prefix used when seeding role hypervectors."""

FHRR_FILLER_SEED_PREFIX: str = "lilli-fhrr-filler"
"""Namespace prefix used when seeding filler (value) hypervectors."""

TIER_INFO: dict[str, object] = {
    "backend": "fhrr",
    "D": 10000,
    "bytes_per_hv": 10000,
    "use_case": "semantic",
}
"""Tier metadata dictionary consumed by lilli.tier_info() dispatcher."""

_TWO_PI: float = 2.0 * math.pi


def random_hv(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    phases = rng.integers(0, 256, size=LILLI_FHRR_DIM, dtype=np.uint8)
    return phases.tobytes()


def role_hv(role: str) -> bytes:
    seed = seed_from_str(FHRR_ROLE_SEED_PREFIX, role)
    return random_hv(seed)


def filler_hv(value: str) -> bytes:
    seed = seed_from_str(FHRR_FILLER_SEED_PREFIX, value)
    return random_hv(seed)


def bind(a: bytes, b: bytes) -> bytes:
    if len(a) != len(b):
        raise ValueError(
            f"bind: length mismatch -- len(a)={len(a)}, len(b)={len(b)}"
        )
    if backend.use_rust():
        return bytes(backend.native().fhrr_bind(a, b))
    aa = np.frombuffer(a, dtype=np.uint8)
    bb = np.frombuffer(b, dtype=np.uint8)
    result = (aa.astype(np.uint16) + bb.astype(np.uint16)) & 0xFF
    return result.astype(np.uint8).tobytes()


def unbind(bound: bytes, key: bytes) -> bytes:
    if len(bound) != len(key):
        raise ValueError(
            f"unbind: length mismatch -- len(bound)={len(bound)}, len(key)={len(key)}"
        )
    if backend.use_rust():
        return bytes(backend.native().fhrr_unbind(bound, key))
    bb = np.frombuffer(bound, dtype=np.uint8)
    kk = np.frombuffer(key, dtype=np.uint8)
    result = (bb.astype(np.int16) - kk.astype(np.int16)) & 0xFF
    return result.astype(np.uint8).tobytes()


def bundle(hvs: list[bytes]) -> bytes:
    if not hvs:
        return bytes(LILLI_FHRR_DIM)

    if len(hvs) == 1:
        return hvs[0]

    if backend.use_rust():
        return bytes(backend.native().fhrr_bundle(list(hvs)))

    mat = np.frombuffer(b"".join(hvs), dtype=np.uint8).reshape(len(hvs), -1)
    D = mat.shape[1]

    radians = (mat.astype(np.float64) / 256.0) * _TWO_PI

    c = np.cos(radians).sum(axis=0)
    s = np.sin(radians).sum(axis=0)
    mean_angle = np.arctan2(s, c)

    mean_angle = mean_angle % _TWO_PI

    quantised = (mean_angle / _TWO_PI * 256.0).astype(np.int32) & 0xFF
    return quantised.astype(np.uint8).tobytes()


def permute(hv: bytes, shift: int) -> bytes:
    if backend.use_rust():
        return bytes(backend.native().fhrr_permute(hv, shift))
    arr = np.frombuffer(hv, dtype=np.uint8)
    return np.roll(arr, shift).tobytes()


def similarity(a: bytes, b: bytes) -> float:
    if len(a) != len(b):
        return 0.0
    if backend.use_rust():
        return float(backend.native().fhrr_similarity(a, b))
    aa_rad = (np.frombuffer(a, dtype=np.uint8).astype(np.float64) / 256.0) * _TWO_PI
    bb_rad = (np.frombuffer(b, dtype=np.uint8).astype(np.float64) / 256.0) * _TWO_PI
    diff = aa_rad - bb_rad
    return float(np.cos(diff).mean())

from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

import numpy as np

from iai_mcp.lilli.core import hd_backend as backend
from iai_mcp.lilli.core.seed import hv_from_seed, seed_from_str
from iai_mcp.lilli.core.similarity import hamming

if TYPE_CHECKING:
    from iai_mcp.store import MemoryStore

log = logging.getLogger(__name__)


LILLI_BSC_DEFAULT_DIM: int = 4096
"""Default dimensionality for the EPISODIC tier. 4096 bits = 512 bytes per HV."""


BSC_ROLE_SEED_PREFIX: str = "tem-role-v1"
"""Seed prefix for role codebook vectors. Matches tem.py for byte-identical output at D=10000."""

BSC_FILLER_SEED_PREFIX: str = "tem-filler-v1"
"""Seed prefix for filler hypervectors. Matches tem.py for byte-identical output at D=10000."""


BSC_ROLE_VOCABULARY: tuple[str, ...] = (
    "WHEN",
    "WHERE",
    "ROLE",
    "PROJECT",
    "COMMUNITY_ID",
    "TEMPORAL_POSITION",
    "ACTOR",
    "OBJECT",
    "INTENT",
    "MODALITY",
    "LANG",
    "SESSION_ID",
    "TIER",
    "VALENCE",
    "CERTAINTY",
    "SOURCE",
    "TOPIC",
    "PARENT_ID",
)


BSC_CAPACITY_DIVISOR: int = 400
"""Divisor for computing the max bundle pair count per dimension D."""

BSC_SATURATION_WARN_RATIO: float = 0.8
"""Fraction of max bundle pairs at which to emit a saturation warning event."""

_TELEMETRY_ROLE_SATURATION_KIND: str = "role_saturation_warning"


def _max_bundle_pairs(D: int) -> int:
    return max(1, D // BSC_CAPACITY_DIVISOR)


BSC_MAX_BUNDLE_PAIRS: int = _max_bundle_pairs(LILLI_BSC_DEFAULT_DIM)
"""Default-D hard cap: _max_bundle_pairs(4096) == 10."""


TIER_INFO: dict = {
    "backend": "bsc",
    "D": LILLI_BSC_DEFAULT_DIM,
    "bytes_per_hv": LILLI_BSC_DEFAULT_DIM // 8,
    "use_case": "episodic",
    "max_bundle_pairs": BSC_MAX_BUNDLE_PAIRS,
}


@lru_cache(maxsize=256)
def role_hv(role: str, *, D: int = LILLI_BSC_DEFAULT_DIM) -> bytes:
    seed = seed_from_str(BSC_ROLE_SEED_PREFIX, role)
    return hv_from_seed(seed, D)


@lru_cache(maxsize=256)
def filler_hv(value: str, *, D: int = LILLI_BSC_DEFAULT_DIM) -> bytes:
    seed = seed_from_str(BSC_FILLER_SEED_PREFIX, value)
    return hv_from_seed(seed, D)


def bind(a: bytes, b: bytes) -> bytes:
    if len(a) != len(b):
        raise ValueError(
            f"bind requires equal-length hypervectors, got {len(a)} and {len(b)}"
        )
    if backend.use_rust():
        return bytes(backend.native().bsc_bind(a, b))
    aa = np.frombuffer(a, dtype=np.uint8)
    bb = np.frombuffer(b, dtype=np.uint8)
    return np.bitwise_xor(aa, bb).tobytes()


def unbind(bound: bytes, key: bytes) -> bytes:
    return bind(bound, key)


def bundle(
    pairs: list[tuple[str, bytes]],
    *,
    D: int = LILLI_BSC_DEFAULT_DIM,
    store: "Optional[MemoryStore]" = None,
) -> bytes:
    if not pairs:
        return bytes(D // 8)

    max_pairs = _max_bundle_pairs(D)
    n = len(pairs)
    warn_threshold = math.ceil(BSC_SATURATION_WARN_RATIO * max_pairs)

    if n >= warn_threshold and store is not None:
        try:
            from iai_mcp import events

            events.write_event(
                store,
                _TELEMETRY_ROLE_SATURATION_KIND,
                {"D": D, "n_pairs": n, "max_pairs": max_pairs, "ratio": n / max_pairs},
                severity="warning",
                domain="lilli.tiers.bsc",
            )
        except Exception:  # noqa: BLE001 — telemetry must never crash bundle
            log.warning("role_saturation telemetry emit failed (non-fatal)", exc_info=True)

    if n > max_pairs:
        from iai_mcp.lilli.errors import BundleCapacityError

        # Emit-then-raise: the saturation signal must be recorded before raising,
        # including when no store is threaded through (degrade to a log-only emit).
        if store is not None:
            try:
                from iai_mcp import events

                events.write_event(
                    store,
                    _TELEMETRY_ROLE_SATURATION_KIND,
                    {
                        "D": D,
                        "n_pairs": n,
                        "max_pairs": max_pairs,
                        "ratio": n / max_pairs,
                        "fatal": True,
                    },
                    severity="error",
                    domain="lilli.tiers.bsc",
                )
            except Exception:  # noqa: BLE001 — telemetry must never mask the raise
                log.error(
                    "role_saturation (fatal) telemetry emit failed", exc_info=True
                )
        else:
            log.error(
                "BSC bundle saturation at D=%d: %d > %d pairs "
                "(no store for telemetry)",
                D,
                n,
                max_pairs,
            )

        raise BundleCapacityError(
            f"BSC bundle at D={D} accepts at most {max_pairs} pairs "
            f"({BSC_CAPACITY_DIVISOR}:1 capacity ratio); got {n}. "
            f"Reduce role count or migrate this pair set to the FHRR tier "
            f"(semantic, higher capacity)."
        )

    bound_bytes: list[bytes] = [
        bind(role_hv(role, D=D), filler) for role, filler in pairs
    ]

    if backend.use_rust():
        return bytes(backend.native().bsc_bundle(bound_bytes, D))

    stacked_bytes = np.stack(
        [np.frombuffer(hv, dtype=np.uint8) for hv in bound_bytes]
    )
    bits = np.unpackbits(stacked_bytes, axis=1).astype(np.int32)
    sums = bits.sum(axis=0)
    voted = (sums * 2 >= n).astype(np.uint8)
    return np.packbits(voted).tobytes()


def permute(hv: bytes, shift: int) -> bytes:
    if backend.use_rust():
        return bytes(backend.native().bsc_permute(hv, shift))
    bits = np.unpackbits(np.frombuffer(hv, dtype=np.uint8))
    shifted = np.roll(bits, shift)
    return np.packbits(shifted).tobytes()


def similarity(a: bytes, b: bytes) -> float:
    if len(a) != len(b):
        return 0.0
    if backend.use_rust():
        return 1.0 - float(backend.native().bsc_hamming(a, b))
    return 1.0 - hamming(a, b)


def unpack_role(hv: bytes, role: str, *, D: int = LILLI_BSC_DEFAULT_DIM) -> bytes:
    return unbind(hv, role_hv(role, D=D))

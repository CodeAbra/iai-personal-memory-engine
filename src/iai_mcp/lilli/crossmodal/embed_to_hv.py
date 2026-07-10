from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from iai_mcp.lilli.core.projection import EMBED_DIM, HV_DIM, P

if TYPE_CHECKING:
    from iai_mcp.store import MemoryStore

log = logging.getLogger(__name__)


RANK_DEFICIENCY_MIN_BATCH_SIZE: int = 8

RANK_DEFICIENCY_DEFAULT_THRESHOLD: float = 0.2

_TELEMETRY_RANK_DEFICIENCY_KIND: str = "rank_deficiency_warning"

_HV_BYTES: int = HV_DIM // 8


def from_embedding(emb: list[float]) -> bytes:
    arr = np.asarray(emb, dtype=np.float32)
    if arr.shape != (EMBED_DIM,):
        raise ValueError(
            f"from_embedding expects a length-{EMBED_DIM} embedding, "
            f"got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("from_embedding: embedding contains non-finite values")
    # Projection always runs on numpy, in BOTH backends. `emb @ P` dispatches to
    # the platform BLAS sgemv, which is memory-bandwidth-bound and already at the
    # physical ceiling — a scalar native matmul is slower AND it is the bit-parity
    # reference. So this is the deliberate hybrid exception: the hot-path ops
    # (similarity/hamming, majority-vote bundle) route to the native backend under
    # IAI_MCP_HD_BACKEND=rust, but projection stays on numpy regardless. The native
    # projection kernel is kept (and proven byte-identical on 100 frozen
    # embeddings) but production projection is never routed through it.
    projected = arr @ P
    bits = (projected >= 0).astype(np.uint8)
    return np.packbits(bits).tobytes()


def from_embedding_batch(
    embs: list[list[float]],
    *,
    store: "MemoryStore | None" = None,
    deviation_threshold: float = RANK_DEFICIENCY_DEFAULT_THRESHOLD,
) -> list[bytes]:
    hvs = [from_embedding(e) for e in embs]

    if store is None or len(embs) < RANK_DEFICIENCY_MIN_BATCH_SIZE:
        return hvs

    try:
        packed = np.frombuffer(b"".join(hvs), dtype=np.uint8).reshape(
            len(hvs), _HV_BYTES
        )
        bit_matrix = np.unpackbits(packed, axis=1)
        freq = bit_matrix.mean(axis=0)
        deviation = float(np.abs(freq - 0.5).mean())

        if deviation > deviation_threshold:
            from iai_mcp import events

            events.write_event(
                store,
                _TELEMETRY_RANK_DEFICIENCY_KIND,
                {
                    "batch_size": len(embs),
                    "deviation": deviation,
                    "threshold": deviation_threshold,
                    "hv_dim": HV_DIM,
                },
                severity="warning",
                domain="lilli.crossmodal.embed_to_hv",
            )
    except Exception:  # noqa: BLE001 — telemetry must never crash batch path
        log.warning(
            "rank_deficiency telemetry emit failed (non-fatal)", exc_info=True
        )

    return hvs


def to_embedding_neighbors(
    hv: bytes,
    store: "MemoryStore",
    k: int = 5,
) -> list:
    if len(hv) != _HV_BYTES:
        log.warning(
            "to_embedding_neighbors: expected %d bytes, got %d — returning empty",
            _HV_BYTES,
            len(hv),
        )
        return []

    packed = np.frombuffer(hv, dtype=np.uint8)
    bits = np.unpackbits(packed)
    hv_signed = bits.astype(np.float32) * 2.0 - 1.0
    approx_emb = hv_signed @ P.T
    norm = float(np.linalg.norm(approx_emb))
    if norm == 0.0:
        log.warning("to_embedding_neighbors: zero-norm reconstructed embedding")
        return []
    approx_emb = approx_emb / norm
    return store.query_similar(approx_emb.tolist(), k=k)

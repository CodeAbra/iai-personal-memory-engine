from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

from iai_mcp.lilli.tiers import bsc, fhrr, sparse_vsa
from iai_mcp.lilli.ops import (
    cleanup,
    consolidation,
    continual,
    delta,
    orthogonalize,
    replay,
    separation,
)
from iai_mcp.lilli.crossmodal import embed_to_hv

log = logging.getLogger(__name__)


class Brain:

    def __init__(self, hippo_conn: Any = None) -> None:
        self.cognitive_mode: str = "autistic"

        self.bsc = bsc
        self.fhrr = fhrr
        self.sparse_vsa = sparse_vsa

        self.ops = SimpleNamespace(
            continual=continual,
            consolidation=consolidation,
            replay=replay,
            orthogonalize=orthogonalize,
            cleanup=cleanup,
            delta=delta,
            separation=separation,
        )

        self.crossmodal = SimpleNamespace(
            embed_to_hv=embed_to_hv,
            hv_to_neighbors=embed_to_hv.to_embedding_neighbors,
        )

        self.profile = SimpleNamespace()

        self.hippo_conn = hippo_conn

        from iai_mcp.lilli.cycle.orchestrator import (
            run_consolidation,
            run_rem,
            run_sws,
        )
        self.cycle = SimpleNamespace(
            run_rem=run_rem,
            run_sws=run_sws,
            run_consolidation=run_consolidation,
        )

    def recall(self, cue: str, *, limit: int = 5, session_id: str = "brain-recall") -> list:
        if self.hippo_conn is None:
            raise RuntimeError(
                "Brain.recall requires hippo_conn (MemoryStore-like instance)"
            )
        from iai_mcp.embed import embed_query, embedder_for_store
        from iai_mcp.retrieve import recall as _retrieve_recall

        embedder = embedder_for_store(self.hippo_conn)
        cue_embedding = embed_query(embedder, cue)
        response = _retrieve_recall(
            self.hippo_conn,
            cue_embedding=cue_embedding,
            cue_text=cue,
            session_id=session_id,
            k_hits=limit,
        )
        return response.hits

    def emit_telemetry(self, kind: str, data: dict) -> None:
        if self.hippo_conn is None:
            return
        try:
            from iai_mcp.events import write_event
            write_event(self.hippo_conn, kind=kind, data=data)
        except Exception as exc:  # noqa: BLE001 -- telemetry must never crash callers
            log.warning("emit_telemetry failed for kind=%s: %s", kind, exc)

    def emit_rank_deficiency_warning(self, data: dict) -> None:
        from iai_mcp.events import TELEMETRY_RANK_DEFICIENCY
        self.emit_telemetry(TELEMETRY_RANK_DEFICIENCY, data)

    def emit_role_saturation_warning(self, data: dict) -> None:
        from iai_mcp.events import TELEMETRY_ROLE_SATURATION
        self.emit_telemetry(TELEMETRY_ROLE_SATURATION, data)

    def emit_codec_marker_missing(self, data: dict) -> None:
        from iai_mcp.events import TELEMETRY_CODEC_MARKER_MISSING
        self.emit_telemetry(TELEMETRY_CODEC_MARKER_MISSING, data)

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID


DEFAULT_EMBED_DIM = 384
EMBED_DIM = DEFAULT_EMBED_DIM

SCHEMA_VERSION_LEGACY = 1
SCHEMA_VERSION_V2 = 2
SCHEMA_VERSION_V3 = 3
SCHEMA_VERSION_V4 = 4
SCHEMA_VERSION_V5 = 5
SCHEMA_VERSION_V6 = 6
SCHEMA_VERSION_CURRENT = SCHEMA_VERSION_V6
SCHEMA_VERSION_ACCEPTED = frozenset({
    SCHEMA_VERSION_LEGACY,
    SCHEMA_VERSION_V2,
    SCHEMA_VERSION_V3,
    SCHEMA_VERSION_V4,
    SCHEMA_VERSION_V5,
    SCHEMA_VERSION_V6,
})

STRUCTURE_HV_DIM: int = 10000
STRUCTURE_HV_BYTES: int = STRUCTURE_HV_DIM // 8

HV_TIER_ENUM: frozenset[str] = frozenset({"bsc", "fhrr", "sparse_vsa"})

EPISTEMIC_STATUS_ENUM: frozenset[str] = frozenset({
    "fact", "estimate", "hypothesis", "opinion", "unknown",
})

SALIENCE_LEVEL_ENUM: frozenset[str] = frozenset({
    "unflagged", "notable", "critical",
})
SALIENCE_LEVEL_RANK: dict[str, int] = {
    "unflagged": 0, "notable": 1, "critical": 2,
}

CLS_SUMMARY_PREFIX_MARKER: str = "Cluster summary ("
"""Fixed boilerplate prefix the rich_club renderer strips before capping cls_summary lines."""

CLS_SUMMARY_PREFIX_RE = re.compile(
    r"\A" + re.escape(CLS_SUMMARY_PREFIX_MARKER) + r"\d+ records, lang=[\w-]+\): "
)
"""Full boilerplate pattern: marker + record count + language tag. Anchored
at string start (`\\A`) so any match method, not only `.match()`, stays
leading-only."""

SEMANTIC_PRUNED_TIER: str = "semantic_pruned"
TIER_ENUM = frozenset({
    "working",
    "episodic",
    "semantic",
    "procedural",
    "parametric",
    SEMANTIC_PRUNED_TIER,
})


@dataclass
class MemoryRecord:

    id: UUID
    tier: str

    literal_surface: str
    aaak_index: str

    embedding: list[float]

    community_id: UUID | None
    centrality: float
    detail_level: int
    pinned: bool

    stability: float
    difficulty: float
    last_reviewed: datetime | None
    never_decay: bool
    never_merge: bool

    provenance: list[dict[str, Any]]

    created_at: datetime
    updated_at: datetime

    language: str

    tags: list[str] = field(default_factory=list)
    s5_trust_score: float = 0.5
    profile_modulation_gain: dict[str, float] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION_CURRENT
    structure_hv: bytes = field(default=b"")
    hv_tier: str = "bsc"
    structure_hv_payload: bytes = field(default=b"")
    embedding_pending: int = 0
    role: str | None = None  # denormalized projection of the role:<x> tag; None when absent
    epistemic_status: str = "unknown"
    salience_level: str = "unflagged"
    valence: float = 0.0
    directive: bool = False

    def __post_init__(self) -> None:
        if self.detail_level >= 3:
            self.never_decay = True
        if self.tier not in TIER_ENUM:
            raise ValueError(
                f"invalid tier {self.tier!r}; must be one of {sorted(TIER_ENUM)}"
            )
        if not self.language or not isinstance(self.language, str):
            raise ValueError(
                "language is a required non-empty ISO-639-1 string field"
            )
        if not (0.0 <= self.s5_trust_score <= 1.0):
            raise ValueError(
                f"s5_trust_score must be in [0, 1], got {self.s5_trust_score}"
            )
        if self.schema_version not in SCHEMA_VERSION_ACCEPTED:
            raise ValueError(
                f"schema_version must be one of {sorted(SCHEMA_VERSION_ACCEPTED)}, "
                f"got {self.schema_version}"
            )
        if not isinstance(self.structure_hv, (bytes, bytearray)):
            raise ValueError(
                f"structure_hv must be bytes, got {type(self.structure_hv).__name__}"
            )
        if self.structure_hv and len(self.structure_hv) != STRUCTURE_HV_BYTES:
            raise ValueError(
                f"structure_hv must be empty (pre-migration) or exactly "
                f"{STRUCTURE_HV_BYTES} bytes (D={STRUCTURE_HV_DIM} BSC packed), "
                f"got {len(self.structure_hv)} bytes"
            )
        if self.hv_tier not in HV_TIER_ENUM:
            raise ValueError(
                f"hv_tier must be one of {sorted(HV_TIER_ENUM)}, got {self.hv_tier!r}; "
                f"HV_TIER_ENUM = {sorted(HV_TIER_ENUM)}"
            )
        if not isinstance(self.structure_hv_payload, (bytes, bytearray)):
            raise ValueError(
                f"structure_hv_payload must be bytes (expected bytes), "
                f"got {type(self.structure_hv_payload).__name__}"
            )
        if self.epistemic_status not in EPISTEMIC_STATUS_ENUM:
            raise ValueError(
                f"epistemic_status must be one of {sorted(EPISTEMIC_STATUS_ENUM)}, "
                f"got {self.epistemic_status!r}"
            )
        if self.salience_level not in SALIENCE_LEVEL_ENUM:
            raise ValueError(
                f"salience_level must be one of {sorted(SALIENCE_LEVEL_ENUM)}, "
                f"got {self.salience_level!r}"
            )


@dataclass
class MemoryHit:

    record_id: UUID
    score: float
    reason: str
    literal_surface: str
    adjacent_suggestions: list[UUID]
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    session_id: str | None = None
    captured_at: str | None = None
    community_id: UUID | None = None
    epistemic_status: str | None = None
    salience_level: str | None = None


@dataclass
class RecallResponse:

    hits: list[MemoryHit]
    anti_hits: list[MemoryHit]
    activation_trace: list[UUID]
    budget_used: int
    hints: list[dict] = field(default_factory=list)
    cue_mode: str = "concept"
    patterns_observed: list[dict] = field(default_factory=list)
    ann_path_used: bool = False
    # Call-local IAI_MCP_STAGE_PROFILE sub-stage timers -- never read from a
    # module global here, a concurrent request must not see another
    # request's timings.
    stage_timings: dict = field(default_factory=dict)


@dataclass
class EdgeUpdate:

    edges_boosted: int
    pairs: list[tuple[UUID, UUID]]
    new_weights: dict[str, float]


@dataclass
class ReconsolidationReceipt:

    original_id: UUID
    new_record_id: UUID
    edge_type: str
    ts: datetime

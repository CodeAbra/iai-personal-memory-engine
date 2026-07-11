"""Synthetic corpus builder for the RSS attribution harness.

Produces deterministic ``MemoryRecord`` instances with random unit-norm embeddings,
a varied tag distribution that triggers tag-cooccurrence based schema induction
during the first sleep cycle, and a populated ``structure_hv_payload`` so the
HD-tier surface is actually exercised (rather than left empty by default and
trivially classified as clean by absence).

Lifted in shape from the existing hermetic bench at
``bench/consolidation_rss_peak.py:_make_record`` — extended with varied tags and
a small BSC role-filler bundle per record.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Iterable, Sequence
from uuid import uuid4

import numpy as np

if TYPE_CHECKING:
    from iai_mcp.store import MemoryStore
    from iai_mcp.types import MemoryRecord


# Deterministic tag lattice — seven cells cycle by record index so the synthetic
# corpus exhibits enough pair-cooccurrence skew for the schema-induction step to
# fire. Verified against ``lilli/cycle/schema.py`` cooccurrence thresholding.
_TAG_LATTICE: tuple[tuple[str, ...], ...] = (
    ("bench",),
    ("bench", "preference"),
    ("bench", "fact"),
    ("bench", "routine"),
    ("bench", "todo"),
    ("bench", "preference", "fact"),
    ("bench", "routine", "todo"),
)


# Fixed role lattice for the BSC bundle. Six roles per record stays well under
# the bundle-saturation cap (``_max_bundle_pairs`` returns 10 at D=4096) so the
# corpus builder never trips ``BundleCapacityError``.
_ROLE_LATTICE: tuple[str, ...] = (
    "subject",
    "predicate",
    "object",
    "time",
    "place",
    "mood",
)


def _build_structure_hv_payload(i: int, dim: int) -> tuple[str, bytes]:
    """Return ``(hv_tier, structure_hv_payload)`` for record index ``i``.

    Imports of ``iai_mcp.lilli.tiers.bsc`` happen lazily inside this helper so
    the ``tests.rss_rca.corpus`` module is cheap to import for callers that
    only need ``make_record`` to skip the HD payload (e.g. for a fast smoke
    that does not care to exercise Surface 5).

    The ``dim`` argument here is the embedding dimension passed by the caller
    (e.g. 384 for the default model); it is intentionally *not* threaded into
    the BSC ``D`` parameter — BSC uses its own default-D codebook
    (``LILLI_BSC_DEFAULT_DIM=4096``) regardless of the embedder's d.
    """
    from iai_mcp.lilli.tiers.bsc import bundle, filler_hv

    # Three role-filler pairs per record. Cycle filler values by index so the
    # ``@lru_cache(maxsize=256)`` codebook fills naturally over the corpus.
    pairs: list[tuple[str, bytes]] = []
    for k, role in enumerate(_ROLE_LATTICE[:3]):
        value = f"filler-{role}-{(i * 7 + k * 13) % 211}"
        pairs.append((role, filler_hv(value)))
    payload = bundle(pairs)
    return "bsc", payload


def make_record(
    i: int,
    rng: np.random.Generator,
    dim: int,
    *,
    varied_tags: bool = True,
    with_hv_payload: bool = True,
) -> "MemoryRecord":
    """Build one synthetic ``MemoryRecord`` for the harness corpus.

    The embedding shape and tier match the existing hermetic bench. Tags rotate
    through ``_TAG_LATTICE`` when ``varied_tags=True`` so the synthetic store
    has enough tag-pair skew to drive the schema-induction step; when
    ``varied_tags=False`` the bench's uniform ``["bench", "rss-peak"]`` tags
    return for tight smoke reproducibility.

    When ``with_hv_payload=True`` (default) the record carries a small BSC
    role-filler bundle in ``structure_hv_payload`` so the HD-tier surface is
    exercised by ``store.insert`` and ``OPTIMIZE_HIPPO``. Setting it to
    ``False`` returns the bench-shaped record with empty payload.
    """
    from iai_mcp.types import MemoryRecord

    now = datetime.now(timezone.utc)
    vec = rng.standard_normal(dim)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm

    if varied_tags:
        tags = list(_TAG_LATTICE[i % len(_TAG_LATTICE)])
    else:
        tags = ["bench", "rss-peak"]

    if with_hv_payload:
        hv_tier, structure_hv_payload = _build_structure_hv_payload(i, dim)
    else:
        hv_tier = "bsc"
        structure_hv_payload = b""

    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=(
            f"alice daily routine note {i}: she logged a habit and a plan"
        ),
        aaak_index="",
        embedding=vec.tolist(),
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=tags,
        language="en",
        hv_tier=hv_tier,
        structure_hv_payload=structure_hv_payload,
    )


def build_corpus(
    store: "MemoryStore",
    n: int,
    *,
    seed: int = 42,
    dim: int | None = None,
    varied_tags: bool = True,
    with_hv_payload: bool = True,
) -> int:
    """Insert ``n`` synthetic records into ``store`` and return the count.

    ``dim`` defaults to ``store.embed_dim`` so the corpus matches whatever
    embedding dimension the store was configured with (typically 384). The
    deterministic seed lets the smoke test assert a reproducible row count
    without flakiness.
    """
    eff_dim = int(dim) if dim is not None else int(store.embed_dim)
    rng = np.random.default_rng(seed)
    inserted = 0
    for i in range(n):
        rec = make_record(
            i, rng, eff_dim,
            varied_tags=varied_tags,
            with_hv_payload=with_hv_payload,
        )
        store.insert(rec)
        inserted += 1
    return inserted


__all__: Sequence[str] = (
    "make_record",
    "build_corpus",
)

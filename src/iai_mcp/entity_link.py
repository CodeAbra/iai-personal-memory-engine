"""Entity-shared edge mining over lexical postings.

A rare token appearing in a handful of records is an entity-grade anchor
(a name, a place, a project, a distinctive noun). Records sharing such an
anchor get an `entity_shared` edge so the 2-hop spread can cross a
semantic gap similarity cannot bridge: a cue that seeds on one caffeine
mention reaches the sleep-tracker note that shares the anchor even though
the cue itself never says caffeine.

The edge is INFERRED structure: it widens spread reach but is excluded
from the ranking degree — shared vocabulary must not manufacture hubs.
"""
from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)

ENTITY_EDGE_TYPE = "entity_shared"
ENTITY_MIN_DF = 2
ENTITY_MAX_DF = 8
ENTITY_MIN_TOKEN_LEN = 5
ENTITY_MAX_PAIRS_PER_TOKEN = 28
ENTITY_MAX_EDGES_PER_RUN = 500
ENTITY_EDGE_WEIGHT = 0.3


def _entity_grade(token: str) -> bool:
    return len(token) >= ENTITY_MIN_TOKEN_LEN and token.isalpha()


def mine_entity_edges(
    store: Any,
    *,
    min_df: int = ENTITY_MIN_DF,
    max_df: int = ENTITY_MAX_DF,
    max_pairs_per_token: int = ENTITY_MAX_PAIRS_PER_TOKEN,
    max_edges_per_run: int = ENTITY_MAX_EDGES_PER_RUN,
) -> "dict[str, int]":
    """Mint entity_shared edges from the warm lexical postings.

    Runs on the consolidation side only: ensuring the lexical index may
    pay the O(corpus)-decrypt build, which is never allowed on the awake
    recall path. Edge writes go through boost_edges (canonicalized pair
    keys, merge-insert), so re-running over the same corpus is idempotent.
    """
    try:
        store.lexical_search("entity-link warm-up", k=1)
    except Exception as exc:  # noqa: BLE001 -- no index means nothing to mine
        logger.debug("entity_link lexical warm-up failed: %s", exc)
        return {"tokens_scanned": 0, "tokens_used": 0, "edges_minted": 0}

    idx = getattr(store, "_lexical_idx", None)
    if idx is None:
        return {"tokens_scanned": 0, "tokens_used": 0, "edges_minted": 0}

    postings = idx.iter_token_postings()
    tokens_used = 0
    minted = 0
    pairs_batch: "list[tuple[UUID, UUID]]" = []
    for token, bucket in postings:
        if minted >= max_edges_per_run:
            break
        if not _entity_grade(token):
            continue
        df = len(bucket)
        if df < min_df or df > max_df:
            continue
        rids = sorted(bucket)
        pairs = [
            (rids[i], rids[j])
            for i in range(len(rids))
            for j in range(i + 1, len(rids))
        ][:max_pairs_per_token]
        if not pairs:
            continue
        tokens_used += 1
        for a, b in pairs:
            if minted >= max_edges_per_run:
                break
            try:
                pairs_batch.append((UUID(a), UUID(b)))
            except ValueError:
                continue
            minted += 1

    if pairs_batch:
        store.boost_edges(
            pairs_batch, delta=ENTITY_EDGE_WEIGHT, edge_type=ENTITY_EDGE_TYPE,
        )
        try:
            from iai_mcp.store import flush_edge_buffer
            flush_edge_buffer(store)
        except Exception as exc:  # noqa: BLE001 -- graph builds drain the buffer anyway
            logger.debug("entity_link edge flush deferred: %s", exc)

    result = {
        "tokens_scanned": len(postings),
        "tokens_used": tokens_used,
        "edges_minted": minted,
    }
    logger.info("entity_link %s", result)
    return result

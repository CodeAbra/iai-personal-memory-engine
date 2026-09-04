"""Discovery-stage candidate-membership capture and miss-cause classification
for the recall retrieval-verdict measurement.

``capture_discovery_membership`` dispatches one ``memory_recall`` cue
through the real production entry point (``core.dispatch``) and observes,
per discovery source (ANN cosine, exact-cosine authority, graph-edge
2-hop spread, rich-club), which ids it returned -- reconciled against the
ids that actually survived into the assembled per-call candidate pool (the
fresh ``MemoryGraph`` node set the production code builds immediately
before ranking). A gold id a source returned but that its own per-source
cap then dropped is reported as never-surviving-into-the-pool, never as
in-pool.

The capture wraps (never replaces) whatever is currently installed at
each of the four discovery call sites plus the fresh per-call graph
construction: a caller-supplied test stub is honored exactly as
production code is, since the wrapper always delegates to the callable it
found at wrap time and restores it afterward.

``classify_miss_cause`` maps one gold id's discovery-membership summary
onto exactly one of four miss causes: discovery-excluded (never survived
into the post-cap pool, whether never returned by any source or returned
and then capped out), 2-hop-spread-did-not-reach (a caller-supplied signal
that the only structurally possible discovery path was a graph edge to an
already-discovered node, and that edge does not exist), rank/budget
dropped (survived into the pool but fell outside the budget-packed
top-K), and lexical-cold-or-IDF-gated (the only rescue path was the
lexical lane and it contributed nothing).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
_REPO_PATH = str(Path(__file__).resolve().parent.parent)
if _REPO_PATH not in sys.path:
    sys.path.insert(0, _REPO_PATH)

SOURCE_ANN = "ann"
SOURCE_EXACT_AUTHORITY = "exact_authority"
SOURCE_TWO_HOP = "two_hop"
SOURCE_RICH_CLUB = "rich_club"
ALL_SOURCES = (SOURCE_ANN, SOURCE_EXACT_AUTHORITY, SOURCE_TWO_HOP, SOURCE_RICH_CLUB)

CAUSE_DISCOVERY_EXCLUDED = "discovery_excluded"
CAUSE_SPREAD_DID_NOT_REACH = "spread_did_not_reach"
CAUSE_RANK_BUDGET_DROPPED = "rank_budget_dropped"
CAUSE_LEXICAL_COLD_OR_IDF_GATED = "lexical_cold_or_idf_gated"
ALL_CAUSES = (
    CAUSE_DISCOVERY_EXCLUDED,
    CAUSE_SPREAD_DID_NOT_REACH,
    CAUSE_RANK_BUDGET_DROPPED,
    CAUSE_LEXICAL_COLD_OR_IDF_GATED,
)


@dataclass
class DiscoveryMembership:
    """Per-cue discovery-stage capture: which of the four sources returned
    each id (raw, as returned by that source) and which ids actually
    survived into the assembled post-cap candidate pool."""

    raw_sources: "dict[str, set]"
    post_cap_ids: set
    response: dict = field(default_factory=dict)

    def sources_for(self, record_id: Any) -> "set[str]":
        return {source for source, ids in self.raw_sources.items() if record_id in ids}

    def survived(self, record_id: Any) -> bool:
        return record_id in self.post_cap_ids

    def gold_membership(
        self, gold_id: Any, *, spread_edge_missing: bool = False,
    ) -> "GoldMembership":
        """Project this cue-level capture down to one gold id's membership
        summary -- the shape ``classify_miss_cause`` consumes.

        A gold id returned by at least one source but absent from the
        post-cap pool did not survive that source's own per-source cap
        (spread top-5 each, rich-club top-50, exact-authority live
        filter) -- it is discovery-excluded, never in-pool, regardless of
        the raw return.
        """
        sources = self.sources_for(gold_id)
        in_pool = gold_id in self.post_cap_ids
        capped_out = bool(sources) and not in_pool
        return GoldMembership(
            gold_id=gold_id,
            in_post_cap_pool=in_pool,
            returned_by_sources=frozenset(sources),
            capped_out=capped_out,
            spread_edge_missing=spread_edge_missing and not capped_out,
        )


@dataclass(frozen=True)
class GoldMembership:
    """A single gold record's discovery-membership summary, reconciled
    against the post-cap candidate pool for one dispatched cue."""

    gold_id: Any
    in_post_cap_pool: bool
    returned_by_sources: "frozenset[str]" = frozenset()
    capped_out: bool = False
    spread_edge_missing: bool = False


def _patch_instance_attr(obj: Any, name: str, new_value: Any):
    """Patch an instance attribute, returning (original_value, restore).

    ``restore`` reverts to the exact pre-patch state: if no instance-level
    attribute existed (the common case -- the callable was resolved via
    the class), restoring deletes the shadow rather than re-setting it, so
    a store reused across many captures never accumulates a permanent
    instance-level shadow of its own class methods.
    """
    had_instance_attr = name in obj.__dict__
    original = getattr(obj, name)
    setattr(obj, name, new_value)

    def _restore() -> None:
        if had_instance_attr:
            setattr(obj, name, original)
        else:
            delattr(obj, name)

    return original, _restore


def capture_discovery_membership(store: Any, params: dict) -> DiscoveryMembership:
    """Dispatch one ``memory_recall`` cue through the real production entry
    point and reconcile what each of the four discovery sources returned
    against what actually survived into the assembled post-cap candidate
    pool (the fresh per-call ``MemoryGraph`` node set).
    """
    from iai_mcp import core
    from iai_mcp import graph as graph_mod
    from iai_mcp import runtime_graph_cache as rgc

    raw: "dict[str, set]" = {source: set() for source in ALL_SOURCES}

    original_query_similar, restore_query_similar = _patch_instance_attr(
        store, "query_similar", None,
    )

    def _query_similar(*args, **kwargs):
        result = original_query_similar(*args, **kwargs)
        for rec, _score in result:
            raw[SOURCE_ANN].add(rec.id)
        return result

    store.query_similar = _query_similar

    original_exact_top_k, restore_exact_top_k = _patch_instance_attr(
        store, "exact_top_k", None,
    )

    def _exact_top_k(*args, **kwargs):
        result = original_exact_top_k(*args, **kwargs)
        for rid, _score in result:
            raw[SOURCE_EXACT_AUTHORITY].add(rid)
        return result

    store.exact_top_k = _exact_top_k

    original_incident_edges = core._incident_edges_warm

    def _incident_edges_warm(active_store, ids, top_k=5):
        result = original_incident_edges(active_store, ids, top_k=top_k)
        for _src, neighbors in result.items():
            for nbr, _et, _wt in neighbors:
                raw[SOURCE_TWO_HOP].add(nbr)
        return result

    original_load_structural = rgc.load_recall_structural

    def _load_recall_structural(*args, **kwargs):
        result = original_load_structural(*args, **kwargs)
        _assignment, rc, *_rest = result
        for rid in (rc or []):
            raw[SOURCE_RICH_CLUB].add(rid)
        return result

    created_graphs: list = []
    add_node_calls: "dict[int, set]" = {}

    original_graph_init = graph_mod.MemoryGraph.__init__
    original_add_node = graph_mod.MemoryGraph.add_node

    def _graph_init(self, *a, **kw):
        original_graph_init(self, *a, **kw)
        created_graphs.append(self)

    def _add_node(self, node_id, *a, **kw):
        add_node_calls.setdefault(id(self), set()).add(node_id)
        return original_add_node(self, node_id, *a, **kw)

    core._incident_edges_warm = _incident_edges_warm
    rgc.load_recall_structural = _load_recall_structural
    graph_mod.MemoryGraph.__init__ = _graph_init
    graph_mod.MemoryGraph.add_node = _add_node
    try:
        response = core.dispatch(store, "memory_recall", params)
    finally:
        restore_query_similar()
        restore_exact_top_k()
        core._incident_edges_warm = original_incident_edges
        rgc.load_recall_structural = original_load_structural
        graph_mod.MemoryGraph.__init__ = original_graph_init
        graph_mod.MemoryGraph.add_node = original_add_node

    # The fresh, per-call candidate pool graph (core/__init__.py's
    # `graph = MemoryGraph()`, populated only from `_candidate_recs`) is
    # the only `MemoryGraph` this dispatch call constructs OTHER than the
    # store-resident rank-builder graph (constructed at most once, lazily,
    # per store) -- exclude that one by identity rather than by
    # construction order, which is robust regardless of whether this was
    # the store's first-ever dispatch.
    builder_graph = getattr(store, "_rank_builder_graph", None)
    pool_candidates = [g for g in created_graphs if g is not builder_graph]
    if len(pool_candidates) != 1:
        raise RuntimeError(
            "expected exactly one non-resident MemoryGraph construction "
            f"during dispatch, got {len(pool_candidates)} -- the post-cap "
            "pool capture assumption no longer holds"
        )
    post_cap_ids = add_node_calls.get(id(pool_candidates[0]), set())

    return DiscoveryMembership(raw_sources=raw, post_cap_ids=post_cap_ids, response=response)


def classify_miss_cause(
    membership: GoldMembership, rank_context: dict, lexical_context: dict,
) -> str:
    """Classify one confirmed miss into exactly one of the four causes.

    ``rank_context`` carries ``in_budget_topk`` (bool): whether the gold
    id, if it survived into the post-cap pool, also made the final
    budget-packed top-K.

    ``lexical_context`` carries ``lexical_only_rescue`` (bool, the ONLY
    structurally possible rescue path for this miss was the lexical lane)
    and ``lexical_contributed`` (bool, the lexical lane actually served
    the gold id).

    Priority, evaluated in order, so every miss lands in exactly one
    category:
      1. survived into the pool but missed the budget cutoff -> rank/budget dropped
      2. returned by a source but dropped by that source's own cap -> discovery excluded
      3. the only structural path was a graph edge that does not exist -> spread did not reach
      4. the only rescue path was the lexical lane and it contributed nothing -> lexical cold/gated
      5. otherwise -> discovery excluded (never returned by any source, no known edge, no lexical path)
    """
    if membership.in_post_cap_pool:
        if rank_context.get("in_budget_topk", False):
            raise ValueError(
                "gold id was served (in the pool and within budget) -- not "
                "a miss, there is no cause to classify"
            )
        return CAUSE_RANK_BUDGET_DROPPED

    if membership.capped_out:
        return CAUSE_DISCOVERY_EXCLUDED

    if membership.spread_edge_missing:
        return CAUSE_SPREAD_DID_NOT_REACH

    if lexical_context.get("lexical_only_rescue", False) and not lexical_context.get(
        "lexical_contributed", False,
    ):
        return CAUSE_LEXICAL_COLD_OR_IDF_GATED

    return CAUSE_DISCOVERY_EXCLUDED

from __future__ import annotations

from math import ceil
from uuid import UUID

from iai_mcp.graph import MemoryGraph


def rich_club_nodes(
    graph: MemoryGraph,
    percent: float = 0.10,
    centrality: "dict[UUID, float] | None" = None,
) -> list[UUID]:
    """Top-``percent`` nodes by betweenness centrality.

    When ``centrality`` is supplied the ranking reuses it directly — the caller
    has already computed (or loaded) the betweenness map and passes it in so this
    function never triggers a second exact betweenness pass. The long-lived
    recall process always supplies it: an exact in-parent Brandes pass on a large
    graph spikes the resident set toward the watchdog cap, so the parent must
    reuse the child-computed / cached / neutral map rather than recompute here.

    When ``centrality`` is None the map is computed on the graph in-process. That
    is reserved for callers that run in a short-lived child process (which
    reclaims its own arenas on exit) or operate on a graph small enough that the
    in-process pass is genuinely bounded.

    Ranking blends a staleness decay into the centrality value so a
    structurally-central-but-stale node loses ground to a recently-reinforced
    one; a node with no ``created_at`` payload decays 0.0 (no-op), keeping the
    ranking pure centrality for graphs that never wrote the field.
    """
    if graph.node_count() == 0:
        return []
    if centrality is None:
        centrality = graph.centrality()
    if not centrality:
        return []

    # Function-local: avoids pulling pipeline's heavier top-level imports
    # (embed.py et al.) onto richclub's own module load.
    from iai_mcp.pipeline import _age_penalty, _payload_created_at

    def _decayed_score(node_id: UUID, value: float) -> float:
        created_at = _payload_created_at(graph.get_payload(node_id).get("created_at", ""))
        return value * (1.0 - _age_penalty(created_at))

    k = max(1, ceil(len(centrality) * percent))
    ranked = sorted(
        centrality.items(), key=lambda kv: _decayed_score(kv[0], kv[1]), reverse=True
    )
    return [node_id for node_id, _ in ranked[:k]]

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from iai_mcp.store import MemoryStore


def _pairs_from_contradicts_edges(
    contradicts_edges: dict, hit_ids_set: set[str],
) -> list[tuple[str, str]]:
    """Derive contracts pairs from an already-fetched incident-edges
    adjacency (candidate id -> [(neighbour_str, edge_type, weight), ...]).

    The adjacency is direction-lossy for a given stored edge: a single
    ``(src, dst)`` row surfaces as one entry on EACH side (once under
    ``src``'s list, once under ``dst``'s), so the true stored row direction
    cannot be recovered from it alone (unlike the raw ``SELECT`` this
    replaces, which has no ``ORDER BY`` and is not a direction contract
    across engines either). Each unordered pair is therefore deduped and
    emitted exactly once, in the pair's own lexicographic order — a
    deterministic, engine-independent convention rather than a re-derivation
    of unspecified row-return order.
    """
    seen: set[frozenset] = set()
    pairs: list[tuple[str, str]] = []
    for qid, edges in contradicts_edges.items():
        src_s = str(qid)
        if src_s not in hit_ids_set:
            continue
        for (dst_s, _et, _wt) in edges:
            if dst_s not in hit_ids_set or dst_s == src_s:
                continue
            key = frozenset((src_s, dst_s))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(tuple(sorted((src_s, dst_s))))
    pairs.sort()
    return pairs


def verify_hit_set(
    store: "MemoryStore",
    hit_record_ids: list[UUID],
    contradicts_edges: "dict | None" = None,
) -> dict:
    """`contradiction_pairs` tuple order is a display convention, not a
    stored-direction contract -- it differs between kill-switch states for
    the identical stored edge (lexicographic when `contradicts_edges` is
    reused, raw storage order on the own-query path).
    """
    hit_count = len(hit_record_ids)
    if hit_count < 2:
        return {
            "has_contradictions": False,
            "contradiction_pairs": [],
            "teachback_summary": f"All {hit_count} memories appear mutually consistent.",
            "hit_count": hit_count,
        }

    hit_ids_str = [str(h) for h in hit_record_ids]

    if contradicts_edges is not None:
        pairs = _pairs_from_contradicts_edges(contradicts_edges, set(hit_ids_str))
    else:
        from iai_mcp.store import EDGES_TABLE

        df = None
        try:
            tbl = store.db.open_table(EDGES_TABLE)
            id_list = ", ".join(f"'{i}'" for i in hit_ids_str)
            where = (
                f"edge_type = 'contradicts' "
                f"AND src IN ({id_list}) "
                f"AND dst IN ({id_list})"
            )
            df = tbl.search().where(where).to_pandas()
        except (OSError, RuntimeError, ValueError):
            df = None

        pairs = []
        if df is not None and len(df) > 0:
            for _, row in df.iterrows():
                pairs.append((str(row["src"]), str(row["dst"])))

    has_contradictions = len(pairs) > 0
    if has_contradictions:
        sample = pairs[0]
        summary = (
            f"WARNING: {len(pairs)} conflicting memory pair(s) surfaced "
            f"among {hit_count} hits — example: ({sample[0]}, {sample[1]})."
        )
    else:
        summary = f"All {hit_count} memories appear mutually consistent."

    return {
        "has_contradictions": has_contradictions,
        "contradiction_pairs": pairs,
        "teachback_summary": summary,
        "hit_count": hit_count,
    }

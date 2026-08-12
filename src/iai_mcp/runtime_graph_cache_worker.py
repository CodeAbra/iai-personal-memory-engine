"""Spawn-context workers that own the heavy graph allocations.

Two entrypoints share one Pipe protocol for receiving the graph (compact
records + edges chunks) and differ only in what they compute and stream back:

  - `_worker_entry` — builds a MemoryGraph, runs community detection +
    rich-club ranking + max-degree, and streams the full result. Drives the
    periodic runtime-graph rebuild.
  - `_community_only_worker_entry` — builds a MemoryGraph and runs community
    detection ONLY; the result stream omits rich-club and max-degree (the
    callers that use it compute rich-club in-parent or do not need it). Drives
    the crisis reclustering pass and the background topology snapshot, keeping
    the numba JIT arenas out of the long-lived daemon address space.

Both exit when their protocol completes — the OS reclaims the worker's address
space so the per-cycle JIT/allocation footprint leaves zero residue in the
parent daemon.

AES-key fence (invariant):
Neither entrypoint opens the storage backend or sees the encryption key. The
module's top-level import surface is empty by design (only
`from __future__ import annotations`). All compute-side imports are performed
unconditionally at the top of each entrypoint, BEFORE the recv loop and BEFORE
any branch on message kind, so the import-surface guard tests can drive the
abort path and still observe the full late-import closure.

Adding any top-level or late import of `iai_mcp.hippo`, `iai_mcp.store`,
`iai_mcp.daemon`, or `iai_mcp.crypto` in either entrypoint breaks the fence and
must be caught by the import-surface guard tests in
`tests/test_runtime_graph_cache_worker.py`.

TEST-ONLY env-var affordance:
    IAI_MCP_RGC_WORKER_CRASH_AFTER_N_NODES — if set to a non-negative integer
    N, the worker exits with code 1 after receiving N node chunks (or
    immediately if N == 0). Used exclusively by the worker-crash chaos test
    to verify fail-fast disposition and last-good snapshot preservation; the
    daemon never sets this in production.
"""
from __future__ import annotations


_ASSIGN_CHUNK = 2000


def _worker_entry(conn) -> None:
    """Recv-then-compute-then-stream over `conn`.

    Hard contract: every late import below MUST execute unconditionally on
    function entry, before the recv loop. The first message handled is the
    abort sentinel `("abort", None)` which returns immediately AFTER the late
    imports have fired — this is both a real cancellation path and the
    mechanism that makes the import-surface guard test exercise the full
    module closure.

    Recognized message kinds:
      - `("abort", None)` — clean early exit (no compute, no result stream).
      - `("nodes", [(id_str, embedding_blob), ...])` — accumulate node tuples.
      - `("nodes_end", None)` — sentinel; no action.
      - `("edges", [(src_str, dst_str, weight), ...])` — accumulate edges.
      - `("edges_end", None)` — break the recv loop and proceed to compute.

    Result stream (parent reassembles into a CommunityAssignment):
      - `("community_table", [(comm_uuid_bytes, centroid_bytes_or_None), ...])`
      - repeated `("assign", [(node_uuid_bytes, comm_idx_int), ...])` of size
        `<= _ASSIGN_CHUNK`, then one `("assign_end", None)`
      - `("backend", str)` — streamed verbatim from `assignment.backend`
      - `("top_communities", [uuid_bytes, ...])`
      - `("mid_regions", [(comm_uuid_bytes, [node_uuid_bytes, ...]), ...])`
      - `("rich_club", [uuid_bytes, ...])`
      - `("max_degree", int)`
      - `("done", None)`
    """
    # Late imports — execute FIRST, unconditionally. These are the only
    # transitive surface the worker is allowed to load. Anything that pulls
    # in storage, encryption, or the daemon must be rejected here.
    import os
    import sys
    import numpy as np
    from uuid import UUID
    from iai_mcp.graph import MemoryGraph
    from iai_mcp.community import detect_communities
    from iai_mcp.centrality_approx import centrality_for_runtime
    from iai_mcp.richclub import rich_club_nodes

    # TEST-ONLY: opt-in crash-after-N-node-chunks hook for the chaos test.
    crash_after_env = os.environ.get("IAI_MCP_RGC_WORKER_CRASH_AFTER_N_NODES")
    try:
        crash_after = int(crash_after_env) if crash_after_env is not None else None
    except ValueError:
        crash_after = None
    node_chunks_received = 0

    nodes_buf: list = []
    edges_buf: list = []

    try:
        # Recv loop.
        while True:
            envelope = conn.recv()
            if not (isinstance(envelope, tuple) and len(envelope) == 2):
                raise RuntimeError(f"malformed envelope: {envelope!r}")
            kind, payload = envelope
            if kind == "abort":
                return
            elif kind == "nodes":
                for id_str, emb_blob in payload:
                    # Keep the buffer compact: .tolist() boxes every float and
                    # a corpus-scale graph in this child is what the memory
                    # watchdog kills. The frame is reused, so copy.
                    emb = np.frombuffer(emb_blob, dtype=np.float32).copy()
                    nodes_buf.append((UUID(id_str), None, emb, {}))
                node_chunks_received += 1
                if crash_after is not None and node_chunks_received >= crash_after:
                    # TEST-ONLY: simulate a mid-stream crash. Skip the
                    # ack/result loop and exit non-zero so the parent
                    # observes broken-pipe / pipe-EOF + nonzero-exit.
                    sys.stderr.write(
                        f"rgc_worker test-hook crash after "
                        f"{node_chunks_received} node chunks\n"
                    )
                    sys.exit(1)
            elif kind == "nodes_end":
                continue
            elif kind == "edges":
                for edge_row in payload:
                    src_str, dst_str, weight = edge_row[0], edge_row[1], edge_row[2]
                    edge_type = edge_row[3] if len(edge_row) > 3 else "hebbian"
                    edges_buf.append((
                        UUID(src_str),
                        UUID(dst_str),
                        float(weight),
                        str(edge_type or "hebbian"),
                    ))
            elif kind == "edges_end":
                break
            else:
                raise RuntimeError(f"unknown chunk kind: {kind!r}")

        # Compute.
        graph = MemoryGraph()
        graph.clear_and_rebuild(nodes_buf, edges_buf)
        assignment = detect_communities(graph, prior=None, prior_mode="cold")
        # Rank the rich club from the warm-graph centrality (exact below the
        # cutoff, bounded k-source sampled betweenness above it) rather than
        # letting rich_club_nodes trigger its own exact O(V*E) Brandes pass --
        # that exact pass is what stalled this child at scale.
        worker_centrality = centrality_for_runtime(graph)
        rc = rich_club_nodes(graph, centrality=worker_centrality)
        # Ranking degree counts EARNED edges only — same exclusion as the
        # cold recall path: inferred links (similarity seeds, entity
        # anchors) must not inflate hub rank on look-alike corpora.
        _degree_pairs = list(graph.degrees(
            exclude_types=graph.RANKING_DEGREE_EXCLUDED
        ))
        max_degree = max((d for _, d in _degree_pairs), default=0)
        # Build per-node degree map as a byproduct of the same pass — emitted
        # alongside max_degree so the parent can populate ranking degree from
        # the cached true values rather than from a bounded traversal count.
        node_degree_map = [(nid.bytes, d) for nid, d in _degree_pairs]

        # Result stream — community table first so the parent can index assigns.
        # Deterministic order: sort communities by UUID bytes.
        unique_comms = sorted(set(assignment.node_to_community.values()), key=lambda u: u.bytes)
        comm_index = {cu: i for i, cu in enumerate(unique_comms)}
        community_table_payload = []
        for cu in unique_comms:
            cent = assignment.community_centroids.get(cu)
            if cent is None or len(cent) == 0:
                cent_bytes = None
            else:
                cent_bytes = np.asarray(cent, dtype=np.float32).tobytes()
            community_table_payload.append((cu.bytes, cent_bytes))
        conn.send(("community_table", community_table_payload))

        # Assign stream — sorted by node UUID bytes for determinism.
        sorted_nodes = sorted(
            assignment.node_to_community.items(), key=lambda kv: kv[0].bytes
        )
        chunk: list = []
        for node_uuid, comm_uuid in sorted_nodes:
            chunk.append((node_uuid.bytes, comm_index[comm_uuid]))
            if len(chunk) >= _ASSIGN_CHUNK:
                conn.send(("assign", chunk))
                chunk = []
        if chunk:
            conn.send(("assign", chunk))
        conn.send(("assign_end", None))

        # Backend, top, mid, rich-club, max_degree, done.
        conn.send(("backend", str(assignment.backend)))
        conn.send((
            "top_communities",
            [u.bytes for u in assignment.top_communities],
        ))
        mid_payload = [
            (comm.bytes, [m.bytes for m in members])
            for comm, members in assignment.mid_regions.items()
        ]
        conn.send(("mid_regions", mid_payload))
        conn.send(("modularity", float(assignment.modularity)))
        conn.send(("rich_club", [u.bytes for u in rc]))
        conn.send(("max_degree", int(max_degree)))
        # Emit per-node degree map in chunks to mirror the assign chunking shape.
        _ND_CHUNK = _ASSIGN_CHUNK
        nd_chunk: list = []
        for node_bytes, deg in node_degree_map:
            nd_chunk.append((node_bytes, deg))
            if len(nd_chunk) >= _ND_CHUNK:
                conn.send(("node_degrees", nd_chunk))
                nd_chunk = []
        if nd_chunk:
            conn.send(("node_degrees", nd_chunk))
        conn.send(("done", None))
    except RuntimeError:
        # Protocol violation — re-raise so the in-process driver tests see it.
        # The subprocess wrapper below converts this into ("error", ...) + exit(1).
        try:
            import traceback
            err = traceback.format_exc(limit=2)[:500]
            conn.send(("error", err))
        except Exception:  # noqa: BLE001
            pass
        raise
    except Exception as exc:  # noqa: BLE001
        try:
            conn.send(("error", repr(exc)[:500]))
        except Exception:  # noqa: BLE001
            pass
        sys.exit(1)
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def _stream_centrality(conn, graph) -> None:
    """Compute the warm-graph centrality and stream it back in chunks.

    The map is `node_uuid -> float`. Chunks are sorted by node UUID bytes for a
    reproducible stream, mirroring the assign-stream discipline. The values come
    from `centrality_for_runtime`: exact Brandes betweenness on a small graph,
    and a bounded deterministic approximation (k-source sampled betweenness)
    above a node-count cutoff, since exact Brandes is O(V*E) and intractable on a
    large corpus. Both branches are deterministic and percentile-normalized, so
    consecutive warm cycles on an unchanged graph reassemble an identical map and
    the seed blend never flickers.
    """
    from iai_mcp.centrality_approx import centrality_for_runtime

    centrality_map = centrality_for_runtime(graph)
    sorted_items = sorted(centrality_map.items(), key=lambda kv: kv[0].bytes)
    chunk: list = []
    for node_uuid, value in sorted_items:
        chunk.append((node_uuid.bytes, float(value)))
        if len(chunk) >= _ASSIGN_CHUNK:
            conn.send(("centrality", chunk))
            chunk = []
    if chunk:
        conn.send(("centrality", chunk))


def _community_only_worker_entry(conn) -> None:
    """Recv-then-compute-communities-then-stream over `conn`.

    Same AES fence + late-import discipline as `_worker_entry`. The recv
    protocol is identical, with one addition: a leading `("config", {...})`
    envelope carries the `prior_mode` for `detect_communities` plus two optional
    flags that select what the worker streams back:

      - `with_centrality` — when true, the worker also computes the full
        betweenness centrality on the same graph it just built and streams it
        as `("centrality", ...)` chunks before `("done", None)`. This folds the
        community partition and the centrality map into ONE child graph-build,
        so the parent never holds either the community-detection arenas or the
        betweenness intermediate.
      - `centrality_only` — when true, the worker skips community detection
        entirely and streams only the `("centrality", ...)` chunks. Used by the
        cache-hit path that already has a community assignment but needs the
        centrality recomputed.

    The result stream omits `rich_club` and `max_degree` — this entry computes
    communities (and optionally centrality) only.

    Recognized message kinds:
      - `("config", {"prior_mode": "seeded"|"cold", "with_centrality": bool,
        "centrality_only": bool})` — optional, sent first.
      - `("abort", None)` — clean early exit (no compute, no result stream).
      - `("nodes", [(id_str, embedding_blob), ...])` — accumulate node tuples
        with full embeddings (used by community detection and mixed paths).
      - `("nodes_topology", [id_str, ...])` — topology-only variant: accumulates
        node ids with empty embeddings. Sufficient for betweenness centrality
        since centrality reads only graph structure. Avoids embedding decode and
        materialization cost in centrality-only mode.
      - `("nodes_end", None)` — sentinel; no action.
      - `("edges", [(src_str, dst_str, weight), ...])` — accumulate edges.
      - `("edges_end", None)` — break the recv loop and proceed to compute.

    Result stream (parent reassembles into a CommunityAssignment):
      - `("community_table", [(comm_uuid_bytes, centroid_bytes_or_None), ...])`
      - repeated `("assign", [(node_uuid_bytes, comm_idx_int), ...])` of size
        `<= _ASSIGN_CHUNK`, then one `("assign_end", None)`
      - `("backend", str)` — streamed verbatim from `assignment.backend`
      - `("top_communities", [uuid_bytes, ...])`
      - `("mid_regions", [(comm_uuid_bytes, [node_uuid_bytes, ...]), ...])`
      - repeated `("centrality", [(node_uuid_bytes, float), ...])` of size
        `<= _ASSIGN_CHUNK` (only when `with_centrality`/`centrality_only`)
      - `("done", None)`

    In `centrality_only` mode the community/assign/backend/top/mid envelopes are
    omitted; the stream carries only the `("centrality", ...)` chunks + `done`.
    """
    # Late imports — execute FIRST, unconditionally. These are the only
    # transitive surface the worker is allowed to load. Anything that pulls
    # in storage, encryption, or the daemon must be rejected here.
    import sys
    import numpy as np
    from uuid import UUID
    from iai_mcp.graph import MemoryGraph
    from iai_mcp.community import detect_communities

    prior_mode = "seeded"
    with_centrality = False
    centrality_only = False
    nodes_buf: list = []
    edges_buf: list = []

    try:
        # Recv loop.
        while True:
            envelope = conn.recv()
            if not (isinstance(envelope, tuple) and len(envelope) == 2):
                raise RuntimeError(f"malformed envelope: {envelope!r}")
            kind, payload = envelope
            if kind == "config":
                if isinstance(payload, dict):
                    mode = payload.get("prior_mode")
                    if mode in ("seeded", "cold"):
                        prior_mode = mode
                    with_centrality = bool(payload.get("with_centrality", False))
                    centrality_only = bool(payload.get("centrality_only", False))
            elif kind == "abort":
                return
            elif kind == "nodes":
                for id_str, emb_blob in payload:
                    # Keep the buffer compact: .tolist() boxes every float and
                    # a corpus-scale graph in this child is what the memory
                    # watchdog kills. The frame is reused, so copy.
                    emb = np.frombuffer(emb_blob, dtype=np.float32).copy()
                    nodes_buf.append((UUID(id_str), None, emb, {}))
            elif kind == "nodes_topology":
                # Topology-only node stream: ids only, no embedding blobs.
                # Betweenness centrality depends only on graph topology (node ids
                # and edges), so embedding-free nodes are sufficient for centrality
                # computation. The empty embedding list keeps the MemoryGraph node
                # record valid without any decode or materialization cost.
                for id_str in payload:
                    nodes_buf.append((UUID(id_str), None, [], {}))
            elif kind == "nodes_end":
                continue
            elif kind == "edges":
                for edge_row in payload:
                    src_str, dst_str, weight = edge_row[0], edge_row[1], edge_row[2]
                    edge_type = edge_row[3] if len(edge_row) > 3 else "hebbian"
                    edges_buf.append((
                        UUID(src_str),
                        UUID(dst_str),
                        float(weight),
                        str(edge_type or "hebbian"),
                    ))
            elif kind == "edges_end":
                break
            else:
                raise RuntimeError(f"unknown chunk kind: {kind!r}")

        # Build the graph once; both community detection and centrality read it.
        graph = MemoryGraph()
        graph.clear_and_rebuild(nodes_buf, edges_buf)

        if centrality_only:
            # Skip detection — stream only the centrality map.
            _stream_centrality(conn, graph)
            conn.send(("done", None))
            return

        assignment = detect_communities(
            graph, prior=None, prior_mode=prior_mode
        )

        # Result stream — community table first so the parent can index assigns.
        # Deterministic order: sort communities by UUID bytes.
        unique_comms = sorted(
            set(assignment.node_to_community.values()), key=lambda u: u.bytes
        )
        comm_index = {cu: i for i, cu in enumerate(unique_comms)}
        community_table_payload = []
        for cu in unique_comms:
            cent = assignment.community_centroids.get(cu)
            if cent is None or len(cent) == 0:
                cent_bytes = None
            else:
                cent_bytes = np.asarray(cent, dtype=np.float32).tobytes()
            community_table_payload.append((cu.bytes, cent_bytes))
        conn.send(("community_table", community_table_payload))

        # Assign stream — sorted by node UUID bytes for determinism.
        sorted_nodes = sorted(
            assignment.node_to_community.items(), key=lambda kv: kv[0].bytes
        )
        chunk: list = []
        for node_uuid, comm_uuid in sorted_nodes:
            chunk.append((node_uuid.bytes, comm_index[comm_uuid]))
            if len(chunk) >= _ASSIGN_CHUNK:
                conn.send(("assign", chunk))
                chunk = []
        if chunk:
            conn.send(("assign", chunk))
        conn.send(("assign_end", None))

        # Backend, top, mid. No rich_club / max_degree.
        conn.send(("backend", str(assignment.backend)))
        conn.send((
            "top_communities",
            [u.bytes for u in assignment.top_communities],
        ))
        mid_payload = [
            (comm.bytes, [m.bytes for m in members])
            for comm, members in assignment.mid_regions.items()
        ]
        conn.send(("mid_regions", mid_payload))
        conn.send(("modularity", float(assignment.modularity)))

        # Optional centrality on the SAME graph — one child, both results.
        if with_centrality:
            _stream_centrality(conn, graph)

        conn.send(("done", None))
    except RuntimeError:
        # Protocol violation — re-raise so the in-process driver tests see it.
        # The subprocess wrapper below converts this into ("error", ...) + exit(1).
        try:
            import traceback
            err = traceback.format_exc(limit=2)[:500]
            conn.send(("error", err))
        except Exception:  # noqa: BLE001
            pass
        raise
    except Exception as exc:  # noqa: BLE001
        try:
            conn.send(("error", repr(exc)[:500]))
        except Exception:  # noqa: BLE001
            pass
        sys.exit(1)
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit("not directly invocable")

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import sys
import time
from pathlib import Path

_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
_ROOT_PATH = str(Path(__file__).resolve().parent.parent)
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)


D_CUES = [
    "what did we cover about auth yesterday?",
    "explain the db migration plan",
    "how does the web cache invalidation work",
    "summary of the cli subcommand changes",
    "recent network stack bug report",
]

# The columns RankCandidateView / _from_row_rank_view actually reads, for
# empirical narrowing comparisons run against the shipped SELECT *.
_NARROWED_COLS = [
    "vec_label", "id", "tier", "literal_surface", "aaak_index", "embedding",
    "structure_hv", "community_id", "stability", "created_at", "tags_json",
    "language", "schema_version", "salience_level",
]


def _percentile(values: "list[float]", pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(len(s) * pct)))
    return float(s[idx])


def _build_synthetic_store(n: int, seed: int, path: Path):
    from bench.neural_map import _BenchEmbedder, _make_record
    from iai_mcp.store import MemoryStore, flush_edge_buffer, flush_record_buffer

    store = MemoryStore(path=path)
    embedder = _BenchEmbedder(base_seed=seed, dim=store.embed_dim)
    if store.active_records_count() < n:
        tag_pool = [
            ["topic:auth"], ["topic:db"], ["topic:web"],
            ["topic:net"], ["topic:cli"],
        ]
        for i in range(n):
            vec = embedder.embed(f"seed-{i}")
            tags = list(tag_pool[i % len(tag_pool)])
            rec = _make_record(vec, text=f"synthetic fact {i}", tags=tags)
            store.insert(rec)
        try:
            flush_record_buffer(store)
            flush_edge_buffer(store)
        except Exception:
            pass
    return store, embedder


def _open_real_clone(clone_root: Path):
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=clone_root, read_only=True)
    return store


def _representative_vec(store, embedder, cue: str) -> "list[float]":
    return embedder.embed(cue)


def _raw_fetch_for_labels(store, labels: "list[int]", select_cols: "list[str] | None"):
    """Mirror HippoQuery._ann_knn_fetch_core's chunked vec_label IN fetch, at
    a caller-chosen column projection, so the SAME production decode loop can
    be timed over a SELECT * fetch and a narrowed-column fetch of the exact
    same rows without touching production source.
    """
    from iai_mcp.hippo._table import _IN_LIST_CHUNK

    col_clause = ", ".join(select_cols) if select_cols else "*"
    seen: set = set()
    dedup = [lbl for lbl in labels if not (lbl in seen or seen.add(lbl))]
    fetched: list = []
    first_description = None
    with store.db.ro_conn() as conn:
        for start in range(0, len(dedup), _IN_LIST_CHUNK):
            chunk = dedup[start:start + _IN_LIST_CHUNK]
            placeholders = ", ".join("?" for _ in chunk)
            sql = f"SELECT {col_clause} FROM records WHERE vec_label IN ({placeholders})"
            cur = conn.execute(sql, chunk)
            rows = cur.fetchall()
            if first_description is None and cur.description is not None:
                first_description = cur.description
            fetched.extend(rows)
    return first_description, fetched


def _time_decode_substeps(store, description, fetched: "list", dist_map: "dict[int, float]", label: str) -> dict:
    """Manually time each sub-step of to_row_dicts' decode loop plus
    _from_row_rank_view, using the REAL production functions, over a REAL
    fetched row set (either driver, either column projection). Returns ms
    totals for the whole fetched set (not per-row) so callers can sum
    directly against escalation_decode_ms.
    """
    from iai_mcp.hippo._table import _decode_raw_row_embedding

    columns = [d[0] for d in description]

    t0 = time.perf_counter()
    tuples = [tuple(r) for r in fetched]
    t_tuple = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    row_dicts = [dict(zip(columns, values)) for values in tuples]
    t_dictzip = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    decoded = [_decode_raw_row_embedding(r) for r in row_dicts]
    t_decode_embed = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    for row in decoded:
        vec_label = row.get("vec_label")
        row["_distance"] = (
            dist_map.get(int(vec_label), float("nan")) if vec_label is not None else float("nan")
        )
    t_distmap = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    decoded.sort(key=lambda r: (r["_distance"] != r["_distance"], r["_distance"]))
    t_sort = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    rank_views = []
    decode_errors = 0
    for row in decoded:
        try:
            rank_views.append(store._from_row_rank_view(row))
        except Exception:
            decode_errors += 1
    t_rank_view = (time.perf_counter() - t0) * 1000.0

    total_to_row_dicts_equiv = t_tuple + t_dictzip + t_decode_embed + t_distmap + t_sort

    return {
        "label": label,
        "n_rows": len(fetched),
        "n_cols": len(columns),
        "tuple_ms": t_tuple,
        "dict_zip_ms": t_dictzip,
        "decode_embed_ms": t_decode_embed,
        "dist_map_ms": t_distmap,
        "sort_ms": t_sort,
        "to_row_dicts_equivalent_ms": total_to_row_dicts_equiv,
        "from_row_rank_view_ms": t_rank_view,
        "decode_errors": decode_errors,
        "combined_escalation_decode_equivalent_ms": total_to_row_dicts_equiv + t_rank_view,
    }


def _sql_window_labels(store, k_effective: int, window_index: int) -> "list[int]":
    """Fallback label source for a store opened read_only=True: HippoDB never
    loads the HNSW vector index on that path (hippo/_db.py sets self._hnsw =
    None unconditionally when read_only), so no ANN knn_query is possible.
    Picks a plain SQL window of live vec_labels instead — the per-row decode
    cost this script measures depends on row content/byte-width, not on
    which rows were chosen by cosine rank, so an arbitrary live-row window is
    a faithful stand-in for an ANN-selected candidate set of the same size.
    """
    where_clause = "tombstoned_at IS NULL AND COALESCE(embedding_pending, 0) = 0"
    offset = (window_index * k_effective) % max(1, store.active_records_count())
    sql = (
        f"SELECT vec_label FROM records WHERE {where_clause} "
        f"ORDER BY vec_label LIMIT {k_effective} OFFSET {offset}"
    )
    with store.db.ro_conn() as conn:
        rows = conn.execute(sql).fetchall()
    labels = [int(r["vec_label"]) for r in rows]
    if len(labels) < k_effective:
        # Window ran off the end of the table — wrap from the start so the
        # request is still satisfied at (close to) the requested size.
        remaining = k_effective - len(labels)
        sql2 = (
            f"SELECT vec_label FROM records WHERE {where_clause} "
            f"ORDER BY vec_label LIMIT {remaining}"
        )
        with store.db.ro_conn() as conn:
            rows2 = conn.execute(sql2).fetchall()
        labels.extend(int(r["vec_label"]) for r in rows2)
    return labels


def _get_ann_query_and_labels(store, vec, k_effective: int, window_index: int = 0):
    """Run the ANN branch through HippoQuery, matching query_similar's own
    where-clause and limit, and return (query, labels, dist_map) for reuse
    across a SELECT * fetch and a narrowed fetch. Falls back to a plain SQL
    label window when the store has no live HNSW index (read_only=True
    opens never load one — see _sql_window_labels).
    """
    from iai_mcp.store import RECORDS_TABLE

    tbl = store.db.open_table(RECORDS_TABLE)
    q = tbl.search(list(vec)).distance_type("cosine")
    where_clause = "tombstoned_at IS NULL AND COALESCE(embedding_pending, 0) = 0"
    q = q.where(where_clause)
    q = q.limit(k_effective)

    if getattr(store.db, "_hnsw", None) is None:
        labels = _sql_window_labels(store, k_effective, window_index)
        dist_map = {lbl: 0.0 for lbl in labels}
        return q, labels, dist_map

    core = q._ann_knn_fetch_core()
    if core is None:
        return q, [], {}
    cur, fetched, dist_map = core
    labels = [int(r["vec_label"]) for r in fetched]
    return q, labels, dist_map


def run_substep_comparison(store, embedder, k_effective: int, repeats: int, driver_label: str, corpus_label: str) -> dict:
    per_cue_results = []
    for cue_idx, cue in enumerate(D_CUES):
        vec = _representative_vec(store, embedder, cue)
        q, labels, dist_map = _get_ann_query_and_labels(store, vec, k_effective, window_index=cue_idx)
        if not labels:
            continue

        star_runs = []
        narrow_runs = []
        for _ in range(repeats):
            desc_star, fetched_star = _raw_fetch_for_labels(store, labels, None)
            star_runs.append(_time_decode_substeps(store, desc_star, fetched_star, dist_map, "select_star"))

            desc_narrow, fetched_narrow = _raw_fetch_for_labels(store, labels, _NARROWED_COLS)
            narrow_runs.append(_time_decode_substeps(store, desc_narrow, fetched_narrow, dist_map, "select_narrowed_14col"))

        per_cue_results.append({
            "cue": cue,
            "k_effective": k_effective,
            "n_labels": len(labels),
            "select_star_runs": star_runs,
            "select_narrowed_runs": narrow_runs,
        })

    def _mean(key_path, runs):
        vals = [r[key_path] for r in runs]
        return sum(vals) / len(vals) if vals else 0.0

    agg = {"select_star": {}, "select_narrowed_14col": {}}
    for variant, key in (("select_star", "select_star_runs"), ("select_narrowed_14col", "select_narrowed_runs")):
        all_runs = [r for cue_res in per_cue_results for r in cue_res[key]]
        for metric in (
            "tuple_ms", "dict_zip_ms", "decode_embed_ms", "dist_map_ms", "sort_ms",
            "to_row_dicts_equivalent_ms", "from_row_rank_view_ms",
            "combined_escalation_decode_equivalent_ms", "n_rows", "n_cols",
        ):
            agg[variant][metric] = _mean(metric, all_runs)

    return {
        "driver": driver_label,
        "corpus": corpus_label,
        "k_effective": k_effective,
        "repeats_per_cue": repeats,
        "per_cue": per_cue_results,
        "aggregate_mean": agg,
    }


def run_cprofile_decode_loop(store, embedder, k_effective: int, calls: int) -> str:
    """cProfile the exact production to_row_dicts() call, `calls` times
    across the cue set, function/line-level attribution. Falls back to the
    SQL-window label source (see _sql_window_labels) plus a manual
    tuple/dict/decode replay when the store has no live HNSW index
    (read_only=True opens never load one), so this still exercises the real
    production _decode_raw_row_embedding / driver Row materialization on
    real rows even though no ANN knn_query is possible.
    """
    profiler = cProfile.Profile()
    use_ann = getattr(store.db, "_hnsw", None) is not None

    def _body():
        from iai_mcp.hippo._table import _decode_raw_row_embedding
        for i in range(calls):
            cue = D_CUES[i % len(D_CUES)]
            vec = _representative_vec(store, embedder, cue)
            if use_ann:
                from iai_mcp.store import RECORDS_TABLE
                tbl = store.db.open_table(RECORDS_TABLE)
                q = tbl.search(list(vec)).distance_type("cosine")
                where_clause = "tombstoned_at IS NULL AND COALESCE(embedding_pending, 0) = 0"
                q = q.where(where_clause).limit(k_effective)
                q.to_row_dicts()
            else:
                labels = _sql_window_labels(store, k_effective, i)
                description, fetched = _raw_fetch_for_labels(store, labels, None)
                if description is None or not fetched:
                    continue
                columns = [d[0] for d in description]
                rows = []
                for raw_row in fetched:
                    values = tuple(raw_row)
                    row = dict(zip(columns, values))
                    row = _decode_raw_row_embedding(row)
                    row["_distance"] = 0.0
                    rows.append(row)
                rows.sort(key=lambda r: (r["_distance"] != r["_distance"], r["_distance"]))

    profiler.enable()
    _body()
    profiler.disable()

    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf)
    stats.strip_dirs()
    buf.write("\n===== sort by CUMULATIVE time (top 35) =====\n")
    stats.sort_stats("cumulative").print_stats(35)
    buf.write("\n===== sort by TOTAL (self) time (top 35) =====\n")
    stats.sort_stats("tottime").print_stats(35)
    return buf.getvalue()


def run_cprofile_full_recall(store, graph, assignment, rich_club, embedder, iterations: int) -> "tuple[str, list[float]]":
    from iai_mcp.pipeline import recall_for_response

    profiler = cProfile.Profile()
    latencies: list[float] = []

    os.environ["IAI_MCP_STAGE_PROFILE"] = "1"

    def _body():
        import random
        rng = random.Random(1234)
        for _ in range(iterations):
            cue = D_CUES[rng.randrange(len(D_CUES))]
            t0 = time.perf_counter()
            recall_for_response(
                store=store, graph=graph, assignment=assignment,
                rich_club=rich_club, embedder=embedder, cue=cue,
                session_id="bench-profile", budget_tokens=1500,
            )
            latencies.append((time.perf_counter() - t0) * 1000.0)

    profiler.enable()
    _body()
    profiler.disable()

    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf)
    stats.strip_dirs()
    buf.write("\n===== FULL RECALL PATH — sort by CUMULATIVE time (top 50) =====\n")
    stats.sort_stats("cumulative").print_stats(50)
    buf.write("\n===== FULL RECALL PATH — sort by TOTAL (self) time (top 50) =====\n")
    stats.sort_stats("tottime").print_stats(50)
    return buf.getvalue(), latencies


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", choices=["stdlib", "lilli"], required=True)
    parser.add_argument("--corpus", choices=["synthetic", "real"], required=True)
    parser.add_argument("--real-clone-path", type=str, default=None)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k-effective", type=int, default=3600)
    parser.add_argument("--substep-repeats", type=int, default=8)
    parser.add_argument("--cprofile-decode-calls", type=int, default=40)
    parser.add_argument("--cprofile-full-iterations", type=int, default=15)
    parser.add_argument("--store-path", type=str, default=None)
    parser.add_argument("--out-json", type=str, required=True)
    parser.add_argument("--out-cprofile-decode", type=str, required=True)
    parser.add_argument("--out-cprofile-full", type=str, required=True)
    args = parser.parse_args()

    os.environ["LILLI_STORAGE_DRIVER"] = args.driver

    if args.corpus == "synthetic":
        store_path = Path(args.store_path) if args.store_path else Path(
            f"/tmp/iai-bench-decode-profile-{args.driver}"
        )
        store, embedder = _build_synthetic_store(args.n, args.seed, store_path)
    else:
        if not args.real_clone_path:
            raise SystemExit("--real-clone-path required for --corpus real")
        store = _open_real_clone(Path(args.real_clone_path))
        from bench.neural_map import _BenchEmbedder
        embedder = _BenchEmbedder(base_seed=args.seed, dim=store.embed_dim)

    result: dict = {
        "driver": args.driver,
        "corpus": args.corpus,
        "k_effective": args.k_effective,
        "active_records": store.active_records_count(),
    }

    substep = run_substep_comparison(
        store, embedder, args.k_effective, args.substep_repeats, args.driver, args.corpus,
    )
    result["substep_comparison"] = substep

    decode_profile_text = run_cprofile_decode_loop(
        store, embedder, args.k_effective, args.cprofile_decode_calls,
    )
    Path(args.out_cprofile_decode).write_text(decode_profile_text)

    if args.corpus == "synthetic":
        from iai_mcp.retrieve import build_runtime_graph
        graph, assignment, rich_club = build_runtime_graph(store)
        full_profile_text, latencies = run_cprofile_full_recall(
            store, graph, assignment, rich_club, embedder, args.cprofile_full_iterations,
        )
        Path(args.out_cprofile_full).write_text(full_profile_text)
        result["full_recall_latencies_ms"] = {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "mean": sum(latencies) / len(latencies) if latencies else 0.0,
            "n": len(latencies),
        }
        result["stage_timings_last_call_ms"] = dict(
            __import__("iai_mcp.pipeline", fromlist=["_last_stage_timings_ms"])._last_stage_timings_ms
        )
    else:
        Path(args.out_cprofile_full).write_text("(skipped for real-store corpus — see decode cprofile only)\n")

    Path(args.out_json).write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: v for k, v in result.items() if k != "substep_comparison"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

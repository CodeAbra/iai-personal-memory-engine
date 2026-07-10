"""Trace which pipeline stage drops the gold session in loss cases.

Usage: python bench/lme500/debug_pipeline_loss.py <question_id> [<question_id> ...]
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np

from iai_mcp.embed import embedder_for_store
from iai_mcp.pipeline import (
    _collect_graph_pool,
    _community_gate,
    _pick_seeds,
    recall_for_benchmark,
)
from iai_mcp.retrieve import build_runtime_graph, recall as retrieve_recall
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord

from bench.adapters.longmemeval import LongMemEvalAdapter


def _make_record(content: str, session_id: str, role: str, embedding: list[float]) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=content,
        aaak_index="",
        embedding=embedding,
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
        tags=["longmemeval", f"role:{role}", f"session:{session_id}"],
        language="en",
    )


def find_row(qid: str):
    adapter = LongMemEvalAdapter()
    sessions = []
    question = None
    answer_session_ids = None
    qtype = None
    for lme_session in adapter.load_dataset(split="S"):
        q = lme_session.queries[0]
        if q["question_id"] == qid:
            sessions.append(lme_session)
            if question is None:
                question = q["query"]
                answer_session_ids = set(q.get("relevant_turn_ids", []))
                qtype = q.get("question_type", "?")
    return question, qtype, answer_session_ids, sessions


def trace_one(qid: str) -> dict:
    print(f"\n{'=' * 78}\n=== qid={qid} ===\n{'=' * 78}", flush=True)
    question, qtype, gold_session_ids, sessions = find_row(qid)
    if question is None:
        print(f"  qid={qid} NOT FOUND in dataset", flush=True)
        return {}

    print(f"  type={qtype}", flush=True)
    print(f"  question[0:120]={question[:120]!r}", flush=True)
    print(f"  gold session_ids={gold_session_ids}", flush=True)
    print(f"  haystack sessions={len(sessions)}", flush=True)

    tmp_root = Path(tempfile.mkdtemp(prefix="lme_dbg_"))
    store_dir = tmp_root / f"row-{qid}"
    store_dir.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(path=store_dir / "lancedb")
    asyncio.run(store.enable_async_writes(coalesce_ms=50, max_batch=128))
    embedder = embedder_for_store(store)

    id_to_session: dict[UUID, str] = {}
    gold_record_ids: set[UUID] = set()
    n_inserted = 0
    for sess in sessions:
        for turn in sess.turns:
            content = str(turn.get("content", "")).strip()
            if not content:
                continue
            vec = embedder.embed(content)
            rec = _make_record(
                content=content,
                session_id=sess.session_id,
                role=str(turn.get("role", "user")),
                embedding=vec,
            )
            store.insert(rec)
            id_to_session[rec.id] = sess.session_id
            if sess.session_id in gold_session_ids:
                gold_record_ids.add(rec.id)
            n_inserted += 1

    asyncio.run(store.disable_async_writes())
    print(f"  records inserted: {n_inserted}", flush=True)
    print(f"  gold records: {len(gold_record_ids)}", flush=True)

    graph, assignment, rich_club = build_runtime_graph(store)
    print(f"  graph nodes: {graph.node_count()}", flush=True)
    print(f"  communities: {len(assignment.mid_regions)}", flush=True)
    print(f"  rich-club: {len(rich_club)}", flush=True)
    cue_emb = embedder.embed(question)

    resp_x = retrieve_recall(
        store=store,
        cue_embedding=cue_emb,
        cue_text=question,
        session_id=f"debug-{qid}",
        budget_tokens=1500,
        k_hits=10,
        k_anti=0,
    )
    x_ids = [h.record_id for h in resp_x.hits]
    x_sessions = [id_to_session.get(r, "?") for r in x_ids]
    x_gold_pos = [i for i, s in enumerate(x_sessions) if s in gold_session_ids]
    print(f"\n  --- retrieve_recall (X) ---", flush=True)
    print(f"    top-10 sessions: {x_sessions}", flush=True)
    print(f"    gold hit positions: {x_gold_pos}", flush=True)

    print(f"\n  --- recall_for_benchmark (Y) stage-by-stage ---", flush=True)

    gated = _community_gate(cue_emb, assignment, top_n=3)
    candidates_set: set[UUID] = set()
    for gc in gated:
        for cid in assignment.mid_regions.get(gc, []):
            candidates_set.add(cid)
    if not candidates_set:
        candidates_set = set(graph.iter_nodes())
        print(f"    Stage 2 (community gate): EMPTY, fallback to all nodes", flush=True)
    print(f"    Stage 2 (community gate): top-3 communities = {gated}", flush=True)
    print(f"      candidates after gate: {len(candidates_set)}", flush=True)
    gold_in_gate = gold_record_ids & candidates_set
    print(f"      gold survives gate: {len(gold_in_gate)} / {len(gold_record_ids)}", flush=True)

    centrality: dict[UUID, float] = {}
    for nid in graph.iter_nodes():
        payload = graph.get_payload(nid)
        if "centrality" in payload:
            try:
                centrality[nid] = float(payload["centrality"])
            except (TypeError, ValueError):
                centrality[nid] = 0.0
    if not centrality:
        try:
            centrality = graph.centrality()
        except Exception:
            centrality = {}
    pool_ids, pool_embs = _collect_graph_pool(graph, None, store)
    cue_vec_norm = np.asarray(cue_emb, dtype=np.float32)
    cn = float(np.linalg.norm(cue_vec_norm))
    if cn > 0.0:
        cue_vec_norm = cue_vec_norm / cn
    if pool_embs.size:
        shared_cos = (pool_embs @ cue_vec_norm).astype(np.float32)
    else:
        shared_cos = np.empty(0, dtype=np.float32)
    id_to_idx = {rid: i for i, rid in enumerate(pool_ids)}
    cand_idx = np.array(
        [id_to_idx[c] for c in candidates_set if c in id_to_idx],
        dtype=np.int64,
    )
    centrality_arr = np.array(
        [centrality.get(rid, 0.0) for rid in pool_ids],
        dtype=np.float32,
    )
    seed_idx = _pick_seeds(cand_idx, shared_cos, centrality_arr, n=3)
    seeds = [pool_ids[int(i)] for i in seed_idx]
    print(f"    Stage 3 (seeds, top-3 by cosine in gated): {len(seeds)}", flush=True)
    seeds_sessions = [id_to_session.get(s, "?") for s in seeds]
    print(f"      seed sessions: {seeds_sessions}", flush=True)
    gold_in_seeds = gold_record_ids & set(seeds)
    print(f"      gold in seeds: {len(gold_in_seeds)}", flush=True)

    spread = graph.two_hop_neighborhood(seeds, top_k=5)
    reachable = set(seeds) | set(spread) | set(rich_club)
    print(f"    Stage 4 (spread + rich-club union):", flush=True)
    print(f"      seeds={len(seeds)} spread={len(spread)} rich={len(rich_club)} reachable={len(reachable)}", flush=True)
    gold_in_reachable = gold_record_ids & reachable
    print(f"      gold in reachable: {len(gold_in_reachable)} / {len(gold_record_ids)}", flush=True)

    resp_y = recall_for_benchmark(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rich_club,
        embedder=embedder,
        cue=question,
        session_id=f"debug-{qid}",
        k_hits=10,
        profile_state=None,
        turn=0,
        mode="concept",
    )
    y_ids = [h.record_id for h in resp_y.hits]
    y_sessions = [id_to_session.get(r, "?") for r in y_ids]
    y_gold_pos = [i for i, s in enumerate(y_sessions) if s in gold_session_ids]
    print(f"    Stage 5 (rank + budget pack):", flush=True)
    print(f"      final hits: {len(y_ids)}", flush=True)
    print(f"      top-10 sessions: {y_sessions}", flush=True)
    print(f"      gold hit positions: {y_gold_pos}", flush=True)

    print(f"\n  --- VERDICT ---", flush=True)
    if y_gold_pos:
        print(f"    gold present in top-10 (positions {y_gold_pos}) — no_loss", flush=True)
        if not gold_in_gate:
            print(f"      (gate would have killed it; augmentation rescued)", flush=True)
        verdict = "no_loss"
    elif not gold_in_gate:
        print(f"    >>> GOLD KILLED at STAGE 2 (community gate) — augmentation also failed <<<", flush=True)
        verdict = "stage_2_community_gate"
    elif not gold_in_reachable:
        print(f"    >>> GOLD KILLED at STAGE 3-4 (seeds + spread)  <<<", flush=True)
        print(f"      gold was {len(gold_in_gate)} candidate(s); none became "
              f"a seed and none was reached within 2 hops of the chosen seeds", flush=True)
        verdict = "stage_3_4_seeds_or_spread"
    else:
        print(f"    >>> GOLD KILLED at STAGE 5 (rank + budget pack) <<<", flush=True)
        print(f"      gold was reachable ({len(gold_in_reachable)}) but not in top-10 hits", flush=True)
        verdict = "stage_5_rank"

    return {
        "qid": qid,
        "qtype": qtype,
        "verdict": verdict,
        "n_records": n_inserted,
        "n_communities": len(assignment.mid_regions),
        "n_rich_club": len(rich_club),
        "n_gold_records": len(gold_record_ids),
        "gold_in_gate": len(gold_in_gate),
        "gold_in_reachable": len(gold_in_reachable),
        "x_gold_pos": x_gold_pos,
        "y_gold_pos": y_gold_pos,
    }


def main(qids: list[str]) -> int:
    summary = []
    for qid in qids:
        try:
            summary.append(trace_one(qid))
        except Exception as exc:
            print(f"\n  qid={qid} TRACE FAILED: {type(exc).__name__}: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            summary.append({"qid": qid, "verdict": "trace_failed"})

    print("\n\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'qid':16} {'qtype':28} {'verdict':32} gold(gate→reach)")
    print("-" * 100)
    for s in summary:
        if not s:
            continue
        gate = s.get("gold_in_gate", "?")
        reach = s.get("gold_in_reachable", "?")
        ngold = s.get("n_gold_records", "?")
        print(
            f"{s.get('qid', '?'):16} {s.get('qtype', '?'):28} "
            f"{s.get('verdict', '?'):32} "
            f"{gate}→{reach} (of {ngold})"
        )
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    sys.exit(main(sys.argv[1:]))

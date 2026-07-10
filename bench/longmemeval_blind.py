"""LongMemEval-S blind-run orchestrator: public-API retrieval, no per-dataset tuning."""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shutil
import statistics
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import sys
from pathlib import Path
_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
_ROOT_PATH = str(Path(__file__).resolve().parent.parent)
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)

from iai_mcp.embed import Embedder, embedder_for_store
from iai_mcp.pipeline import recall_for_benchmark
from iai_mcp.retrieve import build_runtime_graph, recall as retrieve_recall
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord

from bench.adapters.longmemeval import (
    DATASET_ID,
    PINNED_REVISION,
    LMESession,
    LongMemEvalAdapter,
)

from bench.tokens import _char4_count, _tiktoken_count


def _count_tokens(text: str) -> int:
    try:
        return _tiktoken_count(text)
    except Exception:  # pragma: no cover
        return _char4_count(text)


_BM25_K1: float = 1.5
_BM25_B: float = 0.75


def _tokenize(text: str) -> list[str]:
    return text.lower().replace("/", " ").split()


class _BM25Index:

    def __init__(self, docs: list[str]) -> None:
        self._n = len(docs)
        tokenized = [_tokenize(d) for d in docs]
        self._avgdl: float = (
            sum(len(t) for t in tokenized) / self._n if self._n else 1.0
        )
        self._tf: list[dict[str, int]] = []
        df: dict[str, int] = {}
        for toks in tokenized:
            counts: dict[str, int] = {}
            for tok in toks:
                counts[tok] = counts.get(tok, 0) + 1
            self._tf.append(counts)
            for tok in counts:
                df[tok] = df.get(tok, 0) + 1
        self._df = df
        self._dl: list[int] = [len(t) for t in tokenized]

    def scores(self, query: str) -> list[float]:
        qtoks = _tokenize(query)
        out = [0.0] * self._n
        n = self._n
        avgdl = self._avgdl
        k1 = _BM25_K1
        b = _BM25_B
        for tok in qtoks:
            df_t = self._df.get(tok, 0)
            if df_t == 0:
                continue
            idf = math.log((n - df_t + 0.5) / (df_t + 0.5) + 1.0)
            for idx in range(n):
                tf = self._tf[idx].get(tok, 0)
                if tf == 0:
                    continue
                dl = self._dl[idx]
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * dl / avgdl)
                out[idx] += idf * numerator / denominator
        return out


def _rrf_fuse(
    dense_pairs: list[tuple[str, float]],
    bm25_pairs: list[tuple[str, float]],
    k: int = 60,
) -> list[str]:
    rrf: dict[str, float] = {}
    for rank, (sid, _) in enumerate(dense_pairs, start=1):
        rrf[sid] = rrf.get(sid, 0.0) + 1.0 / (k + rank)
    for rank, (sid, _) in enumerate(bm25_pairs, start=1):
        rrf[sid] = rrf.get(sid, 0.0) + 1.0 / (k + rank)
    return [sid for sid, _ in sorted(rrf.items(), key=lambda x: -x[1])]


def preflight_crypto_or_exit() -> None:
    if os.environ.get("IAI_MCP_CRYPTO_PASSPHRASE"):
        return
    os.environ["IAI_MCP_CRYPTO_PASSPHRASE"] = (
        "iai-mcp-bench-falsifiability-deterministic-2026"
    )


def _percentile(xs: list[int], p: float) -> int:
    if not xs:
        return 0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((len(s) - 1) * p / 100.0))))
    return s[k]


def _make_record(
    content: str,
    session_id: str,
    role: str,
    embedding: list[float],
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    from uuid import uuid4

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
        tags=[
            "longmemeval",
            f"role:{role}",
            f"session:{session_id}",
        ],
        language="en",
    )


def _run_one_row(
    row_id: str,
    question: str,
    question_type: str,
    answer_session_ids: set[str],
    sessions: list[LMESession],
    tmp_root: Path,
    granularity: str = "turn",
    embedder_key: str = "bge-small-en-v1.5",
    run_hybrid: bool = False,
) -> dict[str, Any]:
    t0 = time.time()

    store_dir = tmp_root / f"row-{row_id}"
    store_dir.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(path=store_dir / "lancedb")

    asyncio.run(store.enable_async_writes(coalesce_ms=50, max_batch=128))

    inserted_text_tokens = 0

    embedder = Embedder(model_key=embedder_key)
    _ = embedder_for_store

    id_to_session: dict[str, str] = {}
    session_docs: list[tuple[str, str, str]] = []
    if granularity == "session":
        for sess in sessions:
            user_turns = [
                str(turn.get("content", "")).strip()
                for turn in sess.turns
                if str(turn.get("role", "user")) == "user"
                and str(turn.get("content", "")).strip()
            ]
            if not user_turns:
                continue
            doc_text = "\n".join(user_turns)
            vec = embedder.embed(doc_text)
            rec = _make_record(
                content=doc_text,
                session_id=sess.session_id,
                role="user",
                embedding=vec,
            )
            store.insert(rec)
            id_to_session[str(rec.id)] = sess.session_id
            if run_hybrid:
                session_docs.append((doc_text, sess.session_id, str(rec.id)))
            inserted_text_tokens += _count_tokens(doc_text)
    else:
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
                id_to_session[str(rec.id)] = sess.session_id
                inserted_text_tokens += _count_tokens(content)

    asyncio.run(store.disable_async_writes())
    t_after_insert = time.time()

    graph = None
    assignment = None
    rich_club = None
    try:
        graph, assignment, rich_club = build_runtime_graph(store)
    except Exception as exc:  # pragma: no cover — cache helpers should be robust
        print(
            f"[LME] row={row_id} build_runtime_graph failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
    t_after_graph = time.time()

    cue_embedding = embedder.embed(question)
    resp_x = retrieve_recall(
        store=store,
        cue_embedding=cue_embedding,
        cue_text=question,
        session_id=f"lme-{row_id}",
        budget_tokens=1500,
        k_hits=10,
        k_anti=0,
    )
    t_after_x = time.time()

    resp_y = None
    pipeline_error: str | None = None
    if graph is not None:
        try:
            resp_y = recall_for_benchmark(
                store=store,
                graph=graph,
                assignment=assignment,
                rich_club=rich_club,
                embedder=embedder,
                cue=question,
                session_id=f"lme-{row_id}",
                k_hits=10,
                profile_state=None,
                turn=0,
                mode="concept",
            )
        except Exception as exc:
            pipeline_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            print(
                f"[LME] row={row_id} recall_for_benchmark failed: "
                f"{pipeline_error}",
                file=sys.stderr,
            )
    else:
        pipeline_error = "graph_build_failed"
    t_after_y = time.time()

    def _retrieved_session_ids(resp) -> list[str]:
        if resp is None:
            return []
        out: list[str] = []
        for hit in resp.hits:
            sid = id_to_session.get(str(hit.record_id))
            if sid is not None:
                out.append(sid)
        return out

    sids_x = _retrieved_session_ids(resp_x)
    sids_y = _retrieved_session_ids(resp_y)

    def _hit_at_k(sids: list[str], k: int) -> float:
        top = sids[:k]
        return 1.0 if any(s in answer_session_ids for s in top) else 0.0

    r5_x = _hit_at_k(sids_x, 5)
    r10_x = _hit_at_k(sids_x, 10)
    r5_y = _hit_at_k(sids_y, 5) if resp_y is not None else 0.0
    r10_y = _hit_at_k(sids_y, 10) if resp_y is not None else 0.0

    r5_z: float = 0.0
    r10_z: float = 0.0
    sids_z: list[str] = []
    hybrid_error: str | None = None
    t_after_z = t_after_y
    if run_hybrid and session_docs:
        try:
            n_docs = len(session_docs)
            dense_raw = store.query_similar(cue_embedding, k=n_docs)
            if len(dense_raw) != n_docs:
                raise RuntimeError(
                    f"GUARD1 violation: query_similar returned {len(dense_raw)} "
                    f"items but n_docs={n_docs}. Dense side is partial."
                )
            dense_pairs: list[tuple[str, float]] = []
            for rec_obj, score in dense_raw:
                sid = id_to_session.get(str(rec_obj.id))
                if sid is not None:
                    dense_pairs.append((sid, score))

            bm25_idx = _BM25Index([dt for dt, _, _ in session_docs])
            bm25_scores = bm25_idx.scores(question)
            bm25_pairs: list[tuple[str, float]] = [
                (session_docs[i][1], bm25_scores[i])
                for i in range(n_docs)
                if bm25_scores[i] > 0.0
            ]
            bm25_pairs.sort(key=lambda x: -x[1])

            sids_z = _rrf_fuse(dense_pairs, bm25_pairs, k=60)
            r5_z = _hit_at_k(sids_z, 5)
            r10_z = _hit_at_k(sids_z, 10)
        except Exception as exc:
            hybrid_error = f"{type(exc).__name__}: {str(exc)[:300]}"
            print(
                f"[LME] row={row_id} hybrid prong Z failed: {hybrid_error}",
                file=sys.stderr,
            )
    t_after_z = time.time()

    query_tokens = _count_tokens(question)

    result: dict[str, Any] = {
        "question_id": row_id,
        "question_type": question_type,
        "r_at_5_retrieve": r5_x,
        "r_at_10_retrieve": r10_x,
        "r_at_5_pipeline": r5_y,
        "r_at_10_pipeline": r10_y,
        "pipeline_error": pipeline_error,
        "query_tokens": query_tokens,
        "inserted_text_tokens": inserted_text_tokens,
        "n_haystack_sessions": len(sessions),
        "n_turns_inserted": len(id_to_session),
        "timing_seconds": {
            "insert": round(t_after_insert - t0, 2),
            "graph": round(t_after_graph - t_after_insert, 2),
            "recall_retrieve": round(t_after_x - t_after_graph, 2),
            "recall_pipeline": round(t_after_y - t_after_x, 2),
            "total": round(t_after_y - t0, 2),
        },
    }
    if run_hybrid:
        result["r_at_5_hybrid"] = r5_z
        result["r_at_10_hybrid"] = r10_z
        result["hybrid_error"] = hybrid_error
        result["timing_seconds"]["recall_hybrid"] = round(t_after_z - t_after_y, 2)
        result["timing_seconds"]["total"] = round(t_after_z - t0, 2)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        default="S",
        choices=["S", "M", "oracle"],
        help="LongMemEval split",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "practical-cap on rows evaluated. LongMemEval-S = 500 rows; "
            "at ~500 turns/row and 11ms/embed on a 16GB M1 laptop, the "
            "full 500-row run is multi-hour. --limit lets the blind pilot "
            "finish; the SUMMARY discloses the cap honestly."
        ),
    )
    parser.add_argument(
        "--out",
        default="/tmp/p11_lme_full.json",
        help="output JSON path",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "JSONL checkpoint path for crash-resume; default = <out>.jsonl. "
            "Each completed (or errored) row is appended with fsync as one "
            "JSON line. On restart, rows whose question_id already appears "
            "in the checkpoint are skipped."
        ),
    )
    parser.add_argument(
        "--granularity",
        choices=["session", "turn"],
        default="session",
        help=(
            "corpus-construction granularity. "
            "'session' (default): one record per session, "
            "content = '\\n'.join(user-only turns) — matches mempalace's "
            "reference. 'turn': one record per turn (v1/v2 baseline; "
            "use with --dataset raw to reproduce v2's 0.956)."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=["cleaned", "raw"],
        default="cleaned",
        help=(
            "dataset variant. 'cleaned' (default): "
            "xiaowu0162/longmemeval-cleaned, SHA pinned via repo_info(). "
            "'raw' (v1/v2 baseline): xiaowu0162/longmemeval rev "
            "2ec2a557f339... — use with --granularity turn to reproduce "
            "v2's 0.956."
        ),
    )
    parser.add_argument(
        "--qid-include",
        default=None,
        help=(
            "comma-separated list of question_ids; if set, only these "
            "rows run (used by smoke tests for per-qid baseline "
            "verification). Applied after --limit."
        ),
    )
    parser.add_argument(
        "--embedder",
        choices=["bge-small-en-v1.5", "all-MiniLM-L6-v2"],
        default="bge-small-en-v1.5",
        help=(
            "embedder model_key. 'bge-small-en-v1.5' (default "
            "baseline) routes via the production English-only embedder. "
            "'all-MiniLM-L6-v2' (ablation) is mempalace's "
            "ChromaDB default — bench-only swap, production unchanged."
        ),
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        default=False,
        help=(
            "Enable prong Z: dense+BM25 RRF hybrid (bench-only variant, "
            "NOT the product default). Uses in-bench BM25 (k1=1.5, b=0.75) "
            "fused with FULL dense ranking via RRF (k=60, parameter-free). "
            "Reports r_at_5_hybrid / r_at_10_hybrid alongside raw prong X. "
            "Only meaningful with --granularity session."
        ),
    )

    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Keep the existing checkpoint as-is even if it contains rows "
            "classified as ERROR. Without this flag, a checkpoint containing "
            "prior errors is auto-cleaned (the default; errors are usually a "
            "fail-fast crypto / env miss and the right action is to fix the "
            "environment and rerun from scratch). Mutually exclusive with "
            "--fresh."
        ),
    )
    resume_group.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Force-clean the checkpoint even if it contains no errors. "
            "Mutually exclusive with --resume."
        ),
    )
    args = parser.parse_args(argv)

    preflight_crypto_or_exit()

    run_hybrid: bool = args.hybrid
    print(
        f"[LME] blind run starting "
        f"split={args.split} limit={args.limit} "
        f"granularity={args.granularity} dataset={args.dataset} "
        f"embedder={args.embedder} "
        f"hybrid={run_hybrid} "
        f"out={args.out}",
        file=sys.stderr,
        flush=True,
    )

    if args.dataset == "cleaned":
        from bench.adapters.longmemeval_cleaned import (
            CLEANED_DATASET_ID,
            CleanedLongMemEvalAdapter,
        )
        adapter = CleanedLongMemEvalAdapter()
        dataset_id_emit = CLEANED_DATASET_ID
        revision_emit = adapter.revision
    else:
        adapter = LongMemEvalAdapter()
        dataset_id_emit = DATASET_ID
        revision_emit = PINNED_REVISION
    grouped: dict[str, dict[str, Any]] = {}
    row_order: list[str] = []
    for lme_session in adapter.load_dataset(split=args.split):
        q = lme_session.queries[0]
        qid = q["question_id"]
        if qid not in grouped:
            grouped[qid] = {
                "question": q["query"],
                "question_type": q.get("question_type", "unknown"),
                "answer_session_ids": set(q.get("relevant_turn_ids", [])),
                "sessions": [],
            }
            row_order.append(qid)
        grouped[qid]["sessions"].append(lme_session)

    if args.limit is not None:
        row_order = row_order[: args.limit]

    if args.qid_include is not None:
        wanted = {q.strip() for q in str(args.qid_include).split(",") if q.strip()}
        row_order = [qid for qid in row_order if qid in wanted]
        print(
            f"[LME] qid-include filter: kept {len(row_order)} of "
            f"{len(wanted)} requested qids",
            file=sys.stderr,
            flush=True,
        )

    tmp_root = Path(tempfile.mkdtemp(prefix="lme_blind_"))
    print(f"[LME] per-row stores rooted at {tmp_root}", file=sys.stderr, flush=True)

    per_row: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    r5_x_values: list[float] = []
    r10_x_values: list[float] = []
    r5_y_values: list[float] = []
    r10_y_values: list[float] = []
    r5_z_values: list[float] = []
    r10_z_values: list[float] = []
    query_tokens: list[int] = []
    session_tokens: list[int] = []

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(str(args.out) + ".jsonl")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_path.exists():
        prior_error_count = 0
        prior_total_count = 0
        with open(checkpoint_path, "r", encoding="utf-8") as cp_dispo:
            for line in cp_dispo:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prior_total_count += 1
                if (
                    rec.get("classification") == "ERROR"
                    or "error" in rec
                ):
                    prior_error_count += 1

        if args.fresh:
            print(
                f"[LME] --fresh: discarding {prior_total_count}-row "
                f"checkpoint at {checkpoint_path}",
                file=sys.stderr,
                flush=True,
            )
            checkpoint_path.unlink()
        elif prior_error_count > 0 and not args.resume:
            print(
                f"[LME] Resuming from prior run with {prior_error_count} "
                f"errors; starting fresh. Pass --resume to keep checkpoint.",
                file=sys.stderr,
                flush=True,
            )
            checkpoint_path.unlink()

    completed_ids: set[str] = set()
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as cp_f:
            for line in cp_f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    print(
                        f"[LME] WARN: skipping corrupt checkpoint line: {line[:80]!r}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                qid = rec.get("question_id")
                if not qid:
                    continue
                completed_ids.add(qid)
                is_error_row = (
                    rec.get("classification") == "ERROR"
                    or (
                        "error" in rec and isinstance(rec.get("error"), dict)
                    )
                )
                if is_error_row:
                    errors.append(
                        {
                            "question_id": qid,
                            "error_class": rec["error"].get("error_class", "Unknown"),
                            "error": rec["error"].get("error", ""),
                        }
                    )
                    r5_x_values.append(0.0)
                    r10_x_values.append(0.0)
                    r5_y_values.append(0.0)
                    r10_y_values.append(0.0)
                    if run_hybrid:
                        r5_z_values.append(0.0)
                        r10_z_values.append(0.0)
                    query_tokens.append(0)
                    session_tokens.append(0)
                else:
                    per_row.append(rec)
                    r5_x_values.append(float(rec.get("r_at_5_retrieve", 0.0)))
                    r10_x_values.append(float(rec.get("r_at_10_retrieve", 0.0)))
                    r5_y_values.append(float(rec.get("r_at_5_pipeline", 0.0)))
                    r10_y_values.append(float(rec.get("r_at_10_pipeline", 0.0)))
                    if run_hybrid:
                        r5_z_values.append(float(rec.get("r_at_5_hybrid", 0.0)))
                        r10_z_values.append(float(rec.get("r_at_10_hybrid", 0.0)))
                    query_tokens.append(int(rec.get("query_tokens", 0)))
                    session_tokens.append(int(rec.get("inserted_text_tokens", 0)))
    if completed_ids:
        print(
            f"[LME] resume: {len(completed_ids)} rows already in checkpoint "
            f"{checkpoint_path}; processing {len(row_order) - len(completed_ids)} remaining",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            f"[LME] checkpoint: writing per-row durable JSONL to {checkpoint_path}",
            file=sys.stderr,
            flush=True,
        )

    def _checkpoint_append(rec: dict[str, Any]) -> None:
        with open(checkpoint_path, "a", encoding="utf-8") as cp_a:
            cp_a.write(json.dumps(rec) + "\n")
            cp_a.flush()
            os.fsync(cp_a.fileno())

    run_t0 = time.time()
    for i, qid in enumerate(row_order):
        if qid in completed_ids:
            continue
        row = grouped[qid]
        try:
            res = _run_one_row(
                row_id=qid,
                question=row["question"],
                question_type=row["question_type"],
                answer_session_ids=row["answer_session_ids"],
                sessions=row["sessions"],
                tmp_root=tmp_root,
                granularity=args.granularity,
                embedder_key=args.embedder,
                run_hybrid=run_hybrid,
            )
            per_row.append(res)
            r5_x_values.append(res["r_at_5_retrieve"])
            r10_x_values.append(res["r_at_10_retrieve"])
            r5_y_values.append(res["r_at_5_pipeline"])
            r10_y_values.append(res["r_at_10_pipeline"])
            if run_hybrid:
                r5_z_values.append(res.get("r_at_5_hybrid", 0.0))
                r10_z_values.append(res.get("r_at_10_hybrid", 0.0))
            query_tokens.append(res["query_tokens"])
            session_tokens.append(res["inserted_text_tokens"])
            res_with_class = dict(res)
            res_with_class["classification"] = "SUCCESS"
            _checkpoint_append(res_with_class)
            elapsed = time.time() - run_t0
            hybrid_suffix = (
                f" R@5_z={res.get('r_at_5_hybrid', 0.0):.0f}"
                if run_hybrid else ""
            )
            print(
                f"[LME] row {i+1}/{len(row_order)} qid={qid} "
                f"qtype={res['question_type']} "
                f"R@5_x={res['r_at_5_retrieve']:.0f} R@5_y={res['r_at_5_pipeline']:.0f}"
                f"{hybrid_suffix} "
                f"R@10_x={res['r_at_10_retrieve']:.0f} R@10_y={res['r_at_10_pipeline']:.0f} "
                f"t_row={res['timing_seconds']['total']:.1f}s "
                f"t_total={elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:
            err_payload = {
                "error_class": type(exc).__name__,
                "error": str(exc)[:500],
            }
            errors.append({"question_id": qid, **err_payload})
            r5_x_values.append(0.0)
            r10_x_values.append(0.0)
            r5_y_values.append(0.0)
            r10_y_values.append(0.0)
            if run_hybrid:
                r5_z_values.append(0.0)
                r10_z_values.append(0.0)
            query_tokens.append(0)
            session_tokens.append(0)
            _checkpoint_append(
                {
                    "question_id": qid,
                    "question_type": row.get("question_type", "unknown"),
                    "classification": "ERROR",
                    "error": err_payload,
                }
            )
            print(
                f"[LME] ERROR row={qid}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc(file=sys.stderr)
        finally:
            row_dir = tmp_root / f"row-{qid}"
            if row_dir.exists():
                shutil.rmtree(row_dir, ignore_errors=True)

    shutil.rmtree(tmp_root, ignore_errors=True)

    def _mean(xs: list[float]) -> float:
        return (sum(xs) / len(xs)) if xs else 0.0

    n_hits = sum(
        1
        for r in per_row
        if float(r.get("r_at_5_retrieve", 0.0)) == 1.0
    )
    n_misses = sum(
        1
        for r in per_row
        if float(r.get("r_at_5_retrieve", 0.0)) == 0.0
    )
    n_errors = len(errors)

    hybrid_stats: dict[str, Any] | None = None
    if run_hybrid and r5_z_values:
        raw_hits_5 = [
            float(r.get("r_at_5_retrieve", 0.0)) == 1.0 for r in per_row
        ]
        hybrid_hits_5 = [
            float(r.get("r_at_5_hybrid", 0.0)) == 1.0 for r in per_row
        ]

        flip_miss_to_hit = [
            r["question_id"]
            for r, rx, rz in zip(per_row, raw_hits_5, hybrid_hits_5)
            if not rx and rz
        ]
        flip_hit_to_miss = [
            r["question_id"]
            for r, rx, rz in zip(per_row, raw_hits_5, hybrid_hits_5)
            if rx and not rz
        ]

        b = len(flip_miss_to_hit)
        c = len(flip_hit_to_miss)
        net_flips = b - c

        mcnemar_p: float | None = None
        significance_verdict: str = "n/a"
        if b + c > 0:
            try:
                from scipy.stats import binomtest as _binomtest
                result_bt = _binomtest(b, n=b + c, p=0.5, alternative="two-sided")
                mcnemar_p = float(result_bt.pvalue)
                if mcnemar_p < 0.05:
                    significance_verdict = (
                        "SIGNIFICANT (p<0.05) — hybrid lift is real"
                        if net_flips > 0
                        else "SIGNIFICANT (p<0.05) — hybrid is WORSE"
                    )
                else:
                    significance_verdict = (
                        f"NOT SIGNIFICANT (p={mcnemar_p:.3f}) — "
                        f"difference is within noise"
                    )
            except ImportError:
                n_disc = b + c
                from math import comb
                prob_ge_b = sum(
                    comb(n_disc, k) * (0.5 ** n_disc)
                    for k in range(b, n_disc + 1)
                )
                mcnemar_p = float(min(1.0, 2.0 * prob_ge_b))
                significance_verdict = (
                    f"{'SIGNIFICANT' if mcnemar_p < 0.05 else 'NOT SIGNIFICANT'} "
                    f"(p={mcnemar_p:.3f}, fallback exact binomial)"
                )

        hybrid_stats = {
            "method": "dense+BM25 RRF",
            "rrf_k": 60,
            "bm25_k1": _BM25_K1,
            "bm25_b": _BM25_B,
            "tuning": "parameter-free (RRF k=60 is the fixed literature default)",
            "r_at_5_hybrid": _mean(r5_z_values),
            "r_at_10_hybrid": _mean(r10_z_values),
            "r_at_5_hybrid_hits": sum(1 for v in r5_z_values if v == 1.0),
            "r_at_5_delta_vs_raw": _mean(r5_z_values) - _mean(r5_x_values),
            "r_at_10_delta_vs_raw": _mean(r10_z_values) - _mean(r10_x_values),
            "flip_miss_to_hit_n": b,
            "flip_miss_to_hit_qids": flip_miss_to_hit,
            "flip_hit_to_miss_n": c,
            "flip_hit_to_miss_qids": flip_hit_to_miss,
            "net_flips": net_flips,
            "discordant_pairs": b + c,
            "mcnemar_p_value": mcnemar_p,
            "significance_verdict": significance_verdict,
            "label": (
                "HYBRID VARIANT (bench-only, RRF parameter-free). "
                "NOT the product default. Raw prong X is the production number."
            ),
        }

    out = {
        "split": args.split,
        "dataset_id": dataset_id_emit,
        "revision": revision_emit,
        "granularity": args.granularity,
        "dataset_choice": args.dataset,
        "embedder_model_key": args.embedder,
        "embedder_hf_id": Embedder(model_key=args.embedder).model_name,
        "n_rows": len(row_order),
        "r_at_5_retrieve": _mean(r5_x_values),
        "r_at_10_retrieve": _mean(r10_x_values),
        "r_at_5_pipeline": _mean(r5_y_values),
        "r_at_10_pipeline": _mean(r10_y_values),
        "r_at_5_lift": _mean(r5_y_values) - _mean(r5_x_values),
        "r_at_10_lift": _mean(r10_y_values) - _mean(r10_x_values),
        "token_p50": _percentile(query_tokens, 50),
        "token_p95": _percentile(query_tokens, 95),
        "session_tokens_mean": (
            statistics.fmean(session_tokens) if session_tokens else 0.0
        ),
        "errors": errors,
        "n_hits": n_hits,
        "n_misses": n_misses,
        "n_errors": n_errors,
        "hard_limit": args.limit,
        "metric_def": (
            "Session-ID hit-at-k: R@k = 1.0 if any of top-k retrieved records "
            "belongs to a gold session_id, else 0.0 (LongMemEval standard)."
        ),
        "per_row": per_row,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_wall_seconds": round(time.time() - run_t0, 2),
    }
    if hybrid_stats is not None:
        out["hybrid"] = hybrid_stats

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    hybrid_done_suffix = ""
    if hybrid_stats:
        hybrid_done_suffix = (
            f" R@5_hybrid={hybrid_stats['r_at_5_hybrid']:.3f} "
            f"delta={hybrid_stats['r_at_5_delta_vs_raw']:+.4f} "
            f"flips(b={hybrid_stats['flip_miss_to_hit_n']},c={hybrid_stats['flip_hit_to_miss_n']}) "
            f"p={hybrid_stats.get('mcnemar_p_value', 'n/a')} "
            f"verdict={hybrid_stats['significance_verdict']!r}"
        )
    print(
        f"[LME] DONE n_rows={out['n_rows']} "
        f"R@5_retrieve={out['r_at_5_retrieve']:.3f} "
        f"R@5_pipeline={out['r_at_5_pipeline']:.3f} "
        f"lift_R@5={out['r_at_5_lift']:+.3f} "
        f"R@10_retrieve={out['r_at_10_retrieve']:.3f} "
        f"R@10_pipeline={out['r_at_10_pipeline']:.3f} "
        f"lift_R@10={out['r_at_10_lift']:+.3f} "
        f"hits={n_hits} misses={n_misses} errors={n_errors}"
        f"{hybrid_done_suffix} "
        f"-> {args.out}",
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

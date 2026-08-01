from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

_HF_CACHE = Path(
    os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface")
)
HAS_LONGMEMEVAL_CACHE = any(_HF_CACHE.rglob("longmemeval_s")) if _HF_CACHE.exists() else False
HAS_BGE_SMALL_CACHE = any(_HF_CACHE.rglob("*bge-small-en*")) if _HF_CACHE.exists() else False

def _make_record(content: str, session_id: str, role: str, embedding: list[float]):
    from iai_mcp.types import MemoryRecord
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
        tags=["lme_synthetic", f"role:{role}", f"session:{session_id}"],
        language="en",
    )

def _r_at_k_session_ids(retrieved_record_ids, id_to_session, gold_session_ids, k):
    retrieved_sessions = [id_to_session.get(rid, "?") for rid in retrieved_record_ids[:k]]
    return 1.0 if any(s in gold_session_ids for s in retrieved_sessions) else 0.0

@pytest.mark.skipif(
    not HAS_BGE_SMALL_CACHE,
    reason="bge-small-en-v1.5 model not cached locally; synthetic fence requires real embeddings",
)
@pytest.mark.parametrize(
    "n_haystack,n_gold_session,gold_session_count,cue_text,gold_text_template",
    [
        (60, 1, 4, "what did I tell you about my dog Rex on Tuesday?",
         "I have a dog named Rex who is a golden retriever and loves the park"),
        (120, 3, 12, "tell me about the Python build error I was debugging",
         "The build error was traced to a missing __init__.py in the bench module"),
        (80, 1, 3, "what coffee preference did I share?",
         "My favorite coffee is a single-origin Ethiopian pour-over with no milk"),
    ],
    ids=["single-session-user", "multi-session", "single-session-preference"],
)
def test_synthetic_pipeline_no_regression_vs_baseline(
    tmp_path,
    n_haystack,
    n_gold_session,
    gold_session_count,
    cue_text,
    gold_text_template,
):
    import asyncio
    from iai_mcp.embed import embedder_for_store
    from iai_mcp.pipeline import recall_for_benchmark
    from iai_mcp.retrieve import build_runtime_graph, recall as retrieve_recall
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "lancedb")
    asyncio.run(store.enable_async_writes(coalesce_ms=50, max_batch=128))
    embedder = embedder_for_store(store)

    id_to_session: dict[UUID, str] = {}
    gold_record_ids: set[UUID] = set()
    gold_session_ids: set[str] = set()

    try:
        for gs_idx in range(n_gold_session):
            session_id = f"gold-{gs_idx:03d}"
            gold_session_ids.add(session_id)
            for k in range(gold_session_count):
                content = f"{gold_text_template} (turn {k} session {gs_idx})"
                vec = embedder.embed(content)
                rec = _make_record(content, session_id, role="user", embedding=vec)
                store.insert(rec)
                id_to_session[rec.id] = session_id
                gold_record_ids.add(rec.id)

        distractor_topics = [
            "I went to the grocery store today and bought apples",
            "The weather has been rainy all week long here",
            "I am learning to play the piano this year",
            "My favorite TV show is about cooking competitions",
            "The new garden tools arrived in the mail yesterday",
            "I read an interesting book about ancient Rome recently",
            "The car needs an oil change next month sometime",
            "I decided to repaint the bedroom walls light blue",
            "My friend recommended a great Italian restaurant nearby",
            "I found an old photograph from my college years today",
        ]
        for i in range(n_haystack):
            session_id = f"distractor-{i // 3:04d}"
            content = distractor_topics[i % len(distractor_topics)] + f" (#{i})"
            vec = embedder.embed(content)
            rec = _make_record(content, session_id, role="user", embedding=vec)
            store.insert(rec)
            id_to_session[rec.id] = session_id
    finally:
        asyncio.run(store.disable_async_writes())

    graph, assignment, rich_club = build_runtime_graph(store)

    cue_emb = embedder.embed(cue_text)

    resp_x = retrieve_recall(
        store=store,
        cue_embedding=cue_emb,
        cue_text=cue_text,
        session_id="loss-fence-x",
        budget_tokens=1500,
        k_hits=10,
        k_anti=0,
    )
    x_record_ids = [h.record_id for h in resp_x.hits]
    r5_x = _r_at_k_session_ids(x_record_ids, id_to_session, gold_session_ids, 5)
    r10_x = _r_at_k_session_ids(x_record_ids, id_to_session, gold_session_ids, 10)

    resp_y = recall_for_benchmark(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rich_club,
        embedder=embedder,
        cue=cue_text,
        session_id="loss-fence-y",
        k_hits=10,
        profile_state=None,
        turn=0,
        mode="concept",
    )
    y_record_ids = [h.record_id for h in resp_y.hits]
    r5_y = _r_at_k_session_ids(y_record_ids, id_to_session, gold_session_ids, 5)
    r10_y = _r_at_k_session_ids(y_record_ids, id_to_session, gold_session_ids, 10)

    assert r5_y >= r5_x, (
        f"recall_for_benchmark R@5 ({r5_y}) regressed against retrieve_recall R@5 ({r5_x}); "
        f"this is exactly the historical loss-of-QIDs regression. "
        f"Y record_ids: {y_record_ids[:5]}; X record_ids: {x_record_ids[:5]}; "
        f"gold_sessions: {gold_session_ids}; n_gold_records: {len(gold_record_ids)}; "
        f"n_communities: {len(assignment.mid_regions)}"
    )
    assert r10_y >= r10_x, (
        f"recall_for_benchmark R@10 ({r10_y}) regressed against retrieve_recall R@10 ({r10_x})"
    )

@pytest.mark.slow
@pytest.mark.skipif(
    not (HAS_LONGMEMEVAL_CACHE and HAS_BGE_SMALL_CACHE),
    reason="LongMemEval-S dataset or bge-small-en-v1.5 embedder not cached locally",
)
def test_real_qids_smoke_no_loss_verdict():
    qids = [
        "726462e0",
        "06f04340",
        "38146c39",
        "d3ab962e",
        "8e91e7d9",
        "gpt4_b0863698",
        "9a707b82",
    ]
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "bench" / "lme500" / "debug_pipeline_loss.py"
    assert script.exists(), f"missing script: {script}"

    env = dict(os.environ)
    env.setdefault("PYTHONPATH", f"{repo_root / 'src'}:{repo_root}")
    env["TRANSFORMERS_VERBOSITY"] = "error"

    proc = subprocess.run(
        [sys.executable, str(script), *qids],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=1200,
        env=env,
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    assert proc.returncode == 0, (
        f"debug_pipeline_loss.py exited rc={proc.returncode}\n"
        f"--- stderr ---\n{stderr[-2000:]}"
    )

    verdicts: dict[str, str] = {}
    for ln in stdout.splitlines():
        for q in qids:
            if ln.startswith(q):
                for tok in ln.split():
                    if tok in (
                        "no_loss",
                        "stage_2_community_gate",
                        "stage_3_4_seeds_or_spread",
                        "stage_5_rank",
                        "trace_failed",
                    ):
                        verdicts[q] = tok
                        break
                break

    assert len(verdicts) == 7, (
        f"expected verdicts for all 7 qids; got {verdicts}\n"
        f"--- stdout tail ---\n{stdout[-3000:]}"
    )
    for qid in qids:
        assert verdicts[qid] == "no_loss", (
            f"qid={qid} expected 'no_loss' (post-redesign); "
            f"got {verdicts[qid]!r}. Full verdicts={verdicts}.\n"
            f"--- stdout tail ---\n{stdout[-3000:]}"
        )

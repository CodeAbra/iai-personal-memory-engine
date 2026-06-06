"""LongMemEval adapter — external-bench gate.

Wires the public LongMemEval memory benchmark (Xie et al., 2024) into the
IAI-MCP public API (MemoryStore.insert + retrieve.recall). Strict blind-run
discipline: no per-dataset tuning, no field-mapping optimisation, no
embedder finetune. The adapter is the ONLY translation layer; everything
downstream is stock IAI-MCP.

## Dataset source

The plan text (05-11-) cites ``lxucs/longmemeval`` — that repo does
NOT exist on HuggingFace Hub (returns 401/Not Found). The canonical public
mirror shipped by the paper authors is ``xiaowu0162/longmemeval``.
Discovered mid-execution; documented as a Rule 3 deviation in the SUMMARY. DATASET_ID points at the live mirror; PINNED_REVISION is
the 40-char commit hash resolved at execution time so numbers reproduce.

## Row schema (longmemeval_s split, 500 rows)

Each row is:

    {
      "question_id": str (8-hex),
      "question_type": str (single-session-user, multi-session,...),
      "question": str,
      "answer": str,
      "question_date": str ("YYYY/MM/DD (Day) HH:MM"),
      "haystack_dates": list[str],
      "haystack_session_ids": list[str] # len ~54
      "haystack_sessions": list[list[{"role","content"}]]
      "answer_session_ids": list[str] # gold evidence (len typically 1)
    }

## LMESession mapping (deviation, Rule 1/3)

The plan's interface says "one session -> many queries". The actual dataset
is "one query -> many haystack sessions". We therefore flatten each row to
a list of LMESession objects — one per haystack session — with the single
eval query attached to every session in the row (so
bench/longmemeval_blind.py can iterate LMESessions, insert haystack turns,
and run the query against the store). The orchestrator (not the adapter)
scores at the standard LongMemEval session-ID granularity.

The ``score_r_at_k`` method in this module implements the plan's literal
formula ``|retrieved ∩ relevant| / |relevant|`` over UUIDs — it is unit-
testable and matches the Test 4 contract. The orchestrator also
reports session-level R@k using the dataset's native session_id gold.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from uuid import UUID, uuid4

# Local imports kept lazy-friendly by using a distinct alias so tests can
# mock ``bench.adapters.longmemeval.retrieve_recall`` without touching the
# production retrieve module wholesale.
from iai_mcp.retrieve import recall as retrieve_recall
from iai_mcp.embed import embedder_for_store
from iai_mcp.types import MemoryRecord


DATASET_ID: str = "xiaowu0162/longmemeval"
# Pinned against the canonical LongMemEval HuggingFace mirror.
# Reproducers MUST load this exact revision or disclose the drift.
PINNED_REVISION: str = "2ec2a557f339b6c0369619b1ed5793734cc87533"
# Split -> filename (the repo ships configs ``longmemeval_s``,
# ``longmemeval_m``, ``longmemeval_oracle``). runs the S split.
_SPLIT_FILENAMES: dict[str, str] = {
    "S": "longmemeval_s",
    "M": "longmemeval_m",
    "oracle": "longmemeval_oracle",
}


@dataclass
class LMESession:
    """One flattened haystack session + its attached eval query.

    See module docstring for why this differs from the plan's original
    "one session many queries" spec.
    """

    session_id: str
    turns: list[dict]  # [{"role": "user"|"assistant", "content": str}]
    queries: list[dict]  # [{"query": str, "relevant_turn_ids": list[str]}]


class LongMemEvalAdapter:
    """Public API: load_dataset / session_to_inserts / query_to_recall /
    score_r_at_k."""

    DATASET_ID: str = DATASET_ID
    PINNED_REVISION: str = PINNED_REVISION

    def __init__(self, revision: str | None = None) -> None:
        self.revision = revision or self.PINNED_REVISION

    # --------------------------------------------------------------- load

    def load_dataset(self, split: str = "S") -> Iterable[LMESession]:
        """Stream LMESessions out of the LongMemEval-<split> JSON file.

        Uses ``huggingface_hub.hf_hub_download`` to grab the split file at
        the pinned revision (the datasets library's JSON auto-detection
        breaks on this repo because the files ship without a ``.json``
        extension — see README). Falls back to raising a clear error if
        HuggingFace is unreachable and nothing is cached.
        """
        import json

        filename = _SPLIT_FILENAMES.get(split)
        if filename is None:
            raise ValueError(
                f"unknown LongMemEval split {split!r}; "
                f"expected one of {sorted(_SPLIT_FILENAMES)}"
            )

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover — dev extra
            raise RuntimeError(
                "huggingface_hub not installed; run "
                "`pip install 'datasets>=2.18' huggingface_hub`"
            ) from exc

        print(
            f"[LongMemEval] resolving split={split} "
            f"revision={self.revision} filename={filename}",
            file=sys.stderr,
            flush=True,
        )
        path = hf_hub_download(
            repo_id=self.DATASET_ID,
            filename=filename,
            repo_type="dataset",
            revision=self.revision,
        )
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)

        for row in rows:
            qid = row["question_id"]
            question = row["question"]
            # bench/lme500: capture question_type for per-type breakdown.
            question_type = str(row.get("question_type", "unknown"))
            answer_session_ids = list(row.get("answer_session_ids", []))
            haystack_session_ids: list[str] = list(
                row.get("haystack_session_ids", [])
            )
            haystack_sessions: list[list[dict]] = list(
                row.get("haystack_sessions", [])
            )

            # Emit one LMESession per haystack session; attach the eval
            # query to every one so the orchestrator can run ONE recall
            # per row after inserting all haystack turns.
            #
            # The "relevant_turn_ids" field stays session-id-based (the
            # paper's native gold). We record which session is "gold" so
            # the orchestrator can score hits.
            for sess_id, turns in zip(
                haystack_session_ids, haystack_sessions
            ):
                yield LMESession(
                    session_id=sess_id,
                    turns=list(turns),
                    queries=[
                        {
                            "query": question,
                            "question_id": qid,
                            "question_type": question_type,
                            # Gold at session granularity; the orchestrator
                            # decides how to use it. score_r_at_k in this
                            # adapter takes whatever the caller passes.
                            "relevant_turn_ids": answer_session_ids,
                            "is_gold_session": sess_id in answer_session_ids,
                        }
                    ],
                )

    # ------------------------------------------------------- session_to_inserts

    def session_to_inserts(self, session: LMESession) -> list[MemoryRecord]:
        """Map each turn to one MemoryRecord (tier=episodic, literal_surface=content).

        Produces a placeholder embedding sized to the default embed dim.
        The blind-run orchestrator overrides the embedding with the real
        one from ``embedder_for_store(store).embed(text)`` before calling
        ``store.insert`` — this keeps ``session_to_inserts`` cheap for
        unit tests that don't want to load sentence-transformers.
        """
        from iai_mcp.embed import Embedder

        dim = Embedder.DEFAULT_DIM
        records: list[MemoryRecord] = []
        now = datetime.now(timezone.utc)
        for turn in session.turns:
            content = str(turn.get("content", ""))
            rec = MemoryRecord(
                id=uuid4(),
                tier="episodic",
                literal_surface=content,
                aaak_index="",
                embedding=[0.0] * dim,  # placeholder; orchestrator overrides
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
                    f"role:{turn.get('role','user')}",
                    f"session:{session.session_id}",
                ],
                language="en",
            )
            records.append(rec)
        return records

    # ------------------------------------------------------- query_to_recall

    def query_to_recall(self, query: dict, store) -> list[UUID]:
        """Call retrieve.recall(cue_text=query['query'], k_hits=10).

        Returns the retrieved record ids in rank order. The orchestrator
        uses these ids to compute R@k.
        """
        cue_text = str(query["query"])
        embedder = embedder_for_store(store)
        cue_embedding = embedder.embed(cue_text)
        resp = retrieve_recall(
            store=store,
            cue_embedding=cue_embedding,
            cue_text=cue_text,
            session_id="longmemeval-blind",
            budget_tokens=1500,
            k_hits=10,
            k_anti=0,
        )
        return [hit.record_id for hit in resp.hits]

    # ------------------------------------------------------- score_r_at_k

    def score_r_at_k(
        self,
        retrieved_ids: list,
        gold_turn_ids: list,
        k: int = 5,
    ) -> float:
        """R@k = |retrieved_top_k ∩ relevant| / |relevant|.

        Empty ``gold_turn_ids`` returns 1.0 (convention — avoids div-by-zero
        and matches the "no evidence to miss" semantics).

        Both lists are normalised to ``str`` so UUID vs session-id ids work.
        """
        if not gold_turn_ids:
            return 1.0
        top_k = retrieved_ids[: max(0, int(k))]
        gold_set = {str(g) for g in gold_turn_ids}
        hit = sum(1 for rid in top_k if str(rid) in gold_set)
        return hit / float(len(gold_set))

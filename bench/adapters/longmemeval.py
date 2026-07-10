from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable
from uuid import UUID, uuid4

from iai_mcp.retrieve import recall as retrieve_recall
from iai_mcp.embed import embedder_for_store
from iai_mcp.types import MemoryRecord


DATASET_ID: str = "xiaowu0162/longmemeval"
PINNED_REVISION: str = "2ec2a557f339b6c0369619b1ed5793734cc87533"
_SPLIT_FILENAMES: dict[str, str] = {
    "S": "longmemeval_s",
    "M": "longmemeval_m",
    "oracle": "longmemeval_oracle",
}


@dataclass
class LMESession:

    session_id: str
    turns: list[dict]
    queries: list[dict]


class LongMemEvalAdapter:

    DATASET_ID: str = DATASET_ID
    PINNED_REVISION: str = PINNED_REVISION

    def __init__(self, revision: str | None = None) -> None:
        self.revision = revision or self.PINNED_REVISION


    def load_dataset(self, split: str = "S") -> Iterable[LMESession]:
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
            question_type = str(row.get("question_type", "unknown"))
            answer_session_ids = list(row.get("answer_session_ids", []))
            haystack_session_ids: list[str] = list(
                row.get("haystack_session_ids", [])
            )
            haystack_sessions: list[list[dict]] = list(
                row.get("haystack_sessions", [])
            )

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
                            "relevant_turn_ids": answer_session_ids,
                            "is_gold_session": sess_id in answer_session_ids,
                        }
                    ],
                )


    def session_to_inserts(self, session: LMESession) -> list[MemoryRecord]:
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
                embedding=[0.0] * dim,
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


    def query_to_recall(self, query: dict, store) -> list[UUID]:
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


    def score_r_at_k(
        self,
        retrieved_ids: list,
        gold_turn_ids: list,
        k: int = 5,
    ) -> float:
        if not gold_turn_ids:
            return 1.0
        top_k = retrieved_ids[: max(0, int(k))]
        gold_set = {str(g) for g in gold_turn_ids}
        hit = sum(1 for rid in top_k if str(rid) in gold_set)
        return hit / float(len(gold_set))

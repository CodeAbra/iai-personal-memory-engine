"""Foresight precision/recall over the labelled real-thread fixture.

Replays every labelled cue as a live conversational turn against a read-only
COPY of the operator's own store and scores what the next-turn pack would
have injected. Two numbers matter:

- pack-hit-rate: cues whose pack contains at least one labelled relevant
  record (the anticipation found the thread);
- labelled-precision: of the injected items, how many are labelled relevant
  for their cue (labels are not exhaustive, so this is a floor, reported
  for trend, not gated).

Run: .venv/bin/python bench/foresight_real_eval.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
_REPO_PATH = str(Path(__file__).resolve().parent.parent)
if _REPO_PATH not in sys.path:
    sys.path.insert(0, _REPO_PATH)

from bench.recall_accuracy_real import (  # noqa: E402
    _copy_real_store,
    _open_copy_store_shared,
    fixture_path,
)


def run_eval(copy_root: "Path | None" = None) -> dict:
    from iai_mcp.embed import embedder_for_store
    from iai_mcp.foresight import refresh_pack

    fixture = json.loads(fixture_path().read_text(encoding="utf-8"))
    cues = fixture["cues"]

    if copy_root is None:
        prepared = os.environ.get("IAI_MCP_FORESIGHT_EVAL_STORE")
        if prepared:
            # A pre-copied store (made outside any HOME-sandboxed context).
            copy_root = Path(prepared).expanduser()
        else:
            copy_root = Path(tempfile.mkdtemp(prefix="foresight-eval-"))
            _copy_real_store(copy_root)
    store = _open_copy_store_shared(copy_root)

    embedder = embedder_for_store(store)
    hits = 0
    injected_total = 0
    injected_relevant = 0
    empty = 0
    per_cue: list[dict] = []
    try:
        for cue in cues:
            emb = embedder.embed(cue["cue_text"][:512])
            report = refresh_pack(
                store,
                cue_text=cue["cue_text"],
                cue_embedding=list(emb),
                session_id=f"foresight-eval-{cue['cue_id']}",
            )
            packed = set(report.get("packed_ids") or [])
            relevant = set(str(r) for r in cue["relevant_record_ids"])
            got = bool(packed & relevant)
            hits += int(got)
            injected_total += len(packed)
            injected_relevant += len(packed & relevant)
            empty += int(not packed)
            per_cue.append(
                {"cue_id": cue["cue_id"], "hit": got,
                 "packed": len(packed), "overlap": len(packed & relevant)}
            )
    finally:
        try:
            store.close()
        except Exception:  # noqa: BLE001
            pass

    n = len(cues)
    summary = {
        "cues": n,
        "pack_hit_rate": round(hits / n, 3) if n else 0.0,
        "empty_rate": round(empty / n, 3) if n else 0.0,
        "avg_items": round(injected_total / n, 2) if n else 0.0,
        "labelled_precision_floor": (
            round(injected_relevant / injected_total, 3) if injected_total else None
        ),
        "min_cos": os.environ.get("IAI_MCP_FORESIGHT_MIN_COS", "default"),
        "per_cue_misses": [c["cue_id"] for c in per_cue if not c["hit"]],
    }
    return summary


if __name__ == "__main__":
    result = run_eval()
    print(json.dumps(result, indent=2))

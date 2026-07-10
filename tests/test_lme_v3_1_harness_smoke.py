from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

PINNED_QID_FOR_SMOKE = "e47becba"

FIXTURES = REPO / "tests" / "fixtures"

_HF_CACHE = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
HAS_LONGMEMEVAL_CACHE = any(_HF_CACHE.rglob("longmemeval_s")) if _HF_CACHE.exists() else False
HAS_BGE_SMALL_CACHE = any(_HF_CACHE.rglob("*bge-small-en*")) if _HF_CACHE.exists() else False

pytestmark = pytest.mark.skipif(
    not (HAS_LONGMEMEVAL_CACHE and HAS_BGE_SMALL_CACHE),
    reason="LongMemEval-S dataset or bge-small-en-v1.5 model not cached",
)


def _run_harness(embedder: str, qid_filter: str = PINNED_QID_FOR_SMOKE) -> dict:
    out_path = FIXTURES / f"smoke-v3.1-{embedder}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "bench.longmemeval_blind",
        "--split",
        "S",
        "--granularity",
        "session",
        "--dataset",
        "cleaned",
        "--embedder",
        embedder,
        "--qid-include",
        qid_filter,
        "--out",
        str(out_path),
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"harness failed for --embedder {embedder}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    return json.loads(out_path.read_text())


def test_bge_small_baseline_reproduces_v3() -> None:
    out = _run_harness(embedder="bge-small-en-v1.5")
    per_row = out["per_row"]
    assert len(per_row) == 1, f"expected 1 row, got {len(per_row)}"
    row = per_row[0]
    assert row["question_id"] == PINNED_QID_FOR_SMOKE

    v3 = json.loads((REPO / "bench" / "lme500" / "output" / "lme500-v3.json").read_text())
    v3_per_row = v3["per_row"]
    v3_row = next(r for r in v3_per_row if r["question_id"] == PINNED_QID_FOR_SMOKE)

    assert row["r_at_5_pipeline"] == v3_row["r_at_5_pipeline"], (
        f"R@5 pipeline drift: smoke={row['r_at_5_pipeline']} v3={v3_row['r_at_5_pipeline']}"
    )
    assert row["r_at_10_pipeline"] == v3_row["r_at_10_pipeline"]
    assert row["r_at_5_retrieve"] == v3_row["r_at_5_retrieve"]
    assert row["r_at_10_retrieve"] == v3_row["r_at_10_retrieve"]


def test_embedder_metadata_recorded_in_output() -> None:
    bge_path = FIXTURES / "smoke-v3.1-bge-small-en-v1.5.json"
    if not bge_path.exists():
        _run_harness(embedder="bge-small-en-v1.5")
    bge_out = json.loads(bge_path.read_text())
    assert bge_out.get("embedder_model_key") == "bge-small-en-v1.5"
    assert bge_out.get("embedder_hf_id") == "BAAI/bge-small-en-v1.5"

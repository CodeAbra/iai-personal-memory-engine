"""Monotonic recall@k regression gate for the Rust forward-pass embedder."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BLIND_SCRIPT = REPO / "bench" / "longmemeval_blind.py"
DEFAULT_OUT_DIR = REPO / "bench"

PINNED_MODEL_REVISION_SHA = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
PINNED_DATASET_ID = "xiaowu0162/longmemeval-cleaned"
PINNED_DATASET_REVISION_SHA = "98d7416c24c778c2fee6e6f3006e7a073259d48f"


def _harness_output_path(backend: str, out_dir: Path) -> Path:
    return out_dir / f"_embedder_recall_compare.{backend}.raw.json"


def _compare_output_path(backend: str, out_dir: Path) -> Path:
    return out_dir / f"embedder_recall_compare.{backend}.json"


def run_backend(
    backend: str,
    n_questions: int,
    seed: int,
    out_dir: Path,
) -> dict:
    raw_out = _harness_output_path(backend, out_dir)
    cmp_out = _compare_output_path(backend, out_dir)

    if raw_out.exists():
        raw_out.unlink()

    cp = Path(str(raw_out) + ".jsonl")
    if cp.exists():
        cp.unlink()

    env = os.environ.copy()
    env["IAI_MCP_EMBED_BACKEND"] = backend
    env.setdefault("IAI_MCP_TEST_MODE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    cmd = [
        sys.executable,
        str(BLIND_SCRIPT),
        "--split", "S",
        "--limit", str(n_questions),
        "--granularity", "session",
        "--dataset", "cleaned",
        "--embedder", "bge-small-en-v1.5",
        "--out", str(raw_out),
        "--fresh",
    ]
    print(
        f"[recall_compare] backend={backend} cmd={' '.join(cmd)}",
        file=sys.stderr,
        flush=True,
    )
    t0 = time.time()
    result = subprocess.run(
        cmd,
        env=env,
        check=False,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    wall = round(time.time() - t0, 2)

    if result.returncode != 0:
        raise SystemExit(
            f"backend={backend} run failed (exit {result.returncode}). "
            f"See stderr above for the harness's own diagnostic output."
        )
    if not raw_out.exists():
        raise SystemExit(
            f"backend={backend} run completed but no output file at {raw_out}; "
            f"the harness contract may have changed."
        )

    raw = json.loads(raw_out.read_text())
    compare_doc = _to_compare_shape(
        backend=backend,
        raw=raw,
        seed=seed,
        n_questions_requested=n_questions,
        wall_seconds=wall,
    )

    cmp_out.parent.mkdir(parents=True, exist_ok=True)
    cmp_out.write_text(json.dumps(compare_doc, indent=2) + "\n")

    return compare_doc


def _to_compare_shape(
    backend: str,
    raw: dict,
    seed: int,
    n_questions_requested: int,
    wall_seconds: float,
) -> dict:
    per_row = raw.get("per_row", [])
    per_question = []
    for row in per_row:
        per_question.append(
            {
                "qid": row.get("question_id"),
                "recall_at_5": int(float(row.get("r_at_5_retrieve", 0.0))),
                "recall_at_10": int(float(row.get("r_at_10_retrieve", 0.0))),
                "question_type": row.get("question_type", "unknown"),
            }
        )
    aggregate = {
        "recall_at_5": float(raw.get("r_at_5_retrieve", 0.0)),
        "recall_at_10": float(raw.get("r_at_10_retrieve", 0.0)),
    }
    return {
        "backend": backend,
        "model_revision_sha": PINNED_MODEL_REVISION_SHA,
        "dataset_id": raw.get("dataset_id", PINNED_DATASET_ID),
        "dataset_revision_sha": raw.get("revision", PINNED_DATASET_REVISION_SHA),
        "split": raw.get("split", "S"),
        "seed": seed,
        "n_questions_requested": n_questions_requested,
        "n_questions": int(raw.get("n_rows", len(per_question))),
        "granularity": raw.get("granularity", "session"),
        "embedder_model_key": raw.get("embedder_model_key", "bge-small-en-v1.5"),
        "embedder_hf_id": raw.get("embedder_hf_id"),
        "retrieval_prong": "X_retrieve_flat_cosine",
        "per_question": per_question,
        "aggregate": aggregate,
        "n_errors": int(raw.get("n_errors", 0)),
        "wall_seconds": wall_seconds,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def assert_monotonic(rust: dict, pytorch: dict) -> list[str]:
    violations: list[str] = []
    by_qid_rust = {q["qid"]: q for q in rust["per_question"]}
    by_qid_py = {q["qid"]: q for q in pytorch["per_question"]}

    rust_qids = set(by_qid_rust)
    py_qids = set(by_qid_py)
    if rust_qids != py_qids:
        only_rust = rust_qids - py_qids
        only_py = py_qids - rust_qids
        if only_rust:
            violations.append(
                f"QID-SET DRIFT: {len(only_rust)} qids in rust missing from pytorch "
                f"(sample: {sorted(only_rust)[:5]})"
            )
        if only_py:
            violations.append(
                f"QID-SET DRIFT: {len(only_py)} qids in pytorch missing from rust "
                f"(sample: {sorted(only_py)[:5]})"
            )

    intersection = sorted(rust_qids & py_qids)
    for qid in intersection:
        r = by_qid_rust[qid]
        p = by_qid_py[qid]
        if r["recall_at_5"] < p["recall_at_5"]:
            violations.append(
                f"REGRESSION qid={qid} R@5 rust={r['recall_at_5']} "
                f"< pytorch={p['recall_at_5']} (qtype={p.get('question_type', 'unknown')})"
            )
        if r["recall_at_10"] < p["recall_at_10"]:
            violations.append(
                f"REGRESSION qid={qid} R@10 rust={r['recall_at_10']} "
                f"< pytorch={p['recall_at_10']} (qtype={p.get('question_type', 'unknown')})"
            )

    if rust["aggregate"]["recall_at_5"] < pytorch["aggregate"]["recall_at_5"]:
        violations.append(
            f"REGRESSION AGGREGATE R@5: rust={rust['aggregate']['recall_at_5']:.4f} "
            f"< pytorch={pytorch['aggregate']['recall_at_5']:.4f}"
        )
    if rust["aggregate"]["recall_at_10"] < pytorch["aggregate"]["recall_at_10"]:
        violations.append(
            f"REGRESSION AGGREGATE R@10: rust={rust['aggregate']['recall_at_10']:.4f} "
            f"< pytorch={pytorch['aggregate']['recall_at_10']:.4f}"
        )
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-questions",
        type=int,
        default=500,
        help=(
            "LongMemEval-S row cap forwarded as --limit to the blind "
            "harness. SPEC.md R8 target is 500 (full dataset). Smaller "
            "values are valid for smoke iteration but NOT for the final "
            "gate."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Audit metadata only -- the LongMemEval dataset order is "
            "deterministic from the adapter, and the embedders are "
            "deterministic per backend. The value is recorded in each "
            "output JSON for reproducibility."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory where compare JSONs are written (default: bench/).",
    )
    parser.add_argument(
        "--skip-pytorch",
        action="store_true",
        help=(
            "Reuse a previously written pytorch.json instead of re-running "
            "the PyTorch backend. Debug only -- the final gate requires "
            "both backends run from clean."
        ),
    )
    parser.add_argument(
        "--skip-rust",
        action="store_true",
        help=(
            "Reuse a previously written rust.json instead of re-running "
            "the Rust backend. Debug only -- the final gate requires "
            "both backends run from clean."
        ),
    )
    args = parser.parse_args(argv)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pytorch_compare_path = _compare_output_path("pytorch", out_dir)
    rust_compare_path = _compare_output_path("rust", out_dir)

    if args.skip_pytorch:
        if not pytorch_compare_path.exists():
            raise SystemExit(
                f"--skip-pytorch was passed but {pytorch_compare_path} does not exist."
            )
        print(
            f"[recall_compare] reusing cached PyTorch baseline at "
            f"{pytorch_compare_path}",
            file=sys.stderr,
        )
        pytorch_doc = json.loads(pytorch_compare_path.read_text())
    else:
        print(
            f"[recall_compare] running PyTorch backend (N={args.n_questions}) ...",
            file=sys.stderr,
            flush=True,
        )
        pytorch_doc = run_backend(
            backend="pytorch",
            n_questions=args.n_questions,
            seed=args.seed,
            out_dir=out_dir,
        )

    if args.skip_rust:
        if not rust_compare_path.exists():
            raise SystemExit(
                f"--skip-rust was passed but {rust_compare_path} does not exist."
            )
        print(
            f"[recall_compare] reusing cached Rust path at {rust_compare_path}",
            file=sys.stderr,
        )
        rust_doc = json.loads(rust_compare_path.read_text())
    else:
        print(
            f"[recall_compare] running Rust backend (N={args.n_questions}) ...",
            file=sys.stderr,
            flush=True,
        )
        rust_doc = run_backend(
            backend="rust",
            n_questions=args.n_questions,
            seed=args.seed,
            out_dir=out_dir,
        )

    n = rust_doc["n_questions"]
    py_agg = pytorch_doc["aggregate"]
    rs_agg = rust_doc["aggregate"]
    print(
        "\n=== Recall@k monotonicity gate "
        f"({n} LongMemEval-S questions, prong X = flat-cosine) ===",
        flush=True,
    )
    print(
        f"PyTorch  R@5={py_agg['recall_at_5']:.4f}  R@10={py_agg['recall_at_10']:.4f}  "
        f"errors={pytorch_doc.get('n_errors', 0)}"
    )
    print(
        f"Rust     R@5={rs_agg['recall_at_5']:.4f}  R@10={rs_agg['recall_at_10']:.4f}  "
        f"errors={rust_doc.get('n_errors', 0)}"
    )
    print(
        f"Delta    R@5={rs_agg['recall_at_5'] - py_agg['recall_at_5']:+.4f}  "
        f"R@10={rs_agg['recall_at_10'] - py_agg['recall_at_10']:+.4f}",
        flush=True,
    )

    violations = assert_monotonic(rust_doc, pytorch_doc)
    if violations:
        report_path = out_dir / "embedder_recall_compare.violations.txt"
        report_path.write_text("\n".join(violations) + "\n")
        print(
            f"\n*** {len(violations)} MONOTONICITY VIOLATION(S) "
            f"-- recall-parity gate FAILS ***",
            file=sys.stderr,
        )
        for v in violations[:20]:
            print(f"  {v}", file=sys.stderr)
        if len(violations) > 20:
            print(f"  ... ({len(violations) - 20} more)", file=sys.stderr)
        print(
            f"\n  Full violation report: {report_path}",
            file=sys.stderr,
        )
        print(
            "\n  Rule: "
            "results may only RISE, never DROP. "
            "Route back to the embedder forward pass to "
            "investigate FP-determinism drift on the failing qids.",
            file=sys.stderr,
        )
        return 1

    print(
        "\n*** MONOTONIC: Rust >= PyTorch on every per-question + "
        "aggregate metric ***",
        flush=True,
    )
    print(
        f"  SPEC.md Acceptance Criteria 49a items 10 & 11: CLOSED",
        flush=True,
    )
    print(
        f"  Per-question check: {n} questions, 0 regressions",
        flush=True,
    )
    print(
        f"  Outputs: {pytorch_compare_path}",
        flush=True,
    )
    print(
        f"           {rust_compare_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


V2_BASELINE_QIDS: list[tuple[str, str]] = [
    ("e47becba",        "single-session-user"),
    ("0a995998",        "multi-session"),
    ("8a2466db",        "single-session-preference"),
    ("gpt4_59149c77",   "temporal-reasoning"),
    ("6a1eabeb",        "knowledge-update"),
    ("7161e7e2",        "single-session-assistant"),
]


REPO_ROOT = Path(__file__).resolve().parent.parent
V2_JSONL = REPO_ROOT / "bench" / "lme500" / "output" / "lme500-v2.json.jsonl"

_HF_CACHE = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
HAS_LONGMEMEVAL_CACHE = (
    any(_HF_CACHE.rglob("longmemeval_s")) if _HF_CACHE.exists() else False
) and importlib.util.find_spec("huggingface_hub") is not None


@pytest.mark.skipif(
    not HAS_LONGMEMEVAL_CACHE,
    reason="LongMemEval-S HF dataset not cached or huggingface_hub not installed",
)
@pytest.mark.skipif(
    os.environ.get("IAI_MCP_SKIP_LME_V3_SMOKE") == "1",
    reason="IAI_MCP_SKIP_LME_V3_SMOKE=1; smoke is host-portable to bench host",
)
class TestV2BaselineReproduction:

    def _load_v2_truth(self) -> dict[str, dict]:
        truth: dict[str, dict] = {}
        wanted = {qid for qid, _ in V2_BASELINE_QIDS}
        with V2_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                qid = rec.get("question_id")
                if qid in wanted:
                    truth[qid] = rec
        missing = wanted - set(truth.keys())
        assert not missing, (
            f"v2 baseline JSONL missing pinned qids: {missing} — "
            f"the v2-baseline-reproduction fence is invalid"
        )
        return truth

    @pytest.mark.slow
    def test_v2_baseline_reproduction_six_qids(self, tmp_path: Path) -> None:
        truth = self._load_v2_truth()
        qid_csv = ",".join(qid for qid, _ in V2_BASELINE_QIDS)

        out_path = tmp_path / "lme_v3_smoke_v2_repro.json"
        ckpt_path = tmp_path / "lme_v3_smoke_v2_repro.jsonl"

        cmd = [
            sys.executable,
            "-m",
            "bench.longmemeval_blind",
            "--split", "S",
            "--granularity", "turn",
            "--dataset", "raw",
            "--qid-include", qid_csv,
            "--out", str(out_path),
            "--checkpoint", str(ckpt_path),
        ]

        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30 * 60,
        )
        assert proc.returncode == 0, (
            f"harness subprocess failed:\n"
            f"stdout: {proc.stdout[-2000:]}\n"
            f"stderr: {proc.stderr[-2000:]}"
        )

        assert out_path.exists(), f"output JSON not written: {out_path}"
        per_row: dict[str, dict] = {}
        with ckpt_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                qid = rec.get("question_id")
                if qid:
                    per_row[qid] = rec

        mismatches: list[str] = []
        for qid, qtype in V2_BASELINE_QIDS:
            assert qid in per_row, (
                f"qid {qid} ({qtype}) missing from v3 smoke output"
            )
            v2 = truth[qid]
            v3 = per_row[qid]
            for metric in (
                "r_at_5_retrieve",
                "r_at_10_retrieve",
                "r_at_5_pipeline",
                "r_at_10_pipeline",
            ):
                v2_val = float(v2[metric])
                v3_val = float(v3.get(metric, -1.0))
                if v2_val != v3_val:
                    mismatches.append(
                        f"qid={qid} qtype={qtype} {metric}: "
                        f"v2={v2_val} v3={v3_val}"
                    )
        assert not mismatches, (
            "v2 baseline reproduction FAILED — harness has drifted on the "
            "six pinned qids:\n  " + "\n  ".join(mismatches)
        )


class TestFourCombinationCoverage:

    def _build_parser(self):
        return None

    def _normalize_help(self, text: str) -> str:
        import re as _re
        return _re.sub(r"\s+", " ", text)

    def test_help_lists_granularity_flag_with_session_default(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "bench.longmemeval_blind", "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, f"--help failed: {proc.stderr}"
        help_text = proc.stdout
        assert "--granularity" in help_text, "--granularity flag missing from --help"
        assert "{session,turn}" in help_text, (
            "--granularity choices != {session, turn}"
        )
        flat = self._normalize_help(help_text)
        assert "'session' (default)" in flat, (
            "--granularity default disclosure ('session' (default)) "
            "missing from --help text"
        )

    def test_help_lists_dataset_flag_with_cleaned_default(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "bench.longmemeval_blind", "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, f"--help failed: {proc.stderr}"
        help_text = proc.stdout
        assert "--dataset" in help_text, "--dataset flag missing from --help"
        assert "{cleaned,raw}" in help_text, (
            "--dataset choices != {cleaned, raw}"
        )
        flat = self._normalize_help(help_text)
        assert "'cleaned' (default)" in flat, (
            "--dataset default disclosure ('cleaned' (default)) "
            "missing from --help text"
        )

    def test_help_lists_qid_include_flag(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "bench.longmemeval_blind", "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, f"--help failed: {proc.stderr}"
        help_text = proc.stdout
        assert "--qid-include" in help_text, (
            "--qid-include flag missing from --help"
        )

    @pytest.mark.skipif(
        not HAS_LONGMEMEVAL_CACHE,
        reason="LongMemEval-S HF dataset not cached or huggingface_hub not "
    "installed; the harness subprocess needs both",
    )
    def test_four_combinations_are_argparse_valid(self) -> None:
        combos = [
            ("turn", "raw"),
            ("session", "raw"),
            ("turn", "cleaned"),
            ("session", "cleaned"),
        ]
        with tempfile.TemporaryDirectory() as tdir:
            tdir_p = Path(tdir)
            for granularity, dataset in combos:
                out_path = tdir_p / f"smoke_{granularity}_{dataset}.json"
                ckpt_path = tdir_p / f"smoke_{granularity}_{dataset}.jsonl"
                cmd = [
                    sys.executable,
                    "-m", "bench.longmemeval_blind",
                    "--split", "S",
                    "--granularity", granularity,
                    "--dataset", dataset,
                    "--qid-include", "__nonexistent_qid_smoke_test__",
                    "--out", str(out_path),
                    "--checkpoint", str(ckpt_path),
                ]
                proc = subprocess.run(
                    cmd,
                    cwd=str(REPO_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=10 * 60,
                )
                assert proc.returncode == 0, (
                    f"combo granularity={granularity} dataset={dataset} "
                    f"failed:\n"
                    f"stdout: {proc.stdout[-1500:]}\n"
                    f"stderr: {proc.stderr[-1500:]}"
                )
                assert out_path.exists(), (
                    f"combo {granularity}/{dataset}: output JSON not written"
                )
                with out_path.open("r", encoding="utf-8") as f:
                    out = json.load(f)
                assert out["granularity"] == granularity, (
                    f"combo {granularity}/{dataset}: output JSON granularity "
                    f"= {out['granularity']!r}, expected {granularity!r}"
                )
                assert out["dataset_choice"] == dataset, (
                    f"combo {granularity}/{dataset}: output JSON "
                    f"dataset_choice = {out['dataset_choice']!r}, "
                    f"expected {dataset!r}"
                )
                assert out["n_rows"] == 0, (
                    f"combo {granularity}/{dataset}: expected n_rows=0 "
                    f"after no-match qid filter, got {out['n_rows']}"
                )

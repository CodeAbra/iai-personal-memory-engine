
from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path

import pytest

from bench import contradiction_longitudinal as bench


def test_probe_result_has_route_and_cue_hash_columns() -> None:
    pr = bench.ProbeResult(
        probe_id="p-001",
        seed=13,
        n_slice=0,
        condition="post_flip",
        topic="launch_date",
        pipeline_rank=1,
        cosine_rank=2,
        pipeline_hit_at_k=True,
        cosine_hit_at_k=True,
        pipeline_top1_text="some text",
        route="efe_real",
        cue_hash="deadbeef",
    )
    assert isinstance(pr.route, str)
    assert isinstance(pr.cue_hash, str)
    assert pr.route == "efe_real"
    assert pr.cue_hash == "deadbeef"

    pr_default = bench.ProbeResult(
        probe_id="p-002",
        seed=13,
        n_slice=0,
        condition="post_flip",
        topic="launch_date",
        pipeline_rank=-1,
        cosine_rank=-1,
        pipeline_hit_at_k=False,
        cosine_hit_at_k=False,
        pipeline_top1_text="",
    )
    assert pr_default.route == ""
    assert pr_default.cue_hash == ""


@pytest.mark.parametrize(
    "cue",
    [
        "When does the product launch?",
        "Quote the original CEO announcement verbatim.",
        "What is the current price?",
        "x",
        "another cue, totally different bytes",
    ],
)
def test_bench_route_matches_pipeline_formula(cue: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IAI_MCP_EFE_USE_SHADOW", raising=False)
    helper = bench._bench_efe_route_for_cue
    route, cue_hash = helper(cue)

    digest = hashlib.md5(str(cue).encode("utf-8")).digest()
    expected_cue_hash = digest[:4].hex()
    expected_route = "efe_real" if (digest[0] & 1) else "efe_shadow"

    assert cue_hash == expected_cue_hash, (
        f"cue_hash drift: helper={cue_hash} expected={expected_cue_hash}"
    )
    assert route == expected_route, (
        f"route drift: helper={route} expected={expected_route} cue={cue!r}"
    )
    assert len(cue_hash) == 8
    assert all(c in "0123456789abcdef" for c in cue_hash)


def test_bench_route_respects_shadow_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_EFE_USE_SHADOW", "1")
    for cue in ["cue-a", "cue-b", "cue-c", "cue-d"]:
        route, cue_hash = bench._bench_efe_route_for_cue(cue)
        assert route == "efe_shadow", (
            f"shadow override broken for cue={cue!r}: got {route}"
        )
        expected = hashlib.md5(cue.encode("utf-8")).digest()[:4].hex()
        assert cue_hash == expected


def _make_pr(
    probe_id: str, seed: int, route: str, cue_hash: str, *, hit: bool = True
) -> "bench.ProbeResult":
    return bench.ProbeResult(
        probe_id=probe_id,
        seed=seed,
        n_slice=0,
        condition="post_flip",
        topic="launch_date",
        pipeline_rank=1 if hit else -1,
        cosine_rank=1 if hit else -1,
        pipeline_hit_at_k=hit,
        cosine_hit_at_k=hit,
        pipeline_top1_text="top1",
        route=route,
        cue_hash=cue_hash,
    )


def test_csv_writer_emits_route_and_cue_hash_columns(tmp_path: Path) -> None:
    results = [
        _make_pr("probe-a", 13, "efe_real", "aaaaaaaa"),
        _make_pr("probe-b", 13, "efe_shadow", "bbbbbbbb"),
        _make_pr("probe-c", 42, "efe_real", "cccccccc"),
        _make_pr("probe-d", 42, "efe_shadow", "dddddddd", hit=False),
    ]

    summary = {
        "gates": {
            "overall_pass": True,
            "cross_seed_robust": True,
            "per_cell": {},
        },
        "cross_seed": {},
        "per_cell": [],
    }
    env = {"iai_mcp_git_sha": "test-sha"}
    run_id = "test-route-csv"
    bench.write_outputs(
        output_dir=tmp_path,
        run_id=run_id,
        summary=summary,
        all_results=results,
        env=env,
        duration_seconds=0.0,
    )

    csv_path = tmp_path / f"contradiction_longitudinal_{run_id}.csv"
    assert csv_path.exists(), f"CSV not written at {csv_path}"

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        assert "route" in fieldnames, (
            f"`route` column missing from CSV. Fieldnames: {fieldnames}"
        )
        assert "cue_hash" in fieldnames, (
            f"`cue_hash` column missing from CSV. Fieldnames: {fieldnames}"
        )
        rows = list(reader)

    assert len(rows) == 4
    by_id = {r["probe_id"]: r for r in rows}
    assert by_id["probe-a"]["route"] == "efe_real"
    assert by_id["probe-a"]["cue_hash"] == "aaaaaaaa"
    assert by_id["probe-b"]["route"] == "efe_shadow"
    assert by_id["probe-b"]["cue_hash"] == "bbbbbbbb"
    assert by_id["probe-c"]["route"] == "efe_real"
    assert by_id["probe-c"]["cue_hash"] == "cccccccc"
    assert by_id["probe-d"]["route"] == "efe_shadow"
    assert by_id["probe-d"]["cue_hash"] == "dddddddd"

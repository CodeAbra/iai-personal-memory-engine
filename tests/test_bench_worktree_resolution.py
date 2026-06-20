from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

WORKTREE_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = WORKTREE_ROOT / "bench"

BENCH_SCRIPTS_NEEDING_SHIM = [
    "_night_runner.py",
    "capture_dedup_lock.py",
    "community_pipeline_perf.py",
    "consolidation_rss_peak.py",
    "contradiction_longitudinal_claude.py",
    "embedder_baseline.py",
    "embedder_latency.py",
    "longmemeval_blind.py",
    "memory_footprint.py",
    "memorygraph_csr_parity.py",
    "memorygraph_memory.py",
    "neural_map.py",
    "personal_fact_drift.py",
    "pipeline_stage_timings.py",
    "sleep_ablation.py",
    "tokens.py",
    "total_session_cost.py",
    "trajectory.py",
    "verbatim.py",
]

BENCH_SCRIPTS_NO_SHIM = [
    "analyze_arousal_ab.py",
    "analyze_efe_ab.py",
    "arousal_budget_ab.py",
    "contradiction_longitudinal.py",
    "embed_warm_cost.py",
    "embedder_recall_compare.py",
    "make_parity_summary.py",
    "memorygraph_adj_spike.py",
]


def _imports_iai_or_bench(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(
            alias.name == "iai_mcp"
            or alias.name.startswith("iai_mcp.")
            or alias.name == "bench"
            or alias.name.startswith("bench.")
            for alias in node.names
        )
    if isinstance(node, ast.ImportFrom):
        mod = node.module or ""
        return (
            mod == "iai_mcp"
            or mod.startswith("iai_mcp.")
            or mod == "bench"
            or mod.startswith("bench.")
        )
    return False


def _is_src_path_assign(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "_SRC_PATH"
    )


def _is_sys_path_guard_if(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if not (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.NotIn)
    ):
        return False
    left = test.left
    right = test.comparators[0]
    left_is_src_path = isinstance(left, ast.Name) and left.id == "_SRC_PATH"
    right_is_sys_path = (
        isinstance(right, ast.Attribute)
        and right.attr == "path"
        and isinstance(right.value, ast.Name)
        and right.value.id == "sys"
    )
    return left_is_src_path and right_is_sys_path


def _simulate_shim(fake_sys_path: list[str], src_path: str) -> list[str]:
    new_path = list(fake_sys_path)
    if src_path not in new_path:
        new_path.insert(0, src_path)
    return new_path


@pytest.mark.parametrize("script", BENCH_SCRIPTS_NEEDING_SHIM)
def test_bench_script_has_shim_before_iai_import(script: str) -> None:
    path = BENCH_DIR / script
    assert path.exists(), f"bench script missing: {path}"
    tree = ast.parse(path.read_text())

    src_path_idx: int | None = None
    guard_if_idx: int | None = None
    first_toplevel_import_idx: int | None = None
    for i, node in enumerate(tree.body):
        if src_path_idx is None and _is_src_path_assign(node):
            src_path_idx = i
        if guard_if_idx is None and _is_sys_path_guard_if(node):
            guard_if_idx = i
        if first_toplevel_import_idx is None and _imports_iai_or_bench(node):
            first_toplevel_import_idx = i

    assert src_path_idx is not None, (
        f"{script}: no `_SRC_PATH = ...` assignment found at module level"
    )
    assert guard_if_idx is not None, (
        f"{script}: no `if _SRC_PATH not in sys.path:` block found at module level"
    )

    has_any_import = any(_imports_iai_or_bench(n) for n in ast.walk(tree))
    assert has_any_import, (
        f"{script}: no iai_mcp / bench import found anywhere in file "
        f"(authoritative list says this script needs the shim - "
        f"fix the list or the file)"
    )

    if first_toplevel_import_idx is not None:
        assert src_path_idx < first_toplevel_import_idx, (
            f"{script}: `_SRC_PATH` assign at body[{src_path_idx}] must come BEFORE "
            f"first top-level iai_mcp/bench import at body[{first_toplevel_import_idx}]"
        )
        assert guard_if_idx < first_toplevel_import_idx, (
            f"{script}: shim `if` block at body[{guard_if_idx}] must come BEFORE "
            f"first top-level iai_mcp/bench import at body[{first_toplevel_import_idx}]"
        )


@pytest.mark.parametrize("script", BENCH_SCRIPTS_NO_SHIM)
def test_skip_list_bench_scripts_have_no_shim(script: str) -> None:
    path = BENCH_DIR / script
    assert path.exists(), f"skip-list bench script missing: {path}"
    src = path.read_text()
    assert "_SRC_PATH not in sys.path" not in src, (
        f"{script}: skip-list script unexpectedly carries the shim - "
        f"either drop the shim or move the script to BENCH_SCRIPTS_NEEDING_SHIM"
    )
    tree = ast.parse(src)
    for node in ast.walk(tree):
        assert not _imports_iai_or_bench(node), (
            f"{script}: now imports iai_mcp / bench.* - move it out of "
            f"BENCH_SCRIPTS_NO_SHIM and add the shim"
        )


def test_shim_block_idempotency_in_same_process() -> None:
    src_path = str(BENCH_DIR.parent / "src")
    fresh = ["/some/other/path", "/usr/lib/python3.11"]
    after_first = _simulate_shim(fresh, src_path)
    assert after_first[0] == src_path
    assert len(after_first) == len(fresh) + 1
    after_second = _simulate_shim(after_first, src_path)
    assert after_second == after_first
    assert len(after_second) == len(after_first)
    pre_populated = ["/foo", src_path, "/bar"]
    after_third = _simulate_shim(pre_populated, src_path)
    assert after_third == pre_populated


def test_authoritative_script_lists_match_filesystem() -> None:
    on_disk = {
        p.name
        for p in BENCH_DIR.iterdir()
        if p.is_file() and p.suffix == ".py" and p.name != "__init__.py"
    }
    in_lists = set(BENCH_SCRIPTS_NEEDING_SHIM) | set(BENCH_SCRIPTS_NO_SHIM)
    missing_from_lists = on_disk - in_lists
    extra_in_lists = in_lists - on_disk
    assert not missing_from_lists, (
        f"bench/*.py present on disk but missing from authoritative lists: "
        f"{sorted(missing_from_lists)} - classify each as needing-shim or no-shim, "
        f"then add to the matching list."
    )
    assert not extra_in_lists, (
        f"authoritative lists reference scripts not on disk: "
        f"{sorted(extra_in_lists)} - drop the stale entries."
    )
    overlap = set(BENCH_SCRIPTS_NEEDING_SHIM) & set(BENCH_SCRIPTS_NO_SHIM)
    assert not overlap, (
        f"scripts appear in both lists: {sorted(overlap)} - pick one."
    )


_SLOW_XFAIL: dict[str, str] = {}


@pytest.mark.slow
@pytest.mark.parametrize(
    "script",
    [
        pytest.param(
            s,
            marks=pytest.mark.xfail(
                reason=_SLOW_XFAIL[s], strict=False, run=True
            ),
        )
        if s in _SLOW_XFAIL
        else s
        for s in BENCH_SCRIPTS_NEEDING_SHIM
    ],
)
def test_shim_resolves_to_worktree(script: str) -> None:
    path = BENCH_DIR / script
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, str(path), "--help"],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"{script} --help (cwd=/tmp, no PYTHONPATH) exited "
        f"{result.returncode}\nstdout: {result.stdout[:400]}\n"
        f"stderr: {result.stderr[:1200]}"
    )

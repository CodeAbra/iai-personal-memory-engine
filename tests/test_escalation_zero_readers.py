"""Static proof that the confidence-gated escalation mechanism is fully
retired -- every deleted function, constant, and kill-switch env var name
appears NOWHERE under `src/iai_mcp`, `tests/`, or `bench/`, as an identifier,
an attribute access, or a string literal. Mirrors the AST-scan pattern in
`tests/test_rank_cache_retirement.py`, extended to also cover `tests/` and
`bench/` (source-only retirements there only proved absence from
`src/iai_mcp`) and to string-literal env-var names (the cache retirement had
no analogous kill-switch surface).

The retired mechanism, its full mechanism and history, stays inspectable
under the phase's own retirement directory, which lives outside every scan
root -- historical record, not live code.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCAN_ROOTS = (_REPO_ROOT / "src" / "iai_mcp", _REPO_ROOT / "tests", _REPO_ROOT / "bench")

_THIS_FILE = Path(__file__).resolve()
_EXCLUDE_DIR_SUBSTRINGS = ("__pycache__",)

_RETIRED_SYMBOLS = (
    "escalate_recall_candidates",
    "compute_spread_depth",
    "_CONF_HIGH_COSINE_THRESHOLD",
    "_CONF_HIGH_HIT_COUNT_MIN",
    "_CONF_SCORE_THRESHOLD",
    "_escalation_bound_off",
    "_ESCALATION_BOUND_OVER_FETCH_FACTOR",
    "_ESCALATION_UNBOUNDED_OVER_FETCH_FACTOR",
    "_ESCALATION_WIDEN_FAILED_WINDOW_S",
    "_escalation_widen_failed_last_emit",
    "_reset_escalation_widen_failed_state",
    "_emit_escalation_widen_failed",
    "TELEMETRY_ESCALATION_WIDEN_FAILED",
    "_single_pass_recall_off",
)

_RETIRED_ENV_VARS = (
    "IAI_MCP_SINGLE_PASS_RECALL_OFF",
    "IAI_MCP_CONF_ESCALATE_OFF",
    "IAI_MCP_ESCALATION_BOUND_OFF",
)

_RETIRED_CONSTANTS = (
    "ESCALATION_BOUND_CAP",
    "ADAPTIVE_ESCALATION_CAP",
)

_ALL_RETIRED_NAMES = _RETIRED_SYMBOLS + _RETIRED_ENV_VARS + _RETIRED_CONSTANTS


def _iter_py_files():
    for scan_root in _SCAN_ROOTS:
        for root, dirs, files in os.walk(scan_root):
            dirs[:] = [
                d for d in dirs
                if not any(ex in os.path.join(root, d) for ex in _EXCLUDE_DIR_SUBSTRINGS)
            ]
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                path = Path(root) / fn
                if path.resolve() == _THIS_FILE:
                    continue
                yield path


def _identifiers_and_literals_in_file(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.add(node.value)
    return found


def test_escalation_symbols_have_zero_readers_src_and_tests():
    """Every retired escalation symbol, constant, and env-var name is
    absent from `src/iai_mcp`, `tests/`, and `bench/` -- not merely
    unreachable on the default recall path, absent from the tree entirely. A future
    reintroduction (even a dead re-import, or a stray monkeypatch.setenv
    targeting a since-deleted kill-switch) fails this guard loud."""
    hits: dict[str, list[str]] = {name: [] for name in _ALL_RETIRED_NAMES}

    for path in _iter_py_files():
        found = _identifiers_and_literals_in_file(path)
        rel = str(path.relative_to(_REPO_ROOT))
        for name in _ALL_RETIRED_NAMES:
            if name in found:
                hits[name].append(rel)

    failures = {name: files for name, files in hits.items() if files}
    assert not failures, (
        "retired escalation symbol(s) still referenced: "
        f"{failures}"
    )

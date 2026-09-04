"""Resurrection guard: the multi-strategy retrieval selector and its labels
must stay out of src/, while the wired weight-tuner stays importable and
callable.
"""
from __future__ import annotations

import ast
from pathlib import Path

import iai_mcp.learn
import iai_mcp.lilli.profile.tuner

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"

FORBIDDEN_LABELS = {"greedy_2hop", "rich_club_first", "pipeline_default"}


def _iter_py_files(root: Path):
    yield from root.rglob("*.py")


def _find_label_offenses(path: Path) -> list[str]:
    offenses: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in FORBIDDEN_LABELS:
                offenses.append(f"{path}:{node.lineno}")
    return offenses


def test_selector_symbol_absent():
    assert getattr(iai_mcp.learn, "pick_retrieval_strategy", None) is None
    assert getattr(iai_mcp.lilli.profile.tuner, "pick_retrieval_strategy", None) is None
    assert getattr(iai_mcp.learn, "EPSILON_EXPLORE", None) is None
    assert getattr(iai_mcp.lilli.profile.tuner, "EPSILON_EXPLORE", None) is None


def test_strategy_labels_absent_from_src():
    assert SRC_ROOT.exists(), f"scan root missing: {SRC_ROOT}"
    files = list(_iter_py_files(SRC_ROOT))
    assert files, f"scan root yielded zero .py files: {SRC_ROOT}"

    offenses: list[str] = []
    for path in files:
        offenses.extend(_find_label_offenses(path))

    assert offenses == [], f"retired strategy label literals found: {offenses}"


def test_wired_function_retained():
    from iai_mcp.learn import update_retrieval_weights as update_via_learn
    from iai_mcp.lilli.profile.tuner import RetrievalFeedback
    from iai_mcp.lilli.profile.tuner import update_retrieval_weights as update_via_tuner

    feedback = RetrievalFeedback(query_type="smoke", hit_ids=[], used_ids=[])
    result_learn = update_via_learn(feedback, {"W_COSINE": 1.0})
    result_tuner = update_via_tuner(feedback, {"W_COSINE": 1.0})

    assert result_learn == result_tuner
    assert "W_COSINE" in result_learn

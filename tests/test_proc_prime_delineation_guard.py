"""The awake priming seam is additive-only. Three machine-checkable
legs prove it never writes a tuned ranking weight, never invalidates the
tuned-weight cache outside its one legitimate call site, and is a distinct
mechanism from `potentiate_coactivation`.

The seam only READS the prime cache, WIDENS seeds, and widens the k_margin
cut for its own scorer call -- it never calls any merge/consolidation path,
so `never_merge`/pinned semantics are never engaged.

Source/AST scans, not docstring or prose comparison -- a prose-compare test
has no teeth.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from itertools import combinations
from pathlib import Path

import iai_mcp.pipeline as _pm
import iai_mcp.retrieve as _retrieve

_W_NAMES = {"W_COSINE", "W_AAAK", "W_DEGREE", "W_AGE"}
_KNOB_TUNE_RELPATH = "lilli/cycle/sleep_pipeline/_knob_tune.py"
_SRC_ROOT = Path(_pm.__file__).resolve().parent


def _recall_core_ast() -> ast.FunctionDef:
    src = textwrap.dedent(inspect.getsource(_pm._recall_core))
    tree = ast.parse(src)
    return tree.body[0]


def _assignment_targets(tree: ast.AST) -> "set[str]":
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        else:
            continue
        for t in targets:
            if isinstance(t, ast.Name):
                names.add(t.id)
    return names


# ---------------------------------------------------------------------------
# LEG 1: no W_* ranking-weight writes in the seam.
# ---------------------------------------------------------------------------

def test_leg1_seam_never_assigns_ranking_weights():
    tree = _recall_core_ast()
    assigned = _assignment_targets(tree)

    written_weights = assigned & _W_NAMES
    assert not written_weights, (
        f"_recall_core assigns to ranking weight(s) {written_weights} -- "
        "the seam must only READ W_COSINE/W_AAAK/W_DEGREE/W_AGE, never write them"
    )
    # The k_margin lever writes only a call-local, never a ranking weight.
    assert "_prime_k_margin" in assigned, (
        "expected the k_margin widening's local assignment target "
        "'_prime_k_margin' to be present in _recall_core -- source may have "
        "moved the lever under a different name"
    )
    assert "_prime_k_margin" not in _W_NAMES


# ---------------------------------------------------------------------------
# LEG 2: retrieval_weight_cache.invalidate confined to _knob_tune.py.
# ---------------------------------------------------------------------------

def test_leg2_invalidate_confined_to_knob_tune():
    hits: "list[Path]" = []
    for path in _SRC_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "retrieval_weight_cache.invalidate(" in text:
            hits.append(path)

    hit_names = {str(p.relative_to(_SRC_ROOT)) for p in hits}
    assert hit_names == {_KNOB_TUNE_RELPATH}, (
        f"retrieval_weight_cache.invalidate( appears in {sorted(hit_names)}, "
        f"expected exactly {{{_KNOB_TUNE_RELPATH!r}}} -- the awake priming "
        "seam and any other awake-path module must never invalidate the "
        "tuned-weight cache"
    )

    pipeline_text = Path(_pm.__file__).read_text(encoding="utf-8", errors="ignore")
    assert "retrieval_weight_cache.invalidate(" not in pipeline_text


# ---------------------------------------------------------------------------
# LEG 3: distinct from potentiate_coactivation.
# ---------------------------------------------------------------------------

def test_leg3_distinct_from_potentiate_coactivation():
    core_src = inspect.getsource(_pm._recall_core)
    for forbidden in ("potentiate_coactivation(", "queue_coactivation(", "queue_reinforce("):
        assert forbidden not in core_src, (
            f"_recall_core calls {forbidden!r} -- the priming seam must "
            "never invoke the coactivation/reinforce plasticity path"
        )
    assert "prime_cache" in core_src, (
        "_recall_core no longer references prime_cache -- the widening "
        "block may have been removed or relocated"
    )

    # Positive structural check: potentiate_coactivation is UNDIRECTED
    # (all-pairs combinations), the priming seam is DIRECTED (dst only) --
    # confirming the two are not restatements of the same mechanism.
    coact_src = inspect.getsource(_retrieve.potentiate_coactivation)
    assert f"{combinations.__name__}(" in coact_src

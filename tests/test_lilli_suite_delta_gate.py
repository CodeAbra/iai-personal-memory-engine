"""Full-suite delta gate: the Rust storage driver must introduce zero NEW test
failures OR errors versus the stdlib-default baseline.

A known-flaky baseline exists, so the bar is NOT absolute green.  The gate runs
the suite twice -- once under the stdlib default, once under
``LILLI_STORAGE_DRIVER=lilli`` -- and asserts the lilli failure-or-error node-id
set is a SUBSET of the stdlib failure-or-error set (delta >= 0).  A node-id that
fails or errors under BOTH is a pre-existing flaky baseline member, not a
regression.  A node-id that fails or errors ONLY under lilli is a NEW regression.

An ``ERROR`` is structurally distinct from a ``FAILED``: a test that errors
during collection / fixture / setup / import (e.g. an import the Rust engine
cannot satisfy) emits ``ERROR <nodeid>`` -- which the failures-only ``-rf`` flag
hides.  Both subprocess runs therefore use ``-rfE`` and the parser harvests
BOTH ``FAILED`` and ``ERROR`` short-summary node-ids into one set, so a lilli-only
error is caught and named exactly like a lilli-only failure.

Note: under the lilli driver two doctor checks degrade to "check skipped:
ParseError" and return passed=True (the engine covers the store/recall SQL
subset, not arbitrary diagnostic SQL).  A degrade-to-skip emits NO FAILED/ERROR
node-id, so it never enters either set and is correctly not a regression.

The full-suite subprocess run is OPT-IN (``IAI_MCP_FULL_SUITE_DELTA``): the
suite imports the core + loads the embedder/store, so it runs only at phase
verification by the orchestrator in a single process -- never in a parallel
worktree executor.  The parser + subset unit tests always run and need no store.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_THIS_FILE = Path(__file__).name


def _parse_failed_or_errored_nodeids(pytest_stdout: str) -> set[str]:
    """Return the node-ids of every FAILED or ERROR short-summary line.

    Parses the ``-rfE`` short-summary block, where each relevant line looks like
    ``FAILED tests/test_x.py::test_y - AssertionError`` or
    ``ERROR tests/test_x.py::test_z - ImportError``.  The node-id is the token
    after the ``FAILED``/``ERROR`` label, before any `` - `` reason suffix.  Both
    kinds collapse into ONE set so the gate treats a new error like a new failure.
    """
    nodeids: set[str] = set()
    for raw in pytest_stdout.splitlines():
        line = raw.strip()
        for label in ("FAILED ", "ERROR "):
            if line.startswith(label):
                rest = line[len(label):].strip()
                # Drop the trailing " - <reason>" if present.
                nodeid = rest.split(" - ", 1)[0].strip()
                if nodeid:
                    nodeids.add(nodeid)
                break
    return nodeids


def _assert_no_new_regressions(stdlib_set: set[str], lilli_set: set[str]) -> None:
    """Raise naming the NEW node-ids if lilli is not a subset of stdlib."""
    new = lilli_set - stdlib_set
    if new:
        raise AssertionError(
            "the lilli driver introduced NEW failures/errors not present under "
            "stdlib (delta < 0):\n"
            + "\n".join(f"  {nid}" for nid in sorted(new))
        )


# ---------------------------------------------------------------------------
# Parser + subset unit tests (always run; no store / subprocess / env needed).
# ---------------------------------------------------------------------------
def test_parser_harvests_both_failed_and_errored():
    out = (
        "=== short test summary info ===\n"
        "FAILED tests/test_a.py::test_one - AssertionError: nope\n"
        "ERROR tests/test_b.py::test_two - ImportError: missing engine\n"
        "PASSED tests/test_c.py::test_three\n"
    )
    got = _parse_failed_or_errored_nodeids(out)
    assert got == {
        "tests/test_a.py::test_one",
        "tests/test_b.py::test_two",
    }


def test_subset_passes_when_lilli_within_stdlib():
    stdlib = {"tests/test_a.py::test_flaky", "tests/test_b.py::test_other"}
    lilli = {"tests/test_a.py::test_flaky"}
    # No NEW node-ids -> no raise.
    _assert_no_new_regressions(stdlib, lilli)


def test_subset_fails_on_lilli_only_failure_and_names_it():
    stdlib = {"tests/test_a.py::test_flaky"}
    lilli = {"tests/test_a.py::test_flaky", "tests/test_new.py::test_regression"}
    with pytest.raises(AssertionError) as exc:
        _assert_no_new_regressions(stdlib, lilli)
    assert "tests/test_new.py::test_regression" in str(exc.value)


def test_subset_fails_on_lilli_only_error_and_names_it():
    # The critical case: an ERROR present under lilli, ABSENT under stdlib, must
    # break the subset assertion and be named -- a new collection/import error
    # is a regression even though it is not a FAILED.
    stdlib_out = "FAILED tests/test_a.py::test_flaky - AssertionError\n"
    lilli_out = (
        "FAILED tests/test_a.py::test_flaky - AssertionError\n"
        "ERROR tests/test_engine.py::test_loads - ImportError: lilli engine\n"
    )
    stdlib_set = _parse_failed_or_errored_nodeids(stdlib_out)
    lilli_set = _parse_failed_or_errored_nodeids(lilli_out)
    with pytest.raises(AssertionError) as exc:
        _assert_no_new_regressions(stdlib_set, lilli_set)
    assert "tests/test_engine.py::test_loads" in str(exc.value)


def test_fail_or_error_under_both_is_not_a_regression():
    # A node-id that fails under stdlib and ERRORS under lilli (or vice versa)
    # is still "present under both" by node-id, so it is a baseline member, not
    # a new regression -- the set membership is by node-id, not by kind.
    stdlib_out = "ERROR tests/test_flaky.py::test_x - TimeoutError\n"
    lilli_out = "FAILED tests/test_flaky.py::test_x - AssertionError\n"
    stdlib_set = _parse_failed_or_errored_nodeids(stdlib_out)
    lilli_set = _parse_failed_or_errored_nodeids(lilli_out)
    _assert_no_new_regressions(stdlib_set, lilli_set)  # no raise


# ---------------------------------------------------------------------------
# Full-suite subprocess delta gate (opt-in).
# ---------------------------------------------------------------------------
_FULL_SUITE_ENABLED = bool(os.environ.get("IAI_MCP_FULL_SUITE_DELTA"))


def _run_full_suite(*, lilli: bool) -> set[str]:
    """Run the full suite once and return its failed-or-errored node-id set.

    ``-rfE`` is load-bearing: the ``E`` emits the ``ERROR <nodeid>`` short-summary
    lines for collection/fixture/setup/import errors that ``-rf`` (failures only)
    hides.  This gate file is deselected to avoid recursion.
    """
    env = dict(os.environ)
    if lilli:
        env["LILLI_STORAGE_DRIVER"] = "lilli"
        env["IAI_MCP_HD_BACKEND"] = "rust"
    else:
        env.pop("LILLI_STORAGE_DRIVER", None)
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "tests",
            "-q", "-p", "no:cacheprovider", "--tb=no", "-rfE",
            "--deselect", str(Path("tests") / _THIS_FILE),
        ],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    return _parse_failed_or_errored_nodeids(proc.stdout)


@pytest.mark.skipif(
    not _FULL_SUITE_ENABLED,
    reason=(
        "full-suite delta gate is opt-in: set IAI_MCP_FULL_SUITE_DELTA (heavy; "
        "runs only at phase verification in a single process, never in a "
        "worktree executor)"
    ),
)
def test_lilli_suite_introduces_no_new_failures_or_errors():
    stdlib_set = _run_full_suite(lilli=False)
    lilli_set = _run_full_suite(lilli=True)
    print(f"\n[delta-gate] stdlib failed/errored: {len(stdlib_set)}")
    print(f"[delta-gate] lilli  failed/errored: {len(lilli_set)}")
    _assert_no_new_regressions(stdlib_set, lilli_set)
    print("[delta-gate] PASS: lilli failure/error set is a subset of stdlib")

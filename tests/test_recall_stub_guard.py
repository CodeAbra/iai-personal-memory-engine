from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import tests._helpers as _stub_helpers
from tests._helpers import RECALL_STUB_ACTIVE_ENV, stub_embedder_for_store

_TESTS_DIR = Path(__file__).resolve().parent


def _run_nested(body: str) -> subprocess.CompletedProcess:
    """Run a temp test module through the real tests/conftest.py.

    Written inside the tests/ rootdir (never an unrelated tmp dir) so
    pytest picks up the real conftest.py and its autouse guard fixture --
    the one place this proof could itself pass-by-degrading if the guard
    were silently absent from the nested run.
    """
    name = f"_probe_recall_stub_guard_{uuid.uuid4().hex}.py"
    path = _TESTS_DIR / name
    path.write_text(textwrap.dedent(body))
    try:
        env = dict(os.environ)
        env.pop(RECALL_STUB_ACTIVE_ENV, None)
        return subprocess.run(
            [sys.executable, "-m", "pytest", str(path), "-q", "-p", "no:randomly"],
            cwd=str(_TESTS_DIR.parent),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        path.unlink(missing_ok=True)


def test_armed_factory_stub_with_fallback_fails_end_to_end():
    result = _run_nested(
        """
        import logging

        from tests._helpers import stub_embedder_for_store


        def test_body(monkeypatch):
            stub_embedder_for_store(monkeypatch, object())
            logging.getLogger("iai_mcp.core").warning(
                "recall_pipeline_fallback: synthetic degrade for guard proof"
            )
        """
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "recall embedder stub silently degraded" in result.stdout


def test_armed_env_var_with_fallback_fails_end_to_end():
    result = _run_nested(
        """
        import logging
        import os

        from tests._helpers import RECALL_STUB_ACTIVE_ENV


        def test_body():
            os.environ[RECALL_STUB_ACTIVE_ENV] = "1"
            try:
                logging.getLogger("iai_mcp.core").warning(
                    "recall_pipeline_fallback: synthetic degrade for guard proof"
                )
            finally:
                os.environ.pop(RECALL_STUB_ACTIVE_ENV, None)
        """
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "recall embedder stub silently degraded" in result.stdout


def test_unarmed_fallback_leaves_guard_inert():
    result = _run_nested(
        """
        import logging


        def test_body():
            logging.getLogger("iai_mcp.core").warning(
                "recall_pipeline_fallback: synthetic degrade, unarmed"
            )
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_arm_flag_is_the_same_module_object_conftest_reads(monkeypatch):
    stub_embedder_for_store(monkeypatch, object())
    conftest_mod = sys.modules["tests.conftest"]
    assert conftest_mod._stub_helpers is _stub_helpers
    assert conftest_mod._stub_helpers._recall_stub_armed is _stub_helpers._recall_stub_armed
    assert _stub_helpers._recall_stub_armed
    # env var is process-global, so arming is visible regardless of which
    # `_helpers` module object a co-collected test file imported.
    assert os.environ.get(RECALL_STUB_ACTIVE_ENV) == "1"

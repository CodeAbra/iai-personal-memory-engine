from __future__ import annotations

import os
import tempfile
from pathlib import Path

from iai_mcp.store import MemoryStore

# arm flag for the recall-stub-not-degraded guard (tests/conftest.py): a
# per-test sticky signal that a recall test installed a deterministic
# embedder double. MUTATED IN PLACE (append/.clear()) -- never rebind, the
# guard reads this list by module attribute and a rebind would strand that
# read against a stale object.
_recall_stub_armed: list[bool] = []

# shared with bench/recall_accuracy.py, which cannot import this module --
# both sides must agree on this string literal.
RECALL_STUB_ACTIVE_ENV = "IAI_MCP_TEST_RECALL_STUB_ACTIVE"


def arm_recall_stub_guard() -> None:
    # env var is process-global: makes the arm signal independent of which
    # `_helpers` module object a given test file happens to import.
    _recall_stub_armed.append(True)
    os.environ[RECALL_STUB_ACTIVE_ENV] = "1"


def stub_embedder_for_store(monkeypatch, embedder):
    """Install a deterministic recall embedder double and arm the guard.

    ``**_kwargs`` is load-bearing: the recall dispatch always passes
    ``allow_identity_mismatch``/``build_timeout`` to the real funnel; a
    double that does not accept them raises TypeError, which the pipeline
    catches and silently degrades recall instead of using this double.
    """
    import iai_mcp.embed as _embed_mod

    arm_recall_stub_guard()
    monkeypatch.setattr(
        _embed_mod,
        "embedder_for_store",
        lambda _store, **_kwargs: embedder,
    )
    return embedder


def short_socket_path(prefix: str = "iai-sock-") -> Path:
    d = Path(tempfile.mkdtemp(prefix=prefix))
    return d / "d.sock"


def make_tmp_store(tmp_path: Path) -> MemoryStore:
    store_root = tmp_path / "hippo"
    store_root.mkdir(parents=True, exist_ok=True)
    return MemoryStore(path=store_root)


def set_tmp_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "hippo"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "test.sock"))

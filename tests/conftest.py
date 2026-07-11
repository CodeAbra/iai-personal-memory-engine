from __future__ import annotations

import gc
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from _recall_helpers import (  # noqa: E402,F401
    _deterministic_vec,
    _make_gold_record,
    _populate_store,
    _prime_structural_cache,
    _random_vec,
    UUID_HUB,
    UUID_INTER,
    UUID_SEED,
    UUID_TWO_HOP,
    UUID_TWO_HOP_SURFACE,
)


_TEST_PASSPHRASE = "iai-mcp-test-passphrase"


@pytest.fixture(scope="session", autouse=True)
def _lilli_fast_fsync():
    """Set LILLI_FSYNC_MODE=fast for the duration of the test session when the
    lilli storage driver is active and the caller has not already chosen a mode.

    The lilli engine defaults to F_FULLFSYNC on macOS, which flushes the drive
    controller's DRAM cache to NAND on every commit.  At n=1000 commits per
    property-test run that adds minutes of latency and trips the pytest-timeout
    wall.  Test runs verify logical correctness, not power-loss durability, so
    plain os.fsync is correct here.  Production keeps the F_FULLFSYNC default
    because this fixture never runs outside pytest.

    The prior value is saved and restored around the yield so in-process tools
    that inspect LILLI_FSYNC_MODE after the session do not see the test value.
    """
    prior = os.environ.get("LILLI_FSYNC_MODE")
    if (
        os.environ.get("LILLI_STORAGE_DRIVER") == "lilli"
        and prior is None
    ):
        os.environ["LILLI_FSYNC_MODE"] = "fast"
    yield
    if prior is None:
        os.environ.pop("LILLI_FSYNC_MODE", None)
    else:
        os.environ["LILLI_FSYNC_MODE"] = prior


@pytest.fixture(autouse=True)
def _hermetic_default_paths(tmp_path_factory, monkeypatch: pytest.MonkeyPatch):
    base = tmp_path_factory.mktemp("iai-hermetic")
    fake_root = base / ".iai-mcp"
    fake_root.mkdir(parents=True, exist_ok=True)

    from iai_mcp.hippo import _operator_home
    _real_cache = _operator_home() / ".cache" / "huggingface"
    monkeypatch.setenv("HF_HOME", str(_real_cache))
    monkeypatch.setenv("HF_HUB_CACHE", str(_real_cache / "hub"))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(_real_cache / "hub"))

    monkeypatch.setenv("HOME", str(base))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(fake_root / ".daemon.sock"))
    import iai_mcp.hippo as _hippo
    import iai_mcp.store as _store
    import iai_mcp.concurrency as _conc
    import iai_mcp.daemon_state as _ds
    import iai_mcp.lifecycle_state as _lifecycle_state
    import iai_mcp.cli as _cli
    import iai_mcp.lifecycle_event_log as _lel
    import iai_mcp.capture_queue as _cq
    import iai_mcp.lifecycle as _lifecycle
    import iai_mcp.daemon as _daemon
    import iai_mcp.crypto as _crypto
    import iai_mcp.backup as _backup
    monkeypatch.setattr(_hippo, "_DEFAULT_IAI_ROOT", fake_root, raising=False)
    monkeypatch.setattr(_store, "DEFAULT_STORAGE_PATH", fake_root, raising=False)
    monkeypatch.setattr(_conc, "SOCKET_PATH", fake_root / ".daemon.sock", raising=False)
    monkeypatch.setattr(_ds, "STATE_PATH", fake_root / ".daemon-state.json", raising=False)
    monkeypatch.setattr(
        _lifecycle_state, "LIFECYCLE_STATE_PATH",
        fake_root / "lifecycle_state.json", raising=False,
    )
    monkeypatch.setattr(_cli, "LOCK_PATH", fake_root / ".lock", raising=False)
    monkeypatch.setattr(
        _cli, "STATE_PATH", fake_root / ".daemon-state.json", raising=False,
    )
    monkeypatch.setattr(_lel, "DEFAULT_LOG_DIR", fake_root / "logs", raising=False)
    monkeypatch.setattr(_cq, "DEFAULT_QUEUE_DIR", fake_root / "pending", raising=False)
    monkeypatch.setattr(
        _lifecycle, "DEFAULT_LOCK_PATH", fake_root / ".lifecycle.lock", raising=False,
    )
    monkeypatch.setattr(
        _daemon, "SESSION_START_CACHE_PATH",
        fake_root / ".session-start-payload.cached.md", raising=False,
    )
    monkeypatch.setattr(_crypto, "_DEFAULT_STORE_ROOT", fake_root, raising=False)
    monkeypatch.setattr(_backup, "DEFAULT_STORE_PATH", str(fake_root), raising=False)
    yield fake_root


@pytest.fixture(autouse=True)
def _clear_autoflush_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IAI_MCP_TEST_NO_AUTOFLUSH", raising=False)


@pytest.fixture(autouse=True)
def _crypto_passphrase_env(monkeypatch: pytest.MonkeyPatch) -> None:
    if "IAI_MCP_CRYPTO_PASSPHRASE" not in os.environ:
        monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", _TEST_PASSPHRASE)


_GUARDED_ENV = (
    "IAI_MCP_STORE",
    "LILLI_STORAGE_DRIVER",
    "IAI_MCP_CRYPTO_PASSPHRASE",
    "LILLI_FSYNC_MODE",
    "IAI_MCP_EMBED_MODEL",
)


@pytest.fixture(autouse=True)
def _isolate_process_globals(_lilli_fast_fsync, _crypto_passphrase_env):
    """Snapshot the load-bearing process-global env vars at setup and restore
    the exact prior state at teardown.

    A test that mutates one of these vars with a raw ``os.environ`` write would
    otherwise leak it into every later test in the same process; the leaked
    value redirects ``MemoryStore(path=None)`` to a wrong or deleted store root
    (wrong-key reads, empty-store, unable-to-open).  This restores the pristine
    baseline after each test, so collection order can never poison the suite.

    The snapshot is taken after the baseline env fixtures (fast fsync, the test
    passphrase) have run, so it captures the intended baseline rather than a
    pre-baseline empty.  Only the guarded keys are touched -- never a blanket
    ``os.environ.clear()``, which would wipe HOME/HF_HOME and break the
    hermetic fixtures.  Restore is symmetric (pop-if-was-absent, set-otherwise),
    so it composes safely with monkeypatch's own teardown.

    After the env restore, stale (closed) entries are swept from the lilli
    connection registry so it cannot grow unbounded across the whole suite, and
    a final ``gc.collect()`` reclaims any already-released background loop or
    thread objects.  The sweep drops ONLY connections that report themselves
    closed (identity-checked, mirroring the registry's own self-heal); it never
    clears the whole registry and never touches a live or borrowed handle.  The
    gc runs LAST and only reclaims already-released objects -- it is not the
    mechanism that closes a store (the explicit drains make closure observable).
    """
    snapshot = {k: os.environ.get(k) for k in _GUARDED_ENV}
    yield
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    try:
        from iai_mcp.lillibrain import connection as _lilli_conn

        registry = _lilli_conn._LILLI_CONN_REGISTRY
        for path, conn in list(registry.items()):
            if getattr(conn, "is_closed", False):
                # Identity check: only drop the slot if it still maps to this
                # closed connection, so a concurrent re-register is not clobbered.
                if registry.get(path) is conn:
                    registry.pop(path, None)
    except Exception:  # noqa: BLE001 -- not yet installed in some minimal envs
        pass

    gc.collect()


_AUTOFLUSH_OPT_OUT_ENV = "IAI_MCP_TEST_NO_AUTOFLUSH"


@pytest.fixture(autouse=True)
def _autoflush_lance_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        from iai_mcp import store as _store_mod
    except Exception:  # noqa: BLE001 -- env without iai_mcp installed yet
        return

    MemoryStore = getattr(_store_mod, "MemoryStore", None)
    flush_record_buffer = getattr(_store_mod, "flush_record_buffer", None)
    flush_edge_buffer = getattr(_store_mod, "flush_edge_buffer", None)
    if (
        MemoryStore is None
        or flush_record_buffer is None
        or flush_edge_buffer is None
    ):
        return

    try:
        from iai_mcp.events import flush_event_buffer as _flush_event_buffer
    except Exception:  # noqa: BLE001
        _flush_event_buffer = None

    def _opt_out() -> bool:
        return os.environ.get(_AUTOFLUSH_OPT_OUT_ENV) == "1"

    _orig_insert = MemoryStore.insert

    def _insert_then_flush(self, *args, **kwargs):
        result = _orig_insert(self, *args, **kwargs)
        if _opt_out():
            return result
        try:
            flush_record_buffer(self)
            flush_edge_buffer(self)
            if _flush_event_buffer is not None:
                _flush_event_buffer(self)
        except Exception:  # noqa: BLE001 -- flush MUST NOT fail the test
            pass
        return result

    monkeypatch.setattr(MemoryStore, "insert", _insert_then_flush)

    _orig_boost = getattr(MemoryStore, "boost_edges", None)
    if _orig_boost is not None:
        def _boost_then_flush(self, *args, **kwargs):
            result = _orig_boost(self, *args, **kwargs)
            if _opt_out():
                return result
            try:
                flush_edge_buffer(self)
            except Exception:  # noqa: BLE001
                pass
            return result

        monkeypatch.setattr(MemoryStore, "boost_edges", _boost_then_flush)

    _orig_add_contradicts = getattr(MemoryStore, "add_contradicts_edge", None)
    if _orig_add_contradicts is not None:
        def _add_contradicts_then_flush(self, *args, **kwargs):
            result = _orig_add_contradicts(self, *args, **kwargs)
            if _opt_out():
                return result
            try:
                flush_edge_buffer(self)
            except Exception:  # noqa: BLE001
                pass
            return result

        monkeypatch.setattr(
            MemoryStore, "add_contradicts_edge", _add_contradicts_then_flush,
        )

    try:
        from iai_mcp import provenance_buffer as _prov_buf_mod
    except Exception:  # noqa: BLE001
        _prov_buf_mod = None

    if _prov_buf_mod is not None:
        _orig_defer = _prov_buf_mod.defer_provenance
        _orig_flush = _prov_buf_mod.flush_deferred_provenance

        def _defer_then_flush(store, entries):
            result = _orig_defer(store, entries)
            if _opt_out():
                return result
            try:
                _orig_flush(store)
            except Exception:  # noqa: BLE001
                pass
            return result

        monkeypatch.setattr(_prov_buf_mod, "defer_provenance", _defer_then_flush)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.slow (subprocess-heavy bench-shim resolution checks)",
    )
    parser.addoption(
        "--perf",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.perf (wall-clock latency benches, out of the default gate)",
    )
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="run @pytest.mark.live integration tests (real daemon subprocess; out of the default correctness gate)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    if not config.getoption("--perf"):
        skip_perf = pytest.mark.skip(reason="need --perf to run wall-clock bench")
        for item in items:
            if "perf" in item.keywords:
                item.add_marker(skip_perf)
    if not config.getoption("--live"):
        skip_live = pytest.mark.skip(reason="need --live to run the real-daemon E2E gate")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)


@pytest.fixture()
def short_socket():
    d = Path(tempfile.mkdtemp(prefix="iai-sock-"))
    sock = d / "d.sock"
    yield sock
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def hermetic_store(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    store_root = tmp_path / ".iai-mcp"
    store_root.mkdir(parents=True, exist_ok=True)
    dead_socket = tmp_path / "no-such.sock"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(dead_socket))
    yield store_root


@pytest.fixture
def _pdf_fixture(tmp_path):
    """Build a 1-page PDF in tmp_path on demand for ingestion tests.

    Skipped if reportlab is not installed (dev-only dep).
    """
    try:
        from reportlab.pdfgen import canvas
    except ImportError:
        pytest.skip("reportlab not installed (dev-only PDF fixture builder)")
    pdf_path = tmp_path / "fixture.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "This is a test PDF body for the upload integration test.")
    c.save()
    yield pdf_path


@pytest.fixture(autouse=True)
def _reset_module_singletons() -> None:
    try:
        import iai_mcp.runtime_graph_cache as _rgc
        with _rgc._GEN_LOCK:
            _rgc._current_generation = 0
            _rgc._rebuild_timestamp_override = ""
        _rgc.reset_dirty_counter()
    except Exception:  # noqa: BLE001 -- not yet installed in some test envs
        pass

    try:
        import iai_mcp.semantic_recall as _sr
        _sr._WARM_LOCAL_STORE = None
    except Exception:  # noqa: BLE001 -- not yet installed in some test envs
        pass


def _nx_graph_to_memory_graph(nx_g):
    from uuid import uuid4

    from iai_mcp.graph import MemoryGraph

    mg = MemoryGraph()
    node_to_uuid = {n: uuid4() for n in nx_g.nodes()}
    for _n, uid in node_to_uuid.items():
        mg.add_node(uid, community_id=None, embedding=[0.0] * 384)
    for u, v, data in nx_g.edges(data=True):
        w = 1.0
        try:
            w = float(data.get("weight", 1.0))
        except (TypeError, ValueError):
            w = 1.0
        mg.add_edge(node_to_uuid[u], node_to_uuid[v], weight=w)
    return mg

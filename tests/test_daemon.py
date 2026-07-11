from __future__ import annotations

import asyncio
import plistlib
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLIST_PATH = PROJECT_ROOT / "src" / "iai_mcp" / "_deploy" / "launchd" / "com.iai-mcp.daemon.plist"
SERVICE_PATH = PROJECT_ROOT / "src" / "iai_mcp" / "_deploy" / "systemd" / "iai-mcp-daemon.service"


def _fresh_store(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "384")
    from iai_mcp.store import MemoryStore
    return MemoryStore()


def _short_socket_paths(tmp_path, monkeypatch):
    import os
    from iai_mcp import concurrency
    lock_path = tmp_path / ".lock"
    sock_dir = Path(f"/tmp/iai-daemon-{os.getpid()}-{id(tmp_path)}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"
    monkeypatch.setattr(concurrency, "SOCKET_PATH", sock_path)
    return lock_path, sock_path, sock_dir


def test_main_clean_shutdown(tmp_path, monkeypatch):
    from iai_mcp import daemon as daemon_mod
    from iai_mcp import daemon_state as ds_mod

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "384")
    monkeypatch.setattr(ds_mod, "STATE_PATH", tmp_path / ".daemon-state.json")
    _short_socket_paths(tmp_path, monkeypatch)

    def _fake_embedder(store):
        class _Stub:
            def embed(self, text):
                return [0.0]
        return _Stub()
    monkeypatch.setattr("iai_mcp.embed.embedder_for_store", _fake_embedder)

    async def runner():
        task = asyncio.create_task(daemon_mod.main())
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            return await task
        except asyncio.CancelledError:
            return 0

    rc = asyncio.run(runner())
    assert rc == 0


def test_state_machine_transitions(tmp_path, monkeypatch):
    from iai_mcp import daemon as daemon_mod
    from iai_mcp import daemon_state as ds_mod

    monkeypatch.setattr(ds_mod, "STATE_PATH", tmp_path / ".daemon-state.json")

    state: dict = {}

    daemon_mod.transition(state, daemon_mod.STATE_TRANSITIONING)
    assert state["fsm_state"] == daemon_mod.STATE_TRANSITIONING

    daemon_mod.transition(state, daemon_mod.STATE_SLEEP)
    assert state["fsm_state"] == daemon_mod.STATE_SLEEP

    daemon_mod.transition(state, daemon_mod.STATE_DREAMING)
    assert state["fsm_state"] == daemon_mod.STATE_DREAMING

    with pytest.raises(ValueError, match="Illegal transition"):
        daemon_mod.transition(state, daemon_mod.STATE_TRANSITIONING)
    assert state["fsm_state"] == daemon_mod.STATE_DREAMING

    daemon_mod.transition(state, daemon_mod.STATE_SLEEP)
    assert state["fsm_state"] == daemon_mod.STATE_SLEEP

    daemon_mod.transition(state, daemon_mod.STATE_WAKE)
    assert state["fsm_state"] == daemon_mod.STATE_WAKE

    with pytest.raises(ValueError):
        daemon_mod.transition(state, daemon_mod.STATE_SLEEP)

    loaded = ds_mod.load_state()
    assert loaded["fsm_state"] == daemon_mod.STATE_WAKE


def test_scheduler_tick_survives_exceptions(tmp_path, monkeypatch):
    from iai_mcp import daemon as daemon_mod

    store = _fresh_store(tmp_path, monkeypatch)

    monkeypatch.setattr(daemon_mod, "TICK_INTERVAL_SEC", 0)

    state: dict = {}

    call_count = {"n": 0}

    async def flaky_body(store, state):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated tick failure")

    async def runner():
        task = asyncio.create_task(
            daemon_mod._scheduler_tick(store, state, tick_body=flaky_body)
        )
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(runner())

    assert call_count["n"] >= 2, (
        f"tick loop did not continue past first exception; only {call_count['n']} calls"
    )
    from iai_mcp.events import query_events
    err_events = query_events(store, kind="tick_error", limit=5)
    assert len(err_events) >= 1
    assert "simulated tick failure" in err_events[0]["data"].get("error", "")


def test_prewarm_called_once_at_boot(tmp_path, monkeypatch):
    from iai_mcp import daemon as daemon_mod
    from iai_mcp import daemon_state as ds_mod

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "384")
    monkeypatch.setattr(ds_mod, "STATE_PATH", tmp_path / ".daemon-state.json")
    _short_socket_paths(tmp_path, monkeypatch)

    prewarm_calls = {"n": 0}

    class _StubEmbedder:
        def embed(self, text):
            prewarm_calls["n"] += 1
            return [0.0]

    def _fake_embedder(store):
        return _StubEmbedder()

    monkeypatch.setattr("iai_mcp.embed.embedder_for_store", _fake_embedder)

    async def runner():
        task = asyncio.create_task(daemon_mod.main())
        await asyncio.sleep(0.15)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(runner())
    assert prewarm_calls["n"] == 1, (
        f"prewarm expected once, got {prewarm_calls['n']}"
    )


def test_empty_store_shortcut(tmp_path, monkeypatch):
    from iai_mcp import daemon as daemon_mod

    store = _fresh_store(tmp_path, monkeypatch)
    state: dict = {"fsm_state": "WAKE"}

    async def run_once():
        await daemon_mod._tick_body(store, state)

    asyncio.run(run_once())

    assert state.get("last_tick_skipped_reason") == "empty_store"

    from iai_mcp.events import query_events
    rem = query_events(store, kind="rem_cycle_started", limit=5)
    assert rem == []


def test_launchd_plist_valid_xml_with_required_keys():
    assert PLIST_PATH.exists(), f"missing plist at {PLIST_PATH}"

    with open(PLIST_PATH, "rb") as f:
        data = plistlib.load(f)

    assert data["Label"] == "com.iai-mcp.daemon"
    assert data["ProgramArguments"][-1] == "iai_mcp.daemon"
    assert data["RunAtLoad"] is True

    keepalive = data["KeepAlive"]
    assert isinstance(keepalive, dict)
    assert keepalive.get("Crashed") is True
    # SuccessfulExit=false: launchd does not count SIGKILL (jetsam/OOM) as
    # Crashed — without this key a killed daemon stays down until a manual
    # start. Graceful exits (hibernation, stop) are 0 and stay down.
    assert keepalive.get("SuccessfulExit") is False

    assert data["ThrottleInterval"] == 5
    assert "StandardOutPath" in data
    assert "StandardErrorPath" in data
    assert "WorkingDirectory" in data

    env = data["EnvironmentVariables"]
    for required_key in ("PATH", "IAI_MCP_STORE", "HOME", "LANG"):
        assert required_key in env, f"missing env key {required_key}"

    assert "ANTHROPIC_API_KEY" not in env


def test_systemd_unit_required_keys():
    assert SERVICE_PATH.exists(), f"missing unit file at {SERVICE_PATH}"
    text = SERVICE_PATH.read_text()

    assert "[Unit]" in text
    assert "Description=" in text
    assert "[Service]" in text
    assert "Type=simple" in text
    assert "Restart=on-failure" in text
    assert "RestartSec=30" in text
    assert "StartLimitIntervalSec=60" in text
    assert "StartLimitBurst=3" in text
    assert "python3 -m iai_mcp.daemon" in text
    assert "StandardOutput=journal" in text
    assert "StandardError=journal" in text
    assert "SyslogIdentifier=iai-mcp-daemon" in text
    assert "TimeoutStopSec=60" in text
    assert "KillSignal=SIGTERM" in text
    assert "[Install]" in text
    assert "WantedBy=default.target" in text


def test_c3_no_anthropic_api_key_in_artifacts():
    daemon_dir = PROJECT_ROOT / "src" / "iai_mcp" / "daemon"
    daemon_src = (
        (daemon_dir / "__init__.py").read_text()
        + "\n"
        + (daemon_dir / "_watchdog.py").read_text()
    )
    plist_src = PLIST_PATH.read_text()
    service_src = SERVICE_PATH.read_text()

    for name, src in (("daemon", daemon_src), ("plist", plist_src), ("service", service_src)):
        assert "ANTHROPIC_API_KEY" not in src, (
            f"C3 VIOLATION: ANTHROPIC_API_KEY found in {name}"
        )


@pytest.fixture
def _restore_embedder_funnel_after():
    import iai_mcp.embed as _embed_mod

    _orig = _embed_mod.embedder_for_store
    try:
        yield _orig
    finally:
        _embed_mod.embedder_for_store = _orig


class _IdentityStub:

    def embed(self, text):
        return [0.0] * 384


def test_daemon_boot_holds_one_embedder_singleton(
    tmp_path, monkeypatch, _restore_embedder_funnel_after
):
    import iai_mcp.embed as _embed_mod
    from iai_mcp import daemon as daemon_mod

    construct_calls = {"n": 0}

    def _fake_funnel(store):
        construct_calls["n"] += 1
        return _IdentityStub()

    monkeypatch.setattr(_embed_mod, "embedder_for_store", _fake_funnel)

    class _StubStore:
        embed_dim = 384

    store = _StubStore()
    orig_efs, installed = daemon_mod._install_warm_embedder_override(store)

    assert installed is True
    assert construct_calls["n"] == 1
    held = _embed_mod.embedder_for_store(store)
    assert _embed_mod.embedder_for_store(store) is held
    assert _embed_mod.embedder_for_store(store) is held
    assert construct_calls["n"] == 1, "override must not reconstruct per call"

    daemon_mod._restore_embedder_funnel(orig_efs, installed)
    assert _embed_mod.embedder_for_store is _fake_funnel
    assert _embed_mod.embedder_for_store(store) is not held


def test_daemon_prewarm_failure_is_non_fatal(
    tmp_path, monkeypatch, _restore_embedder_funnel_after
):
    import iai_mcp.embed as _embed_mod
    from iai_mcp import daemon as daemon_mod

    def _raising_funnel(store):
        raise RuntimeError("simulated construct failure")

    monkeypatch.setattr(_embed_mod, "embedder_for_store", _raising_funnel)

    class _StubStore:
        embed_dim = 384
        root = tmp_path

    store = _StubStore()
    orig_efs, installed = daemon_mod._install_warm_embedder_override(store)

    assert installed is False, "build failure must not install the override"
    assert orig_efs is _raising_funnel
    assert _embed_mod.embedder_for_store is _raising_funnel
    daemon_mod._restore_embedder_funnel(orig_efs, installed)
    assert _embed_mod.embedder_for_store is _raising_funnel


def test_daemon_early_lock_conflict_does_not_leak_override(
    tmp_path, monkeypatch, _restore_embedder_funnel_after
):
    import iai_mcp.embed as _embed_mod
    from iai_mcp import daemon as daemon_mod
    from iai_mcp import daemon_state as ds_mod
    from iai_mcp import lifecycle_lock as ll_mod

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "384")
    monkeypatch.setattr(ds_mod, "STATE_PATH", tmp_path / ".daemon-state.json")
    _short_socket_paths(tmp_path, monkeypatch)

    sentinel_funnel = _embed_mod.embedder_for_store

    def _conflict(self):
        raise ll_mod.LifecycleLockConflict("simulated live-PID conflict")

    monkeypatch.setattr(ll_mod.LifecycleLock, "acquire", _conflict)

    rc = asyncio.run(daemon_mod.main())

    assert rc == 1, "lock conflict must return exit code 1"
    assert _embed_mod.embedder_for_store is sentinel_funnel, (
        "override must NOT be installed/leaked on an early lock-conflict return"
    )


def test_daemon_boot_raise_after_install_restores_funnel(
    tmp_path, monkeypatch, _restore_embedder_funnel_after
):
    import iai_mcp.embed as _embed_mod
    from iai_mcp import daemon as daemon_mod
    from iai_mcp import daemon_state as ds_mod

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "384")
    monkeypatch.setattr(ds_mod, "STATE_PATH", tmp_path / ".daemon-state.json")
    _short_socket_paths(tmp_path, monkeypatch)

    def _stub_funnel(store):
        class _S:
            def embed(self, text):
                return [0.0] * 384
        return _S()

    monkeypatch.setattr(_embed_mod, "embedder_for_store", _stub_funnel)
    pre_install_funnel = _embed_mod.embedder_for_store
    assert pre_install_funnel is _stub_funnel

    def _raising_save_state(state):
        raise RuntimeError("simulated post-install boot failure")

    monkeypatch.setattr(daemon_mod, "save_state", _raising_save_state)

    with pytest.raises(RuntimeError, match="simulated post-install boot failure"):
        asyncio.run(daemon_mod.main())

    assert _embed_mod.embedder_for_store is pre_install_funnel, (
        "funnel override leaked: restore did not run on a post-install boot raise"
    )


def test_boot_preload_does_not_call_runtime_graph_cache_save_with_empty_payload(
    monkeypatch
):
    """boot_preload must not call runtime_graph_cache.save with an empty (None)
    node payload after build_runtime_graph returns; build_runtime_graph is the
    single save authority.  preload_ready must still be set regardless.

    We drive a coroutine that mirrors the production _boot_preload body exactly
    so we test the logic without spinning up the full daemon.
    """
    import iai_mcp.retrieve as _retrieve_mod
    from iai_mcp import runtime_graph_cache as _rgc_mod

    build_calls: list = []
    save_calls: list = []

    _stub_triple = (object(), object(), object())

    def _fake_build(store):
        build_calls.append(True)
        return _stub_triple

    def _fake_save(*args, **kwargs):
        save_calls.append(args)

    monkeypatch.setattr(_retrieve_mod, "build_runtime_graph", _fake_build)
    monkeypatch.setattr(_rgc_mod, "save", _fake_save)

    _rgc_mod.preload_ready.clear()

    # Mirror the production _boot_preload body (daemon/__init__.py) exactly
    # so that refactors surface here without requiring a full daemon boot.
    store_stub = object()

    async def _boot_preload_mirror():
        try:
            from iai_mcp import retrieve as _retrieve_preload
            await asyncio.to_thread(
                _retrieve_preload.build_runtime_graph, store_stub,
            )
        except Exception as _exc:  # noqa: BLE001
            pass
        finally:
            _rgc_mod.preload_ready.set()

    asyncio.run(_boot_preload_mirror())

    assert build_calls, "build_runtime_graph must have been called during boot_preload"
    # The clobber: save called with node_payload=None (args[3]).
    empty_payload_saves = [a for a in save_calls if len(a) >= 4 and a[3] is None]
    assert not empty_payload_saves, (
        "runtime_graph_cache.save must NOT be called with an empty (None) node "
        "payload from boot_preload — build_runtime_graph is the single save authority"
    )
    assert _rgc_mod.preload_ready.is_set(), "preload_ready must be set after boot_preload completes"

    _rgc_mod.preload_ready.clear()

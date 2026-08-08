"""Observability tail: stop-verb orphan sweep, stderr log cap, embed-identity
visibility in daemon status and doctor.

The sweep tests never call psutil.process_iter — every sweep call passes an
explicit proc_iter of decoy children this file spawned itself, scoped to a
throwaway store path, and conftest additionally arms the
IAI_MCP_DISABLE_ORPHAN_SWEEP kill-switch for the whole suite (tests that
exercise the sweep delete it).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time


def test_daemon_title_matcher_is_own_identity_only() -> None:
    from iai_mcp.cli._daemon import _matches_daemon_title
    from iai_mcp.lifecycle_lock import daemon_process_title

    title = daemon_process_title("/tmp/some-store")
    assert _matches_daemon_title([title], title)
    assert _matches_daemon_title([title + "  "], title)
    # The live rewritten-argv shape: title in argv[0], padding tail.
    assert _matches_daemon_title([title, "", ""], title)
    # Substring in EITHER direction must not match: a sibling store whose
    # path extends this one, or a truncated spelling.
    assert not _matches_daemon_title([title + "-extended"], title)
    assert not _matches_daemon_title([title[:-5]], title)
    # A multi-field command whose JOINED line equals the title must not
    # match either — join matching is the argument-carrier hole.
    assert not _matches_daemon_title(title.split(" "), title)
    # The title as an ARGUMENT of another command must NOT match — an
    # any-field rule fells pgrep/pkill/grep carriers of the title.
    assert not _matches_daemon_title(["pgrep", "-f", title], title)
    assert not _matches_daemon_title(["pkill", "-f", title], title)
    assert not _matches_daemon_title(["python3", "-c", "sleep", title], title)
    # Substring mentions of the module must NOT match.
    assert not _matches_daemon_title(["rg", "-m", "1", "iai_mcp.daemon"], title)
    assert not _matches_daemon_title(
        ["tail", "-f", "iai lilli (iai_mcp.daemon).log"], title
    )
    assert not _matches_daemon_title(["python3", "-m", "iai_mcp.daemon"], title)
    assert not _matches_daemon_title([], title)
    # A daemon of a DIFFERENT store must not match this store's title.
    other = daemon_process_title("/tmp/other-store")
    assert not _matches_daemon_title([other], title)


def test_daemon_sets_the_title_the_sweep_targets(tmp_path, monkeypatch) -> None:
    """The setter and the matcher must share one title builder — drift
    silently disables sweeping."""
    import types

    captured: dict = {}
    fake = types.ModuleType("setproctitle")
    fake.setproctitle = lambda t: captured.__setitem__("title", t)
    monkeypatch.setitem(sys.modules, "setproctitle", fake)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))

    import iai_mcp.daemon as daemon_mod
    from iai_mcp.cli._daemon import _matches_daemon_title
    from iai_mcp.lifecycle_lock import daemon_process_title

    daemon_mod._set_process_title()
    assert captured["title"] == daemon_process_title(tmp_path)
    assert _matches_daemon_title([captured["title"]], daemon_process_title(tmp_path))


def _spawn_decoy(title: str) -> subprocess.Popen:
    # The title travels via env, never argv: an argv carrier would satisfy
    # a substring readiness check BEFORE setproctitle ran, and the sweep
    # would race a decoy that does not yet bear the title. Readiness uses
    # the production matcher itself for the same reason.
    import psutil

    from iai_mcp.cli._daemon import _matches_daemon_title

    code = (
        "import os, time, setproctitle; "
        "setproctitle.setproctitle(os.environ['IAI_TEST_DECOY_TITLE']); "
        "time.sleep(120)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        env={**os.environ, "IAI_TEST_DECOY_TITLE": title},
    )
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            cmdline = [str(x) for x in psutil.Process(proc.pid).cmdline()]
        except psutil.Error:
            cmdline = []
        if _matches_daemon_title(cmdline, title):
            return proc
        time.sleep(0.05)
    proc.kill()
    raise AssertionError("decoy never adopted the requested title")


def test_sweep_kills_own_store_decoy_only(tmp_path, monkeypatch) -> None:
    import psutil

    from iai_mcp.cli._daemon import _sweep_orphan_daemon_processes
    from iai_mcp.lifecycle_lock import daemon_process_title

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.delenv("IAI_MCP_DISABLE_ORPHAN_SWEEP", raising=False)

    mine = _spawn_decoy(daemon_process_title(tmp_path))
    other = _spawn_decoy(daemon_process_title(tmp_path / "other-store"))
    try:
        swept = _sweep_orphan_daemon_processes(
            set(),
            proc_iter=[psutil.Process(mine.pid), psutil.Process(other.pid)],
            term_timeout=5.0,
        )
        assert swept == [mine.pid]
        assert mine.wait(timeout=5.0) is not None
        assert other.poll() is None, "foreign-store decoy must survive"
    finally:
        for p in (mine, other):
            try:
                p.kill()
            except OSError:
                pass


def test_sweep_excludes_self_by_pid(tmp_path, monkeypatch) -> None:
    """Self-exclusion pinned via a surrogate: os.getpid is patched to the
    second decoy's pid, so removing the exclusion kills it."""
    import psutil

    from iai_mcp.cli._daemon import _sweep_orphan_daemon_processes
    from iai_mcp.lifecycle_lock import daemon_process_title

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.delenv("IAI_MCP_DISABLE_ORPHAN_SWEEP", raising=False)

    title = daemon_process_title(tmp_path)
    victim = _spawn_decoy(title)
    surrogate = _spawn_decoy(title)
    try:
        monkeypatch.setattr(os, "getpid", lambda: surrogate.pid)
        swept = _sweep_orphan_daemon_processes(
            set(),
            proc_iter=[psutil.Process(victim.pid), psutil.Process(surrogate.pid)],
            term_timeout=5.0,
        )
        assert swept == [victim.pid]
        assert surrogate.poll() is None, "self (surrogate) must never be swept"
    finally:
        for p in (victim, surrogate):
            try:
                p.kill()
            except OSError:
                pass


def test_sweep_kill_switch_disables(tmp_path, monkeypatch) -> None:
    import psutil

    from iai_mcp.cli._daemon import _sweep_orphan_daemon_processes
    from iai_mcp.lifecycle_lock import daemon_process_title

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_DISABLE_ORPHAN_SWEEP", "1")

    decoy = _spawn_decoy(daemon_process_title(tmp_path))
    try:
        swept = _sweep_orphan_daemon_processes(
            set(),
            proc_iter=[psutil.Process(decoy.pid)],
            term_timeout=1.0,
        )
        assert swept == []
        assert decoy.poll() is None
    finally:
        decoy.kill()


def test_suite_conftest_arms_the_kill_switch() -> None:
    # The suite-wide guarantee the stop-verb unit tests rely on: their
    # cmd_daemon_stop calls must never reach psutil against real daemons.
    assert os.environ.get("IAI_MCP_DISABLE_ORPHAN_SWEEP") == "1"


def test_kill_switch_zero_means_enabled(tmp_path, monkeypatch) -> None:
    """Only explicit truthy values disable — '0' must NOT read as off."""
    import psutil

    from iai_mcp.cli._daemon import _sweep_orphan_daemon_processes
    from iai_mcp.lifecycle_lock import daemon_process_title

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_DISABLE_ORPHAN_SWEEP", "0")
    decoy = _spawn_decoy(daemon_process_title(tmp_path))
    try:
        swept = _sweep_orphan_daemon_processes(
            set(), proc_iter=[psutil.Process(decoy.pid)], term_timeout=5.0
        )
        assert swept == [decoy.pid]
    finally:
        try:
            decoy.kill()
        except OSError:
            pass


def test_stop_verb_wires_the_sweep(monkeypatch) -> None:
    """The sweep must be CALLED by cmd_daemon_stop — the suite kill-switch
    makes it inert in unit tests, so wiring needs its own pin."""
    import argparse

    import iai_mcp.cli._daemon as dmod
    from iai_mcp import cli as cli_pkg

    calls: list = []
    monkeypatch.setattr(
        dmod,
        "_sweep_orphan_daemon_processes",
        lambda excl, **k: calls.append(excl) or [],
    )
    monkeypatch.setattr(cli_pkg, "_is_macos", lambda: True)
    monkeypatch.setattr(
        cli_pkg.subprocess,
        "run",
        lambda *a, **k: type("R", (), {"returncode": 0})(),
    )

    import iai_mcp.lifecycle_lock as ll

    monkeypatch.setattr(ll.LifecycleLock, "read", lambda self: None)

    rc = dmod.cmd_daemon_stop(argparse.Namespace())
    assert rc == 0
    assert len(calls) == 1, "cmd_daemon_stop must invoke the orphan sweep"

    # The Linux branch must wire it too.
    calls.clear()
    monkeypatch.setattr(cli_pkg, "_is_macos", lambda: False)
    monkeypatch.setattr(cli_pkg, "_is_linux", lambda: True)
    rc = dmod.cmd_daemon_stop(argparse.Namespace())
    assert rc == 0
    assert len(calls) == 1, "Linux cmd_daemon_stop must invoke the orphan sweep"


def test_daemon_boot_wires_identity_compute() -> None:
    """The boot path must compute and cache embed_identity — the status
    handler only serves the cache, so a dropped boot wire silently blanks
    the doctor row for every daemon-up host."""
    import inspect

    from iai_mcp import daemon as daemon_mod

    src = inspect.getsource(daemon_mod.main)
    assert "_status_embed_identity" in src
    assert 'state["embed_identity"]' in src


def test_stderr_rotation_caps_and_preserves_append_fd(tmp_path, monkeypatch) -> None:
    from iai_mcp.daemon import STDERR_LOG_CAP_ENV, _rotate_launchd_stderr

    log = tmp_path / "logs" / "launchd-stderr.log"
    log.parent.mkdir(parents=True)
    log.write_bytes(b"x" * 4096)
    monkeypatch.setenv(STDERR_LOG_CAP_ENV, "0.001")

    # The service manager holds an O_APPEND fd across the rotation.
    fd = os.open(log, os.O_WRONLY | os.O_APPEND)
    try:
        assert _rotate_launchd_stderr(log) is True
        rotated = log.parent / (log.name + ".1")
        assert rotated.stat().st_size == 4096
        assert log.stat().st_size == 0
        os.write(fd, b"after-rotation\n")
        assert log.read_bytes() == b"after-rotation\n"
    finally:
        os.close(fd)


def test_stderr_rotation_under_cap_is_noop(tmp_path, monkeypatch) -> None:
    from iai_mcp.daemon import STDERR_LOG_CAP_ENV, _rotate_launchd_stderr

    log = tmp_path / "launchd-stderr.log"
    log.write_bytes(b"small")
    monkeypatch.setenv(STDERR_LOG_CAP_ENV, "5")
    assert _rotate_launchd_stderr(log) is False
    assert log.read_bytes() == b"small"
    assert not (tmp_path / "launchd-stderr.log.1").exists()


def test_stderr_rotation_pathological_caps_never_crash(tmp_path, monkeypatch) -> None:
    from iai_mcp.daemon import STDERR_LOG_CAP_ENV, _rotate_launchd_stderr

    log = tmp_path / "launchd-stderr.log"
    log.write_bytes(b"y" * 4096)
    for raw in ("0", "-3", "inf", "nan", "1e308", "garbage", ""):
        monkeypatch.setenv(STDERR_LOG_CAP_ENV, raw)
        if raw in ("garbage", ""):
            # Unparseable falls back to the 5 MB default: under cap, no-op.
            assert _rotate_launchd_stderr(log) is False
        else:
            assert _rotate_launchd_stderr(log) is False
    assert log.stat().st_size == 4096


def test_stderr_rotation_missing_file_is_noop(tmp_path, monkeypatch) -> None:
    from iai_mcp.daemon import STDERR_LOG_CAP_ENV, _rotate_launchd_stderr

    monkeypatch.setenv(STDERR_LOG_CAP_ENV, "0.001")
    assert _rotate_launchd_stderr(tmp_path / "absent.log") is False


def test_status_serves_boot_cached_identity_without_computing() -> None:
    """The status handler must serve the state-cached block — recomputing
    inline would put a _conn_lock read on the liveness watchdog's path."""
    import asyncio

    from iai_mcp.concurrency import _dispatch_socket_request

    block = {"runtime": "m@pinned|pool=cls|dim=384",
             "stored": "m@pinned|pool=cls|dim=384", "match": True}
    resp = asyncio.run(
        _dispatch_socket_request(
            {"type": "status"}, object(), {"embed_identity": block}
        )
    )
    assert resp["embed_identity"] == block

    resp = asyncio.run(_dispatch_socket_request({"type": "status"}, object(), {}))
    assert resp["embed_identity"] is None


def test_boot_identity_compute_unstamped(tmp_path, monkeypatch) -> None:
    from iai_mcp.concurrency import _status_embed_identity
    from iai_mcp.store import MemoryStore

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    store = MemoryStore(str(tmp_path / "store"))
    try:
        ei = _status_embed_identity(store)
        assert isinstance(ei, dict)
        assert "error" not in ei, ei
        assert ei["stored"] is None
        assert ei["match"] is True
        assert isinstance(ei["runtime"], str) and ei["runtime"]
    finally:
        store.close()


def test_boot_identity_compute_mismatch(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    from iai_mcp.concurrency import _status_embed_identity
    from iai_mcp.embed import stamp_store_embed_identity
    from iai_mcp.store import MemoryStore

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.delenv("IAI_MCP_EMBED_TEXT_PREFIX", raising=False)
    store = MemoryStore(str(tmp_path / "store"))
    try:
        stamp_store_embed_identity(
            store,
            SimpleNamespace(model_name="foreign-model", DIM=384, model_key="f"),
        )
        ei = _status_embed_identity(store)
        assert ei["stored"] and "foreign-model" in ei["stored"]
        assert ei["match"] is False
    finally:
        store.close()


async def _probe_none(*_a, **_k):
    return None


def test_doctor_embed_identity_daemon_down_match(tmp_path, monkeypatch) -> None:
    from iai_mcp import doctor as doctor_pkg
    from iai_mcp.doctor._storage_checks import check_ii_embed_identity
    from iai_mcp.embed import Embedder, stamp_store_embed_identity
    from iai_mcp.store import MemoryStore

    store_root = tmp_path / "store"
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    monkeypatch.setattr(doctor_pkg, "_socket_status_probe", _probe_none)

    store = MemoryStore(str(store_root))
    stamp_store_embed_identity(store, Embedder())
    store.close()

    result = check_ii_embed_identity()
    assert result.passed, result.detail
    assert result.status == "PASS", result.detail
    assert "match" in result.detail


def test_doctor_embed_identity_daemon_down_mismatch(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    from iai_mcp import doctor as doctor_pkg
    from iai_mcp.doctor._storage_checks import check_ii_embed_identity
    from iai_mcp.embed import stamp_store_embed_identity
    from iai_mcp.store import MemoryStore

    store_root = tmp_path / "store"
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    monkeypatch.delenv("IAI_MCP_EMBED_TEXT_PREFIX", raising=False)
    monkeypatch.setattr(doctor_pkg, "_socket_status_probe", _probe_none)

    store = MemoryStore(str(store_root))
    stamp_store_embed_identity(
        store, SimpleNamespace(model_name="foreign-model", DIM=384, model_key="f")
    )
    store.close()

    result = check_ii_embed_identity()
    assert not result.passed, result.detail
    assert "MISMATCH" in result.detail
    assert "re-embedding migration" in result.detail


def test_doctor_embed_identity_daemon_down_unstamped(tmp_path, monkeypatch) -> None:
    from iai_mcp import doctor as doctor_pkg
    from iai_mcp.doctor._storage_checks import check_ii_embed_identity
    from iai_mcp.store import MemoryStore

    store_root = tmp_path / "store"
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    monkeypatch.setattr(doctor_pkg, "_socket_status_probe", _probe_none)

    store = MemoryStore(str(store_root))
    store.close()

    result = check_ii_embed_identity()
    assert result.passed
    assert result.status == "WARN"
    assert "unstamped" in result.detail


def test_doctor_embed_identity_daemon_up_uses_status(monkeypatch) -> None:
    from iai_mcp import doctor as doctor_pkg
    from iai_mcp.doctor._storage_checks import check_ii_embed_identity

    async def _probe_mismatch(*_a, **_k):
        return {
            "ok": True,
            "embed_identity": {
                "runtime": "model-b@pinned|pool=cls|dim=384",
                "stored": "model-a@pinned|pool=cls|dim=384",
                "match": False,
            },
        }

    monkeypatch.setattr(doctor_pkg, "_socket_status_probe", _probe_mismatch)
    result = check_ii_embed_identity()
    assert not result.passed
    assert "MISMATCH" in result.detail
    assert "model-a" in result.detail and "model-b" in result.detail

    async def _probe_match(*_a, **_k):
        return {
            "ok": True,
            "embed_identity": {
                "runtime": "model-a@pinned|pool=cls|dim=384",
                "stored": "model-a@pinned|pool=cls|dim=384",
                "match": True,
            },
        }

    monkeypatch.setattr(doctor_pkg, "_socket_status_probe", _probe_match)
    result = check_ii_embed_identity()
    assert result.passed and result.status == "PASS"


def test_doctor_roster_contains_embed_identity_row(monkeypatch) -> None:
    """The row must be wired into run_diagnosis, not merely defined."""
    import inspect

    from iai_mcp import doctor as doctor_pkg

    src = inspect.getsource(doctor_pkg.run_diagnosis)
    assert "check_ii_embed_identity()" in src


def test_doctor_daemon_down_selects_embedder_by_store_dim(
    tmp_path, monkeypatch
) -> None:
    """The store's embed_dim meta drives WHICH embedder the doctor compares
    against — a dropped dim silently compares an e5 store to the default."""
    from types import SimpleNamespace

    import iai_mcp.embed as embed_mod
    from iai_mcp import doctor as doctor_pkg
    from iai_mcp.doctor._storage_checks import check_ii_embed_identity
    from iai_mcp.embed import Embedder, stamp_store_embed_identity
    from iai_mcp.store import MemoryStore

    store_root = tmp_path / "store"
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    monkeypatch.setattr(doctor_pkg, "_socket_status_probe", _probe_none)

    store = MemoryStore(str(store_root))
    real = Embedder()
    stamp_store_embed_identity(store, real)
    expected_dim = int(store.embed_dim)
    store.close()

    seen: dict = {}
    _orig_efs = embed_mod.embedder_for_store

    def _recording_efs(shim, **kwargs):
        seen["embed_dim"] = getattr(shim, "embed_dim", None)
        return real

    monkeypatch.setattr(embed_mod, "embedder_for_store", _recording_efs)
    result = check_ii_embed_identity()
    assert result.passed, result.detail
    assert seen.get("embed_dim") == expected_dim, (
        f"doctor must pass the store's dim to embedder selection, "
        f"got {seen.get('embed_dim')!r}"
    )
    _ = SimpleNamespace  # keep the import shape used by sibling tests
    _ = _orig_efs


def test_doctor_daemon_down_absent_store_creates_nothing(
    tmp_path, monkeypatch
) -> None:
    """The diagnostic read must not mint a store: an empty stdlib-driver
    brain.sqlite3 left behind would misdirect the next daemon boot."""
    from iai_mcp import doctor as doctor_pkg
    from iai_mcp.doctor._storage_checks import check_ii_embed_identity

    store_root = tmp_path / "no-store-here"
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    monkeypatch.setattr(doctor_pkg, "_socket_status_probe", _probe_none)

    result = check_ii_embed_identity()
    assert result.passed
    assert result.status == "WARN"
    assert not (store_root / "hippo" / "brain.sqlite3").exists(), (
        "doctor row must never create a store file"
    )


def test_doctor_dim_selection_follows_meta_not_a_constant(
    tmp_path, monkeypatch
) -> None:
    """A sentinel embed_dim in meta must reach embedder selection verbatim —
    a hardcoded dimension passes the default-store test and nothing else."""
    import iai_mcp.embed as embed_mod
    from iai_mcp import doctor as doctor_pkg
    from iai_mcp.doctor._storage_checks import check_ii_embed_identity
    from iai_mcp.embed import Embedder, stamp_store_embed_identity
    from iai_mcp.store import MemoryStore

    store_root = tmp_path / "store"
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    monkeypatch.setattr(doctor_pkg, "_socket_status_probe", _probe_none)

    store = MemoryStore(str(store_root))
    real = Embedder()
    stamp_store_embed_identity(store, real)
    with store.db._conn_lock:
        store.db._conn.execute(
            "DELETE FROM _hippo_meta WHERE key = 'embed_dim'"
        )
        store.db._conn.execute(
            "INSERT OR IGNORE INTO _hippo_meta (key, value) VALUES (?, ?)",
            ("embed_dim", "999"),
        )
        store.db._conn.commit()
    store.close()

    seen: dict = {}

    def _recording_efs(shim, **kwargs):
        seen["embed_dim"] = getattr(shim, "embed_dim", None)
        return real

    monkeypatch.setattr(embed_mod, "embedder_for_store", _recording_efs)
    check_ii_embed_identity()
    assert seen.get("embed_dim") == 999, (
        f"selection must receive the meta dim verbatim, got {seen.get('embed_dim')!r}"
    )


def test_stderr_rotation_swallows_non_os_errors(tmp_path, monkeypatch) -> None:
    """Fail-soft covers EVERY exception class, not just OSError — a
    RuntimeError from home-dir resolution must not escape."""
    from pathlib import Path as _P

    from iai_mcp.daemon import STDERR_LOG_CAP_ENV, _rotate_launchd_stderr

    monkeypatch.setenv(STDERR_LOG_CAP_ENV, "0.001")
    monkeypatch.delenv("IAI_MCP_STORE", raising=False)

    def _raising_home():
        raise RuntimeError("no home directory resolvable")

    monkeypatch.setattr(_P, "home", staticmethod(_raising_home))
    assert _rotate_launchd_stderr() is False


def test_status_coerces_non_dict_identity_to_none() -> None:
    """A wrong-typed cache value must degrade to a missing row — never make
    the whole status reply unserializable."""
    import asyncio

    from iai_mcp.concurrency import _dispatch_socket_request

    async def _broken():  # pragma: no cover - never awaited
        return {}

    coro = _broken()
    try:
        resp = asyncio.run(
            _dispatch_socket_request(
                {"type": "status"}, object(), {"embed_identity": coro}
            )
        )
        assert resp["embed_identity"] is None
        import json

        json.dumps(resp)
    finally:
        coro.close()


def test_check_f_absent_store_creates_nothing(tmp_path, monkeypatch) -> None:
    """check_f shares the never-create rule: a doctor run on a fresh host
    must not mint a store file."""
    from iai_mcp.doctor import check_f_hippo_readable

    store_root = tmp_path / "fresh"
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))

    result = check_f_hippo_readable()
    assert result.passed
    assert result.status == "WARN"
    assert not (store_root / "hippo" / "brain.sqlite3").exists(), (
        "check_f must never create a store file"
    )


def test_doctor_never_mints_store_anywhere_in_roster(tmp_path, monkeypatch) -> None:
    """The never-create rule covers the WHOLE diagnosis run and the heal
    audit writer — one unguarded row invalidates the fresh-host report
    within the same run."""
    from iai_mcp import doctor as doctor_pkg
    from iai_mcp.doctor import RepairAction, _execute_action, run_diagnosis

    store_root = tmp_path / "fresh-host"
    store_root.mkdir()
    (store_root / "unrelated.txt").write_text("x")
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    monkeypatch.setattr(doctor_pkg, "_socket_status_probe", _probe_none)

    run_diagnosis(fetch_update=False)
    assert not (store_root / "hippo" / "brain.sqlite3").exists(), (
        "a diagnosis run must not create the store file"
    )

    action = RepairAction(
        label="noop",
        description="probe action",
        destructive=False,
        execute=lambda: (True, "ok", 1),
    )
    ok, _msg, _ms = _execute_action(action)
    assert ok
    assert not (store_root / "hippo" / "brain.sqlite3").exists(), (
        "the heal audit writer must not create the store file"
    )


def test_doctor_identity_row_daemon_up_without_block(monkeypatch) -> None:
    """A daemon that answers without an identity block is a pre-upgrade or
    broken-wire condition — the row must say so and name the remedy, not
    fall into a lock-blocked direct read."""
    from iai_mcp import doctor as doctor_pkg
    from iai_mcp.doctor._storage_checks import check_ii_embed_identity

    async def _probe_no_block(*_a, **_k):
        return {"ok": True, "embed_identity": None}

    monkeypatch.setattr(doctor_pkg, "_socket_status_probe", _probe_no_block)
    result = check_ii_embed_identity()
    assert result.passed
    assert result.status == "WARN"
    assert "restart the daemon" in result.detail

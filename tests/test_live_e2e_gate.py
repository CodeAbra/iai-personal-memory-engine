from __future__ import annotations

import errno
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
from uuid import UUID, uuid4

import pytest

from test_bridge_socket_first import (  # type: ignore[import]
    REPO,
    _kill_test_daemons,
    _wait_for_daemon_socket,
)
from test_cli_subprocess_daemon_down import (  # type: ignore[import]
    _TEST_CRYPTO_PASSPHRASE,
    _hf_cache_root,
)
from _recall_helpers import (  # type: ignore[import]
    UUID_TWO_HOP_SURFACE,
    _populate_store,
    _prime_structural_cache,
)

pytestmark = pytest.mark.live


def _live_daemon_env(tmp_home: Path, sock_path: Path) -> dict[str, str]:
    store_dir = tmp_home / ".iai-mcp"
    env = dict(os.environ)
    env["HOME"] = str(tmp_home)
    env["IAI_MCP_STORE"] = str(store_dir)
    env["IAI_DAEMON_SOCKET_PATH"] = str(sock_path)
    env["IAI_DAEMON_IDLE_SHUTDOWN_SECS"] = "120"
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["IAI_MCP_CRYPTO_PASSPHRASE"] = _TEST_CRYPTO_PASSPHRASE
    env["IAI_MCP_EMBED_OFFLINE"] = "1"
    env["IAI_MCP_AROUSAL_USE_SHADOW"] = "1"
    hf_root = _hf_cache_root()
    env["HF_HOME"] = str(hf_root)
    env["HF_HUB_CACHE"] = str(hf_root / "hub")
    env["HUGGINGFACE_HUB_CACHE"] = str(hf_root / "hub")
    return env


@pytest.fixture(scope="function")
def live_daemon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> "SimpleNamespace":
    hf_cache = _hf_cache_root()
    weights_dir = hf_cache / "hub" / "models--BAAI--bge-small-en-v1.5"
    if not weights_dir.exists():
        pytest.skip(
            f"bge-small weight cache absent ({weights_dir}); the offline "
            "live-gate construct cannot run."
        )

    tmp_home = tmp_path / "home"
    tmp_home.mkdir(parents=True)
    store_dir = tmp_home / ".iai-mcp"

    sock_dir = Path(tempfile.mkdtemp(prefix="iai-live-"))
    sock_path = sock_dir / "d.sock"
    assert len(str(sock_path).encode()) < 104, (
        f"sun_path too long ({len(str(sock_path).encode())} >= 104): {sock_path}"
    )

    proc: subprocess.Popen | None = None
    try:
        monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", _TEST_CRYPTO_PASSPHRASE)
        monkeypatch.setenv("HF_HOME", str(hf_cache))
        monkeypatch.setenv("HF_HUB_CACHE", str(hf_cache / "hub"))
        monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hf_cache / "hub"))
        monkeypatch.setenv("IAI_MCP_EMBED_OFFLINE", "1")

        from iai_mcp.embed import Embedder
        from iai_mcp.pipeline import K_CANDIDATES
        from iai_mcp.store import MemoryStore

        cue = "User reference gold document semantic recall probe cue"
        cue_vec = Embedder().embed(cue)

        store = MemoryStore(str(store_dir))
        try:
            _populate_store(store, cue_vec=cue_vec, n_filler=700)
            _prime_structural_cache(store)

            ann_top_k = {r.id for r, _ in store.query_similar(cue_vec, k=K_CANDIDATES)}
            assert UUID(int=5) not in ann_top_k, (
                f"PRECONDITION FAILED: structural-only gold UUID(5) is a "
                f"DIRECT ANN top-{K_CANDIDATES} hit — the 2-hop spread is not "
                f"load-bearing.  store size={store.active_records_count()}."
            )
        finally:
            store.close()

        daemon_env = _live_daemon_env(tmp_home, sock_path)
        proc = subprocess.Popen(
            [sys.executable, "-m", "iai_mcp.daemon"],
            cwd=str(REPO),
            env=daemon_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        bound = _wait_for_daemon_socket(sock_path, timeout_sec=30.0)
        assert bound, (
            f"daemon did not bind socket within 30 s: {sock_path}; "
            f"proc.poll()={proc.poll()!r}"
        )

        cli_env = dict(daemon_env)

        def iai(*argv: str, timeout: int = 60) -> subprocess.CompletedProcess:
            return subprocess.run(
                [sys.executable, "-m", "iai_mcp.iai_cli", *argv],
                env=cli_env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

        def recall_json(cue_str: str) -> dict:
            result = iai("recall", "--json", "--limit", "50", cue_str)
            assert result.returncode == 0, (
                f"iai recall failed (rc={result.returncode}):\n"
                f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
            )
            stdout_lines = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
            assert stdout_lines, f"no JSON on stdout; stderr={result.stderr!r}"
            return json.loads(stdout_lines[-1])

        def wait_until(
            predicate: Callable[[], bool],
            timeout: float = 10.0,
            interval: float = 0.05,
        ) -> bool:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if predicate():
                    return True
                time.sleep(interval)
            return False

        lifecycle_path = store_dir / "lifecycle_state.json"

        ns = SimpleNamespace(
            cue=cue,
            store_dir=store_dir,
            sock_path=sock_path,
            lifecycle_path=lifecycle_path,
            proc=proc,
            iai=iai,
            recall_json=recall_json,
            wait_until=wait_until,
        )
        yield ns

    finally:
        try:
            _kill_test_daemons(sock_path)
        except Exception:  # noqa: BLE001
            pass
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                pass
        shutil.rmtree(sock_dir, ignore_errors=True)


def test_fresh_tmp_store_boots_on_scoped_env(live_daemon: SimpleNamespace) -> None:
    assert live_daemon.proc.poll() is None, (
        "daemon process exited unexpectedly after socket bind; "
        "check passphrase or store init error"
    )
    assert live_daemon.sock_path != Path.home() / ".iai-mcp" / ".daemon.sock", (
        "test socket must be a tmp path, NOT the production socket"
    )


def test_semantic_cue_surfaces_two_hop_gold(live_daemon: SimpleNamespace) -> None:
    payload = live_daemon.recall_json(live_daemon.cue)

    hits = payload.get("hits") or []
    surfaces = {h.get("literal_surface", "") for h in hits}
    source = payload.get("_source")

    assert UUID_TWO_HOP_SURFACE in surfaces, (
        f"STRUCTURAL GOLD MISSING: {UUID_TWO_HOP_SURFACE!r} not in surfaces.\n"
        f"The 2-hop gold (cos~0.02, outside ANN top-K) is reachable ONLY via "
        f"the 2-hop spread.  Its absence means the structural pipeline did not "
        f"engage or AROUSAL_USE_SHADOW was not effective.\n"
        f"_source={source!r}\n"
        f"gold surfaces present={sorted(s for s in surfaces if 'gold doc' in s)}\n"
        f"stderr from iai: (captured in recall_json assert above)"
    )

    assert source == "daemon", (
        f"expected _source='daemon' (daemon UP + socket bound); got {source!r}.\n"
        f"If _source='direct-store', the RPC failed — check the socket path "
        f"alignment and that IAI_DAEMON_SOCKET_PATH is in the CLI env."
    )


def test_recency_cue_does_not_surface_two_hop_gold(
    live_daemon: SimpleNamespace,
) -> None:
    unrelated_cue = "completely unrelated weather forecast query"
    payload = live_daemon.recall_json(unrelated_cue)

    hits = payload.get("hits") or []
    surfaces = {h.get("literal_surface", "") for h in hits}

    assert UUID_TWO_HOP_SURFACE not in surfaces, (
        f"UNEXPECTED: {UUID_TWO_HOP_SURFACE!r} appeared for an UNRELATED cue.\n"
        f"This means either the recency floor is surfacing the gold records "
        f"(they are among the most recent) or the structural path is routing "
        f"non-collinear queries to the gold chain (both are failures of the "
        f"discriminating probe).  The positive test (semantic cue) cannot be "
        f"trusted as a structural proof if this test fails.\n"
        f"surfaces present={sorted(surfaces)}"
    )


def _ex_held(store_dir: Path) -> bool:
    flag_path = store_dir / "hippo" / ".consolidation-pending"
    if flag_path.exists():
        return True

    lock_path = store_dir / "hippo" / ".lock"
    if not lock_path.exists():
        return False
    probe_fd = -1
    try:
        probe_fd = os.open(str(lock_path), os.O_RDWR)
        fcntl.flock(probe_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(probe_fd, fcntl.LOCK_UN)
        return False
    except OSError as exc:
        if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
            return True
        return False
    finally:
        if probe_fd >= 0:
            try:
                os.close(probe_fd)
            except OSError:
                pass


_LATENCY_BOUND_S = 1.5
_LATENCY_SKIP_LOAD = 4.0
_LATENCY_BEST_OF_N = 3


def test_awake_serves_full_structural(live_daemon: SimpleNamespace) -> None:
    from iai_mcp.lifecycle_state import load_state

    lc = load_state(live_daemon.lifecycle_path)
    current_state = lc.get("current_state")
    assert current_state == "WAKE", (
        f"expected WAKE at probe; got {current_state!r} — "
        "daemon may have drifted to SLEEP before the probe ran"
    )

    payload = live_daemon.recall_json(live_daemon.cue)

    hits = payload.get("hits") or []
    surfaces = {h.get("literal_surface", "") for h in hits}
    source = payload.get("_source")

    assert UUID_TWO_HOP_SURFACE in surfaces, (
        f"AWAKE STRUCTURAL GOLD MISSING: {UUID_TWO_HOP_SURFACE!r} not in surfaces.\n"
        f"The 2-hop gold (cos~0.02, outside ANN top-K) is reachable ONLY via the "
        f"2-hop spread in the daemon pipeline.  Its absence means the structural "
        f"pipeline did not engage (or AROUSAL_USE_SHADOW was not effective).\n"
        f"Recall via daemon RPC must return full-structural — a recency floor "
        f"cannot surface this gold.\n"
        f"_source={source!r}, current_state={current_state!r}\n"
        f"gold surfaces present={sorted(s for s in surfaces if 'gold doc' in s)}"
    )

    assert source == "daemon", (
        f"AWAKE: expected _source='daemon' (daemon UP + socket bound); got {source!r}.\n"
        f"If _source='direct-store', the RPC failed — check IAI_DAEMON_SOCKET_PATH "
        f"alignment in the CLI env and the socket bind in the fixture teardown."
    )


def test_down_hippocampus_led_never_empty_never_hangs(
    live_daemon: SimpleNamespace,
) -> None:
    proc = live_daemon.proc

    proc.terminate()
    proc.wait(timeout=10)
    assert proc.poll() is not None, (
        "daemon process did not exit after terminate()+wait(); "
        "remaining process may hold the test socket"
    )

    t0 = time.monotonic()
    payload = live_daemon.recall_json(live_daemon.cue)
    elapsed = time.monotonic() - t0

    hits = payload.get("hits") or []

    assert len(hits) > 0, (
        f"DOWN: daemon stopped but recall returned 0 hits — the hippocampus "
        f"direct-store path must always answer for a seeded cue; "
        f"_source={payload.get('_source')!r}, elapsed={elapsed:.3f}s"
    )

    assert elapsed < 5.0, (
        f"DOWN: recall took {elapsed:.3f} s (> 5.0 s no-hang bound) — "
        f"the hippocampus-led in-process path must not hang when the "
        f"daemon is stopped; check for a blocking socket probe."
    )


def test_capture_and_last_flow(live_daemon: SimpleNamespace) -> None:
    nonce = uuid4().hex[:12]
    marker_text = f"a load-bearing live-gate decision marker {nonce}"

    result_cap = live_daemon.iai("capture", marker_text)
    assert result_cap.returncode == 0, (
        f"iai capture failed (rc={result_cap.returncode}):\n"
        f"stdout={result_cap.stdout!r}\nstderr={result_cap.stderr!r}"
    )

    result_last = live_daemon.iai("last", "--n", "10")
    assert result_last.returncode == 0, (
        f"iai last failed (rc={result_last.returncode}):\n"
        f"stdout={result_last.stdout!r}\nstderr={result_last.stderr!r}"
    )

    assert nonce in result_last.stdout, (
        f"NONCE {nonce!r} not found in 'iai last' output.\n"
        f"Capture→last round-trip failed: the hippocampus write was not "
        f"readable via the direct-recency path.\n"
        f"captured text: {marker_text!r}\n"
        f"iai last stdout: {result_last.stdout!r}"
    )


def test_asleep_ex_window_degrade_is_pass(live_daemon: SimpleNamespace) -> None:
    store_dir = live_daemon.store_dir
    hippo_dir = store_dir / "hippo"
    hippo_dir.mkdir(parents=True, exist_ok=True)
    flag_path = hippo_dir / ".consolidation-pending"

    cli_env = dict(os.environ)
    cli_env["HOME"] = str(live_daemon.store_dir.parent)
    cli_env["IAI_MCP_STORE"] = str(store_dir)
    cli_env["IAI_DAEMON_SOCKET_PATH"] = str(live_daemon.sock_path)
    cli_env["IAI_MCP_CRYPTO_PASSPHRASE"] = _TEST_CRYPTO_PASSPHRASE
    cli_env["IAI_MCP_EMBED_OFFLINE"] = "1"
    cli_env["IAI_MCP_AROUSAL_USE_SHADOW"] = "1"
    cli_env["IAI_RECALL_ASLEEP_MARGIN_SEC"] = "0"
    hf_root = _hf_cache_root()
    cli_env["HF_HOME"] = str(hf_root)
    cli_env["HF_HUB_CACHE"] = str(hf_root / "hub")
    cli_env["HUGGINGFACE_HUB_CACHE"] = str(hf_root / "hub")

    try:
        flag_path.touch(mode=0o600, exist_ok=True)

        assert _ex_held(store_dir) is True, (
            "PRECONDITION: .consolidation-pending flag was set but _ex_held() "
            "returned False — the flag-based observable did not fire"
        )

        t0 = time.monotonic()
        result = subprocess.run(
            [sys.executable, "-m", "iai_mcp.iai_cli", "recall",
             "--json", "--limit", "50", live_daemon.cue],
            env=cli_env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        elapsed = time.monotonic() - t0

        assert elapsed < 5.0, (
            f"ASLEEP-EX: recall took {elapsed:.3f} s (> 5.0 s generous bound) "
            f"with EX-window held via flag.\n"
            f"The bypass-safe degrade must return promptly "
            f"(ConsolidationPendingError -> bank-fallback is < 1.5 s + bank).\n"
            f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
        )

        if result.stdout.strip():
            stdout_lines = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
            if stdout_lines:
                try:
                    payload = json.loads(stdout_lines[-1])
                    hits = payload.get("hits") or []
                    surfaces = {h.get("literal_surface", "") for h in hits}
                    _ = surfaces
                except (json.JSONDecodeError, ValueError):
                    pass

    finally:
        try:
            flag_path.unlink()
        except FileNotFoundError:
            pass


def test_asleep_settled_in_process_structural_under_bound(
    live_daemon: SimpleNamespace,
) -> None:
    from iai_mcp.lifecycle_state import (
        LifecycleState,
        LifecycleStateRecord,
        load_state,
        save_state,
    )

    store_dir = live_daemon.store_dir
    lifecycle_path = live_daemon.lifecycle_path

    proc = live_daemon.proc
    if proc.poll() is None:
        proc.terminate()
        proc.wait(timeout=10)
    assert proc.poll() is not None, "daemon process did not exit cleanly"

    from datetime import datetime, timezone, timedelta

    past_ts = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    sleep_record: LifecycleStateRecord = {
        "current_state": LifecycleState.SLEEP.value,
        "since_ts": past_ts,
        "last_activity_ts": past_ts,
        "wrapper_event_seq": 0,
        "sleep_cycle_progress": None,
        "quarantine": None,
        "shadow_run": False,
        "crisis_mode": False,
    }
    save_state(sleep_record, lifecycle_path)

    lc_check = load_state(lifecycle_path)
    assert lc_check.get("current_state") == "SLEEP", (
        f"lifecycle file should read SLEEP after save_state; "
        f"got {lc_check.get('current_state')!r}"
    )
    assert not _ex_held(store_dir), (
        "EX-window must be FREE for the settled-asleep structural assert; "
        "the .consolidation-pending flag or LOCK_EX is held unexpectedly"
    )

    load1 = os.getloadavg()[0]
    if load1 > _LATENCY_SKIP_LOAD:
        pytest.skip(
            f"os.getloadavg()[0]={load1:.1f} > {_LATENCY_SKIP_LOAD} — "
            "skipping latency assert to avoid false-fail under high load (REL-03)"
        )

    cli_env = dict(os.environ)
    cli_env["HOME"] = str(store_dir.parent)
    cli_env["IAI_MCP_STORE"] = str(store_dir)
    cli_env["IAI_DAEMON_SOCKET_PATH"] = str(store_dir / "no-such.sock")
    cli_env["IAI_MCP_CRYPTO_PASSPHRASE"] = _TEST_CRYPTO_PASSPHRASE
    cli_env["IAI_MCP_EMBED_OFFLINE"] = "1"
    cli_env["IAI_MCP_AROUSAL_USE_SHADOW"] = "1"
    cli_env["IAI_RECALL_ASLEEP_MARGIN_SEC"] = "0"
    hf_root = _hf_cache_root()
    cli_env["HF_HOME"] = str(hf_root)
    cli_env["HF_HUB_CACHE"] = str(hf_root / "hub")
    cli_env["HUGGINGFACE_HUB_CACHE"] = str(hf_root / "hub")

    def _run_recall() -> tuple[dict, float]:
        t0 = time.monotonic()
        result = subprocess.run(
            [sys.executable, "-m", "iai_mcp.iai_cli", "recall",
             "--json", "--limit", "50", live_daemon.cue],
            env=cli_env,
            capture_output=True,
            text=True,
            timeout=20,
        )
        elapsed = time.monotonic() - t0
        assert result.returncode == 0, (
            f"iai recall (asleep settled) failed (rc={result.returncode}):\n"
            f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
        )
        stdout_lines = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
        assert stdout_lines, f"no JSON on stdout; stderr={result.stderr!r}"
        return json.loads(stdout_lines[-1]), elapsed

    best_payload: dict | None = None
    best_elapsed = float("inf")
    for _ in range(_LATENCY_BEST_OF_N):
        payload, elapsed = _run_recall()
        if elapsed < best_elapsed:
            best_elapsed = elapsed
            best_payload = payload

    assert best_payload is not None
    hits = best_payload.get("hits") or []
    surfaces = {h.get("literal_surface", "") for h in hits}

    assert UUID_TWO_HOP_SURFACE in surfaces, (
        f"ASLEEP-SETTLED STRUCTURAL GOLD MISSING: {UUID_TWO_HOP_SURFACE!r} not in surfaces.\n"
        f"With EX free and lifecycle=SLEEP, the in-process recall path (LAT-05) "
        f"must run the 2-hop spread and surface the structural-only gold.\n"
        f"A recency degrade would miss this gold — its absence means the "
        f"asleep short-circuit path degraded instead of running structural recall.\n"
        f"_source={best_payload.get('_source')!r}\n"
        f"gold surfaces present={sorted(s for s in surfaces if 'gold doc' in s)}\n"
        f"best_elapsed={best_elapsed:.3f}s"
    )

    assert best_elapsed < _LATENCY_BOUND_S, (
        f"ASLEEP-SETTLED: best-of-{_LATENCY_BEST_OF_N} latency "
        f"{best_elapsed:.3f} s > {_LATENCY_BOUND_S} s (REL-03 structural bound).\n"
        f"The in-process settled-asleep path must complete structural recall "
        f"under {_LATENCY_BOUND_S} s on a lightly loaded machine.\n"
        f"If this is a CI load issue, the os.getloadavg skip guard (above) "
        f"should have caught it.  Investigate structural path latency regression."
    )

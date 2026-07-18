"""Isolated-daemon socket churn gate — the deploy gate for the recall path.

Spawns a fully isolated daemon on a fresh clone of a LOCAL prod-scale
snapshot, hammers memory_recall + memory_capture through its socket, forces
a sleep cycle mid-stream, and asserts the awake path never starves:

  G0: boot settles to served recalls;
  G1: no recall exceeds the hard ceiling while the forced cycle is pending;
  G2: recalls settle back under the soft ceiling afterwards;
  G3: at most the one forced cycle completes (the post-cycle refractory
      window holds).

The snapshot is pointed at by ``IAI_CHURN_GATE_SRC`` (a store root with
``hippo/brain.sqlite3`` inside) and never ships with the repo — real stores
stay local-only; without the env var the gate skips. Runtime is minutes, so
the gate is opt-in via the env var rather than part of the default sweep.
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _warm_recall_repro_support import (  # noqa: E402
    await_socket,
    raw_recall,
    spawn_isolated_daemon,
)

HARD_MS = 8000.0
SETTLE_MS = 3000.0
CYCLE_WAIT_S = 420.0
COOLDOWN_OBSERVE_S = 150.0
PROBE_SLEEP_S = 6.0

_SRC = os.environ.get("IAI_CHURN_GATE_SRC")

pytestmark = pytest.mark.skipif(
    not _SRC or not (Path(_SRC) / "hippo" / "brain.sqlite3").exists(),
    reason="IAI_CHURN_GATE_SRC not set to a local prod-scale snapshot",
)


def _clone_store(src: Path, work: Path) -> Path:
    work.mkdir(parents=True)
    for name in (".crypto.key", "config.json", "lifecycle_state.json"):
        p = src / name
        if p.exists():
            subprocess.run(["cp", "-c", str(p), str(work / name)], check=True)
    (work / "hippo").mkdir()
    for p in (src / "hippo").iterdir():
        if p.name.startswith("tmp") or p.name == ".lock":
            continue
        subprocess.run(["cp", "-c", str(p), str(work / "hippo" / p.name)], check=True)
    rgc = src / "runtime_graph_cache.json"
    if rgc.exists():
        subprocess.run(["cp", "-c", str(rgc), str(work / rgc.name)], check=True)
    return work


def _raw_call(socket_path: Path, method: str, params: dict, timeout_s: float = 30.0) -> dict:
    import asyncio

    async def _runner() -> dict:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(socket_path)), timeout=timeout_s,
        )
        try:
            req = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
            writer.write((json.dumps(req) + "\n").encode("utf-8"))
            await writer.drain()
            line = await asyncio.wait_for(reader.readline(), timeout=timeout_s)
            if not line:
                raise ConnectionError("daemon closed connection")
            return json.loads(line.decode("utf-8"))
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except OSError:
                pass

    return asyncio.run(_runner())


def _completed_rebuilds(events_dir: Path) -> int:
    n = 0
    for ev_file in sorted(events_dir.glob("lifecycle-events-*.jsonl")):
        for line in open(ev_file):
            if '"RECALL_INDEX_REBUILD"' in line and '"sleep_step_completed"' in line:
                n += 1
    return n


def test_recall_survives_capture_churn_and_forced_cycle(tmp_path: Path):
    base = _clone_store(Path(_SRC), tmp_path / "gate-store")
    handle = spawn_isolated_daemon(
        base, "lilli",
        scratch_home=tmp_path / "gate-home",
        extra_env={"IAI_MCP_SLEEP_CYCLE_COOLDOWN_SEC": "14400"},
    )
    try:
        await_socket(handle.socket_path, timeout=60.0)

        warm: list[float] = []
        for _ in range(30):
            t0 = time.perf_counter()
            try:
                raw_recall(handle.socket_path, "churn gate warmup cue", timeout_s=30.0)
                warm.append((time.perf_counter() - t0) * 1000)
                if len(warm) >= 3 and all(w < SETTLE_MS for w in warm[-3:]):
                    break
            except Exception:
                warm.append(30000.0)
            time.sleep(2)
        assert warm and warm[-1] < SETTLE_MS, f"G0 boot never settled: {warm[-6:]}"

        stop = threading.Event()

        def capture_loop() -> None:
            i = 0
            while not stop.is_set():
                try:
                    _raw_call(
                        handle.socket_path, "memory_capture",
                        {"text": f"churn gate ambient capture {i} standing in for a turn",
                         "role": "user", "session_id": "churn-gate"},
                        timeout_s=20.0,
                    )
                except Exception:
                    pass
                i += 1
                time.sleep(0.7)

        threading.Thread(target=capture_loop, daemon=True).start()

        repo = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["HOME"] = str(tmp_path / "gate-home")
        env["IAI_MCP_STORE"] = str(base)
        env["IAI_DAEMON_SOCKET_PATH"] = str(handle.socket_path)
        force = subprocess.run(
            [str(repo / ".venv" / "bin" / "iai-mcp"), "daemon", "force-rem"],
            env=env, capture_output=True, timeout=60, text=True,
        )
        assert force.returncode == 0 and "rem_queued" in (force.stdout + force.stderr), (
            f"force-rem must land or every later gate is vacuous: "
            f"rc={force.returncode} out={force.stdout!r} err={force.stderr!r}"
        )

        events_dir = base / "logs"
        during: list[float] = []
        t_start = time.monotonic()
        while time.monotonic() - t_start < CYCLE_WAIT_S:
            t0 = time.perf_counter()
            try:
                raw_recall(handle.socket_path, "churn gate cycle cue", timeout_s=HARD_MS / 1000 + 5)
                during.append((time.perf_counter() - t0) * 1000)
            except Exception:
                during.append(float("inf"))
            if _completed_rebuilds(events_dir) >= 1:
                break
            time.sleep(PROBE_SLEEP_S)
        assert during and max(during) < HARD_MS, (
            f"G1 starvation: max={max(during):.0f}ms over {len(during)} probes"
        )
        assert _completed_rebuilds(events_dir) >= 1, (
            "the forced cycle never completed — G1/G2/G3 would all pass "
            "without exercising the cycle at all"
        )

        settle: list[float] = []
        t_obs = time.monotonic()
        while time.monotonic() - t_obs < COOLDOWN_OBSERVE_S:
            t0 = time.perf_counter()
            try:
                raw_recall(handle.socket_path, "churn gate settle cue", timeout_s=20.0)
                settle.append((time.perf_counter() - t0) * 1000)
            except Exception:
                settle.append(float("inf"))
            time.sleep(4)
        stop.set()
        assert settle and statistics.median(settle) <= SETTLE_MS, (
            f"G2 no settle: p50={statistics.median(settle):.0f}ms"
        )

        assert _completed_rebuilds(events_dir) <= 1, (
            "G3 cooldown breach: a second cycle completed inside the "
            "refractory window"
        )
    finally:
        handle.terminate()

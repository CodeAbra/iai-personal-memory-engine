"""Highest-value product flows driven over the real running daemon.

Every daemon-served flow asserts ``_source=='daemon'`` so a dead daemon can
never pass; the token-budget flow drives the real dispatch method over the
in-process socket transport; the daemon-down flow proves the awake-memory
invariant (the hippocampus answers daemon-independently); the sleep control-plane flow drives the real
``force_rem`` control message and asserts its deterministic, persisted
effect.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from _live_harness import (
    _send_jsonrpc,
    _send_raw,
    _with_socket_server,
    spawn_live_daemon,
)
from _perf_helpers import skip_if_loaded
from _recall_helpers import _deterministic_vec, _populate_store
from test_daemon_lifecycle_uat import (
    _assert_prod_daemon_alive_if_present,
    _prod_daemon_pid,
)

pytestmark = pytest.mark.live


@pytest.fixture(scope="function")
def live_daemon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> "SimpleNamespace":
    # Every flow in this file spawns an isolated daemon; a couple terminate
    # it mid-test (_kill_test_daemons/proc.terminate()). The real prod
    # daemon, if any, must stay alive and untouched throughout.
    prod_pid = _prod_daemon_pid()
    gen = spawn_live_daemon(tmp_path, monkeypatch)
    ns = next(gen)
    try:
        yield ns
    finally:
        try:
            next(gen)
        except StopIteration:
            pass
        _assert_prod_daemon_alive_if_present(prod_pid)


def test_capture_round_trips_to_recall_over_daemon(
    live_daemon: SimpleNamespace,
) -> None:
    nonce = uuid4().hex[:16]
    marker_text = f"alice load-bearing capture round trip marker {nonce}"

    result_cap = live_daemon.iai("capture", marker_text)
    assert result_cap.returncode == 0, (
        f"iai capture failed (rc={result_cap.returncode}):\n"
        f"stdout={result_cap.stdout!r}\nstderr={result_cap.stderr!r}"
    )

    payload = live_daemon.recall_json(marker_text)
    hits = payload.get("hits") or []
    surfaces = {h.get("literal_surface", "") for h in hits}

    assert any(nonce in s for s in surfaces), (
        f"nonce {nonce!r} not found in any recalled hit surface after a "
        f"capture->recall round trip over the live daemon; "
        f"surfaces={sorted(surfaces)!r}"
    )
    assert payload.get("_source") == "daemon", (
        f"expected _source=='daemon' (proving the RPC crossed the wire); "
        f"got {payload.get('_source')!r} -- a dead daemon must not pass "
        f"this flow"
    )


def test_health_over_socket_status_light_and_control_status(
    live_daemon: SimpleNamespace,
) -> None:
    light_resp = asyncio.run(
        _send_jsonrpc(live_daemon.sock_path, "status_light", {})
    )
    assert "error" not in light_resp, f"status_light returned an error: {light_resp!r}"
    light_result = light_resp.get("result") or {}
    assert isinstance(light_result.get("N"), int), (
        f"status_light did not return an integer corpus count over the "
        f"wire: {light_result!r}"
    )

    status_resp = asyncio.run(
        _send_raw(
            live_daemon.sock_path,
            (json.dumps({"type": "status"}) + "\n").encode("utf-8"),
        )
    )
    assert status_resp.get("ok") is True, (
        f"control-plane status over the socket did not report ok=True: "
        f"{status_resp!r}"
    )

    heartbeat = status_resp.get("last_tick_at") or status_resp.get("daemon_started_at")
    assert heartbeat, (
        f"no heartbeat field (last_tick_at/daemon_started_at) present in "
        f"the status reply: {status_resp!r}"
    )

    started_raw = status_resp.get("daemon_started_at")
    assert started_raw, f"status reply missing daemon_started_at: {status_resp!r}"
    started = datetime.fromisoformat(started_raw)
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    age_sec = (datetime.now(timezone.utc) - started).total_seconds()
    assert 0 <= age_sec < 120, (
        f"daemon_started_at is not a fresh heartbeat: age={age_sec:.1f}s, "
        f"value={started_raw!r}"
    )


def test_daemon_down_recall_still_answers_via_direct_store(
    live_daemon: SimpleNamespace,
) -> None:
    nonce = uuid4().hex[:16]
    marker_text = f"alice fallback awake-memory marker {nonce}"

    result_cap = live_daemon.iai("capture", marker_text)
    assert result_cap.returncode == 0, (
        f"iai capture failed (rc={result_cap.returncode}):\n"
        f"stdout={result_cap.stdout!r}\nstderr={result_cap.stderr!r}"
    )

    live_daemon.proc.terminate()
    live_daemon.proc.wait(timeout=10)
    assert live_daemon.proc.poll() is not None, (
        "daemon process did not exit after terminate()+wait(); a still-alive "
        "daemon would make the direct-store fallback assertion meaningless"
    )

    skip_if_loaded()

    t0 = time.monotonic()
    payload = live_daemon.recall_json(marker_text)
    elapsed = time.monotonic() - t0

    hits = payload.get("hits") or []
    assert len(hits) > 0, (
        f"the daemon is down but recall returned 0 hits -- the awake-memory "
        f"invariant requires the hippocampus to answer daemon-independent; "
        f"_source={payload.get('_source')!r}"
    )
    assert payload.get("_source") != "daemon", (
        f"recall reported _source=='daemon' with the daemon process dead "
        f"(poll()={live_daemon.proc.poll()!r}) -- this must be the "
        f"direct-store path, not a stale/impossible daemon answer"
    )
    assert elapsed < 5.0, (
        f"daemon-down recall took {elapsed:.3f}s (> 5.0s no-hang bound); "
        f"the direct-store path must not hang when the daemon is dead"
    )


@pytest.fixture
def short_sock_path(tmp_path: Path):
    sock_dir = Path(f"/tmp/iai-tokbudget-{os.getpid()}-{id(tmp_path)}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"
    assert len(str(sock_path).encode()) < 104, (
        f"sun_path too long ({len(str(sock_path).encode())} >= 104): {sock_path}"
    )
    try:
        yield sock_path
    finally:
        try:
            if sock_path.exists():
                sock_path.unlink()
        except OSError:
            pass
        try:
            sock_dir.rmdir()
        except OSError:
            pass


def test_session_start_token_budget_over_socket(
    tmp_path: Path, short_sock_path: Path,
) -> None:
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "token-budget-store")
    try:
        _populate_store(store, _deterministic_vec(), n_filler=50)
        assert store.active_records_count() > 0, (
            "seeding produced an empty store -- the non-vacuous budget "
            "assertion below would be meaningless"
        )

        async def _drive(sock_path: Path, store) -> dict:
            return await _send_jsonrpc(
                sock_path,
                "session_start_payload",
                {"session_id": "alice-session"},
            )

        resp = asyncio.run(_with_socket_server(short_sock_path, store, _drive))
    finally:
        store.close()

    assert "result" in resp, f"malformed session_start_payload response: {resp!r}"
    total_cached_tokens = resp["result"]["total_cached_tokens"]

    assert total_cached_tokens > 0, (
        f"session_start_payload returned total_cached_tokens=0 over a "
        f"non-empty seeded store -- the budget assertion would be vacuous "
        f"(an empty store legitimately returns 0, a different branch); "
        f"payload={resp['result']!r}"
    )
    assert total_cached_tokens <= 3000, (
        f"session_start_payload total_cached_tokens={total_cached_tokens} "
        f"exceeds the steady-state 3000-token budget invariant; "
        f"payload={resp['result']!r}"
    )


def test_force_rem_control_message_persists_pending_flag(
    live_daemon: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iai_mcp.daemon_state import daemon_state_path

    resp = asyncio.run(
        _send_raw(
            live_daemon.sock_path,
            (
                json.dumps({
                    "type": "force_rem",
                    "ts": datetime.now(timezone.utc).isoformat(),
                })
                + "\n"
            ).encode("utf-8"),
        )
    )
    assert resp.get("ok") is True and resp.get("reason") == "rem_queued", (
        f"force_rem control message was not accepted by the live daemon: "
        f"{resp!r}"
    )

    state_path = daemon_state_path(store_root=live_daemon.store_dir)

    def _flag_pending() -> bool:
        if not state_path.exists():
            return False
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return bool((data.get("force_rem_request") or {}).get("pending"))

    assert live_daemon.wait_until(_flag_pending, timeout=10.0), (
        f"force_rem_request pending flag never persisted to {state_path} "
        f"within 10s of the control message being accepted -- the "
        f"deterministic mandatory assertion for the sleep control-plane "
        f"flow failed"
    )

    # Best-effort tail: only meaningful once the tick loop has actually
    # driven a real sleep_step_completed row -- an empty log makes
    # check_cc_background_liveness a vacuous PASS, so that composition
    # never stands in for the mandatory assertion above.
    skip_if_loaded()

    from iai_mcp.lifecycle_event_log import LifecycleEventLog

    log = LifecycleEventLog(log_dir=live_daemon.store_dir / "logs")

    def _completed_row_with_evidence() -> "dict | None":
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            events = log.read_all(date_str=today)
        except OSError:
            return None
        for ev in events:
            if ev.get("event") != "sleep_step_completed":
                continue
            processed = ev.get("liveness_processed")
            if isinstance(processed, int) and processed > 0:
                return ev
        return None

    from iai_mcp.daemon import TICK_INTERVAL_SEC

    # Past at least one tick (the pending flag is only consumed on a tick),
    # plus margin for the first pipeline step to complete and log a row --
    # a bound shorter than one tick interval could never fire by construction.
    poll_bound_sec = TICK_INTERVAL_SEC + 15.0
    deadline = time.monotonic() + poll_bound_sec
    row: "dict | None" = None
    while time.monotonic() < deadline:
        row = _completed_row_with_evidence()
        if row is not None:
            break
        time.sleep(1.0)

    if row is None:
        pytest.skip(
            f"no sleep_step_completed row with liveness_processed>0 observed "
            f"within the {poll_bound_sec:.0f}s bounded poll window -- the "
            f"tick/cooldown gate did not drive a full pipeline cycle to "
            f"completion in test time; this tail is best-effort and never "
            f"the mandatory assertion"
        )

    monkeypatch.setenv("IAI_MCP_STORE", str(live_daemon.store_dir))
    from iai_mcp.doctor._lifecycle_checks import check_cc_background_liveness

    result = check_cc_background_liveness()
    assert result.status == "PASS", (
        f"check_cc_background_liveness did not PASS after a real "
        f"sleep_step_completed row with "
        f"liveness_processed={row.get('liveness_processed')!r} was "
        f"observed: status={result.status!r}, detail={result.detail!r}"
    )

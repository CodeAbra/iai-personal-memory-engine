"""Headline regression: warm-band socket recall on a freshly-migrated,
prod-scale lilli store under boot-time load.

This module reproduces the exact failure signature previously observed on a
freshly-migrated large lilli store: a drift-mismatched stale runtime-graph
cache plus a full deferred-capture backlog at boot cause socket
``memory_recall`` to stall for many seconds (up to a client timeout) instead
of answering in the warm band. Fired CONCURRENTLY (the same shape a real
cutover gate uses), the recalls previously serialized behind the boot
fleet's shared lock and blew the SLA.

WAVE-0 RED CONTRACT: on HEAD (before the warm-load-index + sleep-maintenance
fixes land), the latency assertion in this module IS EXPECTED TO FAIL. This
is a plain failing assertion, not an xfail -- the fix work that follows
drives it to green by restoring instant, warm-loaded recall off the awake
path and moving maintenance work into sleep.

Every store here is either a COPY of a preserved, non-live snapshot (never
opened in place) or the driver's stdlib no-op control. The daemon under test
is fully isolated: scratch HOME (the deferred-capture backlog directory is
anchored to Path.home() unconditionally, not to IAI_MCP_STORE), scratch store
root, a copied (never minted) crypto key, and a short mkdtemp socket
directory. The real prod daemon (if running) is asserted to be the SAME
process, still alive, before and after every test -- a raw mtime/size compare
of the live store is deliberately avoided since a running prod daemon
legitimately mutates its own store on its own sleep/tick cadence throughout
the test window, independent of anything this module does.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from tests._warm_recall_repro_support import (
    _real_home,
    await_socket,
    locate_repro_source,
    make_fresh_lilli_copy,
    raw_recall,
    seed_deferred_backlog,
    spawn_isolated_daemon,
)

try:
    import iai_mcp_native  # noqa: F401
except ImportError:
    pytest.skip(
        "iai_mcp_native is unavailable; this proof requires the native "
        "engine extension",
        allow_module_level=True,
    )

# Frozen gate cue set (mirrors the frozen-set discipline used by
# bench/boot_warmup_validation.py) -- fingerprinted so the set can never
# silently drift between runs.
_GATE_CUES = (
    "cutover debugging session",
    "deferred backlog fixture turn",
    "cybernetics research sub-agents",
    "phase recall latency",
    "memory consolidation cycle",
)
_GATE_CUES_SHA256 = "a85d5f95e18081f63b5a09b4f6c2d4b0dc40d388a19b2292354aa91d091e765d"

_WARM_SLA_SEC = 1.0
_CLIENT_TIMEOUT_SEC = 30.0
_SOCKET_AWAIT_TIMEOUT_SEC = 30.0


def _assert_frozen_cue_set() -> None:
    fingerprint = hashlib.sha256(
        "\n".join(_GATE_CUES).encode("utf-8")
    ).hexdigest()
    assert fingerprint == _GATE_CUES_SHA256, (
        f"the frozen gate cue set has drifted: expected fingerprint "
        f"{_GATE_CUES_SHA256}, got {fingerprint}. If the cue list changed "
        f"intentionally, update the frozen fingerprint alongside it."
    )


def _prod_daemon_pid() -> "int | None":
    """Best-effort real daemon pid, for the pid-attributed prod-untouched
    check (the 175-05 pattern): identify the daemon BEFORE the test by pid,
    and confirm afterward it is the SAME pid, still alive.

    A raw mtime/size snapshot of ``~/.iai-mcp/hippo`` is deliberately NOT
    used here -- a live prod daemon legitimately mutates files under that
    directory on its own sleep/tick cadence throughout the test window
    (observed: ``.consolidation-pending`` rewritten mid-run), so a byte/mtime
    compare would false-positive on ordinary prod daemon activity that has
    nothing to do with this test's isolated subprocess. The correct
    isolation guard is pid identity: the real prod daemon process must still
    be the SAME process throughout (never killed/replaced by anything this
    test does), which the scratch HOME + scratch store root + short-socket
    isolation already structurally guarantees.
    """
    state_path = _real_home() / ".iai-mcp" / ".daemon-state.json"
    if not state_path.exists():
        return None
    try:
        data = json.loads(state_path.read_text())
    except (OSError, ValueError):
        return None
    pid = data.get("daemon_pid")
    return int(pid) if isinstance(pid, int) else None


def _assert_prod_daemon_alive_if_present(pid: "int | None") -> None:
    if pid is None:
        return
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pytest.fail(
            f"the real prod daemon (pid {pid}) is no longer alive after "
            f"this test ran an isolated daemon -- possible cross-contamination"
        )
    except PermissionError:
        pass  # alive, just not signalable by this uid -- fine
    except OSError:
        pass


@pytest.mark.parametrize("driver", ["lilli", "stdlib"])
def test_warm_recall_prodscale_isolated_daemon(tmp_path: Path, driver: str) -> None:
    _assert_frozen_cue_set()

    src_db = locate_repro_source()
    if src_db is None:
        pytest.skip(
            "the prod-scale repro corpus snapshot is not present on this "
            "machine; this is a local-only fixture, skipping honestly "
            "rather than fabricating a synthetic corpus that would never "
            "reproduce the boot storm"
        )

    prod_daemon_pid = _prod_daemon_pid()

    base = tmp_path / "iso-base"
    scratch_home = tmp_path / "scratch-home"

    if driver == "lilli":
        report = make_fresh_lilli_copy(src_db, base, stale_cache=True)
        assert report.rows_copied.get("records", 0) > 0, (
            "the fresh lilli copy must contain migrated records"
        )
    else:
        # stdlib arm: no-op control proving the harness itself is sound.
        # A plain stdlib copy of the source db, no lilli migrate involved.
        import shutil as _shutil

        base.mkdir(parents=True, exist_ok=True)
        _shutil.copyfile(src_db, base / "brain.sqlite3")
        from tests._warm_recall_repro_support import _create_scratch_crypto_key

        _create_scratch_crypto_key(base)

    backlog_state = seed_deferred_backlog(base, scratch_home=scratch_home, n=50)

    handle = spawn_isolated_daemon(base, driver, scratch_home=scratch_home)
    try:
        await_socket(handle.socket_path, timeout=_SOCKET_AWAIT_TIMEOUT_SEC)

        # --- Boot-drain-fired assertion -------------------------------
        # Poll (bounded) until the seeded backlog file is consumed AND at
        # least one seeded record is recallable via its own fixture text --
        # a dropped file the daemon ignores would leave the backlog file in
        # place forever and no fixture text would ever surface in a recall.
        deadline = time.monotonic() + 60.0
        drain_fired = False
        last_resp: "dict | None" = None
        while time.monotonic() < deadline:
            if not backlog_state.backlog_path.exists():
                resp = raw_recall(
                    handle.socket_path,
                    backlog_state.seeded_texts[0],
                    timeout_s=_CLIENT_TIMEOUT_SEC,
                )
                last_resp = resp
                hits = (resp.get("result") or {}).get("hits") or []
                surfaces = " ".join(h.get("literal_surface", "") for h in hits)
                if backlog_state.session_id in surfaces or len(hits) > 0:
                    drain_fired = True
                    break
            time.sleep(0.5)
        assert drain_fired, (
            f"the boot drain never consumed the seeded deferred-capture "
            f"backlog at {backlog_state.backlog_path} (still exists: "
            f"{backlog_state.backlog_path.exists()}) or the seeded records "
            f"never became recallable; last recall response: {last_resp!r}"
        )

        # --- Concurrent gate-cue firing (the RED assertion) -----------
        import concurrent.futures

        def _timed_recall(cue: str) -> "tuple[str, float, dict]":
            t0 = time.monotonic()
            resp = raw_recall(handle.socket_path, cue, timeout_s=_CLIENT_TIMEOUT_SEC)
            elapsed = time.monotonic() - t0
            return cue, elapsed, resp

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(_GATE_CUES)) as ex:
            futures = [ex.submit(_timed_recall, cue) for cue in _GATE_CUES]
            results = [f.result() for f in futures]

        for cue, elapsed, resp in results:
            assert "error" not in resp, (
                f"memory_recall for cue {cue!r} returned an error: "
                f"{resp.get('error')}"
            )
            result = resp.get("result")
            assert isinstance(result, dict), (
                f"expected a result dict for cue {cue!r}, got {resp!r}"
            )
            hits = result.get("hits")
            assert isinstance(hits, list) and len(hits) > 0, (
                f"expected hits>0 for cue {cue!r} on a {report_or_stdlib(driver)}"
                f" corpus, got {hits!r}"
            )
            assert any(h.get("literal_surface") for h in hits), (
                f"expected at least one hit with non-empty literal_surface "
                f"for cue {cue!r}, got {hits!r}"
            )
            # THE RED ASSERTION: on HEAD, a freshly-migrated lilli store with
            # a drift-mismatched stale cache and a deferred-capture backlog
            # is expected to serialize concurrent recalls behind the boot
            # fleet's shared lock and blow well past this warm SLA. This
            # assertion is intentionally a plain failure on HEAD (no xfail)
            # -- the warm-load-index + sleep-maintenance work that follows
            # must make it pass.
            assert elapsed < _WARM_SLA_SEC, (
                f"cue {cue!r} took {elapsed:.3f}s (SLA {_WARM_SLA_SEC}s) on "
                f"the {driver} driver -- recall must be served from a warm, "
                f"persistent index with maintenance work kept off the awake "
                f"recall path"
            )
    finally:
        handle.terminate()
        _assert_prod_daemon_alive_if_present(prod_daemon_pid)


def report_or_stdlib(driver: str) -> str:
    return "freshly-migrated lilli" if driver == "lilli" else "stdlib"

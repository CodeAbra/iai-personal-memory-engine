"""Live-daemon proof that the awake priming seam widens the served candidate
set and surfaces a genuinely non-overlapping target, plus a first warm-only
ON-vs-OFF recall p95 pair over the real Unix socket.

Two isolated daemon subprocesses are spawned per test, one per priming arm,
each against its own copy of a planted store -- the priming flag is a
per-call environment read inside a fixed process environment, so it cannot be
flipped mid-run on a single daemon; a real A/B requires two processes.

Every teeth-check here is RED-first: the shared assertion helper each GREEN
test depends on is proven, in its own dedicated test against an unplanted
copy of the same corpus, to actually go negative -- a check that can never
fail is not evidence of anything.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import pytest

from tests._proc_prime_live_fixture import CUE, build_fixture
from tests._warm_recall_repro_support import (
    _real_home,
    await_socket,
    raw_recall,
    spawn_isolated_daemon,
)

try:
    import iai_mcp_native  # noqa: F401
except ImportError:
    pytest.skip(
        "iai_mcp_native is unavailable; this proof requires the native engine "
        "extension",
        allow_module_level=True,
    )

_GATE_CUES = (CUE,)
_GATE_CUES_SHA256 = "8b574316c075b92955c416940cc7acda5dba5e05e03a08d499e3e090d2debdd3"

_WARM_SLA_SEC = 1.0
_CLIENT_TIMEOUT_SEC = 30.0
_SOCKET_AWAIT_TIMEOUT_SEC = 60.0
_N_WARMUP = 3
_N_SAMPLES = 50

_DEGRADE_SOURCES = ("embedder-build-degrade", "cold-structural-degrade", "cortex-fallback")
_FULL_QUALITY_STRUCTURAL = ("normal", "overlay")


def _assert_frozen_cue_set() -> None:
    fingerprint = hashlib.sha256("\n".join(_GATE_CUES).encode("utf-8")).hexdigest()
    assert fingerprint == _GATE_CUES_SHA256, (
        f"the frozen gate cue set has drifted: expected fingerprint "
        f"{_GATE_CUES_SHA256}, got {fingerprint}. If the cue changed "
        f"intentionally, update the frozen fingerprint alongside it."
    )


def _prod_daemon_pid() -> "int | None":
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
            f"the real prod daemon (pid {pid}) is no longer alive after this "
            f"test ran an isolated daemon -- possible cross-contamination"
        )
    except PermissionError:
        pass  # alive, just not signalable by this uid -- fine
    except OSError:
        pass


def _assert_full_quality(result: dict) -> None:
    degrade_source = result.get("_source")
    assert degrade_source not in _DEGRADE_SOURCES, (
        f"recall degraded (_source={degrade_source!r}), not a full-quality answer"
    )
    structural_branch = result.get("_structural_source")
    assert structural_branch in _FULL_QUALITY_STRUCTURAL, (
        f"recall structural branch was {structural_branch!r}, not full-quality "
        f"(normal/overlay)"
    )


def _scored_count(resp: dict) -> float:
    result = resp["result"]
    stage_timings = result.get("_stage_timings")
    assert isinstance(stage_timings, dict), (
        f"expected _stage_timings to be populated (IAI_MCP_STAGE_PROFILE=1 was "
        f"set on this arm), got {result!r}"
    )
    scored = stage_timings.get("scored_count")
    assert scored is not None, f"expected scored_count in _stage_timings, got {stage_timings!r}"
    return float(scored)


def _served_id_set(resp: dict) -> "set[str]":
    hits = resp["result"]["hits"]
    return {h.get("record_id") for h in hits}


def _assert_on_scores_wider(on_scored_count: float, off_scored_count: float) -> None:
    assert on_scored_count > off_scored_count, (
        f"expected the priming-ON arm to score a strictly wider candidate set "
        f"than the OFF arm (on={on_scored_count}, off={off_scored_count}); the "
        f"seam did not widen the pool for this cue"
    )


def _assert_dst_exclusive_to_on(on_hits: "set[str]", off_hits: "set[str]", dst_id: str) -> None:
    assert dst_id in on_hits and dst_id not in off_hits, (
        f"expected the target {dst_id!r} to be served on the ON arm and absent "
        f"on the OFF arm (on_has={dst_id in on_hits}, off_has={dst_id in off_hits})"
    )


def _spawn_pair(
    tmp_path: Path, fixture_root: Path, *, extra_env_off: dict, extra_env_on: dict,
):
    off_root = tmp_path / "off-copy"
    on_root = tmp_path / "on-copy"
    shutil.copytree(fixture_root, off_root)
    shutil.copytree(fixture_root, on_root)

    handles: "list" = []
    try:
        handles.append(spawn_isolated_daemon(
            off_root, "lilli",
            scratch_home=tmp_path / "scratch-home-off",
            extra_env=extra_env_off,
        ))
        handles.append(spawn_isolated_daemon(
            on_root, "lilli",
            scratch_home=tmp_path / "scratch-home-on",
            extra_env=extra_env_on,
        ))
        for h in handles:
            await_socket(h.socket_path, timeout=_SOCKET_AWAIT_TIMEOUT_SEC)
    except Exception:
        for h in handles:
            with contextlib.suppress(Exception):
                h.terminate()
        raise
    return tuple(handles)


def _await_warm(
    handle, cue: str = CUE, *,
    min_calls: int = _N_WARMUP, timeout: float = 60.0, label: "str | None" = None,
) -> None:
    """Discard throwaway recalls, at least ``min_calls`` of them, until the
    daemon's lazy structural/graph caches settle into a full-quality answer
    (bounded by ``timeout``). The first calls after boot can legitimately
    answer from a fast degraded path, or from the last-known-good structural
    branch while a background refresh completes, while those caches are
    still loading -- neither is evidence the seam itself is unsound, and
    letting either leak into a measured/asserted call would be. A per-cue
    call must settle its own per-cue lazy state (embed cache, warm BM25,
    community gate) independently of any other cue already warmed on the
    same daemon."""
    deadline = time.monotonic() + timeout
    calls = 0
    last_result: "dict | None" = None
    while calls < min_calls or time.monotonic() < deadline:
        resp = raw_recall(handle.socket_path, cue, timeout_s=_CLIENT_TIMEOUT_SEC)
        result = resp.get("result") or {}
        last_result = result
        calls += 1
        settled = (
            result.get("_source") not in _DEGRADE_SOURCES
            and result.get("_structural_source") in _FULL_QUALITY_STRUCTURAL
        )
        if calls >= min_calls and settled:
            return
    raise AssertionError(
        f"the daemon never settled into a full-quality structural read within "
        f"{timeout}s ({calls} calls, label={label!r}); last result: {last_result!r}"
    )


# ---------------------------------------------------------------------------
# Anchor (A): the widening the latency gate charges -- band-independent.
# ---------------------------------------------------------------------------


@pytest.mark.perf
@pytest.mark.live
@pytest.mark.timeout(900)
def test_widening_anchor_goes_flat_without_plant(tmp_path: Path) -> None:
    _assert_frozen_cue_set()
    prod_pid = _prod_daemon_pid()

    fixture = build_fixture(tmp_path / "base-a-unplanted", plant=False, require_band=False)

    handle_off, handle_on = _spawn_pair(
        tmp_path, fixture.root,
        extra_env_off={"IAI_MCP_PROC_PRIME": "0", "IAI_MCP_STAGE_PROFILE": "1"},
        extra_env_on={"IAI_MCP_PROC_PRIME": "1", "IAI_MCP_STAGE_PROFILE": "1"},
    )
    try:
        _await_warm(handle_off)
        _await_warm(handle_on)

        resp_off = raw_recall(handle_off.socket_path, CUE, timeout_s=_CLIENT_TIMEOUT_SEC)
        resp_on = raw_recall(handle_on.socket_path, CUE, timeout_s=_CLIENT_TIMEOUT_SEC)
        _assert_full_quality(resp_off["result"])
        _assert_full_quality(resp_on["result"])

        scored_off = _scored_count(resp_off)
        scored_on = _scored_count(resp_on)

        with pytest.raises(AssertionError):
            _assert_on_scores_wider(scored_on, scored_off)
    finally:
        handle_off.terminate()
        handle_on.terminate()
        _assert_prod_daemon_alive_if_present(prod_pid)


@pytest.mark.perf
@pytest.mark.live
@pytest.mark.timeout(900)
def test_on_arm_scores_wider_candidate_set(tmp_path: Path) -> None:
    _assert_frozen_cue_set()
    prod_pid = _prod_daemon_pid()

    fixture = build_fixture(tmp_path / "base-a-planted", plant=True, require_band=False)
    assert fixture.pool_count >= 500

    handle_off, handle_on = _spawn_pair(
        tmp_path, fixture.root,
        extra_env_off={"IAI_MCP_PROC_PRIME": "0", "IAI_MCP_STAGE_PROFILE": "1"},
        extra_env_on={"IAI_MCP_PROC_PRIME": "1", "IAI_MCP_STAGE_PROFILE": "1"},
    )
    try:
        _await_warm(handle_off)
        _await_warm(handle_on)

        resp_off = raw_recall(handle_off.socket_path, CUE, timeout_s=_CLIENT_TIMEOUT_SEC)
        resp_on = raw_recall(handle_on.socket_path, CUE, timeout_s=_CLIENT_TIMEOUT_SEC)
        _assert_full_quality(resp_off["result"])
        _assert_full_quality(resp_on["result"])

        scored_off = _scored_count(resp_off)
        scored_on = _scored_count(resp_on)

        _assert_on_scores_wider(scored_on, scored_off)
    finally:
        handle_off.terminate()
        handle_on.terminate()
        _assert_prod_daemon_alive_if_present(prod_pid)


# ---------------------------------------------------------------------------
# Anchor (B): band-verified set-exclusivity, attributed to the nudge.
# ---------------------------------------------------------------------------


@pytest.mark.perf
@pytest.mark.live
@pytest.mark.timeout(900)
def test_teeth_check_goes_red_without_plant(tmp_path: Path) -> None:
    _assert_frozen_cue_set()
    prod_pid = _prod_daemon_pid()

    fixture = build_fixture(tmp_path / "base-b-unplanted", plant=False, require_band=True)

    handle_off, handle_on = _spawn_pair(
        tmp_path, fixture.root,
        extra_env_off={"IAI_MCP_PROC_PRIME": "0"},
        extra_env_on={"IAI_MCP_PROC_PRIME": "1"},
    )
    try:
        _await_warm(handle_off)
        _await_warm(handle_on)

        resp_off = raw_recall(handle_off.socket_path, CUE, timeout_s=_CLIENT_TIMEOUT_SEC)
        resp_on = raw_recall(handle_on.socket_path, CUE, timeout_s=_CLIENT_TIMEOUT_SEC)
        _assert_full_quality(resp_off["result"])
        _assert_full_quality(resp_on["result"])

        off_hits = _served_id_set(resp_off)
        on_hits = _served_id_set(resp_on)

        # A deterministic RED, not merely single-arm jitter: unplanted, both
        # arms are byte-identical code paths and must serve the same set.
        assert on_hits == off_hits, (
            f"expected an unplanted copy to serve identical sets on both arms "
            f"(on={on_hits!r}, off={off_hits!r})"
        )
        assert on_hits and None not in on_hits, (
            f"expected a non-empty, well-formed served-hit set on the unplanted "
            f"copy before checking exclusivity (on_hits={on_hits!r})"
        )

        with pytest.raises(AssertionError):
            _assert_dst_exclusive_to_on(on_hits, off_hits, fixture.dst_id)
    finally:
        handle_off.terminate()
        handle_on.terminate()
        _assert_prod_daemon_alive_if_present(prod_pid)


@pytest.mark.perf
@pytest.mark.live
@pytest.mark.timeout(900)
def test_planted_chunk_fires_on_live_daemon(tmp_path: Path) -> None:
    _assert_frozen_cue_set()
    prod_pid = _prod_daemon_pid()

    fixture = build_fixture(tmp_path / "base-b-planted", plant=True, require_band=True)
    assert fixture.pool_count >= 500

    handle_off, handle_on = _spawn_pair(
        tmp_path, fixture.root,
        extra_env_off={"IAI_MCP_PROC_PRIME": "0"},
        extra_env_on={"IAI_MCP_PROC_PRIME": "1"},
    )
    try:
        _await_warm(handle_off)
        _await_warm(handle_on)

        resp_off = raw_recall(handle_off.socket_path, CUE, timeout_s=_CLIENT_TIMEOUT_SEC)
        resp_on = raw_recall(handle_on.socket_path, CUE, timeout_s=_CLIENT_TIMEOUT_SEC)
        _assert_full_quality(resp_off["result"])
        _assert_full_quality(resp_on["result"])

        off_hits = _served_id_set(resp_off)
        on_hits = _served_id_set(resp_on)

        _assert_dst_exclusive_to_on(on_hits, off_hits, fixture.dst_id)
    finally:
        handle_off.terminate()
        handle_on.terminate()
        _assert_prod_daemon_alive_if_present(prod_pid)


# ---------------------------------------------------------------------------
# Warm-only two-arm p95 measurement with the absolute awake-latency fence.
# ---------------------------------------------------------------------------


def _p95(samples: "list[float]") -> float:
    s = sorted(samples)
    return s[min(int(0.95 * len(s)), len(s) - 1)]


_MAX_DEGRADE_RETRIES_PER_SAMPLE = 40


def _collect_warm_samples(handle, cue: str = CUE, *, n: int) -> "tuple[list[float], int]":
    """Collect n timed samples, each a genuinely full-quality answer.

    A degraded response is never laundered into the p95: it is excluded and
    the slot is retried (bounded), so an occasional transient fast-degrade
    cannot masquerade as a fast warm answer. Returns the samples plus a count
    of how many degraded responses were discarded along the way."""
    samples: "list[float]" = []
    discarded = 0
    for _ in range(n):
        for attempt in range(_MAX_DEGRADE_RETRIES_PER_SAMPLE):
            t0 = time.monotonic()
            resp = raw_recall(handle.socket_path, cue, timeout_s=_CLIENT_TIMEOUT_SEC)
            dt = time.monotonic() - t0
            result = resp["result"]
            if (
                result.get("_source") not in _DEGRADE_SOURCES
                and result.get("_structural_source") in _FULL_QUALITY_STRUCTURAL
            ):
                samples.append(dt)
                break
            discarded += 1
        else:
            raise AssertionError(
                f"a sample slot never produced a full-quality response within "
                f"{_MAX_DEGRADE_RETRIES_PER_SAMPLE} attempts; last result: {result!r}"
            )
    return samples, discarded


@pytest.mark.perf
@pytest.mark.live
@pytest.mark.timeout(900)
def test_live_p95_absolute_fence(tmp_path: Path) -> None:
    _assert_frozen_cue_set()
    prod_pid = _prod_daemon_pid()

    fixture = build_fixture(tmp_path / "base-p95", plant=True, require_band=False)

    handle_off, handle_on = _spawn_pair(
        tmp_path, fixture.root,
        extra_env_off={"IAI_MCP_PROC_PRIME": "0"},
        extra_env_on={"IAI_MCP_PROC_PRIME": "1"},
    )
    try:
        _await_warm(handle_off)
        _await_warm(handle_on)

        samples_off, discarded_off = _collect_warm_samples(handle_off, n=_N_SAMPLES)
        samples_on, discarded_on = _collect_warm_samples(handle_on, n=_N_SAMPLES)
        print(
            f"discarded degraded samples: off={discarded_off} on={discarded_on} "
            f"(excluded from the p95, never laundered into it)"
        )

        p95_off = _p95(samples_off)
        p95_on = _p95(samples_on)
        delta = p95_on - p95_off

        binding_note = (
            "the absolute fence binds at this pool scale"
            if p95_on >= _WARM_SLA_SEC
            else "the absolute fence does not bind at this pool scale -- a "
                 "relative-regression margin over an A/A floor is the "
                 "constraint that would actually decide a production flip, "
                 "and is not measured here"
        )
        print(
            f"p95_off={p95_off:.4f}s p95_on={p95_on:.4f}s delta={delta:.4f}s "
            f"fence={_WARM_SLA_SEC}s -- {binding_note}"
        )

        assert p95_on < _WARM_SLA_SEC, (
            f"p95(ON)={p95_on:.4f}s did not clear the absolute awake fence of "
            f"{_WARM_SLA_SEC}s (p95(OFF)={p95_off:.4f}s, delta={delta:.4f}s) -- "
            f"a legitimate keep-OFF finding, not a threshold to soften"
        )
    finally:
        handle_off.terminate()
        handle_on.terminate()
        _assert_prod_daemon_alive_if_present(prod_pid)


# ---------------------------------------------------------------------------
# Three-concurrent-daemon gate: the A/A floor and the A/B delta share ONE
# interleaved measurement window, over a mixed verbatim+paraphrase cue set.
# ---------------------------------------------------------------------------

_GATE_CUES_4 = (
    CUE,
    "what happens to old idle sessions during archive maintenance",
    "cleanup behavior for sessions that went stale",
    "when does the maintenance job purge inactive session records",
)
_GATE_CUES_4_SHA256 = "ef4f0685bc7cc79c2b8ffac97cb6c378090f696487cbca518081afe40278c991"

_AA_FLOOR_CAP_SEC = 0.10
_REGRESSION_MARGIN_SEC = 0.10
_TRIO_ARM_NAMES = ("off_a", "off_b", "on")

# When set by the caller, a machine-readable verdict artifact is written
# here -- never a hardcoded phase-specific path in this module, the executor
# decides where the durable record lands.
_VERDICT_JSON_ENV = "IAI_MCP_GATE_VERDICT_JSON_PATH"


def _assert_frozen_cue_set_4() -> None:
    fingerprint = hashlib.sha256("\n".join(_GATE_CUES_4).encode("utf-8")).hexdigest()
    assert fingerprint == _GATE_CUES_4_SHA256, (
        f"the frozen 4-cue gate set has drifted: expected fingerprint "
        f"{_GATE_CUES_4_SHA256}, got {fingerprint}. If the cue set changed "
        f"intentionally, update the frozen fingerprint alongside it."
    )


def _diagnose_daemon_failure(handle, label: str) -> str:
    rc = handle.proc.poll()
    stderr_tail = "<process still running, stderr not drained>"
    if rc is not None and handle.proc.stderr is not None:
        try:
            raw = handle.proc.stderr.read()
            stderr_tail = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
        except Exception:
            stderr_tail = "<stderr unreadable>"
    return f"{label} daemon poll={rc!r} stderr_tail={stderr_tail[-4000:]!r}"


def _recall_checked(handle, cue: str, label: str) -> dict:
    try:
        return raw_recall(handle.socket_path, cue, timeout_s=_CLIENT_TIMEOUT_SEC)
    except Exception as exc:
        raise AssertionError(
            f"raw_recall failed on {label} for cue {cue!r}: {exc}; "
            f"{_diagnose_daemon_failure(handle, label)}"
        ) from exc


def _spawn_trio(
    tmp_path: Path, fixture_root: Path, *,
    extra_env_off_a: dict, extra_env_off_b: dict, extra_env_on: dict,
):
    off_a_root = tmp_path / "off-a-copy"
    off_b_root = tmp_path / "off-b-copy"
    on_root = tmp_path / "on-copy"
    shutil.copytree(fixture_root, off_a_root)
    shutil.copytree(fixture_root, off_b_root)
    shutil.copytree(fixture_root, on_root)

    handles: "list" = []
    try:
        handles.append(spawn_isolated_daemon(
            off_a_root, "lilli",
            scratch_home=tmp_path / "scratch-home-off-a",
            extra_env=extra_env_off_a,
        ))
        handles.append(spawn_isolated_daemon(
            off_b_root, "lilli",
            scratch_home=tmp_path / "scratch-home-off-b",
            extra_env=extra_env_off_b,
        ))
        handles.append(spawn_isolated_daemon(
            on_root, "lilli",
            scratch_home=tmp_path / "scratch-home-on",
            extra_env=extra_env_on,
        ))
        for h in handles:
            await_socket(h.socket_path, timeout=_SOCKET_AWAIT_TIMEOUT_SEC)
    except Exception:
        for h in handles:
            with contextlib.suppress(Exception):
                h.terminate()
        raise
    return tuple(handles)


def _fire_one_sample(handle, cue: str, label: str) -> "tuple[float, dict, int]":
    result: "dict | None" = None
    for attempt in range(_MAX_DEGRADE_RETRIES_PER_SAMPLE):
        t0 = time.monotonic()
        resp = _recall_checked(handle, cue, label)
        dt = time.monotonic() - t0
        result = resp["result"]
        if (
            result.get("_source") not in _DEGRADE_SOURCES
            and result.get("_structural_source") in _FULL_QUALITY_STRUCTURAL
        ):
            return dt, resp, attempt
    raise AssertionError(
        f"{label}/{cue!r}: a sample slot never produced a full-quality response "
        f"within {_MAX_DEGRADE_RETRIES_PER_SAMPLE} attempts; last result: {result!r}"
    )


def _collect_trio_interleaved(
    handle_off_a, handle_off_b, handle_on, cues: "tuple[str, ...]", *, n: int,
):
    """One interleaved round-by-round pass: for each round, for each cue, fire
    OFF-a, OFF-b, ON in that exact order, so all three arms see identical
    machine-load conditions within the same round -- the design that ties the
    A/A floor to the same window the A/B delta is drawn from. A degraded
    sample retries its own slot only (bounded); it never blocks or reorders
    the other arms' rounds."""
    handles = {"off_a": handle_off_a, "off_b": handle_off_b, "on": handle_on}
    dt_samples = {cue: {name: [] for name in _TRIO_ARM_NAMES} for cue in cues}
    resp_samples = {cue: {name: [] for name in _TRIO_ARM_NAMES} for cue in cues}
    discarded = {cue: {name: 0 for name in _TRIO_ARM_NAMES} for cue in cues}
    for _round in range(n):
        for cue in cues:
            for name in _TRIO_ARM_NAMES:
                dt, resp, retries = _fire_one_sample(handles[name], cue, name)
                dt_samples[cue][name].append(dt)
                resp_samples[cue][name].append(resp)
                discarded[cue][name] += retries
    return dt_samples, resp_samples, discarded


def _write_verdict_artifact(data: dict) -> "Path | None":
    target = os.environ.get(_VERDICT_JSON_ENV)
    if not target:
        return None
    out_path = Path(target)
    pass_name = data.get("pass")
    if pass_name:
        out_path = out_path.with_name(f"{out_path.stem}-{pass_name}{out_path.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str))
    return out_path


def _warm_all_cues(handle, label: str, cues: "tuple[str, ...]") -> None:
    for cue in cues:
        _await_warm(handle, cue, min_calls=_N_WARMUP, timeout=60.0, label=label)


@pytest.mark.perf
@pytest.mark.live
@pytest.mark.timeout(900)
def test_live_p95_aa_floor_gate(tmp_path: Path) -> None:
    _assert_frozen_cue_set_4()
    prod_pid = _prod_daemon_pid()

    fixture = build_fixture(tmp_path / "base-trio-ceiling", plant=True, require_band=True)
    assert fixture.pool_count >= 500

    handle_off_a, handle_off_b, handle_on = _spawn_trio(
        tmp_path, fixture.root,
        extra_env_off_a={"IAI_MCP_PROC_PRIME": "0"},
        extra_env_off_b={"IAI_MCP_PROC_PRIME": "0"},
        extra_env_on={"IAI_MCP_PROC_PRIME": "1"},
    )
    try:
        _warm_all_cues(handle_off_a, "off_a", _GATE_CUES_4)
        _warm_all_cues(handle_off_b, "off_b", _GATE_CUES_4)
        _warm_all_cues(handle_on, "on", _GATE_CUES_4)

        # Non-vacuity exclusivity precondition, verbatim cue only (the
        # band-verified target was selected against this cue at fixture-build
        # time): a two-OFF stability control -- the target must be exclusive
        # to ON against BOTH independently-booted OFF arms, not just one.
        resp_off_a_cue0 = _recall_checked(handle_off_a, CUE, "off_a")
        resp_off_b_cue0 = _recall_checked(handle_off_b, CUE, "off_b")
        resp_on_cue0 = _recall_checked(handle_on, CUE, "on")
        for resp in (resp_off_a_cue0, resp_off_b_cue0, resp_on_cue0):
            _assert_full_quality(resp["result"])
        off_a_hits_cue0 = _served_id_set(resp_off_a_cue0)
        off_b_hits_cue0 = _served_id_set(resp_off_b_cue0)
        on_hits_cue0 = _served_id_set(resp_on_cue0)
        assert (
            fixture.dst_id in on_hits_cue0
            and fixture.dst_id not in off_a_hits_cue0
            and fixture.dst_id not in off_b_hits_cue0
        ), (
            f"expected the band-verified target {fixture.dst_id!r} to be served "
            f"on ON and absent on BOTH OFF arms on the verbatim cue "
            f"(on={fixture.dst_id in on_hits_cue0}, "
            f"off_a={fixture.dst_id in off_a_hits_cue0}, "
            f"off_b={fixture.dst_id in off_b_hits_cue0})"
        )

        dt_samples, _resp_samples, discarded = _collect_trio_interleaved(
            handle_off_a, handle_off_b, handle_on, _GATE_CUES_4, n=_N_SAMPLES,
        )

        per_cue: dict = {}
        void_cues: "list[str]" = []
        for cue in _GATE_CUES_4:
            p95_off_a = _p95(dt_samples[cue]["off_a"])
            p95_off_b = _p95(dt_samples[cue]["off_b"])
            p95_on = _p95(dt_samples[cue]["on"])
            aa_floor = abs(p95_off_a - p95_off_b)
            c1_pass = p95_on < _WARM_SLA_SEC
            delta = p95_on - p95_off_a
            c2_pass = delta <= _REGRESSION_MARGIN_SEC
            ratio = (p95_on / p95_off_a) if p95_off_a > 0 else float("inf")
            per_cue[cue] = {
                "p95_off_a": p95_off_a, "p95_off_b": p95_off_b, "p95_on": p95_on,
                "aa_floor": aa_floor, "c1_pass": c1_pass, "c2_pass": c2_pass,
                "delta": delta, "ratio_on_over_off_a": ratio,
                "discarded": discarded[cue],
            }
            print(
                f"[ceiling] cue={cue!r} p95_off_a={p95_off_a:.4f}s "
                f"p95_off_b={p95_off_b:.4f}s p95_on={p95_on:.4f}s "
                f"aa_floor={aa_floor:.4f}s delta={delta:.4f}s ratio={ratio:.3f} "
                f"C1={c1_pass} C2={c2_pass} discarded={discarded[cue]!r}"
            )
            if aa_floor > _AA_FLOOR_CAP_SEC:
                void_cues.append(cue)

        artifact = {
            "pass": "ceiling", "cues": list(_GATE_CUES_4), "per_cue": per_cue,
            "aa_floor_cap_sec": _AA_FLOOR_CAP_SEC,
            "regression_margin_sec": _REGRESSION_MARGIN_SEC,
            "warm_sla_sec": _WARM_SLA_SEC,
            "void_cues": void_cues,
            "dst_exclusivity_cue0": {
                "dst_id": fixture.dst_id, "dst_cosine": fixture.dst_cosine,
                "on": fixture.dst_id in on_hits_cue0,
                "off_a": fixture.dst_id in off_a_hits_cue0,
                "off_b": fixture.dst_id in off_b_hits_cue0,
            },
        }

        if void_cues:
            artifact["run_level_verdict"] = "VOID"
            _write_verdict_artifact(artifact)
            pytest.skip(
                f"window VOID: AA_floor exceeded the {_AA_FLOOR_CAP_SEC}s cap "
                f"for cue(s) {void_cues!r} -- machine-load noise dominated this "
                f"window; re-run for a valid measurement, no verdict recorded "
                f"for this run"
            )

        authorize = all(
            per_cue[cue]["c1_pass"] and per_cue[cue]["c2_pass"] for cue in _GATE_CUES_4
        )
        artifact["run_level_verdict"] = "AUTHORIZE FLIP ON" if authorize else "KEEP OFF"
        _write_verdict_artifact(artifact)
        print(f"[ceiling] RUN-LEVEL VERDICT: {artifact['run_level_verdict']}")

        assert authorize, (
            f"run-level gate did not hold for every cue in the frozen set -- "
            f"KEEP OFF is the honest recorded finding here, not a threshold to "
            f"soften. per_cue={per_cue!r}"
        )
    finally:
        handle_off_a.terminate()
        handle_off_b.terminate()
        handle_on.terminate()
        _assert_prod_daemon_alive_if_present(prod_pid)


# ---------------------------------------------------------------------------
# Per-stage attribution pass (STAGE_PROFILE=1): decomposes the ON-vs-OFF-a
# delta across spread / rank / hit_assembly instead of leaving it attributed
# to "the primed hit" alone.
# ---------------------------------------------------------------------------


@pytest.mark.perf
@pytest.mark.live
@pytest.mark.timeout(900)
def test_live_stage_attribution(tmp_path: Path) -> None:
    _assert_frozen_cue_set_4()
    prod_pid = _prod_daemon_pid()

    fixture = build_fixture(tmp_path / "base-trio-attribution", plant=True, require_band=True)
    assert fixture.pool_count >= 500

    handle_off_a, handle_off_b, handle_on = _spawn_trio(
        tmp_path, fixture.root,
        extra_env_off_a={"IAI_MCP_PROC_PRIME": "0", "IAI_MCP_STAGE_PROFILE": "1"},
        extra_env_off_b={"IAI_MCP_PROC_PRIME": "0", "IAI_MCP_STAGE_PROFILE": "1"},
        extra_env_on={"IAI_MCP_PROC_PRIME": "1", "IAI_MCP_STAGE_PROFILE": "1"},
    )
    try:
        _warm_all_cues(handle_off_a, "off_a", _GATE_CUES_4)
        _warm_all_cues(handle_off_b, "off_b", _GATE_CUES_4)
        _warm_all_cues(handle_on, "on", _GATE_CUES_4)

        _dt_samples, resp_samples, discarded = _collect_trio_interleaved(
            handle_off_a, handle_off_b, handle_on, _GATE_CUES_4, n=_N_SAMPLES,
        )

        # NOTE: pipeline.py writes spread/rank/hit_assembly in MILLISECONDS
        # (`(time.perf_counter() - t0) * 1000.0`) -- converted to seconds here
        # so stage deltas are directly comparable to the ceiling pass's
        # wall-clock p95 deltas (also seconds).
        stages = ("spread", "rank", "hit_assembly")
        per_cue: dict = {}
        for cue in _GATE_CUES_4:
            timings_on = [r["result"]["_stage_timings"] for r in resp_samples[cue]["on"]]
            timings_off_a = [r["result"]["_stage_timings"] for r in resp_samples[cue]["off_a"]]
            scored_on = [float(t["scored_count"]) for t in timings_on]
            scored_off_a = [float(t["scored_count"]) for t in timings_off_a]
            reachable_on = [float(t["reachable_count"]) for t in timings_on]

            stage_p95_on = {s: _p95([float(t[s]) / 1000.0 for t in timings_on]) for s in stages}
            stage_p95_off_a = {
                s: _p95([float(t[s]) / 1000.0 for t in timings_off_a]) for s in stages
            }
            stage_delta = {s: stage_p95_on[s] - stage_p95_off_a[s] for s in stages}
            total_delta = sum(stage_delta.values())

            scored_on_min = min(scored_on)
            scored_off_a_max = max(scored_off_a)
            reachable_on_stats = {
                "min": min(reachable_on),
                "max": max(reachable_on),
                "mean": sum(reachable_on) / len(reachable_on),
            }
            per_cue[cue] = {
                "scored_count_on_min": scored_on_min,
                "scored_count_off_a_max": scored_off_a_max,
                "widening_holds": scored_on_min > scored_off_a_max,
                # Recorded, NOT gated on >=500: this synthetic fixture writes
                # no `edges` rows, so the live daemon's graph-reachable pool
                # structurally plateaus well below the >=500 figure borrowed
                # from a real corpus WITH organic edges (see the pool-scale
                # deviation note this test's caller records).
                "reachable_count_on": reachable_on_stats,
                "stage_p95_on_sec": stage_p95_on, "stage_p95_off_a_sec": stage_p95_off_a,
                "stage_delta_sec": stage_delta, "total_stage_delta_sec": total_delta,
                "discarded": discarded[cue],
            }
            print(
                f"[attribution] cue={cue!r} scored_on_min={scored_on_min:.1f} "
                f"scored_off_a_max={scored_off_a_max:.1f} "
                f"widening_holds={scored_on_min > scored_off_a_max} "
                f"reachable_on={reachable_on_stats!r} "
                f"stage_delta_sec={stage_delta!r} total_stage_delta_sec={total_delta:.6f}s"
            )

        # Robust, band-independent non-vacuity backbone -- checked on the
        # verbatim cue, the one cue the fixture's seed record's literal
        # surface guarantees fires as a recall seed. The pool-scale teeth
        # against a tiny fixture (a corpus of tens of records never
        # approaches the unprimed ~82-candidate cut) are re-expressed against
        # the ON arm's own scored_count, reduced across the full sampled
        # window (min over the ON arm's draws, max over OFF-a's) rather than
        # a single sample -- 200 is a floor well below the observed live
        # value (~236-240) but far above what a small fixture could ever
        # reach. A live-daemon reachable-pool floor is not used here: this
        # synthetic fixture writes no graph edges, so the live daemon's
        # graph-reachable pool structurally plateaus well below any figure
        # derived from a real corpus that does carry organic edges,
        # independent of how large this fixture's flat candidate pool is.
        cue0_stats = per_cue[CUE]
        assert cue0_stats["widening_holds"], (
            f"expected min(scored_count(ON)) > max(scored_count(OFF-a)) across "
            f"the sampled window on the verbatim cue (on_min="
            f"{cue0_stats['scored_count_on_min']}, "
            f"off_a_max={cue0_stats['scored_count_off_a_max']}) -- the seam did "
            f"not pay the widening cost this latency gate charges"
        )
        assert cue0_stats["scored_count_on_min"] >= 200, (
            f"pool-scale non-vacuity precondition failed: min(scored_count(ON))="
            f"{cue0_stats['scored_count_on_min']} across the sampled window on "
            f"the verbatim cue, below the required floor of 200 -- the widened "
            f"candidate set is too small for the hit-assembly cost to be "
            f"measurable"
        )
        assert any(abs(cue0_stats["stage_delta_sec"][s]) > 1e-9 for s in stages), (
            f"attribution is vacuous: every named stage delta on the verbatim "
            f"cue is exactly zero (stage_delta_sec={cue0_stats['stage_delta_sec']!r}) "
            f"-- the seam did not change the served-set composition"
        )

        dominant_cue0 = max(stages, key=lambda s: cue0_stats["stage_delta_sec"][s])
        print(
            f"[attribution] dominant stage on the verbatim cue: {dominant_cue0!r} "
            f"(delta={cue0_stats['stage_delta_sec'][dominant_cue0]:.6f}s of total "
            f"{cue0_stats['total_stage_delta_sec']:.6f}s)"
        )

        widened_cues = [cue for cue in _GATE_CUES_4 if per_cue[cue]["widening_holds"]]
        print(
            f"[attribution] cues that exercised the widening cost path "
            f"(scored_count(ON) > scored_count(OFF-a)): {widened_cues!r} of "
            f"{list(_GATE_CUES_4)!r}"
        )

        artifact = {
            "pass": "attribution", "cues": list(_GATE_CUES_4), "per_cue": per_cue,
            "dominant_stage_cue0": dominant_cue0, "widened_cues": widened_cues,
        }
        _write_verdict_artifact(artifact)
    finally:
        handle_off_a.terminate()
        handle_off_b.terminate()
        handle_on.terminate()
        _assert_prod_daemon_alive_if_present(prod_pid)


# ---------------------------------------------------------------------------
# Both-driver fixture + prime_cache round-trip parity (no daemon spawn --
# the store-involving surface only; the authoritative p95 gate stays lilli-
# only, since the p95 leg is client-side arithmetic with no driver-specific
# content).
# ---------------------------------------------------------------------------


@pytest.mark.perf
@pytest.mark.live
@pytest.mark.timeout(900)
@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_fixture_roundtrips_on_both_drivers(tmp_path: Path, driver: str) -> None:
    from tests._proc_prime_live_fixture import _env_scope

    root = tmp_path / f"base-roundtrip-{driver}"
    fixture = build_fixture(root, plant=True, require_band=False, driver=driver)
    assert fixture.pool_count >= 500

    with _env_scope(IAI_MCP_STORE=str(root), LILLI_STORAGE_DRIVER=driver):
        from iai_mcp import prime_cache
        from iai_mcp.store import MemoryStore

        store = MemoryStore(path=root)
        try:
            blob = prime_cache.load(store)
            seed_to_chunks = blob.get("seed_to_chunks", {})
            chunk_members = blob.get("chunk_members", {})
            assert fixture.src_id in seed_to_chunks, (
                f"seed_to_chunks did not round-trip for driver={driver!r}: "
                f"expected {fixture.src_id!r} present, got {seed_to_chunks!r}"
            )
            assert fixture.chunk_id in chunk_members, (
                f"chunk_members did not round-trip for driver={driver!r}: "
                f"expected {fixture.chunk_id!r} present, got {chunk_members!r}"
            )
            members = chunk_members[fixture.chunk_id]
            assert fixture.src_id in members and fixture.dst_id in members, (
                f"chunk membership incomplete for driver={driver!r}: expected "
                f"both {fixture.src_id!r} and {fixture.dst_id!r} in {members!r}"
            )
        finally:
            store.close()

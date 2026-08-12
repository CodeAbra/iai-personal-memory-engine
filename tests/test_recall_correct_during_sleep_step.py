"""Awake recall correctness while a heavy sleep step actively mutates the engine.

Builds a synthetic background corpus plus a frozen set of pinned / never_decay /
never_merge anchor records, drives the recall-index-rebuild sleep step directly
(bypassing the full pipeline orchestration) in a worker thread, and re-fires the
frozen cue set from the main thread for the duration of that step. The step does
genuine in-parent work on the recall read-model (community/graph rebuild
streaming, whole-map structural decode, full-corpus graph reconstruction, and
exact-authority index rebuild) while holding the interpreter, so a concurrent
recall provably contends with it rather than merely running before/after it.

Hermetic: HOME and the store root are redirected under the test's own tmp_path;
a signature of the real ``~/.iai-mcp/`` is compared before and after, and any
drift fails loud. No remote subprocess is ever invoked (stubbed).
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from tests.conftest_shared import retry_once_on_assertion
from tests.rss_rca import corpus as rss_corpus
from tests.rss_rca.harness import (
    _install_remote_stubs,
    _real_store_signature,
    _restore_remote_stubs,
)

# ---------------------------------------------------------------------------
# Frozen anchor/cue phrase set. Each phrase is BOTH the anchor's literal_surface
# AND the cue text -- the cue embedding and the anchor's own embedding are the
# exact same vector (one embed() call per phrase, reused for both), so the
# pre-sleep baseline is a rank-1 hit by construction rather than an emergent
# property of the embedder. Fingerprinted so a silent edit fails loud.
# ---------------------------------------------------------------------------
_ANCHOR_PHRASES: tuple[str, ...] = (
    "alice's favorite hiking trail winds through the aspen grove near her cabin",
    "alice keeps a hand-written recipe for her grandmother's cardamom bread",
    "alice repairs vintage typewriters as a weekend hobby in her garage",
    "alice's cat startled the delivery courier by hiding inside the mailbox",
    "alice practices calligraphy with a fountain pen every sunday morning",
    "alice built a treehouse for her nephew out of reclaimed cedar planks",
    "alice studies constellations through a telescope on the back porch",
    "alice brews her own kombucha and labels each batch by flavor",
    "alice restrings her acoustic guitar before every open mic night",
    "alice grows heirloom tomatoes in raised beds behind the kitchen",
    "alice volunteers at the animal shelter walking rescue dogs on tuesdays",
    "alice collects vintage postage stamps from countries that no longer exist",
)
_ANCHOR_PHRASES_SHA256 = (
    "ae98e9732b0ae1d4345922e18f4b8c5efd35d7246f46827925798ec88bde3b9e"
)

# Calibrated: at 3000 background records the recall-index-rebuild step's
# in-parent work (whole-map structural decode + full-corpus graph stream +
# exact-authority rebuild) reliably spans several hundred ms to a few seconds
# on this box, wide enough for >=3 full cue-set passes to land inside it. If a
# future box makes the window too short, raise this per the phase's
# calibration protocol -- never shrink the round/overlap floors instead.
_BACKGROUND_N = 3000

# The self-match cosine is ~1.0 by construction (same embedding vector); the
# runner-up is either another anchor (a different real sentence) or a random
# background vector (expected cosine near 0 in 384-d). 0.15 stays far below
# any observed self-vs-runner-up gap while catching a genuine rank inversion.
_RANK1_MARGIN_MIN = 0.15

# Anti-vacuous floors (see module docstring for what each guards).
_MIN_ROUNDS = 3
_MIN_ROUNDS_IN_WINDOW = 3
_ENTRY_WAIT_TIMEOUT_SEC = 120.0
_WORKER_JOIN_TIMEOUT_SEC = 300.0
_MAX_LOOP_WALL_SEC = 300.0


def _assert_frozen_anchor_set() -> None:
    fingerprint = hashlib.sha256(
        "\n".join(_ANCHOR_PHRASES).encode("utf-8")
    ).hexdigest()
    assert fingerprint == _ANCHOR_PHRASES_SHA256, (
        f"the frozen anchor/cue phrase set has drifted: expected fingerprint "
        f"{_ANCHOR_PHRASES_SHA256}, got {fingerprint}. If the phrase list "
        f"changed intentionally, update the frozen fingerprint alongside it."
    )


def _redirect_env(tmp_path: Path) -> dict[str, str | None]:
    prev = {
        k: os.environ.get(k)
        for k in (
            "HOME", "IAI_MCP_STORE", "IAI_MCP_USER_MODEL_PATH",
            "IAI_MCP_CRYPTO_PASSPHRASE", "PYTHON_KEYRING_BACKEND",
        )
    }
    os.environ["HOME"] = str(tmp_path)
    os.environ["IAI_MCP_STORE"] = str(tmp_path / ".iai-mcp" / "hippo")
    os.environ["IAI_MCP_USER_MODEL_PATH"] = str(
        tmp_path / ".iai-mcp" / "user_model.json"
    )
    if not prev["IAI_MCP_CRYPTO_PASSPHRASE"]:
        os.environ["IAI_MCP_CRYPTO_PASSPHRASE"] = (
            "iai-mcp-recall-during-sleep-step-deterministic-2026"
        )
    os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"
    (tmp_path / ".iai-mcp").mkdir(parents=True, exist_ok=True)
    return prev


def _restore_env(prev: dict[str, str | None]) -> None:
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _make_anchor(phrase: str, vec: list[float]) -> "MemoryRecord":  # noqa: F821
    from iai_mcp.types import MemoryRecord

    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=phrase,
        aaak_index="",
        embedding=list(vec),
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=True,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=True,
        never_merge=True,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=["bench", "anchor"],
        language="en",
        hv_tier="bsc",
        structure_hv_payload=b"",
    )


class _Scene:
    """Live handles for one built-and-driven scenario. Built by ``_run_scenario``."""

    def __init__(self) -> None:
        self.anchor_ids: dict[str, UUID] = {}
        self.baseline: dict[str, tuple[UUID, ...]] = {}
        self.rounds: list[dict] = []
        self.window: dict[str, float | None] = {"start": None, "end": None}
        self.worker_done: bool | None = None
        self.worker_error: str | None = None
        self.entered_ok: bool = False


def _anchor_hit_set(
    store, cue_vectors: dict[str, list[float]], anchor_ids: set[UUID], phrase: str,
) -> tuple[tuple[UUID, ...], list]:
    from iai_mcp import retrieve

    resp = retrieve.recall(
        store, cue_vectors[phrase], phrase, "recall-during-sleep-step-session",
        k_hits=5, k_anti=3, mode="verbatim",
    )
    # Rank-ORDER preserved (not a set): the invariant under test is that no
    # anchor is dropped, torn, OR reordered relative to the others, and a set
    # cannot witness a reorder.
    restricted = tuple(h.record_id for h in resp.hits if h.record_id in anchor_ids)
    return restricted, resp.hits


def _run_scenario(
    tmp_path: Path, *, step_name: str, n_background: int = _BACKGROUND_N,
) -> _Scene:
    """Build the corpus+anchors, drive one heavy sleep step concurrently with a
    tight recall loop over the frozen cue set, and return the collected scene.

    ``step_name`` selects the step: ``"recall_index_rebuild"`` (primary,
    provable-entry patched for the overlap teeth) or ``"optimize_hippo"``
    (documented weaker-overlap secondary -- its corpus-scaled HNSW rebuild runs
    in a child process and its in-parent VACUUM is deferred whenever a
    foreground read is in-flight, so overlap with THIS test's own recall loop
    is not guaranteed; only step completion + whatever rounds do land are
    asserted for it).
    """
    _assert_frozen_anchor_set()

    sig_before = _real_store_signature()
    prev_env = _redirect_env(tmp_path)
    store = None
    stub_originals: dict[str, object] = {}
    scene = _Scene()

    import iai_mcp.runtime_graph_cache as runtime_graph_cache_mod
    import iai_mcp.lilli.cycle.sleep_pipeline._recall_index as recall_index_mod
    orig_rebuild = runtime_graph_cache_mod._rebuild_and_save_rgc
    orig_prewarm = recall_index_mod._prewarm_recall_read_model

    try:
        stub_originals = _install_remote_stubs()

        from iai_mcp.embed import embedder_for_store
        from iai_mcp.lifecycle_event_log import LifecycleEventLog
        from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline, SleepStep
        from iai_mcp.store import MemoryStore
        from iai_mcp.store._buffers import flush_record_buffer

        store_path = Path(os.environ["IAI_MCP_STORE"])
        store_path.mkdir(parents=True, exist_ok=True)
        store = MemoryStore(path=store_path)

        embedder = embedder_for_store(store)

        rss_corpus.build_corpus(
            store, n=n_background, seed=42, dim=store.embed_dim,
            varied_tags=True, with_hv_payload=True,
        )
        flush_record_buffer(store)

        cue_vectors: dict[str, list[float]] = {}
        for phrase in _ANCHOR_PHRASES:
            vec = list(embedder.embed(phrase))
            cue_vectors[phrase] = vec
            rec = _make_anchor(phrase, vec)
            store.insert(rec)
            scene.anchor_ids[phrase] = rec.id
        flush_record_buffer(store)

        anchor_id_set = set(scene.anchor_ids.values())

        # Pre-sleep frozen baseline: every cue must retrieve its own anchor as
        # the RANK-1 hit with a cosine margin over the runner-up (not merely
        # present anywhere in the hit-set).
        for phrase in _ANCHOR_PHRASES:
            restricted, hits = _anchor_hit_set(store, cue_vectors, anchor_id_set, phrase)
            assert hits, f"pre-sleep recall for {phrase!r} returned no hits"
            top = hits[0]
            assert top.record_id == scene.anchor_ids[phrase], (
                f"pre-sleep baseline for {phrase!r}: expected its own anchor "
                f"{scene.anchor_ids[phrase]} to be rank-1, got {top.record_id}"
            )
            runner_up_score = hits[1].score if len(hits) > 1 else -1.0
            margin = top.score - runner_up_score
            # Unconditional (not just on failure): every run self-documents
            # its own observed margin against the floor, so a future
            # re-check of _RANK1_MARGIN_MIN never needs a fresh diagnostic
            # pass to know how comfortable the floor is.
            print(
                f"rank-1 margin for {phrase[:40]!r}: top={top.score:.4f} "
                f"runner_up={runner_up_score:.4f} margin={margin:.4f} "
                f"(floor={_RANK1_MARGIN_MIN})"
            )
            assert margin > _RANK1_MARGIN_MIN, (
                f"pre-sleep baseline for {phrase!r}: rank-1 cosine margin "
                f"{margin:.4f} <= floor {_RANK1_MARGIN_MIN} (top={top.score:.4f}, "
                f"runner_up={runner_up_score:.4f}) -- ANN/rank jitter risk"
            )
            assert scene.anchor_ids[phrase] in restricted, (
                f"pre-sleep baseline for {phrase!r}: anchor-restricted hit-set "
                f"must contain its own anchor, got {restricted}"
            )
            # The baseline is whatever the anchor-restricted hit-set naturally
            # is (own anchor plus, at this corpus size, other coherent-English
            # anchors that legitimately out-rank the random-noise background --
            # NOT forced to a singleton). The invariant under test is that this
            # exact set survives the heavy step unchanged, not that it is
            # trivially small.
            scene.baseline[phrase] = restricted

        event_log = LifecycleEventLog(log_dir=(tmp_path / ".iai-mcp" / "logs"))
        pipeline = SleepPipeline(
            store=store,
            lifecycle_state_path=tmp_path / ".iai-mcp" / "lifecycle_state.json",
            event_log=event_log,
        )

        entered = threading.Event()

        if step_name == "recall_index_rebuild":
            step = SleepStep.RECALL_INDEX_REBUILD

            def _wrapped_rebuild(store_arg, **kw):
                scene.window["start"] = time.monotonic()
                entered.set()
                return orig_rebuild(store_arg, **kw)

            def _wrapped_prewarm(self_arg, result_arg, interrupt_check_arg):
                # Ends the window at the read-model prewarm's own return --
                # the last phase of the step's in-parent mutating work
                # (structural decode, graph build, exact-authority rebuild) --
                # rather than at step exit, so tooth (c) measures overlap with
                # the mutating phase, not the whole worker-thread lifetime.
                try:
                    return orig_prewarm(self_arg, result_arg, interrupt_check_arg)
                finally:
                    scene.window["end"] = time.monotonic()

            runtime_graph_cache_mod._rebuild_and_save_rgc = _wrapped_rebuild
            recall_index_mod._prewarm_recall_read_model = _wrapped_prewarm
        elif step_name == "optimize_hippo":
            step = SleepStep.OPTIMIZE_HIPPO
        else:
            raise ValueError(f"unknown step_name {step_name!r}")

        worker_result: dict = {}

        def _worker() -> None:
            t0 = time.monotonic()
            if step_name == "optimize_hippo":
                scene.window["start"] = t0
                entered.set()
            try:
                # interrupt_check=lambda: False is MANDATORY: a foreground-
                # signaling check makes _prewarm_recall_read_model skip its
                # heavy phases and short-circuits the very contention this
                # test needs to observe.
                done, payload = pipeline._step_methods[step](lambda: False)
                worker_result["done"] = done
                worker_result["payload"] = payload
            except Exception as exc:  # noqa: BLE001 -- captured, asserted below
                worker_result["error"] = repr(exc)
            finally:
                # For recall_index_rebuild, window["end"] is already set by
                # _wrapped_prewarm at the mutating phase's own return; only
                # fall back to step-exit time here if that never fired (e.g.
                # the step raised or was interrupted before reaching prewarm).
                if scene.window["end"] is None:
                    scene.window["end"] = time.monotonic()

        thread = threading.Thread(
            target=_worker, name="heavy-sleep-step-worker", daemon=True,
        )
        thread.start()

        scene.entered_ok = entered.wait(timeout=_ENTRY_WAIT_TIMEOUT_SEC)

        loop_t0 = time.monotonic()
        round_idx = 0
        while True:
            alive_before = thread.is_alive()
            r_t0 = time.monotonic()
            round_hits: dict[str, tuple[UUID, ...]] = {}
            for phrase in _ANCHOR_PHRASES:
                restricted, _hits = _anchor_hit_set(
                    store, cue_vectors, anchor_id_set, phrase,
                )
                round_hits[phrase] = restricted
            r_t1 = time.monotonic()
            scene.rounds.append({
                "idx": round_idx, "start": r_t0, "end": r_t1, "hits": round_hits,
            })
            round_idx += 1
            if not alive_before:
                break
            if time.monotonic() - loop_t0 > _MAX_LOOP_WALL_SEC:
                break

        thread.join(timeout=_WORKER_JOIN_TIMEOUT_SEC)
        scene.worker_done = worker_result.get("done")
        scene.worker_error = worker_result.get("error")

        return scene

    finally:
        if step_name == "recall_index_rebuild":
            runtime_graph_cache_mod._rebuild_and_save_rgc = orig_rebuild
            recall_index_mod._prewarm_recall_read_model = orig_prewarm
        if store is not None:
            try:
                store.close()
            except Exception:  # noqa: BLE001
                pass
        _restore_remote_stubs(stub_originals)
        _restore_env(prev_env)
        sig_after = _real_store_signature()
        # Never bury an in-flight AssertionError (e.g. a baseline check failing
        # inside the try above) behind an unrelated hermeticity RuntimeError --
        # only raise here when no other exception is already propagating.
        if sig_before != sig_after and sys.exc_info()[0] is None:
            raise RuntimeError(
                "HERMETICITY VIOLATION: ~/.iai-mcp changed during the test."
            )


# ---------------------------------------------------------------------------
# PRIMARY (must-pass, default gate): RECALL_INDEX_REBUILD.
# ---------------------------------------------------------------------------


@pytest.mark.timeout(900)
def test_recall_stays_correct_during_recall_index_rebuild(tmp_path: Path) -> None:
    """Anchor-restricted recall hit-sets survive a live, in-parent, GIL-holding
    recall-index rebuild unchanged -- no pinned/never_decay/never_merge anchor
    is ever dropped, torn, or mis-ordered while the step mutates the engine.

    Three anti-vacuous teeth, all required:
      (a) the step actually completed (``done=True``, no worker exception);
      (b) at least 3 full passes over the frozen cue set landed while the
          worker thread was alive;
      (c) at least 3 of those passes overlap the step's own in-parent
          mutating-phase window (from provable entry into the community/graph
          rebuild through the read-model prewarm's own return, which covers
          the structural decode, graph reconstruction, and exact-authority
          rebuild) -- wall-clock duration alone is never treated as evidence
          of contention.
    """
    scene = _run_scenario(tmp_path, step_name="recall_index_rebuild")

    assert scene.worker_error is None, (
        f"the recall-index-rebuild step raised: {scene.worker_error}"
    )
    assert scene.entered_ok, (
        "the worker thread never signalled provable entry into the "
        "recall-index rebuild within the wait timeout"
    )
    assert scene.worker_done is True, (
        f"the recall-index-rebuild step did not report completion "
        f"(done={scene.worker_done!r})"
    )

    # Tooth (b).
    assert len(scene.rounds) >= _MIN_ROUNDS, (
        f"only {len(scene.rounds)} recall round(s) completed while the "
        f"worker thread was alive; need >= {_MIN_ROUNDS}. Raise the "
        f"background corpus size (calibration protocol) if this is stable."
    )

    # Tooth (c): overlap with the step's own provable-entry-to-prewarm-return
    # mutating-phase window, not with the whole worker-thread lifetime and
    # never with elapsed time.
    window_start = scene.window["start"]
    window_end = scene.window["end"]
    assert window_start is not None and window_end is not None, (
        "the step's rebuild window was never recorded"
    )
    rounds_in_window = [
        r for r in scene.rounds
        if r["start"] <= window_end and r["end"] >= window_start
    ]
    assert len(rounds_in_window) >= _MIN_ROUNDS_IN_WINDOW, (
        f"only {len(rounds_in_window)} of {len(scene.rounds)} recall round(s) "
        f"overlapped the rebuild window [{window_start:.3f}, {window_end:.3f}]; "
        f"need >= {_MIN_ROUNDS_IN_WINDOW}. Raise the background corpus size "
        f"(calibration protocol) if the window is stably too short."
    )

    # Correctness (sound by construction): every round's anchor-restricted
    # hit-set matches the frozen pre-sleep baseline for every cue. Checked
    # over ALL rounds (not just in-window ones) -- the invariant must hold
    # for the step's entire wall-clock duration, before and after entry too.
    mismatches = []
    for r in scene.rounds:
        for phrase in _ANCHOR_PHRASES:
            got = r["hits"][phrase]
            want = scene.baseline[phrase]
            if got != want:
                mismatches.append((r["idx"], phrase, want, got))
    assert not mismatches, (
        f"anchor-restricted hit-set drifted from the frozen baseline during "
        f"the recall-index rebuild: {mismatches[:5]}"
        + (f" (+{len(mismatches) - 5} more)" if len(mismatches) > 5 else "")
    )


# ---------------------------------------------------------------------------
# SECONDARY (documented weaker overlap, default gate): OPTIMIZE_HIPPO.
# ---------------------------------------------------------------------------


@pytest.mark.timeout(900)
def test_recall_stays_correct_during_hippo_optimize_step(tmp_path: Path) -> None:
    """Same correctness invariant against OPTIMIZE_HIPPO, WEAKER overlap.

    OPTIMIZE_HIPPO's corpus-scaled HNSW rebuild runs in a child process (the
    parent only holds the connection lock for a COUNT and a microsecond atomic
    swap) and its in-parent VACUUM is deferred whenever a foreground read is
    in-flight -- i.e. deferred by this test's own recall loop. Overlap with a
    live recall is therefore NOT guaranteed the way it is for
    RECALL_INDEX_REBUILD, so only step completion and correctness over
    whatever rounds actually land are asserted; the round-floor and
    window-overlap teeth are intentionally not required here.
    """
    scene = _run_scenario(tmp_path, step_name="optimize_hippo")

    assert scene.worker_error is None, (
        f"the OPTIMIZE_HIPPO step raised: {scene.worker_error}"
    )
    assert scene.worker_done is True, (
        f"the OPTIMIZE_HIPPO step did not report completion "
        f"(done={scene.worker_done!r})"
    )

    mismatches = []
    for r in scene.rounds:
        for phrase in _ANCHOR_PHRASES:
            got = r["hits"][phrase]
            want = scene.baseline[phrase]
            if got != want:
                mismatches.append((r["idx"], phrase, want, got))
    assert not mismatches, (
        f"anchor-restricted hit-set drifted from the frozen baseline during "
        f"OPTIMIZE_HIPPO: {mismatches[:5]}"
        + (f" (+{len(mismatches) - 5} more)" if len(mismatches) > 5 else "")
    )


# ---------------------------------------------------------------------------
# Latency (perf lane, opt-in via --perf): wall-clock under concurrent sleep
# load is flake-prone on a loaded machine, so it never lives in the default
# gate -- the default gate stays correctness-only.
# ---------------------------------------------------------------------------


@pytest.mark.perf
@pytest.mark.timeout(900)
@retry_once_on_assertion
def test_recall_latency_during_recall_index_rebuild(tmp_path: Path) -> None:
    """Every mid-rebuild recall round completes under a generous WARM ceiling.

    Retried once (``retry_once_on_assertion``) to absorb one-off OS-scheduler
    jitter on a loaded box; a genuine regression misses both attempts.
    """
    _LATENCY_CEILING_SEC = 5.0

    scene = _run_scenario(tmp_path, step_name="recall_index_rebuild")
    assert scene.worker_error is None
    assert scene.worker_done is True
    assert scene.rounds, "no recall rounds executed"

    slow = [
        (r["idx"], r["end"] - r["start"])
        for r in scene.rounds
        if (r["end"] - r["start"]) > _LATENCY_CEILING_SEC
    ]
    assert not slow, (
        f"{len(slow)} recall round(s) exceeded the {_LATENCY_CEILING_SEC}s "
        f"warm ceiling for a full {len(_ANCHOR_PHRASES)}-cue pass: {slow[:5]}"
    )

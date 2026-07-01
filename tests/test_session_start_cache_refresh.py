"""Tests for the WAKE-time SessionStart cache refresh path.

Covers the bug where ``_write_session_start_cache`` was only called from the
SLEEP branch of the lifecycle tick, so a long-lived Claude session (heartbeat
never quiet for 30 minutes) left the precache file frozen for days while the
store kept ingesting records. The fix adds:

  * a watermark sidecar (records_count + max_vec_label + max_*_at) so the cache
    knows what corpus it was built from
  * a cheap probe `_should_refresh_session_start_cache` that compares the live
    watermark to the stored one
  * `_maybe_refresh_session_start_cache`, a best-effort WAKE refresh that gates
    on min-interval + single-flight + runtime-graph-cache-warm
  * lifecycle-tick hooks that call the refresh after `wake_sequence` (when new
    records were just embedded) and as a periodic safety net in WAKE/DROWSY
  * a TTL safety net in the SessionStart shell hook
    (`IAI_MCP_SESSION_CACHE_MAX_AGE_SEC`)

The tests run against a hermetic store, never touch ``~/.iai-mcp``, and never
spawn a real daemon.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX paths + shell hook")

HOOK_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "iai_mcp" / "_deploy" / "hooks" / "iai-mcp-session-recall.sh"
)


def _fresh_store(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "384")
    from iai_mcp.store import MemoryStore
    return MemoryStore()


def _seed(store, n: int = 3) -> None:
    from iai_mcp.core import _seed_l0_identity
    from iai_mcp.types import EMBED_DIM, MemoryRecord
    _seed_l0_identity(store)
    now = datetime.now(timezone.utc)
    for i in range(n):
        store.insert(MemoryRecord(
            id=uuid4(),
            tier="semantic",
            literal_surface=f"Pinned fact {i}: high-detail context.",
            aaak_index="",
            embedding=[0.1] * EMBED_DIM,
            community_id=None,
            centrality=0.5,
            detail_level=5,
            pinned=True,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=True,
            never_merge=False,
            provenance=[],
            created_at=now,
            updated_at=now,
            tags=[],
            language="en",
        ))


def _query_events(store, kind: str) -> list[dict]:
    from iai_mcp.events import query_events
    return list(query_events(store, kind=kind, limit=10000))


def _cache_paths(tmp_path) -> tuple[Path, Path]:
    cache = tmp_path / "cache.md"
    # The canonical sidecar uses Path.with_suffix(".meta.json") which REPLACES
    # the existing suffix — `.cached.md` → `.cached.meta.json` — matching
    # SESSION_START_CACHE_META_PATH. The reader and the writer route through
    # _default_session_start_cache_meta_path so the two cannot drift.
    meta = tmp_path / "cache.meta.json"
    return cache, meta


def test_default_meta_path_matches_module_constant():
    """The default sidecar derived from cache_path MUST match
    SESSION_START_CACHE_META_PATH when cache_path == SESSION_START_CACHE_PATH.
    Guards against a future refactor renaming one side and forgetting the other.
    """
    from iai_mcp import daemon as daemon_mod

    derived = daemon_mod._default_session_start_cache_meta_path(
        daemon_mod.SESSION_START_CACHE_PATH
    )
    assert derived == daemon_mod.SESSION_START_CACHE_META_PATH, (
        f"sidecar default {derived} != module constant "
        f"{daemon_mod.SESSION_START_CACHE_META_PATH}"
    )


def test_write_and_should_refresh_use_same_default_meta_path(tmp_path, monkeypatch):
    """Anti-divergence: if _write_session_start_cache writes the sidecar at one
    path and _should_refresh_session_start_cache reads it at another, the cache
    will refresh on EVERY tick because the watermark is always missing. Pin both
    sides to the helper and verify the round-trip succeeds."""
    from iai_mcp import daemon as daemon_mod

    store = _fresh_store(tmp_path, monkeypatch)
    _seed(store)

    cache = tmp_path / "cache.md"
    # Do NOT pass meta_path explicitly — both helpers must default to the same
    # path on their own.
    result = daemon_mod._write_session_start_cache(
        store, cache_path=cache, trigger="manual", force_rebuild=True,
    )
    assert result["action"] == "wrote"

    # The reader, given only cache_path, must find the sidecar the writer left.
    should, reason, _wm = daemon_mod._should_refresh_session_start_cache(
        store, cache_path=cache,
        meta_path=daemon_mod._default_session_start_cache_meta_path(cache),
        min_interval_sec=0.0,
    )
    assert not should, (
        f"reader didn't see the writer's sidecar — reason={reason}; "
        f"this means the read/write paths use different default sidecar paths"
    )
    assert reason == "no_new_records"


# ---------------------------------------------------------------------------
# Diagnostic: SLEEP-only call path => stale cache (the bug we are fixing)
# ---------------------------------------------------------------------------

def test_diagnostic_old_path_only_runs_in_sleep_branch(tmp_path, monkeypatch):
    """Document the pre-fix bug: _write_session_start_cache was wired *only*
    into the SLEEP branch of the lifecycle tick, so a never-sleeping daemon
    never regenerated the cache."""
    from iai_mcp import daemon as daemon_mod
    import inspect

    src = inspect.getsource(daemon_mod._tick_body) if hasattr(daemon_mod, "_tick_body") else ""
    # The lifecycle tick body lives in this module; just check the file-wide
    # picture: the only *unconditional* call to `_write_session_start_cache`
    # used to sit inside the `current is _LifecycleState.SLEEP` arm. We now
    # also call it via the WAKE refresh wrapper, but the sleep-pipeline call
    # MUST stay (it backs the post-consolidation cache write).
    text = Path(daemon_mod.__file__).read_text()
    assert "_LifecycleState.SLEEP" in text
    assert "_maybe_refresh_session_start_cache" in text, (
        "WAKE-side refresh helper is missing — fix not applied"
    )


# ---------------------------------------------------------------------------
# Watermark probe
# ---------------------------------------------------------------------------

def test_watermark_probe_counts_active_records(tmp_path, monkeypatch):
    from iai_mcp import daemon as daemon_mod

    store = _fresh_store(tmp_path, monkeypatch)
    _seed(store, n=4)

    wm = daemon_mod._session_start_cache_watermark(store)
    assert wm["records_count"] >= 4
    assert wm["max_vec_label"] >= 4
    assert wm["max_created_at"]  # non-empty ISO


def test_watermark_probe_distinguishes_new_record_with_old_created_at(
    tmp_path, monkeypatch,
):
    """A record inserted *now* but stamped with an *old* ``created_at`` (e.g.
    backfilled from a transcript) still bumps ``max_vec_label`` — that's why
    the watermark uses the monotone vec_label, not just MAX(created_at)."""
    from iai_mcp import daemon as daemon_mod
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    store = _fresh_store(tmp_path, monkeypatch)
    _seed(store, n=2)

    before = daemon_mod._session_start_cache_watermark(store)

    old_ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
    store.insert(MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface="Backfilled from old transcript.",
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.5,
        detail_level=5,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=old_ts,
        updated_at=old_ts,
        tags=[],
        language="en",
    ))

    after = daemon_mod._session_start_cache_watermark(store)
    assert after["max_vec_label"] > before["max_vec_label"]
    assert after["records_count"] > before["records_count"]
    # MAX(created_at) likely UNCHANGED (the new record's date is older), proving
    # the vec_label component is what catches this case.
    assert after["max_created_at"] >= before["max_created_at"]


# ---------------------------------------------------------------------------
# should_refresh decision matrix
# ---------------------------------------------------------------------------

def test_should_refresh_when_cache_absent(tmp_path, monkeypatch):
    from iai_mcp import daemon as daemon_mod

    store = _fresh_store(tmp_path, monkeypatch)
    _seed(store)

    cache, meta = _cache_paths(tmp_path)
    should, reason, _wm = daemon_mod._should_refresh_session_start_cache(
        store, cache_path=cache, meta_path=meta, min_interval_sec=60.0,
    )
    assert should
    assert reason == "cache_absent"


def test_should_refresh_blocks_within_min_interval(tmp_path, monkeypatch):
    from iai_mcp import daemon as daemon_mod

    store = _fresh_store(tmp_path, monkeypatch)
    _seed(store)

    cache, meta = _cache_paths(tmp_path)
    # Pretend a refresh just happened: write cache + meta + bump mtime to now.
    cache.write_text("seed")
    meta.write_text(json.dumps({
        "records_count": 999, "max_vec_label": 999,
        "max_created_at": "z", "max_updated_at": "z",
    }))
    os.utime(cache, (time.time(), time.time()))

    should, reason, _wm = daemon_mod._should_refresh_session_start_cache(
        store, cache_path=cache, meta_path=meta, min_interval_sec=300.0,
    )
    assert not should
    assert reason == "min_interval_not_elapsed"


def test_should_refresh_when_meta_absent(tmp_path, monkeypatch):
    from iai_mcp import daemon as daemon_mod

    store = _fresh_store(tmp_path, monkeypatch)
    _seed(store)

    cache, meta = _cache_paths(tmp_path)
    cache.write_text("seed")
    # Make the cache look old enough to bypass the min-interval gate.
    old = time.time() - 600
    os.utime(cache, (old, old))

    should, reason, _wm = daemon_mod._should_refresh_session_start_cache(
        store, cache_path=cache, meta_path=meta, min_interval_sec=60.0,
    )
    assert should
    assert reason == "meta_absent"


def test_should_refresh_no_new_records(tmp_path, monkeypatch):
    from iai_mcp import daemon as daemon_mod

    store = _fresh_store(tmp_path, monkeypatch)
    _seed(store)

    cache, meta = _cache_paths(tmp_path)
    cache.write_text("seed")
    wm = daemon_mod._session_start_cache_watermark(store)
    meta.write_text(json.dumps({
        "records_count": wm["records_count"],
        "max_vec_label": wm["max_vec_label"],
        "max_created_at": wm["max_created_at"],
        "max_updated_at": wm["max_updated_at"],
    }))
    old = time.time() - 600
    os.utime(cache, (old, old))

    should, reason, _wm = daemon_mod._should_refresh_session_start_cache(
        store, cache_path=cache, meta_path=meta, min_interval_sec=60.0,
    )
    assert not should
    assert reason == "no_new_records"


def test_should_refresh_when_watermark_changes(tmp_path, monkeypatch):
    from iai_mcp import daemon as daemon_mod

    store = _fresh_store(tmp_path, monkeypatch)
    _seed(store, n=2)

    cache, meta = _cache_paths(tmp_path)
    cache.write_text("seed")
    wm = daemon_mod._session_start_cache_watermark(store)
    meta.write_text(json.dumps({
        "records_count": wm["records_count"],
        "max_vec_label": wm["max_vec_label"],
        "max_created_at": wm["max_created_at"],
        "max_updated_at": wm["max_updated_at"],
    }))
    old = time.time() - 600
    os.utime(cache, (old, old))

    _seed(store, n=1)  # add one more record

    should, reason, _wm = daemon_mod._should_refresh_session_start_cache(
        store, cache_path=cache, meta_path=meta, min_interval_sec=60.0,
    )
    assert should
    assert reason == "watermark_changed"


def test_should_refresh_skips_when_watermark_probe_fails(tmp_path, monkeypatch):
    from iai_mcp import daemon as daemon_mod

    store = _fresh_store(tmp_path, monkeypatch)
    _seed(store)

    cache, meta = _cache_paths(tmp_path)
    cache.write_text("seed")
    meta.write_text(json.dumps({
        "records_count": 3,
        "max_vec_label": 3,
        "max_created_at": "z",
        "max_updated_at": "z",
    }))
    old = time.time() - 600
    os.utime(cache, (old, old))

    monkeypatch.setattr(
        daemon_mod,
        "_session_start_cache_watermark",
        lambda _store: {
            "records_count": -1,
            "max_vec_label": -1,
            "max_created_at": "",
            "max_updated_at": "",
            "probe_failed": True,
        },
    )

    should, reason, _wm = daemon_mod._should_refresh_session_start_cache(
        store, cache_path=cache, meta_path=meta, min_interval_sec=60.0,
    )
    assert not should
    assert reason == "probe_failed"


# ---------------------------------------------------------------------------
# _write_session_start_cache: telemetry + skip paths
# ---------------------------------------------------------------------------

def test_write_emits_started_and_success_events(tmp_path, monkeypatch):
    from iai_mcp import daemon as daemon_mod

    store = _fresh_store(tmp_path, monkeypatch)
    _seed(store)

    cache, meta = _cache_paths(tmp_path)
    result = daemon_mod._write_session_start_cache(
        store, cache_path=cache, meta_path=meta,
        trigger="manual", force_rebuild=True,
    )
    assert result["action"] == "wrote"
    assert cache.exists()
    assert meta.exists()

    started = _query_events(store, "session_start_cache_write_started")
    success = _query_events(store, "session_start_cache_write_success")
    failed = _query_events(store, "session_start_cache_write_failed")
    assert len(started) == 1, started
    assert len(success) == 1, success
    assert len(failed) == 0

    # Inspect the success-event payload: all the watermark fields are there.
    from iai_mcp.events import query_events
    s = list(query_events(store, kind="session_start_cache_write_success", limit=10))[0]
    data = s.get("data") or {}
    for key in (
        "trigger", "cache_path", "rendered_chars", "records_count",
        "max_vec_label", "max_record_created_at", "max_updated_at", "duration_ms",
    ):
        assert key in data, f"missing {key} in success event: {data}"
    assert data["trigger"] == "manual"
    assert data["rendered_chars"] > 0
    assert data["records_count"] >= 3


def test_write_skips_when_runtime_graph_cache_cold(tmp_path, monkeypatch):
    """When the runtime graph cache is cold and force_rebuild=False, the WAKE
    refresh MUST skip with reason=runtime_graph_cache_cold instead of spawning
    a heavyweight rebuild inside the lifecycle tick."""
    from iai_mcp import daemon as daemon_mod

    store = _fresh_store(tmp_path, monkeypatch)
    _seed(store)

    monkeypatch.setattr(
        daemon_mod, "_runtime_graph_cache_is_warm", lambda _store: False,
    )

    cache, meta = _cache_paths(tmp_path)
    result = daemon_mod._write_session_start_cache(
        store, cache_path=cache, meta_path=meta,
        trigger="periodic_wake", force_rebuild=False,
    )
    assert result["action"] == "skipped"
    assert result["reason"] == "runtime_graph_cache_cold"
    assert not cache.exists(), "cold-cache skip MUST NOT write the cache"

    skipped = _query_events(store, "session_start_cache_write_skipped")
    assert any(
        (e.get("data") or {}).get("reason") == "runtime_graph_cache_cold"
        for e in skipped
    ), skipped


def test_write_force_rebuild_ignores_cold_probe(tmp_path, monkeypatch):
    """The SLEEP branch passes force_rebuild=True; even a cold probe shouldn't
    stop the write — that's the cheapest moment to absorb the rebuild cost."""
    from iai_mcp import daemon as daemon_mod

    store = _fresh_store(tmp_path, monkeypatch)
    _seed(store)

    monkeypatch.setattr(
        daemon_mod, "_runtime_graph_cache_is_warm", lambda _store: False,
    )

    cache, meta = _cache_paths(tmp_path)
    result = daemon_mod._write_session_start_cache(
        store, cache_path=cache, meta_path=meta,
        trigger="sleep_pipeline", force_rebuild=True,
    )
    assert result["action"] == "wrote"
    assert cache.exists()


def test_sleep_pipeline_write_waits_for_single_flight_lock(tmp_path, monkeypatch):
    """The authoritative SLEEP refresh should wait behind an in-flight refresh
    instead of dropping the write for this cycle."""
    from iai_mcp import daemon as daemon_mod

    store = _fresh_store(tmp_path, monkeypatch)
    _seed(store)

    monkeypatch.setattr(daemon_mod, "SESSION_START_CACHE_LOCK_TIMEOUT_SEC", 5.0)
    cache, meta = _cache_paths(tmp_path)
    result: dict[str, dict] = {}
    daemon_mod._session_start_cache_lock.acquire()

    def _run_sleep_write():
        result["value"] = daemon_mod._write_session_start_cache(
            store, cache_path=cache, meta_path=meta,
            trigger="sleep_pipeline", force_rebuild=True,
        )

    t = threading.Thread(target=_run_sleep_write)
    try:
        t.start()
        time.sleep(0.05)
        assert t.is_alive(), "sleep_pipeline write should block while lock is held"
    finally:
        daemon_mod._session_start_cache_lock.release()

    t.join(timeout=5)
    assert not t.is_alive(), "sleep_pipeline write did not finish after lock release"
    assert result["value"]["action"] == "wrote"
    assert cache.exists()


def test_sleep_pipeline_write_times_out_if_single_flight_lock_stalls(
    tmp_path, monkeypatch,
):
    """A stuck in-flight refresh should degrade to an observable skip, not hang
    the daemon's SLEEP pipeline forever."""
    from iai_mcp import daemon as daemon_mod

    store = _fresh_store(tmp_path, monkeypatch)
    _seed(store)

    monkeypatch.setattr(daemon_mod, "SESSION_START_CACHE_LOCK_TIMEOUT_SEC", 0.01)
    cache, meta = _cache_paths(tmp_path)
    daemon_mod._session_start_cache_lock.acquire()
    try:
        result = daemon_mod._write_session_start_cache(
            store, cache_path=cache, meta_path=meta,
            trigger="sleep_pipeline", force_rebuild=True,
        )
    finally:
        daemon_mod._session_start_cache_lock.release()

    assert result == {"action": "skipped", "reason": "refresh_in_progress"}
    assert not cache.exists()
    skipped = _query_events(store, "session_start_cache_write_skipped")
    assert any(
        (e.get("data") or {}).get("reason") == "refresh_in_progress"
        and (e.get("data") or {}).get("trigger") == "sleep_pipeline"
        for e in skipped
    ), skipped


def test_write_emits_failed_event_on_render_crash(tmp_path, monkeypatch):
    from iai_mcp import daemon as daemon_mod
    from iai_mcp import session as session_mod

    store = _fresh_store(tmp_path, monkeypatch)
    _seed(store)

    def _boom(*_a, **_k):
        raise RuntimeError("synthetic-render-failure")

    monkeypatch.setattr(session_mod, "format_payload_as_markdown", _boom)

    cache, meta = _cache_paths(tmp_path)
    result = daemon_mod._write_session_start_cache(
        store, cache_path=cache, meta_path=meta,
        trigger="manual", force_rebuild=True,
    )
    assert result["action"] == "failed"

    failed = _query_events(store, "session_start_cache_write_failed")
    assert failed, "expected at least one _failed event"
    data = failed[-1].get("data") or {}
    assert data.get("reason") == "RuntimeError"
    assert "synthetic-render-failure" in str(data.get("error", ""))
    assert "duration_ms" in data
    assert data.get("trigger") == "manual"


# ---------------------------------------------------------------------------
# Single-flight + min-interval gates on _maybe_refresh
# ---------------------------------------------------------------------------

def test_maybe_refresh_single_flight_under_concurrency(tmp_path, monkeypatch):
    from iai_mcp import daemon as daemon_mod

    store = _fresh_store(tmp_path, monkeypatch)
    _seed(store)

    cache, meta = _cache_paths(tmp_path)

    # Make the write itself slow so the second thread definitely contends.
    real_write = daemon_mod._write_session_start_cache

    def _slow_write(store, **kwargs):
        time.sleep(0.2)
        return real_write(store, **kwargs)

    monkeypatch.setattr(daemon_mod, "_write_session_start_cache", _slow_write)

    results = []
    def _kick():
        results.append(daemon_mod._maybe_refresh_session_start_cache(
            store, trigger="manual", cache_path=cache, meta_path=meta,
            min_interval_sec=0.0, force_rebuild=True,
        ))

    t1 = threading.Thread(target=_kick)
    t2 = threading.Thread(target=_kick)
    t1.start(); t2.start()
    t1.join(); t2.join()

    actions = [r.get("action") for r in results]
    reasons = [r.get("reason") for r in results]
    assert actions.count("wrote") == 1, (actions, reasons)
    assert "refresh_in_progress" in reasons, (actions, reasons)


def test_maybe_refresh_respects_min_interval_env(tmp_path, monkeypatch):
    from iai_mcp import daemon as daemon_mod

    store = _fresh_store(tmp_path, monkeypatch)
    _seed(store)
    cache, meta = _cache_paths(tmp_path)

    monkeypatch.setenv("IAI_MCP_SESSION_CACHE_REFRESH_MIN_SEC", "120")

    # First refresh writes the cache + meta and stamps mtime=now.
    r1 = daemon_mod._maybe_refresh_session_start_cache(
        store, trigger="periodic_wake",
        cache_path=cache, meta_path=meta, force_rebuild=True,
    )
    assert r1["action"] == "wrote"

    # Immediate second call: blocked by the min-interval gate, NOT by single-flight.
    r2 = daemon_mod._maybe_refresh_session_start_cache(
        store, trigger="periodic_wake",
        cache_path=cache, meta_path=meta, force_rebuild=True,
    )
    assert r2 == {"action": "skipped", "reason": "min_interval_not_elapsed"}


def test_periodic_noop_skips_do_not_emit_event_churn(tmp_path, monkeypatch):
    from iai_mcp import daemon as daemon_mod

    store = _fresh_store(tmp_path, monkeypatch)
    _seed(store)

    cache, meta = _cache_paths(tmp_path)
    daemon_mod._write_session_start_cache(
        store, cache_path=cache, meta_path=meta,
        trigger="manual", force_rebuild=True,
    )

    r1 = daemon_mod._maybe_refresh_session_start_cache(
        store, trigger="periodic_wake",
        cache_path=cache, meta_path=meta,
        min_interval_sec=60.0, force_rebuild=True,
    )
    assert r1 == {"action": "skipped", "reason": "min_interval_not_elapsed"}

    old = time.time() - 600
    os.utime(cache, (old, old))
    r2 = daemon_mod._maybe_refresh_session_start_cache(
        store, trigger="periodic_wake",
        cache_path=cache, meta_path=meta,
        min_interval_sec=60.0, force_rebuild=True,
    )
    assert r2 == {"action": "skipped", "reason": "no_new_records"}

    skipped = _query_events(store, "session_start_cache_write_skipped")
    assert not [
        e for e in skipped
        if (e.get("data") or {}).get("trigger") == "periodic_wake"
    ], skipped


def test_maybe_refresh_runs_when_new_records_after_interval(tmp_path, monkeypatch):
    """End-to-end of the WAKE refresh: stale cache + new records + interval
    elapsed → the next call writes a fresh cache *without* the lifecycle tick
    ever entering SLEEP."""
    from iai_mcp import daemon as daemon_mod

    store = _fresh_store(tmp_path, monkeypatch)
    _seed(store, n=2)
    cache, meta = _cache_paths(tmp_path)

    daemon_mod._write_session_start_cache(
        store, cache_path=cache, meta_path=meta,
        trigger="manual", force_rebuild=True,
    )
    assert cache.exists()

    # Pretend a few minutes have passed since the first write so the min-interval
    # gate lets the second refresh through.
    old = time.time() - 600
    os.utime(cache, (old, old))

    _seed(store, n=2)  # new records → watermark moves

    result = daemon_mod._maybe_refresh_session_start_cache(
        store, trigger="periodic_wake",
        cache_path=cache, meta_path=meta,
        min_interval_sec=60.0, force_rebuild=True,
    )
    assert result["action"] == "wrote"
    new_mtime = cache.stat().st_mtime
    # The successful refresh re-stamped the mtime back to "now", so it must be
    # far newer than the artificially-aged value we set above.
    assert new_mtime > old + 60
    # Watermark sidecar reflects the new corpus size.
    sidecar = json.loads(meta.read_text())
    wm_now = daemon_mod._session_start_cache_watermark(store)
    assert sidecar["records_count"] == wm_now["records_count"]
    assert sidecar["max_vec_label"] == wm_now["max_vec_label"]
    # Content may or may not change byte-for-byte depending on payload tail,
    # but the rendered_chars field should match the new file length.
    assert sidecar["rendered_chars"] == len(cache.read_text())


# ---------------------------------------------------------------------------
# Hook TTL safety net (opt-in via IAI_MCP_SESSION_CACHE_MAX_AGE_SEC)
# ---------------------------------------------------------------------------

def _run_hook(home: Path, *, env: dict[str, str], stdin: str = '{"session_id":"x","source":"startup"}'):
    full_env = os.environ.copy()
    full_env["HOME"] = str(home)
    full_env.update(env)
    return subprocess.run(
        ["bash", str(HOOK_PATH)], input=stdin, env=full_env,
        capture_output=True, text=True, timeout=10.0,
    )


def _today_log(home: Path) -> Path:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return home / ".iai-mcp" / "logs" / f"recall-{today}.log"


def _make_stub_cli(dir_: Path, body: str) -> Path:
    cli = dir_ / "iai-mcp"
    cli.write_text(body)
    cli.chmod(cli.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return cli


def test_hook_serves_fresh_cache_when_ttl_set(tmp_path):
    """TTL set + cache fresh → still serve the cache, log cache-hit."""
    home = tmp_path / "home"; home.mkdir()
    (home / ".iai-mcp").mkdir()
    cache = home / ".iai-mcp" / ".session-start-payload.cached.md"
    cache.write_text("fresh-cache-content")

    stub_dir = tmp_path / "stub"; stub_dir.mkdir()
    _make_stub_cli(stub_dir, "#!/usr/bin/env bash\necho CLI_SHOULD_NOT_BE_CALLED\nexit 0\n")
    (home / ".iai-mcp" / ".cli-path").write_text(str(stub_dir / "iai-mcp"))

    proc = _run_hook(home, env={"IAI_MCP_SESSION_CACHE_MAX_AGE_SEC": "3600"})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "fresh-cache-content"
    assert "CLI_SHOULD_NOT_BE_CALLED" not in proc.stdout
    log = _today_log(home).read_text()
    assert "cache-hit age=" in log
    assert "cache-stale" not in log


def test_hook_falls_through_when_cache_older_than_ttl(tmp_path):
    """TTL set + cache stale → fall through to live CLI, log cache-stale."""
    home = tmp_path / "home"; home.mkdir()
    (home / ".iai-mcp").mkdir()
    cache = home / ".iai-mcp" / ".session-start-payload.cached.md"
    cache.write_text("stale-content")
    twenty_five_h_ago = time.time() - (25 * 3600)
    os.utime(cache, (twenty_five_h_ago, twenty_five_h_ago))

    stub_dir = tmp_path / "stub"; stub_dir.mkdir()
    _make_stub_cli(stub_dir, "#!/usr/bin/env bash\nprintf '%s' FRESH_FROM_CLI\nexit 0\n")
    (home / ".iai-mcp" / ".cli-path").write_text(str(stub_dir / "iai-mcp"))

    proc = _run_hook(home, env={"IAI_MCP_SESSION_CACHE_MAX_AGE_SEC": "3600"})
    assert proc.returncode == 0, proc.stderr
    assert "FRESH_FROM_CLI" in proc.stdout
    assert "stale-content" not in proc.stdout
    log = _today_log(home).read_text()
    assert "cache-stale age=" in log
    assert " max=3600s" in log


def test_hook_default_ttl_falls_through_when_unset(tmp_path):
    """Unset env → default-on 12h TTL: a 25h-old cache MUST fall through to the
    live CLI rather than re-injecting a stale prefix. This is the safety-net
    behaviour: if the daemon is down, we'd rather pay the live-CLI cost than
    silently serve a multi-day-old precache."""
    home = tmp_path / "home"; home.mkdir()
    (home / ".iai-mcp").mkdir()
    cache = home / ".iai-mcp" / ".session-start-payload.cached.md"
    cache.write_text("stale-content")
    twenty_five_h_ago = time.time() - (25 * 3600)
    os.utime(cache, (twenty_five_h_ago, twenty_five_h_ago))

    stub_dir = tmp_path / "stub"; stub_dir.mkdir()
    _make_stub_cli(stub_dir, "#!/usr/bin/env bash\nprintf '%s' FRESH_FROM_CLI\nexit 0\n")
    (home / ".iai-mcp" / ".cli-path").write_text(str(stub_dir / "iai-mcp"))

    env_no_ttl = {k: v for k, v in os.environ.items()
                  if k != "IAI_MCP_SESSION_CACHE_MAX_AGE_SEC"}
    full = env_no_ttl.copy()
    full["HOME"] = str(home)
    proc = subprocess.run(
        ["bash", str(HOOK_PATH)],
        input='{"session_id":"x","source":"startup"}',
        env=full, capture_output=True, text=True, timeout=10.0,
    )
    assert proc.returncode == 0, proc.stderr
    assert "FRESH_FROM_CLI" in proc.stdout
    assert "stale-content" not in proc.stdout
    log = _today_log(home).read_text()
    assert "cache-stale age=" in log
    assert " max=43200s" in log, log


def test_hook_ttl_disabled_explicitly_serves_stale(tmp_path):
    """IAI_MCP_SESSION_CACHE_MAX_AGE_SEC=0 → explicit legacy behaviour: cache
    served regardless of age. The opt-out is the escape hatch for operators who
    consciously accept a stale precache (e.g. offline triage)."""
    home = tmp_path / "home"; home.mkdir()
    (home / ".iai-mcp").mkdir()
    cache = home / ".iai-mcp" / ".session-start-payload.cached.md"
    cache.write_text("stale-content")
    twenty_five_h_ago = time.time() - (25 * 3600)
    os.utime(cache, (twenty_five_h_ago, twenty_five_h_ago))

    stub_dir = tmp_path / "stub"; stub_dir.mkdir()
    _make_stub_cli(stub_dir, "#!/usr/bin/env bash\nprintf '%s' CLI_SHOULD_NOT_BE_CALLED\nexit 0\n")
    (home / ".iai-mcp" / ".cli-path").write_text(str(stub_dir / "iai-mcp"))

    proc = _run_hook(home, env={"IAI_MCP_SESSION_CACHE_MAX_AGE_SEC": "0"})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "stale-content"
    assert "CLI_SHOULD_NOT_BE_CALLED" not in proc.stdout


def test_hook_ttl_garbage_env_falls_back_to_default(tmp_path):
    """A non-numeric env value MUST NOT silently disable the safety net — it
    falls back to the 12h default. Documents the case() guard."""
    home = tmp_path / "home"; home.mkdir()
    (home / ".iai-mcp").mkdir()
    cache = home / ".iai-mcp" / ".session-start-payload.cached.md"
    cache.write_text("stale-content")
    twenty_five_h_ago = time.time() - (25 * 3600)
    os.utime(cache, (twenty_five_h_ago, twenty_five_h_ago))

    stub_dir = tmp_path / "stub"; stub_dir.mkdir()
    _make_stub_cli(stub_dir, "#!/usr/bin/env bash\nprintf '%s' FRESH_FROM_CLI\nexit 0\n")
    (home / ".iai-mcp" / ".cli-path").write_text(str(stub_dir / "iai-mcp"))

    proc = _run_hook(home, env={"IAI_MCP_SESSION_CACHE_MAX_AGE_SEC": "not-a-number"})
    assert proc.returncode == 0, proc.stderr
    assert "FRESH_FROM_CLI" in proc.stdout
    log = _today_log(home).read_text()
    assert " max=43200s" in log, log

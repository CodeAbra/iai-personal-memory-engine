from __future__ import annotations

import datetime as _dt
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from iai_mcp.session import SESSION_START_CACHE_MAX_CHARS

pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX shell hook")

HOOK_PATH = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "_deploy" / "hooks" / "iai-mcp-session-recall.sh"
CACHE_REL = ".iai-mcp/.session-start-payload.cached.md"
WAKE_DEPTH_SIDECAR_REL = ".iai-mcp/.session-wake-depth"
SENTINEL = "SENTINEL_LIVE_PATH_OK"
UNAVAILABLE_MARKER = "[iai-mcp memory: UNAVAILABLE]"
# Mirrors the hook's own cache_head_cap = SESSION_START_CACHE_MAX_CHARS -
# len(stale_suffix): the marker's length is reserved out of the char cap on
# EVERY cache-hit, even when no marker is ultimately appended.
_STALE_SUFFIX = "\n\n[iai-mcp memory: STALE]"
CACHE_HEAD_CAP = SESSION_START_CACHE_MAX_CHARS - len(_STALE_SUFFIX)

def _today_log_path(home: Path) -> Path:
    today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    return home / ".iai-mcp" / "logs" / f"recall-{today}.log"

def _fresh_store(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "384")
    from iai_mcp.store import MemoryStore
    return MemoryStore()

def _make_stub_cli(dir_: Path, script: str) -> Path:
    cli = dir_ / "iai-mcp"
    cli.write_text(script)
    cli.chmod(cli.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return cli

def _run_hook(home: Path, *, extra_env: dict[str, str] | None = None,
              stdin_payload: str = '{"session_id":"x","source":"startup","cwd":"/tmp","transcript_path":""}',
              timeout: float = 10.0):
    env = os.environ.copy()
    env["HOME"] = str(home)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(HOOK_PATH)],
        input=stdin_payload,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

def _count_session_started_events(store) -> int:
    from iai_mcp.events import query_events
    rows = query_events(store, kind="session_started", limit=10000)
    return len(list(rows))

def test_daemon_writes_cache_on_rem_completion(tmp_path, monkeypatch):
    from iai_mcp import core, daemon as daemon_mod
    from iai_mcp import retrieve
    from iai_mcp.session import (
        _compose_session_start_payload,
        format_payload_as_markdown,
    )

    store = _fresh_store(tmp_path, monkeypatch)
    monkeypatch.setitem(core._profile_state, "wake_depth", "standard")

    from iai_mcp.core import _seed_l0_identity
    _seed_l0_identity(store)

    from datetime import datetime, timezone
    from uuid import uuid4
    from iai_mcp.types import EMBED_DIM, MemoryRecord
    _now = datetime.now(timezone.utc)
    for _i in range(3):
        store.insert(MemoryRecord(
            id=uuid4(),
            tier="semantic",
            literal_surface=f"Pinned fact {_i}: high-detail context.",
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
            created_at=_now,
            updated_at=_now,
            tags=[],
            language="en",
        ))

    events_before = _count_session_started_events(store)

    cache_path = tmp_path / "session-start-payload.cached.md"
    daemon_mod._write_session_start_cache(store, cache_path=cache_path)

    assert cache_path.exists(), "cache file was not created"
    content = cache_path.read_text(encoding="utf-8")
    assert content, "cache file is empty — wake_depth=standard should produce content"
    assert len(content) <= SESSION_START_CACHE_MAX_CHARS, (
        f"cache content exceeds the {SESSION_START_CACHE_MAX_CHARS}-char cap"
    )

    _g, asgn, rc = retrieve.build_runtime_graph(store)
    payload = _compose_session_start_payload(
        store, asgn, rc,
        session_id="precache",
        profile_state=dict(core._profile_state),
    )
    expected = format_payload_as_markdown(payload)[:SESSION_START_CACHE_MAX_CHARS]
    assert content == expected, "cache content does not match _compose_session_start_payload output"

    events_after = _count_session_started_events(store)
    assert events_after == events_before, (
        f"precache writer emitted {events_after - events_before} session_started "
        f"event(s); should be 0 (compose-vs-emit split is broken)"
    )

def test_precache_writer_composes_from_real_hydrated_wake_depth(tmp_path, monkeypatch):
    """The precache writer must read the store's actual tuned wake_depth,
    not a fixed literal — a store tuned to "deep" must reach composition
    as "deep", never silently downgraded to a hardcoded value."""
    from iai_mcp import core, daemon as daemon_mod
    from iai_mcp import retrieve
    import iai_mcp.session as session_mod
    from iai_mcp.session import _compose_session_start_payload

    store = _fresh_store(tmp_path, monkeypatch)
    monkeypatch.setitem(core._profile_state, "wake_depth", "deep")

    from iai_mcp.core import _seed_l0_identity
    _seed_l0_identity(store)

    captured: dict = {}
    real_compose = session_mod._compose_session_start_payload

    def _capturing_compose(*args, **kwargs):
        result = real_compose(*args, **kwargs)
        captured["profile_state"] = kwargs.get("profile_state")
        captured["payload"] = result
        return result

    monkeypatch.setattr(session_mod, "_compose_session_start_payload", _capturing_compose)

    cache_path = tmp_path / "deep-cache.md"
    daemon_mod._write_session_start_cache(store, cache_path=cache_path)

    assert captured.get("profile_state") is not None, "writer never reached composition"
    assert captured["profile_state"].get("wake_depth") == "deep", (
        "writer composed with a hardcoded value instead of the store's real hydrated wake_depth"
    )
    payload_deep = captured["payload"]
    assert payload_deep.wake_depth == "deep"

    _g, assignment, rc = retrieve.build_runtime_graph(store)
    payload_standard = _compose_session_start_payload(
        store, assignment, rc,
        session_id="precache",
        profile_state={"wake_depth": "standard"},
    )
    assert payload_standard.wake_depth == "standard"
    assert payload_deep != payload_standard, (
        "a deep-tuned store must yield a payload distinct from the standard-tuned composition"
    )

def test_precache_clamps_tampered_wake_depth_to_minimal(tmp_path, monkeypatch):
    """An out-of-enum wake_depth in the hydrated profile (a corrupt or
    tampered blob) must compose as "minimal", never crash and never reach
    composition as an unknown branch."""
    from iai_mcp import core, daemon as daemon_mod
    import iai_mcp.session as session_mod
    from iai_mcp.session import _compose_session_start_payload

    store = _fresh_store(tmp_path, monkeypatch)
    monkeypatch.setitem(core._profile_state, "wake_depth", "corrupted-depth")

    from iai_mcp.core import _seed_l0_identity
    _seed_l0_identity(store)

    captured: dict = {}
    real_compose = session_mod._compose_session_start_payload

    def _capturing_compose(*args, **kwargs):
        captured["profile_state"] = kwargs.get("profile_state")
        return real_compose(*args, **kwargs)

    monkeypatch.setattr(session_mod, "_compose_session_start_payload", _capturing_compose)

    cache_path = tmp_path / "tamper-cache.md"
    daemon_mod._write_session_start_cache(store, cache_path=cache_path)

    assert captured.get("profile_state") is not None
    assert captured["profile_state"].get("wake_depth") == "minimal", (
        "an out-of-enum wake_depth must clamp to minimal before reaching composition"
    )
    assert cache_path.exists(), "cache write must not crash on a tampered wake_depth"

def test_cache_file_mode_is_owner_only(tmp_path, monkeypatch):
    from iai_mcp import daemon as daemon_mod

    store = _fresh_store(tmp_path, monkeypatch)

    from iai_mcp.core import _seed_l0_identity
    _seed_l0_identity(store)

    from datetime import datetime, timezone
    from uuid import uuid4
    from iai_mcp.types import EMBED_DIM, MemoryRecord
    _now = datetime.now(timezone.utc)
    for _i in range(3):
        store.insert(MemoryRecord(
            id=uuid4(),
            tier="semantic",
            literal_surface=f"Pinned fact {_i}: high-detail context.",
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
            created_at=_now,
            updated_at=_now,
            tags=[],
            language="en",
        ))

    cache_path = tmp_path / "session-start-payload.cached.md"
    daemon_mod._write_session_start_cache(store, cache_path=cache_path)

    assert cache_path.exists(), "cache file was not created"
    assert oct(stat.S_IMODE(cache_path.stat().st_mode)) == "0o600", (
        f"cache file mode is not 0o600; got "
        f"{oct(stat.S_IMODE(cache_path.stat().st_mode))}"
    )

def test_precache_does_not_compress_payload(tmp_path, monkeypatch):
    from iai_mcp import daemon as daemon_mod

    store = _fresh_store(tmp_path, monkeypatch)

    from iai_mcp.core import _seed_l0_identity
    _seed_l0_identity(store)

    from datetime import datetime, timezone
    from uuid import uuid4
    from iai_mcp.types import EMBED_DIM, MemoryRecord
    _now = datetime.now(timezone.utc)
    for _i in range(3):
        store.insert(MemoryRecord(
            id=uuid4(),
            tier="semantic",
            literal_surface=f"Pinned fact {_i}: high-detail context.",
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
            created_at=_now,
            updated_at=_now,
            tags=[],
            language="en",
        ))

    cache_path = tmp_path / "c.md"
    daemon_mod._write_session_start_cache(store, cache_path=cache_path)

    assert cache_path.exists(), "cache file was not created by the precache writer"
    assert cache_path.read_text(encoding="utf-8"), "cache file is empty after precache write"

def test_format_payload_as_markdown_wake_depth_marker_both_shapes():
    """format_payload_as_markdown must emit a leading-line wake_depth marker
    matching payload.wake_depth (dataclass) / payload.get("wake_depth")
    (dict), placed right after the source_watermark line."""
    from iai_mcp.session import format_payload_as_markdown, SessionStartPayload

    watermark = "2026-09-06T15:00:00.000000+00:00"

    payload_dc = SessionStartPayload(
        l0="identity content", wake_depth="deep", source_watermark=watermark,
    )
    rendered_dc = format_payload_as_markdown(payload_dc)
    lines_dc = rendered_dc.split("\n")
    assert lines_dc[0] == f"<!-- iai-mcp:source_watermark={watermark} -->"
    assert lines_dc[1] == "<!-- iai-mcp:wake_depth=deep -->"

    payload_dict = {
        "l0": "identity content",
        "wake_depth": "standard",
        "source_watermark": watermark,
    }
    rendered_dict = format_payload_as_markdown(payload_dict)
    lines_dict = rendered_dict.split("\n")
    assert lines_dict[0] == f"<!-- iai-mcp:source_watermark={watermark} -->"
    assert lines_dict[1] == "<!-- iai-mcp:wake_depth=standard -->"


def test_daemon_writes_wake_depth_sidecar_reflecting_current_depth(tmp_path, monkeypatch):
    from iai_mcp import core, daemon as daemon_mod

    store = _fresh_store(tmp_path, monkeypatch)
    monkeypatch.setitem(core._profile_state, "wake_depth", "deep")

    sidecar_path = tmp_path / "wake-depth.sidecar"
    daemon_mod._write_wake_depth_sidecar(store, sidecar_path=sidecar_path)

    assert sidecar_path.exists(), "sidecar was not created"
    assert sidecar_path.read_text(encoding="utf-8").strip() == "deep"
    assert oct(stat.S_IMODE(sidecar_path.stat().st_mode)) == "0o600", (
        f"sidecar mode is not 0o600; got "
        f"{oct(stat.S_IMODE(sidecar_path.stat().st_mode))}"
    )


def test_wake_depth_sidecar_diverges_from_stale_embedded_marker(tmp_path, monkeypatch):
    """The exact staleness signal the hook guard depends on: the cache embeds
    the depth at compose time; when the tuner changes depth afterward
    without a cache refresh, the live sidecar (current) must disagree with
    the marker already embedded in the cache (compose-time)."""
    from iai_mcp import core, daemon as daemon_mod

    store = _fresh_store(tmp_path, monkeypatch)
    monkeypatch.setitem(core._profile_state, "wake_depth", "standard")

    from iai_mcp.core import _seed_l0_identity
    _seed_l0_identity(store)

    cache_path = tmp_path / "cache.md"
    daemon_mod._write_session_start_cache(store, cache_path=cache_path)
    cache_content = cache_path.read_text(encoding="utf-8")
    embedded_line = cache_content.split("\n")[1]
    assert embedded_line == "<!-- iai-mcp:wake_depth=standard -->"

    monkeypatch.setitem(core._profile_state, "wake_depth", "deep")
    sidecar_path = tmp_path / "wake-depth.sidecar"
    daemon_mod._write_wake_depth_sidecar(store, sidecar_path=sidecar_path)
    sidecar_value = sidecar_path.read_text(encoding="utf-8").strip()

    assert sidecar_value == "deep"
    assert sidecar_value != "standard", (
        "the compose-time embedded marker must differ from the live sidecar "
        "after an un-refreshed depth change"
    )


def test_hook_reads_cache_when_fresh(tmp_path):
    assert HOOK_PATH.exists(), f"hook script missing: {HOOK_PATH}"
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp").mkdir()

    cache_content = "# L0 identity\nfresh-cache-content-marker"
    (home / CACHE_REL).write_text(cache_content)

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    _make_stub_cli(stub_dir, "#!/usr/bin/env bash\necho CLI_SHOULD_NOT_BE_CALLED\nexit 0\n")
    (home / ".iai-mcp" / ".cli-path").write_text(str(stub_dir / "iai-mcp"))

    proc = _run_hook(home)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == cache_content, (
        f"hook did not return cache verbatim. stdout={proc.stdout!r}"
    )
    assert "CLI_SHOULD_NOT_BE_CALLED" not in proc.stdout, (
        "hook called the CLI instead of reading the cache"
    )

    log_path = _today_log_path(home)
    assert log_path.exists(), f"hook log missing: {log_path}"
    log_text = log_path.read_text(encoding="utf-8")
    assert "cache-hit age=" in log_text, (
        f"expected 'cache-hit age=' marker in log; got:\n{log_text}"
    )

def test_hook_cache_at_exact_head_cap_not_truncated(tmp_path):
    """A cache sized exactly at cache_head_cap (9975, the marker-reserved
    boundary reserved on EVERY cache-hit even when no marker is ultimately
    appended) must still be served in full -- the existing fresh-cache test
    is far below either cap and cannot catch a regression at this boundary.
    """
    assert HOOK_PATH.exists(), f"hook script missing: {HOOK_PATH}"
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp").mkdir()

    prefix = "# L0 identity\n"
    cache_content = prefix + ("A" * (CACHE_HEAD_CAP - len(prefix)))
    assert len(cache_content) == CACHE_HEAD_CAP
    (home / CACHE_REL).write_text(cache_content)

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    _make_stub_cli(stub_dir, "#!/usr/bin/env bash\necho CLI_SHOULD_NOT_BE_CALLED\nexit 0\n")
    (home / ".iai-mcp" / ".cli-path").write_text(str(stub_dir / "iai-mcp"))

    proc = _run_hook(home)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == cache_content, (
        f"a cache at exactly cache_head_cap must serve byte-identical, no "
        f"truncation. got {len(proc.stdout)} bytes, expected {len(cache_content)}"
    )


def test_hook_cache_over_head_cap_is_truncated(tmp_path):
    """The other half of the boundary proof: a cache sized just OVER
    cache_head_cap must be truncated to exactly that many bytes -- the cap
    is a real ceiling, not a no-op.
    """
    assert HOOK_PATH.exists(), f"hook script missing: {HOOK_PATH}"
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp").mkdir()

    prefix = "# L0 identity\n"
    over_size = CACHE_HEAD_CAP + 100
    cache_content = prefix + ("A" * (over_size - len(prefix)))
    assert len(cache_content) == over_size
    (home / CACHE_REL).write_text(cache_content)

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    _make_stub_cli(stub_dir, "#!/usr/bin/env bash\necho CLI_SHOULD_NOT_BE_CALLED\nexit 0\n")
    (home / ".iai-mcp" / ".cli-path").write_text(str(stub_dir / "iai-mcp"))

    proc = _run_hook(home)
    assert proc.returncode == 0, proc.stderr
    assert len(proc.stdout) == CACHE_HEAD_CAP, (
        f"expected truncation to exactly {CACHE_HEAD_CAP} bytes, "
        f"got {len(proc.stdout)}"
    )
    assert proc.stdout == cache_content[:CACHE_HEAD_CAP]


def test_hook_falls_back_when_cache_absent(tmp_path):
    assert HOOK_PATH.exists(), f"hook script missing: {HOOK_PATH}"
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp").mkdir()

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    _make_stub_cli(stub_dir, f"#!/usr/bin/env bash\nprintf '%s' '{SENTINEL}'\nexit 0\n")
    (home / ".iai-mcp" / ".cli-path").write_text(str(stub_dir / "iai-mcp"))

    proc = _run_hook(home)
    assert proc.returncode == 0, proc.stderr
    assert SENTINEL in proc.stdout, (
        f"fallback CLI sentinel not in stdout. stdout={proc.stdout!r}"
    )

    log_path = _today_log_path(home)
    assert log_path.exists(), f"hook log missing: {log_path}"
    log_text = log_path.read_text(encoding="utf-8")
    assert "cache-miss absent" in log_text, (
        f"expected 'cache-miss absent' marker in log; got:\n{log_text}"
    )

def test_hook_serves_stale_cache(tmp_path):
    """Retired the old silent-stale pin (this test used to assert a 25h-old,
    watermark-less cache is served with NO staleness concept at all -- a
    negative assertion about a marker that never existed). This test now
    exercises a real staleness DETECTOR: a cache whose embedded watermark
    trails the live store sidecar by a full clock-hour must now carry the
    STALE marker, not silently serve as if current.
    """
    assert HOOK_PATH.exists(), f"hook script missing: {HOOK_PATH}"
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp" / "hippo").mkdir(parents=True)

    cache_content = (
        "<!-- iai-mcp:source_watermark=2026-09-01T00:00:00.000000+00:00 -->\n\n"
        "# stale\nold content that must still be served"
    )
    stale_cache = home / CACHE_REL
    stale_cache.write_text(cache_content)
    (home / ".iai-mcp" / "hippo" / ".max-created-at").write_text(
        "2026-09-06T15:30:00.000000+00:00"
    )

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    _make_stub_cli(stub_dir, "#!/usr/bin/env bash\necho CLI_SHOULD_NOT_BE_CALLED\nexit 0\n")
    (home / ".iai-mcp" / ".cli-path").write_text(str(stub_dir / "iai-mcp"))

    proc = _run_hook(home)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == f"{cache_content}\n\n[iai-mcp memory: STALE]", (
        f"hook did not serve the cache plus the STALE marker. stdout={proc.stdout!r}"
    )
    assert "CLI_SHOULD_NOT_BE_CALLED" not in proc.stdout, (
        "hook called the CLI instead of reading the stale cache"
    )

    log_path = _today_log_path(home)
    assert log_path.exists(), f"hook log missing: {log_path}"
    log_text = log_path.read_text(encoding="utf-8")
    assert "cache-hit age=" in log_text, (
        f"expected 'cache-hit age=' marker in log; got:\n{log_text}"
    )
    assert "stale=true" in log_text, (
        f"expected 'stale=true' detection signal in log; got:\n{log_text}"
    )


def test_hook_sub_hour_divergence_does_not_fire(tmp_path):
    """A live sidecar within the SAME clock-hour as the embedded watermark
    must NOT flag stale -- the detector is hour-granularity, not strict
    inequality, to avoid firing on every in-flight sub-hour race.
    """
    assert HOOK_PATH.exists(), f"hook script missing: {HOOK_PATH}"
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp" / "hippo").mkdir(parents=True)

    cache_content = (
        "<!-- iai-mcp:source_watermark=2026-09-06T15:05:00.000000+00:00 -->\n\n"
        "# fresh enough\nsame-hour content"
    )
    (home / CACHE_REL).write_text(cache_content)
    (home / ".iai-mcp" / "hippo" / ".max-created-at").write_text(
        "2026-09-06T15:59:59.999999+00:00"
    )

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    _make_stub_cli(stub_dir, "#!/usr/bin/env bash\necho CLI_SHOULD_NOT_BE_CALLED\nexit 0\n")
    (home / ".iai-mcp" / ".cli-path").write_text(str(stub_dir / "iai-mcp"))

    proc = _run_hook(home)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == cache_content, (
        f"a sub-hour divergence must serve the cache verbatim, no marker. "
        f"stdout={proc.stdout!r}"
    )

    log_text = _today_log_path(home).read_text(encoding="utf-8")
    assert "stale=false" in log_text, (
        f"expected 'stale=false' for a sub-hour divergence; got:\n{log_text}"
    )


def test_hook_honors_iai_mcp_store_for_live_sidecar(tmp_path):
    """Under a custom IAI_MCP_STORE, the staleness comparison must read that
    store's own sidecar, not the default $HOME/.iai-mcp one -- a diverging
    default-store leftover (or its total absence) must not affect the
    verdict. Confirmed by pointing IAI_MCP_STORE at a wholly separate
    directory whose sidecar disagrees with the served pack's embedded
    watermark, while the default $HOME path carries a matching (non-stale)
    value that would otherwise mask the divergence.
    """
    assert HOOK_PATH.exists(), f"hook script missing: {HOOK_PATH}"
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp" / "hippo").mkdir(parents=True)

    cache_content = (
        "<!-- iai-mcp:source_watermark=2026-09-01T00:00:00.000000+00:00 -->\n\n"
        "# custom store\ncontent served from the custom-store deployment"
    )
    (home / CACHE_REL).write_text(cache_content)
    # Default-store sidecar: intentionally matches the embedded watermark's
    # hour, so a hook that wrongly ignores IAI_MCP_STORE would report fresh.
    (home / ".iai-mcp" / "hippo" / ".max-created-at").write_text(
        "2026-09-01T00:30:00.000000+00:00"
    )

    custom_store = tmp_path / "custom-store"
    (custom_store / "hippo").mkdir(parents=True)
    (custom_store / "hippo" / ".max-created-at").write_text(
        "2026-09-06T15:30:00.000000+00:00"
    )

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    _make_stub_cli(stub_dir, "#!/usr/bin/env bash\necho CLI_SHOULD_NOT_BE_CALLED\nexit 0\n")
    (home / ".iai-mcp" / ".cli-path").write_text(str(stub_dir / "iai-mcp"))

    proc = _run_hook(home, extra_env={"IAI_MCP_STORE": str(custom_store)})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == f"{cache_content}\n\n[iai-mcp memory: STALE]", (
        f"hook must compare against the IAI_MCP_STORE sidecar, not the "
        f"default $HOME one. stdout={proc.stdout!r}"
    )


def test_hook_embedded_watermark_matches_real_store_insert(tmp_path, monkeypatch):
    """Discriminating test: drives a REAL MemoryStore insert (which stamps
    the live sidecar via store.py's write path), composes a session-start
    pack from that same store, and confirms the pack's embedded watermark
    equals the sidecar the shell hook resolves. A test that only hand-plants
    both files under a fake HOME cannot catch a wrong production
    path-resolution -- this one can.
    """
    store = _fresh_store(tmp_path, monkeypatch)

    from iai_mcp.core import _seed_l0_identity
    _seed_l0_identity(store)

    from datetime import datetime, timezone
    from uuid import uuid4
    from iai_mcp.types import EMBED_DIM, MemoryRecord
    now = datetime.now(timezone.utc)
    store.insert(MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface="real insert stamps the sidecar",
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

    from iai_mcp import store_watermark
    sidecar_dir = getattr(store.db, "_hippo_dir", store.root / "hippo")
    sidecar_value = store_watermark.read(sidecar_dir)
    assert sidecar_value, "real insert must have stamped the sidecar"

    from iai_mcp import retrieve
    from iai_mcp.session import _compose_session_start_payload
    _g, asgn, rc = retrieve.build_runtime_graph(store)
    payload = _compose_session_start_payload(
        store, asgn, rc,
        session_id="discriminating",
        profile_state={"wake_depth": "standard"},
    )
    assert payload.source_watermark == sidecar_value, (
        f"pack watermark {payload.source_watermark!r} != sidecar {sidecar_value!r}"
    )


def test_hook_wake_depth_mismatch_falls_through_to_live_cli(tmp_path):
    """A proven wake_depth mismatch (embedded marker != live sidecar) must
    never serve the pre-tuning pack -- the hook falls through to the CLI."""
    assert HOOK_PATH.exists(), f"hook script missing: {HOOK_PATH}"
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp" / "hippo").mkdir(parents=True)

    cache_content = (
        "<!-- iai-mcp:source_watermark=2026-09-06T15:30:00.000000+00:00 -->\n"
        "<!-- iai-mcp:wake_depth=minimal -->\n\n"
        "# content\nsome cached content"
    )
    (home / CACHE_REL).write_text(cache_content)
    (home / ".iai-mcp" / "hippo" / ".max-created-at").write_text(
        "2026-09-06T15:45:00.000000+00:00"
    )
    (home / WAKE_DEPTH_SIDECAR_REL).write_text("deep")

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    _make_stub_cli(stub_dir, f"#!/usr/bin/env bash\nprintf '%s' '{SENTINEL}'\nexit 0\n")
    (home / ".iai-mcp" / ".cli-path").write_text(str(stub_dir / "iai-mcp"))

    proc = _run_hook(home)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == SENTINEL, (
        f"hook did not fall through to the live CLI on a proven wake_depth "
        f"mismatch. stdout={proc.stdout!r}"
    )

    log_text = _today_log_path(home).read_text(encoding="utf-8")
    assert "wake_depth_stale=true" in log_text, (
        f"expected 'wake_depth_stale=true' detection signal in log; got:\n{log_text}"
    )


def test_hook_wake_depth_mismatch_serves_stale_cache_when_cli_unavailable(tmp_path):
    """When the wake_depth fall-through's CLI path is unreachable (daemon
    down), the hook must serve the cache plus STALE, never bare UNAVAILABLE
    or zero content -- preserving daemon-independence."""
    assert HOOK_PATH.exists(), f"hook script missing: {HOOK_PATH}"
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp" / "hippo").mkdir(parents=True)

    cache_content = (
        "<!-- iai-mcp:source_watermark=2026-09-06T15:30:00.000000+00:00 -->\n"
        "<!-- iai-mcp:wake_depth=minimal -->\n\n"
        "# content\nsome cached content"
    )
    (home / CACHE_REL).write_text(cache_content)
    (home / ".iai-mcp" / "hippo" / ".max-created-at").write_text(
        "2026-09-06T15:45:00.000000+00:00"
    )
    (home / WAKE_DEPTH_SIDECAR_REL).write_text("deep")

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    _make_stub_cli(
        stub_dir,
        f"#!/usr/bin/env bash\nprintf '%s' '{UNAVAILABLE_MARKER}'\nexit 0\n",
    )
    (home / ".iai-mcp" / ".cli-path").write_text(str(stub_dir / "iai-mcp"))

    proc = _run_hook(home)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == f"{cache_content}\n\n[iai-mcp memory: STALE]", (
        f"hook did not fall back to cache+STALE when the CLI was unavailable "
        f"after a wake_depth fall-through. stdout={proc.stdout!r}"
    )

    log_text = _today_log_path(home).read_text(encoding="utf-8")
    assert "wake-depth-fallthrough cli-unavailable served-stale-cache" in log_text, (
        f"expected fall-through-served-stale marker in log; got:\n{log_text}"
    )


def test_hook_wake_depth_mismatch_serves_stale_cache_when_cli_unresolvable(tmp_path):
    """When the wake_depth fall-through hits the CLI-resolution block and NO
    binary can be resolved at all (no override, no cache, no PATH entry, no
    baked-in candidate) -- distinct from the sibling test above, where the
    CLI resolves but the daemon socket behind it is down -- the hook must
    still serve the cache plus STALE, never exit with empty stdout."""
    assert HOOK_PATH.exists(), f"hook script missing: {HOOK_PATH}"
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp" / "hippo").mkdir(parents=True)

    cache_content = (
        "<!-- iai-mcp:source_watermark=2026-09-06T15:30:00.000000+00:00 -->\n"
        "<!-- iai-mcp:wake_depth=minimal -->\n\n"
        "# content\nsome cached content"
    )
    (home / CACHE_REL).write_text(cache_content)
    (home / ".iai-mcp" / "hippo" / ".max-created-at").write_text(
        "2026-09-06T15:45:00.000000+00:00"
    )
    (home / WAKE_DEPTH_SIDECAR_REL).write_text("deep")
    # No .cli-path file, no candidate present under this fresh $HOME, and
    # PATH below excludes any directory that could contain iai-mcp.
    # The hook's baked-in candidate list also probes two absolute,
    # host-real paths that a sandboxed $HOME/PATH cannot shadow -- assert
    # they are not a real iai-mcp on this host so a false pass here turns
    # into a loud, explained failure instead of silently skipping the
    # branch under test.
    for _abs_candidate in ("/opt/homebrew/bin/iai-mcp", "/usr/local/bin/iai-mcp"):
        assert not os.access(_abs_candidate, os.X_OK), (
            f"test assumes {_abs_candidate} is not an executable iai-mcp on "
            "this host; the CLI-unresolvable path is not hermetically "
            "exercised here"
        )

    proc = _run_hook(
        home,
        # Empty (not unset) IAI_MCP_SESSION_RECALL_CLI: the hook's own `-n`
        # check treats empty as absent, and this makes the test hermetic
        # against a developer shell that exports the override per the
        # hook's own resolution-order comment.
        extra_env={"PATH": "/usr/bin:/bin", "IAI_MCP_SESSION_RECALL_CLI": ""},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == f"{cache_content}\n\n[iai-mcp memory: STALE]", (
        f"hook did not fall back to cache+STALE when the CLI binary was "
        f"unresolvable after a wake_depth fall-through. stdout={proc.stdout!r}"
    )
    assert proc.stdout != "", "session-start must never emit empty when a cached payload exists"

    log_text = _today_log_path(home).read_text(encoding="utf-8")
    assert "skipped: iai-mcp CLI not found, served-stale-cache" in log_text, (
        f"expected CLI-not-found-served-stale marker in log; got:\n{log_text}"
    )


def test_hook_wake_depth_unknown_sidecar_fails_open(tmp_path):
    """An unreadable/missing wake_depth sidecar is UNKNOWN, not a proven
    mismatch -- the hook must fail open and serve the cache normally,
    exactly as session_scope_blocks does for the per-turn hook."""
    assert HOOK_PATH.exists(), f"hook script missing: {HOOK_PATH}"
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp" / "hippo").mkdir(parents=True)

    cache_content = (
        "<!-- iai-mcp:source_watermark=2026-09-06T15:30:00.000000+00:00 -->\n"
        "<!-- iai-mcp:wake_depth=minimal -->\n\n"
        "# content\nsome cached content"
    )
    (home / CACHE_REL).write_text(cache_content)
    (home / ".iai-mcp" / "hippo" / ".max-created-at").write_text(
        "2026-09-06T15:45:00.000000+00:00"
    )
    # No wake_depth sidecar written -- unknown, must fail open.

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    _make_stub_cli(stub_dir, "#!/usr/bin/env bash\necho CLI_SHOULD_NOT_BE_CALLED\nexit 0\n")
    (home / ".iai-mcp" / ".cli-path").write_text(str(stub_dir / "iai-mcp"))

    proc = _run_hook(home)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == cache_content, (
        f"an unknown wake_depth sidecar must fail open and serve the cache "
        f"verbatim, no fall-through. stdout={proc.stdout!r}"
    )
    assert "CLI_SHOULD_NOT_BE_CALLED" not in proc.stdout, (
        "hook called the CLI instead of failing open on an unknown sidecar"
    )

    log_text = _today_log_path(home).read_text(encoding="utf-8")
    assert "wake_depth_stale=false" in log_text, (
        f"expected 'wake_depth_stale=false' for an unknown sidecar; got:\n{log_text}"
    )


def test_marker_vocabulary_matches_session_reserve_source():
    """session.py's marker-reserve constants must match the literal each
    downstream caller actually appends -- an edit to one without the other
    would silently reopen the mid-line-truncation gap the reserve exists
    to close."""
    from iai_mcp.cli._capture import _AVAILABILITY_MARKER_HEALTHY
    from iai_mcp.session import SESSION_START_MARKER_HEALTHY, SESSION_START_MARKER_STALE

    assert SESSION_START_MARKER_HEALTHY == _AVAILABILITY_MARKER_HEALTHY, (
        "session.py's HEALTHY marker literal drifted from cli/_capture.py's"
    )

    hook_body = HOOK_PATH.read_text(encoding="utf-8")
    m = re.search(r'stale_marker="([^"]+)"', hook_body)
    assert m is not None, "hook script's stale_marker literal not found"
    assert m.group(1) == SESSION_START_MARKER_STALE, (
        "session.py's STALE marker literal drifted from the recall hook's"
    )

from __future__ import annotations

import datetime as _dt
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX shell hook")

HOOK_PATH = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "_deploy" / "hooks" / "iai-mcp-session-recall.sh"
CACHE_REL = ".iai-mcp/.session-start-payload.cached.md"
SENTINEL = "SENTINEL_LIVE_PATH_OK"
# Mirrors the hook's own cache_head_cap = 10000 - len(stale_suffix): the
# marker's length is reserved out of the 10000-char cap on EVERY cache-hit,
# even when no marker is ultimately appended.
_STALE_SUFFIX = "\n\n[iai-mcp memory: STALE]"
CACHE_HEAD_CAP = 10000 - len(_STALE_SUFFIX)

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
    from iai_mcp import daemon as daemon_mod
    from iai_mcp import retrieve
    from iai_mcp.session import (
        _compose_session_start_payload,
        format_payload_as_markdown,
    )

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

    events_before = _count_session_started_events(store)

    cache_path = tmp_path / "session-start-payload.cached.md"
    daemon_mod._write_session_start_cache(store, cache_path=cache_path)

    assert cache_path.exists(), "cache file was not created"
    content = cache_path.read_text(encoding="utf-8")
    assert content, "cache file is empty — wake_depth=standard should produce content"
    assert len(content) <= 10000, "cache content exceeds 10000-char cap"

    _g, asgn, rc = retrieve.build_runtime_graph(store)
    payload = _compose_session_start_payload(
        store, asgn, rc,
        session_id="precache",
        profile_state={"wake_depth": "standard"},
    )
    expected = format_payload_as_markdown(payload)[:10000]
    assert content == expected, "cache content does not match _compose_session_start_payload output"

    events_after = _count_session_started_events(store)
    assert events_after == events_before, (
        f"precache writer emitted {events_after - events_before} session_started "
        f"event(s); should be 0 (compose-vs-emit split is broken)"
    )

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

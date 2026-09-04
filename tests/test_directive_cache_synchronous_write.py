"""Directive cache writer is atomic, mode 0600, empty-safe, fail-soft, and
refreshes synchronously from every call site that can set the flag,
including a non-RPC path."""

from __future__ import annotations

import stat
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from iai_mcp.capture import capture_turn
from iai_mcp.directive_cache import write_directives_cache
from iai_mcp.embed import embedder_for_store
from iai_mcp.retrieve import contradict
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "lancedb")


def _directive_record(text: str) -> MemoryRecord:
    rec = MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    )
    rec.directive = True
    return rec


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_write_directives_cache_is_atomic_and_mode_0600(driver, store, monkeypatch, tmp_path):
    _select_driver(driver, monkeypatch)
    store.insert(_directive_record("from now on reply in English"))

    cache_path = tmp_path / "cache" / ".directives.cached.md"
    write_directives_cache(store, cache_path=cache_path)

    assert cache_path.exists()
    assert "from now on reply in English" in cache_path.read_text(encoding="utf-8")
    mode = stat.S_IMODE(cache_path.stat().st_mode)
    assert mode == 0o600


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_write_directives_cache_empty_set_is_safe(driver, store, monkeypatch, tmp_path):
    _select_driver(driver, monkeypatch)

    cache_path = tmp_path / "cache" / ".directives.cached.md"
    write_directives_cache(store, cache_path=cache_path)

    assert cache_path.exists()
    assert cache_path.read_text(encoding="utf-8") == ""


def test_write_directives_cache_is_fail_soft_on_render_error(store, tmp_path, monkeypatch):
    import iai_mcp.session as session_mod

    def _boom(_store):
        raise RuntimeError("render exploded")

    monkeypatch.setattr(session_mod, "render_directive_segment", _boom)

    cache_path = tmp_path / "cache" / ".directives.cached.md"
    write_directives_cache(store, cache_path=cache_path)  # must not raise

    assert not cache_path.exists()


def test_write_directives_cache_is_fail_soft_on_write_error(store, tmp_path):
    store.insert(_directive_record("fail soft on write"))

    blocking_file = tmp_path / "blocked"
    blocking_file.write_text("not a directory")
    cache_path = blocking_file / ".directives.cached.md"

    write_directives_cache(store, cache_path=cache_path)  # must not raise


def test_write_directives_cache_flushes_the_record_buffer_before_reading(
    store, monkeypatch, tmp_path,
):
    """Every other test in the suite runs under the harness's per-insert
    autoflush; this test disables it so the writer's OWN flush is the thing
    proven to make a just-inserted, still-buffered directive visible."""
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")

    store.insert(_directive_record("stay terse under load"))

    cache_path = tmp_path / "cache" / ".directives.cached.md"
    write_directives_cache(store, cache_path=cache_path)

    assert cache_path.exists()
    assert "stay terse under load" in cache_path.read_text(encoding="utf-8")


def test_daemon_boot_precache_calls_directive_cache_writer(store, monkeypatch):
    calls: list = []

    def _spy(passed_store, **_kwargs):
        calls.append(passed_store)

    import iai_mcp.directive_cache as directive_cache_mod
    monkeypatch.setattr(directive_cache_mod, "write_directives_cache", _spy)

    from iai_mcp import daemon as daemon_mod
    daemon_mod._write_session_start_cache(store)

    assert calls == [store]


def test_capture_turn_synchronously_refreshes_directive_cache(store, monkeypatch, tmp_path):
    """capture_turn called DIRECTLY, not through an RPC handler -- the same
    shape deferred_drain_worker.py uses -- must refresh the cache with no
    sleep, and contradict must clear it just as immediately."""
    cache_path = tmp_path / "cache" / ".directives.cached.md"

    import iai_mcp.directive_cache as directive_cache_mod
    _real_write = directive_cache_mod.write_directives_cache

    def _write_to_tmp(passed_store, **_kwargs):
        _real_write(passed_store, cache_path=cache_path)

    monkeypatch.setattr(directive_cache_mod, "write_directives_cache", _write_to_tmp)

    result = capture_turn(
        store=store, cue="c", text="from now on reply in English",
        directive=True, session_id="s1", role="user",
    )
    assert result["status"] == "inserted", result
    assert cache_path.exists()
    assert "from now on reply in English" in cache_path.read_text(encoding="utf-8")

    original_id = UUID(result["record_id"])
    new_fact = "from now on reply in Spanish"
    new_embedding = list(embedder_for_store(store).embed(new_fact))
    contradict(store, original_id, new_fact, new_embedding)

    assert "from now on reply in English" not in cache_path.read_text(encoding="utf-8")


def test_full_budget_directive_cache_stays_under_hook_truncation_cap(
    store, tmp_path,
):
    """iai-mcp-per-turn-recall.sh's emit_directives cannot import
    directive_budget.py (bash, not Python) and hardcodes its own head -c
    cap. This proves the worst-case rendered cache still fits under that
    cap, so a future bump to the shared budget constants that silently
    outgrows the hook's cap fails loudly here instead."""
    from iai_mcp.directive_budget import DIRECTIVE_LINE_CHAR_CAP, DIRECTIVE_MAX_COUNT

    # Mirrors iai-mcp-per-turn-recall.sh's emit_directives `head -c` cap.
    HOOK_TRUNCATION_CAP_BYTES = 4096

    for i in range(DIRECTIVE_MAX_COUNT):
        text = f"directive {i} " + ("x" * DIRECTIVE_LINE_CHAR_CAP)
        store.insert(_directive_record(text))

    cache_path = tmp_path / "cache" / ".directives.cached.md"
    write_directives_cache(store, cache_path=cache_path)

    assert cache_path.exists()
    size = cache_path.stat().st_size
    assert size < HOOK_TRUNCATION_CAP_BYTES, (
        f"full-budget directive cache is {size} bytes, at or over the hook's "
        f"hardcoded {HOOK_TRUNCATION_CAP_BYTES}-byte truncation cap -- "
        f"bump the cap in iai-mcp-per-turn-recall.sh's emit_directives"
    )


def test_ordinary_capture_does_not_rewrite_directive_cache(store, monkeypatch, tmp_path):
    calls: list = []

    import iai_mcp.directive_cache as directive_cache_mod

    def _spy(passed_store, **_kwargs):
        calls.append(passed_store)

    monkeypatch.setattr(directive_cache_mod, "write_directives_cache", _spy)

    result = capture_turn(
        store=store, cue="c", text="alice attended the weekly standup meeting",
        session_id="s1", role="user",
    )
    assert result["status"] == "inserted", result
    assert calls == []


def test_default_call_shape_never_touches_the_real_home_cache(store):
    """The dangerous call shape is exactly the one production uses:
    write_directives_cache(store) with NO explicit cache_path, relying on
    the module-level DIRECTIVES_CACHE_PATH default. Under the hermetic
    fixture that default is monkeypatched to a tmp path; this proves the
    call-time-resolution fix actually honors that patch, and that the
    operator's real ~/.iai-mcp/.directives.cached.md is neither created nor
    modified. The real path is only stat()'d for an existence/mtime
    fingerprint -- never opened or read."""
    import iai_mcp.directive_cache as directive_cache_mod
    import iai_mcp.hippo as hippo_mod

    real_cache_path = hippo_mod._REAL_IAI_ROOT / ".directives.cached.md"
    real_existed_before = real_cache_path.exists()
    real_mtime_before = real_cache_path.stat().st_mtime if real_existed_before else None

    resolved = directive_cache_mod.DIRECTIVES_CACHE_PATH
    assert hippo_mod._REAL_IAI_ROOT.resolve() not in resolved.resolve().parents, (
        f"DIRECTIVES_CACHE_PATH={resolved!r} resolves under the real store "
        f"{hippo_mod._REAL_IAI_ROOT!r} -- the hermetic fixture failed to patch it"
    )

    store.insert(_directive_record("isolation proof directive"))
    directive_cache_mod.write_directives_cache(store)  # no cache_path override

    assert resolved.exists()
    assert "isolation proof directive" in resolved.read_text(encoding="utf-8")

    real_exists_after = real_cache_path.exists()
    if real_existed_before:
        assert real_exists_after
        assert real_cache_path.stat().st_mtime == real_mtime_before, (
            "real ~/.iai-mcp/.directives.cached.md was modified by a "
            "cache_path-less write_directives_cache() call"
        )
    else:
        assert not real_exists_after, (
            "real ~/.iai-mcp/.directives.cached.md was created by a "
            "cache_path-less write_directives_cache() call"
        )

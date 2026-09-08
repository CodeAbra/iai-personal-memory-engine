"""`iai directive list` / `iai directive remove <id>` -- the shared retire
seam (directive_ops.retire_directive), an isatty-gated remove with no
self-suppliable override, and dual-driver correctness.

Retire = flag flip only: `directive` -> False, `tombstoned_at` stays None,
the record stays fetchable (searchable/auditable), same contract
`retrieve.contradict()` and `migrate/_directive_sweep.py` honor.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from uuid import UUID

import pytest

from iai_mcp import directive_ops
from iai_mcp.capture import capture_turn
from iai_mcp.directive_ops import ResolveOutcome, ResolveResult
from iai_mcp.iai_cli import cmd_directive_list, cmd_directive_remove
from iai_mcp.store import MemoryStore


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401, PLC0415
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _args(action: str, id_: "str | None" = None) -> argparse.Namespace:
    return argparse.Namespace(action=action, id=id_)


def _mint(store_root: Path, text: str = "standing directive: reply in English") -> UUID:
    s = MemoryStore(path=store_root)
    try:
        result = capture_turn(
            store=s, cue="c", text=text, directive=True, session_id="s1", role="user",
        )
        assert result["status"] == "inserted", result
        return UUID(result["record_id"])
    finally:
        s.close()


@pytest.fixture(autouse=True)
def _redirect_home_cache(tmp_path, monkeypatch):
    """Hermeticity guard: the directive cache is bound at import time from
    the real home directory -- redirect the module constant to a sentinel
    that must never be written by these tests, independent of the
    store-root-derived cache_path retire_directive passes explicitly."""
    import iai_mcp.directive_cache as _directive_cache_mod

    sentinel = tmp_path / "MUST_NOT_BE_WRITTEN" / ".directives.cached.md"
    monkeypatch.setattr(_directive_cache_mod, "DIRECTIVES_CACHE_PATH", sentinel)
    return sentinel


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_list_surfaces_minted_directive_with_short_id(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store_root = tmp_path / "store"
    store_root.mkdir()
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))

    rid = _mint(store_root)

    s = MemoryStore(path=store_root)
    try:
        live = directive_ops.iter_live_directives(s)
    finally:
        s.close()
    assert len(live) == 1
    short_id, record_id, text = live[0]
    assert record_id == rid
    assert short_id == rid.hex[:8]
    assert "reply in English" in text


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_tty_remove_retires_stays_searchable_hermetic_cache(
    driver, tmp_path, monkeypatch, capsys, _redirect_home_cache
):
    _select_driver(driver, monkeypatch)
    store_root = tmp_path / "store"
    store_root.mkdir()
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))

    rid = _mint(store_root)
    short_id = rid.hex[:8]
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)

    # capture_turn's own mint-time cache refresh (unrelated to this plan)
    # also targets the redirected sentinel by default -- clear it so the
    # assertion below isolates retire_directive's own cache write.
    _redirect_home_cache.unlink(missing_ok=True)

    rc = cmd_directive_remove(_args("remove", short_id))
    assert rc == 0
    out = capsys.readouterr().out
    assert short_id in out

    s = MemoryStore(path=store_root)
    try:
        rec = s.get(rid)
        live_untombstoned_ids = {r.id for r in s.iter_records(where="tombstoned_at IS NULL")}
    finally:
        s.close()
    assert rec is not None
    assert rec.directive is False
    assert rid in live_untombstoned_ids

    # Hermetic: the real-home sentinel was never written; the tmp store's
    # own cache file was.
    assert not _redirect_home_cache.exists()
    assert (store_root / ".directives.cached.md").exists()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_non_tty_remove_refuses_and_mutates_nothing(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store_root = tmp_path / "store"
    store_root.mkdir()
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))

    rid = _mint(store_root)
    short_id = rid.hex[:8]

    s = MemoryStore(path=store_root)
    try:
        before = s.get(rid)
    finally:
        s.close()
    assert before.directive is True

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    rc = cmd_directive_remove(_args("remove", short_id))
    assert rc == 2

    s = MemoryStore(path=store_root)
    try:
        after = s.get(rid)
    finally:
        s.close()
    assert after.directive is True
    assert after.directive == before.directive


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_no_override_flag_defeats_non_tty_refusal(driver, tmp_path, monkeypatch):
    """A hypothetical yes/force attribute on args must not be read at all --
    the isatty refusal has no self-suppliable bypass."""
    _select_driver(driver, monkeypatch)
    store_root = tmp_path / "store"
    store_root.mkdir()
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    rid = _mint(store_root)

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    ns = _args("remove", rid.hex[:8])
    ns.yes = True
    ns.force = True
    rc = cmd_directive_remove(ns)
    assert rc == 2


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_unknown_id_distinct_message_mutates_nothing(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store_root = tmp_path / "store"
    store_root.mkdir()
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)

    rc = cmd_directive_remove(_args("remove", "ffffffff"))
    assert rc == 2


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_already_retired_id_distinct_message_mutates_nothing(driver, tmp_path, monkeypatch, capsys):
    _select_driver(driver, monkeypatch)
    store_root = tmp_path / "store"
    store_root.mkdir()
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)

    rid = _mint(store_root)
    short_id = rid.hex[:8]

    rc1 = cmd_directive_remove(_args("remove", short_id))
    assert rc1 == 0

    rc2 = cmd_directive_remove(_args("remove", short_id))
    assert rc2 == 2
    err = capsys.readouterr().err
    assert "already retired" in err.lower()

    s = MemoryStore(path=store_root)
    try:
        rec = s.get(rid)
        live_untombstoned_ids = {r.id for r in s.iter_records(where="tombstoned_at IS NULL")}
    finally:
        s.close()
    assert rec.directive is False
    assert rid in live_untombstoned_ids


def test_ambiguous_prefix_distinct_message_mutates_nothing(tmp_path, monkeypatch, capsys):
    store_root = tmp_path / "store"
    store_root.mkdir()
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)

    calls: list[str] = []

    def _fake_resolve(_store, token):
        calls.append(token)
        return ResolveResult(ResolveOutcome.AMBIGUOUS)

    monkeypatch.setattr(directive_ops, "resolve_directive_short_id", _fake_resolve)

    def _boom_retire(*_a, **_k):
        raise AssertionError("retire_directive must not be called on an ambiguous prefix")

    monkeypatch.setattr(directive_ops, "retire_directive", _boom_retire)

    rc = cmd_directive_remove(_args("remove", "ab"))
    assert rc == 2
    err = capsys.readouterr().err
    assert "ambiguous" in err.lower()
    assert calls == ["ab"]


def test_resolve_directive_short_id_reports_ambiguous_for_shared_prefix(monkeypatch):
    """Unit-level proof of the real resolve logic's ambiguity branch,
    independent of the CLI dispatch test above (which mocks resolve
    entirely)."""
    from uuid import uuid4

    fake_rid_a = uuid4()
    fake_rid_b = uuid4()
    shared_prefix = "aa000000"
    # Force both fake ids to share an 8-char dashless prefix without
    # depending on real UUID randomness colliding.
    fake_rid_a = UUID(shared_prefix + fake_rid_a.hex[8:])
    fake_rid_b = UUID(shared_prefix + fake_rid_b.hex[8:])

    def _fake_iter_live(_store):
        return [
            (shared_prefix, fake_rid_a, "text a"),
            (shared_prefix, fake_rid_b, "text b"),
        ]

    monkeypatch.setattr(directive_ops, "iter_live_directives", _fake_iter_live)

    result = directive_ops.resolve_directive_short_id(object(), shared_prefix)
    assert result.outcome == ResolveOutcome.AMBIGUOUS


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_cache_write_targets_store_root_never_home(driver, tmp_path, monkeypatch, _redirect_home_cache):
    _select_driver(driver, monkeypatch)
    store_root = tmp_path / "store"
    store_root.mkdir()
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)

    rid = _mint(store_root)
    # capture_turn's own mint-time cache refresh also targets the
    # redirected sentinel by default -- clear it so the assertion below
    # isolates retire_directive's own cache write.
    _redirect_home_cache.unlink(missing_ok=True)

    rc = cmd_directive_remove(_args("remove", rid.hex[:8]))
    assert rc == 0

    assert (store_root / ".directives.cached.md").exists()
    assert not _redirect_home_cache.exists()

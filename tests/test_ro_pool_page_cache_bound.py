"""The pooled read-only readers' page caches are bounded.

``HippoDB.__init__`` bounds its writer connection with ``PRAGMA cache_size``,
but the read-only pool does not go through ``HippoDB``: it opens its slots via
``get_lilli_raw_conn()``, so every slot was constructed with the pager's
``_max_cached_pages`` unset — i.e. unbounded. On a full-store scan that is not a
bounded-budget difference but a retention one: the pager keeps every clean page
it read, for the life of the daemon.

What each test here can and cannot see: the production engine is the NATIVE one,
which holds its page cache in Rust and exposes no pager to Python. So the slot
path is tested by asserting the PRAGMA the pool actually issues (deterministic,
and it is the whole of this change), and the retention contract is tested
against the reference pager, where the cache IS observable:

    4,000 pages written then scanned, 8 KiB pages
    unbounded   resident 4,003 pages   32.8 MB retained
    16 MiB      resident 2,048 pages   16.8 MB retained   <- exactly the budget

``IAI_MCP_RO_POOL_CACHE_KIB`` overrides the budget; ``0`` restores unbounded
behaviour. These tests exercise the engine (lilli) driver only — under the
stdlib driver the pool is not used at all.
"""

from __future__ import annotations

import os

import pytest

from iai_mcp.hippo._db import DEFAULT_STORAGE_DRIVER
from iai_mcp.hippo._ro_pool import (
    RO_POOL_CACHE_KIB_DEFAULT,
    _bound_slot_page_cache,
    _slot_cache_kib,
)

_LILLI = os.environ.get("LILLI_STORAGE_DRIVER", DEFAULT_STORAGE_DRIVER) == "lilli"

pytestmark = pytest.mark.skipif(
    not _LILLI,
    reason="RoConnPool is lilli-only; skipped under stdlib",
)


class _RecordingConn:
    """Minimal stand-in that records execute() calls and can be made to fail."""

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[str] = []
        self._fail = fail

    def execute(self, sql, *args, **kwargs):
        self.calls.append(str(sql))
        if self._fail:
            raise RuntimeError("engine rejected PRAGMA")


def _pragma_calls(conn: _RecordingConn) -> list[str]:
    return [c for c in conn.calls if "cache_size" in c.lower()]


def test_bound_issues_the_cache_size_pragma(monkeypatch) -> None:
    """The bound is applied by issuing the engine's own PRAGMA cache_size."""
    monkeypatch.delenv("IAI_MCP_RO_POOL_CACHE_KIB", raising=False)
    conn = _RecordingConn()

    _bound_slot_page_cache(conn)

    assert _pragma_calls(conn) == [f"PRAGMA cache_size=-{RO_POOL_CACHE_KIB_DEFAULT}"], (
        "the slot must be bounded with a negative-KiB cache_size PRAGMA"
    )


def test_bound_respects_an_override(monkeypatch) -> None:
    monkeypatch.setenv("IAI_MCP_RO_POOL_CACHE_KIB", "4096")
    conn = _RecordingConn()

    _bound_slot_page_cache(conn)

    assert _pragma_calls(conn) == ["PRAGMA cache_size=-4096"]


def test_bound_is_a_no_op_when_disabled(monkeypatch) -> None:
    """``0`` restores upstream behaviour — no PRAGMA is issued at all."""
    monkeypatch.setenv("IAI_MCP_RO_POOL_CACHE_KIB", "0")
    conn = _RecordingConn()

    _bound_slot_page_cache(conn)

    assert _pragma_calls(conn) == []


def test_a_failed_bound_never_prevents_the_slot_opening(monkeypatch) -> None:
    """The bound is advisory: an engine rejection must not raise out of it."""
    monkeypatch.delenv("IAI_MCP_RO_POOL_CACHE_KIB", raising=False)
    conn = _RecordingConn(fail=True)

    _bound_slot_page_cache(conn)  # must not raise

    assert _pragma_calls(conn) == [f"PRAGMA cache_size=-{RO_POOL_CACHE_KIB_DEFAULT}"]


def test_slot_cache_kib_default_override_and_malformed(monkeypatch) -> None:
    """Knob parsing: default, override, disabled, and a malformed value."""
    monkeypatch.delenv("IAI_MCP_RO_POOL_CACHE_KIB", raising=False)
    assert _slot_cache_kib() == RO_POOL_CACHE_KIB_DEFAULT

    monkeypatch.setenv("IAI_MCP_RO_POOL_CACHE_KIB", "4096")
    assert _slot_cache_kib() == 4096

    # 0 (and negatives) mean "unbounded".
    monkeypatch.setenv("IAI_MCP_RO_POOL_CACHE_KIB", "0")
    assert _slot_cache_kib() == 0

    # A malformed value falls back to the default, never crashing the pool.
    monkeypatch.setenv("IAI_MCP_RO_POOL_CACHE_KIB", "not-an-int")
    assert _slot_cache_kib() == RO_POOL_CACHE_KIB_DEFAULT


def test_pool_open_slot_applies_the_bound_end_to_end(tmp_path, monkeypatch) -> None:
    """The pool's OWN open path issues the bound, not just the helper.

    Guards the call site: the helper could be correct while never being wired
    into ``_open_slot``, which is the state this change fixes.
    """
    from iai_mcp.store import MemoryStore

    monkeypatch.setenv("IAI_MCP_RO_POOL_CACHE_KIB", "2048")

    store = MemoryStore(path=tmp_path)
    try:
        store.db._conn.execute("SELECT COUNT(*) FROM records").fetchone()
    except Exception as exc:  # noqa: BLE001 -- engine unavailable in this env
        pytest.skip(f"lilli engine unavailable: {exc}")

    pool = getattr(store.db, "_ro_pool", None)
    if pool is None:
        pytest.skip("RO pool not constructed for this driver")

    real_open = pool._open_slot

    def _spy_open_slot():
        conn = real_open()
        # Record what the bound did to the connection it was handed.
        bound_calls.append(conn is not None)
        return conn

    bound_calls: list[bool] = []
    monkeypatch.setattr(pool, "_open_slot", _spy_open_slot)

    # Drive the real slot path and capture the PRAGMA it issues.
    issued: list[str] = []
    from iai_mcp.hippo import _ro_pool as _mod

    real_bound = _mod._bound_slot_page_cache

    def _spy_bound(conn):
        original_execute = conn.execute

        def _record(sql, *args, **kwargs):
            if "cache_size" in str(sql).lower():
                issued.append(str(sql))
            return original_execute(sql, *args, **kwargs)

        conn.execute = _record
        return real_bound(conn)

    monkeypatch.setattr(_mod, "_bound_slot_page_cache", _spy_bound)

    with pool.borrow():
        pass

    assert issued == ["PRAGMA cache_size=-2048"], (
        f"the pool's open path must bound each slot; saw {issued!r}"
    )


def test_scan_retention_is_capped_at_the_budget(tmp_path) -> None:
    """The bound caps retained pages across a full-store scan.

    The mechanism check: an unbounded pager retains one entry per page scanned;
    a bounded one stays at/under its budget. Exercised against the reference
    pager, which is where that retention lives and is observable.
    """
    from iai_mcp.lillibrain.constants import PAGE_SIZE
    from iai_mcp.lillibrain.pager import Pager

    def _scan_resident_pages(bound_kib: int) -> int:
        budget = None if bound_kib <= 0 else max(1, (bound_kib * 1024) // PAGE_SIZE)
        pager = Pager(tmp_path / f"scan_{bound_kib}.lilli", max_cached_pages=budget)
        try:
            pager.enable_wal_mode()
            n_pages = 1200
            pager.begin_write()
            for pn in range(4, 4 + n_pages):
                while pager.read_db_header_field("db_size") < pn:
                    pager.extend_file()
                pager.write_page(pn, bytes([pn % 251]) * PAGE_SIZE)
            pager.commit()

            for pn in range(4, 4 + n_pages):  # the full-store scan
                pager.read_page(pn)

            return len(pager.page_cache)
        finally:
            pager.close()

    budget_kib = 4096  # 4 MiB -> 512 pages at 8 KiB
    budget_pages = max(1, (budget_kib * 1024) // PAGE_SIZE)

    unbounded = _scan_resident_pages(0)
    bounded = _scan_resident_pages(budget_kib)

    assert unbounded >= 1200, f"unbounded pager retained only {unbounded} pages"
    assert bounded <= budget_pages, (
        f"bounded pager retained {bounded} pages, budget is {budget_pages}"
    )
    assert bounded < unbounded

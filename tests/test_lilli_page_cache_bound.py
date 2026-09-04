"""Engine unit: the lilli pager's page cache is bounded and crash-safe.

These tests poke the pager directly, so they exercise the lilli engine only;
they are skipped on the stdlib driver (which opens a native sqlite3 connection
and never runs any lillibrain code). The contract under test:

1. Under a configured budget, reading more clean pages than the budget evicts
   the least-recently-used clean pages so the cache stays at/under the bound.
2. A dirty (uncommitted) page is NEVER evicted, even when the cache is far over
   budget — its bytes survive and commit intact.
3. A rollback-restored before-image is NEVER evicted — it stays authoritative
   through a subsequent large scan.
4. Read correctness holds: a page evicted by the bound returns identical bytes
   when re-read (re-materialised from the WAL / main file).
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import pytest

from iai_mcp.hippo._db import DEFAULT_STORAGE_DRIVER
from iai_mcp.lillibrain.constants import PAGE_SIZE
from iai_mcp.lillibrain.pager import Pager

_LILLI = os.environ.get("LILLI_STORAGE_DRIVER", DEFAULT_STORAGE_DRIVER) == "lilli"

pytestmark = pytest.mark.skipif(
    not _LILLI,
    reason="engine-internal pager test runs only under LILLI_STORAGE_DRIVER=lilli",
)


def _page_payload(marker: int) -> bytes:
    """A full page whose first 4 bytes encode a recognisable marker."""
    buf = bytearray(PAGE_SIZE)
    struct.pack_into(">I", buf, 0, marker)
    return bytes(buf)


def _open_wal_pager(tmp_path: Path, *, budget: int | None) -> Pager:
    pager = Pager(tmp_path / "cache.lilli", max_cached_pages=budget)
    pager.enable_wal_mode()
    return pager


def _write_committed_pages(pager: Pager, page_nos: list[int]) -> dict[int, bytes]:
    """Allocate+write each page in one committed WAL transaction; return contents."""
    contents: dict[int, bytes] = {}
    pager.begin_write()  # no-op in WAL mode
    for pn in page_nos:
        # Ensure the page exists (extend the file up to pn).
        while pager.read_db_header_field("db_size") < pn:
            pager.extend_file()
        payload = _page_payload(pn)
        pager.write_page(pn, payload)
        contents[pn] = payload
    pager.commit()
    return contents


def test_wal_clean_pages_evicted_at_budget(tmp_path: Path) -> None:
    """Reading more clean pages than the budget keeps the cache at/under bound."""
    budget = 4
    pager = _open_wal_pager(tmp_path, budget=budget)
    try:
        page_nos = list(range(4, 16))  # 12 data pages, well above the 4-page budget
        contents = _write_committed_pages(pager, page_nos)

        # Read every page back; each enters the cache via the WAL-read path.
        for pn in page_nos:
            pager.read_page(pn)

        assert len(pager.page_cache) <= budget, (
            f"cache holds {len(pager.page_cache)} pages, budget is {budget}"
        )

        # Read correctness: an evicted page re-reads to its original bytes.
        for pn in page_nos:
            got = bytes(pager.read_page(pn))
            assert got[:4] == contents[pn][:4], f"page {pn} mis-read after eviction"
    finally:
        pager.close()


def test_wal_dirty_page_never_evicted(tmp_path: Path) -> None:
    """A dirty page survives an over-budget scan and commits intact."""
    budget = 4
    pager = _open_wal_pager(tmp_path, budget=budget)
    try:
        page_nos = list(range(4, 20))
        _write_committed_pages(pager, page_nos)

        # Open a transaction and dirty one page with a distinctive marker.
        dirty_pn = 5
        dirty_marker = 0xDEADBEEF
        pager.begin_write()
        pager.write_page(dirty_pn, _page_payload(dirty_marker))
        assert dirty_pn in pager.dirty

        # Read many OTHER pages to push the cache far over budget.
        for pn in page_nos:
            if pn != dirty_pn:
                pager.read_page(pn)

        # The dirty page must still be resident with its uncommitted bytes.
        assert dirty_pn in pager.page_cache, "dirty page was evicted (data loss)"
        assert struct.unpack_from(">I", pager.page_cache[dirty_pn], 0)[0] == dirty_marker

        pager.commit()

        # After commit the bytes are durable and re-read correctly.
        got = struct.unpack_from(">I", bytes(pager.read_page(dirty_pn)), 0)[0]
        assert got == dirty_marker, "committed dirty page lost its write"
    finally:
        pager.close()


def test_journal_rollback_restored_page_never_evicted(tmp_path: Path) -> None:
    """A rollback-restored before-image survives a subsequent large scan.

    Exercised in rollback-journal mode (the only mode that restores an
    authoritative before-image into the cache); a tiny budget forces eviction
    pressure during the scan that follows the rollback.
    """
    budget = 4
    pager = Pager(tmp_path / "rollback.lilli", max_cached_pages=budget)
    try:
        # Seed several pages with committed content.
        page_nos = list(range(4, 20))
        original: dict[int, bytes] = {}
        pager.begin_write()
        for pn in page_nos:
            while pager.read_db_header_field("db_size") < pn:
                pager.extend_file()
            payload = _page_payload(pn)
            pager.write_page(pn, payload)
            original[pn] = payload
        pager.commit()

        # Start a new transaction, mutate one page, then ROLL BACK.
        restored_pn = 6
        pager.begin_write()
        pager.write_page(restored_pn, _page_payload(0xCAFEF00D))
        pager.rollback()

        # The restored page must be pinned and authoritative.
        assert restored_pn in pager.pinned, "rollback-restored page was not pinned"
        assert restored_pn in pager.page_cache, "rollback-restored page evicted"
        restored_marker = struct.unpack_from(
            ">I", pager.page_cache[restored_pn], 0
        )[0]
        assert restored_marker == restored_pn, "rollback did not restore before-image"

        # Scan all other pages to apply heavy eviction pressure.
        for pn in page_nos:
            if pn != restored_pn:
                pager.read_page(pn)

        # The restored page survived and still reads as its before-image.
        assert restored_pn in pager.page_cache, (
            "rollback-restored page evicted under budget pressure (data divergence)"
        )
        got = struct.unpack_from(">I", bytes(pager.read_page(restored_pn)), 0)[0]
        assert got == restored_pn, "rollback-restored page diverged after scan"
    finally:
        pager.close()


def test_pragma_cache_size_sets_pager_bound(tmp_path: Path) -> None:
    """PRAGMA cache_size on the engine connection actually bounds the pager."""
    from iai_mcp.lillibrain.connection import LilliBrainConnection, _open_lilli_db

    db_file = tmp_path / "pragma.lilli"
    pager, catalog, root_map = _open_lilli_db(db_file)
    conn = LilliBrainConnection(pager, catalog, root_map)
    try:
        # Negative N → KiB of memory → page count.
        conn.execute("PRAGMA cache_size=-16")  # 16 KiB
        expected = max(1, (16 * 1024) // PAGE_SIZE)
        assert pager._max_cached_pages == expected

        # Positive N → page count directly.
        conn.execute("PRAGMA cache_size=128")
        assert pager._max_cached_pages == 128

        # Zero → unbounded (no crash).
        conn.execute("PRAGMA cache_size=0")
        assert pager._max_cached_pages is None

        # Unparseable → unbounded (no crash), empty cursor shape preserved.
        cur = conn.execute("PRAGMA cache_size=notanint")
        assert pager._max_cached_pages is None
        assert list(cur.fetchall()) == []
    finally:
        pager.close()

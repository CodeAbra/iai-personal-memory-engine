"""Structural integrity verification for the LilliBrain B+tree.

Walks the entire tree and freelist, verifying structural invariants. Returns a
list of error strings (empty = clean). Designed to be called as a Hypothesis
@invariant after every mutation step, and as a maintenance diagnostic.

Invariant groups checked:
  1. Page accounting — no page number references a page beyond db_size
  2. No cycle or double-referenced page in the tree walk
  3. Strictly increasing keys within every node
  4. Interior child count — len(children) == len(keys) + 1
  5. All child page numbers within range
  6. All leaves at equal depth
  7. No orphaned pages (every allocated page is either reachable, an overflow
     page, or on the freelist)
  8. No page appears on both the freelist and the live tree
  9. Freelist header count matches the actual freelist set size
 10. Cross-page leaf key ordering forms a strictly ascending global sequence
 11. Leaf sibling chain order matches the tree-walk leaf order
 12. Leaf sibling links point to pages within the allocated range
 13. Overflow pages are not on the freelist and not in the B-tree node set
 14. Overflow chain terminates without cycles and all page numbers are in range
 15. Reassembled overflow payload byte count equals the declared total length
 16. No page appears in more than one role (B-tree node, overflow page, freelist)

The public signature and IntegrityError class are stable; callers may pass
raise_on_error=True to receive a single exception instead of a list.
"""
from __future__ import annotations

import struct

from iai_mcp.lillibrain.btree.page import (
    get_next_leaf,
    page_type,
    read_all_leaf_cells,
    read_interior_node,
)
from iai_mcp.lillibrain.constants import (
    OVERFLOW_DATA_OFFSET,
    OVERFLOW_NEXT_OFFSET,
    OVERFLOW_THRESHOLD,
    PAGE_INTERIOR_TABLE,
    PAGE_LEAF_TABLE,
)
from iai_mcp.lillibrain.freelist import FreelistCorruptionError, freelist_pages
from iai_mcp.lillibrain.pager import Pager

# ---------------------------------------------------------------------------
# Integrity error class
# ---------------------------------------------------------------------------


class IntegrityError(RuntimeError):
    """Raised by check_integrity() when the caller requests raise-on-error mode.

    Default mode returns a list[str]; pass raise_on_error=True to get
    this exception instead, useful in @invariant hooks.
    """


# ---------------------------------------------------------------------------
# Overflow chain walk helper
# ---------------------------------------------------------------------------


def _walk_overflow_chain(
    pager: Pager,
    first_page_no: int,
    total_len: int,
    overflow_pages: set[int],
    errors: list[str],
    db_size: int,
) -> None:
    """Walk an overflow chain starting at first_page_no and validate its structure.

    Collects every page number into overflow_pages, reporting:
      - cycles (a page number repeats in the chain)
      - out-of-range page numbers (< 1 or > db_size)
      - short chains (bytes collected fall short of declared total_len)
      - byte-count mismatch after reassembly (group 15)
    """
    collected = 0
    current = first_page_no
    seen: set[int] = set()

    while current != 0:
        if current in seen:
            errors.append(f"overflow chain cycle at page {current}")
            return
        if current < 1 or current > db_size:
            errors.append(
                f"overflow chain page {current} out of range (db_size={db_size})"
            )
            return
        seen.add(current)
        overflow_pages.add(current)

        page_data = pager.read_page(current)
        (next_pg,) = struct.unpack_from(">I", page_data, OVERFLOW_NEXT_OFFSET)
        chunk_len = len(page_data) - OVERFLOW_DATA_OFFSET  # PAGE_SIZE - 4
        collected += chunk_len
        current = next_pg

    # collected may exceed total_len on the last page due to zero-padding;
    # trimming is correct — only report if insufficient bytes were collected.
    if collected < total_len:
        errors.append(
            f"overflow chain starting at page {first_page_no}: "
            f"collected {collected} bytes < declared {total_len}"
        )
        return

    # Group 15: verify byte-exact reassembly via read_overflow_chain.
    try:
        from iai_mcp.lillibrain.btree.overflow import read_overflow_chain  # noqa: PLC0415
        reassembled = read_overflow_chain(pager, first_page_no, total_len)
        if len(reassembled) != total_len:
            errors.append(
                f"overflow chain starting at page {first_page_no}: "
                f"reassembled {len(reassembled)} bytes != declared {total_len}"
            )
    except Exception as exc:  # noqa: BLE001 — integrity walk must never raise
        errors.append(
            f"overflow chain starting at page {first_page_no}: "
            f"reassembly error: {exc}"
        )


# ---------------------------------------------------------------------------
# Internal tree-walk helpers
# ---------------------------------------------------------------------------


def _walk_tree(
    pager: Pager,
    page_no: int,
    depth: int,
    visited: set[int],
    leaf_order: list[int],
    errors: list[str],
    db_size: int,
    overflow_pages: set[int],
) -> int | None:
    """Recursively walk the tree rooted at page_no.

    Returns the leaf depth from this node (all leaves must share the same
    depth for invariant group 6). Returns None if a cycle or structural error
    prevented reaching leaves.

    Collected side effects:
      - visited: updated with each page encountered
      - leaf_order: list of leaf page numbers in tree-walk order (leftmost first)
      - errors: accumulated violation strings
    """
    # ---------------------------------------------------------------------------
    # Invariant group 1: page-number bounds
    # ---------------------------------------------------------------------------
    if page_no < 1 or page_no > db_size:
        errors.append(
            f"page {page_no}: page number out of allocated range (db_size={db_size})"
        )
        return None

    # ---------------------------------------------------------------------------
    # Invariant group 2: no cycle or double-referenced page
    # ---------------------------------------------------------------------------
    if page_no in visited:
        errors.append(
            f"page {page_no}: referenced more than once (cycle or double-reference)"
        )
        return None
    visited.add(page_no)

    page_data = pager.read_page(page_no)
    mv = memoryview(page_data)

    try:
        ptype = page_type(mv)
    except ValueError as exc:
        errors.append(f"page {page_no}: {exc}")
        return None

    if ptype == PAGE_LEAF_TABLE:
        # ---------------------------------------------------------------------------
        # Invariant group 3 (leaf): strictly increasing keys
        # ---------------------------------------------------------------------------
        # Read without pager: keys are always inline; for spilled cells the
        # raw 4-byte overflow pointer is returned as payload — sufficient for
        # key-order checks and overflow pointer extraction.
        cells = read_all_leaf_cells(mv)
        keys = [c[0] for c in cells]
        for i in range(1, len(keys)):
            if keys[i] <= keys[i - 1]:
                errors.append(
                    f"page {page_no}: leaf keys not strictly in order at position {i} "
                    f"(keys[{i-1}]={keys[i-1]}, keys[{i}]={keys[i]})"
                )

        # ---------------------------------------------------------------------------
        # Groups 13-15: discover and validate overflow chains for spilled cells
        # ---------------------------------------------------------------------------
        from iai_mcp.lillibrain.btree.page import _LEAF_PTR_ARRAY_START  # noqa: PLC0415
        from iai_mcp.lillibrain.record import decode_varint  # noqa: PLC0415

        for ci in range(len(cells)):
            (cell_ptr,) = struct.unpack_from(">H", mv, _LEAF_PTR_ARRAY_START + ci * 2)
            _k, kn = decode_varint(mv, cell_ptr)
            payload_len, vn = decode_varint(mv, cell_ptr + kn)
            if payload_len > OVERFLOW_THRESHOLD:
                ovf_ptr_offset = cell_ptr + kn + vn
                (first_ovf,) = struct.unpack_from(">I", mv, ovf_ptr_offset)
                _walk_overflow_chain(
                    pager, first_ovf, payload_len, overflow_pages, errors, db_size
                )

        leaf_order.append(page_no)
        return depth

    # Interior node
    assert ptype == PAGE_INTERIOR_TABLE
    node_keys, children = read_interior_node(mv)

    # ---------------------------------------------------------------------------
    # Invariant group 3 (interior): strictly increasing keys
    # ---------------------------------------------------------------------------
    for i in range(1, len(node_keys)):
        if node_keys[i] <= node_keys[i - 1]:
            errors.append(
                f"page {page_no}: interior keys not strictly in order at position {i} "
                f"(keys[{i-1}]={node_keys[i-1]}, keys[{i}]={node_keys[i]})"
            )

    # ---------------------------------------------------------------------------
    # Invariant group 4: child count == len(keys) + 1
    # ---------------------------------------------------------------------------
    if len(children) != len(node_keys) + 1:
        errors.append(
            f"page {page_no}: interior child count {len(children)} != len(keys)+1 "
            f"({len(node_keys)+1})"
        )

    # ---------------------------------------------------------------------------
    # Invariant group 5: all child page numbers in range
    # ---------------------------------------------------------------------------
    for child_no in children:
        if child_no < 1 or child_no > db_size:
            errors.append(
                f"page {page_no}: child page {child_no} out of allocated range "
                f"(db_size={db_size})"
            )

    # Recurse into children.
    leaf_depths: list[int] = []
    for child_no in children:
        d = _walk_tree(pager, child_no, depth + 1, visited, leaf_order, errors, db_size, overflow_pages)
        if d is not None:
            leaf_depths.append(d)

    if not leaf_depths:
        return None

    # ---------------------------------------------------------------------------
    # Invariant group 6: all leaves at equal depth (checked incrementally)
    # ---------------------------------------------------------------------------
    first_depth = leaf_depths[0]
    for ld in leaf_depths[1:]:
        if ld != first_depth:
            errors.append(
                f"page {page_no}: subtrees have leaves at different depths "
                f"({first_depth} vs {ld})"
            )
    return first_depth


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def check_integrity(pager: Pager, *, raise_on_error: bool = False) -> list[str]:
    """Verify structural invariants of the B+tree stored in pager.

    Returns a (possibly empty) list of error description strings.
    If raise_on_error is True, raises IntegrityError on the first violation
    instead of collecting all errors.
    """
    errors: list[str] = []

    # ---------------------------------------------------------------------------
    # Test-injection hook — allows tests to force error conditions on the stub
    # ---------------------------------------------------------------------------
    injected: list[str] = getattr(pager, "_injected_errors_for_test", [])
    errors.extend(injected)

    # ---------------------------------------------------------------------------
    # Invariant group 1: page accounting — root page must be in range
    # ---------------------------------------------------------------------------
    db_size = pager.read_db_header_field("db_size")

    if db_size < 1:
        errors.append(f"db_size={db_size}: invalid (must be >= 1)")
        if raise_on_error and errors:
            raise IntegrityError("; ".join(errors))
        return errors

    if pager.root_page_no < 1 or pager.root_page_no > db_size:
        errors.append(
            f"page {pager.root_page_no}: root page number out of allocated range "
            f"(db_size={db_size})"
        )
        if raise_on_error and errors:
            raise IntegrityError("; ".join(errors))
        return errors

    # ---------------------------------------------------------------------------
    # Perform the tree walk — groups 1-6, 13-15 (overflow chains discovered here)
    # ---------------------------------------------------------------------------
    visited: set[int] = set()
    leaf_order: list[int] = []
    overflow_pages: set[int] = set()

    _walk_tree(pager, pager.root_page_no, 0, visited, leaf_order, errors, db_size, overflow_pages)

    # ---------------------------------------------------------------------------
    # Invariant group 8: freelist disjoint from live B-tree pages
    # ---------------------------------------------------------------------------
    try:
        free_set = freelist_pages(pager)
    except FreelistCorruptionError as exc:
        errors.append(f"freelist corruption: {exc}")
        free_set = set()

    overlap = visited & free_set
    if overlap:
        errors.append(
            f"pages on both the live tree and the freelist: {sorted(overlap)}"
        )

    # ---------------------------------------------------------------------------
    # Invariant group 7: no orphaned pages
    # ---------------------------------------------------------------------------
    # Page 1 is the DB header page — never part of the tree or freelist; always
    # a valid resident. Start the orphan scan from page 2.
    # Overflow pages are a third reachable set; a page reachable as an overflow
    # page is not orphaned.
    for page_no in range(2, db_size + 1):
        if page_no not in visited and page_no not in free_set and page_no not in overflow_pages:
            errors.append(
                f"page {page_no}: allocated but not reachable from the tree and "
                f"not on the freelist (orphaned page)"
            )

    # ---------------------------------------------------------------------------
    # Invariant group 9: freelist count matches DB header
    # ---------------------------------------------------------------------------
    header_freelist_count = pager.read_db_header_field("freelist_count")
    if header_freelist_count != len(free_set):
        errors.append(
            f"freelist count mismatch: header says {header_freelist_count}, "
            f"actual freelist has {len(free_set)} pages"
        )

    # ---------------------------------------------------------------------------
    # Invariant group 10: cross-page leaf key ordering forms one ascending sequence
    # ---------------------------------------------------------------------------
    all_leaf_keys: list[int] = []
    for leaf_no in leaf_order:
        page_data = pager.read_page(leaf_no)
        mv = memoryview(page_data)
        cells = read_all_leaf_cells(mv)
        all_leaf_keys.extend(c[0] for c in cells)

    for i in range(1, len(all_leaf_keys)):
        if all_leaf_keys[i] <= all_leaf_keys[i - 1]:
            errors.append(
                f"cross-page key ordering violation: global key sequence not "
                f"strictly ascending at position {i} "
                f"(key[{i-1}]={all_leaf_keys[i-1]}, key[{i}]={all_leaf_keys[i]})"
            )
            break  # one report is enough; avoids flooding

    # ---------------------------------------------------------------------------
    # Invariant groups 11 & 12: sibling chain order and bounds
    # ---------------------------------------------------------------------------
    if leaf_order:
        # Walk the sibling chain starting from the leftmost leaf.
        sibling_chain: list[int] = []
        current = leaf_order[0]
        seen_siblings: set[int] = set()

        while current != 0:
            # Range check for sibling target
            if current > db_size or current < 1:
                errors.append(
                    f"leaf sibling link points to out-of-range page {current} "
                    f"(db_size={db_size})"
                )
                break
            if current in seen_siblings:
                errors.append(
                    f"leaf sibling chain cycle: page {current} visited twice"
                )
                break
            seen_siblings.add(current)
            sibling_chain.append(current)
            next_no = get_next_leaf(pager, current)
            current = next_no

        # Sibling chain order must match tree-walk leaf order
        if sibling_chain != leaf_order:
            # Only report if there is an actual mismatch — could be a subset
            # if the chain terminated early (already reported above).
            if len(errors) == 0 or not any("sibling" in e for e in errors):
                errors.append(
                    f"leaf sibling chain order does not match tree-walk order: "
                    f"sibling={sibling_chain}, tree-walk={leaf_order}"
                )

    # ---------------------------------------------------------------------------
    # Invariant group 13: overflow pages are not on the freelist and not in the
    # B-tree node set (already collected by _walk_overflow_chain; verify here)
    # ---------------------------------------------------------------------------
    cross_tree_ovf = visited & overflow_pages
    if cross_tree_ovf:
        errors.append(
            f"pages are both B-tree nodes and overflow pages: {sorted(cross_tree_ovf)}"
        )

    cross_free_ovf = free_set & overflow_pages
    if cross_free_ovf:
        errors.append(
            f"pages are both on the freelist and overflow pages: {sorted(cross_free_ovf)}"
        )

    # ---------------------------------------------------------------------------
    # Invariant group 16: no page appears in more than one role
    # (Reported above via cross_tree_ovf and cross_free_ovf; the tree/freelist
    # cross is already reported in group 8.  All three roles checked.)
    # ---------------------------------------------------------------------------

    if raise_on_error and errors:
        raise IntegrityError("; ".join(errors))

    return errors

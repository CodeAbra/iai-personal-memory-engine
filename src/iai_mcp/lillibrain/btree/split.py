"""B+tree split code: copy-up leaf split and push-up interior split.

Two completely separate public functions enforce distinct semantics:

  leaf_split_and_insert  — COPY-UP:
    The separator is the first key of the right leaf and stays there. After the
    split both children together hold all N+1 cells; the separator appears in the
    right child exactly once. The parent receives a copy of the separator key.

  interior_split_and_insert — PUSH-UP:
    The median key is extracted and sent to the parent. It is absent from both
    resulting children. After the split the parent gains one key and the total
    key count is conserved: len(left_keys) + 1 (push_up) + len(right_keys) == N.

The two functions are intentionally kept separate; a shared function with a
node-type flag is an anti-pattern because it makes the copy-up/push-up
distinction easy to misconfigure silently.

Root split discipline:
  The root page number (pager.root_page_no) is FIXED for the lifetime of the
  database. When the root overflows, _root_from_children reinitialises the fixed
  root page in place: the old root content moves to a freshly allocated child
  page and the fixed root page becomes an interior node over the N children.

Inline invariants (always active, not test-only):
  Leaf:     separator_key == right_cells[0][0]
            separator_key in [c[0] for c in right_cells]
  Interior: push_up_key not in left_keys and push_up_key not in right_keys
            len(left_keys) + 1 + len(right_keys) == len(all_keys)
"""
from __future__ import annotations

import bisect
import logging
import struct
from typing import TYPE_CHECKING

from iai_mcp.lillibrain.btree.page import (
    get_next_leaf,
    read_all_leaf_cells,
    read_interior_node,
    set_next_leaf,
    write_interior_node,
    write_leaf_cells,
)
from iai_mcp.lillibrain.constants import INTERIOR_HEADER_SIZE, PAGE_SIZE
from iai_mcp.lillibrain.freelist import alloc_page

if TYPE_CHECKING:
    from iai_mcp.lillibrain.pager import Journal, Pager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helper: insert a separator into an interior node or split it
# ---------------------------------------------------------------------------


def _interior_fits(keys: list[int]) -> bool:
    """Return True when an interior node with these keys fits one page by bytes.

    Interior layout: INTERIOR_HEADER_SIZE header + n*2 pointer array
    + sum(4 + varint(key)) cell bytes (the rightmost child lives in the header,
    so it is not charged a cell). Matches _interior_has_room_for's arithmetic.
    """
    from iai_mcp.lillibrain.record import encode_varint  # noqa: PLC0415

    n = len(keys)
    ptr_area = INTERIOR_HEADER_SIZE + n * 2
    cell_bytes = sum(4 + len(encode_varint(k)) for k in keys)
    return ptr_area + cell_bytes <= PAGE_SIZE


def _insert_run_into_interior(
    pager: Pager,
    journal: Journal,
    path_stack: list[tuple[int, int]],
    parent_page_no: int,
    old_child_page_no: int,
    new_child_page_nos: list[int],
    separators: list[int],
) -> None:
    """Splice a run of (separator, new_child) pairs into an interior node.

    The old child pointer stays at its existing slot; each new child follows it
    in key order, every new child preceded by its separator. When the resulting
    node fits one page it is rewritten in place; otherwise it is N-way split and
    the overflow propagates upward (consuming path_stack).

    The splice index is located by the OLD CHILD'S PAGE NUMBER, not by bisecting
    a separator — all run separators fall in the same parent gap, so a
    per-separator bisect would be ambiguous.
    """
    from iai_mcp.lillibrain.constants import LEAF_MAX_CELLS  # noqa: PLC0415

    page_data = pager.read_page(parent_page_no)
    mv = memoryview(page_data)
    old_keys, old_children = read_interior_node(mv)

    # Locate the old child by page number; the run goes immediately after it.
    j = old_children.index(old_child_page_no)
    new_keys = old_keys[:j] + list(separators) + old_keys[j:]
    new_children = (
        old_children[: j + 1] + list(new_child_page_nos) + old_children[j + 1 :]
    )

    # Byte capacity governs the split in production; LEAF_MAX_CELLS is a secondary
    # ceiling tests inject so the interior split path fires on a small key count.
    if len(new_keys) <= LEAF_MAX_CELLS and _interior_fits(new_keys):
        new_page = bytearray(PAGE_SIZE)
        write_interior_node(new_page, new_keys, new_children)
        pager.write_page(parent_page_no, new_page)
        logger.debug(
            "_insert_run_into_interior: parent=%d keys=%s children=%s",
            parent_page_no, new_keys, new_children,
        )
    else:
        # Parent interior node overflows — capture its before-image explicitly,
        # then N-way split it directly from the in-memory overflow state. Writing
        # the overflow content to the page cache is intentionally avoided: with a
        # large interior capacity the overflow write would push the pointer array
        # into the cell-content area, corrupting the page before the split can
        # repair it. The before-image lets crash recovery restore the pre-split
        # content.
        if pager._journal is not None and parent_page_no not in pager._journal._journaled:
            pager._journal.record_before_image(
                parent_page_no, pager.read_page(parent_page_no)
            )
        interior_split_and_insert(
            pager, journal, path_stack, parent_page_no, new_keys, new_children
        )


# ---------------------------------------------------------------------------
# Public function 1: Leaf split — COPY-UP
# ---------------------------------------------------------------------------


def leaf_split_and_insert(
    pager: Pager,
    journal: Journal,
    path_stack: list[tuple[int, int]],
    leaf_page_no: int,
    new_key: int,
    new_value: bytes,
) -> None:
    """Split a full leaf and insert (new_key, new_value), using copy-up semantics.

    The new (key, value) is merged into the sorted cell list (N+1 cells), then
    partitioned by bytes into N>=2 groups so every group fits one page. Each
    separator is the first key of a right-hand group and stays in that group
    (copy-up); a copy of each separator is promoted to the parent.

    Allocation: group 0 stays on the original leaf page; one new leaf per
    right-hand group is allocated from the freelist (trunk and header pages
    journaled before any freelist mutation). The sibling chain is rethreaded:
    group0 → new1 → ... → new(N-1) → old_right_sibling.

    After the split, per boundary:
      assert separator == groups[i][0][0]
      assert separator in [c[0] for c in groups[i]]

    The run of (separator, new_child) pairs is inserted into the parent via
    _insert_run_into_interior, which recurses upward as needed. If the path
    stack is empty (the leaf is also the root), _root_from_children promotes the
    single-leaf root into a two-level tree.
    """
    page_data = pager.read_page(leaf_page_no)
    mv = memoryview(page_data)
    # Pass pager so spilled cells are fully reassembled — the split code
    # hands (key, payload) tuples to write_leaf_cells which will re-spill
    # oversized payloads onto the new pages via the pager/journal pair.
    old_cells = read_all_leaf_cells(mv, pager=pager)

    # Free OLD overflow chains for all spilled cells on the original leaf before
    # rewriting it.  write_leaf_cells will allocate fresh chains for oversized
    # payloads when it writes both halves; without freeing the originals first,
    # the old chains become orphaned pages not reachable from the tree or freelist.
    from iai_mcp.lillibrain.btree.delete import _free_spilled_cell_chains  # noqa: PLC0415
    n_old = len(old_cells)
    if n_old:
        _free_spilled_cell_chains(pager, journal, mv, list(range(n_old)))

    # Merge new cell into sorted position (N+1 cells total).
    idx = bisect.bisect_left([c[0] for c in old_cells], new_key)
    all_cells = old_cells[:idx] + [(new_key, new_value)] + old_cells[idx:]

    # Byte-balanced N-way partition: distribute the sorted cells into N>=2 groups
    # so every group fits one page. A count-midpoint split can co-locate two
    # large cells on one half and overflow it, and an unequal set can have no
    # valid two-way cut at all — the byte partition always succeeds because every
    # cell fits a page alone (inline <= OVERFLOW_THRESHOLD, spilled costs only a
    # 4-byte on-page pointer).
    from iai_mcp.lillibrain.btree.delete import _balanced_nway_leaf_split  # noqa: PLC0415
    groups = _balanced_nway_leaf_split(all_cells)

    # Single-group result = no split needed. The byte partition fits every cell on
    # one page, so there is nothing to divide: rewrite the merged cells back onto
    # the original leaf in place, preserving its sibling link, and return without
    # allocating pages or touching the parent. This keeps a split that yields one
    # group a no-op instead of building a degenerate single-child root — the split
    # dispatch is gated by a secondary cell-count ceiling that can fire on a
    # byte-fitting leaf, so the splitter must tolerate a no-op partition.
    if len(groups) == 1:
        old_right_sibling = struct.unpack_from(">I", page_data, 8)[0]
        new_leaf = bytearray(PAGE_SIZE)
        write_leaf_cells(new_leaf, all_cells, pager=pager, journal=journal)
        pager.write_page(leaf_page_no, new_leaf)
        set_next_leaf(pager, leaf_page_no, old_right_sibling)
        logger.debug(
            "leaf_split_and_insert: leaf=%d single-group no-op (byte-fitting)",
            leaf_page_no,
        )
        return

    # Group 0 stays on the original leaf page; allocate one fresh leaf per
    # right-hand group (N-1 new pages), journaled.
    new_page_nos = [alloc_page(pager, journal=journal) for _ in groups[1:]]
    group0_page_no = leaf_page_no
    group_page_nos = [group0_page_no] + new_page_nos

    # Read the original right sibling BEFORE rewriting so the chain tail is
    # preserved across all N pages.
    old_right_sibling = struct.unpack_from(">I", page_data, 8)[0]

    # Write every group to its page, re-spilling oversized payloads onto fresh
    # chains. Sibling links are set after writing so the chain rethreads as
    # group0 -> new1 -> ... -> new(N-1) -> old_right_sibling.
    for page_no, group in zip(group_page_nos, groups):
        new_leaf = bytearray(PAGE_SIZE)
        write_leaf_cells(new_leaf, group, pager=pager, journal=journal)
        pager.write_page(page_no, new_leaf)

    for i, page_no in enumerate(group_page_nos):
        next_link = (
            group_page_nos[i + 1] if i + 1 < len(group_page_nos) else old_right_sibling
        )
        set_next_leaf(pager, page_no, next_link)

    # COPY-UP separators: first key of each right-hand group, in key order.
    separators = [groups[i][0][0] for i in range(1, len(groups))]

    # Inline copy-up invariants per boundary (always on, not test-only): each
    # separator is the first key of its right group and present in that group.
    for i in range(1, len(groups)):
        sep = separators[i - 1]
        right_keys = [c[0] for c in groups[i]]
        assert sep == groups[i][0][0], (
            f"copy-up invariant violated at boundary {i}: separator={sep} "
            f"!= group first key {groups[i][0][0]}"
        )
        assert sep in right_keys, (
            f"copy-up invariant violated at boundary {i}: separator={sep} "
            f"not found in right group keys {right_keys}"
        )

    # Off-by-one guard on the rethreaded chain head and tail.
    assert get_next_leaf(pager, group_page_nos[-1]) == old_right_sibling, (
        "sibling chain tail does not point to the old right sibling"
    )
    if len(group_page_nos) > 1:
        assert get_next_leaf(pager, group0_page_no) == group_page_nos[1], (
            "sibling chain head does not point to the first new leaf"
        )

    logger.debug(
        "leaf_split_and_insert: leaf=%d groups=%d new_pages=%s separators=%s",
        leaf_page_no, len(groups), new_page_nos, separators,
    )

    # Ascend to the parent with the run of (separator, new_child) pairs.
    if not path_stack:
        # The leaf is also the root — build a new root holding all N children.
        _root_from_children(pager, journal, group_page_nos, separators)
    else:
        parent_page_no, _ = path_stack.pop()
        _insert_run_into_interior(
            pager, journal, path_stack, parent_page_no,
            group0_page_no, new_page_nos, separators,
        )


# ---------------------------------------------------------------------------
# Public function 2: Interior split — PUSH-UP
# ---------------------------------------------------------------------------


def _balanced_nway_interior_split(
    all_keys: list[int],
    all_children: list[int],
) -> tuple[list[tuple[list[int], list[int]]], list[int]]:
    """Partition an over-capacity interior node into N>=2 page-fitting nodes (PUSH-UP).

    Returns (nodes, push_up_keys) where nodes is a list of (keys, children) each
    fitting one interior page, and push_up_keys are the N-1 boundary keys sent to
    the parent. Each pushed-up key separates two adjacent child groups and is
    absent from both children — the PUSH-UP discipline. Key count is conserved:
    sum(len(node_keys)) + len(push_up_keys) == len(all_keys).

    Greedy-pack children left -> right: keep adding (left_child, key) pairs while
    the node's keys still fit one interior page; when the next key would overflow,
    that key becomes the boundary push-up (removed from the node) and a new node
    opens with the child that would have followed it. Interior keys are bounded
    varint rowids, so every singleton node fits a page.
    """
    # Build nodes as runs of children; a key sits between consecutive children.
    # node_children accumulates child pointers; node_keys accumulates the keys
    # strictly between them. A boundary push-up closes a node and starts the next.
    nodes: list[tuple[list[int], list[int]]] = []
    push_up_keys: list[int] = []

    cur_children: list[int] = [all_children[0]]
    cur_keys: list[int] = []

    for i, key in enumerate(all_keys):
        next_child = all_children[i + 1]
        # Would appending this key (and its right child) overflow the node?
        if cur_keys and not _interior_fits(cur_keys + [key]):
            # Close the current node; this key is pushed up as the boundary.
            nodes.append((cur_keys, cur_children))
            push_up_keys.append(key)
            cur_keys = []
            cur_children = [next_child]
        else:
            cur_keys.append(key)
            cur_children.append(next_child)

    nodes.append((cur_keys, cur_children))
    return nodes, push_up_keys


def interior_split_and_insert(
    pager: Pager,
    journal: Journal,
    path_stack: list[tuple[int, int]],
    interior_page_no: int,
    all_keys: list[int],
    all_children: list[int],
) -> None:
    """N-way split a full interior node, pushing N-1 boundary keys up to the parent.

    all_keys and all_children represent the over-capacity state produced by
    _insert_run_into_interior after a run of keys/children was spliced into the
    in-memory lists beyond the page byte budget. The node is partitioned into
    N>=2 page-fitting interior nodes; the N-1 boundary keys are extracted and
    sent to the parent (PUSH-UP — absent from both adjacent children).

    After the split (per boundary):
      assert push_up_key not in left_keys and push_up_key not in right_keys
    and key count is conserved across all nodes plus the pushed-up keys.

    Node 0 stays on interior_page_no; N-1 new interior pages are allocated. If
    path_stack is empty, the topmost overflow reinitialises the fixed root in
    place via _root_from_children.
    """
    nodes, push_up_keys = _balanced_nway_interior_split(all_keys, all_children)

    # Single-node result = no split needed. The byte partition fits every key on
    # one interior page, so rewrite the node in place and return without
    # allocating pages or touching the parent. As with the leaf path, the split
    # dispatch carries a secondary key-count ceiling that can fire on a
    # byte-fitting node; a one-node partition must stay a no-op rather than build
    # a degenerate single-child root.
    if len(nodes) == 1:
        node_page = bytearray(PAGE_SIZE)
        write_interior_node(node_page, all_keys, all_children)
        pager.write_page(interior_page_no, node_page)
        logger.debug(
            "interior_split_and_insert: interior=%d single-node no-op (byte-fitting)",
            interior_page_no,
        )
        return

    # Push-up invariants (always on): each boundary key is absent from the two
    # nodes it separates, and the total key count is conserved.
    total_node_keys = sum(len(keys) for keys, _ in nodes)
    assert total_node_keys + len(push_up_keys) == len(all_keys), (
        f"push-up invariant violated: key count not conserved: "
        f"{total_node_keys} + {len(push_up_keys)} != {len(all_keys)}"
    )
    for b, push_up_key in enumerate(push_up_keys):
        left_keys = nodes[b][0]
        right_keys = nodes[b + 1][0]
        assert push_up_key not in left_keys and push_up_key not in right_keys, (
            f"push-up invariant violated at boundary {b}: push_up_key={push_up_key} "
            f"found in adjacent node keys"
        )

    # Node 0 stays on the original page; allocate one new page per right node.
    new_page_nos = [alloc_page(pager, journal=journal) for _ in nodes[1:]]
    node_page_nos = [interior_page_no] + new_page_nos

    for page_no, (keys, children) in zip(node_page_nos, nodes):
        node_page = bytearray(PAGE_SIZE)
        write_interior_node(node_page, keys, children)
        pager.write_page(page_no, node_page)

    logger.debug(
        "interior_split_and_insert: interior=%d new_pages=%s push_up=%s",
        interior_page_no, new_page_nos, push_up_keys,
    )

    # Ascend to the parent with the run of pushed-up (separator, child) pairs.
    if not path_stack:
        # This interior node is the root — reinitialise the fixed root in place
        # with all N children.
        _root_from_children(pager, journal, node_page_nos, push_up_keys)
    else:
        parent_page_no, _ = path_stack.pop()
        _insert_run_into_interior(
            pager, journal, path_stack, parent_page_no,
            interior_page_no, new_page_nos, push_up_keys,
        )


# ---------------------------------------------------------------------------
# Private: root split helpers
# ---------------------------------------------------------------------------


def _root_from_children(
    pager: Pager,
    journal: Journal,
    node_page_nos: list[int],
    separators: list[int],
) -> None:
    """Reinitialise the fixed root page as an interior node over N children.

    node_page_nos[0] is the fixed root page, which currently holds node 0's
    content (the leftmost group of the split that just ran — leaf cells after a
    leaf split, or interior content after an interior split). Its content is
    copied to a freshly allocated child page so the fixed root page can be
    overwritten as an interior node; the remaining node_page_nos are the new
    right children already written by the caller.

    The root page number never changes (fixed-root discipline): only the page
    content flips to an interior node with len(node_page_nos) children and
    len(separators) keys. Works for both the leaf-root promotion (first split of
    a single-leaf root) and the interior-root overflow.
    """
    root_page_no = node_page_nos[0]

    # Move node 0's content off the root page to a fresh child page. For a leaf
    # node-0 this carries the sibling link verbatim (it was set to point at the
    # next group during the leaf split), so the leaf chain stays intact.
    new_node0_page_no = alloc_page(pager, journal=journal)
    root_data = pager.read_page(root_page_no)
    pager.write_page(new_node0_page_no, bytearray(root_data))

    children = [new_node0_page_no] + node_page_nos[1:]

    new_root_page = bytearray(PAGE_SIZE)
    write_interior_node(new_root_page, keys=list(separators), children=children)
    pager.write_page(root_page_no, new_root_page)

    logger.debug(
        "_root_from_children: root=%d node0->%d children=%s separators=%s",
        root_page_no, new_node0_page_no, children, separators,
    )

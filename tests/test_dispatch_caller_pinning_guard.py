"""Default-lane static guard: every core.dispatch branch is either sent by
a real client surface or carries an explicit, rot-guarded list entry. This
is the static inverse of the ``--live`` reachability sweep -- it parses
source only, never imports ``iai_mcp.core``, and runs in the normal
sequential gate.
"""

from __future__ import annotations

from pathlib import Path

from _live_harness import (
    KNOWN_DEAD_DISPATCH_METHODS,
    derive_client_sent_methods,
    dispatch_method_branches,
)

CORE = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "core" / "__init__.py"

# Methods dispatched only from an internal path (sleep pipeline, control
# plane) and never expected from any client surface, each individually
# justified here. Empty today -- nothing is legitimately internal-only at
# the dispatch-method layer.
JUSTIFIED_INTERNAL: "frozenset[str]" = frozenset()


def unpinned_dispatch_methods(branches, client, known_dead, justified):
    return set(branches) - set(client) - set(known_dead) - set(justified)


def test_every_dispatch_handler_has_a_client_sender() -> None:
    branches = dispatch_method_branches(CORE)
    client = derive_client_sent_methods()

    assert client <= branches, (
        f"client-sent method(s) with no parsed branch: {sorted(client - branches)} "
        "-- the AST branch parser under-collected"
    )
    assert len(branches) >= 20, (
        f"only {len(branches)} dispatch branches parsed -- the AST parser "
        "looks broken"
    )
    for anchor in ("memory_recall", "topology", "session_start_payload"):
        assert anchor in branches, (
            f"live anchor {anchor!r} missing from parsed branches -- the "
            "AST parser looks broken"
        )

    unpinned = unpinned_dispatch_methods(
        branches, client, KNOWN_DEAD_DISPATCH_METHODS, JUSTIFIED_INTERNAL
    )
    assert not unpinned, (
        f"dispatch handler(s) with no client sender and no list entry: "
        f"{sorted(unpinned)} -- wire a real sender or add a known-dead "
        "entry with a delete-tracking annotation"
    )


def test_known_dead_and_whitelist_do_not_rot() -> None:
    branches = dispatch_method_branches(CORE)

    orphans = (KNOWN_DEAD_DISPATCH_METHODS | JUSTIFIED_INTERNAL) - branches
    assert not orphans, (
        f"list entry with no matching branch: {sorted(orphans)} -- remove "
        "the stale entry"
    )

    resurrected = KNOWN_DEAD_DISPATCH_METHODS & derive_client_sent_methods()
    assert not resurrected, (
        f"known-dead handler(s) now have a real client sender: "
        f"{sorted(resurrected)} -- move off the known-dead list"
    )


def test_guard_trips_on_a_synthetic_dead_handler(tmp_path: Path) -> None:
    synthetic = tmp_path / "synthetic_dispatch.py"
    synthetic.write_text(
        'if method == "memory_recall":\n'
        "    pass\n"
        'if method == "totally_fake_dead_method":\n'
        "    pass\n",
        encoding="utf-8",
    )

    enumerated = dispatch_method_branches(synthetic)
    assert "totally_fake_dead_method" in enumerated, (
        "AST parser failed to enumerate a synthetic branch"
    )

    unpinned = unpinned_dispatch_methods(
        enumerated, {"memory_recall"}, frozenset(), frozenset()
    )
    assert unpinned == {"totally_fake_dead_method"}, (
        "pinning helper failed to report the synthetic dead branch"
    )


def test_guard_trips_on_a_reversed_idiom_dead_handler(tmp_path: Path) -> None:
    synthetic = tmp_path / "synthetic_dispatch_reversed.py"
    synthetic.write_text(
        'if method == "memory_recall":\n'
        "    pass\n"
        'if "totally_fake_reversed" == method:\n'
        "    pass\n"
        'if method in {"fake_in_a", "fake_in_b"}:\n'
        "    pass\n",
        encoding="utf-8",
    )

    enumerated = dispatch_method_branches(synthetic)
    assert "totally_fake_reversed" in enumerated, (
        "AST parser failed to enumerate a reversed-idiom branch "
        "(literal on the left of ==)"
    )
    assert {"fake_in_a", "fake_in_b"} <= enumerated, (
        "AST parser regressed the method-in-{...} collection form"
    )

    unpinned = unpinned_dispatch_methods(
        enumerated, {"memory_recall"}, frozenset(), frozenset()
    )
    assert unpinned == {"totally_fake_reversed", "fake_in_a", "fake_in_b"}, (
        "pinning helper failed to report the reversed-idiom and/or "
        "in-collection dead branches"
    )

"""Negative-space guard: the priming cache has exactly one awake reader, and
the persisted blob is metadata-only. The cache is built at sleep and
warm-loaded at boot; every *other* awake recall module (`core`, `retrieve`,
`session`, `working_tier`) never references it, and `pipeline._recall_core`
is the cache's one gated reader (see the sibling test below). This module
proves that structurally, not by convention.
"""
from __future__ import annotations

import inspect
from pathlib import Path

from iai_mcp import pipeline, prime_cache
from iai_mcp.store import MemoryStore
from tests.test_prime_cache import _plant_transition

_TARGET_MODULES = (
    "iai_mcp.core",
    "iai_mcp.retrieve",
    "iai_mcp.session",
    "iai_mcp.working_tier",
)


def _source_files_for(mod: object) -> list[Path]:
    """A package (has __path__) contributes every submodule source file, not
    just __init__.py -- a plain single-file module contributes itself."""
    top = Path(mod.__file__)  # type: ignore[arg-type]
    if hasattr(mod, "__path__"):
        return sorted(top.parent.rglob("*.py"))
    return [top]


def test_no_awake_recall_call_site_yet() -> None:
    import importlib

    for name in _TARGET_MODULES:
        mod = importlib.import_module(name)
        for src_path in _source_files_for(mod):
            src = src_path.read_text(encoding="utf-8")
            assert "prime_cache" not in src, (
                f"{src_path} references 'prime_cache' -- the awake recall "
                f"path must not reference prime_cache; its only reader is "
                f"the boot warm-load"
            )


def test_prime_cache_referenced_only_inside_recall_core() -> None:
    """The seam's single call site: `prime_cache` is referenced inside
    `_recall_core` and nowhere else in the module. Structural-presence
    assertions below prove the multi-part mechanism (gate, dual-loop nudge,
    clamp, k_margin truncation-survival lever) is wired, not half-wired --
    verified by mutant (revert-after): dropping the nudge from one loop
    reddens the twice-count assertion, dropping the clamp block reddens the
    clamp-token assertion, and restoring the bare RUST_SCORER_K_MARGIN
    argument reddens the flag_reachable_indices assertion."""
    core_src = inspect.getsource(pipeline._recall_core)
    assert "prime_cache" in core_src, (
        "_recall_core does not reference prime_cache -- the seam is not wired"
    )

    module_src = Path(pipeline.__file__).read_text(encoding="utf-8")
    module_without_core = module_src.replace(core_src, "", 1)
    assert "prime_cache" not in module_without_core, (
        "pipeline.py references prime_cache outside _recall_core -- the "
        "awake priming reader must live only inside the recall entry point"
    )

    assert core_src.count('"IAI_MCP_PROC_PRIME"') == 1, (
        "the priming gate must be checked in exactly one place"
    )
    assert core_src.count("xproc_prime") == 2, (
        "the nudge must be applied at BOTH winners loops (Rust and Python-"
        "fallback) -- a single-loop nudge silently no-ops on whichever "
        "scorer path the daemon actually runs"
    )
    assert core_src.count("proc_prime-clamped") == 1, (
        "the clamp must be wired exactly once, after the branch rejoin"
    )
    assert core_src.count("primed_ids") >= 3, (
        "primed_ids must thread through the widening, k_margin-widening, "
        "and clamp regions"
    )
    assert "flag_reachable_indices.size" in core_src, (
        "the k_margin truncation-survival lever is not wired -- a primed "
        "candidate would be silently dropped by winners.truncate on the "
        "production Rust scorer path"
    )


def _fresh_store(tmp_path: Path) -> "tuple[MemoryStore, Path]":
    home = tmp_path / "operator-home"
    store_root = home / ".iai-mcp"
    store = MemoryStore(path=store_root)
    return store, home


def test_prime_cache_is_metadata_only(tmp_path: Path) -> None:
    # Build-produced, not test-invented: plant real proc_transitions rows
    # and run the real prime_cache.build(store) producer -- a hand-built
    # blob would make the id-shape assertions below circular (they would
    # just check the test's own literals).
    store, _ = _fresh_store(tmp_path)
    _plant_transition(store, src="A", dst="B", chunk_id="c1")
    _plant_transition(store, src="A", dst="C", chunk_id="c2")
    blob = prime_cache.build(store)
    assert prime_cache.save(store, blob) is True
    loaded = prime_cache.load(store)

    for chunk_ids in loaded["seed_to_chunks"].values():
        assert isinstance(chunk_ids, list)
        for chunk_id in chunk_ids:
            assert isinstance(chunk_id, str)
            assert " " not in chunk_id

    for members in loaded["chunk_members"].values():
        assert isinstance(members, list)
        for record_id in members:
            assert isinstance(record_id, str)
            assert " " not in record_id

    with store.db._conn_lock:
        meta_row = store.db._conn.execute(
            "SELECT value FROM _hippo_meta WHERE key = ?",
            (prime_cache.PRIME_CACHE_META_KEY,),
        ).fetchone()
        record_row = store.db._conn.execute(
            "SELECT id FROM records WHERE id = ?",
            (prime_cache.PRIME_CACHE_META_KEY,),
        ).fetchone()
    assert meta_row is not None
    assert record_row is None


def test_prime_cache_build_never_reads_literal_surface() -> None:
    # Structural companion to the content check above: build()'s
    # proc_transitions read stays column-scoped, and its records liveness
    # join (_dead_chunk_ids) is id-only -- neither path can leak free text
    # by construction. Mutant (apply-observe-revert, verified manually):
    # injecting a `row["literal_surface"]` read into either function's
    # source makes this assertion RED.
    combined = inspect.getsource(prime_cache.build) + inspect.getsource(
        prime_cache._dead_chunk_ids
    )
    assert "literal_surface" not in combined
    assert "FROM proc_transitions" in combined
    assert "SELECT id FROM records" in combined

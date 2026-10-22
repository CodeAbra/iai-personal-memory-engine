"""CLI-callable-layer idempotency proof for `iai_mcp.tools.index_all`.

Each lane's Rust-internal layer is proved idempotent independently; this
drives the same two contracts -- convergence on an unchanged tree and
clean replace on a changed file -- through the layer the SessionStart
hook actually calls, across all three lanes together, on a hermetic
`tmp_path` fixture with explicit tmp store roots. Never the default/live
store roots.
"""

from __future__ import annotations

from pathlib import Path

from iai_mcp import tools


def _write_fixture_tree(root: Path, *, old_marker: str, symbol_body: str) -> dict[str, Path]:
    """Builds the three-file fixture tree shared by both tests.

    `target_mod.py` is the stable xref-import target and shared-token
    bearer -- never mutated. `importer_mod.py` carries the old/new marker
    (both the whole-file comment and the symbol name), an import edge to
    `target_mod`, and is the file the replace test mutates in place.
    `distinct_mod.py` is a filler file with its own unique token/symbol,
    unrelated to the shared token, to prove non-participating files are
    unaffected.
    """
    target_mod = root / "target_mod.py"
    target_mod.write_text(
        "def define_zzzsharedzzz_target():\n"
        "    return \"target_body\"\n"
    )

    importer_mod = root / "importer_mod.py"
    importer_mod.write_text(
        f"# {old_marker}\n"
        "from target_mod import define_zzzsharedzzz_target\n"
        "\n"
        f"def helper_zzzsharedzzz_{symbol_body}():\n"
        "    return \"importer_body\"\n"
    )

    distinct_mod = root / "distinct_mod.py"
    distinct_mod.write_text(
        "def helper_distinctfillertoken():\n"
        "    return \"filler\"\n"
    )

    return {
        "target_mod": target_mod,
        "importer_mod": importer_mod,
        "distinct_mod": distinct_mod,
    }


def _paths_str(fixture: dict[str, Path]) -> list[str]:
    return [str(p) for p in fixture.values()]


def _hit_paths(hits: list[tuple[str, float]]) -> list[str]:
    return [path for path, _score in hits]


def test_index_all_over_unchanged_tree_converges(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    symbol_root = tmp_path / "symbols"
    xref_root = tmp_path / "xref"

    fixture = _write_fixture_tree(
        tmp_path, old_marker="zzzoldmarkerzzz", symbol_body="zzzoldmarkerzzz"
    )
    paths = _paths_str(fixture)

    indexed_1 = tools.index_all(
        paths, store_root=store_root, symbol_store_root=symbol_root, xref_store_root=xref_root
    )
    assert indexed_1 == 3

    whole_1 = tools.search("zzzsharedzzz", store_root=store_root)
    symbol_1 = tools.search_symbol("zzzsharedzzz", store_root=symbol_root)
    xref_1 = tools.search_xref("zzzsharedzzz", store_root=store_root, xref_store_root=xref_root)

    # Every lane must find the shared token exactly once per bearing file --
    # no doubled entry for the same path even on the FIRST index.
    expected_shared_bearers = sorted(
        [str(fixture["importer_mod"]), str(fixture["target_mod"])]
    )
    assert sorted(_hit_paths(whole_1)) == expected_shared_bearers, whole_1
    assert len(_hit_paths(whole_1)) == len(set(_hit_paths(whole_1))), whole_1
    assert str(fixture["importer_mod"]) in _hit_paths(symbol_1), symbol_1
    assert len(_hit_paths(symbol_1)) == len(set(_hit_paths(symbol_1))), symbol_1
    assert len(_hit_paths(xref_1)) == len(set(_hit_paths(xref_1))), xref_1

    indexed_2 = tools.index_all(
        paths, store_root=store_root, symbol_store_root=symbol_root, xref_store_root=xref_root
    )
    assert indexed_2 == 3

    whole_2 = tools.search("zzzsharedzzz", store_root=store_root)
    symbol_2 = tools.search_symbol("zzzsharedzzz", store_root=symbol_root)
    xref_2 = tools.search_xref("zzzsharedzzz", store_root=store_root, xref_store_root=xref_root)

    # Convergence: re-indexing the unchanged tree adds zero net new
    # postings -- the hit set (as a set of paths) is identical, and no
    # lane doubles an entry for the same path.
    assert sorted(_hit_paths(whole_2)) == sorted(_hit_paths(whole_1)), (whole_1, whole_2)
    assert len(_hit_paths(whole_2)) == len(set(_hit_paths(whole_2))), whole_2

    assert sorted(_hit_paths(symbol_2)) == sorted(_hit_paths(symbol_1)), (symbol_1, symbol_2)
    assert len(_hit_paths(symbol_2)) == len(set(_hit_paths(symbol_2))), symbol_2

    assert sorted(_hit_paths(xref_2)) == sorted(_hit_paths(xref_1)), (xref_1, xref_2)
    assert len(_hit_paths(xref_2)) == len(set(_hit_paths(xref_2))), xref_2

    # The xref lane's injection must still surface the import target.
    assert str(fixture["target_mod"]) in _hit_paths(xref_2), xref_2


def test_index_all_replaces_changed_file(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    symbol_root = tmp_path / "symbols"
    xref_root = tmp_path / "xref"

    old_marker = "zzzoldmarkerzzz"
    new_marker = "zzznewmarkerzzz"
    fixture = _write_fixture_tree(tmp_path, old_marker=old_marker, symbol_body=old_marker)
    paths = _paths_str(fixture)

    tools.index_all(
        paths, store_root=store_root, symbol_store_root=symbol_root, xref_store_root=xref_root
    )

    # Mutate importer_mod.py in place: old marker -> new marker, in both
    # the whole-file comment and the symbol name, keeping the shared
    # token and the import edge unchanged.
    fixture["importer_mod"].write_text(
        f"# {new_marker}\n"
        "from target_mod import define_zzzsharedzzz_target\n"
        "\n"
        f"def helper_zzzsharedzzz_{new_marker}():\n"
        "    return \"importer_body\"\n"
    )

    tools.index_all(
        paths, store_root=store_root, symbol_store_root=symbol_root, xref_store_root=xref_root
    )

    importer_path = str(fixture["importer_mod"])

    # Whole-file lane: old marker gone, new marker present, exactly once.
    whole_old = tools.search(old_marker, store_root=store_root)
    whole_new = tools.search(new_marker, store_root=store_root)
    assert importer_path not in _hit_paths(whole_old), whole_old
    assert _hit_paths(whole_new).count(importer_path) == 1, whole_new

    # Symbol lane: old symbol name unfindable, new symbol name findable,
    # exactly one hit for the path.
    symbol_old = tools.search_symbol(old_marker, store_root=symbol_root)
    symbol_new = tools.search_symbol(new_marker, store_root=symbol_root)
    assert importer_path not in _hit_paths(symbol_old), symbol_old
    assert _hit_paths(symbol_new).count(importer_path) == 1, symbol_new

    # Shared token still resolves to exactly one hit for the changed path
    # in every lane -- no stale duplicate left behind by the replace.
    whole_shared = tools.search("zzzsharedzzz", store_root=store_root)
    symbol_shared = tools.search_symbol("zzzsharedzzz", store_root=symbol_root)
    xref_shared = tools.search_xref(
        "zzzsharedzzz", store_root=store_root, xref_store_root=xref_root
    )
    assert _hit_paths(whole_shared).count(importer_path) == 1, whole_shared
    assert _hit_paths(symbol_shared).count(importer_path) == 1, symbol_shared
    assert _hit_paths(xref_shared).count(importer_path) == 1, xref_shared

    # The xref edge (importer_mod -> target_mod) survives the replace,
    # re-emitted (not duplicated) since importer_mod's content changed.
    assert _hit_paths(xref_shared).count(str(fixture["target_mod"])) == 1, xref_shared


def test_search_fused_over_reindexed_fixture_is_single_copy(tmp_path: Path) -> None:
    """`search_fused`'s `_dedup_preserve_order` calls dedup the whole/symbol
    lanes before RRF fusion -- confirms the fused result stays single-copy
    now that the underlying lanes no longer duplicate on re-index, i.e.
    those dedup calls are harmless no-ops rather than load-bearing."""
    store_root = tmp_path / "store"
    symbol_root = tmp_path / "symbols"
    xref_root = tmp_path / "xref"
    vector_root = tmp_path / "vectors"

    fixture = _write_fixture_tree(
        tmp_path, old_marker="zzzoldmarkerzzz", symbol_body="zzzoldmarkerzzz"
    )
    paths = _paths_str(fixture)

    tools.index_all(
        paths, store_root=store_root, symbol_store_root=symbol_root, xref_store_root=xref_root
    )
    tools.index_all(
        paths, store_root=store_root, symbol_store_root=symbol_root, xref_store_root=xref_root
    )

    # weak_threshold=-1.0 keeps the base fused score above the gate, so
    # this stays on the base-lanes-only path with no embed subprocess.
    fused = tools.search_fused(
        "zzzsharedzzz",
        weak_threshold=-1.0,
        store_root=store_root,
        symbol_store_root=symbol_root,
        xref_store_root=xref_root,
        vector_store_root=vector_root,
    )
    fused_paths = _hit_paths(fused)
    assert len(fused_paths) == len(set(fused_paths)), fused
    assert str(fixture["importer_mod"]) in fused_paths, fused

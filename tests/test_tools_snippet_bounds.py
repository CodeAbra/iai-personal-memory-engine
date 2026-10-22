"""Multi-hit combined-budget coverage for the snippet-attachment step:
`tail.rs`'s existing bounded-read tests are all single-window, so this is
the first test covering the TOTAL-response byte budget across several
hits at once, plus the xref lane's "may exceed k" contract under that
budget.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from iai_mcp.tools import (  # noqa: E402
    SNIPPET_LINE_CAP,
    SNIPPET_TOTAL_BYTE_BUDGET,
    attach_snippets,
    search,
    search_with_snippets,
)


def _write_long_matching_file(path: Path, marker: str, *, line_count: int = 40) -> None:
    """Every line is long and contains `marker` many times over, so the
    per-hit snippet extractor has plenty of high-overlap lines to choose
    from and the resulting snippet content is large."""
    line = f"# {marker} " + ("alice_payload_segment " * 20) + f"{marker}\n"
    path.write_text(line * line_count)


def test_pinned_bound_constants() -> None:
    assert SNIPPET_LINE_CAP == 10
    assert SNIPPET_TOTAL_BYTE_BUDGET == 24 * 1024


def test_per_hit_snippet_never_exceeds_the_line_cap(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    store_root = tmp_path / "store"
    for i in range(6):
        _write_long_matching_file(corpus / f"file{i}.py", "alice_broad_marker")

    from iai_mcp.tools import index_paths

    index_paths([str(p) for p in corpus.glob("*.py")], store_root=store_root)

    hits = search_with_snippets("alice_broad_marker", k=10, store_root=store_root)
    assert hits
    for hit in hits:
        line_count = hit["snippet"].count("\n") + 1 if hit["snippet"] else 0
        assert line_count <= SNIPPET_LINE_CAP, hit


def test_total_snippet_budget_bounds_combined_content_without_dropping_rows(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    store_root = tmp_path / "store"
    # Each file's full-cap snippet is roughly 10 * ~180 bytes ~= 1.8KB;
    # 20 hits comfortably exceed the 24KiB total budget.
    file_count = 20
    for i in range(file_count):
        _write_long_matching_file(corpus / f"file{i:02d}.py", "alice_budget_marker")

    from iai_mcp.tools import index_paths

    index_paths([str(p) for p in corpus.glob("*.py")], store_root=store_root)

    hits = search_with_snippets("alice_budget_marker", k=file_count, store_root=store_root)
    assert len(hits) == file_count, "the budget must never drop a hit row"

    total_snippet_bytes = sum(len(h["snippet"].encode("utf-8")) for h in hits)
    assert total_snippet_bytes <= SNIPPET_TOTAL_BYTE_BUDGET

    # At least one hit must have been budget-exhausted (empty snippet with
    # an elision marker) given 20 hits comfortably exceed the byte budget.
    exhausted = [h for h in hits if h["snippet"] == ""]
    assert exhausted, "expected at least one hit past the total budget"
    for h in exhausted:
        # budget_elided_lines is the "withheld for budget" signal; it must
        # never be conflated with elided_lines (the count of matching
        # lines outside the window, which extract_snippet computes on its
        # own and this budget step must never overwrite).
        assert h["budget_elided_lines"] > 0
        assert h["elided_lines"] >= 0
        # The row itself (path, score, line range) is preserved even
        # though the snippet content was elided.
        assert h["path"]
        assert h["line_end"] >= h["line_start"] >= 1


def test_budget_elided_lines_never_overwrites_the_rust_computed_elided_lines(
    tmp_path: Path,
) -> None:
    """`elided_lines` (query-matching lines outside the returned window,
    computed by `extract_snippet` itself) and `budget_elided_lines`
    (0 unless this hit's snippet was withheld for the total byte budget)
    are two DISTINCT counts -- budget exhaustion must never clobber the
    former with the latter's value."""
    marker = "alice_disambiguation_marker"
    p = tmp_path / "file.py"
    _write_long_matching_file(p, marker, line_count=15)

    attached = attach_snippets([(str(p), 1.0)], marker, total_byte_budget=0)
    assert len(attached) == 1
    h = attached[0]

    assert h["snippet"] == "", "a zero byte budget must withhold every snippet"
    # window covers SNIPPET_LINE_CAP=10 lines out of 15 matching lines.
    assert h["budget_elided_lines"] == SNIPPET_LINE_CAP
    assert h["elided_lines"] == 15 - SNIPPET_LINE_CAP
    assert h["budget_elided_lines"] != h["elided_lines"]


def test_hit_order_unchanged_by_snippet_attachment(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    store_root = tmp_path / "store"
    for i in range(8):
        _write_long_matching_file(corpus / f"file{i}.py", "alice_order_marker")

    from iai_mcp.tools import index_paths

    index_paths([str(p) for p in corpus.glob("*.py")], store_root=store_root)

    base_hits = search("alice_order_marker", k=8, store_root=store_root)
    snippeted_hits = search_with_snippets("alice_order_marker", k=8, store_root=store_root)

    assert [p for p, _ in base_hits] == [h["path"] for h in snippeted_hits]


def test_index_paths_resolves_a_relative_input_path_to_absolute(
    tmp_path: Path, monkeypatch
) -> None:
    """`index_paths` (and the sibling index_* functions) must resolve every
    input path to absolute before it reaches the store, regardless of
    whether the caller passed a relative or absolute entry --
    `_normalize_hit_path`/`attach_snippets` assume every hit is already
    absolute (or was normalized from one) and never fall back to the
    process cwd to resolve a relative hit."""
    from iai_mcp.tools import index_paths, search

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "rel_marker.py").write_text("alice_relpath_marker\n")

    store_root = tmp_path / "store"
    monkeypatch.chdir(corpus)
    index_paths(["rel_marker.py"], store_root=store_root)

    hits = search("alice_relpath_marker", k=5, store_root=store_root)
    assert hits
    assert Path(hits[0][0]).is_absolute(), (
        f"stored hit path must be absolute, got: {hits[0][0]}"
    )


def test_fused_lane_snippet_resolves_from_indexed_repo_root_not_process_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    """A fused/vector hit's path is rewritten repo-relative by
    `_normalize_hit_path` before fusion; the snippet must still resolve
    against the indexed corpus's OWN repo root, never whatever directory
    the calling process happens to have as its cwd. Builds two SEPARATE
    git-tracked repos that share the same relative file path but hold
    different content, indexes only the first, then chdir's into the
    second before resolving the fused-lane snippet -- a cwd-anchored
    resolve would either come up empty or, worse, silently return the
    second repo's content.
    """
    from iai_mcp.tools import attach_snippets, index_all, search_fused_detailed

    corpus_repo = tmp_path / "corpus_repo"
    (corpus_repo / ".git").mkdir(parents=True)
    (corpus_repo / "src").mkdir()
    marker_file = corpus_repo / "src" / "mod.py"
    marker_file.write_text("def alice_marker_fn():\n    return 'CORPUS_REPO_TRUE_CONTENT'\n")

    decoy_repo = tmp_path / "decoy_repo"
    (decoy_repo / ".git").mkdir(parents=True)
    (decoy_repo / "src").mkdir()
    (decoy_repo / "src" / "mod.py").write_text(
        "def alice_marker_fn():\n    return 'DECOY_REPO_WRONG_CONTENT'\n"
    )

    store_root = tmp_path / "store"
    symbol_store_root = tmp_path / "symbol_store"
    xref_store_root = tmp_path / "xref_store"
    index_all(
        [str(marker_file)],
        store_root=store_root,
        symbol_store_root=symbol_store_root,
        xref_store_root=xref_store_root,
    )

    monkeypatch.chdir(decoy_repo)

    detailed = search_fused_detailed(
        "alice_marker_fn",
        k=5,
        store_root=store_root,
        symbol_store_root=symbol_store_root,
        xref_store_root=xref_store_root,
    )
    hits_out = attach_snippets(
        detailed["hits"], "alice_marker_fn", abs_paths=detailed["abs_paths"]
    )
    assert hits_out, "expected at least one fused hit"
    target = next((h for h in hits_out if h["path"] == "src/mod.py"), None)
    assert target is not None, f"expected the normalized repo-relative hit, got: {hits_out}"

    assert "DECOY_REPO_WRONG_CONTENT" not in target["snippet"], (
        "snippet resolution must never silently substitute the decoy repo's "
        f"file for the indexed corpus's own file: {target}"
    )
    assert "CORPUS_REPO_TRUE_CONTENT" in target["snippet"], (
        f"expected the snippet to resolve from the indexed corpus repo, got: {target}"
    )


def testattach_snippets_degrades_gracefully_for_an_unresolvable_relative_hit_path() -> None:
    """A repo-relative hit path with no matching `abs_paths` entry (e.g. a
    caller that forgot to thread the map `_normalize_hit_path` built) must
    degrade to an empty snippet rather than guessing a path relative to
    the process cwd -- never a crash, never wrong content."""
    from iai_mcp.tools import attach_snippets

    attached = attach_snippets([("src/mod.py", 1.0)], "anything")
    assert len(attached) == 1
    assert attached[0]["snippet"] == ""
    assert attached[0]["path"] == "src/mod.py"


def test_xref_shaped_result_exceeding_k_keeps_every_row_under_budget(tmp_path: Path) -> None:
    """`search_xref` may legitimately return more than `k` rows (injected
    definition files, lib.rs:156-161) -- the total budget must cap
    snippet CONTENT only, never re-truncate the row count back to `k`.
    This test constructs an xref-shaped hit list directly (more rows than
    a plausible `k`) and feeds it through the shared attach step.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    files = []
    for i in range(12):
        p = corpus / f"injected{i}.py"
        _write_long_matching_file(p, "alice_xref_marker")
        files.append(p)

    hits = [(str(p), 1.0 - i * 0.01) for i, p in enumerate(files)]
    k = 5
    assert len(hits) > k, "fixture must exceed k to exercise the xref over-k contract"

    attached = attach_snippets(hits, "alice_xref_marker")
    assert len(attached) == len(hits), "the budget must never re-truncate rows back to k"
    for h in attached:
        line_count = h["snippet"].count("\n") + 1 if h["snippet"] else 0
        assert line_count <= SNIPPET_LINE_CAP

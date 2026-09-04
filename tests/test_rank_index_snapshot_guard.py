"""Two invariants for the columnar rank index's transitional bulk export:

- Memo: a repeated `_RankIndexHandle.snapshot()` call at an unchanged
  (generation, tokens) key returns the memoized export instead of
  recomputing the matrix/degree/postings artifacts.
- Guard: the ONLY production call sites under src/iai_mcp (tests excluded)
  that invoke `.snapshot()` on the rank index are the adapter's own
  internal `self._index.snapshot(...)` line and the live recall path's
  `_rank_handle.snapshot(...)` call -- the first, and only sanctioned,
  production consumer this adapter was built for. Any THIRD call site is
  still a violation: this guard keeps the surface exactly as wide as the
  live path requires, never wider.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp import retrieve
from iai_mcp.store import MemoryStore
from iai_mcp.store._rank_index import rank_index_for
from iai_mcp.types import MemoryRecord

SRC = Path(__file__).resolve().parent.parent / "src" / "iai_mcp"
RANK_INDEX_FILE = SRC / "store" / "_rank_index.py"
CORE_DISPATCH_FILE = SRC / "core" / "__init__.py"

# Named, justified exemption: MetaAnalyst().snapshot(store, hours) inside
# the DMN reflection sleep-step is the meta-analyst's OWN unrelated
# snapshot -- it predates this guard by seven weeks and has nothing to do
# with the rank index or the recall path. Widening this set is a guard
# failure to review, not a rubber stamp: it must stay exactly one entry,
# and the entry must resolve to a real file.
_NON_RANK_INDEX_SNAPSHOT_MODULES = frozenset({
    SRC / "lilli" / "cycle" / "sleep_pipeline" / "_dmn.py",
})


class _SnapshotCallVisitor(ast.NodeVisitor):
    """Flags every `.snapshot(` call site, whitelisting exactly two: the
    `self._index.snapshot(` line inside `_RankIndexHandle.snapshot`'s own
    body, and the live recall path's `_rank_handle.snapshot(` call inside
    `core.dispatch` -- both by class/function/receiver context, never by
    excluding a whole file. A third `.snapshot(` anywhere else in either
    file, or any other production module, still registers as a
    violation."""

    def __init__(self, file: Path) -> None:
        self.file = file
        self.violations: list[tuple[Path, int]] = []
        self.candidates = 0
        self._class_stack: list[str] = []
        self._func_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._func_stack.append(node.name)
        self.generic_visit(node)
        self._func_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "snapshot":
            self.candidates += 1
            if not self._is_whitelisted(node.func):
                self.violations.append((self.file, node.lineno))
        self.generic_visit(node)

    def _is_whitelisted(self, func: ast.Attribute) -> bool:
        if self.file.name == "_rank_index.py":
            return self._is_adapter_internal_call(func)
        if self.file == CORE_DISPATCH_FILE:
            return self._is_live_path_call(func)
        return False

    def _is_adapter_internal_call(self, func: ast.Attribute) -> bool:
        if self._class_stack[-1:] != ["_RankIndexHandle"]:
            return False
        if self._func_stack[-1:] != ["snapshot"]:
            return False
        value = func.value
        return (
            isinstance(value, ast.Attribute)
            and value.attr == "_index"
            and isinstance(value.value, ast.Name)
            and value.value.id == "self"
        )

    def _is_live_path_call(self, func: ast.Attribute) -> bool:
        if self._func_stack[-1:] != ["dispatch"]:
            return False
        value = func.value
        return isinstance(value, ast.Name) and value.id == "_rank_handle"


# ---------------------------------------------------------------------------
# No-production-snapshot-consumer guard (also the recall-path zero-delta
# evidence)
# ---------------------------------------------------------------------------

def test_no_production_snapshot_consumer_guard():
    """The only production call sites under src/iai_mcp (tests excluded)
    that invoke `.snapshot()` on the rank index are the adapter's own
    `self._index.snapshot(...)` call inside `_RankIndexHandle.snapshot`,
    and the live recall path's `_rank_handle.snapshot(...)` call inside
    `core.dispatch` -- the first, and only sanctioned, production consumer
    this adapter was built for. DMN's unrelated `MetaAnalyst().snapshot()`
    is a named module exemption (see `_NON_RANK_INDEX_SNAPSHOT_MODULES`),
    applied at scan level, never inside the visitor.

    A THIRD call site anywhere else is still a violation: this guard keeps
    the recall-path surface exactly as wide as the live path requires.
    """
    all_violations: list[tuple[Path, int]] = []
    total_candidates = 0
    whitelisted_hits = 0
    for f in sorted(SRC.rglob("*.py")):
        tree = ast.parse(f.read_text(), filename=str(f))
        visitor = _SnapshotCallVisitor(f)
        visitor.visit(tree)
        total_candidates += visitor.candidates
        if f in _NON_RANK_INDEX_SNAPSHOT_MODULES:
            continue
        all_violations.extend(visitor.violations)
        whitelisted_hits += visitor.candidates - len(visitor.violations)

    assert not all_violations, (
        "production .snapshot() call site(s) found outside the two "
        "sanctioned call sites (_RankIndexHandle.snapshot's own body and "
        "core.dispatch's live-path call):\n"
        + "\n".join(f"  {p}:{lineno}" for p, lineno in all_violations)
    )
    assert total_candidates >= 3, (
        f"only found {total_candidates} .snapshot( call sites across "
        "src/iai_mcp/ -- expected at least 3 (the rank index's own call, "
        "the live recall path's call, plus DMN's unrelated MetaAnalyst "
        "call); the scan is broken (renamed method, moved file, or a "
        "pattern that stopped matching), not clean"
    )
    assert whitelisted_hits == 2, (
        f"expected exactly 2 whitelisted rank-index snapshot() call sites "
        f"(the adapter's own and the live recall path's), found "
        f"{whitelisted_hits} -- the whitelist logic may have started "
        "matching too much or too little"
    )
    for exempt in _NON_RANK_INDEX_SNAPSHOT_MODULES:
        assert exempt.exists(), f"exempted module no longer exists: {exempt}"


def test_snapshot_guard_scan_is_actually_populated():
    """The guard above fails silent if the scan resolves to an empty file
    list. Pin the scanned list non-empty and confirm the rank index's own
    module is actually in it."""
    scanned = sorted(SRC.rglob("*.py"))
    assert scanned, "snapshot guard scan resolved to an empty file list"
    assert RANK_INDEX_FILE in scanned


def test_snapshot_guard_visitor_flags_planted_violation_in_unrelated_file():
    """Positive control: a `.snapshot()` call in an unrelated production
    module must be flagged, not silently pass."""
    source = "def foo(idx):\n    return idx.snapshot(1, [])\n"
    tree = ast.parse(source, filename="<synthetic>")
    visitor = _SnapshotCallVisitor(Path("not_rank_index.py"))
    visitor.visit(tree)
    assert len(visitor.violations) == 1
    assert visitor.candidates == 1


def test_snapshot_guard_visitor_flags_second_call_in_rank_index_file():
    """Positive control: a second, non-whitelisted `.snapshot()` call
    inside `_rank_index.py` must still fail -- the whitelist is scoped to
    one call site inside one method, never the whole file."""
    source = (
        "class _RankIndexHandle:\n"
        "    def snapshot(self, graph, tokens=None):\n"
        "        return self._index.snapshot(1, tokens)\n"
        "    def other(self):\n"
        "        return self._index.snapshot(2, [])\n"
    )
    tree = ast.parse(source, filename="<synthetic>")
    visitor = _SnapshotCallVisitor(Path("_rank_index.py"))
    visitor.visit(tree)
    assert len(visitor.violations) == 1
    assert visitor.candidates == 2


def test_snapshot_guard_module_allowlist_does_not_leak_into_visitor():
    """The module-level exemption for DMN's unrelated `.snapshot()` call
    is applied by the scan (a module allow-list), never inside the
    visitor: the same `self._index.snapshot(` snippet, scanned under a
    filename other than `_rank_index.py`, must still register as a
    violation at the visitor level -- proving the exemption is a
    scan-time decision a real rank-index consumer could not hide behind
    by simply living in an exempted module."""
    source = (
        "class _RankIndexHandle:\n"
        "    def snapshot(self, graph, tokens=None):\n"
        "        return self._index.snapshot(1, tokens)\n"
    )
    tree = ast.parse(source, filename="<synthetic>")
    visitor = _SnapshotCallVisitor(Path("_dmn.py"))
    visitor.visit(tree)
    assert len(visitor.violations) == 1, (
        "scanned under a filename other than '_rank_index.py', the "
        "whitelist condition must not match -- the module allow-list, "
        "not the visitor, is what exempts _dmn.py"
    )


# ---------------------------------------------------------------------------
# Per-generation memo for the transitional bulk snapshot export
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(
        _keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p)
    )
    monkeypatch.setattr(
        _keyring, "delete_password", lambda s, u: fake.pop((s, u), None)
    )
    yield fake


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(path=tmp_path / "lancedb")
    s.root = tmp_path
    return s


def _make_record(
    store: MemoryStore,
    text: str = "hello",
    vec_seed: float = 0.1,
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[vec_seed] * store.embed_dim,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=["t"],
        language="en",
    )


def test_export_memoized_once_per_generation_change(store: MemoryStore):
    """A repeated `snapshot()` call at an unchanged generation must hit
    the per-generation export memo, not recompute the bulk matrix/degree/
    postings export; a generation change must invalidate the memo and
    recompute exactly once."""
    rec = _make_record(store, "memo-once", 0.2)
    store.insert(rec)
    graph, _assignment, _rc = retrieve.build_runtime_graph(store)

    handle = rank_index_for(store, graph)
    handle._build(graph)  # populate self._index; the export memo stays empty

    calls = {"n": 0}
    real_index = handle._index
    real_snapshot = real_index.snapshot

    class _SnapshotSpy:
        def snapshot(self, *args, **kwargs):
            calls["n"] += 1
            return real_snapshot(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(real_index, name)

    handle._index = _SnapshotSpy()

    gen1, ids1, *_rest1 = handle.snapshot(graph, [])
    assert calls["n"] == 1, "the first call after a fresh build must compute the export"

    gen1b, ids1b, *_rest1b = handle.snapshot(graph, [])
    assert calls["n"] == 1, "a repeated call at the same generation must hit the memo"
    assert gen1b == gen1
    assert list(ids1b) == list(ids1)

    rec2 = _make_record(store, "advance-generation", 0.4)
    store.insert(rec2)
    gen2, ids2, *_rest2 = handle.snapshot(graph, [])
    assert gen2 > gen1, (
        "the fixture must actually advance graph._pool_content_version, "
        "or the recompute-once assertion below is vacuous"
    )
    assert calls["n"] == 2, "a generation change must invalidate the memo and recompute exactly once"
    assert rec2.id.int in ids2


def test_export_memo_keyed_on_tokens_not_generation_alone(store: MemoryStore):
    """A same-generation call with a DIFFERENT token set must not be
    served from the other token set's cached postings -- the memo key is
    (generation, tokens), not generation alone."""
    rec = _make_record(store, "hello searchable world", 0.3)
    store.insert(rec)
    graph, _assignment, _rc = retrieve.build_runtime_graph(store)
    handle = rank_index_for(store, graph)

    gen_a, _ids_a, _mat_a, _degree_a, postings_a = handle.snapshot(graph, ["hello"])
    assert "hello" in postings_a

    gen_b, _ids_b, _mat_b, _degree_b, postings_b = handle.snapshot(graph, ["searchable"])
    assert gen_b == gen_a, "generation must not have changed between the two calls"
    assert "searchable" in postings_b
    assert set(postings_b) == {"searchable"}, (
        "a differing token set at the same generation must yield a fresh "
        "export, not the previous call's cached postings"
    )

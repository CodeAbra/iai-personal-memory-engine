from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest


def _rec(
    *,
    text: str = "t",
    tags: list[str] | None = None,
    language: str = "en",
    tier: str = "semantic",
    detail_level: int = 2,
    created_at: datetime | None = None,
):
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    now = created_at or datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
        aaak_index="",
        embedding=[1.0] + [0.0] * (EMBED_DIM - 1),
        community_id=None,
        centrality=0.0,
        detail_level=detail_level,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=list(tags or []),
        language=language,
    )


@pytest.fixture(autouse=True)
def _patch_embedder(monkeypatch):
    from iai_mcp import embed as embed_mod
    from iai_mcp.types import EMBED_DIM

    class _FakeEmbedder:
        DIM = EMBED_DIM
        DEFAULT_DIM = EMBED_DIM
        DEFAULT_MODEL_KEY = "fake"

        def __init__(self, *args, **kwargs):
            self.DIM = EMBED_DIM

        def embed(self, text: str) -> list[float]:
            return [1.0] + [0.0] * (EMBED_DIM - 1)

        def embed_batch(self, texts):
            return [self.embed(t) for t in texts]

    monkeypatch.setattr(embed_mod, "Embedder", _FakeEmbedder)
    yield


def test_semantic_pruned_tier_constant_and_enum_membership():
    from iai_mcp.types import SEMANTIC_PRUNED_TIER, TIER_ENUM

    assert SEMANTIC_PRUNED_TIER == "semantic_pruned", (
        "The constant value must be 'semantic_pruned' (used as a soft-delete "
        "sentinel by cleanup_schema_duplicates)."
    )
    assert SEMANTIC_PRUNED_TIER in TIER_ENUM, (
        "TIER_ENUM must include 'semantic_pruned' so MemoryRecord.__post_init__ "
        "tier validation accepts pruned rows when reading them back from the store."
    )


def test_memoryrecord_accepts_semantic_pruned_tier():
    rec = _rec(tier="semantic_pruned", text="pruned dup")
    assert rec.tier == "semantic_pruned"


def test_memoryrecord_existing_tiers_still_accepted():
    for tier in ("working", "episodic", "semantic", "procedural", "parametric"):
        rec = _rec(tier=tier, text=f"t-{tier}")
        assert rec.tier == tier


def test_memoryrecord_invalid_tier_still_raises():
    with pytest.raises(ValueError, match="invalid tier"):
        _rec(tier="garbage")


def _seed_dup_store(
    tmp_path: Path,
    n_per_pattern: int = 4,
    n_patterns: int = 3,
    extra_singletons: int = 0,
):
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    base = datetime.now(timezone.utc)
    patterns: list[str] = []

    for p_idx in range(n_patterns):
        pattern = f"tags:capture+role:user+p{p_idx}"
        patterns.append(pattern)
        schema_ids = []
        for s_idx in range(n_per_pattern):
            sched_at = base + timedelta(seconds=p_idx * 60 + s_idx)
            sch = _rec(
                text=f"schema-p{p_idx}-i{s_idx}",
                tier="semantic",
                tags=[f"pattern:{pattern}", "schema"],
                created_at=sched_at,
            )
            store.insert(sch)
            schema_ids.append(sch.id)

            ev = _rec(
                text=f"ev-p{p_idx}-i{s_idx}",
                tier="episodic",
                tags=["capture", "role:user"],
                created_at=sched_at,
            )
            store.insert(ev)
            store.boost_edges(
                [(ev.id, sch.id)],
                edge_type="schema_instance_of",
                delta=0.1,
            )

    for s_idx in range(extra_singletons):
        pattern = f"singleton-p{s_idx}"
        patterns.append(pattern)
        sch = _rec(
            text=f"singleton-{s_idx}",
            tier="semantic",
            tags=[f"pattern:{pattern}", "schema"],
            created_at=base + timedelta(seconds=10000 + s_idx),
        )
        store.insert(sch)

    return store, patterns


def _count_semantic_pattern_records(store, pattern_tag_prefix: str = "pattern:") -> int:
    return sum(
        1
        for r in store.all_records()
        if r.tier == "semantic"
        and any(t.startswith(pattern_tag_prefix) for t in (r.tags or []))
    )


def _count_pruned(store) -> int:
    return sum(1 for r in store.all_records() if r.tier == "semantic_pruned")


def test_cleanup_dry_run_does_not_mutate_store(tmp_path):
    from iai_mcp.migrate import cleanup_schema_duplicates

    store, _patterns = _seed_dup_store(
        tmp_path, n_per_pattern=4, n_patterns=3
    )
    pre_semantic = _count_semantic_pattern_records(store)
    pre_pruned = _count_pruned(store)
    assert pre_semantic == 12
    assert pre_pruned == 0

    siblings_pre = sorted(p.name for p in tmp_path.iterdir())

    summary = cleanup_schema_duplicates(store, apply=False)

    assert summary["mode"] == "dry-run"
    assert summary["groups"] == 3
    assert summary["keepers"] == 3
    assert summary["pruned"] == 9
    assert summary["snapshot_dir"] is None

    assert _count_semantic_pattern_records(store) == 12
    assert _count_pruned(store) == 0

    siblings_post = sorted(p.name for p in tmp_path.iterdir())
    assert siblings_pre == siblings_post, (
        f"--dry-run must not create any sibling directories; saw new entries: "
        f"{set(siblings_post) - set(siblings_pre)}"
    )


def test_cleanup_apply_creates_snapshot_directory_before_writes(tmp_path):
    from iai_mcp.migrate import cleanup_schema_duplicates

    store, _patterns = _seed_dup_store(tmp_path, n_per_pattern=4, n_patterns=3)
    summary = cleanup_schema_duplicates(store, apply=True)

    assert summary["mode"] == "apply"
    assert summary["snapshot_dir"] is not None

    snap = Path(summary["snapshot_dir"])
    assert snap.exists() and snap.is_dir()
    assert snap.parent == Path(store.root)
    assert snap.name.startswith("hippo-pre-cleanup-")
    suffix = snap.name[len("hippo-pre-cleanup-"):]
    assert len(suffix) == 16 and suffix.endswith("Z")

    snap_entries = {p.name for p in snap.iterdir()}
    assert snap_entries, f"snapshot must be non-empty, got: {snap_entries}"
    has_hippo = "brain.sqlite3" in snap_entries
    has_lance = any(e.endswith(".lance") for e in snap_entries)
    assert has_hippo or has_lance, (
        f"snapshot must contain brain.sqlite3 or .lance dirs; got: {snap_entries}"
    )


def test_cleanup_apply_soft_deletes_duplicates_via_tier_rename(tmp_path):
    from iai_mcp.migrate import cleanup_schema_duplicates

    store, _patterns = _seed_dup_store(tmp_path, n_per_pattern=4, n_patterns=3)
    cleanup_schema_duplicates(store, apply=True)

    assert _count_semantic_pattern_records(store) == 3
    assert _count_pruned(store) == 9


def test_cleanup_apply_reinforces_edges_onto_keeper(tmp_path):
    from iai_mcp.migrate import cleanup_schema_duplicates
    from iai_mcp.store import EDGES_TABLE

    store, patterns = _seed_dup_store(tmp_path, n_per_pattern=4, n_patterns=3)

    edges_pre = store.db.open_table(EDGES_TABLE).to_pandas()
    pre_total_sio = int(
        (edges_pre["edge_type"] == "schema_instance_of").sum()
    )
    assert pre_total_sio == 12

    pattern_to_keeper_id = {}
    for p in patterns:
        recs = sorted(
            (
                r
                for r in store.all_records()
                if r.tier == "semantic"
                and f"pattern:{p}" in (r.tags or [])
            ),
            key=lambda r: r.created_at,
        )
        if recs:
            pattern_to_keeper_id[p] = recs[0].id

    summary = cleanup_schema_duplicates(store, apply=True)
    assert summary["edges_reinforced"] >= 9

    edges_post = store.db.open_table(EDGES_TABLE).to_pandas()
    for pattern, keeper_id in pattern_to_keeper_id.items():
        keeper_str = str(keeper_id)
        sio = edges_post[
            (edges_post["edge_type"] == "schema_instance_of")
            & ((edges_post["dst"] == keeper_str) | (edges_post["src"] == keeper_str))
        ]
        assert len(sio) == 4, (
            f"pattern {pattern!r}: keeper {keeper_str[:8]} should have 4 "
            f"schema_instance_of edges (1 original + 3 redirected from dups), "
            f"got {len(sio)}"
        )


def test_cleanup_apply_keeper_is_oldest_per_pattern(tmp_path):
    from iai_mcp.migrate import cleanup_schema_duplicates

    store, patterns = _seed_dup_store(tmp_path, n_per_pattern=4, n_patterns=3)

    expected_keeper_ids = {}
    for p in patterns:
        recs = sorted(
            (
                r
                for r in store.all_records()
                if r.tier == "semantic"
                and f"pattern:{p}" in (r.tags or [])
            ),
            key=lambda r: r.created_at,
        )
        expected_keeper_ids[p] = recs[0].id

    cleanup_schema_duplicates(store, apply=True)

    for p, expected_id in expected_keeper_ids.items():
        survivors = [
            r
            for r in store.all_records()
            if r.tier == "semantic"
            and f"pattern:{p}" in (r.tags or [])
        ]
        assert len(survivors) == 1, (
            f"pattern {p!r}: expected exactly 1 keeper, got {len(survivors)}"
        )
        assert survivors[0].id == expected_id, (
            f"pattern {p!r}: keeper should be the oldest record "
            f"({str(expected_id)[:8]}), got {str(survivors[0].id)[:8]}"
        )


def test_cleanup_apply_skips_single_record_groups(tmp_path):
    from iai_mcp.migrate import cleanup_schema_duplicates

    store, _patterns = _seed_dup_store(
        tmp_path, n_per_pattern=4, n_patterns=2, extra_singletons=2
    )
    assert _count_semantic_pattern_records(store) == 10

    summary = cleanup_schema_duplicates(store, apply=True)
    assert summary["groups"] == 2
    assert summary["keepers"] == 2
    assert summary["pruned"] == 6

    assert _count_semantic_pattern_records(store) == 4


def test_cleanup_emits_schema_cleanup_run_event(tmp_path):
    from iai_mcp.events import query_events
    from iai_mcp.migrate import cleanup_schema_duplicates

    store, _patterns = _seed_dup_store(tmp_path, n_per_pattern=4, n_patterns=3)
    cleanup_schema_duplicates(store, apply=True)

    events = query_events(store, kind="schema_cleanup_run")
    assert len(events) >= 1, "schema_cleanup_run event must be emitted"
    e = events[0]
    payload = e["data"]
    for required_key in (
        "mode",
        "groups",
        "keepers",
        "pruned",
        "edges_reinforced",
        "snapshot_dir",
    ):
        assert required_key in payload, (
            f"schema_cleanup_run event payload missing '{required_key}'"
        )
    assert payload["mode"] == "apply"
    assert payload["groups"] == 3
    assert payload["keepers"] == 3
    assert payload["pruned"] == 9


def test_cleanup_apply_is_idempotent_on_second_run(tmp_path):
    from iai_mcp.migrate import cleanup_schema_duplicates

    store, _patterns = _seed_dup_store(tmp_path, n_per_pattern=4, n_patterns=3)

    summary1 = cleanup_schema_duplicates(store, apply=True)
    assert summary1["groups"] == 3
    assert summary1["pruned"] == 9

    summary2 = cleanup_schema_duplicates(store, apply=True)
    assert summary2["groups"] == 0, (
        f"second --apply must report 0 groups (idempotent), got {summary2}"
    )
    assert summary2["keepers"] == 0
    assert summary2["pruned"] == 0
    assert _count_semantic_pattern_records(store) == 3
    assert _count_pruned(store) == 9


def _run_cli(argv: list[str]) -> tuple[int, str]:
    import io
    from contextlib import redirect_stdout

    from iai_mcp.cli import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            code = main(argv)
        except SystemExit as exc:
            code = int(exc.code) if exc.code is not None else 0
    return code, buf.getvalue()


def test_cli_schema_cleanup_default_is_dry_run(tmp_path):
    _seed_dup_store(tmp_path, n_per_pattern=4, n_patterns=3)
    code, out = _run_cli(
        ["schema-cleanup", "--store-path", str(tmp_path)]
    )
    assert code == 0, f"CLI exited non-zero: {code!r}; output:\n{out}"
    assert "[dry-run]" in out, (
        f"default mode must report '[dry-run]' header; got:\n{out}"
    )
    assert "groups" in out
    assert "keepers" in out
    assert "pruned" in out


def test_cli_schema_cleanup_apply_runs_end_to_end(tmp_path):
    from iai_mcp.store import MemoryStore

    _seed_dup_store(tmp_path, n_per_pattern=4, n_patterns=3)
    code, out = _run_cli(
        ["schema-cleanup", "--apply", "--store-path", str(tmp_path)]
    )
    assert code == 0, f"CLI exited non-zero: {code!r}; output:\n{out}"
    assert "[apply]" in out
    assert "snapshot" in out.lower()

    store = MemoryStore(path=tmp_path)
    assert _count_semantic_pattern_records(store) == 3
    assert _count_pruned(store) == 9


def test_cli_schema_cleanup_dry_run_and_apply_mutually_exclusive(tmp_path):
    _seed_dup_store(tmp_path, n_per_pattern=4, n_patterns=2)
    code, _out = _run_cli(
        [
            "schema-cleanup",
            "--dry-run",
            "--apply",
            "--store-path",
            str(tmp_path),
        ]
    )
    assert code != 0, (
        "--dry-run and --apply must be mutually exclusive (argparse-rejected)"
    )


def test_cli_schema_cleanup_honours_store_path_argument(tmp_path):
    store_a_root = tmp_path / "a"
    store_b_root = tmp_path / "b"
    store_a_root.mkdir()
    store_b_root.mkdir()

    _seed_dup_store(store_a_root, n_per_pattern=4, n_patterns=2)

    code, out_b = _run_cli(
        ["schema-cleanup", "--store-path", str(store_b_root)]
    )
    assert code == 0
    assert "0" in out_b

    from iai_mcp.store import MemoryStore

    store_a = MemoryStore(path=store_a_root)
    assert _count_semantic_pattern_records(store_a) == 8


def test_cli_schema_cleanup_argparse_contract():
    from iai_mcp.cli import _build_parser

    p = _build_parser()
    ns = p.parse_args(["schema-cleanup", "--apply"])
    assert ns.cmd == "schema-cleanup"
    assert ns.apply is True
    assert ns.dry_run is False
    assert ns.store_path is None

    ns2 = p.parse_args(["schema-cleanup", "--dry-run"])
    assert ns2.dry_run is True
    assert ns2.apply is False

    ns3 = p.parse_args(["schema-cleanup", "--store-path", "/tmp/foo"])
    assert ns3.store_path == "/tmp/foo"
    assert ns3.apply is False
    assert ns3.dry_run is False

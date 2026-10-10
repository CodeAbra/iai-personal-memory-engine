"""Unit coverage for the on-disk stats aggregation glue (`gather_stats`)."""

from __future__ import annotations

import json
from pathlib import Path

from iai_mcp.tools import index_paths, index_symbols, index_xref
from iai_mcp.tools.session import capture_event
from iai_mcp.tools.stats import gather_stats


def test_code_index_lanes_report_segment_count_bytes_and_mtime(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    target = corpus / "alpha.py"
    target.write_text("def alpha():\n    return 1\n")

    store_root = tmp_path / "tools"
    symbol_root = store_root / "symbols"
    xref_root = store_root / "xref"

    index_paths([str(target)], store_root=store_root)
    index_symbols([str(target)], store_root=symbol_root)
    index_xref([str(target)], store_root=xref_root)

    stats = gather_stats(
        store_root=store_root,
        symbol_store_root=symbol_root,
        xref_store_root=xref_root,
        session_store_root=tmp_path / "absent-sessions",
        artifacts_root=tmp_path / "absent-artifacts",
    )

    for lane in ("whole_file", "symbol", "xref"):
        lane_stats = stats["code_index"][lane]
        assert lane_stats["segment_count"] == 1, lane
        assert lane_stats["total_bytes"] > 0, lane
        assert lane_stats["latest_modified"] is not None, lane


def test_stray_non_segment_file_in_a_lane_root_is_never_counted_as_a_segment(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    target = corpus / "alpha.py"
    target.write_text("def alpha():\n    return 1\n")

    store_root = tmp_path / "tools"
    index_paths([str(target)], store_root=store_root)
    # A filesystem marker file (e.g. a stray `.DS_Store`) sitting next to the
    # real segment file must not inflate the count -- the allowlist is by
    # segment-file suffix, not by excluding the manifest name.
    (store_root / ".DS_Store").write_text("not a segment")

    stats = gather_stats(
        store_root=store_root,
        symbol_store_root=tmp_path / "absent-symbol",
        xref_store_root=tmp_path / "absent-xref",
        session_store_root=tmp_path / "absent-sessions",
        artifacts_root=tmp_path / "absent-artifacts",
    )

    assert stats["code_index"]["whole_file"]["segment_count"] == 1


def test_sessions_report_count_total_events_and_per_category_totals(tmp_path: Path) -> None:
    sessions_root = tmp_path / "tools-root" / "sessions"

    capture_event(
        {
            "tool_name": "Edit",
            "file_path": "src/a.py",
            "session_id": "sess-a",
            "source_hook": "PostToolUse",
        },
        store_root=sessions_root,
    )
    capture_event(
        {
            "tool_name": "Bash",
            "command": "false",
            "is_error": True,
            "session_id": "sess-a",
            "source_hook": "PostToolUse",
        },
        store_root=sessions_root,
    )
    capture_event(
        {
            "tool_name": "Edit",
            "file_path": "src/b.py",
            "session_id": "sess-b",
            "source_hook": "PostToolUse",
        },
        store_root=sessions_root,
    )

    stats = gather_stats(
        store_root=tmp_path / "absent-index",
        symbol_store_root=tmp_path / "absent-symbol",
        xref_store_root=tmp_path / "absent-xref",
        session_store_root=sessions_root,
        artifacts_root=tmp_path / "absent-artifacts",
    )

    session_stats = stats["sessions"]
    assert session_stats["session_count"] == 2
    assert session_stats["total_events"] == 3
    assert session_stats["events_by_category"] == {"file": 2, "error": 1}


def test_executor_reports_run_count_from_artifact_subdirectories(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    (artifacts_root / "aaaa-1111-01").mkdir(parents=True)
    (artifacts_root / "bbbb-2222-02").mkdir(parents=True)
    # A stray file sitting alongside the run directories must not count as a run.
    (artifacts_root / "not-a-run.txt").write_text("stray")

    stats = gather_stats(
        store_root=tmp_path / "absent-index",
        symbol_store_root=tmp_path / "absent-symbol",
        xref_store_root=tmp_path / "absent-xref",
        session_store_root=tmp_path / "absent-sessions",
        artifacts_root=artifacts_root,
    )

    assert stats["executor"]["run_count"] == 2


def test_absent_stores_report_zeros_and_never_raise(tmp_path: Path) -> None:
    stats = gather_stats(
        store_root=tmp_path / "absent-index",
        symbol_store_root=tmp_path / "absent-symbol",
        xref_store_root=tmp_path / "absent-xref",
        session_store_root=tmp_path / "absent-sessions",
        artifacts_root=tmp_path / "absent-artifacts",
        savings_log=tmp_path / "absent-savings.jsonl",
    )

    for lane in ("whole_file", "symbol", "xref"):
        lane_stats = stats["code_index"][lane]
        assert lane_stats == {
            "segment_count": 0,
            "total_bytes": 0,
            "latest_modified": None,
        }, lane

    assert stats["sessions"] == {
        "session_count": 0,
        "total_events": 0,
        "events_by_category": {},
    }
    assert stats["executor"] == {"run_count": 0, "redirect_count": 0, "bytes_captured": 0}


def test_redirect_count_from_savings_log(tmp_path: Path) -> None:
    savings_log = tmp_path / "tools-savings.jsonl"
    savings_log.write_text(
        "\n".join(
            '{"ts": "2026-09-16T00:00:00Z", "cmd_class": "curl"}' for _ in range(4)
        )
        + "\n"
    )

    stats = gather_stats(
        store_root=tmp_path / "absent-index",
        symbol_store_root=tmp_path / "absent-symbol",
        xref_store_root=tmp_path / "absent-xref",
        session_store_root=tmp_path / "absent-sessions",
        artifacts_root=tmp_path / "absent-artifacts",
        savings_log=savings_log,
    )

    assert stats["executor"]["redirect_count"] == 4


def test_redirect_count_skips_blank_and_malformed_lines(tmp_path: Path) -> None:
    savings_log = tmp_path / "tools-savings.jsonl"
    savings_log.write_text(
        '{"ts": "a", "cmd_class": "curl"}\n'
        "\n"
        "not json\n"
        '{"ts": "b", "cmd_class": "gradle"}\n'
    )

    stats = gather_stats(
        store_root=tmp_path / "absent-index",
        symbol_store_root=tmp_path / "absent-symbol",
        xref_store_root=tmp_path / "absent-xref",
        session_store_root=tmp_path / "absent-sessions",
        artifacts_root=tmp_path / "absent-artifacts",
        savings_log=savings_log,
    )

    assert stats["executor"]["redirect_count"] == 2


def test_redirect_count_absent_log_is_zero(tmp_path: Path) -> None:
    stats = gather_stats(
        store_root=tmp_path / "absent-index",
        symbol_store_root=tmp_path / "absent-symbol",
        xref_store_root=tmp_path / "absent-xref",
        session_store_root=tmp_path / "absent-sessions",
        artifacts_root=tmp_path / "absent-artifacts",
        savings_log=tmp_path / "does-not-exist.jsonl",
    )

    assert stats["executor"]["redirect_count"] == 0


def test_bytes_captured_sums_meta_json_offsets_across_run_dirs(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    run1 = artifacts_root / "aaaa-1111-01"
    run2 = artifacts_root / "bbbb-2222-02"
    run1.mkdir(parents=True)
    run2.mkdir(parents=True)
    (run1 / "meta.json").write_text(json.dumps({"stdout_offset": 100, "stderr_offset": 20}))
    (run2 / "meta.json").write_text(json.dumps({"stdout_offset": 5, "stderr_offset": 0}))

    stats = gather_stats(
        store_root=tmp_path / "absent-index",
        symbol_store_root=tmp_path / "absent-symbol",
        xref_store_root=tmp_path / "absent-xref",
        session_store_root=tmp_path / "absent-sessions",
        artifacts_root=artifacts_root,
        savings_log=tmp_path / "absent-savings.jsonl",
    )

    assert stats["executor"]["bytes_captured"] == 125


def test_bytes_captured_absent_artifacts_root_is_zero(tmp_path: Path) -> None:
    stats = gather_stats(
        store_root=tmp_path / "absent-index",
        symbol_store_root=tmp_path / "absent-symbol",
        xref_store_root=tmp_path / "absent-xref",
        session_store_root=tmp_path / "absent-sessions",
        artifacts_root=tmp_path / "absent-artifacts",
        savings_log=tmp_path / "absent-savings.jsonl",
    )

    assert stats["executor"]["bytes_captured"] == 0


def test_savings_log_retention_bound_caps_redirect_count_and_trims_file(tmp_path: Path) -> None:
    savings_log = tmp_path / "tools-savings.jsonl"
    over_cap = 10005
    savings_log.write_text(
        "\n".join(f'{{"ts": "t{i}", "cmd_class": "curl"}}' for i in range(over_cap)) + "\n"
    )

    stats = gather_stats(
        store_root=tmp_path / "absent-index",
        symbol_store_root=tmp_path / "absent-symbol",
        xref_store_root=tmp_path / "absent-xref",
        session_store_root=tmp_path / "absent-sessions",
        artifacts_root=tmp_path / "absent-artifacts",
        savings_log=savings_log,
    )

    assert stats["executor"]["redirect_count"] == 10000
    remaining = [ln for ln in savings_log.read_text().splitlines() if ln.strip()]
    assert len(remaining) == 10000

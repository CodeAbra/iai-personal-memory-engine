"""Cross-layer staleness fence: source_watermark == derived_watermark
across episodic -> consolidation -> pack. Two policies, one hard-FAIL:
derived>source is the sole impossible/hard-FAIL condition; consolidation lag
is expected (WARN only past CONSOLIDATION_LAG_WARN_SEC); pack lag is always
flagged (any inequality -> lagging).
"""

from __future__ import annotations

import ast
import inspect

import pytest

from iai_mcp import watermark_fence as wf
from iai_mcp.doctor import _storage_checks as sc


# ---------------------------------------------------------------------------
# check_fence: the pure comparator
# ---------------------------------------------------------------------------

def test_consolidation_lag_beyond_threshold_is_lagging():
    episodic = "2026-09-06T12:00:00+00:00"
    consolidation = "2026-09-04T00:00:00+00:00"  # ~60h behind, past the 48h WARN
    result = wf.check_fence(episodic, consolidation, None)

    assert result.consolidation.status == "lagging"
    assert result.ok is True, "lag is a WARN condition, never a hard-FAIL"


def test_consolidation_lag_within_threshold_is_current():
    episodic = "2026-09-06T12:00:00+00:00"
    consolidation = "2026-09-05T12:00:00+00:00"  # 24h behind, within the 48h WARN
    result = wf.check_fence(episodic, consolidation, None)

    assert result.consolidation.status == "current"
    assert result.ok is True


def test_derived_ahead_of_source_pack_is_impossible():
    episodic = "2026-09-06T00:00:00+00:00"
    pack = "2026-09-07T00:00:00+00:00"  # a day AHEAD of its source
    result = wf.check_fence(episodic, None, pack)

    assert result.pack.status == "impossible"
    assert result.ok is False


def test_derived_ahead_of_source_consolidation_is_impossible():
    episodic = "2026-09-06T00:00:00+00:00"
    consolidation = "2026-09-07T00:00:00+00:00"
    result = wf.check_fence(episodic, consolidation, None)

    assert result.consolidation.status == "impossible"
    assert result.ok is False


def test_idle_store_equal_values_is_current():
    """After days of no cycle, an idle store where consolidation == episodic
    (nothing new to consolidate) must read as current, never as lagging."""
    ts = "2026-08-01T00:00:00+00:00"
    result = wf.check_fence(ts, ts, ts)

    assert result.consolidation.status == "current"
    assert result.pack.status == "current"
    assert result.ok is True


def test_missing_consolidation_watermark_is_never_run_not_fail():
    result = wf.check_fence("2026-09-06T00:00:00+00:00", None, None)

    assert result.consolidation.status == "never_run"
    assert result.ok is True


def test_pack_any_lag_is_flagged_lagging():
    episodic = "2026-09-06T12:00:00+00:00"
    pack = "2026-09-06T11:59:00+00:00"  # one minute behind -- still flagged
    result = wf.check_fence(episodic, None, pack)

    assert result.pack.status == "lagging"
    assert result.ok is True


def test_z_and_plus_zero_suffix_equal_values_normalize_to_current():
    episodic = "2026-09-06T12:00:00Z"
    pack = "2026-09-06T12:00:00+00:00"
    result = wf.check_fence(episodic, None, pack)

    assert result.pack.status == "current", (
        "a suffix-format-only difference must never flip the verdict"
    )


def test_check_fence_never_crashes_on_malformed_input():
    result = wf.check_fence("not-a-timestamp", "also-not-one", "still-not-one")

    assert result.ok is True
    assert result.pack.status in ("current", "never_run")
    assert result.consolidation.status in ("current", "never_run")


def test_real_readers_chained_into_check_fence_detect_consolidation_lag(tmp_path):
    """A consolidation layer trailing its source beyond the warn threshold
    is flagged lagging by the real reader wiring (not a stubbed value),
    while the fence overall stays ok -- a lag is a WARN, never a hard-FAIL."""
    from iai_mcp import store_watermark
    from iai_mcp.lifecycle_state import default_state, lifecycle_state_path, save_state

    store_root = tmp_path / "store"
    store_watermark.emit(store_root / "hippo", "2026-09-06T12:00:00+00:00")

    record = default_state()
    record["consolidated_watermark"] = "2026-09-03T00:00:00+00:00"  # ~3.5 days behind
    save_state(record, lifecycle_state_path(store_root))

    episodic = wf.read_episodic_watermark(store_root)
    consolidation = wf.read_consolidation_watermark(store_root)
    result = wf.check_fence(episodic, consolidation, None)

    assert result.consolidation.status == "lagging"
    assert result.ok is True, "lag is a WARN condition, never a hard-FAIL"


def test_no_daemon_package_import():
    """Daemon-independence: the module is importable and callable without
    pulling in the daemon package -- a both-storage-driver caller can drive
    it with fabricated timestamps and no live store."""
    source = inspect.getsource(wf)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("iai_mcp.daemon"), (
                    f"unexpected daemon import: {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith("iai_mcp.daemon"), (
                f"unexpected daemon import: {module}"
            )
    assert "iai_mcp.daemon" not in source


# ---------------------------------------------------------------------------
# read_pack_watermark: anchored line-1 parse
# ---------------------------------------------------------------------------

def test_read_pack_watermark_parses_leading_sentinel_line(tmp_path):
    cache_path = tmp_path / "pack.md"
    cache_path.write_text(
        "<!-- iai-mcp:source_watermark=2026-09-06T12:00:00+00:00 -->\n\nrest\n",
        encoding="utf-8",
    )
    assert wf.read_pack_watermark(cache_path) == "2026-09-06T12:00:00+00:00"


def test_read_pack_watermark_absent_file_returns_none(tmp_path):
    assert wf.read_pack_watermark(tmp_path / "missing.md") is None


def test_read_pack_watermark_no_sentinel_returns_none(tmp_path):
    cache_path = tmp_path / "pack.md"
    cache_path.write_text("## Identity\nsome content\n", encoding="utf-8")
    assert wf.read_pack_watermark(cache_path) is None


def test_read_pack_watermark_only_anchors_line_one(tmp_path):
    """A crafted cache cannot inject a fake sentinel anywhere but line 1."""
    cache_path = tmp_path / "pack.md"
    cache_path.write_text(
        "## Identity\n<!-- iai-mcp:source_watermark=2026-01-01T00:00:00Z -->\n",
        encoding="utf-8",
    )
    assert wf.read_pack_watermark(cache_path) is None


# ---------------------------------------------------------------------------
# doctor wrapper: check_jj_watermark_fence CheckResult escalation
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_store_env(monkeypatch):
    monkeypatch.delenv("IAI_MCP_STORE", raising=False)


def _stub_readers(monkeypatch, *, episodic, consolidation, pack):
    monkeypatch.setattr(wf, "read_episodic_watermark", lambda root: episodic)
    monkeypatch.setattr(wf, "read_consolidation_watermark", lambda root: consolidation)
    monkeypatch.setattr(wf, "read_pack_watermark", lambda cache_path=None: pack)


def test_doctor_fails_on_impossible_layer(monkeypatch):
    _stub_readers(
        monkeypatch,
        episodic="2026-09-06T00:00:00+00:00",
        consolidation="2026-09-07T00:00:00+00:00",  # ahead of source
        pack=None,
    )
    result = sc.check_jj_watermark_fence()
    assert result.status == "FAIL"
    assert result.passed is False


def test_doctor_warns_on_consolidation_lag(monkeypatch):
    _stub_readers(
        monkeypatch,
        episodic="2026-09-06T00:00:00+00:00",
        consolidation="2026-09-01T00:00:00+00:00",  # far past the 48h threshold
        pack=None,
    )
    result = sc.check_jj_watermark_fence()
    assert result.status == "WARN"
    assert result.passed is True


def test_doctor_passes_when_all_current(monkeypatch):
    ts = "2026-09-06T00:00:00+00:00"
    _stub_readers(monkeypatch, episodic=ts, consolidation=ts, pack=ts)
    result = sc.check_jj_watermark_fence()
    assert result.status == "PASS"
    assert result.passed is True


def test_doctor_pack_lag_alone_does_not_escalate(monkeypatch):
    """Pack lag is the common, expected steady state -- informational only."""
    _stub_readers(
        monkeypatch,
        episodic="2026-09-06T12:00:00+00:00",
        consolidation=None,
        pack="2026-09-06T11:00:00+00:00",
    )
    result = sc.check_jj_watermark_fence()
    assert result.status == "PASS"
    assert result.passed is True
    assert "lagging" in result.detail

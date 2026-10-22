"""Regression guard for the phys_footprint harness's hermeticity-by-construction
audit hook. Exercises the pure decision functions on synthetic path strings and
synthetic audit-event tuples only -- never reads the real store, never spawns
the worker, never installs a real `sys.addaudithook`.
"""
from __future__ import annotations

import os

from tests.rss_rca.phys_footprint_harness import (
    _build_allowlist_roots,
    _open_event_is_write,
    _path_is_allowlisted,
    _paths_for_audit_event,
)

# Synthetic top-level roots guaranteed to not exist and not be a symlink on
# any platform (unlike /tmp or /var, which are symlinks on macOS) -- keeps
# os.path.realpath a no-op so the test is platform-independent.
_TMP_ROOT = "/iai-test-tmproot"
_SYS_PREFIX = "/iai-test-venv"
_OUT_DIR = "/iai-test-outdir"
_ROOTS = (_TMP_ROOT, _SYS_PREFIX, _OUT_DIR)


# ---------------------------------------------------------------------------
# _path_is_allowlisted
# ---------------------------------------------------------------------------


def test_tmp_root_path_is_allowed() -> None:
    assert _path_is_allowlisted(f"{_TMP_ROOT}/store/records.dat", _ROOTS)


def test_sys_prefix_path_is_allowed() -> None:
    assert _path_is_allowlisted(f"{_SYS_PREFIX}/lib/python3.12/__pycache__/x.pyc", _ROOTS)


def test_out_dir_path_is_allowed() -> None:
    assert _path_is_allowlisted(f"{_OUT_DIR}/phases-n3000.jsonl", _ROOTS)


def test_exact_root_match_is_allowed() -> None:
    assert _path_is_allowlisted(_TMP_ROOT, _ROOTS)


def test_real_store_path_off_allowlist_is_a_violation() -> None:
    real_store = os.path.join("/iai-test-realhome", ".iai-mcp", "tools", "session-index.json")
    assert not _path_is_allowlisted(real_store, _ROOTS)


def test_arbitrary_off_allowlist_root_is_a_violation() -> None:
    assert not _path_is_allowlisted("/etc/passwd", _ROOTS)


def test_sibling_string_prefix_is_not_a_false_allow() -> None:
    # Shares a STRING prefix with the allowlisted root but is a DIFFERENT
    # directory -- a naive str.startswith(root) without the separator guard
    # would wrongly allow this.
    assert not _path_is_allowlisted(f"{_TMP_ROOT}-evil/leak", _ROOTS)


# ---------------------------------------------------------------------------
# _build_allowlist_roots
# ---------------------------------------------------------------------------


def test_build_allowlist_roots_includes_tmp_root_and_out_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    roots = _build_allowlist_roots(tmp_path, out_dir)
    assert os.path.realpath(tmp_path) in roots
    assert os.path.realpath(out_dir) in roots


def test_build_allowlist_roots_includes_hf_home_when_set(tmp_path, monkeypatch) -> None:
    hf_cache = tmp_path / "hf-cache"
    hf_cache.mkdir()
    monkeypatch.setenv("HF_HOME", str(hf_cache))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    roots = _build_allowlist_roots(tmp_path, None)
    assert os.path.realpath(hf_cache) in roots


def test_build_allowlist_roots_omits_out_dir_when_none(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    roots = _build_allowlist_roots(tmp_path, None)
    assert os.path.realpath(tmp_path) in roots


# ---------------------------------------------------------------------------
# _open_event_is_write -- write-intent filtering for the shared "open" event
# (builtin open()/io.open() AND os.open(), which reuses the same event).
# ---------------------------------------------------------------------------


def test_string_read_mode_is_not_a_write() -> None:
    assert not _open_event_is_write(("/some/path", "rb", None))


def test_string_write_mode_is_a_write() -> None:
    assert _open_event_is_write(("/some/path", "wb", None))


def test_string_read_plus_mode_is_a_write() -> None:
    assert _open_event_is_write(("/some/path", "r+", None))


def test_os_open_readonly_flags_is_not_a_write() -> None:
    # os.open() reuses the "open" event with mode=None and real int flags in
    # position 2 (confirmed empirically -- no separate "os.open" event).
    assert not _open_event_is_write(("/some/path", None, os.O_RDONLY))


def test_os_open_creat_flag_is_a_write() -> None:
    assert _open_event_is_write(("/some/path", None, os.O_RDONLY | os.O_CREAT))


def test_unrecognized_shape_defaults_to_write() -> None:
    assert _open_event_is_write(("/some/path",))


# ---------------------------------------------------------------------------
# _paths_for_audit_event -- path + dir_fd extraction per event shape.
# ---------------------------------------------------------------------------


def test_open_event_extracts_single_path() -> None:
    assert _paths_for_audit_event("open", ("/some/path", "rb", None)) == ["/some/path"]


def test_os_mkdir_extracts_path_ignoring_fd_when_none() -> None:
    assert _paths_for_audit_event("os.mkdir", ("/some/dir", 0o777, None)) == ["/some/dir"]


def test_os_remove_relative_path_joined_via_resolved_dir_fd(monkeypatch) -> None:
    import tests.rss_rca.phys_footprint_harness as harness

    monkeypatch.setattr(
        harness, "_resolve_dir_fd",
        lambda fd: "/iai-test-tmproot/hub" if fd == 42 else None,
    )
    result = _paths_for_audit_event("os.remove", ("model.safetensors", 42))
    assert result == ["/iai-test-tmproot/hub/model.safetensors"]


def test_os_remove_relative_path_with_unresolvable_dir_fd_stays_relative(monkeypatch) -> None:
    import tests.rss_rca.phys_footprint_harness as harness

    monkeypatch.setattr(harness, "_resolve_dir_fd", lambda fd: None)
    result = _paths_for_audit_event("os.remove", ("bare-name", 7))
    assert result == ["bare-name"]


def test_os_rename_extracts_both_src_and_dst() -> None:
    result = _paths_for_audit_event("os.rename", ("/a/src", "/b/dst", None, None))
    assert result == ["/a/src", "/b/dst"]


def test_fd_argument_never_resolves_to_a_path() -> None:
    assert _paths_for_audit_event("open", (7, "rb", None)) == []

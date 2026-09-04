"""Boot-time source stamp for the roots a running daemon pins, and the doctor
row that turns a silently stale sleep path into a visible warning."""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def isolated_daemon_state(tmp_path, monkeypatch):
    """Redirect daemon-state resolution into tmp_path.

    Without this the checks under test read and the fixture writes the
    operator's real ~/.iai-mcp/.daemon-state.json.
    """
    from iai_mcp import daemon_state

    store_dir = tmp_path / "store"
    store_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("IAI_MCP_STORE", str(store_dir))
    state_path = daemon_state.daemon_state_path()
    monkeypatch.setattr(daemon_state, "STATE_PATH", state_path)
    return state_path


@pytest.fixture
def fake_pinned_root(tmp_path, monkeypatch):
    from iai_mcp import code_stamp

    root = tmp_path / "pinned"
    root.mkdir()
    (root / "_a.py").write_text("A = 1\n", encoding="utf-8")
    (root / "_b.py").write_text("B = 2\n", encoding="utf-8")
    monkeypatch.setattr(code_stamp, "pinned_source_roots", lambda: [root])
    return root


def _write_state(state_path, payload: dict) -> None:
    state_path.write_text(json.dumps(payload), encoding="utf-8")


def test_stamp_is_content_addressed_not_mtime(fake_pinned_root) -> None:
    from iai_mcp.code_stamp import compute_code_stamp

    first = compute_code_stamp()
    assert first["files"] == 2

    (fake_pinned_root / "_a.py").write_text("A = 1\n", encoding="utf-8")
    assert compute_code_stamp()["digest"] == first["digest"]

    (fake_pinned_root / "_a.py").write_text("A = 2\n", encoding="utf-8")
    assert compute_code_stamp()["digest"] != first["digest"]


def test_stamp_covers_new_and_nested_files(fake_pinned_root) -> None:
    from iai_mcp.code_stamp import compute_code_stamp

    before = compute_code_stamp()
    nested = fake_pinned_root / "sub"
    nested.mkdir()
    (nested / "_c.py").write_text("C = 3\n", encoding="utf-8")
    after = compute_code_stamp()

    assert after["files"] == before["files"] + 1
    assert after["digest"] != before["digest"]


def test_real_pinned_roots_resolve_to_the_sleep_pipeline_package() -> None:
    from iai_mcp.code_stamp import compute_code_stamp, pinned_source_roots

    roots = pinned_source_roots()
    assert roots and all(root.is_dir() for root in roots)
    assert any((root / "_knob_tune.py").exists() for root in roots)
    assert compute_code_stamp()["files"] > 0


def test_row_passes_when_no_daemon_booted(isolated_daemon_state, fake_pinned_root) -> None:
    from iai_mcp.doctor import check_ff_daemon_code_current

    _write_state(isolated_daemon_state, {"fsm_state": "WAKE"})

    result = check_ff_daemon_code_current()

    assert result.status == "PASS"
    assert result.passed is True


def test_row_passes_when_boot_stamp_matches_disk(
    isolated_daemon_state, fake_pinned_root
) -> None:
    from iai_mcp.code_stamp import compute_code_stamp
    from iai_mcp.doctor import check_ff_daemon_code_current

    _write_state(
        isolated_daemon_state,
        {"daemon_pid": 4242, "code_stamp": compute_code_stamp()},
    )

    result = check_ff_daemon_code_current()

    assert result.status == "PASS"


def test_row_warns_when_source_changed_after_boot(
    isolated_daemon_state, fake_pinned_root
) -> None:
    from iai_mcp.code_stamp import compute_code_stamp
    from iai_mcp.doctor import check_ff_daemon_code_current

    _write_state(
        isolated_daemon_state,
        {"daemon_pid": 4242, "code_stamp": compute_code_stamp()},
    )
    (fake_pinned_root / "_a.py").write_text("A = 99\n", encoding="utf-8")

    result = check_ff_daemon_code_current()

    assert result.status == "WARN"
    assert result.passed is True
    assert "older than the source on disk" in result.detail


def test_row_warns_when_daemon_booted_from_another_root(
    isolated_daemon_state, fake_pinned_root, tmp_path
) -> None:
    from iai_mcp.code_stamp import compute_code_stamp
    from iai_mcp.doctor import check_ff_daemon_code_current

    stamp = compute_code_stamp()
    stamp["roots"] = [str(tmp_path / "elsewhere")]
    _write_state(isolated_daemon_state, {"daemon_pid": 4242, "code_stamp": stamp})

    result = check_ff_daemon_code_current()

    assert result.status == "WARN"
    assert "different sleep-path source root" in result.detail


def test_row_warns_when_daemon_booted_without_a_stamp(
    isolated_daemon_state, fake_pinned_root
) -> None:
    from iai_mcp.doctor import check_ff_daemon_code_current

    _write_state(isolated_daemon_state, {"daemon_pid": 4242})

    result = check_ff_daemon_code_current()

    assert result.status == "WARN"
    assert "without a code stamp" in result.detail


def test_stale_code_row_never_plans_a_repair(
    isolated_daemon_state, fake_pinned_root
) -> None:
    """The only remedy is a daemon stop+start; the unattended reflex must
    never take that decision."""
    from iai_mcp.code_stamp import compute_code_stamp
    from iai_mcp.doctor import _plan_repair_actions, check_ff_daemon_code_current

    _write_state(
        isolated_daemon_state,
        {"daemon_pid": 4242, "code_stamp": compute_code_stamp()},
    )
    (fake_pinned_root / "_b.py").write_text("B = 99\n", encoding="utf-8")

    result = check_ff_daemon_code_current()

    assert _plan_repair_actions([result]) == []


def test_run_diagnosis_includes_the_row(isolated_daemon_state, fake_pinned_root) -> None:
    """Registration coverage: dropping the row from run_diagnosis makes the
    warning invisible, which is the failure mode this row exists to prevent."""
    from iai_mcp import doctor

    _write_state(isolated_daemon_state, {"fsm_state": "WAKE"})

    names = {r.name for r in doctor.run_diagnosis(fetch_update=False)}

    assert "(ff) daemon sleep-path code current" in names
    assert "check_ff_daemon_code_current" in doctor.__all__


def test_status_reply_carries_the_boot_stamp() -> None:
    import asyncio

    from iai_mcp.concurrency import _dispatch_socket_request

    state = {"fsm_state": "WAKE", "code_stamp": {"digest": "abc", "roots": [], "files": 3}}
    resp = asyncio.run(_dispatch_socket_request({"type": "status"}, None, state))

    assert resp["code_stamp"] == {"digest": "abc", "roots": [], "files": 3}


def test_status_reply_coerces_a_wrong_typed_stamp_to_none() -> None:
    import asyncio

    from iai_mcp.concurrency import _dispatch_socket_request

    resp = asyncio.run(
        _dispatch_socket_request({"type": "status"}, None, {"code_stamp": "not-a-dict"})
    )

    assert resp["code_stamp"] is None

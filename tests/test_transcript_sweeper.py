"""`iai-mcp cowork install`/`uninstall` install and remove the background
sweeper (a scheduled process, never a task inside the memory daemon), and
`sweep_once` never double-captures a transcript the Stop hook already
captured today."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.fixture()
def cowork_env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("APPDATA", str(home / "AppData" / "Roaming"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setattr(
        "iai_mcp.cli._capture._patch_claude_desktop_config",
        lambda action: "Claude Desktop: stubbed",
    )
    return {"home": home}


def _snapshot(root: Path) -> dict:
    if not root.is_dir():
        return {}
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _record_loads(monkeypatch, calls: list) -> None:
    import iai_mcp.cli._cowork as cowork_mod

    def _fake_load(paths):
        calls.append(paths)

    monkeypatch.setattr(cowork_mod, "_issue_sweeper_load", _fake_load)


def _record_unloads(monkeypatch, calls: list) -> None:
    import iai_mcp.cli._cowork as cowork_mod

    def _fake_unload(paths):
        calls.append(paths)

    monkeypatch.setattr(cowork_mod, "_issue_sweeper_unload", _fake_unload)


# ---------------------------------------------------------------------------
# Task 1: install writes and loads a scheduled sweeper.
# ---------------------------------------------------------------------------


def test_install_writes_launchd_unit_and_flag(cowork_env, monkeypatch):
    from iai_mcp.cli._cowork import cmd_cowork_install, _sweeper_unit_paths

    monkeypatch.setattr(sys, "platform", "darwin")
    load_calls = []
    _record_loads(monkeypatch, load_calls)

    rc = cmd_cowork_install(argparse.Namespace())
    assert rc == 0

    paths = _sweeper_unit_paths(cowork_env["home"])
    unit = paths["unit"]
    assert unit.exists()
    content = unit.read_text(encoding="utf-8")
    assert "<key>StartInterval</key>" in content
    assert "<integer>120</integer>" in content
    assert "transcript-sweep" in content
    assert "run" in content

    flag = cowork_env["home"] / ".iai-mcp" / ".cowork-sweep-enabled"
    assert flag.exists()

    assert len(load_calls) == 1


def test_install_writes_systemd_service_and_timer(cowork_env, monkeypatch):
    from iai_mcp.cli._cowork import cmd_cowork_install, _sweeper_unit_paths

    monkeypatch.setattr(sys, "platform", "linux")
    load_calls = []
    _record_loads(monkeypatch, load_calls)

    rc = cmd_cowork_install(argparse.Namespace())
    assert rc == 0

    paths = _sweeper_unit_paths(cowork_env["home"])
    assert paths["service"].exists()
    assert paths["timer"].exists()
    timer_content = paths["timer"].read_text(encoding="utf-8")
    assert "OnUnitActiveSec=120" in timer_content
    service_content = paths["service"].read_text(encoding="utf-8")
    assert "transcript-sweep" in service_content
    assert len(load_calls) == 1


def test_install_is_idempotent_by_content_hash(cowork_env, monkeypatch):
    from iai_mcp.cli._cowork import cmd_cowork_install, _sweeper_unit_paths

    monkeypatch.setattr(sys, "platform", "darwin")
    _record_loads(monkeypatch, [])

    assert cmd_cowork_install(argparse.Namespace()) == 0
    unit = _sweeper_unit_paths(cowork_env["home"])["unit"]
    first_hash = hashlib.sha256(unit.read_bytes()).hexdigest()

    assert cmd_cowork_install(argparse.Namespace()) == 0
    second_hash = hashlib.sha256(unit.read_bytes()).hexdigest()
    assert first_hash == second_hash


def test_install_writes_nothing_under_desktop_tree(cowork_env, monkeypatch):
    from iai_mcp.cli._cowork import cmd_cowork_install

    monkeypatch.setattr(sys, "platform", "darwin")
    _record_loads(monkeypatch, [])

    desktop_root = cowork_env["home"] / "Library" / "Application Support" / "Claude"
    desktop_root.mkdir(parents=True)
    (desktop_root / "cowork_settings.json").write_text("{}", encoding="utf-8")
    before = _snapshot(desktop_root)

    assert cmd_cowork_install(argparse.Namespace()) == 0

    after = _snapshot(desktop_root)
    assert before == after


def test_install_declines_on_unsupported_platform(cowork_env, monkeypatch):
    from iai_mcp.cli._cowork import cmd_cowork_install

    monkeypatch.setattr(sys, "platform", "cygwin")
    monkeypatch.setattr("os.name", "posix")

    rc = cmd_cowork_install(argparse.Namespace())
    assert rc != 0

    launchd = cowork_env["home"] / "Library" / "LaunchAgents"
    systemd = cowork_env["home"] / ".config" / "systemd"
    assert not launchd.exists()
    assert not systemd.exists()


def test_no_daemon_env_var_governs_sweep_cadence():
    import iai_mcp.cli._cowork as cowork_mod
    import iai_mcp.transcript_sweep as sweep_mod

    for mod in (cowork_mod, sweep_mod):
        source = inspect.getsource(mod)
        assert "TRANSCRIPT_SWEEP_SEC" not in source


def test_materialize_plugin_write_path_is_gone():
    import iai_mcp.cli._cowork as cowork_mod

    assert not hasattr(cowork_mod, "_materialize_plugin")
    assert not hasattr(cowork_mod, "_build_hooks_json")
    assert not hasattr(cowork_mod, "_hook_entry")


def test_install_source_never_writes_marketplace_registration():
    import iai_mcp.cli._cowork as cowork_mod

    source = inspect.getsource(cowork_mod.cmd_cowork_install)
    assert "extraKnownMarketplaces" not in source
    assert "_patch_cowork_settings" not in source


# ---------------------------------------------------------------------------
# Task 2: uninstall reverses all five categories; the probe and the
# plugin-registration write path are retired.
# ---------------------------------------------------------------------------

_ACCT = "11111111-2222-4333-8444-555555555555"
_DEVICE = "66666666-7777-4888-9999-aaaaaaaaaaaa"

_FOREIGN_SETTINGS = {
    "extraKnownMarketplaces": {
        "knowledge-work-plugins": {
            "source": {"source": "github", "repo": "anthropics/knowledge-work-plugins"}
        }
    },
    "enabledPlugins": {"productivity@knowledge-work-plugins": True},
}


def _seed_cowork_home(home: Path) -> Path:
    root = home / "Library" / "Application Support" / "Claude" / "local-agent-mode-sessions"
    dev = root / _ACCT / _DEVICE
    dev.mkdir(parents=True)
    return dev


def test_uninstall_round_trip_restores_pristine_snapshot(cowork_env, monkeypatch):
    from iai_mcp.cli._cowork import cmd_cowork_install, cmd_cowork_uninstall

    monkeypatch.setattr(sys, "platform", "darwin")
    _record_loads(monkeypatch, [])
    _record_unloads(monkeypatch, [])

    home = cowork_env["home"]
    dev = _seed_cowork_home(home)
    (dev / "cowork_settings.json").write_text(
        json.dumps(_FOREIGN_SETTINGS), encoding="utf-8"
    )

    before = _snapshot(home)

    assert cmd_cowork_install(argparse.Namespace()) == 0
    assert cmd_cowork_uninstall(argparse.Namespace()) == 0

    after = _snapshot(home)
    assert after == before


def test_uninstall_strips_legacy_registration_never_installed_by_current_version(
    cowork_env, monkeypatch
):
    from iai_mcp.cli._cowork import (
        MARKETPLACE_NAME, PLUGIN_NAME, cmd_cowork_uninstall,
    )

    monkeypatch.setattr(sys, "platform", "darwin")
    _record_unloads(monkeypatch, [])

    home = cowork_env["home"]
    dev = _seed_cowork_home(home)
    legacy_key = f"{PLUGIN_NAME}@{MARKETPLACE_NAME}"
    legacy_settings = {
        "extraKnownMarketplaces": {
            MARKETPLACE_NAME: {
                "source": {"source": "directory", "path": "/some/old/install/path"}
            }
        },
        "enabledPlugins": {legacy_key: True},
    }
    (dev / "cowork_settings.json").write_text(json.dumps(legacy_settings), encoding="utf-8")

    # current version never ran install -- no unit, no flag -- uninstall
    # must still strip what an OLDER version wrote.
    assert cmd_cowork_uninstall(argparse.Namespace()) == 0

    data = json.loads((dev / "cowork_settings.json").read_text())
    assert MARKETPLACE_NAME not in data["extraKnownMarketplaces"]
    assert legacy_key not in data["enabledPlugins"]


def test_uninstall_preserves_foreign_keys_byte_identical(cowork_env, monkeypatch):
    from iai_mcp.cli._cowork import cmd_cowork_uninstall

    monkeypatch.setattr(sys, "platform", "darwin")
    _record_unloads(monkeypatch, [])

    home = cowork_env["home"]
    dev = _seed_cowork_home(home)
    (dev / "cowork_settings.json").write_text(
        json.dumps(_FOREIGN_SETTINGS), encoding="utf-8"
    )

    assert cmd_cowork_uninstall(argparse.Namespace()) == 0

    data = json.loads((dev / "cowork_settings.json").read_text())
    assert data == _FOREIGN_SETTINGS


def test_uninstall_on_clean_machine_is_noop(cowork_env, monkeypatch):
    from iai_mcp.cli._cowork import cmd_cowork_uninstall

    monkeypatch.setattr(sys, "platform", "darwin")
    _record_unloads(monkeypatch, [])

    home = cowork_env["home"]
    before = _snapshot(home)
    rc = cmd_cowork_uninstall(argparse.Namespace())
    after = _snapshot(home)

    assert rc == 0
    assert after == before


def test_uninstall_second_run_is_noop(cowork_env, monkeypatch):
    from iai_mcp.cli._cowork import cmd_cowork_install, cmd_cowork_uninstall

    monkeypatch.setattr(sys, "platform", "darwin")
    _record_loads(monkeypatch, [])
    _record_unloads(monkeypatch, [])

    home = cowork_env["home"]
    dev = _seed_cowork_home(home)
    (dev / "cowork_settings.json").write_text(
        json.dumps(_FOREIGN_SETTINGS), encoding="utf-8"
    )

    assert cmd_cowork_install(argparse.Namespace()) == 0
    assert cmd_cowork_uninstall(argparse.Namespace()) == 0
    after_first = _snapshot(home)

    assert cmd_cowork_uninstall(argparse.Namespace()) == 0
    after_second = _snapshot(home)

    assert after_second == after_first


def test_uninstall_removes_retired_reachability_check_artifacts(cowork_env, monkeypatch):
    from iai_mcp.cli._cowork import cmd_cowork_uninstall

    monkeypatch.setattr(sys, "platform", "darwin")
    _record_unloads(monkeypatch, [])

    home = cowork_env["home"]
    dev = _seed_cowork_home(home)

    retired_marketplace = "local-desktop-app-uploads"
    retired_plugin = "iai-mcp-cowork-probe"
    marketplace_dir = dev / "cowork_plugins" / "marketplaces" / retired_marketplace
    plugin_dir = marketplace_dir / retired_plugin
    (plugin_dir / "hooks").mkdir(parents=True)
    (plugin_dir / "hooks" / f"{retired_plugin}.sh").write_text("#!/bin/sh\n", encoding="utf-8")

    plugin_key = f"{retired_plugin}@{retired_marketplace}"
    settings = {
        "extraKnownMarketplaces": {
            retired_marketplace: {"source": {"source": "directory", "path": str(marketplace_dir)}}
        },
        "enabledPlugins": {plugin_key: True},
    }
    (dev / "cowork_settings.json").write_text(json.dumps(settings), encoding="utf-8")

    host_tree = home / ".iai-mcp" / "cowork-probe"
    (host_tree / "hooks").mkdir(parents=True)
    (host_tree / "hooks" / f"{retired_plugin}.sh").write_text("#!/bin/sh\n", encoding="utf-8")

    assert cmd_cowork_uninstall(argparse.Namespace()) == 0

    assert not marketplace_dir.exists()
    assert not host_tree.exists()
    data = json.loads((dev / "cowork_settings.json").read_text())
    assert retired_marketplace not in data["extraKnownMarketplaces"]
    assert plugin_key not in data["enabledPlugins"]


def test_uninstall_warns_about_unreversed_retired_hook_mutation(cowork_env, monkeypatch, capsys):
    from iai_mcp.cli._cowork import cmd_cowork_uninstall

    monkeypatch.setattr(sys, "platform", "darwin")
    _record_unloads(monkeypatch, [])

    home = cowork_env["home"]
    dev = _seed_cowork_home(home)
    (dev / "cowork_settings.json").write_text("{}", encoding="utf-8")

    settings_path = dev / "agent" / "local_ditto_alice" / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {
                            "matcher": "startup|resume|clear|compact",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": 'bash "/some/host/path/iai-mcp-cowork-probe.sh"',
                                    "timeout": 15,
                                }
                            ],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    assert cmd_cowork_uninstall(argparse.Namespace()) == 0
    out = capsys.readouterr().out
    assert "warning:" in out
    assert str(settings_path) in out
    assert "iai-mcp-cowork-probe" in out
    # The unreversible entry is never silently removed or overwritten.
    data = json.loads(settings_path.read_text())
    assert data["hooks"]["SessionStart"]


def test_uninstall_reports_nothing_for_a_clean_agent_settings_file(cowork_env, monkeypatch, capsys):
    from iai_mcp.cli._cowork import cmd_cowork_uninstall

    monkeypatch.setattr(sys, "platform", "darwin")
    _record_unloads(monkeypatch, [])

    home = cowork_env["home"]
    dev = _seed_cowork_home(home)
    (dev / "cowork_settings.json").write_text("{}", encoding="utf-8")

    settings_path = dev / "agent" / "local_ditto_alice" / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(json.dumps({"hooks": {"SessionStart": []}}), encoding="utf-8")

    assert cmd_cowork_uninstall(argparse.Namespace()) == 0
    out = capsys.readouterr().out
    assert "iai-mcp-cowork-probe" not in out


def test_no_probe_or_retired_helper_symbols_remain():
    import iai_mcp.cli._cowork as cowork_mod

    function_names = [
        n for n, f in vars(cowork_mod).items() if inspect.isfunction(f)
    ]
    for name in function_names:
        assert "probe" not in name.lower(), name
    assert not hasattr(cowork_mod, "cmd_cowork_probe")
    assert not hasattr(cowork_mod, "PROBE_CHANNELS")
    assert not hasattr(cowork_mod, "_materialize_plugin")
    assert not hasattr(cowork_mod, "_plugin_version")


def test_no_stale_backup_sidecar_read_write_or_unlink():
    source = inspect.getsource(__import__(
        "iai_mcp.cli._cowork", fromlist=["_"]
    ))
    for line in source.splitlines():
        if "iai-backup" in line:
            assert (
                "read_bytes" not in line
                and "write_bytes" not in line
                and "unlink" not in line
            ), line


def test_cowork_probe_files_removed_from_tree():
    import iai_mcp
    from pathlib import Path as _P

    pkg_root = _P(iai_mcp.__file__).resolve().parent
    hook_script = pkg_root / "_deploy" / "hooks" / "iai-mcp-cowork-probe.sh"
    assert not hook_script.exists()


# ---------------------------------------------------------------------------
# Task 3: a shared-home transcript seen by both the Stop hook and the sweep
# is captured exactly once.
# ---------------------------------------------------------------------------


@pytest.fixture
def sweep_store_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-transcript-sweeper-passphrase")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))
    import keyring.core

    keyring.core._keyring_backend = None
    yield tmp_path
    keyring.core._keyring_backend = None


def _write_transcript(path: Path, lines: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for obj in lines:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _user_line(text: str, *, uuid: str, ts: str) -> dict:
    return {
        "type": "user",
        "uuid": uuid,
        "timestamp": ts,
        "message": {"role": "user", "content": text},
    }


def _assistant_line(text: str, *, uuid: str, ts: str) -> dict:
    return {
        "type": "assistant",
        "uuid": uuid,
        "timestamp": ts,
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


def _open_store():
    from iai_mcp.store import MemoryStore

    return MemoryStore()


def _write_turn_capture_receipt(iai_home: Path, *, session_id: str, when: datetime) -> None:
    log_dir = iai_home / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"turn-capture-{when:%Y-%m-%d}.log"
    line = f"{when:%Y-%m-%dT%H:%M:%SZ} session={session_id} channel=settings\n"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line)


def test_sweep_skips_session_with_same_day_capture_hooks_receipt(sweep_store_env):
    from iai_mcp.transcript_sweep import _sweep_state_path, sweep_once

    session_id = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
    claude_root = sweep_store_env / ".claude"
    transcript_path = claude_root / "projects" / "-Users-alice-guard" / f"{session_id}.jsonl"
    _write_transcript(
        transcript_path,
        [_user_line(
            "guard test nonce long enough to be a real turn",
            uuid="u-1", ts="2026-09-02T20:00:00.000000+00:00",
        )],
    )

    iai_home = sweep_store_env / ".iai-mcp"
    _write_turn_capture_receipt(
        iai_home, session_id=session_id, when=datetime.now(timezone.utc),
    )

    summary = sweep_once(roots=[claude_root])

    assert summary["sessions_staged"] == 0
    assert summary["lines_staged"] == 0
    assert not _sweep_state_path(session_id).exists()


def test_dedup_collapses_double_staging_to_one_row_per_turn(sweep_store_env):
    from iai_mcp.capture import drain_capture_backlog, write_deferred_captures
    from iai_mcp.transcript_sweep import sweep_once

    session_id = "dedup-session-0001"
    claude_root = sweep_store_env / ".claude"
    transcript_path = claude_root / "projects" / "-Users-alice-dedup" / f"{session_id}.jsonl"
    nonce = "dedup double capture proof nonce eight eight"
    _write_transcript(
        transcript_path,
        [
            _user_line(nonce, uuid="u-1", ts="2026-09-02T20:00:00.000000+00:00"),
            _assistant_line(
                "acknowledged the dedup nonce in full",
                uuid="a-1", ts="2026-09-02T20:00:01.000000+00:00",
            ),
        ],
    )

    # The Stop-hook path: stages the whole transcript once, directly.
    write_deferred_captures(
        session_id=session_id, transcript_path=transcript_path,
        cwd=str(transcript_path.parent),
    )

    # The sweep path, guard disabled: stages the SAME transcript
    # independently, proving the store-side dedup carries the guarantee
    # even without the skip guard's help.
    summary = sweep_once(roots=[claude_root], skip_receipted_sessions=False)
    assert summary["sessions_staged"] == 1

    store = _open_store()
    try:
        drain_capture_backlog(store)
        turns = store.recent_user_turns(50, session_id=session_id)
        matches = [t for t in turns if nonce in (t.literal_surface or "")]
        assert len(matches) == 1, (
            f"expected exactly one row per turn; got {len(matches)}: {matches!r}"
        )
    finally:
        store.close()


def test_sweep_twice_stages_nothing_new_and_inserts_no_new_row(sweep_store_env):
    from iai_mcp.capture import drain_capture_backlog
    from iai_mcp.transcript_sweep import sweep_once

    session_id = "twice-session-0001"
    claude_root = sweep_store_env / ".claude"
    transcript_path = claude_root / "projects" / "-Users-alice-twice" / f"{session_id}.jsonl"
    nonce = "sweep twice no growth nonce seven seven seven"
    _write_transcript(
        transcript_path,
        [_user_line(nonce, uuid="u-1", ts="2026-09-02T20:00:00.000000+00:00")],
    )

    summary1 = sweep_once(roots=[claude_root])
    assert summary1["sessions_staged"] == 1

    store = _open_store()
    try:
        drain_capture_backlog(store)
        first = [
            t for t in store.recent_user_turns(50, session_id=session_id)
            if nonce in (t.literal_surface or "")
        ]
        assert len(first) == 1

        summary2 = sweep_once(roots=[claude_root])
        assert summary2["sessions_staged"] == 0
        assert summary2["lines_staged"] == 0

        drain_capture_backlog(store)
        second = [
            t for t in store.recent_user_turns(50, session_id=session_id)
            if nonce in (t.literal_surface or "")
        ]
        assert len(second) == len(first)
    finally:
        store.close()


def test_sweep_catches_session_with_no_receipt(sweep_store_env):
    from iai_mcp.capture import drain_capture_backlog
    from iai_mcp.transcript_sweep import sweep_once

    session_id = "catch-session-0001"
    claude_root = sweep_store_env / ".claude"
    transcript_path = claude_root / "projects" / "-Users-alice-catch" / f"{session_id}.jsonl"
    nonce = "sweep catches an uncaptured session nonce six"
    _write_transcript(
        transcript_path,
        [_user_line(nonce, uuid="u-1", ts="2026-09-02T20:00:00.000000+00:00")],
    )

    summary = sweep_once(roots=[claude_root])
    assert summary["sessions_staged"] == 1

    store = _open_store()
    try:
        drain_capture_backlog(store)
        turns = store.recent_user_turns(50, session_id=session_id)
        assert any(nonce in (t.literal_surface or "") for t in turns)
    finally:
        store.close()


def test_guard_reuses_capture_hooks_receipt_reader_not_a_reimplementation():
    import iai_mcp.transcript_sweep as mod

    source = inspect.getsource(mod)
    assert "_read_hook_receipts" in source

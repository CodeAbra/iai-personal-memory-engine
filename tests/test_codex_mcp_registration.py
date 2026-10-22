"""Codex MCP registration: surgical upsert of only [mcp_servers.iai-tools]
in config.toml. The fixtures below mirror the shape of a real Codex
config.toml -- a foreign server with its own env sub-table, a
shell-environment-policy-shaped table, and a top-level scalar key, all
holding fake tokens -- to prove the writer never touches anything
outside its own table.
"""
from __future__ import annotations

import tomllib

import pytest

from iai_mcp.cli._codex_mcp import install_codex_mcp, uninstall_codex_mcp

_FOREIGN_TOKEN = "sk-fake-foreign-0123456789abcdef"
_SHELL_TOKEN = "sk-fake-shell-policy-token-zzzz"

_FIXTURE = f'''history_max_bytes = 1048576

[mcp_servers.other-server]
command = "node"
args = ["/opt/other/server.js"]

[mcp_servers.other-server.env]
API_TOKEN = "{_FOREIGN_TOKEN}"

[shell_environment_policy.set]
SECRET_TOKEN = "{_SHELL_TOKEN}"

[model]
provider = "openai"
'''


@pytest.fixture()
def codex_home(tmp_path, monkeypatch):
    home = tmp_path / "codex-home"
    monkeypatch.setenv("IAI_MCP_CODEX_HOME", str(home))
    return home


@pytest.fixture()
def fake_wrapper(tmp_path, monkeypatch):
    wrapper = tmp_path / "tools-index.js"
    wrapper.write_text("// fake tools wrapper\n", encoding="utf-8")
    monkeypatch.setenv("IAI_MCP_TOOLS_WRAPPER_PATH", str(wrapper))
    return wrapper


class TestInstall:
    def test_adds_table_with_env_subtable_to_absent_config(
        self, codex_home, fake_wrapper
    ):
        assert install_codex_mcp() == 0
        text = (codex_home / "config.toml").read_text(encoding="utf-8")
        data = tomllib.loads(text)
        entry = data["mcp_servers"]["iai-tools"]
        assert entry["command"] == "node"
        assert entry["args"] == [str(fake_wrapper)]
        assert "env" in entry
        assert entry["env"]["IAI_MCP_PYTHON"]

    def test_second_install_is_byte_identical(self, codex_home, fake_wrapper):
        install_codex_mcp()
        first = (codex_home / "config.toml").read_bytes()
        install_codex_mcp()
        second = (codex_home / "config.toml").read_bytes()
        assert first == second

    def test_foreign_content_survives_byte_for_byte_on_fresh_install(
        self, codex_home, fake_wrapper
    ):
        codex_home.mkdir(parents=True)
        cfg = codex_home / "config.toml"
        cfg.write_text(_FIXTURE, encoding="utf-8")

        install_codex_mcp()

        new_text = cfg.read_text(encoding="utf-8")
        assert new_text.startswith(_FIXTURE)
        assert _FOREIGN_TOKEN in new_text
        assert _SHELL_TOKEN in new_text
        assert "history_max_bytes = 1048576" in new_text
        assert '[mcp_servers.other-server]' in new_text
        assert '[shell_environment_policy.set]' in new_text

    def test_no_backup_file_created(self, codex_home, fake_wrapper):
        codex_home.mkdir(parents=True)
        cfg = codex_home / "config.toml"
        cfg.write_text(_FIXTURE, encoding="utf-8")
        install_codex_mcp()
        siblings = {p.name for p in codex_home.iterdir()}
        assert siblings == {"config.toml"}

    def test_replaces_only_our_table_when_present_between_foreign_tables(
        self, codex_home, fake_wrapper
    ):
        codex_home.mkdir(parents=True)
        cfg = codex_home / "config.toml"
        stale_block = (
            "[mcp_servers.iai-tools]\n"
            'command = "node"\n'
            'args = ["/stale/path.js"]\n'
            "\n"
            "[mcp_servers.iai-tools.env]\n"
            'IAI_MCP_PYTHON = "/stale/python"\n'
        )
        before_table = (
            "[mcp_servers.before-server]\n"
            'command = "node"\n'
            f'args = ["/opt/before.js"]\n\n'
        )
        after_table = (
            "\n[mcp_servers.after-server]\n"
            'command = "node"\n'
            f'args = ["/opt/after.js"]\n'
        )
        cfg.write_text(before_table + stale_block + after_table, encoding="utf-8")

        install_codex_mcp()

        new_text = cfg.read_text(encoding="utf-8")
        assert new_text.startswith(before_table)
        assert new_text.endswith(after_table)
        assert "/stale/path.js" not in new_text
        assert "/stale/python" not in new_text
        data = tomllib.loads(new_text)
        assert data["mcp_servers"]["iai-tools"]["args"] == [str(fake_wrapper)]
        assert data["mcp_servers"]["before-server"]["args"] == ["/opt/before.js"]
        assert data["mcp_servers"]["after-server"]["args"] == ["/opt/after.js"]

    def test_similarly_named_table_is_not_swallowed_as_own_subtable(
        self, codex_home, fake_wrapper
    ):
        codex_home.mkdir(parents=True)
        cfg = codex_home / "config.toml"
        near_miss = (
            "[mcp_servers.iai-tools-other]\n"
            'command = "node"\n'
            'args = ["/opt/iai-tools-other.js"]\n'
        )
        cfg.write_text(near_miss, encoding="utf-8")

        install_codex_mcp()

        new_text = cfg.read_text(encoding="utf-8")
        assert "/opt/iai-tools-other.js" in new_text
        data = tomllib.loads(new_text)
        assert data["mcp_servers"]["iai-tools-other"]["args"] == [
            "/opt/iai-tools-other.js"
        ]
        assert data["mcp_servers"]["iai-tools"]["args"] == [str(fake_wrapper)]

    def test_commented_foreign_header_after_ours_survives_byte_for_byte(
        self, codex_home, fake_wrapper
    ):
        codex_home.mkdir(parents=True)
        cfg = codex_home / "config.toml"
        stale_block = (
            "[mcp_servers.iai-tools]\n"
            'command = "node"\n'
            'args = ["/stale/path.js"]\n'
        )
        commented_after = (
            "[mcp_servers.work-server]  # personal token below\n"
            'command = "node"\n'
            'args = ["/opt/work.js"]\n\n'
            "[mcp_servers.work-server.env]\n"
            f'API_TOKEN = "{_FOREIGN_TOKEN}"\n'
        )
        cfg.write_text(stale_block + "\n" + commented_after, encoding="utf-8")

        install_codex_mcp()

        new_text = cfg.read_text(encoding="utf-8")
        assert _FOREIGN_TOKEN in new_text
        assert commented_after in new_text
        assert "/stale/path.js" not in new_text
        data = tomllib.loads(new_text)
        assert data["mcp_servers"]["work-server"]["env"]["API_TOKEN"] == _FOREIGN_TOKEN
        assert data["mcp_servers"]["iai-tools"]["args"] == [str(fake_wrapper)]

    def test_quoted_own_header_is_recognized_not_duplicated(
        self, codex_home, fake_wrapper
    ):
        codex_home.mkdir(parents=True)
        cfg = codex_home / "config.toml"
        quoted_block = (
            '[mcp_servers."iai-tools"]\n'
            'command = "node"\n'
            'args = ["/stale/path.js"]\n'
        )
        cfg.write_text(quoted_block, encoding="utf-8")

        install_codex_mcp()

        new_text = cfg.read_text(encoding="utf-8")
        # A duplicate append (text scan missed the quoted header while
        # tomllib found it) would leave two top-level iai-tools headers
        # and a stale /stale/path.js entry alongside the new one.
        assert new_text.count('command = "node"') == 1
        assert "/stale/path.js" not in new_text
        data = tomllib.loads(new_text)
        assert data["mcp_servers"]["iai-tools"]["args"] == [str(fake_wrapper)]

    def test_multiline_string_containing_our_header_text_refuses_not_splices(
        self, codex_home, fake_wrapper, capsys
    ):
        codex_home.mkdir(parents=True)
        cfg = codex_home / "config.toml"
        # The literal bracket text lives inside a multi-line string value of
        # a foreign table -- tomllib sees no real iai-tools table (existing
        # is None), but the naive line scanner sees a header-shaped line
        # and reports a range (rng is not None). The symmetric guard must
        # refuse on this disagreement instead of splicing at the phantom
        # boundary and corrupting the string.
        config = (
            '[mcp_servers.other-server]\n'
            'command = "node"\n'
            'args = ["/opt/other/server.js"]\n'
            'notes = """\n'
            'Example config:\n'
            '[mcp_servers.iai-tools]\n'
            'command = "node"\n'
            '"""\n'
        )
        cfg.write_text(config, encoding="utf-8")

        rc = install_codex_mcp()

        assert rc == 1
        out = capsys.readouterr().out
        assert "disagree" in out.lower()
        assert cfg.read_text(encoding="utf-8") == config

    def test_genuinely_broken_toml_gets_a_parse_error_message_not_disagree(
        self, codex_home, fake_wrapper, capsys
    ):
        codex_home.mkdir(parents=True)
        cfg = codex_home / "config.toml"
        # Invalid TOML syntax (unterminated string) but the scanner still
        # sees a header-shaped line -- existing is None because parsing
        # fails outright, not because the table is genuinely absent. The
        # refusal message must say so, not send the user hunting for a
        # header disagreement that isn't the real cause.
        config = '[mcp_servers.iai-tools]\ncommand = "node\n'
        cfg.write_text(config, encoding="utf-8")

        rc = install_codex_mcp()

        assert rc == 1
        out = capsys.readouterr().out
        assert "does not parse as toml" in out.lower()
        assert "disagree" not in out.lower()
        assert cfg.read_text(encoding="utf-8") == config


class TestUninstall:
    def test_removes_only_own_table_and_subtables(self, codex_home, fake_wrapper):
        codex_home.mkdir(parents=True)
        cfg = codex_home / "config.toml"
        cfg.write_text(_FIXTURE, encoding="utf-8")

        install_codex_mcp()
        assert uninstall_codex_mcp() == 0

        new_text = cfg.read_text(encoding="utf-8")
        assert new_text == _FIXTURE
        assert "iai-tools" not in tomllib.loads(new_text).get("mcp_servers", {})

    def test_own_env_subtable_not_orphaned_by_reinstall_then_uninstall(
        self, codex_home, fake_wrapper
    ):
        codex_home.mkdir(parents=True)
        cfg = codex_home / "config.toml"
        cfg.write_text(_FIXTURE, encoding="utf-8")

        install_codex_mcp()
        install_codex_mcp()
        uninstall_codex_mcp()

        new_text = cfg.read_text(encoding="utf-8")
        assert new_text == _FIXTURE

    def test_uninstall_on_absent_config_is_noop(self, codex_home, fake_wrapper):
        assert uninstall_codex_mcp() == 0
        assert not (codex_home / "config.toml").exists()

    def test_uninstall_without_our_table_leaves_file_unchanged(
        self, codex_home, fake_wrapper
    ):
        codex_home.mkdir(parents=True)
        cfg = codex_home / "config.toml"
        cfg.write_text(_FIXTURE, encoding="utf-8")
        assert uninstall_codex_mcp() == 0
        assert cfg.read_text(encoding="utf-8") == _FIXTURE

    def test_uninstall_refuses_on_multiline_string_containing_our_header_text(
        self, codex_home, fake_wrapper, capsys
    ):
        codex_home.mkdir(parents=True)
        cfg = codex_home / "config.toml"
        config = (
            '[mcp_servers.other-server]\n'
            'command = "node"\n'
            'args = ["/opt/other/server.js"]\n'
            'notes = """\n'
            'Example config:\n'
            '[mcp_servers.iai-tools]\n'
            'command = "node"\n'
            '"""\n'
        )
        cfg.write_text(config, encoding="utf-8")

        rc = uninstall_codex_mcp()

        assert rc == 1
        out = capsys.readouterr().out
        assert "disagree" in out.lower()
        assert cfg.read_text(encoding="utf-8") == config

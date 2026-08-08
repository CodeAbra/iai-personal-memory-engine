"""Reflection provider selection and the vendor-CLI adapter: config-over-env
with fail-loud unknown/corrupt configs, allow-list child environment (no
foreign key material or endpoint overrides), locked-down argv with the prompt
on stdin, resolution ladder shared by daemon and CLI, output cap, and insight
routing that skips Anthropic-only machinery for non-claude providers."""
from __future__ import annotations

import asyncio
import json
import os
import stat

import pytest


def test_default_provider_is_claude(tmp_path, monkeypatch) -> None:
    from iai_mcp.reflection_provider import configured_reflection_provider

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    assert configured_reflection_provider() == "claude"


def test_config_selects_provider_and_roundtrips(tmp_path, monkeypatch) -> None:
    from iai_mcp.reflection_provider import (
        configured_reflection_provider,
        set_reflection_provider,
    )

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    (tmp_path / "config.json").write_text(
        json.dumps({"user": {"timezone": "UTC"}}), encoding="utf-8"
    )
    set_reflection_provider("gemini")
    assert configured_reflection_provider() == "gemini"
    cfg = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert cfg["user"]["timezone"] == "UTC", "other config keys must survive"
    assert not (tmp_path / "config.json.tmp").exists(), (
        "the write must be tmp+rename, with the tmp renamed away"
    )

    set_reflection_provider("claude")
    assert configured_reflection_provider() == "claude"


def test_unknown_provider_refuses(tmp_path, monkeypatch) -> None:
    from iai_mcp.reflection_provider import (
        configured_reflection_provider,
        set_reflection_provider,
    )

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    (tmp_path / "config.json").write_text(
        json.dumps({"reflection": {"provider": "shady-llm"}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="shady-llm"):
        configured_reflection_provider()
    with pytest.raises(ValueError, match="no-such"):
        set_reflection_provider("no-such")


def test_corrupt_config_refuses_not_reroutes(tmp_path, monkeypatch) -> None:
    """An unreadable config must refuse — a silent claude default would
    reroute the nightly call away from the user's chosen vendor."""
    from iai_mcp.reflection_provider import (
        configured_reflection_provider,
        set_reflection_provider,
    )

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    (tmp_path / "config.json").write_text("NOT JSON", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        configured_reflection_provider()
    with pytest.raises(ValueError, match="unreadable"):
        set_reflection_provider("gemini")
    assert (tmp_path / "config.json").read_text(encoding="utf-8") == "NOT JSON", (
        "a refusing setter must not clobber the corrupt file"
    )


def _fake_bin(root, name: str, script: str) -> None:
    p = root / "bin" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("#!/bin/sh\n" + script, encoding="utf-8")
    p.chmod(p.stat().st_mode | stat.S_IXUSR)


@pytest.fixture
def fake_path(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("PATH", f"{tmp_path}/bin:{os.environ['PATH']}")
    return tmp_path


def test_gemini_adapter_argv_and_stdin_prompt(fake_path) -> None:
    """The prompt travels on stdin (argv is world-readable via ps) and the
    argv carries the read-only flags: plan approval mode, no extensions."""
    from iai_mcp.reflection_provider import _invoke_subscription_cli

    _fake_bin(fake_path, "gemini", 'printf "ARGV:%s\\n" "$@"; printf "STDIN:"; cat -')
    result = asyncio.run(
        _invoke_subscription_cli("gemini", "the store content prompt")
    )
    assert result["ok"], result
    assert "ARGV:--approval-mode\nARGV:plan\n" in result["text"]
    assert "ARGV:--skip-trust\n" in result["text"]
    assert "ARGV:-e\nARGV:none\n" in result["text"]
    assert "STDIN:the store content prompt" in result["text"]
    assert "ARGV:the store content prompt" not in result["text"], (
        "the prompt must never appear in argv"
    )
    assert result["data"]["result"] == result["text"]
    assert result["tokens_in"] == 0 and result["tokens_out"] == 0


def test_codex_adapter_argv_and_stdin_prompt(fake_path) -> None:
    from iai_mcp.reflection_provider import _invoke_subscription_cli

    _fake_bin(fake_path, "codex", 'printf "ARGV:%s\\n" "$@"; printf "STDIN:"; cat -')
    result = asyncio.run(_invoke_subscription_cli("codex", "probe prompt"))
    assert result["ok"], result
    assert result["text"].startswith(
        "ARGV:exec\nARGV:--sandbox\nARGV:read-only\n"
        "ARGV:--skip-git-repo-check\nARGV:-\n"
    ), result["text"]
    assert "STDIN:probe prompt" in result["text"]


def test_adapter_runs_in_a_scratch_cwd(fake_path) -> None:
    """The vendor child must never inherit the daemon's working directory
    (the service manager pins the user's home)."""
    from iai_mcp.reflection_provider import _invoke_subscription_cli

    _fake_bin(fake_path, "gemini", "pwd; cat - >/dev/null")
    result = asyncio.run(_invoke_subscription_cli("gemini", "probe"))
    assert result["ok"], result
    assert result["text"].strip().endswith(".reflection-scratch")
    assert str(fake_path / "store") in result["text"]


def test_missing_provider_cli_is_a_typed_failure(tmp_path, monkeypatch) -> None:
    from iai_mcp.reflection_provider import _invoke_subscription_cli

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    monkeypatch.delenv("IAI_MCP_CODEX_BIN", raising=False)
    result = asyncio.run(_invoke_subscription_cli("codex", "probe"))
    assert not result["ok"]
    assert result["reason"] == "provider_cli_missing_codex"


def test_provider_cli_failure_carries_exit_and_detail(fake_path) -> None:
    from iai_mcp.reflection_provider import _invoke_subscription_cli

    _fake_bin(fake_path, "gemini", 'cat - >/dev/null; echo "quota exhausted" >&2; exit 7')
    result = asyncio.run(_invoke_subscription_cli("gemini", "probe"))
    assert not result["ok"]
    assert "gemini_exit_7" in result["reason"]
    assert "quota exhausted" in result["reason"]


def test_provider_empty_output_is_a_failure(fake_path) -> None:
    from iai_mcp.reflection_provider import _invoke_subscription_cli

    _fake_bin(fake_path, "gemini", "cat - >/dev/null; exit 0")
    result = asyncio.run(_invoke_subscription_cli("gemini", "probe"))
    assert not result["ok"]
    assert result["reason"] == "gemini_empty_output"


def test_oversized_output_is_refused(fake_path) -> None:
    from iai_mcp.reflection_provider import (
        REFLECTION_OUTPUT_MAX_BYTES,
        _invoke_subscription_cli,
    )

    _fake_bin(
        fake_path, "gemini",
        f'cat - >/dev/null; head -c {REFLECTION_OUTPUT_MAX_BYTES + 1} /dev/zero'
        ' | tr "\\0" "x"',
    )
    result = asyncio.run(_invoke_subscription_cli("gemini", "probe"))
    assert not result["ok"]
    assert result["reason"].startswith("gemini_output_too_large_")


def test_child_env_is_an_allowlist_no_key_material(fake_path, monkeypatch) -> None:
    """Per denied variable, with values that cannot substring-alias: the
    child dumps its whole environment and none of the planted key material
    or endpoint overrides may appear in it."""
    from iai_mcp.reflection_provider import _invoke_subscription_cli

    _fake_bin(fake_path, "gemini", "cat - >/dev/null; env | sort")
    planted = {
        "ANTHROPIC_API_KEY": "sk-ant-LEAK",
        "GEMINI_API_KEY": "AIza-LEAK",
        "GOOGLE_API_KEY": "goog-LEAK",
        "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/creds-LEAK.json",
        "OPENAI_API_KEY": "sk-oai-LEAK",
        "OPENAI_BASE_URL": "https://evil.example/v1",
        "AZURE_OPENAI_API_KEY": "az-LEAK",
    }
    for var, val in planted.items():
        monkeypatch.setenv(var, val)
    result = asyncio.run(_invoke_subscription_cli("gemini", "probe"))
    assert result["ok"], result
    for var in planted:
        assert f"{var}=" not in result["text"], f"{var} leaked to the child"
    assert "LEAK" not in result["text"]
    assert "PATH=" in result["text"], "the allow-list must still pass PATH"
    assert "HOME=" in result["text"], "the CLIs read their config under HOME"


def test_resolution_ladder_finds_home_local_bin(tmp_path, monkeypatch) -> None:
    """The daemon's service PATH omits ~/.local/bin; the ladder must find
    the binary there anyway, and the env override must win outright."""
    from iai_mcp.reflection_provider import resolve_provider_bin

    home = tmp_path / "home"
    ladder_bin = home / ".local" / "bin" / "gemini"
    ladder_bin.parent.mkdir(parents=True)
    ladder_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    ladder_bin.chmod(0o755)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", "/nonexistent-for-this-test")
    monkeypatch.delenv("IAI_MCP_GEMINI_BIN", raising=False)
    assert resolve_provider_bin("gemini") == str(ladder_bin)

    explicit = tmp_path / "elsewhere" / "gemini"
    explicit.parent.mkdir(parents=True)
    explicit.write_text("#!/bin/sh\n", encoding="utf-8")
    explicit.chmod(0o755)
    monkeypatch.setenv("IAI_MCP_GEMINI_BIN", str(explicit))
    assert resolve_provider_bin("gemini") == str(explicit)


def test_unsupported_provider_is_typed(tmp_path, monkeypatch) -> None:
    from iai_mcp.reflection_provider import _invoke_subscription_cli

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    result = asyncio.run(_invoke_subscription_cli("claude", "probe"))
    assert not result["ok"]
    assert result["reason"] == "provider_unsupported_claude"


def test_claude_dispatch_routes_to_invoke_claude_once(monkeypatch) -> None:
    import iai_mcp.claude_cli as claude_cli
    from iai_mcp.reflection_provider import invoke_reflection_once

    seen: dict = {}

    async def fake_claude(prompt, *, model="haiku"):
        seen["prompt"] = prompt
        seen["model"] = model
        return {"ok": True, "data": {"result": "claude text"},
                "tokens_in": 1, "tokens_out": 2, "cost_usd": 0.0}

    monkeypatch.setattr(claude_cli, "invoke_claude_once", fake_claude)
    result = asyncio.run(
        invoke_reflection_once("p1", model="sonnet", provider="claude")
    )
    assert result["ok"] and result["data"]["result"] == "claude text"
    assert seen == {"prompt": "p1", "model": "sonnet"}


def test_claude_sync_dispatch_routes_to_invoke_claude_sync(monkeypatch) -> None:
    import iai_mcp.claude_cli as claude_cli
    from iai_mcp.reflection_provider import invoke_reflection_sync

    seen: dict = {}

    def fake_claude(prompt, *, model="haiku"):
        seen["prompt"] = prompt
        seen["model"] = model
        return {"ok": True, "data": {"result": "claude text"},
                "tokens_in": 1, "tokens_out": 2, "cost_usd": 0.0}

    monkeypatch.setattr(claude_cli, "invoke_claude_sync", fake_claude)
    result = invoke_reflection_sync("p2", model="haiku", provider="claude")
    assert result["ok"]
    assert seen == {"prompt": "p2", "model": "haiku"}


def test_sync_on_a_running_loop_is_a_typed_failure(tmp_path, monkeypatch) -> None:
    from iai_mcp.reflection_provider import invoke_reflection_sync

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))

    async def _on_loop():
        return invoke_reflection_sync("probe", provider="gemini")

    result = asyncio.run(_on_loop())
    assert not result["ok"]
    assert result["reason"] == "sync_call_on_event_loop"


def test_insight_routes_non_claude_without_anthropic_machinery(
    tmp_path, monkeypatch
) -> None:
    """With a non-claude provider, the Anthropic credential check and the
    claude budget must not run — they gate the wrong subscription."""
    import iai_mcp.insight as insight_mod

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    (tmp_path / "config.json").write_text(
        json.dumps({"reflection": {"provider": "gemini"}}), encoding="utf-8"
    )

    def _must_not_run():
        raise AssertionError("Anthropic credential check ran for gemini provider")

    monkeypatch.setattr(
        insight_mod, "verify_credentials_subscription", _must_not_run
    )

    async def _fake_reflect(prompt, **kwargs):
        return {"ok": True, "text": "reflection text",
                "data": {"result": "reflection text"},
                "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}

    monkeypatch.setattr(insight_mod, "invoke_reflection_once", _fake_reflect)

    from iai_mcp.store import MemoryStore

    store = MemoryStore(str(tmp_path / "store"))
    try:
        result = asyncio.run(
            insight_mod.generate_overnight_insight(store, "test-session")
        )
    finally:
        store.close()
    # An empty store has no evidence sources — the call must stop at the
    # provenance gate, NOT at Anthropic credential machinery.
    assert result["reason"] in ("no_evidence_sources",), result


def test_insight_refuses_on_invalid_provider_config(
    tmp_path, monkeypatch
) -> None:
    import iai_mcp.insight as insight_mod

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    (tmp_path / "config.json").write_text(
        json.dumps({"reflection": {"provider": "bogus"}}), encoding="utf-8"
    )

    from iai_mcp.store import MemoryStore

    store = MemoryStore(str(tmp_path / "store"))
    try:
        result = asyncio.run(
            insight_mod.generate_overnight_insight(store, "test-session")
        )
    finally:
        store.close()
    assert result["ok"] is False
    assert result["reason"].startswith("reflection_provider_invalid")


def test_insight_claude_path_still_checks_credentials(
    tmp_path, monkeypatch
) -> None:
    import iai_mcp.insight as insight_mod

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))

    seen: dict = {}

    def _creds():
        seen["checked"] = True
        return {"ok": False, "reason": "probe"}

    monkeypatch.setattr(insight_mod, "verify_credentials_subscription", _creds)

    from iai_mcp.store import MemoryStore

    store = MemoryStore(str(tmp_path / "store"))
    try:
        result = asyncio.run(
            insight_mod.generate_overnight_insight(store, "test-session")
        )
    finally:
        store.close()
    assert seen.get("checked") is True
    assert result["reason"] == "credentials_check_failed"


def test_unavailable_scratch_cwd_is_a_typed_refusal(fake_path, monkeypatch) -> None:
    """A failed scratch dir must refuse, never fall back to the inherited
    cwd — that would silently undo the workspace isolation."""
    import iai_mcp.reflection_provider as rp

    _fake_bin(fake_path, "gemini", "cat - >/dev/null; echo ok")
    monkeypatch.setattr(rp, "_reflection_cwd", lambda: None)
    result = asyncio.run(rp._invoke_subscription_cli("gemini", "probe"))
    assert not result["ok"]
    assert result["reason"] == "gemini_scratch_cwd_unavailable"


def test_cancellation_terminates_the_vendor_child(fake_path, monkeypatch) -> None:
    import iai_mcp.claude_cli as claude_cli
    import iai_mcp.reflection_provider as rp

    killed: list = []

    class _StubProc:
        pid = 4242
        returncode = None

        async def communicate(self, input=None):
            raise asyncio.CancelledError()

    async def _fake_exec(*argv, **kwargs):
        return _StubProc()

    async def _fake_kill(proc, grace):
        killed.append((proc, grace))

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    monkeypatch.setattr(claude_cli, "_terminate_then_kill", _fake_kill)

    async def _run():
        return await rp._invoke_subscription_cli("gemini", "probe")

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_run())
    assert len(killed) == 1, "a cancelled reflection must terminate its child"


def test_output_cap_value_is_pinned() -> None:
    from iai_mcp.reflection_provider import REFLECTION_OUTPUT_MAX_BYTES

    assert REFLECTION_OUTPUT_MAX_BYTES == 8192


def test_critic_passes_the_resolved_provider_down(tmp_path, monkeypatch) -> None:
    import uuid as _uuid

    import iai_mcp.reconsolidation_critic as critic_mod

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    (tmp_path / "config.json").write_text(
        json.dumps({"reflection": {"provider": "gemini"}}), encoding="utf-8"
    )

    seen: dict = {}

    def _fake_sync(prompt, **kwargs):
        seen.update(kwargs)
        return {"ok": True, "data": {"result": "[]"},
                "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}

    # The critic imports the invoker inside the function body — patch the
    # source module, not the critic's namespace.
    monkeypatch.setattr(
        "iai_mcp.reflection_provider.invoke_reflection_sync", _fake_sync
    )
    critic_mod.evaluate_batch_reconsolidation([(_uuid.uuid4(), "text")])
    assert seen.get("provider") == "gemini", (
        "the critic must pass the provider it gated on to the invoker"
    )


def test_critic_warns_on_unparseable_output(tmp_path, monkeypatch, caplog) -> None:
    import logging
    import uuid as _uuid

    import iai_mcp.reconsolidation_critic as critic_mod

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    (tmp_path / "config.json").write_text(
        json.dumps({"reflection": {"provider": "gemini"}}), encoding="utf-8"
    )

    def _fake_sync(prompt, **kwargs):
        return {"ok": True, "data": {"result": "agent prose, not a JSON array"},
                "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}

    monkeypatch.setattr(
        "iai_mcp.reflection_provider.invoke_reflection_sync", _fake_sync
    )
    with caplog.at_level(logging.WARNING):
        scores = critic_mod.evaluate_batch_reconsolidation(
            [(_uuid.uuid4(), "text")]
        )
    assert scores == {}
    assert any("unparseable" in r.message for r in caplog.records), (
        "a broken provider must be distinguishable from nothing-was-labile"
    )


def test_insight_passes_the_resolved_provider_down(tmp_path, monkeypatch) -> None:
    import uuid as _uuid

    import iai_mcp.insight as insight_mod

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    (tmp_path / "config.json").write_text(
        json.dumps({"reflection": {"provider": "gemini"}}), encoding="utf-8"
    )

    sid = _uuid.uuid4()
    monkeypatch.setattr(
        insight_mod, "_gather_patterns", lambda store: (["pattern"], [sid])
    )
    monkeypatch.setattr(
        insight_mod, "_gather_surprise", lambda store: ("surprise", [])
    )

    class _Store:
        def get_batch(self, ids):
            return {sid: object()}

    seen: dict = {}

    async def _fake_once(prompt, **kwargs):
        seen.update(kwargs)
        return {"ok": False, "reason": "probe_stop"}

    monkeypatch.setattr(insight_mod, "invoke_reflection_once", _fake_once)
    result = asyncio.run(
        insight_mod.generate_overnight_insight(_Store(), "session")
    )
    assert result["reason"] == "probe_stop"
    assert seen.get("provider") == "gemini", (
        "insight must pass the provider it gated on to the invoker"
    )

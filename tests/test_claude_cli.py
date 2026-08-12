from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest


@pytest.fixture(autouse=True)
def _stub_keychain_credentials(monkeypatch):
    """Stop verify_credentials_subscription from reading the developer's real
    macOS login Keychain when a test monkeypatches CREDENTIALS_PATH to a tmp
    path that does not exist. Otherwise the Keychain fallback returns valid
    OAuth credentials and tests that expect ``credentials_file_missing`` see
    ``ok=True`` instead.

    Also drop any ambient ``IAI_MCP_CLAUDE_BARE`` from the shell environment
    so the bare-flag default behaviour is exercised hermetically regardless
    of how the developer's shell is configured.
    """
    from iai_mcp import claude_cli
    monkeypatch.setattr(claude_cli, "_read_keychain_credentials", lambda: None)
    monkeypatch.delenv("IAI_MCP_CLAUDE_BARE", raising=False)


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    from iai_mcp import daemon_state
    state_path = tmp_path / ".daemon-state.json"
    monkeypatch.setattr(daemon_state, "STATE_PATH", state_path)
    return state_path


@pytest.fixture
def fake_creds(tmp_path, monkeypatch):
    creds = tmp_path / ".credentials.json"
    creds.write_text(json.dumps({"billingType": "stripe_subscription"}))
    from iai_mcp import claude_cli
    monkeypatch.setattr(claude_cli, "CREDENTIALS_PATH", creds)
    return creds


class _FakeProc:

    def __init__(
        self,
        stdout: bytes = b"{}",
        stderr: bytes = b"",
        returncode: int = 0,
        *,
        hang: bool = False,
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self._hang = hang
        self.terminate_called = False
        self.kill_called = False

    async def communicate(self, input=None):  # noqa: ARG002
        if self._hang:
            await asyncio.sleep(3600)
        return (self._stdout, self._stderr)

    def terminate(self) -> None:
        self.terminate_called = True
        if self.returncode is None:
            self.returncode = -15

    def kill(self) -> None:
        self.kill_called = True
        if self.returncode is None:
            self.returncode = -9

    async def wait(self):
        return self.returncode


def _install_subprocess_mock(monkeypatch, proc: _FakeProc) -> dict:
    capture: dict = {"args": None, "env": None, "kwargs": None}

    async def fake_spawn(*args, **kwargs):
        capture["args"] = args
        capture["env"] = kwargs.get("env")
        capture["kwargs"] = kwargs
        return proc

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_spawn)
    return capture


def test_invoke_uses_argv_and_required_flags(monkeypatch, fake_creds, isolated_state):
    from iai_mcp.claude_cli import invoke_claude_once

    proc = _FakeProc(stdout=json.dumps({
        "result": "ok",
        "cost_usd": 0,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }).encode("utf-8"))
    cap = _install_subprocess_mock(monkeypatch, proc)

    result = asyncio.run(invoke_claude_once("hello", model="haiku"))

    assert result["ok"] is True
    args = cap["args"]
    import os as _os
    assert _os.path.basename(args[0]) == "claude"
    assert "--bare" in args
    assert "-p" in args
    assert "hello" in args
    assert "--output-format" in args and "json" in args
    assert "--max-turns" in args and "1" in args
    assert "--tools" in args
    assert "--no-session-persistence" in args
    assert "--model" in args and "haiku" in args


def test_env_scrubbed(monkeypatch, fake_creds, isolated_state):
    from iai_mcp.claude_cli import invoke_claude_once

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-hostile-1")
    monkeypatch.setenv("CLAUDE_API_KEY", "sk-hostile-2")
    monkeypatch.setenv("CLAUDE_CODE_API_KEY", "sk-hostile-3")
    monkeypatch.setenv("KEEP_ME", "benign")

    proc = _FakeProc(stdout=json.dumps({
        "result": "ok", "cost_usd": 0, "usage": {"input_tokens": 1, "output_tokens": 1},
    }).encode("utf-8"))
    cap = _install_subprocess_mock(monkeypatch, proc)

    asyncio.run(invoke_claude_once("hi", model="haiku"))

    env = cap["env"]
    assert env is not None
    for key in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "CLAUDE_CODE_API_KEY"):
        assert key not in env, f"C3 violation: {key} leaked to subprocess env"
    assert env.get("KEEP_ME") == "benign"


def test_happy_path_parses_tokens_and_cost(monkeypatch, fake_creds, isolated_state):
    from iai_mcp.claude_cli import invoke_claude_once

    payload = {
        "result": "unifying insight text",
        "cost_usd": 0,
        "usage": {"input_tokens": 150, "output_tokens": 40},
        "is_error": False,
        "session_id": "sess-x",
        "duration_ms": 500,
        "num_turns": 1,
    }
    proc = _FakeProc(stdout=json.dumps(payload).encode("utf-8"))
    _install_subprocess_mock(monkeypatch, proc)

    result = asyncio.run(invoke_claude_once("hi", model="haiku"))
    assert result["ok"] is True
    assert result["cost_usd"] == 0.0
    assert result["tokens_in"] == 150
    assert result["tokens_out"] == 40
    assert result["data"]["result"] == "unifying insight text"


def test_c3_auto_disable(monkeypatch, fake_creds, isolated_state):
    from iai_mcp.claude_cli import BudgetTracker, invoke_claude_once
    from iai_mcp.daemon_state import load_state

    payload = {
        "result": "billing detected text",
        "cost_usd": 0.05,
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }
    proc = _FakeProc(stdout=json.dumps(payload).encode("utf-8"))
    _install_subprocess_mock(monkeypatch, proc)

    result = asyncio.run(invoke_claude_once("hi", model="haiku"))
    assert result["ok"] is False
    assert result["reason"] == "api_billing_detected"
    assert result["cost_usd"] == 0.05

    tracker = BudgetTracker(load_state())
    assert tracker.claude_disabled_after_billing_event() is True


def test_timeout_terminates_then_kills(monkeypatch, fake_creds, isolated_state):
    from iai_mcp import claude_cli
    from iai_mcp.claude_cli import invoke_claude_once

    monkeypatch.setattr(claude_cli, "CLAUDE_TIMEOUT_SEC", 0.05)
    monkeypatch.setattr(claude_cli, "TERMINATE_WAIT_SEC", 0.05)
    monkeypatch.setattr(claude_cli, "KILL_WAIT_SEC", 0.05)

    proc = _FakeProc(hang=True, returncode=None)

    async def slow_wait():
        await asyncio.sleep(3600)
        return -9

    proc.wait = slow_wait  # type: ignore[assignment]
    _install_subprocess_mock(monkeypatch, proc)

    result = asyncio.run(invoke_claude_once("hi", model="haiku"))
    assert result["ok"] is False
    assert result["reason"] == "timeout"
    assert proc.terminate_called is True
    assert proc.kill_called is True


def test_nonzero_exit(monkeypatch, fake_creds, isolated_state):
    from iai_mcp.claude_cli import invoke_claude_once

    proc = _FakeProc(stdout=b"", stderr=b"subscription expired", returncode=1)
    _install_subprocess_mock(monkeypatch, proc)

    result = asyncio.run(invoke_claude_once("hi", model="haiku"))
    assert result["ok"] is False
    assert result["reason"] == "nonzero_exit"
    assert result["exit_code"] == 1
    assert "subscription expired" in result["stderr"]


def test_unparseable_output(monkeypatch, fake_creds, isolated_state):
    from iai_mcp.claude_cli import invoke_claude_once

    proc = _FakeProc(stdout=b"not valid json at all", returncode=0)
    _install_subprocess_mock(monkeypatch, proc)

    result = asyncio.run(invoke_claude_once("hi", model="haiku"))
    assert result["ok"] is False
    assert result["reason"] == "unparseable_output"


def test_credentials_gate(tmp_path, monkeypatch):
    from iai_mcp import claude_cli
    from iai_mcp.claude_cli import verify_credentials_subscription

    creds = tmp_path / ".credentials.json"
    monkeypatch.setattr(claude_cli, "CREDENTIALS_PATH", creds)

    assert verify_credentials_subscription()["ok"] is False

    creds.write_text(json.dumps({"billingType": "api_key"}))
    r = verify_credentials_subscription()
    assert r["ok"] is False
    assert r["reason"] == "not_subscription"

    creds.write_text(json.dumps({"billingType": "stripe_subscription"}))
    r2 = verify_credentials_subscription()
    assert r2["ok"] is True
    assert r2["billing_type"] == "stripe_subscription"


def _new_schema_creds(sub_type: str = "max", scopes=None, expires_at_ms=None):
    if scopes is None:
        scopes = ["user:inference", "user:profile"]
    if expires_at_ms is None:
        expires_at_ms = int(
            (datetime.now(tz=timezone.utc) + timedelta(days=365)).timestamp() * 1000
        )
    return {
        "claudeAiOauth": {
            "accessToken": "sk-ant-oat01-stub",
            "refreshToken": "sk-ant-ort01-stub",
            "expiresAt": expires_at_ms,
            "scopes": scopes,
            "subscriptionType": sub_type,
            "rateLimitTier": f"default_claude_{sub_type}_5x",
        }
    }


@pytest.mark.parametrize("sub_type", ["pro", "pro_max", "max", "team", "enterprise"])
def test_new_schema_accepts_any_valid_tier(tmp_path, monkeypatch, sub_type):
    from iai_mcp import claude_cli
    from iai_mcp.claude_cli import verify_credentials_subscription

    creds = tmp_path / ".credentials.json"
    creds.write_text(json.dumps(_new_schema_creds(sub_type=sub_type)))
    monkeypatch.setattr(claude_cli, "CREDENTIALS_PATH", creds)

    r = verify_credentials_subscription()
    assert r["ok"] is True, f"expected ok=True for tier {sub_type!r}, got {r}"
    assert r["subscription_type"] == sub_type


def test_new_schema_rejects_invalid_tier(tmp_path, monkeypatch):
    from iai_mcp import claude_cli
    from iai_mcp.claude_cli import verify_credentials_subscription

    creds = tmp_path / ".credentials.json"
    creds.write_text(json.dumps(_new_schema_creds(sub_type="community")))
    monkeypatch.setattr(claude_cli, "CREDENTIALS_PATH", creds)

    r = verify_credentials_subscription()
    assert r["ok"] is False
    assert r["reason"] == "not_subscription"
    assert r["subscription_type"] == "community"


def test_new_schema_rejects_missing_inference_scope(tmp_path, monkeypatch):
    from iai_mcp import claude_cli
    from iai_mcp.claude_cli import verify_credentials_subscription

    creds = tmp_path / ".credentials.json"
    creds.write_text(json.dumps(_new_schema_creds(
        sub_type="max",
        scopes=["user:profile", "user:mcp_servers"],
    )))
    monkeypatch.setattr(claude_cli, "CREDENTIALS_PATH", creds)

    r = verify_credentials_subscription()
    assert r["ok"] is False
    assert r["reason"] == "missing_inference_scope"


def test_new_schema_rejects_expired_credentials_when_no_refresh_token(
    tmp_path, monkeypatch,
):
    from iai_mcp import claude_cli
    from iai_mcp.claude_cli import verify_credentials_subscription

    creds = tmp_path / ".credentials.json"
    expired_ms = int(
        (datetime.now(tz=timezone.utc) - timedelta(days=1)).timestamp() * 1000
    )
    payload = _new_schema_creds(
        sub_type="max",
        expires_at_ms=expired_ms,
    )
    del payload["claudeAiOauth"]["refreshToken"]
    creds.write_text(json.dumps(payload))
    monkeypatch.setattr(claude_cli, "CREDENTIALS_PATH", creds)

    r = verify_credentials_subscription()
    assert r["ok"] is False
    assert r["reason"] == "credentials_expired"


def test_new_schema_accepts_expired_access_token_with_refresh_token(
    tmp_path, monkeypatch,
):
    from iai_mcp import claude_cli
    from iai_mcp.claude_cli import verify_credentials_subscription

    creds = tmp_path / ".credentials.json"
    expired_ms = int(
        (datetime.now(tz=timezone.utc) - timedelta(hours=2)).timestamp() * 1000
    )
    creds.write_text(json.dumps(_new_schema_creds(
        sub_type="max",
        expires_at_ms=expired_ms,
    )))
    monkeypatch.setattr(claude_cli, "CREDENTIALS_PATH", creds)

    r = verify_credentials_subscription()
    assert r["ok"] is True
    assert r["subscription_type"] == "max"


def test_new_schema_takes_precedence_over_legacy_billingType(
    tmp_path, monkeypatch,
):
    from iai_mcp import claude_cli
    from iai_mcp.claude_cli import verify_credentials_subscription

    creds = tmp_path / ".credentials.json"
    payload = _new_schema_creds(sub_type="pro")
    payload["billingType"] = "api_key"
    creds.write_text(json.dumps(payload))
    monkeypatch.setattr(claude_cli, "CREDENTIALS_PATH", creds)

    r = verify_credentials_subscription()
    assert r["ok"] is True
    assert r["subscription_type"] == "pro"


def test_budget_cap(isolated_state):
    from iai_mcp.claude_cli import (
        BUDGET_STATE_KEY,
        BudgetTracker,
        DAILY_QUOTA_BUDGET_PCT,
        ESTIMATED_DAILY_TOKEN_CEILING,
    )

    daily_cap = int(DAILY_QUOTA_BUDGET_PCT * ESTIMATED_DAILY_TOKEN_CEILING)

    state = {BUDGET_STATE_KEY: {
        "daily_used_tokens": daily_cap - 500,
        "weekly_buffer_used_tokens": 0,
        "last_reset_date": "2026-04-18",
        "claude_disabled": False,
        "claude_disabled_reason": None,
    }}
    assert BudgetTracker(state).can_spend(100) is True

    state2 = {BUDGET_STATE_KEY: {
        "daily_used_tokens": daily_cap,
        "weekly_buffer_used_tokens": 0,
        "last_reset_date": "2026-04-18",
        "claude_disabled": False,
        "claude_disabled_reason": None,
    }}
    assert BudgetTracker(state2).can_spend(ESTIMATED_DAILY_TOKEN_CEILING) is False

    state3 = {BUDGET_STATE_KEY: {
        "daily_used_tokens": 0,
        "weekly_buffer_used_tokens": 0,
        "last_reset_date": "2026-04-18",
        "claude_disabled": True,
        "claude_disabled_reason": "api_billing_detected",
    }}
    assert BudgetTracker(state3).can_spend(1) is False


def test_reset_if_new_day(isolated_state):
    from iai_mcp.claude_cli import BUDGET_STATE_KEY, BudgetTracker

    tz = ZoneInfo("Asia/Dubai")
    state = {BUDGET_STATE_KEY: {
        "daily_used_tokens": 8000,
        "weekly_buffer_used_tokens": 0,
        "last_reset_date": "2026-04-17",
        "claude_disabled": False,
        "claude_disabled_reason": None,
    }}
    t = BudgetTracker(state)

    now_same_day = datetime(2026, 4, 17, 23, 0, tzinfo=tz)
    t.reset_if_new_day(now_same_day, tz)
    assert state[BUDGET_STATE_KEY]["daily_used_tokens"] == 8000

    now_new_day = datetime(2026, 4, 18, 1, 0, tzinfo=tz)
    t.reset_if_new_day(now_new_day, tz)
    assert state[BUDGET_STATE_KEY]["daily_used_tokens"] == 0
    assert state[BUDGET_STATE_KEY]["last_reset_date"] == "2026-04-18"


def test_weekly_buffer_exceeded(isolated_state):
    from iai_mcp.claude_cli import (
        BUDGET_STATE_KEY,
        BudgetTracker,
        ESTIMATED_DAILY_TOKEN_CEILING,
        WEEKLY_BUFFER_PCT,
    )

    weekly_cap = int(WEEKLY_BUFFER_PCT * ESTIMATED_DAILY_TOKEN_CEILING * 7)
    state_under = {BUDGET_STATE_KEY: {
        "daily_used_tokens": 0,
        "weekly_buffer_used_tokens": weekly_cap - 1,
        "last_reset_date": "2026-04-18",
        "claude_disabled": False,
        "claude_disabled_reason": None,
    }}
    assert BudgetTracker(state_under).weekly_buffer_exceeded() is False

    state_over = {BUDGET_STATE_KEY: {
        "daily_used_tokens": 0,
        "weekly_buffer_used_tokens": weekly_cap,
        "last_reset_date": "2026-04-18",
        "claude_disabled": False,
        "claude_disabled_reason": None,
    }}
    assert BudgetTracker(state_over).weekly_buffer_exceeded() is True


def test_force_wake_does_not_crash_daemon(monkeypatch, fake_creds, isolated_state):
    from iai_mcp import claude_cli
    from iai_mcp.claude_cli import invoke_claude_once

    monkeypatch.setattr(claude_cli, "FORCE_WAKE_GRACE_SEC", 0.05)
    monkeypatch.setattr(claude_cli, "KILL_WAIT_SEC", 0.05)

    proc = _FakeProc(hang=True, returncode=None)

    async def slow_wait():
        await asyncio.sleep(3600)
        return -9

    proc.wait = slow_wait  # type: ignore[assignment]
    _install_subprocess_mock(monkeypatch, proc)

    async def runner():
        task = asyncio.create_task(invoke_claude_once("hi", model="haiku"))
        await asyncio.sleep(0)
        task.cancel()
        return await task

    result = asyncio.run(runner())
    assert isinstance(result, dict)
    assert result["ok"] is False
    assert result["reason"] == "force_wake_killed"
    assert proc.terminate_called is True
    assert proc.kill_called is True


def test_claude_binary_resolves_without_user_path(monkeypatch, tmp_path):
    """Service managers start the daemon with a minimal PATH; the claude
    binary must still resolve (explicit env first, then PATH, then the
    standard install locations) instead of raising FileNotFoundError."""
    from iai_mcp.claude_cli import _build_cmd, _resolve_claude_bin

    fake = tmp_path / "claude"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o755)

    monkeypatch.setenv("CLAUDE_BIN", str(fake))
    assert _resolve_claude_bin() == str(fake)
    assert _build_cmd("hi", "haiku")[0] == str(fake)

    monkeypatch.delenv("CLAUDE_BIN")
    monkeypatch.setenv("PATH", str(tmp_path))
    assert _resolve_claude_bin() == str(fake)


def test_bare_login_failure_retries_without_bare(
    monkeypatch, fake_creds, isolated_state,
):
    """A --bare attempt that reports "Not logged in" (Keychain-credential
    setups) retries once without the flag; hooks stay disabled via
    --settings on the retry."""
    from iai_mcp.claude_cli import invoke_claude_once

    login_error = json.dumps({
        "is_error": True,
        "result": "Not logged in · Please run /login",
    }).encode("utf-8")
    ok_payload = json.dumps({
        "result": "insight", "cost_usd": 0,
        "usage": {"input_tokens": 7, "output_tokens": 3},
    }).encode("utf-8")

    procs = [
        _FakeProc(stdout=login_error, returncode=1),
        _FakeProc(stdout=ok_payload, returncode=0),
    ]
    calls: list = []

    async def fake_spawn(*args, **kwargs):
        calls.append(args)
        return procs[len(calls) - 1]

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_spawn)

    result = asyncio.run(invoke_claude_once("hello", model="haiku"))

    assert result["ok"] is True
    assert result["bare_fallback_used"] is True
    assert len(calls) == 2
    assert "--bare" in calls[0]
    assert "--bare" not in calls[1]
    assert "--settings" in calls[1]
    settings = json.loads(calls[1][calls[1].index("--settings") + 1])
    assert settings == {"disableAllHooks": True}


def test_zero_exit_error_result_is_not_success(
    monkeypatch, fake_creds, isolated_state,
):
    from iai_mcp.claude_cli import invoke_claude_once

    proc = _FakeProc(stdout=json.dumps({
        "is_error": True, "result": "overloaded",
    }).encode("utf-8"), returncode=0)
    _install_subprocess_mock(monkeypatch, proc)

    result = asyncio.run(invoke_claude_once("hi", model="haiku"))

    assert result["ok"] is False
    assert result["reason"] == "error_result"
    assert "overloaded" in result["detail"]


def test_non_login_failure_does_not_retry(
    monkeypatch, fake_creds, isolated_state,
):
    from iai_mcp.claude_cli import invoke_claude_once

    calls: list = []

    async def fake_spawn(*args, **kwargs):
        calls.append(args)
        return _FakeProc(stdout=b"boom", stderr=b"fatal", returncode=2)

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_spawn)

    result = asyncio.run(invoke_claude_once("hi", model="haiku"))

    assert result["ok"] is False
    assert len(calls) == 1, "only a Not-logged-in --bare failure may retry"


def test_sync_bare_login_failure_retries_without_bare(
    monkeypatch, fake_creds, isolated_state,
):
    import subprocess as _subprocess

    from iai_mcp.claude_cli import invoke_claude_sync

    login_error = json.dumps({
        "is_error": True,
        "result": "Not logged in · Please run /login",
    }).encode("utf-8")
    ok_payload = json.dumps({
        "result": "insight", "cost_usd": 0,
        "usage": {"input_tokens": 7, "output_tokens": 3},
    }).encode("utf-8")

    calls: list = []

    class _Done:
        def __init__(self, rc, out):
            self.returncode = rc
            self.stdout = out
            self.stderr = b""

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if len(calls) == 1:
            return _Done(1, login_error)
        return _Done(0, ok_payload)

    monkeypatch.setattr(_subprocess, "run", fake_run)

    result = invoke_claude_sync("hello", model="haiku")

    assert result["ok"] is True
    assert result["bare_fallback_used"] is True
    assert "--bare" in calls[0]
    assert "--bare" not in calls[1] and "--settings" in calls[1]


def test_scrubbed_env_fills_user_identity(monkeypatch, fake_creds, isolated_state):
    """A launchd child without USER/LOGNAME cannot refresh the Keychain
    OAuth session; the scrubbed env fills both from the process owner."""
    import pwd

    from iai_mcp.claude_cli import _scrubbed_env

    monkeypatch.delenv("USER", raising=False)
    monkeypatch.delenv("LOGNAME", raising=False)

    env = _scrubbed_env()

    expected = pwd.getpwuid(__import__("os").getuid()).pw_name
    assert env.get("USER") == expected
    assert env.get("LOGNAME") == expected


def test_scrubbed_env_keeps_existing_identity(monkeypatch, fake_creds, isolated_state):
    from iai_mcp.claude_cli import _scrubbed_env

    monkeypatch.setenv("USER", "alice")
    monkeypatch.setenv("LOGNAME", "alice")

    env = _scrubbed_env()

    assert env["USER"] == "alice"
    assert env["LOGNAME"] == "alice"


def test_login_marker_beyond_stdout_truncation_still_retries(
    monkeypatch, fake_creds, isolated_state,
):
    """The real CLI front-loads a large usage block, pushing the result
    text past any fixed stdout prefix — detection must ride the parsed
    result field, not the truncation."""
    from iai_mcp.claude_cli import invoke_claude_once

    padding = {"usage": {("k%d" % i): 0 for i in range(80)}}
    login_error = json.dumps({
        "is_error": True,
        **padding,
        "result": "Not logged in · Please run /login",
    }).encode("utf-8")
    assert b"Not logged in" not in login_error[:500], "fixture must exceed prefix"
    ok_payload = json.dumps({
        "result": "insight", "cost_usd": 0,
        "usage": {"input_tokens": 7, "output_tokens": 3},
    }).encode("utf-8")

    procs = [
        _FakeProc(stdout=login_error, returncode=1),
        _FakeProc(stdout=ok_payload, returncode=0),
    ]
    calls: list = []

    async def fake_spawn(*args, **kwargs):
        calls.append(args)
        return procs[len(calls) - 1]

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_spawn)

    result = asyncio.run(invoke_claude_once("hello", model="haiku"))

    assert result["ok"] is True
    assert result["bare_fallback_used"] is True
    assert len(calls) == 2

from __future__ import annotations

import os
import secrets
from pathlib import Path

import pytest


def test_check_h_pass_when_file_present_and_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iai_mcp.doctor import check_h_crypto_file_state

    key_path = tmp_path / ".crypto.key"
    key_path.write_bytes(secrets.token_bytes(32))
    os.chmod(key_path, 0o600)

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))

    result = check_h_crypto_file_state()
    assert result.status == "PASS", f"unexpected status={result.status} detail={result.detail}"
    assert result.passed is True
    assert ".crypto.key" in result.detail


def test_check_h_warn_when_file_missing_and_keyring_has_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iai_mcp.doctor import check_h_crypto_file_state

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    assert not (tmp_path / ".crypto.key").exists()

    import keyring as _keyring

    fake_b64 = "Zm9vYmFyZm9vYmFyZm9vYmFyZm9vYmFyZm9vYmFyZm9vYmE="

    def fake_get(service: str, username: str) -> str | None:
        return fake_b64

    monkeypatch.setattr(_keyring, "get_password", fake_get)

    result = check_h_crypto_file_state()
    assert result.status == "WARN", f"unexpected status={result.status} detail={result.detail}"
    assert "migrate-to-file" in result.detail.lower()
    assert result.passed is True


def test_check_h_pass_when_file_missing_and_no_keyring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iai_mcp.doctor import check_h_crypto_file_state

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    assert not (tmp_path / ".crypto.key").exists()

    import keyring as _keyring

    def fake_get(service: str, username: str) -> str | None:
        return None

    monkeypatch.setattr(_keyring, "get_password", fake_get)

    result = check_h_crypto_file_state()
    assert result.status == "PASS", f"unexpected status={result.status} detail={result.detail}"
    assert result.passed is True
    detail_l = result.detail.lower()
    assert "init" in detail_l or "passphrase" in detail_l


def test_check_h_pass_when_keyring_backend_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iai_mcp.doctor import check_h_crypto_file_state

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    assert not (tmp_path / ".crypto.key").exists()

    import keyring as _keyring
    import keyring.errors as _keyring_errors

    def raise_no_backend(service: str, username: str) -> str | None:
        raise _keyring_errors.NoKeyringError("no backend available (test-stub)")

    monkeypatch.setattr(_keyring, "get_password", raise_no_backend)

    result = check_h_crypto_file_state()
    assert result.status == "PASS", f"unexpected status={result.status} detail={result.detail}"
    assert result.passed is True


def test_check_h_fail_when_file_malformed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iai_mcp.doctor import check_h_crypto_file_state

    key_path = tmp_path / ".crypto.key"
    key_path.write_bytes(b"\x00" * 31)
    os.chmod(key_path, 0o600)

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))

    result = check_h_crypto_file_state()
    assert result.status == "FAIL", f"unexpected status={result.status} detail={result.detail}"
    assert result.passed is False
    assert "wrong length" in result.detail.lower() or "malformed" in result.detail.lower()


def test_format_top_of_output_hint_emits_line_when_check_h_warns() -> None:
    from iai_mcp.doctor import CheckResult, _format_top_of_output_hint

    results = [
        CheckResult("(a) daemon process alive", True, "PID 12345 (iai_mcp.daemon)", status="PASS"),
        CheckResult(
            "(h) crypto key file state",
            True,
            "crypto key file missing at /tmp/x/.crypto.key, but a Keychain entry was found.\n"
            "  Run `iai-mcp crypto migrate-to-file` from a Terminal to migrate the key.",
            status="WARN",
        ),
    ]

    hint = _format_top_of_output_hint(results)
    assert hint is not None, "WARN row for check_h must produce a hint"
    assert hint.startswith("> hint:"), f"hint must be prefixed with `> hint:`, got: {hint!r}"
    assert "migrate-to-file" in hint, f"hint must name migrate-to-file, got: {hint!r}"


def test_format_top_of_output_hint_returns_none_when_no_warn() -> None:
    from iai_mcp.doctor import CheckResult, _format_top_of_output_hint

    results = [
        CheckResult("(a) daemon process alive", True, "PID 12345 (iai_mcp.daemon)", status="PASS"),
        CheckResult("(h) crypto key file state", True, "key file present", status="PASS"),
    ]

    assert _format_top_of_output_hint(results) is None


def test_run_diagnosis_includes_check_h(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from iai_mcp.doctor import run_diagnosis

    key_path = tmp_path / ".crypto.key"
    key_path.write_bytes(secrets.token_bytes(32))
    os.chmod(key_path, 0o600)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))

    results = run_diagnosis()
    h_rows = [r for r in results if "(h)" in r.name and "crypto" in r.name.lower()]
    assert len(h_rows) == 1, (
        f"expected exactly one (h) crypto row in run_diagnosis(); "
        f"got {len(h_rows)} from {[r.name for r in results]}"
    )
    i_rows = [r for r in results if "(i)" in r.name]
    assert len(i_rows) == 1, (
        f"expected exactly one (i) row in run_diagnosis(); "
        f"got {len(i_rows)} from {[r.name for r in results]}"
    )


def test_cmd_doctor_prints_hint_at_top_when_check_h_warns(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import argparse

    from iai_mcp import doctor as _doctor

    synthetic = [
        _doctor.CheckResult("(a) daemon process alive", True, "synthetic", status="PASS"),
        _doctor.CheckResult("(b) socket file fresh", True, "synthetic", status="PASS"),
        _doctor.CheckResult("(c) lock file healthy", True, "synthetic", status="PASS"),
        _doctor.CheckResult("(d) no orphan iai_mcp.core procs", True, "synthetic", status="PASS"),
        _doctor.CheckResult("(e) daemon state file valid", True, "synthetic", status="PASS"),
        _doctor.CheckResult("(f) hippo storage readable", True, "synthetic", status="PASS"),
        _doctor.CheckResult("(g) no dup binders", True, "synthetic", status="PASS"),
        _doctor.CheckResult(
            "(h) crypto key file state",
            True,
            (
                "crypto key file missing at /tmp/.crypto.key, but a Keychain entry was found.\n"
                "  Run `iai-mcp crypto migrate-to-file` from a Terminal to migrate the key."
            ),
            status="WARN",
        ),
    ]
    monkeypatch.setattr(_doctor, "run_diagnosis", lambda: synthetic)

    args = argparse.Namespace(apply=False, yes=False)
    rc = _doctor.cmd_doctor(args)

    captured = capsys.readouterr().out

    hint_idx = captured.find("> hint:")
    header_idx = captured.find("iai doctor")
    assert hint_idx >= 0, f"expected `> hint:` line in stdout, got:\n{captured!r}"
    assert header_idx >= 0, f"expected checklist header in stdout, got:\n{captured!r}"
    assert hint_idx < header_idx, (
        f"hint (idx {hint_idx}) must appear BEFORE checklist header (idx {header_idx})\n"
        f"stdout was:\n{captured}"
    )
    assert "migrate-to-file" in captured[: header_idx], (
        f"hint must name `migrate-to-file` ABOVE the checklist header; "
        f"top-of-output region was: {captured[:header_idx]!r}"
    )
    assert rc == 0, f"WARN rows must not change exit code; got rc={rc}"


def test_check_result_three_arg_constructor_still_works() -> None:
    from iai_mcp.doctor import CheckResult

    r_pass = CheckResult("(x) example", True, "ok")
    assert r_pass.passed is True
    assert r_pass.detail == "ok"
    assert r_pass.status in ("PASS", "FAIL")
    assert r_pass.status == "PASS"

    r_fail = CheckResult("(y) example", False, "broken")
    assert r_fail.status == "FAIL"

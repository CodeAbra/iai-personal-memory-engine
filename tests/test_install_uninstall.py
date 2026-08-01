from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO / "scripts" / "install.sh"
UNINSTALL_SH = REPO / "scripts" / "uninstall.sh"
PLIST_TEMPLATE = REPO / "scripts" / "com.iai-mcp.daemon.plist.template"


def _bash_available() -> bool:
    return shutil.which("bash") is not None


def _dry_run_env() -> dict[str, str]:
    return {**os.environ, "DRY_RUN": "1", "IAI_TEST_SKIP_BUILD": "1"}


@pytest.fixture(autouse=True)
def _scripts_exist() -> None:
    if not INSTALL_SH.exists():
        pytest.skip(f"{INSTALL_SH} missing")
    if not UNINSTALL_SH.exists():
        pytest.skip(f"{UNINSTALL_SH} missing")


@pytest.mark.skipif(not _bash_available(), reason="bash unavailable")
@pytest.mark.skipif(platform.system() != "Darwin", reason="DRY_RUN message only emitted on Darwin")
def test_install_dry_run_succeeds() -> None:
    result = subprocess.run(
        ["bash", str(INSTALL_SH)],
        env=_dry_run_env(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"install.sh DRY_RUN failed:\n--- STDOUT ---\n{result.stdout}\n"
        f"--- STDERR ---\n{result.stderr}\n"
    )
    assert "DRY_RUN=1 — skipping launchctl calls" in result.stdout, (
        f"missing DRY_RUN marker in stdout:\n{result.stdout}"
    )


@pytest.mark.skipif(not _bash_available(), reason="bash unavailable")
def test_install_dry_run_idempotent() -> None:
    env = _dry_run_env()
    for attempt in (1, 2):
        result = subprocess.run(
            ["bash", str(INSTALL_SH)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"install.sh DRY_RUN attempt {attempt} failed:\n"
            f"--- STDOUT ---\n{result.stdout}\n"
            f"--- STDERR ---\n{result.stderr}\n"
        )


@pytest.mark.skipif(not _bash_available(), reason="bash unavailable")
def test_uninstall_dry_run_succeeds() -> None:
    result = subprocess.run(
        ["bash", str(UNINSTALL_SH)],
        env={**os.environ, "DRY_RUN": "1"},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, (
        f"uninstall.sh DRY_RUN failed:\n--- STDOUT ---\n{result.stdout}\n"
        f"--- STDERR ---\n{result.stderr}\n"
    )
    assert "iai-mcp uninstalled" in result.stdout, (
        f"uninstall.sh stdout missing terminator:\n{result.stdout}"
    )


@pytest.mark.skipif(not _bash_available(), reason="bash unavailable")
def test_uninstall_dry_run_idempotent() -> None:
    env = {**os.environ, "DRY_RUN": "1"}
    for attempt in (1, 2):
        result = subprocess.run(
            ["bash", str(UNINSTALL_SH)],
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, (
            f"uninstall.sh DRY_RUN attempt {attempt} failed:\n"
            f"--- STDOUT ---\n{result.stdout}\n"
            f"--- STDERR ---\n{result.stderr}\n"
        )


@pytest.mark.skipif(not _bash_available(), reason="bash unavailable")
@pytest.mark.skipif(not shutil.which("sed"), reason="sed unavailable")
def test_install_renders_template_with_substitutions() -> None:
    if not PLIST_TEMPLATE.exists():
        pytest.skip(f"{PLIST_TEMPLATE} missing")

    fake_python = "/fake/path/.venv/bin/python"
    fake_home = "/tmp/iai-fake-home-test-7-1-03"

    result = subprocess.run(
        [
            "sed",
            "-e", f"s|{{PYTHON_PATH}}|{fake_python}|g",
            "-e", f"s|{{HOME}}|{fake_home}|g",
            str(PLIST_TEMPLATE),
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, f"sed failed: {result.stderr}"
    rendered = result.stdout

    assert fake_python in rendered, "PYTHON_PATH not substituted"
    assert fake_home in rendered, "HOME not substituted"

    assert "{PYTHON_PATH}" not in rendered, "{PYTHON_PATH} placeholder remains"
    assert "{HOME}" not in rendered, "{HOME} placeholder remains"

    assert "<plist version=\"1.0\">" in rendered
    assert "com.iai-mcp.daemon" in rendered


@pytest.mark.skipif(not _bash_available(), reason="bash unavailable")
def test_uninstall_purge_state_dry_run() -> None:
    result = subprocess.run(
        ["bash", str(UNINSTALL_SH), "--purge-state"],
        env={**os.environ, "DRY_RUN": "1"},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, (
        f"uninstall.sh --purge-state DRY_RUN failed:\n"
        f"--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}\n"
    )
    assert "skipping rm of state files" in result.stdout, (
        f"purge-state DRY_RUN gate did not fire:\n{result.stdout}"
    )


@pytest.mark.skipif(not _bash_available(), reason="bash unavailable")
def test_install_sh_syntax_valid() -> None:
    result = subprocess.run(
        ["bash", "-n", str(INSTALL_SH)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, (
        f"install.sh has syntax errors:\n--- STDERR ---\n{result.stderr}\n"
    )


@pytest.mark.skipif(not _bash_available(), reason="bash unavailable")
def test_uninstall_sh_syntax_valid() -> None:
    result = subprocess.run(
        ["bash", "-n", str(UNINSTALL_SH)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, (
        f"uninstall.sh has syntax errors:\n--- STDERR ---\n{result.stderr}\n"
    )

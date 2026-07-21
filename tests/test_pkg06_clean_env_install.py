
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_MASKING_VARS = {"PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME", "PYTHONSTARTUP"}

_REPO_ROOT = Path(__file__).resolve().parent.parent

@pytest.fixture(scope="session")
def clean_install_whl(tmp_path_factory):
    whl_dir = tmp_path_factory.mktemp("whl", numbered=False)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "-w",
            str(whl_dir),
        ],
        cwd=str(_REPO_ROOT),
        check=True,
    )
    wheels = list(whl_dir.glob("iai_mcp-*.whl"))
    assert len(wheels) == 1, f"Expected exactly 1 wheel, got: {wheels}"
    return wheels[0]

@pytest.fixture
def clean_install_env(tmp_path, clean_install_whl):
    venv_dir = tmp_path / "clean-env"
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    venv_bin = venv_dir / "bin"

    clean_env = {k: v for k, v in os.environ.items() if k not in _MASKING_VARS}
    clean_env["HOME"] = str(tmp_path)
    clean_env["IAI_MCP_STORE"] = str(tmp_path / ".iai-mcp")

    subprocess.run(
        [str(venv_bin / "pip"), "install", str(clean_install_whl), "--quiet"],
        env=clean_env,
        check=True,
    )
    return venv_bin, clean_env

def _assert_not_masked(venv_bin: Path, clean_env: dict) -> None:
    result = subprocess.run(
        [
            str(venv_bin / "python"),
            "-c",
            "import iai_mcp; print(iai_mcp.__file__)",
        ],
        capture_output=True,
        text=True,
        env=clean_env,
        check=True,
    )
    pkg_file = result.stdout.strip()
    assert "site-packages" in pkg_file, (
        f"iai_mcp loaded from wrong location — masking still active: {pkg_file}"
    )
    assert str(venv_bin.parent) in pkg_file, (
        f"iai_mcp not loaded from the clean venv: {pkg_file}"
    )

@pytest.mark.slow
def test_wheel_builds_and_installs_clean(clean_install_env):
    venv_bin, clean_env = clean_install_env
    _assert_not_masked(venv_bin, clean_env)

@pytest.mark.slow
def test_daemon_install_finds_plist(clean_install_env, tmp_path):
    venv_bin, clean_env = clean_install_env
    _assert_not_masked(venv_bin, clean_env)
    env = {
        **clean_env,
        "IAI_DAEMON_SOCKET_PATH": str(tmp_path / "no.sock"),
        "IAI_MCP_CRYPTO_PASSPHRASE": "test-passphrase",
    }
    result = subprocess.run(
        [str(venv_bin / "iai-mcp"), "daemon", "install", "--dry-run", "--yes"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"daemon install --dry-run --yes failed:\n{result.stderr}"
    )
    assert "com.iai-mcp.daemon" in result.stdout, (
        "Plist content not printed by --dry-run"
    )
    assert str(venv_bin / "python") in result.stdout, (
        "Venv python not found in rendered plist"
    )
    assert "/usr/local/bin/python3" not in result.stdout, (
        "Hard-coded system python3 still present in rendered plist"
    )

@pytest.mark.slow
def test_capture_hooks_install_finds_hooks(clean_install_env, tmp_path):
    venv_bin, clean_env = clean_install_env
    _assert_not_masked(venv_bin, clean_env)
    result = subprocess.run(
        [str(venv_bin / "iai-mcp"), "capture-hooks", "install"],
        capture_output=True,
        text=True,
        env=clean_env,
    )
    assert result.returncode == 0, (
        f"capture-hooks install failed:\n{result.stderr}"
    )
    hooks_dir = tmp_path / ".claude" / "hooks"
    assert (hooks_dir / "iai-mcp-session-capture.sh").exists(), (
        "session-capture hook not installed"
    )
    assert (hooks_dir / "iai-mcp-turn-capture.sh").exists(), (
        "turn-capture hook not installed"
    )
    assert (hooks_dir / "iai-mcp-session-recall.sh").exists(), (
        "session-recall hook not installed"
    )

    import json

    claude_json = tmp_path / ".claude.json"
    assert claude_json.exists(), ".claude.json not written by capture-hooks install"
    data = json.loads(claude_json.read_text())
    iai_mcp_python = (
        data.get("mcpServers", {}).get("iai-mcp", {}).get("env", {}).get("IAI_MCP_PYTHON", "")
    )
    venv_root = str(venv_bin.parent)
    assert iai_mcp_python.startswith(venv_root), (
        f"IAI_MCP_PYTHON={iai_mcp_python!r} does not start with venv root {venv_root!r}"
    )
    assert str(_REPO_ROOT) not in iai_mcp_python, (
        f"IAI_MCP_PYTHON points into the source repo: {iai_mcp_python!r}"
    )

@pytest.mark.slow
def test_session_capture_hook_candidates(clean_install_env, tmp_path):
    venv_bin, clean_env = clean_install_env
    _assert_not_masked(venv_bin, clean_env)
    result = subprocess.run(
        [str(venv_bin / "iai-mcp"), "capture-hooks", "install"],
        capture_output=True,
        text=True,
        env=clean_env,
    )
    assert result.returncode == 0, (
        f"capture-hooks install failed:\n{result.stderr}"
    )
    hook_path = tmp_path / ".claude" / "hooks" / "iai-mcp-session-capture.sh"
    assert hook_path.exists(), "session-capture hook file not created"
    hook_text = hook_path.read_text()
    assert "command -v iai-mcp" in hook_text, (
        "Hook does not probe PATH for iai-mcp CLI"
    )
    assert ".pyenv/shims/iai-mcp" in hook_text, (
        "Hook does not fall back to pyenv shim path"
    )

@pytest.mark.slow
def test_wheel_contains_wrapper_and_rust_ext(clean_install_env):
    venv_bin, clean_env = clean_install_env
    _assert_not_masked(venv_bin, clean_env)
    probe = (
        "import importlib.resources as r, pathlib\n"
        "wrapper = pathlib.Path(str(r.files('iai_mcp') / '_wrapper' / 'index.js'))\n"
        "print('wrapper_exists:', wrapper.exists())\n"
        "print('bridge_exists:', (wrapper.parent / 'bridge.js').exists())\n"
        "siblings = list(wrapper.parent.glob('*.js'))\n"
        "print('js_count:', len(siblings))\n"
        "import iai_mcp_native\n"
        "print('rust_ext_ok:', True)\n"
    )
    result = subprocess.run(
        [str(venv_bin / "python"), "-c", probe],
        capture_output=True,
        text=True,
        env=clean_env,
    )
    out = result.stdout
    assert "wrapper_exists: True" in out, (
        f"_wrapper/index.js absent from wheel:\n{result.stderr}"
    )
    assert "bridge_exists: True" in out, (
        "_wrapper/bridge.js absent from wheel"
    )
    assert "js_count: 7" in out, (
        f"Expected 7 JS files in _wrapper/, got:\n{out}"
    )
    assert "rust_ext_ok: True" in out, (
        "Rust extension import failed inside clean venv"
    )

@pytest.mark.slow
def test_fresh_editable_install_resolver(tmp_path_factory, clean_install_whl):
    editable_tmp = tmp_path_factory.mktemp("editable", numbered=True)

    venv_dir = editable_tmp / "editable-env"
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    venv_bin = venv_dir / "bin"

    clean_env = {k: v for k, v in os.environ.items() if k not in _MASKING_VARS}
    clean_env["HOME"] = str(editable_tmp)
    clean_env["IAI_MCP_STORE"] = str(editable_tmp / ".iai-mcp")
    # HOME is masked in clean_env above, so recover the real home portably
    # (pwd/getuid do not exist on Windows). Path.home() reads the unmasked
    # process environment, which still holds the real value here.
    _real_home = str(Path.home())
    clean_env.setdefault("RUSTUP_HOME", os.path.join(_real_home, ".rustup"))
    clean_env.setdefault("CARGO_HOME", os.path.join(_real_home, ".cargo"))

    result = subprocess.run(
        [str(venv_bin / "pip"), "install", "-e", str(_REPO_ROOT), "--quiet"],
        env=clean_env,
        check=False,
    )
    assert result.returncode == 0, (
        f"pip install -e failed:\n{getattr(result, 'stderr', '')}"
    )

    result2 = subprocess.run(
        [
            str(venv_bin / "python"),
            "-c",
            "from iai_mcp.cli import _resolve_wrapper_path; print(_resolve_wrapper_path())",
        ],
        capture_output=True,
        text=True,
        env=clean_env,
    )
    assert result2.returncode == 0, (
        f"_resolve_wrapper_path() failed:\n{result2.stderr}"
    )
    resolved = result2.stdout.strip()
    assert resolved.endswith("mcp-wrapper/dist/index.js"), (
        f"Editable resolver returned unexpected path: {resolved!r}"
    )

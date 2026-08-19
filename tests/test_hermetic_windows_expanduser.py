from __future__ import annotations

import ntpath
import os
import pwd
from pathlib import Path


def _under(child: Path, ancestor: Path) -> bool:
    child = Path(child).resolve()
    ancestor = Path(ancestor).resolve()
    return child == ancestor or ancestor in child.parents


def test_posix_home_still_sandboxed() -> None:
    assert Path.home() != Path(pwd.getpwuid(os.getuid()).pw_dir)


def test_windows_order_sandboxed_under_fixture() -> None:
    base = Path(os.environ["HOME"])
    assert "USERPROFILE" in os.environ and _under(Path(os.environ["USERPROFILE"]), base)
    assert _under(Path(ntpath.expanduser("~")), base)


def test_windows_fallback_without_userprofile(monkeypatch) -> None:
    base = Path(os.environ["HOME"])
    monkeypatch.delenv("USERPROFILE", raising=False)
    assert _under(Path(ntpath.expanduser("~")), base)


def test_helper_neutralizes_outside_decoy(tmp_path, monkeypatch) -> None:
    from conftest import _sandbox_home_env

    monkeypatch.setenv("USERPROFILE", "/outside/decoy")
    monkeypatch.setenv("HOMEDRIVE", "")
    monkeypatch.setenv("HOMEPATH", "/outside/decoy")
    _sandbox_home_env(monkeypatch, tmp_path)
    assert _under(Path(ntpath.expanduser("~")), tmp_path)


def test_home_sandbox_call_stays_after_hf_block() -> None:
    src = (Path(__file__).parent / "conftest.py").read_text()
    assert src.index('setenv("HF_HOME"') < src.index("_sandbox_home_env(monkeypatch, base)")

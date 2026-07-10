
from __future__ import annotations

from pathlib import Path

import pytest

def test_missing_wrapper_raises_actionable(tmp_path, monkeypatch):
    import iai_mcp.cli as _cli_module

    monkeypatch.delenv("IAI_MCP_WRAPPER_PATH", raising=False)

    import importlib.resources as _res

    original_files = _res.files

    def _fake_files(package):
        if package == "iai_mcp":
            return original_files.__class__
        return original_files(package)

    fake_pkg_init = tmp_path / "iai_mcp" / "__init__.py"
    fake_pkg_init.parent.mkdir(parents=True, exist_ok=True)
    fake_pkg_init.write_text("")

    import iai_mcp as _pkg
    monkeypatch.setattr(_pkg, "__file__", str(fake_pkg_init))

    monkeypatch.setattr(_cli_module, "iai_mcp", _pkg, raising=False)  # type: ignore[attr-defined]


    class _FakeTraversable:

        def __init__(self, base: Path):
            self._base = base

        def __truediv__(self, child: str) -> "_FakeTraversable":
            return _FakeTraversable(self._base / child)

        def __str__(self) -> str:
            return str(self._base)

        def exists(self) -> bool:
            return self._base.exists()


    original_res_files = None
    try:
        import importlib.resources as _ir

        original_res_files = _ir.files

        def _fake_ir_files(pkg):
            if pkg == "iai_mcp":
                return _FakeTraversable(tmp_path)
            return original_res_files(pkg)  # type: ignore[misc]

        monkeypatch.setattr(_ir, "files", _fake_ir_files)
    except Exception:
        pass

    from iai_mcp.cli import _resolve_wrapper_path  # type: ignore[attr-defined]

    with pytest.raises(FileNotFoundError) as exc_info:
        _resolve_wrapper_path()

    msg = str(exc_info.value)
    assert "npm run build" in msg, (
        f"Error message lacks 'npm run build' instruction:\n{msg}"
    )
    assert "scripts/install.sh" in msg, (
        f"Error message lacks 'scripts/install.sh' instruction:\n{msg}"
    )

def test_env_override_wins(tmp_path, monkeypatch):
    fake_wrapper = tmp_path / "index.js"
    fake_wrapper.write_text("// stub\n")

    monkeypatch.setenv("IAI_MCP_WRAPPER_PATH", str(fake_wrapper))

    from iai_mcp.cli import _resolve_wrapper_path  # type: ignore[attr-defined]

    result = _resolve_wrapper_path()
    assert result == fake_wrapper, (
        f"Env override not respected: got {result!r}, expected {fake_wrapper!r}"
    )

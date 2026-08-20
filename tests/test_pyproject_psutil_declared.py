from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"

def test_psutil_declared_in_project_dependencies() -> None:
    text = PYPROJECT.read_text()
    project_marker = text.find("\n[project]")
    if project_marker < 0:
        project_marker = text.find("[project]") if text.startswith("[project]") else -1
    assert project_marker >= 0, "[project] block not found in pyproject.toml"
    next_section = text.find("\n[", project_marker + len("\n[project]"))
    section_end = next_section if next_section >= 0 else len(text)
    project_block = text[project_marker:section_end]
    assert "psutil" in project_block, (
        "psutil missing from [project] block. The declaration exists so a "
        "clean `pip install -e .` reaches psutil at runtime. Restore the line."
    )
    import re
    match = re.search(r'"\s*psutil\s*>=\s*\d+', project_block)
    assert match, (
        'Expected `"psutil>=X.Y.Z"` style declaration in [project] '
        "dependencies."
    )

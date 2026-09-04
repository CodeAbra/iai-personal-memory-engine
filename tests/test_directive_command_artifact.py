"""The shipped /iai-directive command artifact must invoke the CLI
mechanism with the exact flag name -- pinned here so the wrapper can never
silently drift from `iai capture --directive`.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ARTIFACT = (
    _REPO_ROOT / "src" / "iai_mcp" / "_deploy" / "commands" / "iai-directive.md"
)


def test_command_artifact_exists() -> None:
    assert _ARTIFACT.is_file(), f"missing command artifact: {_ARTIFACT}"


def test_command_artifact_invokes_the_exact_cli_mechanism() -> None:
    text = _ARTIFACT.read_text(encoding="utf-8")
    assert "capture --directive" in text, (
        "artifact must invoke the exact `capture --directive` flag"
    )
    assert "$ARGUMENTS" in text, (
        "artifact must forward the user's typed rule via $ARGUMENTS"
    )

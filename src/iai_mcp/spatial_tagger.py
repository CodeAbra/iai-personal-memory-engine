

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

DEFAULT_WINGS: frozenset[str] = frozenset(
    {
        "IAI-MCP",
        "iai-mcp",
        "Documents",
        "Projects",
        "src",
        "code",
        "repos",
    }
)

DESKTOP_MARKER: str = "Desktop"


def tag(
    record: Any,
    source_path: str | None,
    default_wing: str = "general",
) -> tuple[str | None, str | None, str | None]:
    if source_path is None or not source_path.strip():
        return (None, None, None)

    path = PurePosixPath(source_path)
    components = path.parts

    wing: str | None = None

    for i, part in enumerate(components):
        if part == DESKTOP_MARKER and i + 1 < len(components):
            wing = components[i + 1]
            break

    if wing is None and any(
        part in {"IAI-MCP", "iai-mcp", "iai_mcp"} for part in components
    ):
        wing = "IAI-MCP"

    if wing is None:
        for part in components:
            if part in DEFAULT_WINGS:
                wing = part
                break

    if wing is None:
        wing = default_wing

    parent_name = path.parent.name
    room: str | None = parent_name if parent_name else None

    stem = path.stem
    drawer: str | None = stem if stem else None

    return (wing, room, drawer)


class SpatialTagger:

    @classmethod
    def tag(
        cls,
        record: Any,
        source_path: str | None,
        default_wing: str = "general",
    ) -> tuple[str | None, str | None, str | None]:
        return tag(record, source_path, default_wing=default_wing)

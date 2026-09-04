"""Content stamp of the source roots a long-lived daemon pins at boot."""

from __future__ import annotations

import hashlib
from pathlib import Path

# Modules under these roots are imported once at daemon boot and their
# functions bound onto long-lived objects; a file landing here afterwards
# runs stale until the process restarts.
PINNED_SOURCE_ROOTS: tuple[tuple[str, ...], ...] = (
    ("lilli", "cycle", "sleep_pipeline"),
)


def pinned_source_roots() -> list[Path]:
    import iai_mcp

    base = Path(iai_mcp.__file__).resolve().parent
    return [base.joinpath(*parts) for parts in PINNED_SOURCE_ROOTS]


def compute_code_stamp() -> dict:
    digest = hashlib.sha256()
    roots: list[str] = []
    files = 0
    for root in pinned_source_roots():
        roots.append(str(root))
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
            files += 1
    return {"roots": roots, "digest": digest.hexdigest(), "files": files}


def stamp_divergence(boot_stamp: object) -> str | None:
    """None when the boot stamp still matches the source on disk."""
    if not isinstance(boot_stamp, dict) or not boot_stamp.get("digest"):
        return "missing"
    current = compute_code_stamp()
    if boot_stamp.get("roots") != current["roots"]:
        return "roots"
    if boot_stamp["digest"] != current["digest"]:
        return "digest"
    return None

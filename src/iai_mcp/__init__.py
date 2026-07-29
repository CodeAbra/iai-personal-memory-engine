"""IAI-MCP -- autistic-style persistent memory MCP server."""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import tomllib

from iai_mcp.types import (
    MemoryRecord,
    MemoryHit,
    RecallResponse,
    EdgeUpdate,
    ReconsolidationReceipt,
    TIER_ENUM,
)

def _distribution_version() -> str:
    """Resolve the installed version from wheel metadata. The distribution
    is `iai-pme`; the old name keeps pre-rename editable installs honest."""
    for _dist in ("iai-pme", "iai-mcp"):
        try:
            return version(_dist)
        except PackageNotFoundError:
            continue
    return "0+unknown"


_pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
if _pyproject.is_file():
    __version__ = tomllib.loads(_pyproject.read_text())["project"]["version"]
else:
    __version__ = _distribution_version()
__all__ = [
    "MemoryRecord",
    "MemoryHit",
    "RecallResponse",
    "EdgeUpdate",
    "ReconsolidationReceipt",
    "TIER_ENUM",
]

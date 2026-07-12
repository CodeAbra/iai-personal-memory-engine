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

_pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
if _pyproject.is_file():
    __version__ = tomllib.loads(_pyproject.read_text())["project"]["version"]
else:
    try:
        __version__ = version("iai-mcp")
    except PackageNotFoundError:
        __version__ = "0+unknown"
__all__ = [
    "MemoryRecord",
    "MemoryHit",
    "RecallResponse",
    "EdgeUpdate",
    "ReconsolidationReceipt",
    "TIER_ENUM",
]

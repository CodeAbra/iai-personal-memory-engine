from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


def _native_available() -> bool:
    try:
        import iai_mcp_native  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _native_available(), reason="iai_mcp_native wheel not installed"
)
def test_embed_submodule_importable() -> None:
    from iai_mcp_native import embed
    assert embed.Embedder is not None


@pytest.mark.skipif(
    not _native_available(), reason="iai_mcp_native wheel not installed"
)
def test_graph_submodule_importable() -> None:
    from iai_mcp_native import graph
    assert callable(graph.answer)


@pytest.mark.skipif(
    not _native_available(), reason="iai_mcp_native wheel not installed"
)
def test_graph_answer_returns_42() -> None:
    from iai_mcp_native import graph
    assert graph.answer() == 42


@pytest.mark.skipif(
    not _native_available(), reason="iai_mcp_native wheel not installed"
)
def test_dotted_imports_register_sys_modules() -> None:
    import iai_mcp_native  # noqa: F401
    import iai_mcp_native.embed  # noqa: F401
    import iai_mcp_native.graph  # noqa: F401
    assert "iai_mcp_native.embed" in sys.modules
    assert "iai_mcp_native.graph" in sys.modules


@pytest.mark.skipif(
    not _native_available(), reason="iai_mcp_native wheel not installed"
)
def test_both_submodules_share_one_so() -> None:
    from iai_mcp_native import embed as _embed
    from iai_mcp_native import graph as _graph
    import iai_mcp_native

    pkg_dir = Path(iai_mcp_native.__file__).parent
    binaries = [
        f for f in os.listdir(pkg_dir)
        if f.endswith((".so", ".dylib"))
    ]
    assert len(binaries) == 1, (
        f"expected exactly one extension binary in {pkg_dir}, got {binaries}"
    )

    assert id(_embed) == id(iai_mcp_native.embed) == id(sys.modules["iai_mcp_native.embed"])
    assert id(_graph) == id(iai_mcp_native.graph) == id(sys.modules["iai_mcp_native.graph"])

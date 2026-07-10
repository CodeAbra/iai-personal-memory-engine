from __future__ import annotations


def _require_native() -> None:
    try:
        import iai_mcp_native  # noqa: F401
        from iai_mcp_native import embed as _e, graph as _g, hd as _h

        assert callable(_g.is_connected), "iai_mcp_native.graph.is_connected not callable"
        assert callable(_e.Embedder), "iai_mcp_native.embed.Embedder not callable"
        assert callable(_h.project), "iai_mcp_native.hd.project not callable"
    except Exception as exc:
        raise RuntimeError(
            "iai_mcp_native (the Rust native extension) is not available.\n"
            "\n"
            "The Rust embedder and graph algorithms have no Python fallback.\n"
            "A missing or broken native module prevents the daemon and the\n"
            "MCP server from starting.\n"
            "\n"
            "To rebuild the extension, run:\n"
            "\n"
            "    iai-mcp build-native\n"
            "\n"
            "Then restart the daemon or MCP server.\n"
            "\n"
            "Advanced / manual build from a source checkout:\n"
            "\n"
            "    cd rust/iai_mcp_native && maturin develop --release"
        ) from exc

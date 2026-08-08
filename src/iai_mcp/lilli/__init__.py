"""Lilli/HD -- memory-architecture HDC library.

Multi-tier (BSC episodic D=4096, FHRR semantic D=10000 uint8, Sparse VSA procedural D=2048).
Pure numpy.

Public API:
    from iai_mcp.lilli import Brain, tier_info, from_embedding, to_embedding_neighbors
"""
from __future__ import annotations

__author__ = "Areg Noya"

from iai_mcp.lilli.brain import Brain
from iai_mcp.lilli.tier_info import tier_info, list_tiers
from iai_mcp.lilli.crossmodal.embed_to_hv import from_embedding, to_embedding_neighbors

__all__ = [
    "Brain",
    "tier_info",
    "list_tiers",
    "from_embedding",
    "to_embedding_neighbors",
]

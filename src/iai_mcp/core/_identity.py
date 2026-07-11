"""L0 identity seed (idempotent, one-time at boot)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from uuid import UUID

from iai_mcp.aaak import enforce_english_raw, generate_aaak_index
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord

L0_ID = UUID("00000000-0000-0000-0000-000000000001")


_DEFAULT_L0_SEED = (
    "User identity not yet configured. "
    "Run `iai-mcp config identity` to set your name, language, and role."
)


def _load_l0_identity_seed() -> str:
    config_path = os.path.join(
        os.environ.get("IAI_MCP_STORE", os.path.expanduser("~/.iai-mcp")),
        "config.json",
    )
    if os.path.isfile(config_path):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            identity = cfg.get("identity", {})
            parts = []
            if identity.get("name"):
                parts.append(f"User: {identity['name']}.")
            if identity.get("languages"):
                parts.append(f"Primary languages: {identity['languages']}.")
            if identity.get("role"):
                parts.append(f"Role: {identity['role']}.")
            if identity.get("project"):
                parts.append(f"Active project: {identity['project']}.")
            if identity.get("extra"):
                parts.append(identity["extra"])
            if parts:
                return " ".join(parts)
        except (json.JSONDecodeError, OSError):
            pass
    return _DEFAULT_L0_SEED


def _seed_l0_identity(store: MemoryStore) -> None:
    existing = store.get(L0_ID)
    if existing is not None:
        return
    now = datetime.now(timezone.utc)
    seed_dim = store.embed_dim
    surface = _load_l0_identity_seed()
    # The identity anchor must be semantically findable: a zero-embedded seed
    # has cosine ~0 to every cue and never surfaces on identity questions.
    # Embed failure degrades to zeros (seeding must not block store init) —
    # the record stays verbatim/lexically findable either way.
    try:
        from iai_mcp.embed import embedder_for_store
        seed_embedding = list(embedder_for_store(store).embed(surface))
    except Exception:  # noqa: BLE001 -- init must not fail on the embedder
        seed_embedding = [0.0] * seed_dim
    seed = MemoryRecord(
        id=L0_ID,
        tier="semantic",
        literal_surface=surface,
        aaak_index="",
        embedding=seed_embedding,
        community_id=None,
        centrality=1.0,
        detail_level=5,
        pinned=True,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=True,
        never_merge=True,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=["identity", "l0", "pinned"],
        language="en",
    )
    enforce_english_raw(seed)
    seed.aaak_index = generate_aaak_index(seed)
    store.insert(seed)

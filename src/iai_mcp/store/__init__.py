from __future__ import annotations

import asyncio
import base64
import enum
import functools
import json
import os
import random
import re
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Callable, Union
from uuid import UUID

import logging

from iai_mcp.hippo import _REAL_IAI_ROOT, AccessMode, HippoDB, HippoIntegrityError
from iai_mcp.model_attribution import normalize_model

CPU_HAS_AVX2: bool = True


from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from iai_mcp.crypto import (
    CIPHERTEXT_PREFIX,
    NONCE_BYTES,
    CryptoKey,
    encrypt_field,
    is_encrypted,
)
from iai_mcp.exceptions import (
    StoreCorruptionError,
    StoreInsertError,
    StoreQueryError,
    StoreSchemaError,
)
from iai_mcp.types import (
    DEFAULT_EMBED_DIM,
    EMBED_DIM,
    HV_TIER_ENUM,
    SCHEMA_VERSION_CURRENT,
    MemoryRecord,
    TIER_ENUM,
)

logger = logging.getLogger(__name__)

DEFAULT_STORAGE_PATH = Path.home() / ".iai-mcp"

RECORDS_TABLE = "records"
EDGES_TABLE = "edges"

EVENTS_TABLE = "events"
BUDGET_TABLE = "budget_ledger"
RATELIMIT_TABLE = "ratelimit_ledger"

_STC_TIER_ORDER: dict[str, int] = {"semantic": 0, "episodic": 1, "procedural": 2}

EDGE_TYPES: frozenset[str] = frozenset({
    "hebbian",
    "contradicts",
    "consolidated_from",
    "schema_instance_of",
    "temporal_next",
    "invariant_anchor",
    "curiosity_bridge",
    "profile_modulates",
    "hebbian_structure",
    "pattern_separation_seed",
    "hebbian_cluster_replay",
    "entity_shared",
})


class GateAction(enum.Enum):
    SKIP = "skip"
    INSERT = "insert"


GatePayload = Union[UUID, list[tuple[UUID, float]]]


_UUID_STR_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def _uuid_literal(value: UUID | str) -> str:
    s = str(value).lower()
    if not _UUID_STR_RE.match(s):
        raise ValueError(f"not a canonical UUID: {value!r}")
    return s


def _resolve_embed_dim() -> int:
    env_dim = os.environ.get("IAI_MCP_EMBED_DIM")
    if env_dim:
        try:
            return int(env_dim)
        except ValueError:
            pass
    return DEFAULT_EMBED_DIM


class _PendingTurn:

    __slots__ = (
        "_text",
        "_session_id",
        "_ts",
        "_idem_tag",
        "_source_uuid",
        "_role",
        "_model",
    )

    def __init__(
        self,
        *,
        text: str,
        session_id: str,
        ts: "datetime",
        idem_tag: str,
        source_uuid: "str | None",
        role: str = "user",
        model: object = None,
    ) -> None:
        self._text = text
        self._session_id = session_id
        self._ts = ts
        self._idem_tag = idem_tag
        self._source_uuid = source_uuid
        self._role = role
        self._model = normalize_model(model)

    @property
    def id(self):
        return None

    @property
    def tier(self) -> str:
        return "episodic"

    @property
    def literal_surface(self) -> str:
        return self._text

    @property
    def tags(self) -> list:
        return [f"role:{self._role}", self._idem_tag]

    @property
    def provenance(self) -> list:
        prov: dict = {"session_id": self._session_id, "role": self._role}
        if self._source_uuid is not None:
            prov["source_uuid"] = self._source_uuid
        if self._model is not None:
            prov["model"] = self._model
        return [prov]

    @property
    def created_at(self):
        return self._ts

    @property
    def _pending_idem_tag(self) -> str:
        return self._idem_tag

    @property
    def _pending_source_uuid(self) -> "str | None":
        return self._source_uuid


from iai_mcp.store._buffers import (
    _record_buffer, _record_last_flush_at,
    flush_record_buffer, should_flush_record_buffer, should_flush_record_buffer_by_time,
    _edge_buffer, _edge_last_flush_at,
    flush_edge_buffer, should_flush_edge_buffer, should_flush_edge_buffer_by_time,
)

from iai_mcp.store._store import MemoryStore

__all__ = [
    "MemoryStore",
    "DEFAULT_STORAGE_PATH",
    "AESGCM",
    "RECORDS_TABLE",
    "EDGES_TABLE",
    "EVENTS_TABLE",
    "BUDGET_TABLE",
    "RATELIMIT_TABLE",
    "EDGE_TYPES",
    "_STC_TIER_ORDER",
    "GateAction",
    "GatePayload",
    "_UUID_STR_RE",
    "_uuid_literal",
    "_resolve_embed_dim",
    "_PendingTurn",
    "EMBED_DIM",
    "DEFAULT_EMBED_DIM",
    "_record_buffer",
    "_record_last_flush_at",
    "flush_record_buffer",
    "should_flush_record_buffer",
    "should_flush_record_buffer_by_time",
    "_edge_buffer",
    "_edge_last_flush_at",
    "flush_edge_buffer",
    "should_flush_edge_buffer",
    "should_flush_edge_buffer_by_time",
]

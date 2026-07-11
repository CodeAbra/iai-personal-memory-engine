"""Compression is allowed ONLY on retrieval views and summaries, NEVER on raw
content. Enforcement lives in `is_compressible`:

Forbidden:
- pinned records (includes L0 identity)
- invariant_anchor records (s5_trust_score >= 0.9)
- user-tagged raw: records (raw:en, raw:ru, ...)
- normal episodic records (default reject; literal_surface is constitutional

Allowed:
- records tagged cls_summary (CLS consolidation output)
- records tagged schema (LEARN-03 induction output)
- records tagged session_summary

Runtime fallback: when `llmlingua` is not installed, `compress_llmlingua2`
returns the input unchanged and emits an llm_health event. This keeps the
Tier-0 path green on minimal installs (CI, fresh user machines).

Constants:
- COMPRESSION_TARGET_L2 = 0.5 (community descriptors)
- COMPRESSION_TARGET_SUMMARY = 0.3 (session summaries)
"""
from __future__ import annotations

import logging
import os
import platform
import shutil
import sys
import threading
from typing import Any

from iai_mcp.events import write_event

logger = logging.getLogger(__name__)


COMPRESSION_TARGET_L2 = 0.5
COMPRESSION_TARGET_SUMMARY = 0.3

INVARIANT_TRUST_THRESHOLD = 0.9


def is_compressible(record) -> tuple[bool, str]:
    """Return (allowed, reason) for a given MemoryRecord.

    Reason is a short English diagnostic consumed only in tests / debug logs.
    """
    if getattr(record, "pinned", False):
        return False, "pinned record (D-14 L0 / user-pinned)"

    trust = getattr(record, "s5_trust_score", 0.5)
    try:
        if float(trust) >= INVARIANT_TRUST_THRESHOLD:
            return False, (
                f"invariant anchor (trust={float(trust):.2f} >= "
                f"{INVARIANT_TRUST_THRESHOLD}); D-22 forbids compression"
            )
    except (TypeError, ValueError):
        pass

    tags = getattr(record, "tags", None) or []
    for tag in tags:
        if tag.startswith("raw:"):
            return False, f"raw-tagged record ({tag}); user flagged as raw"

    # Explicit allowlist.
    allow_tags = {"cls_summary", "schema", "session_summary"}
    for tag in tags:
        if tag in allow_tags:
            return True, ""

    return False, "literal_surface constitutional (D-25 default deny)"


_LLMLINGUA_LOCK = threading.Lock()
_LLMLINGUA_CACHE: dict[str, Any] = {}


def _load_llmlingua2():
    """Lazy-load llmlingua's PromptCompressor (LLMLingua-2 model).

    Returns the compressor instance on success; None if the package is absent
    or fails to instantiate. Callers log a fallback event and passthrough.
    """
    with _LLMLINGUA_LOCK:
        if "instance" in _LLMLINGUA_CACHE:
            return _LLMLINGUA_CACHE["instance"]
        try:
            from llmlingua import PromptCompressor  # type: ignore
        except (ImportError, ModuleNotFoundError):
            _LLMLINGUA_CACHE["instance"] = None
            return None
        try:
            # Device auto-detection without torch: Apple Silicon -> mps,
            # Linux with NVIDIA -> cuda, everything else -> cpu.
            # llmlingua defaults to CUDA which breaks on macOS ARM64.
            if sys.platform == "darwin" and platform.machine() == "arm64":
                device_map = "mps"
            elif os.path.exists("/dev/nvidia0") or shutil.which("nvcc") is not None:
                device_map = "cuda"
            else:
                device_map = "cpu"
            # microsoft/llmlingua-2-xlm-roberta-large-meetingbank (default in
            # llmlingua>=0.2).
            compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                use_llmlingua2=True,
                device_map=device_map,
            )
        except (OSError, RuntimeError, ImportError, ValueError) as exc:
            logger.debug("llmlingua_init_failed", extra={"err": str(exc)[:120]})
            _LLMLINGUA_CACHE["instance"] = None
            return None
        _LLMLINGUA_CACHE["instance"] = compressor
        return compressor


def compress_llmlingua2(
    text: str,
    target_ratio: float = 0.5,
    store=None,
) -> str:
    """Compress `text` to approximately `target_ratio` of original tokens.

        On any failure (package missing, model load error, runtime exception):
        - Return `text` unchanged (passthrough).
        - If `store` is provided, emit an llm_health event of kind
          'compression_fallback' with severity='warning'.

        consulted BEFORE reaching this function).
    """
    if not text:
        return text

    compressor = _load_llmlingua2()
    if compressor is None:
        if store is not None:
            try:
                write_event(
                    store,
                    kind="llm_health",
                    data={
                        "component": "compress_llmlingua2",
                        "tier": "fallback",
                        "reason": "llmlingua package unavailable or model load failed",
                    },
                    severity="warning",
                )
            except (OSError, RuntimeError, ValueError):
                pass
        return text

    try:
        result = compressor.compress_prompt(text, rate=float(target_ratio))
        if isinstance(result, dict):
            return str(result.get("compressed_prompt", text))
        return str(result)
    except (RuntimeError, ValueError, TypeError) as exc:  # pragma: no cover -- runtime failure passthrough
        logger.warning("compress_llmlingua2_failed", extra={"err": str(exc)[:120]})
        if store is not None:
            try:
                write_event(
                    store,
                    kind="llm_health",
                    data={
                        "component": "compress_llmlingua2",
                        "tier": "fallback",
                        "error": str(exc),
                    },
                    severity="warning",
                )
            except (OSError, RuntimeError, ValueError):
                pass
        return text


def compress_l2_descriptor(descriptor: str, store=None) -> str:
    """"""
    return compress_llmlingua2(
        descriptor, target_ratio=COMPRESSION_TARGET_L2, store=store,
    )


def compress_summary(summary: str, store=None) -> str:
    """"""
    return compress_llmlingua2(
        summary, target_ratio=COMPRESSION_TARGET_SUMMARY, store=store,
    )


# Recall payload compression: compress the assembled recall response
# text crossing the MCP wire. Does NOT modify stored literal_surface
COMPRESSION_TARGET_PAYLOAD = 0.5


def compress_recall_payload(hits_text: str, store=None) -> str:
    """Compress assembled recall payload for MCP wire transfer.

    Target: ~50% compression (3000 tokens → ~1500 high-density shards).
    Preserves named entities, code symbols, and key facts via LLMLingua-2's
    token-level importance scoring.

    Falls back to uncompressed on any failure (graceful degradation).
    """
    if not hits_text or len(hits_text) < 200:
        return hits_text
    return compress_llmlingua2(
        hits_text, target_ratio=COMPRESSION_TARGET_PAYLOAD, store=store,
    )

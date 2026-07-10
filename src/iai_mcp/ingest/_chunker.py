"""Token-based sliding-window chunker for file ingestion.

Emits ~target_tokens-sized windows with overlap_tokens carry-over so each
chunk has shared context with its neighbours. The cl100k_base encoder is
instantiated once at module scope (lazy-init on first call); per-chunk
re-instantiation would reload the vocab on every invocation.

Short-tail merge: the capture spine drops sub-12-char text outright
(``capture.MIN_CAPTURE_LEN`` at ``src/iai_mcp/capture.py:33``); a trailing
sub-12-char chunk is folded into its predecessor so verbatim/lossless
ingestion is honoured. A whole-document shorter than that threshold
returns an empty list — the caller surfaces the no-chunks outcome.
"""
from __future__ import annotations

import re

import tiktoken

_CL100K = None


def _encoder():
    global _CL100K
    if _CL100K is None:
        _CL100K = tiktoken.get_encoding("cl100k_base")
    return _CL100K


_SENTENCE_BOUNDARY = re.compile(r"\.\s+")


def chunk_text(
    text: str, *, target_tokens: int = 256, overlap: int = 48
) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    enc = _encoder()
    token_ids = enc.encode(text)
    if not token_ids:
        return []
    chunks: list[str] = []
    base_stride = max(1, target_tokens - overlap)
    i = 0
    while i < len(token_ids):
        window = token_ids[i : i + target_tokens]
        chunk_str = enc.decode(window).strip()
        snapped = False
        snapped_consumed = base_stride  # default: as if no snap
        if len(window) >= target_tokens:
            m = list(_SENTENCE_BOUNDARY.finditer(chunk_str))
            if m and m[-1].end() >= len(chunk_str) // 2:
                chunk_str = chunk_str[: m[-1].end()].rstrip()
                # Re-encode the snapped text to learn how many tokens it
                # represents; the next window resumes from the snap point
                # (keeping the intended overlap) so no tokens fall in the gap
                # between the snap boundary and the next window start.
                snapped_consumed = len(enc.encode(chunk_str))
                snapped = True
        if chunk_str:
            chunks.append(chunk_str)
        if snapped:
            # Advance by the snapped token count minus overlap, but never
            # past the normal stride and never less than 1.
            step = min(base_stride, max(1, snapped_consumed - overlap))
        else:
            # Un-snapped window (full window or partial last window):
            # use the normal stride so short documents do not loop.
            step = base_stride
        i += step

    MIN = 12
    if len(chunks) >= 2 and len(chunks[-1]) < MIN:
        chunks[-2] = (chunks[-2] + " " + chunks[-1]).strip()
        chunks.pop()
    if len(chunks) == 1 and len(chunks[0]) < MIN:
        return []
    return chunks

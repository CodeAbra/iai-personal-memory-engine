"""Resident lexical (identifier-grade) index over decrypted surfaces.

Semantic similarity is weak exactly where code work is strongest: exact
identifiers (`_hippea_cascade_loop`, `IAI_MCP_FORESIGHT_OFF`). This index
gives the awake store a lexical lane beside the semantic one: an inverted
token map built in RAM from decrypted surfaces — plaintext lives only in
process memory, never on disk, so the at-rest encryption posture is
untouched.

Freshness model: the index remembers the corpus-count-cache generation it
was built against and rebuilds on demand when the generation moved (any
corpus-changing write bumps it). Queries are AND-of-tokens ranked by term
frequency; camelCase and snake_case identifiers match both whole and by
their parts.
"""
from __future__ import annotations

import logging
import re
import threading
from typing import Any

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}|[0-9]{3,}")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

MAX_QUERY_TOKENS = 8


def tokenize(text: str) -> list[str]:
    """Whole identifiers plus their snake/camel parts, lowercased."""
    out: list[str] = []
    for raw in _TOKEN_RE.findall(text or ""):
        low = raw.lower()
        out.append(low)
        parts = [p for p in low.split("_") if len(p) > 2]
        if len(parts) > 1:
            out.extend(parts)
        camel = [p.lower() for p in _CAMEL_RE.split(raw) if len(p) > 2]
        if len(camel) > 1:
            out.extend(camel)
    return out


_BM25_K1 = 1.2
_BM25_B = 0.75


class LexicalIndex:
    """Inverted token index over (record_id, surface) pairs, BM25-ranked.

    BM25 over the raw term-frequency sum matters here for two reasons: a
    ubiquitous token ("store") must not outrank a rare identifier hit, and a
    short precise record must not lose to a long rambling one that merely
    repeats the term (Robertson/Sparck Jones probabilistic weighting)."""

    def __init__(self) -> None:
        self._postings: dict[str, dict[str, int]] = {}
        self._doc_len: dict[str, int] = {}
        self._avg_len: float = 1.0
        self._n_docs: int = 0
        self._generation: Any = None
        self._lock = threading.Lock()
        # Single-flight for the O(corpus)-decrypt rebuild: two concurrent
        # searches on a moved generation must not both pay it.
        self.build_lock = threading.Lock()

    @property
    def generation(self) -> Any:
        return self._generation

    def build(self, rows: "list[tuple[str, str]]", generation: Any) -> None:
        import math

        postings: dict[str, dict[str, int]] = {}
        doc_len: dict[str, int] = {}
        for rid, surface in rows:
            tokens = tokenize(surface)
            doc_len[rid] = len(tokens)
            for tok in tokens:
                bucket = postings.setdefault(tok, {})
                bucket[rid] = bucket.get(rid, 0) + 1
        n_docs = len(doc_len)
        avg_len = (sum(doc_len.values()) / n_docs) if n_docs else 1.0
        with self._lock:
            self._postings = postings
            self._doc_len = doc_len
            self._avg_len = max(avg_len, 1.0)
            self._n_docs = n_docs
            self._generation = generation
        logger.debug(
            "lexical index built: %d docs, %d tokens, avg_len %.1f (%s)",
            n_docs, len(postings), avg_len, math.floor(avg_len),
        )

    @staticmethod
    def _bm25(tf: int, df: int, length: int, n_docs: int, avg_len: float) -> float:
        import math

        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        denom = tf + _BM25_K1 * (
            1.0 - _BM25_B + _BM25_B * length / avg_len
        )
        return idf * tf * (_BM25_K1 + 1.0) / denom

    def query(self, text: str, k: int = 10) -> "list[tuple[str, float]]":
        tokens = list(dict.fromkeys(tokenize(text)))[:MAX_QUERY_TOKENS]
        if not tokens:
            return []
        with self._lock:
            # ONE snapshot for everything scoring reads: pairing snapshotted
            # postings with a newer n_docs/avg_len from a concurrent build()
            # swap skews IDF.
            postings = self._postings
            per_token = [postings.get(t) for t in tokens]
            doc_len = self._doc_len
            n_docs = self._n_docs
            avg_len = self._avg_len
        if any(p is None for p in per_token):
            # AND semantics with a graceful fallback: if the full conjunction
            # is empty, rank by the rarest tokens that DO occur.
            per_token = [p for p in per_token if p]
            if not per_token:
                return []
        ids = set(per_token[0])
        for p in per_token[1:]:
            nxt = ids & set(p)
            if nxt:
                ids = nxt
        scored = [
            (
                rid,
                sum(
                    self._bm25(p[rid], len(p), doc_len.get(rid, 1), n_docs, avg_len)
                    for p in per_token
                    if rid in p
                ),
            )
            for rid in ids
        ]
        scored.sort(key=lambda t: (-t[1], t[0]))
        return scored[:k]

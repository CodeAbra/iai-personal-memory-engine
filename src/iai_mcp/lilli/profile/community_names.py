"""Human-readable topic names for communities: a pure keyphrase derivation
over each community's own member text, plus the AES-encrypted union map
that survives restart.

Storage mirrors ``persistence.py``: one JSON blob in the store's
``_hippo_meta`` table, encrypted with the same field-level AES-256-GCM
boundary as ``literal_surface`` -- topic vocabulary is derived directly
from user content, so it carries the same sensitivity.
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone
from typing import Any

from cryptography.exceptions import InvalidTag

from iai_mcp.crypto import CryptoKeyError, decrypt_field, encrypt_field, is_encrypted
from iai_mcp.lilli.profile.persistence import _hippo_db
from iai_mcp.store._lexical_index import tokenize

logger = logging.getLogger(__name__)

COMMUNITY_NAMES_META_KEY = "community_names"
COMMUNITY_NAMES_AAD = b"community_names"
COMMUNITY_NAMES_BLOB_VERSION = 1

#: A candidate must appear in at least this fraction of a community's
#: members to count as shared vocabulary rather than one record's fluke.
NAME_MIN_MEMBER_DF = 0.30
NAME_MIN_LEN = 3
NAME_MAX_LEN = 24

_NAME_SHAPE_RE = re.compile(r"^[a-z][a-z-]*$")

_STOPWORDS = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can",
    "had", "has", "have", "her", "his", "its", "our", "out", "she", "that",
    "them", "they", "this", "was", "were", "what", "when", "where", "which",
    "who", "will", "with", "your", "about", "after", "again", "before",
    "being", "between", "both", "does", "down", "during", "each", "from",
    "here", "how", "into", "just", "more", "most", "off", "once", "only",
    "other", "over", "own", "same", "some", "such", "than", "then", "there",
    "these", "those", "through", "too", "under", "until", "very", "while",
    "why", "im", "ive", "dont", "didnt", "cant", "yeah",
    "okay", "like", "just", "really", "actually", "also", "still", "even",
})


def _corpus_idf(df: int, n_docs: int) -> float:
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


def _accepted_shape(token: str) -> bool:
    if not (NAME_MIN_LEN <= len(token) <= NAME_MAX_LEN):
        return False
    if token in _STOPWORDS:
        return False
    return bool(_NAME_SHAPE_RE.match(token))


def _ranked_qualifying(
    member_surfaces: "list[str]",
    corpus_df: "dict[str, int]",
    n_docs: int,
) -> "list[str]":
    """Discriminating tokens for one community, ranked by
    ``in_community_tf * corpus_idf``, highest first. A token must appear in
    >= NAME_MIN_MEMBER_DF of the community's members to qualify; a token
    absent from the corpus df table (df<=0) is skipped rather than scored --
    an unindexed token would otherwise carry a fabricated IDF spike."""
    if n_docs <= 0 or not member_surfaces:
        return []
    n_members = len(member_surfaces)
    tf_sum: "dict[str, int]" = {}
    member_hits: "dict[str, int]" = {}
    for surface in member_surfaces:
        seen: "set[str]" = set()
        for tok in tokenize(surface or ""):
            tf_sum[tok] = tf_sum.get(tok, 0) + 1
            if tok not in seen:
                seen.add(tok)
                member_hits[tok] = member_hits.get(tok, 0) + 1
    scored: "list[tuple[float, str]]" = []
    for tok, hits in member_hits.items():
        if hits / n_members < NAME_MIN_MEMBER_DF:
            continue
        if not _accepted_shape(tok):
            continue
        df = corpus_df.get(tok, 0)
        if df <= 0:
            continue
        score = tf_sum[tok] * _corpus_idf(df, n_docs)
        scored.append((score, tok))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [tok for _, tok in scored]


def derive_community_name(
    member_surfaces: "list[str]",
    corpus_df: "dict[str, int]",
    n_docs: int,
    *,
    prior_name: "str | None" = None,
) -> "str | None":
    """Top discriminating keyphrase for one community, or None when no
    token clears the acceptance gate -- the caller renders the honest
    ``topic-<id8>`` fallback, never a fabricated word.

    HYSTERESIS: when `prior_name` still clears the gate this night, it is
    returned unchanged even if a new token edges ahead on score. A name
    flips only when the prior name genuinely drops out of the qualifying
    set, so the dict key stays stable and per-key learning accumulates.
    """
    ranked = _ranked_qualifying(member_surfaces, corpus_df, n_docs)
    if prior_name is not None and prior_name in ranked:
        return prior_name
    return ranked[0] if ranked else None


def derive_second_term(
    member_surfaces: "list[str]",
    corpus_df: "dict[str, int]",
    n_docs: int,
    *,
    exclude: "str | None",
    prior_name: "str | None" = None,
) -> "str | None":
    """Second-ranked discriminating token, for the de-dup disambiguator.

    HYSTERESIS: mirrors `derive_community_name` -- when `prior_name` still
    clears the gate this night (and is not the base token), it is kept even
    if a new token edges ahead on score. Without this, a compound display
    name (`music-jazz`) can rotate its disambiguation term night-to-night
    while the base topic stays put, resetting the depth key that name
    carries even though nothing about the topic changed.
    """
    ranked = _ranked_qualifying(member_surfaces, corpus_df, n_docs)
    if prior_name is not None and prior_name != exclude and prior_name in ranked:
        return prior_name
    for tok in ranked:
        if tok != exclude:
            return tok
    return None


def neutral_name(community_id: Any) -> str:
    return f"topic-{str(community_id)[:8]}"


def disambiguate(
    names_by_cid: "dict[str, str]",
    second_terms: "dict[str, str]",
) -> "dict[str, str]":
    """When two or more communities share a top token, each colliding cid
    gets its own second discriminating term appended (`music-jazz`); never
    a numeric suffix. If the two-term names also collide, they are left
    equal -- the merge/max-depth survivor on the consuming side handles it."""
    groups: "dict[str, list[str]]" = {}
    for cid, name in names_by_cid.items():
        groups.setdefault(name, []).append(cid)
    out = dict(names_by_cid)
    for name, cids in groups.items():
        if len(cids) <= 1:
            continue
        for cid in cids:
            second = second_terms.get(cid)
            out[cid] = f"{name}-{second}" if second else name
    return out


def save_community_names(
    store: Any,
    *,
    reverse_index: "dict[str, str]",
    provenance: "dict[str, dict]",
    base_index: "dict[str, str] | None" = None,
) -> bool:
    db = _hippo_db(store)
    if db is None:
        return False
    payload = {
        "version": COMMUNITY_NAMES_BLOB_VERSION,
        "reverse_index": dict(reverse_index),
        # Pre-disambiguation base name per cid -- hysteresis compares against
        # this, never against a compound display name: the tokenizer that
        # feeds the qualifying set never emits the hyphen a disambiguated
        # name carries, so hysteresis would silently no-op if fed
        # `reverse_index` instead.
        "base_index": dict(base_index or {}),
        "provenance": dict(provenance),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    # AES boundary: identical to the profile blob and literal_surface --
    # never a plaintext row, never the profile blob, never a plaintext event.
    ct = encrypt_field(
        json.dumps(payload), store._key(), associated_data=COMMUNITY_NAMES_AAD,
    )
    with db._conn_lock:
        db._conn.execute(
            "DELETE FROM _hippo_meta WHERE key = ?", (COMMUNITY_NAMES_META_KEY,),
        )
        db._conn.execute(
            "INSERT OR IGNORE INTO _hippo_meta (key, value) VALUES (?, ?)",
            (COMMUNITY_NAMES_META_KEY, ct),
        )
        db._conn.commit()
    return True


def load_community_names(store: Any) -> "dict[str, dict]":
    db = _hippo_db(store)
    if db is None:
        return {}
    with db._conn_lock:
        row = db._conn.execute(
            "SELECT value FROM _hippo_meta WHERE key = ?",
            (COMMUNITY_NAMES_META_KEY,),
        ).fetchone()
    if row is None:
        return {}
    value = row["value"]
    if not is_encrypted(value):
        return {}
    try:
        plaintext = decrypt_field(
            value, store._key(), associated_data=COMMUNITY_NAMES_AAD,
        )
    except (InvalidTag, ValueError, CryptoKeyError):
        # Fail-safe, mirroring persistence.py: never overwrite ciphertext this
        # process cannot open -- a miss here is silent, not destructive.
        return {}
    try:
        payload = json.loads(plaintext)
    except (ValueError, TypeError):
        return {}
    return {
        "reverse_index": dict(payload.get("reverse_index") or {}),
        "base_index": dict(payload.get("base_index") or {}),
        "provenance": dict(payload.get("provenance") or {}),
    }

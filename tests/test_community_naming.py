from __future__ import annotations

import pytest

from iai_mcp import core
from iai_mcp.lilli.profile.community_names import (
    derive_community_name,
    derive_second_term,
    disambiguate,
    load_community_names,
    neutral_name,
    save_community_names,
)
from iai_mcp.store import MemoryStore


# ---------------------------------------------------------------------------
# Pure derivation
# ---------------------------------------------------------------------------


def _corpus_df(*surfaces: str) -> dict:
    from iai_mcp.store._lexical_index import tokenize

    df: dict = {}
    for surface in surfaces:
        for tok in set(tokenize(surface)):
            df[tok] = df.get(tok, 0) + 1
    return df


JAZZ_MEMBERS = [
    "alice loves jazz music at the downtown club",
    "alice heard the jazz trio play until midnight",
    "alice keeps jazz records on the shelf in the study",
    "alice hums a jazz tune while cooking dinner",
]

OTHER_DOC = "alice wrote a grocery list for the week ahead"


def test_coherent_corpus_yields_alpha_topic_word() -> None:
    # "alice" recurs in every document (a name is common across a personal
    # corpus, low idf); "jazz" is shared by every member but rarer
    # corpus-wide, so it -- not the ubiquitous name -- wins the topic slot.
    # Every one-off word (club, trio, midnight, records, ...) appears in
    # exactly one of the 4 members (25% < the 30% membership gate).
    corpus_df = _corpus_df(*JAZZ_MEMBERS, OTHER_DOC)
    name = derive_community_name(
        JAZZ_MEMBERS, corpus_df, n_docs=len(JAZZ_MEMBERS) + 1,
    )
    assert name == "jazz"
    assert name.isalpha()


NO_SHARED_VOCAB_MEMBERS = [
    "zebra fingerpaint doorway umbrella",
    "kangaroo telephone origami wristband",
    "octopus lantern seaweed compass",
    "giraffe telescope acorn blanket",
]


def test_incoherent_community_yields_none() -> None:
    # 4 members, zero words shared between any two -- every candidate's
    # membership fraction (1/4 = 0.25) falls below the 0.30 gate.
    corpus_df = _corpus_df(*NO_SHARED_VOCAB_MEMBERS)
    name = derive_community_name(
        NO_SHARED_VOCAB_MEMBERS, corpus_df, n_docs=len(NO_SHARED_VOCAB_MEMBERS),
    )
    assert name is None


EDGE_MEMBERS = [
    "alice loves jazz trio music every evening",
    "alice hears the jazz trio again tonight",
    "alice mentions jazz almost daily now",
    "alice still enjoys jazz on weekends",
]


def test_hysteresis_keeps_prior_name_despite_new_edge_ahead() -> None:
    # "trio" clears the 30% membership gate (2 of 4 members) and, being
    # rarer corpus-wide than "jazz" (shared by all 4), edges ahead on raw
    # score -- but "jazz" (the prior name) still clears the qualifying gate
    # this night, so hysteresis keeps it.
    corpus_df = _corpus_df(*EDGE_MEMBERS, OTHER_DOC)
    n_docs = len(EDGE_MEMBERS) + 1
    fresh = derive_community_name(EDGE_MEMBERS, corpus_df, n_docs=n_docs)
    assert fresh == "trio", "the token designed to edge ahead must actually do so"

    stable = derive_community_name(
        EDGE_MEMBERS, corpus_df, n_docs=n_docs, prior_name="jazz",
    )
    assert stable == "jazz", "a still-qualifying prior name must be retained"


def test_hysteresis_flips_when_prior_drops_out_of_qualifying_set() -> None:
    corpus_df = _corpus_df(*NO_SHARED_VOCAB_MEMBERS)
    name = derive_community_name(
        NO_SHARED_VOCAB_MEMBERS, corpus_df,
        n_docs=len(NO_SHARED_VOCAB_MEMBERS), prior_name="jazz",
    )
    assert name is None, "a prior name absent from tonight's set must not survive"


def test_compound_prior_can_never_survive_hysteresis() -> None:
    # Documents why the step persists a separate pre-disambiguation
    # base_index alongside reverse_index: the tokenizer that builds the
    # qualifying set never emits a hyphen, so a compound display name fed
    # back in as prior_name can never be found -- hysteresis must compare
    # against the bare token, never the disambiguated display name.
    corpus_df = _corpus_df(*JAZZ_MEMBERS, OTHER_DOC)
    name = derive_community_name(
        JAZZ_MEMBERS, corpus_df, n_docs=len(JAZZ_MEMBERS) + 1,
        prior_name="jazz-alice",
    )
    assert name != "jazz-alice"
    assert name == "jazz"


def test_df_zero_token_never_scored() -> None:
    # "jazz" tokenized in members but absent from the supplied corpus_df --
    # an unindexed token must not fabricate a name via a spurious idf spike.
    corpus_df = {}
    name = derive_community_name(JAZZ_MEMBERS, corpus_df, n_docs=4)
    assert name is None


def test_n_docs_zero_always_none() -> None:
    corpus_df = _corpus_df(*JAZZ_MEMBERS)
    assert derive_community_name(JAZZ_MEMBERS, corpus_df, n_docs=0) is None


def test_member_fraction_gate_rejects_single_member_fluke() -> None:
    members = [
        "alice mentions a rare xenocrystal artifact",
        "totally unrelated sentence one",
        "totally unrelated sentence two",
        "totally unrelated sentence three",
    ]
    corpus_df = _corpus_df(*members)
    name = derive_community_name(members, corpus_df, n_docs=len(members))
    assert name != "xenocrystal"


def test_code_identifier_shape_rejected() -> None:
    members = [
        "call merge_insert on the batch of rows",
        "merge_insert handles the upsert path",
        "we always use merge_insert for writes",
    ]
    corpus_df = _corpus_df(*members)
    name = derive_community_name(members, corpus_df, n_docs=len(members))
    assert name != "merge_insert"


def test_neutral_fallback_shape() -> None:
    cid = "12345678-aaaa-bbbb-cccc-1234567890ab"
    assert neutral_name(cid) == "topic-12345678"


def test_disambiguate_appends_second_term_for_collision() -> None:
    names_by_cid = {"c1": "music", "c2": "music"}
    second_terms = {"c1": "jazz", "c2": "film"}
    out = disambiguate(names_by_cid, second_terms)
    assert out["c1"] == "music-jazz"
    assert out["c2"] == "music-film"


def test_disambiguate_no_numeric_suffix_on_double_collision() -> None:
    names_by_cid = {"c1": "music", "c2": "music"}
    second_terms = {"c1": "jazz", "c2": "jazz"}
    out = disambiguate(names_by_cid, second_terms)
    assert out["c1"] == out["c2"] == "music-jazz"
    for name in out.values():
        assert not any(ch.isdigit() for ch in name)


def test_disambiguate_leaves_unique_names_alone() -> None:
    names_by_cid = {"c1": "music", "c2": "cooking"}
    out = disambiguate(names_by_cid, {})
    assert out == names_by_cid


def test_derive_second_term_excludes_top_token() -> None:
    corpus_df = _corpus_df(*JAZZ_MEMBERS, OTHER_DOC)
    n_docs = len(JAZZ_MEMBERS) + 1
    top = derive_community_name(JAZZ_MEMBERS, corpus_df, n_docs=n_docs)
    second = derive_second_term(
        JAZZ_MEMBERS, corpus_df, n_docs=n_docs, exclude=top,
    )
    assert second is not None
    assert second != top


def test_derive_second_term_hysteresis_keeps_prior_despite_new_edge_ahead() -> None:
    # EDGE_MEMBERS: excluding "jazz", "trio" edges ahead of "alice" fresh
    # (higher idf, rarer corpus-wide) -- but a still-qualifying prior second
    # term ("alice") must survive, the same protection base-name hysteresis
    # gets, so a compound display name's disambiguation term does not churn
    # night to night while the underlying topic is unchanged.
    corpus_df = _corpus_df(*EDGE_MEMBERS, OTHER_DOC)
    n_docs = len(EDGE_MEMBERS) + 1
    fresh = derive_second_term(EDGE_MEMBERS, corpus_df, n_docs=n_docs, exclude="jazz")
    assert fresh == "trio", "the token designed to edge ahead must actually do so"

    stable = derive_second_term(
        EDGE_MEMBERS, corpus_df, n_docs=n_docs, exclude="jazz", prior_name="alice",
    )
    assert stable == "alice", "a still-qualifying prior second term must be retained"


def test_derive_second_term_hysteresis_flips_when_prior_drops_out() -> None:
    corpus_df = _corpus_df(*NO_SHARED_VOCAB_MEMBERS)
    result = derive_second_term(
        NO_SHARED_VOCAB_MEMBERS, corpus_df,
        n_docs=len(NO_SHARED_VOCAB_MEMBERS), exclude="zebra", prior_name="jazz",
    )
    assert result is None, "a prior second term absent from tonight's set must not survive"


def test_derive_second_term_prior_equal_to_exclude_is_never_returned() -> None:
    # A base name can never also win the second slot -- if `prior_name`
    # happens to equal `exclude` (a stale caller bug), hysteresis must not
    # hand the base token back out as the disambiguator.
    corpus_df = _corpus_df(*EDGE_MEMBERS, OTHER_DOC)
    n_docs = len(EDGE_MEMBERS) + 1
    result = derive_second_term(
        EDGE_MEMBERS, corpus_df, n_docs=n_docs, exclude="jazz", prior_name="jazz",
    )
    assert result != "jazz"
    assert result == "trio"


# ---------------------------------------------------------------------------
# AES storage round-trip
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-community-names")
    return MemoryStore(path=tmp_path / "lancedb")


class _NonHippoStore:
    db = object()

    def _key(self) -> bytes:
        return b"0" * 32


def test_save_load_round_trip(store) -> None:
    reverse_index = {"c1": "jazz", "c2": "topic-deadbeef"}
    provenance = {"c1": {"community_id": "c1", "member_count": 3}}
    assert save_community_names(
        store, reverse_index=reverse_index, provenance=provenance,
    ) is True

    loaded = load_community_names(store)
    assert loaded["reverse_index"] == reverse_index
    assert loaded["provenance"] == provenance


def test_load_miss_returns_empty_dict(store) -> None:
    assert load_community_names(store) == {}


def test_ciphertext_at_rest(store) -> None:
    from iai_mcp.lilli.profile.community_names import COMMUNITY_NAMES_META_KEY
    from iai_mcp.crypto import is_encrypted

    save_community_names(
        store, reverse_index={"c1": "jazz"}, provenance={},
    )
    with store.db._conn_lock:
        row = store.db._conn.execute(
            "SELECT value FROM _hippo_meta WHERE key = ?",
            (COMMUNITY_NAMES_META_KEY,),
        ).fetchone()
    assert row is not None
    assert is_encrypted(row["value"])
    assert "jazz" not in row["value"]


def test_non_hippo_store_is_noop() -> None:
    store = _NonHippoStore()
    assert save_community_names(store, reverse_index={}, provenance={}) is False
    assert load_community_names(store) == {}


def test_fresh_load_reflects_latest_save(store) -> None:
    save_community_names(store, reverse_index={"c1": "jazz"}, provenance={})
    save_community_names(store, reverse_index={"c1": "film"}, provenance={})
    loaded = load_community_names(store)
    assert loaded["reverse_index"] == {"c1": "film"}


# ---------------------------------------------------------------------------
# Boot-cache
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_community_names_cache():
    saved = dict(core._community_names_cache)
    yield
    core._community_names_cache.clear()
    core._community_names_cache.update(saved)


def test_get_before_set_is_empty_dict_never_raises() -> None:
    core.set_community_names({})
    assert core.get_community_names() == {}


def test_set_then_get_round_trips() -> None:
    core.set_community_names({"c1": "jazz"})
    assert core.get_community_names() == {"c1": "jazz"}

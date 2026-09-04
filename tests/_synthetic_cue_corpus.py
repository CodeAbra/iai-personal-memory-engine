"""Synthetic, non-owner, production-shaped corpus and cue generator.

Underscore-prefixed so pytest never collects this as a test module.

Builds a wholly synthetic natural-language corpus (alice-style, never derived
from any real store or session transcript) and a 3-band cue set (specific /
vague / novel) through the real ``Embedder()`` -- never the hash-based bench
fake, so the semantic distances between "specific"/"vague"/"novel" cues and
the corpus are the real ones an embedding model produces, not an approximation
of them.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID, uuid4

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_recall_stage_profile import _monkeypatch_env  # noqa: E402

from iai_mcp.embed import Embedder  # noqa: E402
from iai_mcp.store import MemoryStore, flush_record_buffer  # noqa: E402
from iai_mcp.types import MemoryRecord  # noqa: E402

# Anchored to build-time "now" minus a fixed offset, NOT a historical
# constant: the age penalty (T4, `_pm.AGE_HALF_LIFE_DAYS = 30`) clamps to 1.0
# once `created_at` is far enough in the past, which would make a frozen-now
# straddle differential vacuous (saturated age never moves). Keeping records
# 7-10 days old holds the age term inside its live (non-saturated) range
# regardless of when this module runs.
_CORPUS_AGE_FLOOR_DAYS = 7
_CORPUS_AGE_JITTER_HOURS = 3 * 24

# ---------------------------------------------------------------------------
# Synthetic corpus content -- alice-style, wholly fictional, template text.
# Each topic ships 4 near-duplicate paraphrases of ONE fact (not 4 different
# facts) so a "specific" cue paraphrase of that fact has >=3 corpus neighbors
# above 0.75 raw cosine -- topically-related but differently-worded sentences
# plateau around 0.4-0.6 cosine, well below that band.
# ---------------------------------------------------------------------------

_TOPICS: "list[dict[str, object]]" = [
    {
        "topic": "sourdough_baking",
        "records": [
            "Alice's sourdough starter needs feeding every twelve hours during the summer heat wave.",
            "During the summer heat wave, Alice has to feed her sourdough starter every twelve hours.",
            "Alice feeds her sourdough starter twice a day, every twelve hours, because of the summer heat.",
            "The summer heat wave means Alice must feed her sourdough starter on a strict twelve-hour schedule.",
        ],
        "specific_cue": "How often does Alice need to feed her sourdough starter during the summer heat wave?",
        "vague_cue": "Any updates on Alice's sourdough starter this year?",
    },
    {
        "topic": "garden_tomatoes",
        "records": [
            "Alice planted six heirloom tomato varieties in the raised beds behind the garage.",
            "Behind the garage, Alice put six different heirloom tomato varieties into the raised beds.",
            "Alice chose six heirloom tomato varieties for the raised garden beds behind her garage.",
            "In the raised beds behind the garage, Alice ended up planting six heirloom tomato varieties.",
        ],
        "specific_cue": "How many heirloom tomato varieties did Alice plant in the raised beds behind the garage?",
        "vague_cue": "How is Alice's tomato garden doing this season?",
    },
    {
        "topic": "book_club",
        "records": [
            "Alice's book club picked a Scandinavian mystery novel for next month's meeting.",
            "For next month's meeting, Alice's book club chose a Scandinavian mystery novel.",
            "The book club Alice belongs to selected a Scandinavian mystery novel to read before next month.",
            "Alice's book club settled on a Scandinavian mystery novel as next month's pick.",
        ],
        "specific_cue": "What Scandinavian mystery novel did Alice's book club pick for next month's meeting?",
        "vague_cue": "What's going on with Alice's book club lately?",
    },
    {
        "topic": "marathon_training",
        "records": [
            "Alice increased her weekly marathon mileage to forty miles ahead of the spring race.",
            "Ahead of the spring race, Alice bumped her weekly marathon mileage up to forty miles.",
            "Alice's weekly marathon mileage climbed to forty miles as the spring race approached.",
            "To prepare for the spring race, Alice raised her weekly marathon mileage to forty miles.",
        ],
        "specific_cue": "Why did Alice increase her weekly marathon mileage to forty miles before the spring race?",
        "vague_cue": "How is Alice's marathon training going?",
    },
    {
        "topic": "home_renovation",
        "records": [
            "Alice's kitchen renovation contractor recommended quartz countertops over granite.",
            "For the kitchen renovation, Alice's contractor suggested quartz countertops instead of granite.",
            "Alice's contractor advised quartz countertops over granite for the kitchen renovation.",
            "Instead of granite, Alice's renovation contractor recommended quartz countertops for the kitchen.",
        ],
        "specific_cue": "Did Alice's kitchen renovation contractor recommend quartz countertops over granite?",
        "vague_cue": "What did Alice originally decide about the kitchen renovation budget?",
    },
    {
        "topic": "cat_adoption",
        "records": [
            "Alice adopted a three-year-old tabby cat named Juniper from the county shelter.",
            "From the county shelter, Alice adopted a tabby cat named Juniper who was three years old.",
            "Alice's newly adopted cat, a three-year-old tabby named Juniper, came from the county shelter.",
            "Juniper, the three-year-old tabby cat Alice adopted, came from the county animal shelter.",
        ],
        "specific_cue": "What three-year-old tabby cat did Alice adopt from the county shelter?",
        "vague_cue": "How's Alice's cat Juniper doing these days?",
    },
    {
        "topic": "piano_lessons",
        "records": [
            "Alice's piano teacher assigned a Chopin nocturne for the spring recital.",
            "For the spring recital, Alice's piano teacher assigned her a Chopin nocturne.",
            "Alice's piano teacher gave her a Chopin nocturne to prepare for the spring recital.",
            "A Chopin nocturne was assigned to Alice by her piano teacher for the spring recital.",
        ],
        "specific_cue": "Which Chopin nocturne did Alice's piano teacher assign for the spring recital?",
        "vague_cue": "What was Alice's piano practice schedule before the recital got rescheduled?",
    },
    {
        "topic": "road_trip_planning",
        "records": [
            "Alice mapped a coastal road trip route from Oregon down to Big Sur.",
            "From Oregon down to Big Sur, Alice mapped out a coastal road trip route.",
            "Alice planned a coastal driving route that goes from Oregon all the way to Big Sur.",
            "The coastal road trip route Alice mapped runs from Oregon down to Big Sur.",
        ],
        "specific_cue": "What coastal road trip route did Alice map from Oregon down to Big Sur?",
        "vague_cue": "How did Alice's road trip planning turn out?",
    },
    {
        "topic": "freelance_invoicing",
        "records": [
            "Alice switched freelance invoicing software after her old client management tool shut down.",
            "After her old client management tool shut down, Alice switched to new freelance invoicing software.",
            "Alice's old client management tool shut down, so she switched freelance invoicing software.",
            "When the old client management tool shut down, Alice moved to different freelance invoicing software.",
        ],
        "specific_cue": "Why did Alice switch freelance invoicing software after her old client management tool shut down?",
        "vague_cue": "Any updates on Alice's freelance invoicing setup?",
    },
    {
        "topic": "language_learning",
        "records": [
            "Alice's Spanish tutor switched their weekly lessons to focus on subjunctive verb tenses.",
            "Their weekly lessons now focus on subjunctive verb tenses, after Alice's Spanish tutor made the switch.",
            "Alice's Spanish tutor changed the weekly lesson focus to subjunctive verb tenses.",
            "The weekly Spanish lessons Alice takes now focus on subjunctive verb tenses, per her tutor's switch.",
        ],
        "specific_cue": "What did Alice's Spanish tutor switch their weekly lessons to focus on?",
        "vague_cue": "How is Alice's Spanish learning progressing?",
    },
    {
        "topic": "pottery_class",
        "records": [
            "Alice's pottery class moved to Thursday evenings at the community art studio.",
            "At the community art studio, Alice's pottery class now meets on Thursday evenings.",
            "Alice's pottery class shifted to Thursday evenings, held at the community art studio.",
            "The community art studio moved Alice's pottery class to Thursday evenings.",
        ],
        "specific_cue": "What day did Alice's pottery class move to at the community art studio?",
        "vague_cue": "What's new with Alice's pottery class?",
    },
    {
        "topic": "bike_commute",
        "records": [
            "Alice's bike commute takes eighteen minutes along the river trail to downtown.",
            "Along the river trail to downtown, Alice's bike commute takes eighteen minutes.",
            "It takes Alice eighteen minutes to bike along the river trail into downtown.",
            "Alice's eighteen-minute bike commute follows the river trail into downtown.",
        ],
        "specific_cue": "How long does Alice's bike commute take along the river trail to downtown?",
        "vague_cue": "How's Alice's bike commute been lately?",
    },
    {
        "topic": "wine_tasting",
        "records": [
            "Alice's wine tasting group compared three Oregon pinot noirs from the same vintage.",
            "Three Oregon pinot noirs from the same vintage were compared by Alice's wine tasting group.",
            "Alice's wine tasting group tried three Oregon pinot noirs, all from the same vintage.",
            "At Alice's wine tasting group, three Oregon pinot noirs from the same vintage were compared side by side.",
        ],
        "specific_cue": "Which three Oregon pinot noirs did Alice's wine tasting group compare from the same vintage?",
        "vague_cue": "What did Alice's wine tasting group get up to recently?",
    },
    {
        "topic": "woodworking_shop",
        "records": [
            "Alice's woodworking shop got a new dust collection system installed last weekend.",
            "Last weekend, a new dust collection system was installed in Alice's woodworking shop.",
            "Alice had a new dust collection system installed in her woodworking shop last weekend.",
            "The new dust collection system in Alice's woodworking shop was installed just last weekend.",
        ],
        "specific_cue": "What new dust collection system did Alice install in her woodworking shop last weekend?",
        "vague_cue": "What's Alice been building in the woodworking shop?",
    },
    {
        "topic": "meditation_practice",
        "records": [
            "Alice's meditation practice shifted from ten minutes to twenty-five minutes each morning.",
            "Each morning, Alice's meditation practice went from ten minutes up to twenty-five minutes.",
            "Alice now meditates twenty-five minutes each morning, up from her original ten minutes.",
            "What used to be a ten-minute morning meditation for Alice is now twenty-five minutes.",
        ],
        "specific_cue": "How long did Alice's morning meditation practice shift to, from ten minutes?",
        "vague_cue": "What was Alice's meditation practice like before the retreat?",
    },
    {
        "topic": "photography_hobby",
        "records": [
            "Alice's photography hobby moved from digital back to shooting on film this year.",
            "This year, Alice's photography hobby shifted from digital back to shooting on film.",
            "Alice went back to shooting film for her photography hobby this year, after years of digital.",
            "Shooting on film is where Alice's photography hobby returned to this year, away from digital.",
        ],
        "specific_cue": "Did Alice's photography hobby move from digital back to shooting on film this year?",
        "vague_cue": "What's Alice been doing with her photography hobby?",
    },
]

# Cues about topics wholly absent from the corpus above -- "novel" band,
# expected low raw cosine against every corpus record. A subset carries
# historical-intent trigger words ("originally", "before", "previously") to
# exercise the T14/T15/T16 Bucket-B gates under low-confidence recall too.
_NOVEL_CUES: "list[str]" = [
    "What's the best way to insulate an attic before winter?",
    "How do I renew my passport before the international trip?",
    "What's Alice's opinion on quantum computing hardware roadmaps?",
    "Details about deep sea coral reef restoration projects.",
    "Notes on antique typewriter restoration techniques.",
    "Thoughts on urban beekeeping regulations downtown.",
    "What did Alice decide about the vintage motorcycle restoration budget?",
    "Summary of the community solar panel co-op meeting.",
    "Alice's plans for the backyard chicken coop expansion.",
    "Details on the neighborhood tool-lending library proposal.",
    "What does Alice think about the new light rail extension?",
    "Notes from Alice's calligraphy workshop last spring.",
    "Alice's review of the new espresso machine she bought.",
    "What did Alice originally plan for the kayak trip before the weather changed?",
    "Summary of Alice's notes on beekeeping before winter preparation.",
    "Details about the community mural project Alice volunteered for.",
    "Alice's thoughts on switching to a standing desk earlier this year.",
    "What was Alice's original budget for the greenhouse project?",
    "Notes on the origami workshop Alice attended previously.",
    "Alice's plans for the rooftop stargazing party.",
]

# Topics whose specific-band cue is additionally wrapped as a verbatim-style
# quote, so it drives `mode="verbatim"` at the harness call site (mode is a
# caller-supplied recall_for_response parameter, not derived from cue text).
_VERBATIM_TOPICS = frozenset({
    "sourdough_baking", "garden_tomatoes", "book_club",
    "marathon_training", "cat_adoption", "wine_tasting",
})


@dataclass(frozen=True)
class CueSpec:
    text: str
    band: str  # "specific" | "vague" | "novel"
    mode: str  # "concept" | "verbatim" -- passed to recall_for_response


@dataclass
class SyntheticCorpusAndCues:
    records: "list[MemoryRecord]"
    cues_by_band: "dict[str, list[CueSpec]]"
    total_cues: int


def _unit(v: "np.ndarray | list[float]") -> list[float]:
    arr = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    if n > 0:
        arr = arr / n
    return arr.tolist()


def build_corpus_records(seed: int = 0, embedder: "Embedder | None" = None) -> "list[MemoryRecord]":
    """Deterministic (given `seed`) synthetic corpus, embedded through the
    REAL Embedder(). `created_at` is explicit and deterministic -- never
    `datetime.now()` -- so the NULL-created_at clock-read fallbacks on the
    recall path structurally never fire for this corpus."""
    embedder = embedder or Embedder()
    rng = np.random.default_rng(seed)
    corpus_base_ts = datetime.now(timezone.utc) - timedelta(days=_CORPUS_AGE_FLOOR_DAYS + 1)
    records: list[MemoryRecord] = []
    for topic in _TOPICS:
        for sentence in topic["records"]:
            vec = _unit(embedder.embed(sentence))
            created_at = corpus_base_ts + timedelta(
                hours=int(rng.integers(0, _CORPUS_AGE_JITTER_HOURS))
            )
            rec = MemoryRecord(
                id=uuid4(),
                tier="episodic",
                literal_surface=sentence,
                aaak_index="",
                embedding=vec,
                community_id=None,
                centrality=0.0,
                detail_level=2,
                pinned=False,
                stability=0.5,
                difficulty=0.0,
                last_reviewed=None,
                never_decay=False,
                never_merge=False,
                provenance=[],
                created_at=created_at,
                updated_at=created_at,
                tags=[],
                language="en",
            )
            records.append(rec)
    return records


def build_cue_set(seed: int = 0) -> "dict[str, list[CueSpec]]":
    """Deterministic 3-band cue set. `seed` is accepted for interface parity
    with `build_corpus_records`; cue content is fixed template text, so the
    set is already deterministic without consuming the seed."""
    del seed
    specific: list[CueSpec] = []
    vague: list[CueSpec] = []
    for topic in _TOPICS:
        topic_id = topic["topic"]
        mode = "verbatim" if topic_id in _VERBATIM_TOPICS else "concept"
        specific_text = topic["specific_cue"]
        if mode == "verbatim":
            specific_text = f'Did Alice write "{topic["records"][0]}"?'
        specific.append(CueSpec(text=specific_text, band="specific", mode=mode))
        vague.append(CueSpec(text=topic["vague_cue"], band="vague", mode="concept"))
    novel = [CueSpec(text=t, band="novel", mode="concept") for t in _NOVEL_CUES]
    return {"specific": specific, "vague": vague, "novel": novel}


def flatten_cues(cues_by_band: "dict[str, list[CueSpec]]") -> "list[CueSpec]":
    """Stable flat ordering (specific, vague, novel; within-band template
    order) -- used as the deterministic index source for cue_seed values."""
    out: list[CueSpec] = []
    for band in ("specific", "vague", "novel"):
        out.extend(cues_by_band.get(band, []))
    return out


def insert_corpus(store: MemoryStore, records: "list[MemoryRecord]") -> None:
    for rec in records:
        store.insert(rec)
    flush_record_buffer(store)
    store._build_exact_index_sync()


def build_synthetic_corpus_and_cues(seed: int = 0) -> SyntheticCorpusAndCues:
    """All-in-one entrypoint: builds an ephemeral store isolated from any
    real user data (fake HOME/IAI_MCP_STORE/keyring) and inserts the corpus
    through the normal write path."""
    import keyring as _keyring

    mp = pytest.MonkeyPatch()
    fake_keyring: dict = {}
    try:
        mp.setattr(_keyring, "get_password", lambda s, u: fake_keyring.get((s, u)))
        mp.setattr(_keyring, "set_password", lambda s, u, p: fake_keyring.__setitem__((s, u), p))
        mp.setattr(_keyring, "delete_password", lambda s, u: fake_keyring.pop((s, u), None))

        with TemporaryDirectory(prefix="synthetic-cue-corpus-") as tmp:
            tmp_path = Path(tmp)
            _monkeypatch_env(mp, tmp_path)
            store_root = tmp_path / "store"
            mp.setenv("IAI_MCP_STORE", str(store_root))
            # Standalone (non-pytest) invocations skip conftest's autouse
            # _crypto_passphrase_env fixture; set the same test passphrase
            # explicitly so a fresh store never needs Keychain access.
            mp.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "iai-mcp-test-passphrase")

            embedder = Embedder()
            records = build_corpus_records(seed=seed, embedder=embedder)
            cues_by_band = build_cue_set(seed=seed)

            store = MemoryStore(path=store_root)
            insert_corpus(store, records)
    finally:
        mp.undo()

    total_cues = sum(len(v) for v in cues_by_band.values())
    return SyntheticCorpusAndCues(
        records=records,
        cues_by_band=cues_by_band,
        total_cues=total_cues,
    )


# ---------------------------------------------------------------------------
# Bucket-A per-term discrimination fixture.
#
# A separate, small, purpose-built corpus -- NOT the 3-band corpus above --
# so extending it never risks the already-verified path-vs-itself
# determinism of the main harness. Every Bucket-A term (T1-T4, T6, T9, T11,
# T12 fire under default settings; T5 and T7 are OFF by default in
# production -- W_SPREAD_ACT=0.0 and structural_weight defaults to 0.0 via
# profile_state -- so their controls explicitly enable the real, existing
# kill-switch for the duration of one probe call, which is legitimate: the
# Rust port must still implement the formula correctly for when the knob is
# on) gets a "twin" pair wherever possible -- two records sharing IDENTICAL
# literal_surface (hence identical embedding, hence identical cosine/
# trigram/fts contribution) that differ ONLY in the field the term reads,
# isolating that term's contribution from every other term.
# ---------------------------------------------------------------------------


@dataclass
class TermDiscriminationFixture:
    records: "list[MemoryRecord]"
    # (src_id, dst_id, edge_type, delta) applied via store.boost_edges after insert_corpus.
    edges: "list[tuple[object, object, str, float]]"
    # term -> probe cue text.
    probe_cue: "dict[str, str]"
    # Named record-id handles term controls need to target for corruption.
    ids: "dict[str, object]"


def build_term_discrimination_fixture(
    seed: int = 0, embedder: "Embedder | None" = None,
) -> TermDiscriminationFixture:
    embedder = embedder or Embedder()
    rng = np.random.default_rng(seed)
    base_ts = datetime.now(timezone.utc) - timedelta(days=_CORPUS_AGE_FLOOR_DAYS + 1)

    def _rec(text, *, created_at=None, aaak_index="", stability=0.5,
              tags=None, structure_hv=b"") -> MemoryRecord:
        vec = _unit(embedder.embed(text))
        ts = created_at or (base_ts + timedelta(hours=int(rng.integers(0, _CORPUS_AGE_JITTER_HOURS))))
        return MemoryRecord(
            id=uuid4(), tier="episodic", literal_surface=text, aaak_index=aaak_index,
            embedding=vec, community_id=None, centrality=0.0, detail_level=2,
            pinned=False, stability=stability, difficulty=0.0, last_reviewed=None,
            never_decay=False, never_merge=False, provenance=[],
            created_at=ts, updated_at=ts, tags=list(tags or []), language="en",
            structure_hv=structure_hv,
        )

    records: list[MemoryRecord] = []
    ids: dict[str, object] = {}
    probe_cue: dict[str, str] = {}

    # T1 cosine / T6 community_contrib (shares this pair -- community_contrib
    # is mode_bias * cos * graded_weight; with the varying cos already
    # present here, zeroing the community-bias constant is independently
    # detectable on the same probe).
    probe_cue["T1_cosine"] = "Alice recorded weather patterns for the mountain cabin project."
    probe_cue["T6_community_contrib"] = probe_cue["T1_cosine"]
    r_near = _rec("Alice recorded detailed weather patterns for the mountain cabin project this spring.")
    r_far = _rec("Alice's neighbor bought a new kayak for the summer lake trips.")
    records += [r_near, r_far]
    ids["cosine_near"] = r_near.id
    ids["cosine_far"] = r_far.id

    # T2 aaak -- twin pair, identical text AND created_at, only aaak_index
    # differs (a shared created_at is required for a true twin: the shared
    # `_rec` helper otherwise jitters each call's created_at independently
    # via the running rng, which would leak T4 variance into this pair).
    probe_cue["T2_aaak"] = "Notes about the zephyrcabin9k renovation timeline."
    t2_text = "Renovation timeline notes are filed under the shared project folder."
    t2_ts = base_ts + timedelta(hours=1)
    r_aaak_hit = _rec(t2_text, aaak_index="E:zephyrcabin9k", created_at=t2_ts)
    r_aaak_miss = _rec(t2_text, aaak_index="", created_at=t2_ts)
    records += [r_aaak_hit, r_aaak_miss]
    ids["aaak_hit"] = r_aaak_hit.id
    ids["aaak_miss"] = r_aaak_miss.id

    # T3 degree -- twin pair, identical text AND created_at, only graph
    # degree differs. `graph.degrees()` only counts edges between nodes that
    # both exist in the runtime graph -- a boosted edge to a synthetic,
    # never-inserted UUID is silently absent from degree, so the hub needs
    # 15 real (if throwaway) filler records to connect to, not bare UUIDs.
    probe_cue["T3_degree"] = "Alice organized the woodshop tool inventory over the weekend."
    t3_text = "The woodshop tool inventory got reorganized into labeled bins."
    t3_ts = base_ts + timedelta(hours=2)
    r_degree_hub = _rec(t3_text, created_at=t3_ts)
    r_degree_peer = _rec(t3_text, created_at=t3_ts)
    records += [r_degree_hub, r_degree_peer]
    ids["degree_hub"] = r_degree_hub.id
    ids["degree_peer"] = r_degree_peer.id
    degree_fillers = [
        _rec(f"Unrelated filler note number {i} about nothing in particular.", created_at=t3_ts)
        for i in range(15)
    ]
    records += degree_fillers
    edges: list[tuple[object, object, str, float]] = [
        (r_degree_hub.id, filler.id, "hebbian", 1.0) for filler in degree_fillers
    ]

    # T3 second arm -- an excluded-edge-type hub: every edge is
    # "entity_shared" (a `RANKING_DEGREE_EXCLUDED` type), so the normal
    # exclusion holds this hub's ranking degree at zero regardless of its
    # real edge count; only an UNFILTERED degree computation (one that
    # ignores the exclusion set entirely) makes this hub's degree nonzero.
    # Distinct from `T3_degree` above, which corrupts via `W_DEGREE=0`
    # (an all-zero degree term) -- this corrupts via the exclusion set
    # itself, isolating the "ignores the exclusion" failure mode.
    probe_cue["T3_degree_unfiltered_exclusion"] = "Alice logged the compost bin temperature readings."
    t3x_text = "Compost bin temperature readings got logged twice daily."
    t3x_ts = base_ts + timedelta(hours=2, minutes=30)
    r_degree_excluded_hub = _rec(t3x_text, created_at=t3x_ts)
    r_degree_excluded_peer = _rec(t3x_text, created_at=t3x_ts)
    records += [r_degree_excluded_hub, r_degree_excluded_peer]
    ids["degree_excluded_hub"] = r_degree_excluded_hub.id
    ids["degree_excluded_peer"] = r_degree_excluded_peer.id
    excluded_fillers = [
        _rec(f"Unrelated excluded-edge filler note {i}.", created_at=t3x_ts)
        for i in range(15)
    ]
    records += excluded_fillers
    edges += [
        (r_degree_excluded_hub.id, filler.id, "entity_shared", 1.0) for filler in excluded_fillers
    ]

    # T4 age -- twin pair, identical text, only created_at differs
    # (near-zero age vs near the AGE_HALF_LIFE_DAYS=30 half-life).
    probe_cue["T4_age"] = "Alice updated the herb garden watering schedule."
    t4_text = "The herb garden watering schedule got a fresh update this week."
    r_age_young = _rec(t4_text, created_at=datetime.now(timezone.utc) - timedelta(hours=6))
    r_age_old = _rec(t4_text, created_at=datetime.now(timezone.utc) - timedelta(days=28))
    records += [r_age_young, r_age_old]
    ids["age_young"] = r_age_young.id
    ids["age_old"] = r_age_old.id

    # T5 spread_contrib -- OFF by default (W_SPREAD_ACT=0.0). seed -[entity_
    # shared]-> intermediate -[entity_shared]-> target chain; target's own
    # cosine to the probe cue is deliberately low, so its ONLY score
    # contribution beyond the base floor comes from the activation-transfer
    # widen when the kill-switch is explicitly enabled for the probe.
    probe_cue["T5_spread_contrib"] = "Alice discussed the lighthouse restoration grant proposal."
    r_spread_seed = _rec("Alice discussed the lighthouse restoration grant proposal with the historical society.")
    r_spread_intermediate = _rec("The historical society keeps a ledger of restoration grant recipients.")
    r_spread_target = _rec("A ceramic mug collection sits on the kitchen windowsill.")
    records += [r_spread_seed, r_spread_intermediate, r_spread_target]
    ids["spread_seed"] = r_spread_seed.id
    ids["spread_intermediate"] = r_spread_intermediate.id
    ids["spread_target"] = r_spread_target.id
    edges += [
        (r_spread_seed.id, r_spread_intermediate.id, "entity_shared", 3.0),
        (r_spread_intermediate.id, r_spread_target.id, "entity_shared", 3.0),
    ]

    # T7 structural_similarity -- OFF by default (structural_weight defaults
    # to 0.0 via profile_state). Twin pair, identical text; only one carries
    # a TOPIC-tagged structure_hv matching the probe cue text exactly (same
    # string through filler_hv on both sides -> maximal structural cosine).
    from iai_mcp import tem

    probe_cue["T7_structural_similarity"] = "structural probe marker"
    t7_text = "Documentation review notes are attached to this entry."
    t7_ts = base_ts + timedelta(hours=3)
    r_struct_hit = _rec(t7_text, tags=["structural probe marker"], created_at=t7_ts)
    r_struct_hit.structure_hv = tem.bind_structure(r_struct_hit)
    r_struct_miss = _rec(t7_text, created_at=t7_ts)
    records += [r_struct_hit, r_struct_miss]
    ids["structural_hit"] = r_struct_hit.id
    ids["structural_miss"] = r_struct_miss.id

    # T9 stability -- twin pair, identical text AND created_at, only
    # stability differs.
    probe_cue["T9_stability"] = "Alice catalogued vintage postcards from the estate sale."
    t9_text = "Vintage postcards from the estate sale got catalogued by date."
    t9_ts = base_ts + timedelta(hours=4)
    r_stability_low = _rec(t9_text, stability=0.02, created_at=t9_ts)
    r_stability_high = _rec(t9_text, stability=0.98, created_at=t9_ts)
    records += [r_stability_low, r_stability_high]
    ids["stability_low"] = r_stability_low.id
    ids["stability_high"] = r_stability_high.id

    # T10 valence -- both MemoryRecord and SimpleRecordView carry a
    # `valence` field now, but this fixture's records are built through
    # `_rec()` with no valence kwarg, so they hold the default 0.0. The
    # control injects a non-default value via a getattr shadow (test-only),
    # documented as such wherever it is used.
    probe_cue["T10_valence"] = "Alice reviewed the community garden plot assignments."
    t10_text = "Community garden plot assignments were reviewed for the new season."
    t10_ts = base_ts + timedelta(hours=5)
    r_valence_a = _rec(t10_text, created_at=t10_ts)
    r_valence_b = _rec(t10_text, created_at=t10_ts)
    records += [r_valence_a, r_valence_b]
    ids["valence_a"] = r_valence_a.id
    ids["valence_b"] = r_valence_b.id

    # T11 trigram -- literal-text overlap with the cue itself (>0.3 jaccard)
    # for one record, a differently-worded record for the other.
    probe_cue["T11_trigram"] = "Alice fixed the squeaky garden gate hinge yesterday afternoon."
    r_trigram_hit = _rec("Alice fixed the squeaky garden gate hinge yesterday afternoon.")
    r_trigram_miss = _rec("The neighborhood association scheduled a fence repair crew visit.")
    records += [r_trigram_hit, r_trigram_miss]
    ids["trigram_hit"] = r_trigram_hit.id
    ids["trigram_miss"] = r_trigram_miss.id

    # T12 fts_hits -- exact-substring containment. The gate is
    # `cue.lower() in rec.literal_surface.lower()` -- the WHOLE lowercased
    # CUE must appear verbatim inside the record's surface, not the other
    # way around -- so the record's text is built to literally contain the
    # probe cue as a substring. The candidate control removes that
    # substring from ONLY that hydrated record's surface (never the store)
    # so trigram/cosine variance is neutralized separately in the control.
    probe_cue["T12_fts_hits"] = "quokka77bridge maintenance schedule"
    r_fts_hit = _rec(
        "Notes about the quokka77bridge maintenance schedule were filed by the crew last week."
    )
    records.append(r_fts_hit)
    ids["fts_hit"] = r_fts_hit.id

    # T13 lex_rank -- identifier-grade cue (snake_case token), warm BM25
    # index seeded directly (see warm_lexical_index_for_fixture). The cue
    # deliberately avoids common words shared with the filler records above
    # ("about", "notes"): the index's AND-of-tokens query keeps whichever
    # token comes first in cue order even when its intersection with a
    # later, rarer token is empty (a graceful-fallback, not a strict AND) --
    # opening on a word common to the 15 filler records would let THEM win
    # the match instead of the intended target.
    probe_cue["T13_lex_rank"] = "zephyr_cache_9f2 settings reference."
    r_lex_hit = _rec("The zephyr_cache_9f2 configuration file moved to the shared drive.")
    records.append(r_lex_hit)
    ids["lex_hit"] = r_lex_hit.id

    return TermDiscriminationFixture(records=records, edges=edges, probe_cue=probe_cue, ids=ids)


def apply_term_discrimination_edges(store: MemoryStore, fixture: TermDiscriminationFixture) -> None:
    for src, dst, edge_type, delta in fixture.edges:
        store.boost_edges([(src, dst)], edge_type=edge_type, delta=[delta])


def warm_lexical_index_for_fixture(store: MemoryStore, fixture: TermDiscriminationFixture) -> None:
    """Seed the store's warm lexical index directly from the fixture's own
    known (id, surface) pairs, using the real `LexicalIndex.build` +
    `lexical_query_warm` read path unchanged. `store.lexical_search`'s own
    rebuild (iter_record_columns + get_batch) was observed, on this small
    fixture, to occasionally return ids absent from the store entirely --
    an environment-dependent rebuild issue out of scope for this harness to
    root-cause; seeding directly is hermetic and exercises the identical
    downstream query/BM25/min_idf code the recall path reads from."""
    from iai_mcp.store._lexical_index import LexicalIndex

    idx = LexicalIndex()
    rows = [(str(r.id), r.literal_surface or "") for r in fixture.records]
    generation = store._corpus_count_cache.generation()
    idx.build(rows, generation)
    store._lexical_idx = idx


# ---------------------------------------------------------------------------
# Bucket-B evidence fixture: T8/T14/T15/T16/T17 are Python-applied to only
# the winner rows after the resident-scoring call returns, so they are NOT
# part of the per-term miscompute controls above -- both sides of the
# differential apply them identically regardless of which side did the
# resident (Bucket-A) scoring. This fixture instead records, once, that
# each term measurably moves a score and (where the term's own mechanics
# allow it) flips rank order against a Bucket-A-only baseline -- the
# obligation a future candidate producer inherits is asserting a Bucket-B
# promotion is never lost past a margin-bounded top-(k+margin) window.
# ---------------------------------------------------------------------------


@dataclass
class BucketBEvidenceFixture:
    records: "list[MemoryRecord]"
    edges: "list[tuple[object, object, str, float]]"
    probe_cue: "dict[str, str]"
    ids: "dict[str, object]"


def build_bucket_b_evidence_fixture(
    seed: int = 0, embedder: "Embedder | None" = None,
) -> BucketBEvidenceFixture:
    embedder = embedder or Embedder()
    rng = np.random.default_rng(seed)
    base_ts = datetime.now(timezone.utc) - timedelta(days=_CORPUS_AGE_FLOOR_DAYS + 1)

    def _rec(text, *, created_at=None, tier="episodic", salience_level="unflagged", id=None) -> MemoryRecord:
        vec = _unit(embedder.embed(text))
        ts = created_at or (base_ts + timedelta(hours=int(rng.integers(0, _CORPUS_AGE_JITTER_HOURS))))
        return MemoryRecord(
            id=id or uuid4(), tier=tier, literal_surface=text, aaak_index="",
            embedding=vec, community_id=None, centrality=0.0, detail_level=2,
            pinned=False, stability=0.5, difficulty=0.0, last_reviewed=None,
            never_decay=False, never_merge=False, provenance=[],
            created_at=ts, updated_at=ts, tags=[], language="en",
            salience_level=salience_level,
        )

    records: list[MemoryRecord] = []
    ids: dict[str, object] = {}
    probe_cue: dict[str, str] = {}

    # T8 profile modulation -- interest_boost/dunn_quadrant apply the SAME
    # gain to every candidate (uniform multiplier), so by construction
    # neither can flip relative rank order between two candidates; only the
    # community-keyed monotropism_depth component varies per candidate, and
    # demonstrating THAT requires distinct multi-community targeting this
    # small fixture does not build. This probe records the (real, measured)
    # score change from interest_boost, documented as score-only evidence.
    probe_cue["T8_profile_modulation"] = "Alice reorganized her recipe card collection by cuisine."
    r_t8 = _rec("Alice reorganized her recipe card collection by cuisine this weekend.")
    records.append(r_t8)
    ids["t8_target"] = r_t8.id

    # T14 tier/knowledge-boost (xtier, default 1.05x) -- a near-tied pair;
    # the semantic-tier record needs the boost to overtake the episodic one.
    probe_cue["T14_tier_boost"] = "Alice's espresso machine needed a new water filter cartridge."
    r_t14_episodic = _rec(
        "Alice's espresso machine needed a new water filter cartridge this month.",
        tier="episodic",
    )
    r_t14_semantic = _rec(
        "The water filter cartridge in Alice's espresso machine got replaced recently.",
        tier="semantic",
    )
    records += [r_t14_episodic, r_t14_semantic]
    ids["t14_episodic"] = r_t14_episodic.id
    ids["t14_semantic"] = r_t14_semantic.id

    # T15 salience multiplier (default step 0.05/level; critical = rank 2 ->
    # 1.10x) -- a near-tied pair, one flagged critical.
    probe_cue["T15_salience"] = "Alice replaced the smoke detector batteries in the hallway."
    r_t15_unflagged = _rec(
        "Alice replaced the smoke detector batteries in the hallway last night.",
        salience_level="unflagged",
    )
    r_t15_critical = _rec(
        "The hallway smoke detector got new batteries from Alice recently.",
        salience_level="critical",
    )
    records += [r_t15_unflagged, r_t15_critical]
    ids["t15_unflagged"] = r_t15_unflagged.id
    ids["t15_critical"] = r_t15_critical.id

    # T16 temporal-match boost (default 1.15x) -- a twin pair (IDENTICAL
    # text -> identical cosine/trigram/fts, a perfect Bucket-A tie), only
    # created_at differs: one exactly matches an explicit date mention in
    # the cue. A tied baseline makes any nonzero boost a guaranteed flip.
    probe_cue["T16_temporal_match"] = "Alice's project notes from March 3, 2024."
    t16_text = "Alice's project notes covered budget details for the quarter."
    t16_shared_ts = base_ts + timedelta(hours=6)
    t16_match_ts = datetime(2024, 3, 3, 12, 0, tzinfo=timezone.utc)
    r_t16_nomatch = _rec(t16_text, created_at=t16_shared_ts)
    r_t16_match = _rec(t16_text, created_at=t16_match_ts)
    records += [r_t16_nomatch, r_t16_match]
    ids["t16_nomatch"] = r_t16_nomatch.id
    ids["t16_match"] = r_t16_match.id

    # T17 historical_verbatim anchor rewrite -- wholesale score REPLACEMENT
    # of a contradicts-outgoing anchor's score, to sit just under its
    # correction's score. Cue carries historical trigger words ("originally",
    # "before") so cue_intent becomes "historical_verbatim" automatically.
    probe_cue["T17_historical_verbatim_rewrite"] = (
        "What did Alice originally say about the vacation schedule before it changed?"
    )
    # `store.boost_edges` canonicalizes (src, dst) by lexicographic sort of
    # the id STRINGS (documented store invariant), independent of caller
    # intent -- the historical-anchor/"src" role `build_temporal_validity_
    # maps` resolves goes to whichever id sorts first, which `uuid4()` makes
    # a 50/50 coin flip per run. The anchor text's literal overlap with the
    # cue fires T11's x2.0 trigram multiplier (the correction text does
    # not), so the anchor's own natural score is always ABOVE the
    # correction's -- the rewrite can only ever promote the LOWER-scoring
    # side toward the higher one, so fixed ids force the correction (lower
    # natural score) into the "src"/rewritten role every run, instead of
    # leaving which side gets rewritten to chance.
    r_t17_anchor = _rec(
        "Alice's vacation schedule was originally set for the first week of June.",
        id=UUID(int=2),
    )
    r_t17_correction = _rec(
        "Alice's vacation schedule is now set for the second week of July.",
        id=UUID(int=1),
    )
    records += [r_t17_anchor, r_t17_correction]
    ids["t17_anchor"] = r_t17_anchor.id
    ids["t17_correction"] = r_t17_correction.id
    edges: list[tuple[object, object, str, float]] = [
        (r_t17_anchor.id, r_t17_correction.id, "contradicts", 1.0),
    ]

    return BucketBEvidenceFixture(records=records, edges=edges, probe_cue=probe_cue, ids=ids)


# ---------------------------------------------------------------------------
# Live-path-shaped corpus (>200 records) -- sized past the bounded Layer-1
# pool so ranking/scoring is exercised at scale and deep-rank planted
# verbatim controls have somewhere to live, unlike the 60-record fixture
# above.
# ---------------------------------------------------------------------------

_LIVE_PATH_ACTIVITIES: "list[str]" = [
    "baking", "gardening", "reading", "running", "painting", "coding",
    "hiking", "knitting", "cycling", "photography", "cooking", "sailing",
    "birdwatching", "woodworking", "pottery", "chess", "yoga", "fishing",
    "kayaking", "climbing", "swimming", "dancing", "singing", "writing",
    "sketching", "gaming", "camping", "surfing", "skiing", "archery",
    "beekeeping", "brewing", "quilting", "carpentry", "astronomy",
    "calligraphy", "origami", "juggling", "fencing", "rowing",
]

_LIVE_PATH_TEMPLATES: "list[str]" = [
    "Alice spent the {period} on her {activity} project, focusing on {detail}.",
    "During the {period}, Alice worked on {activity}, paying attention to {detail}.",
    "Alice's {activity} session in the {period} centered on {detail}.",
    "In the {period}, Alice's {activity} routine emphasized {detail}.",
    "Alice dedicated the {period} to {activity}, especially {detail}.",
]

_LIVE_PATH_PERIODS: "list[str]" = ["morning", "afternoon", "evening", "weekend"]
_LIVE_PATH_DETAILS: "list[str]" = ["technique", "consistency", "precision", "patience"]


def build_live_path_shaped_corpus(
    seed: int = 0, embedder: "Embedder | None" = None,
) -> "tuple[list[MemoryRecord], list[tuple[UUID, UUID]]]":
    """>200-record corpus (40 activities x 5 phrasings x 4 periods = 800),
    through the REAL Embedder() -- each activity forms a dense near-
    duplicate cluster. Also returns explicit within-cluster edge pairs
    (chain-linked, 3 forward neighbors each): the Layer-1 hop expansion
    needs real graph edges to fan out over -- insert-time similarity-
    linking alone is not guaranteed dense enough to push the bounded pool
    (K_CANDIDATES=200 ANN seed + authority + hop1/hop2 + rich-club) past
    200 on a fixture this size, so the edges are authored explicitly via
    `store.boost_edges` at insert time (caller's responsibility)."""
    embedder = embedder or Embedder()
    rng = np.random.default_rng(seed)
    base_ts = datetime.now(timezone.utc) - timedelta(days=_CORPUS_AGE_FLOOR_DAYS + 1)
    records: "list[MemoryRecord]" = []
    edges: "list[tuple[UUID, UUID]]" = []
    for activity in _LIVE_PATH_ACTIVITIES:
        cluster_start = len(records)
        for template in _LIVE_PATH_TEMPLATES:
            for period in _LIVE_PATH_PERIODS:
                detail = _LIVE_PATH_DETAILS[
                    (len(records)) % len(_LIVE_PATH_DETAILS)
                ]
                sentence = template.format(period=period, activity=activity, detail=detail)
                vec = _unit(embedder.embed(sentence))
                created_at = base_ts + timedelta(
                    hours=int(rng.integers(0, _CORPUS_AGE_JITTER_HOURS))
                )
                rec = MemoryRecord(
                    id=uuid4(), tier="episodic", literal_surface=sentence, aaak_index="",
                    embedding=vec, community_id=None, centrality=0.0, detail_level=2,
                    pinned=False, stability=0.5, difficulty=0.0, last_reviewed=None,
                    never_decay=False, never_merge=False, provenance=[],
                    created_at=created_at, updated_at=created_at, tags=[], language="en",
                )
                records.append(rec)
        cluster = records[cluster_start:]
        for i, rec in enumerate(cluster):
            for j in range(i + 1, min(i + 4, len(cluster))):
                edges.append((rec.id, cluster[j].id))
    return records, edges


def insert_live_path_corpus(
    store: MemoryStore, records: "list[MemoryRecord]", edges: "list[tuple[UUID, UUID]]",
) -> None:
    """Insert the live-path-shaped corpus and wire its within-cluster
    edges -- the hop1/hop2 fan-out the bounded Layer-1 gather relies on."""
    insert_corpus(store, records)
    if edges:
        store.boost_edges(edges, edge_type="hebbian")
        from iai_mcp.store import flush_edge_buffer
        flush_edge_buffer(store)


def build_production_shaped_cue_set() -> "dict[str, list[CueSpec]]":
    """3-band natural-language cue set for the live-path-shaped corpus
    (real text, embedded through the production Embedder() at call time --
    never vector injection): one cue per activity cluster ("specific"), one
    loosely-worded cue that names no single activity ("vague"), and cues
    naming no corpus activity at all ("novel")."""
    specific = [
        CueSpec(text=f"What does Alice focus on during her {activity} sessions?", band="specific", mode="concept")
        for activity in _LIVE_PATH_ACTIVITIES
    ]
    vague = [
        CueSpec(text="Any updates on what Alice has been up to with her hobbies lately?", band="vague", mode="concept"),
        CueSpec(text="What has Alice been spending her free time on recently?", band="vague", mode="concept"),
        CueSpec(text="How are Alice's various weekend projects coming along?", band="vague", mode="concept"),
    ]
    novel = [
        CueSpec(text="the history of Roman aqueduct engineering", band="novel", mode="concept"),
        CueSpec(text="thermodynamic efficiency of a Stirling engine", band="novel", mode="concept"),
        CueSpec(text="migratory routes of humpback whales in the Pacific", band="novel", mode="concept"),
    ]
    return {"specific": specific, "vague": vague, "novel": novel}

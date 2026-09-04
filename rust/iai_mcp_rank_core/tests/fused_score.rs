//! Unit proofs for the fused scorer: the four-way ascending
//! candidate union, the `(-score, id)` tie-break (proven equal to the
//! `str(UUID(int=id))` tie-break), edge-type-excluded degree, internally
//! computed BM25/FTS (no precomputed lexical side-input), overlay-aware
//! post-write reads, the coverage invariant, and the k+margin window.
//!
//! Integration test target: sees only the crate's public API (`Inner`,
//! `DoubleBuffer`, `PendingOp`, `fused_score`, `FusedScoreParams`), matching
//! how a real caller reaches this crate.

use std::collections::{HashMap, HashSet};

use iai_mcp_rank_core::{fused_score, DoubleBuffer, FusedScoreParams, Inner, PendingOp};

const DIM: usize = 4;

/// Builds a minimal fixture `Inner` from parallel per-record rows. Every
/// numeric field defaults to a value chosen so unrelated terms stay inert
/// (zero cosine input externally, neutral stability, empty aaak/surface) —
/// each test overrides only the fields its assertion depends on.
struct Row {
    id: u128,
    surface: &'static str,
    aaak_index: &'static str,
    created_at: i64,
    stability: f32,
    tier: &'static str,
    edges: Vec<(u128, f32, &'static str)>,
}

fn row(id: u128) -> Row {
    Row {
        id,
        surface: "",
        aaak_index: "",
        created_at: 0,
        stability: 1.0,
        tier: "episodic",
        edges: Vec::new(),
    }
}

fn build_inner(rows: Vec<Row>) -> Inner {
    let n = rows.len();
    let ids: Vec<u128> = rows.iter().map(|r| r.id).collect();
    let vectors_flat = vec![0.0f32; n * DIM];
    let mut edge_map: HashMap<u128, Vec<(u128, f32, String)>> = HashMap::new();
    let surfaces: Vec<String> = rows.iter().map(|r| r.surface.to_string()).collect();
    let aaak_index: Vec<String> = rows.iter().map(|r| r.aaak_index.to_string()).collect();
    let created_at: Vec<i64> = rows.iter().map(|r| r.created_at).collect();
    let stability: Vec<f32> = rows.iter().map(|r| r.stability).collect();
    let tier: Vec<String> = rows.iter().map(|r| r.tier.to_string()).collect();
    let tags: Vec<Vec<String>> = vec![Vec::new(); n];
    let salience_level: Vec<u8> = vec![0u8; n];
    let centrality: Vec<f32> = vec![0.0f32; n];
    let pending: Vec<bool> = vec![false; n];
    for r in &rows {
        edge_map.insert(
            r.id,
            r.edges.iter().map(|(nb, w, et)| (*nb, *w, et.to_string())).collect(),
        );
    }

    Inner::from_columns(
        DIM,
        1,
        ids,
        vectors_flat,
        edge_map,
        surfaces,
        aaak_index,
        created_at,
        stability,
        tier,
        tags,
        salience_level,
        centrality,
        pending,
    )
    .expect("fixture build must not fail")
}

/// All-`false` flag vec sized to `n` -- the default T11/T12 input for
/// every test that does not itself assert the flag-driven boost.
fn zero_flags(n: usize) -> Vec<bool> {
    vec![false; n]
}

/// Baseline params over `pool_ids` with every optional lane disabled --
/// each test only overrides what its assertion depends on.
#[allow(clippy::too_many_arguments)]
fn base_params<'a>(
    pool_ids: &'a [u128],
    cosine: &'a [f64],
    structure_hv: &'a [Option<Vec<u8>>],
    t11_flags: &'a [bool],
    t12_flags: &'a [bool],
) -> FusedScoreParams<'a> {
    FusedScoreParams {
        pool_ids,
        cosine,
        structure_hv,
        cosine_top_indices: &[],
        spread_indices: &[],
        rich_indices: &[],
        lex_indices: &[],
        t11_flags,
        t12_flags,
        verbatim_filter: false,
        cue: "",
        now: 0,
        effective_w_degree: 0.0,
        effective_w_cosine: 1.0,
        excluded_edge_types: EMPTY_EXCLUDED.get_or_init(HashSet::new),
        spread_provenance: EMPTY_SPREAD.get_or_init(HashMap::new),
        w_spread_act: 0.0,
        spread_act_decay: 0.0,
        community_id_by_member: EMPTY_COMMUNITY_MEMBER.get_or_init(HashMap::new),
        community_scores: EMPTY_COMMUNITY_SCORES.get_or_init(HashMap::new),
        max_community_score: 0.0,
        mode_bias: 0.0,
        cos_spread_min: 0.02,
        structural_weight: 0.0,
        cue_structure_hv: None,
        lex_lane_enabled: false,
        min_idf: 0.0,
        lex_fusion_w: 0.0,
        k: 100,
        k_margin: 0,
    }
}

// Process-lifetime empty containers so `base_params`'s borrowed fields have
// something stable to point at across a whole test body.
static EMPTY_EXCLUDED: std::sync::OnceLock<HashSet<String>> = std::sync::OnceLock::new();
static EMPTY_SPREAD: std::sync::OnceLock<HashMap<u128, iai_mcp_rank_core::SpreadEntry>> = std::sync::OnceLock::new();
static EMPTY_COMMUNITY_MEMBER: std::sync::OnceLock<HashMap<u128, u128>> = std::sync::OnceLock::new();
static EMPTY_COMMUNITY_SCORES: std::sync::OnceLock<HashMap<u128, f64>> = std::sync::OnceLock::new();

fn winner_ids(result: &iai_mcp_rank_core::FusedScoreResult) -> HashSet<u128> {
    result.winners.iter().map(|w| w.id).collect()
}

// ---------------------------------------------------------------------
// Four-way ascending union + verbatim filter + coverage invariant
// ---------------------------------------------------------------------

#[test]
fn four_way_union_is_ascending_deduped_and_covers_every_member() {
    let ids: Vec<u128> = (100..=106).collect(); // positions 0..=6
    let inner = build_inner(ids.iter().map(|&id| row(id)).collect());
    let cosine = vec![0.0f64; ids.len()];
    let structure_hv: Vec<Option<Vec<u8>>> = vec![None; ids.len()];
    let t11_flags = zero_flags(cosine.len());
    let t12_flags = zero_flags(cosine.len());
    let mut params = base_params(&ids, &cosine, &structure_hv, &t11_flags, &t12_flags);
    // Overlapping on purpose: position 3 is in both cosine_top and spread;
    // position 0 is in both lex and cosine_top.
    params.cosine_top_indices = &[0, 3];
    params.spread_indices = &[3, 4];
    params.rich_indices = &[1];
    params.lex_indices = &[5, 0];
    params.k = 10;

    let result = fused_score(&inner, &params).expect("valid input");
    let expected_positions = [0u32, 1, 3, 4, 5];
    assert_eq!(result.coverage.expected, expected_positions.len() as u32);
    assert_eq!(result.coverage.resident, expected_positions.len() as u32);
    assert!(result.coverage.all_resident);

    let expected_ids: HashSet<u128> = expected_positions.iter().map(|&p| ids[p as usize]).collect();
    assert_eq!(winner_ids(&result), expected_ids);
}

#[test]
fn verbatim_filter_keeps_only_episodic_tier_members() {
    let mut rows = vec![row(1), row(2), row(3)];
    rows[0].tier = "episodic";
    rows[1].tier = "semantic";
    rows[2].tier = "episodic";
    let ids: Vec<u128> = rows.iter().map(|r| r.id).collect();
    let inner = build_inner(rows);
    let cosine = vec![0.0f64; ids.len()];
    let structure_hv: Vec<Option<Vec<u8>>> = vec![None; ids.len()];
    let t11_flags = zero_flags(cosine.len());
    let t12_flags = zero_flags(cosine.len());
    let mut params = base_params(&ids, &cosine, &structure_hv, &t11_flags, &t12_flags);
    params.cosine_top_indices = &[0, 1, 2];
    params.verbatim_filter = true;

    let result = fused_score(&inner, &params).expect("valid input");
    assert_eq!(result.coverage.expected, 2);
    assert_eq!(winner_ids(&result), HashSet::from([1u128, 3u128]));
}

#[test]
fn verbatim_filter_treats_an_unset_resident_tier_as_episodic() {
    // `feed`'s tier defaults to an empty string when the caller omits it;
    // Python's `getattr(rec, "tier", "episodic")` treats a genuinely
    // missing attribute as episodic. An empty resident tier must match
    // that default, not be excluded as if it were some other tier.
    let mut rows = vec![row(1), row(2)];
    rows[0].tier = "";
    rows[1].tier = "semantic";
    let ids: Vec<u128> = rows.iter().map(|r| r.id).collect();
    let inner = build_inner(rows);
    let cosine = vec![0.0f64; ids.len()];
    let structure_hv: Vec<Option<Vec<u8>>> = vec![None; ids.len()];
    let t11_flags = zero_flags(cosine.len());
    let t12_flags = zero_flags(cosine.len());
    let mut params = base_params(&ids, &cosine, &structure_hv, &t11_flags, &t12_flags);
    params.cosine_top_indices = &[0, 1];
    params.verbatim_filter = true;

    let result = fused_score(&inner, &params).expect("valid input");
    assert_eq!(winner_ids(&result), HashSet::from([1u128]));
}

#[test]
fn coverage_invariant_reports_a_pool_position_with_no_resident_row() {
    let ids: Vec<u128> = vec![1, 2];
    let inner = build_inner(ids.iter().map(|&id| row(id)).collect());
    // pool_ids names a THIRD id (999) at position 2 that is not resident in
    // `inner` at all -- the candidate-selection side named a candidate the
    // index does not (yet, or ever) hold.
    let pool_ids: Vec<u128> = vec![1, 2, 999];
    let cosine = vec![0.0f64; 3];
    let structure_hv: Vec<Option<Vec<u8>>> = vec![None; 3];
    let t11_flags = zero_flags(cosine.len());
    let t12_flags = zero_flags(cosine.len());
    let mut params = base_params(&pool_ids, &cosine, &structure_hv, &t11_flags, &t12_flags);
    params.cosine_top_indices = &[0, 1, 2];

    let result = fused_score(&inner, &params).expect("valid input");
    assert_eq!(result.coverage.expected, 3);
    assert_eq!(result.coverage.resident, 2);
    assert!(!result.coverage.all_resident);
    assert_eq!(winner_ids(&result), HashSet::from([1u128, 2u128]));
}

// ---------------------------------------------------------------------
// Tie-break: (-score, id) matches Python's (-score, str(uuid))
// ---------------------------------------------------------------------

/// Canonical `str(UUID(int=n))` form: 32 lowercase hex digits, hyphens at
/// fixed offsets 8/13/18/23 -- built by hand (no `uuid` crate dependency)
/// to keep this proof independent of any third-party formatting choice.
fn uuid_string(id: u128) -> String {
    let hex = format!("{id:032x}");
    format!("{}-{}-{}-{}-{}", &hex[0..8], &hex[8..12], &hex[12..16], &hex[16..20], &hex[20..32])
}

#[test]
fn tie_break_matches_uuid_string_order() {
    // Values straddling the 9->a hex-nibble boundary at several digit
    // positions, plus a spread of magnitudes.
    let ids: Vec<u128> = vec![
        0,
        1,
        9,
        10,
        0x99,
        0x9a,
        0xa0,
        0x9999_9999,
        0x9999_999a,
        u128::MAX,
        u128::MAX - 1,
        (1u128 << 64) - 1,
        1u128 << 64,
    ];
    let mut by_id = ids.clone();
    by_id.sort();

    let mut by_string = ids.clone();
    by_string.sort_by_key(|a| uuid_string(*a));

    assert_eq!(by_id, by_string, "ascending u128 order must equal ascending str(UUID(int=id)) order");
}

#[test]
fn winners_tie_break_ascending_id_on_equal_score() {
    // Three candidates with byte-identical inputs (same cosine, no other
    // active term) -- their scores tie exactly, so serialization order
    // must be ascending id.
    let ids: Vec<u128> = vec![500, 100, 300];
    let inner = build_inner(ids.iter().map(|&id| row(id)).collect());
    let cosine = vec![0.7f64; 3];
    let structure_hv: Vec<Option<Vec<u8>>> = vec![None; 3];
    let t11_flags = zero_flags(cosine.len());
    let t12_flags = zero_flags(cosine.len());
    let mut params = base_params(&ids, &cosine, &structure_hv, &t11_flags, &t12_flags);
    params.cosine_top_indices = &[0, 1, 2];

    let result = fused_score(&inner, &params).expect("valid input");
    let observed: Vec<u128> = result.winners.iter().map(|w| w.id).collect();
    assert_eq!(observed, vec![100u128, 300, 500]);
}

// ---------------------------------------------------------------------
// Degree excludes the ranking-excluded edge types, not just raw adjacency
// count
// ---------------------------------------------------------------------

#[test]
fn degree_term_excludes_configured_edge_types_not_just_counts_all() {
    // A<->B only via an excluded type; C<->D only via a non-excluded type.
    // A fixture with NO excluded-type edges would not catch a regression to
    // the unfiltered `resolved_degree`.
    let mut rows = vec![row(1), row(2), row(3), row(4)];
    rows[0].edges = vec![(2, 1.0, "consolidated_from")];
    rows[1].edges = vec![(1, 1.0, "consolidated_from")];
    rows[2].edges = vec![(4, 1.0, "hebbian")];
    rows[3].edges = vec![(3, 1.0, "hebbian")];
    let ids: Vec<u128> = rows.iter().map(|r| r.id).collect();
    let inner = build_inner(rows);
    let cosine = vec![0.0f64; 4];
    let structure_hv: Vec<Option<Vec<u8>>> = vec![None; 4];
    let mut excluded = HashSet::new();
    excluded.insert("consolidated_from".to_string());
    let t11_flags = zero_flags(cosine.len());
    let t12_flags = zero_flags(cosine.len());
    let mut params = base_params(&ids, &cosine, &structure_hv, &t11_flags, &t12_flags);
    params.cosine_top_indices = &[0, 1, 2, 3];
    params.effective_w_degree = 1.0;
    params.excluded_edge_types = &excluded;
    // All-zero cosine makes the candidate head fully flat; disable the
    // flat-cosine damp (a separate, correctly-firing term) so it does not
    // zero out effective_w_degree and mask this test's own assertion.
    params.cos_spread_min = 0.0;

    let result = fused_score(&inner, &params).expect("valid input");
    let by_id: HashMap<u128, f64> = result.winners.iter().map(|w| (w.id, w.partial_score)).collect();
    // A/B's only edge is excluded -> filtered degree 0 -> deg_norm 0.
    assert!((by_id[&1] - 0.0).abs() < 1e-12);
    assert!((by_id[&2] - 0.0).abs() < 1e-12);
    // C/D's edge is not excluded -> filtered degree 1 == the corpus max ->
    // deg_norm 1.0 -> partial_score == effective_w_degree.
    assert!((by_id[&3] - 1.0).abs() < 1e-12);
    assert!((by_id[&4] - 1.0).abs() < 1e-12);
}

// ---------------------------------------------------------------------
// T11/T12 flag-driven boosts (the flags themselves are computed Python-
// side from the candidate's own hydrated surface -- see
// `_t11_t12_flags`/`pipeline.py` -- fused_score only applies the
// multiplier at the passed-in flag's position) and BM25 lexical fusion
// computed internally from the resident postings, no precomputed
// lexical side-input.
// ---------------------------------------------------------------------

#[test]
fn t12_flag_true_applies_the_x3_fts_multiplier_at_its_pool_position() {
    let ids: Vec<u128> = vec![1, 2];
    let inner = build_inner(ids.iter().map(|&id| row(id)).collect());
    let cosine = vec![0.5f64; 2]; // nonzero base so the x3.0 multiplier is observable
    let structure_hv: Vec<Option<Vec<u8>>> = vec![None; 2];
    let t11_flags = zero_flags(cosine.len());
    let mut t12_flags = zero_flags(cosine.len());
    t12_flags[0] = true;
    let mut params = base_params(&ids, &cosine, &structure_hv, &t11_flags, &t12_flags);
    params.cosine_top_indices = &[0, 1];

    let result = fused_score(&inner, &params).expect("valid input");
    let by_id: HashMap<u128, f64> = result.winners.iter().map(|w| (w.id, w.partial_score)).collect();
    assert!((by_id[&1] - 1.5).abs() < 1e-9, "fts x3.0 must apply: got {}", by_id[&1]);
    assert!((by_id[&2] - 0.5).abs() < 1e-9, "flag false, no boost: got {}", by_id[&2]);
}

#[test]
fn t11_flag_true_applies_the_x2_trigram_multiplier_at_its_pool_position() {
    let ids: Vec<u128> = vec![1, 2];
    let inner = build_inner(ids.iter().map(|&id| row(id)).collect());
    let cosine = vec![0.5f64; 2];
    let structure_hv: Vec<Option<Vec<u8>>> = vec![None; 2];
    let mut t11_flags = zero_flags(cosine.len());
    t11_flags[0] = true;
    let t12_flags = zero_flags(cosine.len());
    let mut params = base_params(&ids, &cosine, &structure_hv, &t11_flags, &t12_flags);
    params.cosine_top_indices = &[0, 1];

    let result = fused_score(&inner, &params).expect("valid input");
    let by_id: HashMap<u128, f64> = result.winners.iter().map(|w| (w.id, w.partial_score)).collect();
    assert!((by_id[&1] - 1.0).abs() < 1e-9, "trigram x2.0 must apply: got {}", by_id[&1]);
    assert!((by_id[&2] - 0.5).abs() < 1e-9, "flag false, no boost: got {}", by_id[&2]);
}

#[test]
fn bm25_lex_rank_matches_via_a_shared_token_computed_from_resident_postings() {
    let mut rows = vec![row(1), row(2)];
    rows[0].surface = "a marimba concert last night";
    rows[1].surface = "completely different subject entirely";
    let ids: Vec<u128> = rows.iter().map(|r| r.id).collect();
    let inner = build_inner(rows);
    let cosine = vec![0.5f64; 2];
    let structure_hv: Vec<Option<Vec<u8>>> = vec![None; 2];
    let t11_flags = zero_flags(cosine.len());
    let t12_flags = zero_flags(cosine.len());
    let mut params = base_params(&ids, &cosine, &structure_hv, &t11_flags, &t12_flags);
    params.cosine_top_indices = &[0, 1];
    params.cue = "xylophone_quartz_marimba"; // shares the "marimba" token only
    params.lex_lane_enabled = true;
    params.min_idf = 0.0;
    params.lex_fusion_w = 1.0;

    let result = fused_score(&inner, &params).expect("valid input");
    let by_id: HashMap<u128, f64> = result.winners.iter().map(|w| (w.id, w.partial_score)).collect();
    // rank 0 (only match) -> + lex_fusion_w / (1 + 0) == 1.0 on top of base 0.5.
    assert!((by_id[&1] - 1.5).abs() < 1e-9, "lex fusion add must apply to the token match: got {}", by_id[&1]);
    assert!((by_id[&2] - 0.5).abs() < 1e-9, "no shared token, no lex contribution: got {}", by_id[&2]);
}

#[test]
fn bm25_lex_lane_disabled_never_contributes() {
    let mut rows = vec![row(1)];
    rows[0].surface = "a marimba concert last night";
    let ids: Vec<u128> = rows.iter().map(|r| r.id).collect();
    let inner = build_inner(rows);
    let cosine = vec![0.5f64; 1];
    let structure_hv: Vec<Option<Vec<u8>>> = vec![None; 1];
    let t11_flags = zero_flags(cosine.len());
    let t12_flags = zero_flags(cosine.len());
    let mut params = base_params(&ids, &cosine, &structure_hv, &t11_flags, &t12_flags);
    params.cosine_top_indices = &[0];
    // Shares the "marimba" token (would match BM25 if the lane fired) but
    // is neither a literal substring nor trigram-similar enough to also
    // trigger fts/trigram -- isolates the lex-lane-disabled assertion.
    params.cue = "xylophone_quartz_marimba";
    params.lex_lane_enabled = false; // e.g. cue is not identifier-grade
    params.lex_fusion_w = 1.0;

    let result = fused_score(&inner, &params).expect("valid input");
    assert!((result.winners[0].partial_score - 0.5).abs() < 1e-9);
}

// ---------------------------------------------------------------------
// Overlay-aware reads: a post-write recall must score CURRENT, not
// committed-stale, adjacency/postings/surface text
// ---------------------------------------------------------------------

#[test]
fn post_write_recall_scores_current_degree_and_postings_not_committed_stale() {
    let mut rows = vec![row(1), row(2), row(3)];
    rows[0].surface = "original text with no shared token";
    let inner0 = build_inner(rows);
    let buf = DoubleBuffer::new(inner0);

    // Before any write: id 1 has no edges and a surface with no "marimba"
    // token.
    let baseline = buf.snapshot(1).expect("matching-generation read");
    let cosine = vec![0.5f64; 3];
    let structure_hv: Vec<Option<Vec<u8>>> = vec![None; 3];
    let ids: Vec<u128> = vec![1, 2, 3];
    let t11_flags = zero_flags(cosine.len());
    let t12_flags = zero_flags(cosine.len());
    let mut params = base_params(&ids, &cosine, &structure_hv, &t11_flags, &t12_flags);
    params.cosine_top_indices = &[0];
    // Shares the "marimba" token with the post-write surface below but is
    // neither a literal substring nor trigram-similar, isolating this test
    // to the degree + BM25 terms it names.
    params.cue = "xylophone_quartz_marimba";
    params.lex_lane_enabled = true;
    params.min_idf = 0.0;
    params.lex_fusion_w = 1.0;
    params.effective_w_degree = 1.0;
    let before = fused_score(&baseline, &params).expect("valid input");
    assert!((before.winners[0].partial_score - 0.5).abs() < 1e-9, "no edges, no token match yet");

    // Feed an upsert that gives id 1 a new edge (to 2) AND a surface
    // containing "marimba" -- then apply it via a stale-generation
    // snapshot (the incremental read-path apply, not a wholesale rebuild).
    buf.feed(PendingOp::Upsert {
        id: 1,
        vector: vec![0.0; DIM],
        edges: vec![(2, 1.0, "hebbian".to_string())],
        surface: "now mentions marimba explicitly".to_string(),
        aaak_index: String::new(),
        created_at: 0,
        stability: 1.0,
        tier: "episodic".to_string(),
        tags: Vec::new(),
        salience_level: 0,
        centrality: 0.0,
        pending: false,
    })
    .expect("feed must succeed");
    let updated = buf.snapshot(2).expect("stale-generation apply");

    let after = fused_score(&updated, &params).expect("valid input");
    // Degree now 1 (max degree across the corpus is 1 -> deg_norm 1.0) and
    // the BM25 lane now matches "marimba" -> + lex_fusion_w / 1.
    let expected = 0.5 /* cosine */ + 1.0 /* effective_w_degree * deg_norm */ + 1.0 /* lex fusion add */;
    assert!(
        (after.winners[0].partial_score - expected).abs() < 1e-9,
        "post-write read must reflect the CURRENT overlay, not the committed-stale CSR: got {}",
        after.winners[0].partial_score
    );
}

// ---------------------------------------------------------------------
// Single hybrid call: k + k_margin over-fetch window
// ---------------------------------------------------------------------

#[test]
fn winners_window_is_bounded_to_k_plus_margin() {
    let ids: Vec<u128> = (1..=10).collect();
    let inner = build_inner(ids.iter().map(|&id| row(id)).collect());
    // Distinct, strictly descending-by-id-would-not-hold scores: use id
    // itself as the cosine driver so ordering is unambiguous.
    let cosine: Vec<f64> = ids.iter().map(|&id| id as f64 / 100.0).collect();
    let structure_hv: Vec<Option<Vec<u8>>> = vec![None; ids.len()];
    let t11_flags = zero_flags(cosine.len());
    let t12_flags = zero_flags(cosine.len());
    let mut params = base_params(&ids, &cosine, &structure_hv, &t11_flags, &t12_flags);
    let all_positions: Vec<u32> = (0u32..10).collect();
    params.cosine_top_indices = &all_positions;
    params.k = 2;
    params.k_margin = 1;

    let result = fused_score(&inner, &params).expect("valid input");
    assert_eq!(result.winners.len(), 3);
    let observed: Vec<u128> = result.winners.iter().map(|w| w.id).collect();
    // Highest cosine == highest id here -> winners are the top-3 ids
    // descending.
    assert_eq!(observed, vec![10u128, 9, 8]);
}

// ---------------------------------------------------------------------
// pre_gain_base + term_multiplier let a caller reinsert a per-call
// multiplicative gain at the same insertion point today's formula uses,
// without a second scoring pass.
// ---------------------------------------------------------------------

#[test]
fn pre_gain_base_and_term_multiplier_reconstruct_a_later_applied_gain() {
    let ids: Vec<u128> = vec![1];
    let inner = build_inner(ids.iter().map(|&id| row(id)).collect());
    let cosine = vec![0.5f64]; // base_s == 0.5 (degree/aaak/age/spread/community all 0)
    let structure_hv: Vec<Option<Vec<u8>>> = vec![None];
    // Both flags true so term_multiplier != 1.0 -- a multiplier-of-1.0
    // fixture would pass this reconstruction formula vacuously.
    let t11_flags = vec![true];
    let t12_flags = vec![true];
    let mut params = base_params(&ids, &cosine, &structure_hv, &t11_flags, &t12_flags);
    params.cosine_top_indices = &[0];

    let result = fused_score(&inner, &params).expect("valid input");
    let winner = &result.winners[0];
    assert!((winner.pre_gain_base - 0.5).abs() < 1e-9, "got {}", winner.pre_gain_base);
    assert!((winner.term_multiplier - 6.0).abs() < 1e-9, "trigram*fts must both fire: got {}", winner.term_multiplier);

    // Simulate a later per-call multiplicative gain g applied at the
    // documented insertion point (onto pre_gain_base, BEFORE the
    // stability lift and the trigram/fts multipliers) -- this is the
    // "true" value today's Python formula would have produced.
    let g = 2.0f64;
    let stability_lift = 0.0; // stability == 1.0 in the fixture default
    let lex_add = 0.0; // lex lane disabled by default
    let true_value = (winner.pre_gain_base * g + stability_lift) * winner.term_multiplier + lex_add;

    // The reconstruction formula: recover `true_value` from
    // `partial_score` + the two extra fields, with NO second scoring pass
    // and no need for stability_lift/lex_add individually.
    let reconstructed = winner.partial_score + winner.pre_gain_base * winner.term_multiplier * (g - 1.0);
    assert!(
        (reconstructed - true_value).abs() < 1e-9,
        "reconstructed {reconstructed} must equal the true gain-applied value {true_value}"
    );

    // A gain of 1.0 (no profile modulation) must reconstruct to exactly
    // partial_score itself.
    let identity_reconstructed = winner.partial_score + winner.pre_gain_base * winner.term_multiplier * (1.0 - 1.0);
    assert!((identity_reconstructed - winner.partial_score).abs() < 1e-12);
}

// ---------------------------------------------------------------------
// FFI-boundary input validation: a length/index mismatch must
// raise, never silently mis-score.
// ---------------------------------------------------------------------

#[test]
fn mismatched_cosine_array_length_is_rejected() {
    let ids: Vec<u128> = vec![1, 2];
    let inner = build_inner(ids.iter().map(|&id| row(id)).collect());
    let cosine = vec![0.0f64; 1]; // wrong length vs pool_ids
    let structure_hv: Vec<Option<Vec<u8>>> = vec![None; 2];
    let t11_flags = zero_flags(cosine.len());
    let t12_flags = zero_flags(cosine.len());
    let params = base_params(&ids, &cosine, &structure_hv, &t11_flags, &t12_flags);
    assert!(fused_score(&inner, &params).is_err());
}

#[test]
fn out_of_bounds_pool_position_in_an_index_array_is_rejected() {
    let ids: Vec<u128> = vec![1, 2];
    let inner = build_inner(ids.iter().map(|&id| row(id)).collect());
    let cosine = vec![0.0f64; 2];
    let structure_hv: Vec<Option<Vec<u8>>> = vec![None; 2];
    let t11_flags = zero_flags(cosine.len());
    let t12_flags = zero_flags(cosine.len());
    let mut params = base_params(&ids, &cosine, &structure_hv, &t11_flags, &t12_flags);
    params.cosine_top_indices = &[5]; // out of bounds for a 2-element pool
    assert!(fused_score(&inner, &params).is_err());
}

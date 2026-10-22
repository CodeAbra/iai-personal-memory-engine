//! fusion -- pure Reciprocal Rank Fusion (K=60) over already-ranked lane
//! outputs, plus the weak-base-signal gate. No I/O, no brain dependency;
//! callers pass in per-lane path lists already deduped and normalized to
//! the same join key (see `_normalize_hit_path` on the Python side).
//!
//! Gating is one code path: the semantic lane's rank list is simply
//! omitted from the `lanes` slice for a given query when the gate does
//! not fire -- `rrf_merge` treats any number of lanes identically.

use std::collections::HashMap;

pub const RRF_K: f64 = 60.0;

/// Dev-split-tuned gate threshold, frozen and pinned by a regression test
/// (see `frozen_gate_threshold_is_pinned` below and
/// `tests/test_fusion_gate_frozen.py`). Update this value together with
/// both tests when a dev-split sweep produces a winner -- never after the
/// held-out test split has run.
pub const FROZEN_GATE_THRESHOLD: f64 = 0.03;

/// Reciprocal-rank fusion over `lanes`. Each lane is already rank-ordered
/// (index 0 == rank 1) and pre-normalized by the caller to the same join
/// key. A path present in multiple lanes collapses to exactly ONE row,
/// scored as the sum of every lane's `1/(RRF_K + rank)` contribution --
/// never duplicated, never split. A lane that omits a path contributes 0
/// for it. Result is sorted by fused score descending; ties keep the
/// lane-insertion (first-seen) order.
pub fn rrf_merge(lanes: &[Vec<String>]) -> Vec<(String, f64)> {
    let mut scores: HashMap<&str, f64> = HashMap::new();
    let mut order: Vec<&str> = Vec::new();
    for lane in lanes {
        for (idx, path) in lane.iter().enumerate() {
            let rank = idx + 1;
            let contribution = 1.0 / (RRF_K + rank as f64);
            match scores.get_mut(path.as_str()) {
                Some(score) => *score += contribution,
                None => {
                    scores.insert(path.as_str(), contribution);
                    order.push(path.as_str());
                }
            }
        }
    }
    let mut merged: Vec<(String, f64)> = order
        .into_iter()
        .map(|path| (path.to_string(), scores[path]))
        .collect();
    merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    merged
}

/// True when the base fused top result is weak: the top fused score is
/// below `threshold` (or `FROZEN_GATE_THRESHOLD` when `threshold` is
/// `None` -- every shipped caller that omits the threshold gets the
/// dev-split-tuned value, never an ad hoc fallback). Several lanes
/// agreeing at the top produces a high summed RRF score; a lone weak hit
/// produces a low one -- computable from a single query's own base-lane
/// outputs, no corpus-wide distribution. An empty base fusion (no base
/// lane returned anything) is weak by construction.
pub fn gate_is_weak(base_fused: &[(String, f64)], threshold: Option<f64>) -> bool {
    let t = threshold.unwrap_or(FROZEN_GATE_THRESHOLD);
    match base_fused.first() {
        None => true,
        Some((_, score)) => *score < t,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_k_is_60() {
        assert_eq!(RRF_K, 60.0);
    }

    #[test]
    fn same_document_across_lanes_collapses_to_one_summed_row() {
        let lane_a = vec!["shared.py".to_string(), "only_a.py".to_string()];
        let lane_b = vec!["shared.py".to_string(), "only_b.py".to_string()];
        let merged = rrf_merge(&[lane_a, lane_b]);

        let shared_rows: Vec<&(String, f64)> =
            merged.iter().filter(|(p, _)| p == "shared.py").collect();
        assert_eq!(shared_rows.len(), 1, "shared.py must appear exactly once");

        let expected = 1.0 / (RRF_K + 1.0) + 1.0 / (RRF_K + 1.0);
        assert!((shared_rows[0].1 - expected).abs() < 1e-12);
    }

    #[test]
    fn disjoint_lanes_concatenate_ranked_by_descending_fused_score() {
        let lane_a = vec!["a1.py".to_string(), "a2.py".to_string()];
        let lane_b = vec!["b1.py".to_string()];
        let merged = rrf_merge(&[lane_a, lane_b]);

        // a1.py: rank 1 in lane_a only -> 1/61. b1.py: rank 1 in lane_b only -> 1/61.
        // a2.py: rank 2 in lane_a only -> 1/62. Descending: a1/b1 tie above a2.
        let scores: HashMap<&str, f64> = merged.iter().map(|(p, s)| (p.as_str(), *s)).collect();
        assert!((scores["a1.py"] - 1.0 / 61.0).abs() < 1e-12);
        assert!((scores["b1.py"] - 1.0 / 61.0).abs() < 1e-12);
        assert!((scores["a2.py"] - 1.0 / 62.0).abs() < 1e-12);
        assert!(scores["a1.py"] > scores["a2.py"]);
        assert_eq!(merged.len(), 3);
    }

    #[test]
    fn a_lane_that_omits_a_document_contributes_zero_for_it() {
        let lane_a = vec!["present_everywhere.py".to_string(), "only_a.py".to_string()];
        let lane_b = vec!["present_everywhere.py".to_string()];
        let lane_c: Vec<String> = Vec::new();
        let merged = rrf_merge(&[lane_a, lane_b, lane_c]);

        let scores: HashMap<&str, f64> = merged.iter().map(|(p, s)| (p.as_str(), *s)).collect();
        let expected_shared = 1.0 / (RRF_K + 1.0) + 1.0 / (RRF_K + 1.0);
        assert!((scores["present_everywhere.py"] - expected_shared).abs() < 1e-12);
        assert!((scores["only_a.py"] - 1.0 / (RRF_K + 2.0)).abs() < 1e-12);
    }

    #[test]
    fn gate_fires_on_a_weak_base_top_and_not_on_a_strong_one() {
        let weak_base = vec![("lone_hit.py".to_string(), 1.0 / (RRF_K + 1.0))];
        assert!(gate_is_weak(&weak_base, None));

        // Three lanes agreeing at rank 1 produces a high fused top score.
        let strong_score = 1.0 / (RRF_K + 1.0) * 3.0;
        let strong_base = vec![("agreed_hit.py".to_string(), strong_score)];
        assert!(!gate_is_weak(&strong_base, None));
    }

    #[test]
    fn caller_supplied_threshold_overrides_the_frozen_default() {
        let base = vec![("hit.py".to_string(), 0.03)];
        // Default threshold (frozen at 0.03): 0.03 is not weak.
        assert!(!gate_is_weak(&base, None));
        // A caller-supplied override takes precedence over the default and
        // can flip the decision either way.
        assert!(gate_is_weak(&base, Some(0.05)));
        assert!(!gate_is_weak(&base, Some(0.01)));
    }

    #[test]
    fn empty_base_fusion_is_weak_by_construction() {
        assert!(gate_is_weak(&[], None));
    }

    #[test]
    fn frozen_gate_threshold_is_pinned() {
        assert_eq!(FROZEN_GATE_THRESHOLD, 0.03);
    }

    #[test]
    fn omitted_threshold_resolves_to_the_frozen_gate_value() {
        // A score between the old un-tuned 0.02 and the frozen 0.03 only
        // gates as weak if a caller passing no explicit threshold actually
        // resolves to FROZEN_GATE_THRESHOLD, not a stale fallback.
        let base = vec![("hit.py".to_string(), 0.025)];
        assert!(gate_is_weak(&base, None));
    }
}

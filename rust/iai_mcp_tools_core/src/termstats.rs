//! Resident cross-segment term-stat table -- corpus-wide document
//! frequency, incrementally folded in on publish/tombstone, never
//! recomputed from scratch on a query. A per-segment df alone cannot give
//! comparable IDF across segments; this table is the resident sum.

use std::collections::HashMap;

use crate::postings::PostingsCsr;

#[derive(Clone, Default)]
pub struct TermStats {
    /// feature_id -> document frequency, summed across every folded-in
    /// segment. Never a raw token string. Crate-visible so the on-disk
    /// manifest (index.rs) can serialize/reconstruct this table without
    /// re-scanning any segment.
    pub(crate) df: HashMap<u32, u32>,
    pub(crate) n_docs: u32,
    pub(crate) total_len: u64,
}

impl TermStats {
    pub fn new() -> Self {
        TermStats::default()
    }

    pub fn n_docs(&self) -> u32 {
        self.n_docs
    }

    pub fn df(&self, feature_id: u32) -> u32 {
        self.df.get(&feature_id).copied().unwrap_or(0)
    }

    /// Matches `rebuild_postings_csr`'s own n_docs==0 floor of 1.0.
    pub fn avg_len(&self) -> f64 {
        if self.n_docs == 0 {
            1.0
        } else {
            (self.total_len as f64 / self.n_docs as f64).max(1.0)
        }
    }

    /// Folds one segment's distinct-token df contributions in -- O(distinct
    /// tokens in the segment), never a corpus-wide rescan.
    pub fn add_segment(&mut self, postings: &PostingsCsr) {
        for (&feature_id, &compact_id) in &postings.token_lookup {
            let start = postings.post_offsets[compact_id as usize];
            let end = postings.post_offsets[compact_id as usize + 1];
            let local_df = end - start;
            *self.df.entry(feature_id).or_insert(0) += local_df;
        }
        self.n_docs += postings.n_docs;
        self.total_len += postings.total_len;
    }

    /// Reverses `add_segment`'s exact contributions for the same segment.
    /// A zero-count entry is dropped so `df` never carries stale zero
    /// rows. Must restore the prior totals exactly -- used on tombstone
    /// (writes arrive in a later plan), never on a merge (a merge does
    /// not change which files/tokens exist, so it must not touch this
    /// table at all).
    #[allow(dead_code)] // caller (per-file tombstone removal) arrives in a later plan
    pub fn remove_segment(&mut self, postings: &PostingsCsr) {
        for (&feature_id, &compact_id) in &postings.token_lookup {
            let start = postings.post_offsets[compact_id as usize];
            let end = postings.post_offsets[compact_id as usize + 1];
            let local_df = end - start;
            if let Some(entry) = self.df.get_mut(&feature_id) {
                *entry = entry.saturating_sub(local_df);
                if *entry == 0 {
                    self.df.remove(&feature_id);
                }
            }
        }
        self.n_docs = self.n_docs.saturating_sub(postings.n_docs);
        self.total_len = self.total_len.saturating_sub(postings.total_len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postings::rebuild_postings_csr;
    use crate::tokenize::{feature_id, tokenize};
    use std::collections::HashMap as StdHashMap;

    fn csr_from_docs(docs: &[&str]) -> PostingsCsr {
        let freqs: Vec<StdHashMap<u32, u32>> = docs
            .iter()
            .map(|text| {
                let mut freq: StdHashMap<u32, u32> = StdHashMap::new();
                for tok in tokenize(text) {
                    *freq.entry(feature_id(&tok)).or_insert(0) += 1;
                }
                freq
            })
            .collect();
        rebuild_postings_csr(docs.len(), &freqs)
    }

    #[test]
    fn term_stats_global_df_sums_across_segments_sharing_a_token() {
        let seg_a = csr_from_docs(&["zebra runs fast", "zebra sleeps"]); // local df(zebra) = 2
        let seg_b = csr_from_docs(&["zebra hides", "zebra eats", "zebra jumps"]); // local df(zebra) = 3

        let mut stats = TermStats::new();
        stats.add_segment(&seg_a);
        stats.add_segment(&seg_b);

        let zebra = feature_id("zebra");
        assert_eq!(
            stats.df(zebra),
            5,
            "global df must be the SUM across segments, not either segment's local count"
        );
        assert_eq!(stats.n_docs(), 5);
    }

    #[test]
    fn term_stats_remove_segment_restores_prior_totals_exactly() {
        let seg_a = csr_from_docs(&["giraffe walks", "giraffe eats"]);
        let seg_b = csr_from_docs(&["giraffe runs"]);

        let mut stats = TermStats::new();
        stats.add_segment(&seg_a);
        let prior_n_docs = stats.n_docs();
        let prior_df = stats.df(feature_id("giraffe"));
        let prior_total_len = stats.total_len;

        stats.add_segment(&seg_b);
        stats.remove_segment(&seg_b);

        assert_eq!(stats.n_docs(), prior_n_docs);
        assert_eq!(stats.df(feature_id("giraffe")), prior_df);
        assert_eq!(stats.total_len, prior_total_len);
    }

    #[test]
    fn term_stats_remove_segment_drops_zero_count_entries() {
        let seg = csr_from_docs(&["falcon soars"]);
        let mut stats = TermStats::new();
        stats.add_segment(&seg);
        stats.remove_segment(&seg);

        assert_eq!(stats.df(feature_id("falcon")), 0);
        assert_eq!(stats.n_docs(), 0);
        assert!(stats.df.is_empty(), "zero-count df entries must be dropped, not left as stale zeros");
    }
}

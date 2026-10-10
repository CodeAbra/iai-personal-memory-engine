//! Robertson/Sparck-Jones BM25 scoring, k1=1.2, b=0.75.

use crate::termstats::TermStats;

pub const BM25_K1: f64 = 1.2;
pub const BM25_B: f64 = 0.75;

/// Length normalization off -- for the symbol lane's 1-3-token symbol-name
/// "documents", where document length is not a meaningful signal.
pub const SYMBOL_BM25_B: f64 = 0.0;

/// BM25 with `b` as an explicit parameter. At `b=0` the length term
/// vanishes (denom reduces to `tf + k1`), so document length does not
/// affect the score.
pub fn bm25_score_b(tf: f64, df: f64, length: f64, n_docs: f64, avg_len: f64, b: f64) -> f64 {
    let idf = (1.0 + (n_docs - df + 0.5) / (df + 0.5)).ln();
    let denom = tf + BM25_K1 * (1.0 - b + b * length / avg_len);
    idf * tf * (BM25_K1 + 1.0) / denom
}

pub fn bm25_score(tf: f64, df: f64, length: f64, n_docs: f64, avg_len: f64) -> f64 {
    bm25_score_b(tf, df, length, n_docs, avg_len, BM25_B)
}

/// BM25 read through the resident global term-stat table: `df`/`n_docs`
/// (and thus IDF) come from the corpus-wide table, comparable across
/// segments; `tf` and `length` stay local to the per-segment CSR the hit
/// was found in.
pub fn bm25_score_global(tf: f64, feature_id: u32, length: f64, stats: &TermStats) -> f64 {
    let df = stats.df(feature_id) as f64;
    let n_docs = stats.n_docs() as f64;
    let avg_len = stats.avg_len();
    bm25_score(tf, df, length, n_docs, avg_len)
}

/// Mirrors `bm25_score_global`'s shape at `SYMBOL_BM25_B` -- the caller
/// passes the symbol lane's OWN `TermStats` instance, separate from the
/// whole-file lane's, so `stats` carries symbol-corpus df/n_docs/avg_len.
pub fn symbol_bm25_score_global(tf: f64, feature_id: u32, length: f64, stats: &TermStats) -> f64 {
    let df = stats.df(feature_id) as f64;
    let n_docs = stats.n_docs() as f64;
    let avg_len = stats.avg_len();
    bm25_score_b(tf, df, length, n_docs, avg_len, SYMBOL_BM25_B)
}

#[cfg(test)]
mod tests {
    use super::*;

    // tf=2, df=1, length=10, n_docs=3, avg_len=8.
    // idf = ln(1 + (3-1+0.5)/(1+0.5)) = ln(8/3) = 0.98082925301...
    // denom = 2 + 1.2*(0.25 + 0.75*10/8) = 2 + 1.2*1.1875 = 3.425
    // score = idf * 2 * 2.2 / 3.425 = 1.26004341...
    #[test]
    fn matches_hand_computed_value_tuple_one() {
        let score = bm25_score(2.0, 1.0, 10.0, 3.0, 8.0);
        assert!(
            (score - 1.260_043_41).abs() < 1e-6,
            "score {score} != hand-computed 1.26004341"
        );
    }

    // tf=1, df=2, length==avg_len, n_docs=4 -- length/avg_len cancels to 1,
    // so denom = tf + k1 = 1 + 1.2 = 2.2 = tf*(k1+1), leaving score == idf.
    // idf = ln(1 + (4-2+0.5)/(2+0.5)) = ln(1 + 2.5/2.5) = ln(2) = 0.69314718...
    #[test]
    fn matches_hand_computed_value_tuple_two() {
        let score = bm25_score(1.0, 2.0, 5.0, 4.0, 5.0);
        assert!(
            (score - std::f64::consts::LN_2).abs() < 1e-9,
            "score {score} != ln(2)"
        );
    }

    #[test]
    fn higher_term_frequency_scores_higher_all_else_equal() {
        let low_tf = bm25_score(1.0, 2.0, 5.0, 4.0, 5.0);
        let high_tf = bm25_score(3.0, 2.0, 5.0, 4.0, 5.0);
        assert!(high_tf > low_tf);
    }

    #[test]
    fn bm25_score_delegates_to_bm25_score_b_at_the_default_b_byte_identical() {
        let tuples: &[(f64, f64, f64, f64, f64)] = &[
            (2.0, 1.0, 10.0, 3.0, 8.0),
            (1.0, 2.0, 5.0, 4.0, 5.0),
            (3.0, 2.0, 5.0, 4.0, 5.0),
            (5.0, 7.0, 40.0, 100.0, 12.5),
        ];
        for &(tf, df, length, n_docs, avg_len) in tuples {
            let via_default = bm25_score(tf, df, length, n_docs, avg_len);
            let via_b = bm25_score_b(tf, df, length, n_docs, avg_len, BM25_B);
            assert_eq!(
                via_default.to_bits(),
                via_b.to_bits(),
                "bm25_score must be byte-identical to bm25_score_b at the default b"
            );
        }
    }

    #[test]
    fn b_zero_drops_the_length_normalization_term() {
        let tf = 3.0;
        let df = 2.0;
        let n_docs = 10.0;
        let avg_len = 20.0;

        let short_doc = bm25_score_b(tf, df, 4.0, n_docs, avg_len, SYMBOL_BM25_B);
        let long_doc = bm25_score_b(tf, df, 400.0, n_docs, avg_len, SYMBOL_BM25_B);
        assert_eq!(
            short_doc.to_bits(),
            long_doc.to_bits(),
            "at b=0 document length must not affect the score"
        );

        // denom reduces to tf + k1 at b=0 -- matches the closed form directly.
        let idf = (1.0 + (n_docs - df + 0.5) / (df + 0.5)).ln();
        let expected = idf * tf * (BM25_K1 + 1.0) / (tf + BM25_K1);
        assert!((short_doc - expected).abs() < 1e-12);
    }

    #[test]
    fn nonzero_b_still_penalizes_a_longer_document() {
        let tf = 3.0;
        let df = 2.0;
        let n_docs = 10.0;
        let avg_len = 20.0;

        let short_doc = bm25_score_b(tf, df, 4.0, n_docs, avg_len, BM25_B);
        let long_doc = bm25_score_b(tf, df, 400.0, n_docs, avg_len, BM25_B);
        assert!(
            short_doc > long_doc,
            "at the default b a longer document must score lower for the same tf"
        );
    }

    #[test]
    fn symbol_bm25_score_global_reads_a_separate_term_stats_instance() {
        use crate::postings::rebuild_postings_csr;
        use crate::tokenize::{feature_id, tokenize};
        use std::collections::HashMap;

        let csr_from_docs = |docs: &[&str]| -> crate::postings::PostingsCsr {
            let freqs: Vec<HashMap<u32, u32>> = docs
                .iter()
                .map(|text| {
                    let mut freq: HashMap<u32, u32> = HashMap::new();
                    for tok in tokenize(text) {
                        *freq.entry(feature_id(&tok)).or_insert(0) += 1;
                    }
                    freq
                })
                .collect();
            rebuild_postings_csr(docs.len(), &freqs)
        };

        let mut whole_file_stats = TermStats::new();
        whole_file_stats.add_segment(&csr_from_docs(&["falcon soars", "falcon dives"]));

        let mut symbol_stats = TermStats::new();
        symbol_stats.add_segment(&csr_from_docs(&["falcon"]));

        let fid = feature_id("falcon");
        let global_score = bm25_score_global(1.0, fid, 1.0, &whole_file_stats);
        let symbol_score = symbol_bm25_score_global(1.0, fid, 1.0, &symbol_stats);
        assert_ne!(
            global_score, symbol_score,
            "the symbol-scoped scorer must read its own TermStats instance, distinct from the whole-file lane's"
        );
    }

    #[test]
    fn term_stats_backed_score_uses_global_df_and_n_docs_not_local_segment_values() {
        use crate::postings::rebuild_postings_csr;
        use crate::tokenize::{feature_id, tokenize};
        use std::collections::HashMap;

        let csr_from_docs = |docs: &[&str]| -> crate::postings::PostingsCsr {
            let freqs: Vec<HashMap<u32, u32>> = docs
                .iter()
                .map(|text| {
                    let mut freq: HashMap<u32, u32> = HashMap::new();
                    for tok in tokenize(text) {
                        *freq.entry(feature_id(&tok)).or_insert(0) += 1;
                    }
                    freq
                })
                .collect();
            rebuild_postings_csr(docs.len(), &freqs)
        };

        // seg_a alone: local df(falcon) = 1, n_docs = 1.
        let seg_a = csr_from_docs(&["falcon soars"]);
        // seg_b: local df(falcon) = 2 -- global df across both segments = 3.
        let seg_b = csr_from_docs(&["falcon dives", "falcon glides"]);

        let mut stats = TermStats::new();
        stats.add_segment(&seg_a);
        stats.add_segment(&seg_b);

        let fid = feature_id("falcon");
        let local_score = bm25_score(1.0, 1.0, 2.0, seg_a.n_docs as f64, seg_a.avg_len);
        let global_score = bm25_score_global(1.0, fid, 2.0, &stats);

        assert_eq!(stats.df(fid), 3, "global df is the sum across both segments");
        assert_ne!(
            global_score, local_score,
            "a score computed through the global table must differ from one computed against segment-local df/n_docs alone"
        );
    }
}

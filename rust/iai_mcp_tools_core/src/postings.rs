//! CSR postings builder for whole-file BM25 -- feature-id keyed, two-pass.

use std::collections::HashMap;

pub struct PostingsCsr {
    /// feature_id -> compact CSR index -- never a token string.
    pub token_lookup: HashMap<u32, u32>,
    pub post_offsets: Vec<u32>,
    pub post_doc_slots: Vec<u32>,
    pub post_tfs: Vec<u32>,
    pub doc_len: Vec<u32>,
    pub n_docs: u32,
    pub avg_len: f64,
    pub total_len: u64,
}

/// Two-pass CSR builder: assign compact ids + count doc-frequency per
/// token, then fill `post_doc_slots`/`post_tfs` via a cursor array. Doc
/// order is preserved by the outer `for slot in 0..n` loop, so no sort is
/// needed.
pub fn rebuild_postings_csr(n: usize, token_freqs: &[HashMap<u32, u32>]) -> PostingsCsr {
    let mut token_lookup: HashMap<u32, u32> = HashMap::new();
    let mut token_doc_count: Vec<u32> = Vec::new();
    let mut raw_postings: Vec<(u32, u32, u32)> = Vec::new(); // (compact_index, slot, tf)
    let mut doc_len = Vec::with_capacity(n);
    let mut total_len: u64 = 0;

    for slot in 0..n {
        let local_freq = &token_freqs[slot];
        let slot_len: u32 = local_freq.values().sum();
        doc_len.push(slot_len);
        total_len += slot_len as u64;

        for (&feature_id, &tf) in local_freq {
            let compact_id = match token_lookup.get(&feature_id) {
                Some(&id) => id,
                None => {
                    let id = token_lookup.len() as u32;
                    token_lookup.insert(feature_id, id);
                    id
                }
            };
            if compact_id as usize == token_doc_count.len() {
                token_doc_count.push(0);
            }
            token_doc_count[compact_id as usize] += 1;
            raw_postings.push((compact_id, slot as u32, tf));
        }
    }

    let ntokens = token_lookup.len();
    let mut post_offsets = Vec::with_capacity(ntokens + 1);
    post_offsets.push(0u32);
    for &c in &token_doc_count {
        post_offsets.push(post_offsets.last().unwrap() + c);
    }
    let total_postings = raw_postings.len();
    let mut post_doc_slots = vec![0u32; total_postings];
    let mut post_tfs = vec![0u32; total_postings];
    let mut cursor: Vec<u32> = post_offsets[..ntokens].to_vec();
    for (token_id, slot, tf) in raw_postings {
        let pos = cursor[token_id as usize] as usize;
        post_doc_slots[pos] = slot;
        post_tfs[pos] = tf;
        cursor[token_id as usize] += 1;
    }

    let n_docs = n as u32;
    let avg_len = if n_docs == 0 {
        1.0
    } else {
        (total_len as f64 / n_docs as f64).max(1.0)
    };

    PostingsCsr {
        token_lookup,
        post_offsets,
        post_doc_slots,
        post_tfs,
        doc_len,
        n_docs,
        avg_len,
        total_len,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bm25::bm25_score;
    use crate::tokenize::{feature_id, tokenize};

    /// Fixed 3-doc corpus, tokenized with the real tokenizer so feature ids
    /// match production, then compared against a hand-written expected
    /// table:
    ///   doc0 "cat dog"      -> cat:1, dog:1        (len 2)
    ///   doc1 "dog bird"     -> dog:1, bird:1       (len 2)
    ///   doc2 "cat cat bird" -> cat:2, bird:1        (len 3)
    fn fixed_corpus_csr() -> PostingsCsr {
        let docs = ["cat dog", "dog bird", "cat cat bird"];
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
        rebuild_postings_csr(3, &freqs)
    }

    #[test]
    fn csr_matches_hand_written_table() {
        let csr = fixed_corpus_csr();

        assert_eq!(csr.n_docs, 3);
        assert_eq!(csr.doc_len, vec![2, 2, 3]);
        assert_eq!(csr.total_len, 7);
        assert!((csr.avg_len - 7.0 / 3.0).abs() < 1e-12);

        let postings_for = |token: &str| -> (Vec<u32>, Vec<u32>) {
            let fid = feature_id(token);
            let cid = csr.token_lookup[&fid] as usize;
            let start = csr.post_offsets[cid] as usize;
            let end = csr.post_offsets[cid + 1] as usize;
            (
                csr.post_doc_slots[start..end].to_vec(),
                csr.post_tfs[start..end].to_vec(),
            )
        };

        // "cat": doc0 (tf=1), doc2 (tf=2) -- ascending slot order preserved.
        let (slots, tfs) = postings_for("cat");
        assert_eq!(slots, vec![0, 2]);
        assert_eq!(tfs, vec![1, 2]);

        // "dog": doc0 (tf=1), doc1 (tf=1).
        let (slots, tfs) = postings_for("dog");
        assert_eq!(slots, vec![0, 1]);
        assert_eq!(tfs, vec![1, 1]);

        // "bird": doc1 (tf=1), doc2 (tf=1).
        let (slots, tfs) = postings_for("bird");
        assert_eq!(slots, vec![1, 2]);
        assert_eq!(tfs, vec![1, 1]);
    }

    #[test]
    fn rebuild_postings_csr_counts_document_frequency() {
        let mut doc_a = HashMap::new();
        doc_a.insert(feature_id("zebra"), 2u32);
        let mut doc_b = HashMap::new();
        doc_b.insert(feature_id("zebra"), 1u32);
        doc_b.insert(feature_id("giraffe"), 3u32);
        let csr = rebuild_postings_csr(2, &[doc_a, doc_b]);
        let zebra_id = csr.token_lookup[&feature_id("zebra")];
        let start = csr.post_offsets[zebra_id as usize] as usize;
        let end = csr.post_offsets[zebra_id as usize + 1] as usize;
        assert_eq!(end - start, 2, "zebra appears in both docs");
    }

    /// Single-token query ("cat") over the fixed corpus ranks documents by
    /// descending BM25 score: doc2 (tf=2, longer doc) outscores doc0
    /// (tf=1), both computed through the real CSR + BM25 path.
    #[test]
    fn single_token_query_ranks_by_descending_bm25() {
        let csr = fixed_corpus_csr();
        let fid = feature_id("cat");
        let cid = csr.token_lookup[&fid] as usize;
        let start = csr.post_offsets[cid] as usize;
        let end = csr.post_offsets[cid + 1] as usize;
        let df = (end - start) as f64;
        let n_docs = csr.n_docs as f64;
        let avg_len = csr.avg_len;

        let mut scored: Vec<(u32, f64)> = (start..end)
            .map(|i| {
                let slot = csr.post_doc_slots[i];
                let tf = csr.post_tfs[i] as f64;
                let length = csr.doc_len[slot as usize] as f64;
                (slot, bm25_score(tf, df, length, n_docs, avg_len))
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        assert_eq!(scored[0].0, 2, "doc2 (tf=2) must outrank doc0 (tf=1)");
        assert_eq!(scored[1].0, 0);
        assert!(scored[0].1 > scored[1].1);
    }
}

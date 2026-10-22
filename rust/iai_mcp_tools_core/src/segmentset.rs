//! Generation-tagged double-buffer over the active segment set --
//! re-derives `iai_mcp_rank_core::DoubleBuffer`'s lock order by copy,
//! never by import: a `rebuild` mutex guards the whole clone -> mutate ->
//! commit body; a snapshot is a pure Arc clone taken under `active`'s
//! lock only, so readers never take `rebuild`.
//!
//! `segments` holds lightweight handles (id + on-disk file name), never
//! the loaded postings -- an append or a compaction swap only ever needs
//! to add/remove handles and fold/carry the resident term-stats, never to
//! open a prior segment's file to do so.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::postings::PostingsCsr;
use crate::termstats::TermStats;

/// One tombstoned (segment_id, file_id) pair. The writes that populate
/// this set arrive in a later plan; the field and its atomic pairing with
/// `term_stats` are defined now so the no-torn-pair guarantee is provable
/// today.
pub type Tombstone = (u32, u32);

/// A lightweight reference to an on-disk segment -- an id (unique among
/// currently active segments) and its file name under the store root.
/// Never carries the loaded `PostingsCsr`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentHandle {
    pub id: u32,
    pub file_name: String,
}

/// A published, immutable view. `segments`, `tombstones`, and
/// `term_stats` swap together as ONE tuple -- a reader never observes a
/// torn pairing across the three, at any point in the snapshot's
/// lifetime.
pub struct SegmentSet {
    pub segments: Vec<SegmentHandle>,
    pub tombstones: Arc<HashSet<Tombstone>>,
    pub term_stats: Arc<TermStats>,
    /// path -> (segment_id, file_id, content_hash), resident so an append
    /// can look up a path's prior content-hash without opening a segment.
    pub file_index: Arc<HashMap<String, (u32, u32, u64)>>,
    pub generation: u64,
}

impl SegmentSet {
    pub(crate) fn empty() -> Self {
        SegmentSet {
            segments: Vec::new(),
            tombstones: Arc::new(HashSet::new()),
            term_stats: Arc::new(TermStats::new()),
            file_index: Arc::new(HashMap::new()),
            generation: 0,
        }
    }
}

/// Active/publish double-buffer over `SegmentSet`.
pub struct SegmentSetBuffer {
    active: Mutex<Arc<SegmentSet>>,
    // Serializes the whole clone -> mutate -> commit body of `publish`
    // and `compact`. Readers (`snapshot`) never take this lock -- they
    // only clone the `Arc` under `active`'s lock, held solely for that
    // O(1) pointer read.
    rebuild: Mutex<()>,
}

impl SegmentSetBuffer {
    pub fn new() -> Self {
        SegmentSetBuffer {
            active: Mutex::new(Arc::new(SegmentSet::empty())),
            rebuild: Mutex::new(()),
        }
    }

    /// Seeds the buffer with an already-known `SegmentSet` (e.g. loaded
    /// from the on-disk manifest) rather than starting empty.
    pub fn from_set(set: SegmentSet) -> Self {
        SegmentSetBuffer {
            active: Mutex::new(Arc::new(set)),
            rebuild: Mutex::new(()),
        }
    }

    /// A pure read: the published `Arc` is cloned (O(1)) under `active`'s
    /// lock only. A concurrent `publish` either has not yet swapped
    /// `active` (this call observes the prior tuple, fully consistent) or
    /// already has (this call observes the new tuple, also fully
    /// consistent) -- there is no window in which `segments`,
    /// `tombstones`, and `term_stats` are drawn from different
    /// generations.
    pub fn snapshot(&self) -> Arc<SegmentSet> {
        self.active.lock().clone()
    }

    /// Appends one new segment: clones the current set, folds the new
    /// segment's df into a cloned term-stat table, bumps `generation`,
    /// and commits with a single Arc pointer-swap under `active`'s lock.
    /// Guarded by `rebuild` so two concurrent publishes never race the
    /// clone body against each other. `postings` is only borrowed to
    /// fold term-stats -- the handle, not the CSR, is what gets stored.
    pub fn publish(&self, handle: SegmentHandle, postings: &PostingsCsr) {
        let _rebuild_guard = self.rebuild.lock();
        let current = self.active.lock().clone();

        let mut next_stats = (*current.term_stats).clone();
        next_stats.add_segment(postings);

        let mut segments = current.segments.clone();
        segments.push(handle);

        let next = Arc::new(SegmentSet {
            segments,
            tombstones: Arc::clone(&current.tombstones),
            term_stats: Arc::new(next_stats),
            file_index: Arc::clone(&current.file_index),
            generation: current.generation + 1,
        });

        *self.active.lock() = next;
    }

    /// Appends one new segment carrying its own path->(segment_id,file_id,
    /// content_hash) entries plus tombstones for the paths it replaces --
    /// segments+tombstones+term_stats+file_index swap as ONE atomic tuple,
    /// mirroring `publish`'s no-torn-pair guarantee.
    pub fn publish_incremental(
        &self,
        handle: SegmentHandle,
        postings: &PostingsCsr,
        file_entries: &[(String, u32, u64)],
        new_tombstones: &[Tombstone],
    ) {
        let _rebuild_guard = self.rebuild.lock();
        let current = self.active.lock().clone();

        let mut next_stats = (*current.term_stats).clone();
        next_stats.add_segment(postings);

        let mut segments = current.segments.clone();
        segments.push(handle.clone());

        let mut file_index = (*current.file_index).clone();
        for (path, file_id, content_hash) in file_entries {
            file_index.insert(path.clone(), (handle.id, *file_id, *content_hash));
        }

        let mut tombstones = (*current.tombstones).clone();
        for t in new_tombstones {
            tombstones.insert(*t);
        }

        let next = Arc::new(SegmentSet {
            segments,
            tombstones: Arc::new(tombstones),
            term_stats: Arc::new(next_stats),
            file_index: Arc::new(file_index),
            generation: current.generation + 1,
        });

        *self.active.lock() = next;
    }

    /// Compaction swap: publishes `merged` while retiring every handle
    /// whose id is in `source_ids`, in ONE swap (merged-in, sources-out
    /// together -- never two separate steps, or a reader in between
    /// double-counts or loses files). `term_stats` is carried forward
    /// UNCHANGED -- a merge re-packages already-counted content, it must
    /// never be treated as a new-content publish.
    pub fn compact(&self, source_ids: &HashSet<u32>, merged: SegmentHandle) {
        let _rebuild_guard = self.rebuild.lock();
        let current = self.active.lock().clone();

        let mut segments: Vec<SegmentHandle> = current
            .segments
            .iter()
            .filter(|h| !source_ids.contains(&h.id))
            .cloned()
            .collect();
        segments.push(merged);

        let next = Arc::new(SegmentSet {
            segments,
            tombstones: Arc::clone(&current.tombstones),
            term_stats: Arc::clone(&current.term_stats),
            file_index: Arc::clone(&current.file_index),
            generation: current.generation + 1,
        });

        *self.active.lock() = next;
    }

    /// Compaction swap with reconciliation: retires every handle in
    /// `source_ids`, installs `merged`, and REPLACES (not folds) term_stats
    /// and file_index wholesale with the caller's reconciled values -- a
    /// merge that also dedupes/tombstone-drops must not carry forward stale
    /// per-file bookkeeping. A tombstone whose segment_id is being retired
    /// is cleared; any tombstone against a segment outside this batch
    /// survives.
    pub fn compact_reconciled(
        &self,
        source_ids: &HashSet<u32>,
        merged: SegmentHandle,
        reconciled_stats: TermStats,
        rebuilt_file_index: HashMap<String, (u32, u32, u64)>,
    ) {
        let _rebuild_guard = self.rebuild.lock();
        let current = self.active.lock().clone();

        let mut segments: Vec<SegmentHandle> = current
            .segments
            .iter()
            .filter(|h| !source_ids.contains(&h.id))
            .cloned()
            .collect();
        segments.push(merged);

        let tombstones: HashSet<Tombstone> = current
            .tombstones
            .iter()
            .filter(|(seg_id, _)| !source_ids.contains(seg_id))
            .cloned()
            .collect();

        let next = Arc::new(SegmentSet {
            segments,
            tombstones: Arc::new(tombstones),
            term_stats: Arc::new(reconciled_stats),
            file_index: Arc::new(rebuilt_file_index),
            generation: current.generation + 1,
        });

        *self.active.lock() = next;
    }
}

impl Default for SegmentSetBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postings::rebuild_postings_csr;
    use crate::tokenize::{feature_id, tokenize};
    use std::collections::HashMap;
    use std::sync::Barrier;
    use std::thread;
    use std::time::Duration;

    fn csr_from_docs(docs: &[&str]) -> PostingsCsr {
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
    }

    fn handle(id: u32) -> SegmentHandle {
        SegmentHandle {
            id,
            file_name: format!("segment-{id}.lcidx"),
        }
    }

    #[test]
    fn double_buffer_generation_increases_monotonically_on_publish() {
        let buf = SegmentSetBuffer::new();
        assert_eq!(buf.snapshot().generation, 0);
        buf.publish(handle(1), &csr_from_docs(&["alpha beta"]));
        assert_eq!(buf.snapshot().generation, 1);
        buf.publish(handle(2), &csr_from_docs(&["gamma delta"]));
        assert_eq!(buf.snapshot().generation, 2);
    }

    #[test]
    fn double_buffer_snapshot_taken_before_publish_never_observes_new_segment() {
        let buf = SegmentSetBuffer::new();
        buf.publish(handle(1), &csr_from_docs(&["alpha beta"]));

        let snap = buf.snapshot();
        assert_eq!(snap.segments.len(), 1);

        buf.publish(handle(2), &csr_from_docs(&["gamma delta"]));

        // The held snapshot is an Arc clone taken before the second
        // publish -- its own segment count and generation must be frozen.
        assert_eq!(snap.segments.len(), 1);
        assert_eq!(snap.generation, 1);

        let fresh = buf.snapshot();
        assert_eq!(fresh.segments.len(), 2);
        assert_eq!(fresh.generation, 2);
    }

    #[test]
    fn double_buffer_snapshot_exposes_matching_segments_tombstones_term_stats_tuple() {
        let buf = SegmentSetBuffer::new();
        let seg = csr_from_docs(&["zebra hides", "zebra eats"]);
        let seg_n_docs = seg.n_docs;
        buf.publish(handle(1), &seg);

        let snap = buf.snapshot();
        // No torn pair: term_stats.n_docs must equal the one published
        // segment's own n_docs, since exactly one segment has ever
        // existed.
        assert_eq!(snap.term_stats.n_docs(), seg_n_docs);
        assert_eq!(snap.term_stats.df(feature_id("zebra")), 2);
        assert!(snap.tombstones.is_empty());
    }

    #[test]
    fn double_buffer_concurrent_reader_holds_pre_publish_tuple_for_its_whole_lifetime() {
        let buf = Arc::new(SegmentSetBuffer::new());
        buf.publish(handle(1), &csr_from_docs(&["falcon soars"]));

        let start = Arc::new(Barrier::new(2));

        let reader_buf = Arc::clone(&buf);
        let reader_start = Arc::clone(&start);
        let reader = thread::spawn(move || {
            let snap = reader_buf.snapshot();
            reader_start.wait();
            // Give the publisher a real window to swap `active` while
            // this thread still holds its own snapshot.
            thread::sleep(Duration::from_millis(30));
            assert_eq!(snap.segments.len(), 1);
            assert_eq!(snap.generation, 1);
            assert_eq!(snap.term_stats.df(feature_id("falcon")), 1);
            assert!(snap.tombstones.is_empty());
        });

        start.wait();
        buf.publish(handle(2), &csr_from_docs(&["falcon dives", "falcon glides"]));
        reader.join().unwrap();

        let latest = buf.snapshot();
        assert_eq!(latest.generation, 2);
        assert_eq!(latest.segments.len(), 2);
        assert_eq!(latest.term_stats.df(feature_id("falcon")), 3);
    }

    #[test]
    fn compaction_snapshot_held_across_swap_sees_unchanged_tuple() {
        let buf = SegmentSetBuffer::new();
        buf.publish(handle(1), &csr_from_docs(&["alpha token_a"]));
        buf.publish(handle(2), &csr_from_docs(&["beta token_b"]));

        let snap = buf.snapshot();
        assert_eq!(snap.segments.len(), 2);
        assert_eq!(snap.generation, 2);
        let n_docs_before = snap.term_stats.n_docs();
        let alpha_df_before = snap.term_stats.df(feature_id("alpha"));

        let mut source_ids = HashSet::new();
        source_ids.insert(1u32);
        source_ids.insert(2u32);
        buf.compact(&source_ids, handle(3));

        // A snapshot taken BEFORE the compaction swap must be UNCHANGED:
        // still 2 pre-merge segments, generation 2, untouched term_stats
        // -- no torn pair, no double-count, nothing missing for this
        // reader's whole lifetime.
        assert_eq!(snap.segments.len(), 2);
        assert_eq!(snap.generation, 2);
        assert_eq!(snap.term_stats.n_docs(), n_docs_before);

        let fresh = buf.snapshot();
        assert_eq!(fresh.segments.len(), 1);
        assert_eq!(fresh.segments[0].id, 3);
        assert_eq!(fresh.generation, 3);
        // term_stats carried forward UNCHANGED across a merge -- re-
        // packaging already-counted content must never double-count.
        assert_eq!(fresh.term_stats.n_docs(), n_docs_before);
        assert_eq!(fresh.term_stats.df(feature_id("alpha")), alpha_df_before);
    }
}

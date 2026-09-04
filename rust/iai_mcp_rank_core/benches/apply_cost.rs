//! Read-path apply timing at production-representative volume. Builds a
//! ~20k-record columnar index, feeds a write burst of pending upserts and
//! deletes, then times one stale `DoubleBuffer::snapshot` -- the
//! INCREMENTAL apply: copy columns off the published `Inner`, apply
//! pending ops in place (scalar/vector columns updated directly,
//! adjacency/tags/postings deltas recorded in the bounded overlay), never
//! a wholesale CSR rebuild. Also times `DoubleBuffer::fold`, the separate
//! off-critical-path operation that still performs that wholesale rebuild,
//! for direct contrast against the incremental numbers above it.
//! Disposable fixture harness, no production surface: `harness = false`,
//! run directly by `cargo bench`.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use iai_mcp_rank_core::{tokenize, DoubleBuffer, Inner, PendingOp};

const DIM: usize = 384;
const N_RECORDS: usize = 20_000;
const EDGES_PER_RECORD: usize = 3; // ~60k adjacency entries at 20k records
const WORDS_PER_DOC: usize = 20; // ~387k posting entries at 20k records
const VOCAB_SIZE: u32 = 500;
const BURST_UPSERTS: usize = 1_000;
const BURST_DELETES: usize = 200;
const WARM_ITERS: usize = 5;
// Regression fence, not a latency budget: DoubleBuffer::snapshot's stale
// path applies incrementally now -- cost tracks overlay (touched-id) size,
// not corpus size. Set with headroom above the incremental design's
// expected cost so this stays a feasibility gate against a real
// regression (e.g. a change that reintroduces a wholesale rebuild on this
// path), not a flaky wall-clock lock.
const CEILING_MS: f64 = 500.0;
// A read-path apply this far under CEILING_MS is evidence it took the
// incremental branch, not the ~160-270 ms wholesale-rebuild class this
// same fixture measured before the overlay existed -- a much tighter
// bound than the regression fence above, calibrated with headroom over
// the incremental design's expected cost, not the old wholesale one.
const INCREMENTAL_APPLY_CEILING_MS: f64 = 50.0;

/// Deterministic xorshift32 -- a disposable fixture generator has no
/// business pulling in an RNG crate dependency.
fn xorshift32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

fn synthetic_vector(rng: &mut u32) -> Vec<f32> {
    (0..DIM)
        .map(|_| (xorshift32(rng) as f32 / u32::MAX as f32) - 0.5)
        .collect()
}

fn synthetic_surface(doc: usize, rng: &mut u32) -> String {
    let mut words: Vec<String> = (0..WORDS_PER_DOC)
        .map(|_| format!("token{}", xorshift32(rng) % VOCAB_SIZE))
        .collect();
    words.push(format!("doc{doc}"));
    words.join(" ")
}

fn synthetic_edges(node: usize, rng: &mut u32) -> Vec<(u128, f32, String)> {
    (1..=EDGES_PER_RECORD)
        .map(|k| {
            let neighbor = ((node + k) % N_RECORDS) as u128 + 1;
            let weight = 0.5 + (xorshift32(rng) as f32 / u32::MAX as f32) * 0.5;
            (neighbor, weight, "hebbian".to_string())
        })
        .collect()
}

fn count_postings(surfaces: &[String]) -> usize {
    surfaces
        .iter()
        .map(|s| tokenize(s).into_iter().collect::<HashSet<_>>().len())
        .sum()
}

fn build_fixture() -> (Inner, usize) {
    let mut rng: u32 = 0x9E37_79B9;
    let mut ids = Vec::with_capacity(N_RECORDS);
    let mut vectors_flat = Vec::with_capacity(N_RECORDS * DIM);
    let mut edge_map: HashMap<u128, Vec<(u128, f32, String)>> = HashMap::with_capacity(N_RECORDS);
    let mut surfaces = Vec::with_capacity(N_RECORDS);
    let aaak_index = vec![String::new(); N_RECORDS];
    let created_at = vec![0i64; N_RECORDS];
    let stability = vec![0.5f32; N_RECORDS];
    let tier = vec!["episodic".to_string(); N_RECORDS];
    let tags = vec![Vec::new(); N_RECORDS];
    let salience_level = vec![0u8; N_RECORDS];
    let centrality = vec![0.0f32; N_RECORDS];

    for i in 0..N_RECORDS {
        let id = i as u128 + 1;
        ids.push(id);
        vectors_flat.extend(synthetic_vector(&mut rng));
        surfaces.push(synthetic_surface(i, &mut rng));
        edge_map.insert(id, synthetic_edges(i, &mut rng));
    }

    let posting_count = count_postings(&surfaces);
    let adjacency_count: usize = edge_map.values().map(Vec::len).sum();
    eprintln!(
        "apply_cost: fixture n={N_RECORDS} postings={posting_count} adjacency={adjacency_count}"
    );

    let pending = vec![false; N_RECORDS];
    let inner = Inner::from_columns(
        DIM, 1, ids, vectors_flat, edge_map, surfaces, aaak_index,
        created_at, stability, tier, tags, salience_level, centrality, pending,
    )
    .expect("fixture build must not fail");
    (inner, adjacency_count)
}

// Layer-1 candidate-set scale: a per-call cold `Inner::from_columns` build
// over the ANN/hop-hydrated candidate window (a few hundred to a few
// thousand rows), distinct from the corpus-wide 14-20k scale measured
// above -- feasibility input for a per-call rebuild design.
const LAYER1_SIZES: [usize; 3] = [200, 1_000, 3_600];
const LAYER1_WARM_ITERS: usize = 9;

type LayerColumns = (
    usize, u64, Vec<u128>, Vec<f32>, HashMap<u128, Vec<(u128, f32, String)>>,
    Vec<String>, Vec<String>, Vec<i64>, Vec<f32>, Vec<String>, Vec<Vec<String>>,
    Vec<u8>, Vec<f32>, Vec<bool>,
);

fn synthetic_edges_at(node: usize, n: usize, rng: &mut u32) -> Vec<(u128, f32, String)> {
    (1..=EDGES_PER_RECORD)
        .map(|k| {
            let neighbor = ((node + k) % n) as u128 + 1;
            let weight = 0.5 + (xorshift32(rng) as f32 / u32::MAX as f32) * 0.5;
            (neighbor, weight, "hebbian".to_string())
        })
        .collect()
}

fn build_columns_at_scale(n: usize) -> LayerColumns {
    let mut rng: u32 = 0x1234_5678;
    let mut ids = Vec::with_capacity(n);
    let mut vectors_flat = Vec::with_capacity(n * DIM);
    let mut edge_map: HashMap<u128, Vec<(u128, f32, String)>> = HashMap::with_capacity(n);
    let mut surfaces = Vec::with_capacity(n);
    for i in 0..n {
        let id = i as u128 + 1;
        ids.push(id);
        vectors_flat.extend(synthetic_vector(&mut rng));
        surfaces.push(synthetic_surface(i, &mut rng));
        edge_map.insert(id, synthetic_edges_at(i, n, &mut rng));
    }
    let aaak_index = vec![String::new(); n];
    let created_at = vec![0i64; n];
    let stability = vec![0.5f32; n];
    let tier = vec!["episodic".to_string(); n];
    let tags = vec![Vec::new(); n];
    let salience_level = vec![0u8; n];
    let centrality = vec![0.0f32; n];
    let pending = vec![false; n];
    (
        DIM, 1, ids, vectors_flat, edge_map, surfaces, aaak_index,
        created_at, stability, tier, tags, salience_level, centrality, pending,
    )
}

fn bench_from_columns_layer1() {
    for &n in LAYER1_SIZES.iter() {
        let (dim, generation, ids, vectors_flat, edge_map, surfaces, aaak_index,
             created_at, stability, tier, tags, salience_level, centrality, pending) =
            build_columns_at_scale(n);

        let mut samples_ms = Vec::with_capacity(LAYER1_WARM_ITERS);
        for _ in 0..LAYER1_WARM_ITERS {
            let start = Instant::now();
            let inner = Inner::from_columns(
                dim, generation, ids.clone(), vectors_flat.clone(), edge_map.clone(),
                surfaces.clone(), aaak_index.clone(), created_at.clone(), stability.clone(),
                tier.clone(), tags.clone(), salience_level.clone(), centrality.clone(),
                pending.clone(),
            )
            .expect("layer1 fixture build must not fail");
            samples_ms.push(start.elapsed().as_secs_f64() * 1000.0);
            std::hint::black_box(inner);
        }
        report(&format!("from_columns_layer1 n={n}"), &samples_ms);
    }
}

fn build_burst(rng: &mut u32) -> Vec<PendingOp> {
    let mut ops = Vec::with_capacity(BURST_UPSERTS + BURST_DELETES);
    for i in 0..BURST_UPSERTS {
        let id = (i % N_RECORDS) as u128 + 1;
        ops.push(PendingOp::Upsert {
            id,
            vector: synthetic_vector(rng),
            edges: synthetic_edges(i, rng),
            surface: synthetic_surface(i, rng),
            aaak_index: String::new(),
            created_at: 0,
            stability: 0.5,
            tier: "episodic".to_string(),
            tags: Vec::new(),
            salience_level: 0,
            centrality: 0.0,
            pending: false,
        });
    }
    for i in 0..BURST_DELETES {
        let id = ((BURST_UPSERTS + i) % N_RECORDS) as u128 + 1;
        ops.push(PendingOp::Delete { id });
    }
    ops
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn report(label: &str, samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    let med = median(&mut sorted);
    let min = sorted.first().copied().unwrap_or(0.0);
    let max = sorted.last().copied().unwrap_or(0.0);
    println!(
        "apply_cost: {label}: min={min:.3} ms median={med:.3} ms max={max:.3} ms (n={})",
        samples.len(),
    );
    med
}

fn main() {
    bench_from_columns_layer1();

    let (inner, _adjacency_count) = build_fixture();
    let buf = DoubleBuffer::new(inner);
    let mut rng: u32 = 0xC0FF_EE11;
    let mut gen = 2u64;

    // With-burst: re-feed a fresh pending burst before each generation
    // bump -- times the INCREMENTAL apply (scalar/vector columns updated
    // in place, adjacency/tags/postings deltas recorded in the overlay),
    // never a wholesale CSR rebuild.
    let mut with_burst_ms = Vec::with_capacity(WARM_ITERS);
    for _ in 0..WARM_ITERS {
        for op in build_burst(&mut rng) {
            buf.feed(op).expect("feed must not fail during burst");
        }
        let start = Instant::now();
        let applied = buf.snapshot(gen).expect("incremental apply must not fail");
        with_burst_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(applied);
        gen += 1;
    }

    // Zero-pending: bump the generation with an EMPTY pending queue,
    // isolating the base per-call apply cost (clone + zero ops) from the
    // burst-apply cost above.
    let mut zero_pending_ms = Vec::with_capacity(WARM_ITERS);
    for _ in 0..WARM_ITERS {
        let start = Instant::now();
        let applied = buf.snapshot(gen).expect("apply must not fail");
        zero_pending_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(applied);
        gen += 1;
    }

    let with_burst_median = report("with-burst (incremental apply, feed+apply)", &with_burst_ms);
    let zero_pending_median = report("zero-pending (apply only, no touched ids)", &zero_pending_ms);
    println!(
        "apply_cost: burst-apply marginal cost over zero-pending: {:.3} ms",
        with_burst_median - zero_pending_median,
    );

    assert!(
        with_burst_median < CEILING_MS,
        "incremental apply (with-burst) median {with_burst_median:.3} ms, expected under the \
         {CEILING_MS} ms regression fence -- a write burst must never stall a reader past the \
         double-buffer's guarantees",
    );
    assert!(
        with_burst_median < INCREMENTAL_APPLY_CEILING_MS,
        "incremental apply (with-burst) median {with_burst_median:.3} ms, expected under the \
         {INCREMENTAL_APPLY_CEILING_MS} ms incremental-design ceiling -- this far under the \
         {CEILING_MS} ms wholesale-rebuild fence is the evidence the read path is NOT \
         re-tokenizing the whole corpus on this call",
    );

    // fold cost: the off-critical-path operation that STILL performs the
    // wholesale postings/adjacency/tags CSR rebuild, timed for direct
    // contrast against the incremental apply above -- this must stay far
    // more expensive than the incremental numbers, since it is the same
    // O(corpus) work that used to run inline on the recall path.
    let mut fold_ms = Vec::with_capacity(WARM_ITERS);
    for _ in 0..WARM_ITERS {
        for op in build_burst(&mut rng) {
            buf.feed(op).expect("feed must not fail during burst");
        }
        buf.snapshot(gen).expect("incremental apply must not fail");
        gen += 1;
        let start = Instant::now();
        let folded = buf.fold();
        fold_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(folded);
    }
    let fold_median = report("fold (off-path wholesale CSR rebuild, for contrast)", &fold_ms);
    println!(
        "apply_cost: fold / incremental-apply ratio: {:.1}x (fold={:.3} ms, incremental with-burst={:.3} ms)",
        fold_median / with_burst_median.max(1e-6),
        fold_median,
        with_burst_median,
    );
}

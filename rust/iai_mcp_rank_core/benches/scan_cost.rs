//! Full-term-set fused-scan timing over the columnar `Inner` at a
//! production-shaped corpus. Disposable fixture harness, no production
//! surface: `harness = false`, run directly by `cargo bench`. Walks every
//! resident slot computing cosine, aaak overlap, degree norm, age penalty,
//! and HD-hypervector Hamming, then runs a sparse BM25 posting-lookup pass
//! over a fixed query token set. `structure_hv` is synthesized locally at
//! its BSC-episodic width (D=4096, 512 packed bytes, confirmed from
//! `src/iai_mcp/types.py` / `iai_mcp.lilli.tiers.bsc`) — the columnar
//! `Inner` carries no such column.

use std::collections::HashSet;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use iai_mcp_rank_core::{tokenize, Inner, BM25_B, BM25_K1, MAX_QUERY_TOKENS};

const DIM: usize = 384;
const N_RECORDS: usize = 14_489; // grounded live-store count (264-RESEARCH-DURABLE item 1)
const TOTAL_ADJACENCY: usize = 60_062; // grounded symmetrized resident adjacency count
const VOCAB_SIZE: u32 = 21_192; // grounded unique-token vocabulary
const TOTAL_POSTINGS: usize = 387_583; // grounded sum(unique tokens per doc)
const TOTAL_TOKEN_OCCURRENCES: usize = 603_694; // grounded sum(tokenize(surface).len())
const HD_BITS: usize = 4096; // BSC episodic tier width (LILLI_BSC_DEFAULT_DIM)
const HD_WORDS: usize = HD_BITS / 64; // popcount over u64 words
const WARM_ITERS: usize = 5;

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

/// Builds the fixed VOCAB_SIZE word pool once: `w` + zero-padded id (6
/// chars) + a 6-char lowercase filler derived from the id, so the same
/// vocabulary id always renders to the identical token string wherever it
/// is drawn (a per-occurrence random filler would fragment one vocabulary
/// entry into many distinct tokens and blow the grounded token count).
fn build_vocab(size: u32, seed: u32) -> Vec<String> {
    (0..size)
        .map(|id| {
            let mut rng = id ^ seed;
            let filler: String = (0..6)
                .map(|_| (b'a' + (xorshift32(&mut rng) % 26) as u8) as char)
                .collect();
            format!("w{id:05}{filler}")
        })
        .collect()
}

fn pick_distinct_ids(rng: &mut u32, vocab_size: u32, count: usize, forced: &[u32]) -> Vec<u32> {
    let mut seen: HashSet<u32> = forced.iter().copied().collect();
    let mut out: Vec<u32> = forced.to_vec();
    while out.len() < count {
        let candidate = xorshift32(rng) % vocab_size;
        if seen.insert(candidate) {
            out.push(candidate);
        }
    }
    out
}

/// Expands the doc's distinct-token ids into the full occurrence list
/// (unique ids plus repeats of the first few, mimicking the real corpus's
/// per-doc repeat rate of ~1.56x unique-token count).
fn expand_occurrences(rng: &mut u32, unique_ids: &[u32], total_count: usize) -> Vec<u32> {
    let mut out = unique_ids.to_vec();
    let repeat_pool_len = unique_ids.len().min(5).max(1);
    while out.len() < total_count {
        let idx = (xorshift32(rng) as usize) % repeat_pool_len;
        out.push(unique_ids[idx]);
    }
    out
}

fn synthetic_hv(rng: &mut u32) -> [u64; HD_WORDS] {
    let mut hv = [0u64; HD_WORDS];
    for word in hv.iter_mut() {
        let lo = xorshift32(rng) as u64;
        let hi = xorshift32(rng) as u64;
        *word = (hi << 32) | lo;
    }
    hv
}

struct Corpus {
    inner: Inner,
    query_hv: [u64; HD_WORDS],
    record_hvs: Vec<[u64; HD_WORDS]>,
}

/// Builds the columnar `Inner` at the grounded 14,489-record scale, with
/// per-doc unique/total token counts exactly summing to the grounded
/// postings (387,583) and total-occurrence (603,694) targets, and
/// per-record adjacency exactly summing to the grounded 60,062 count.
/// `structure_hv` is synthesized in parallel, off the columnar struct
/// entirely -- `Inner` has no such column to populate.
fn build_corpus() -> Corpus {
    let mut rng: u32 = 0x1234_5678;
    let vocab = build_vocab(VOCAB_SIZE, 0x9E37_79B9);
    let aaak_vocab = build_vocab(200, 0x517C_C1B7);

    // Exact-sum adjacency split: base 4 edges/record, remainder records get
    // one extra edge, summing to TOTAL_ADJACENCY exactly.
    let base_edges = TOTAL_ADJACENCY / N_RECORDS;
    let extra_edge_records = TOTAL_ADJACENCY - base_edges * N_RECORDS;

    // Exact-sum postings split: base unique tokens/doc, remainder records
    // get one extra unique token, summing to TOTAL_POSTINGS exactly.
    let base_unique = TOTAL_POSTINGS / N_RECORDS;
    let extra_unique_records = TOTAL_POSTINGS - base_unique * N_RECORDS;

    // Exact-sum occurrence split: base total tokens/doc, remainder records
    // get one extra occurrence, summing to TOTAL_TOKEN_OCCURRENCES exactly.
    let base_total = TOTAL_TOKEN_OCCURRENCES / N_RECORDS;
    let extra_total_records = TOTAL_TOKEN_OCCURRENCES - base_total * N_RECORDS;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    let mut ids = Vec::with_capacity(N_RECORDS);
    let mut vectors_flat = Vec::with_capacity(N_RECORDS * DIM);
    let mut edge_map = std::collections::HashMap::with_capacity(N_RECORDS);
    let mut surfaces = Vec::with_capacity(N_RECORDS);
    let mut aaak_index = Vec::with_capacity(N_RECORDS);
    let mut created_at = Vec::with_capacity(N_RECORDS);
    let mut stability = Vec::with_capacity(N_RECORDS);
    let tier = vec!["episodic".to_string(); N_RECORDS];
    let tags = vec![Vec::new(); N_RECORDS];
    let mut salience_level = Vec::with_capacity(N_RECORDS);
    let centrality = vec![0.0f32; N_RECORDS];
    let mut record_hvs = Vec::with_capacity(N_RECORDS);

    let query_hv = synthetic_hv(&mut rng);
    // The MAX_QUERY_TOKENS query tokens the BM25 pass looks up must be
    // resident; slot 0 forces vocabulary ids [0, MAX_QUERY_TOKENS) into its
    // unique set so the sparse lookup never silently misses.
    let forced_query_ids: Vec<u32> = (0..MAX_QUERY_TOKENS as u32).collect();

    for i in 0..N_RECORDS {
        let id = i as u128 + 1;
        ids.push(id);
        vectors_flat.extend(synthetic_vector(&mut rng));
        record_hvs.push(synthetic_hv(&mut rng));

        let edges_here = base_edges + usize::from(i < extra_edge_records);
        let neighbors: Vec<(u128, f32, String)> = (1..=edges_here)
            .map(|k| {
                let neighbor = ((i + k) % N_RECORDS) as u128 + 1;
                let weight = 0.5 + (xorshift32(&mut rng) as f32 / u32::MAX as f32) * 0.5;
                (neighbor, weight, "hebbian".to_string())
            })
            .collect();
        edge_map.insert(id, neighbors);

        let unique_here = base_unique + usize::from(i < extra_unique_records);
        let total_here = base_total + usize::from(i < extra_total_records);
        let forced: &[u32] = if i == 0 { &forced_query_ids } else { &[] };
        let unique_ids = pick_distinct_ids(&mut rng, VOCAB_SIZE, unique_here, forced);
        let occurrence_ids = expand_occurrences(&mut rng, &unique_ids, total_here);
        let surface = occurrence_ids
            .iter()
            .map(|&wid| vocab[wid as usize].as_str())
            .collect::<Vec<_>>()
            .join(" ");
        surfaces.push(surface);

        // Grounded ratio: ~5,325 / 14,489 records carry non-empty aaak_index.
        if i < 5_325 {
            let aaak_ids = pick_distinct_ids(&mut rng, 200, 5, &[]);
            let text = aaak_ids
                .iter()
                .map(|&wid| aaak_vocab[wid as usize].as_str())
                .collect::<Vec<_>>()
                .join(" ");
            aaak_index.push(text);
        } else {
            aaak_index.push(String::new());
        }

        let days_ago = (xorshift32(&mut rng) % 730) as i64;
        let secs_jitter = (xorshift32(&mut rng) % 86_400) as i64;
        created_at.push(now - days_ago * 86_400 - secs_jitter);
        stability.push((xorshift32(&mut rng) as f32 / u32::MAX as f32).clamp(0.0, 1.0));
        salience_level.push((xorshift32(&mut rng) % 3) as u8);
    }

    let posting_count: usize = surfaces
        .iter()
        .map(|s| tokenize(s).into_iter().collect::<HashSet<_>>().len())
        .sum();
    let occurrence_count: usize = surfaces.iter().map(|s| tokenize(s).len()).sum();
    let adjacency_count: usize = edge_map.values().map(Vec::len).sum();
    let avg_surface_chars: f64 =
        surfaces.iter().map(|s| s.len()).sum::<usize>() as f64 / N_RECORDS as f64;
    eprintln!(
        "scan_cost: fixture n={N_RECORDS} postings={posting_count} \
         (target {TOTAL_POSTINGS}) occurrences={occurrence_count} \
         (target {TOTAL_TOKEN_OCCURRENCES}) adjacency={adjacency_count} \
         (target {TOTAL_ADJACENCY}) avg_surface_chars={avg_surface_chars:.1} (target 559.0)"
    );

    let pending = vec![false; N_RECORDS];
    let inner = Inner::from_columns(
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
    .expect("fixture build must not fail");

    Corpus {
        inner,
        query_hv,
        record_hvs,
    }
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
        "scan_cost: {label}: min={min:.3} ms median={med:.3} ms max={max:.3} ms (n={})",
        samples.len(),
    );
    med
}

/// One full fused-scan pass: dense per-slot terms (cosine, aaak overlap,
/// degree norm, age penalty, HD Hamming) over every resident record, then a
/// sparse BM25 posting-lookup pass over `MAX_QUERY_TOKENS` query tokens.
/// `record_norms` is precomputed once outside the timed loop (a repeated
/// scan against a persistent index would not re-derive vector norms on
/// every query). Representative per-term arithmetic to size the scan --
/// not the real scorer: no summation-order or tie-break fidelity is
/// claimed.
fn fused_scan(
    corpus: &Corpus,
    query_vector: &[f32],
    record_norms: &[f32],
    query_aaak: &HashSet<String>,
    now: i64,
) -> f64 {
    let inner = &corpus.inner;
    let n = inner.len();

    let query_norm = query_vector.iter().map(|x| x * x).sum::<f32>().sqrt();

    let mut scores = vec![0.0f64; n];
    let created_at_col = inner.created_at();

    for slot in 0..n {
        let row = inner.vector_row(slot);
        let dot: f32 = query_vector.iter().zip(row).map(|(a, b)| a * b).sum();
        let cosine = dot / (query_norm * record_norms[slot] + 1e-9);

        let aaak = inner.aaak_text(slot);
        let mut overlap = 0u32;
        if !aaak.is_empty() {
            for tok in tokenize(aaak) {
                if query_aaak.contains(&tok) {
                    overlap += 1;
                }
            }
        }

        let degree = inner.degree(slot);
        let degree_term = ((degree + 1) as f32).ln();

        let age_days = ((now - created_at_col[slot]) as f64 / 86_400.0).max(0.0);
        let age_term = 1.0 / (1.0 + age_days / 30.0);

        let mut hamming = 0u32;
        let record_hv = &corpus.record_hvs[slot];
        for w in 0..HD_WORDS {
            hamming += (corpus.query_hv[w] ^ record_hv[w]).count_ones();
        }
        let hamming_term = 1.0 - (hamming as f64 / HD_BITS as f64);

        let stability = inner.stability(slot);
        let salience = inner.salience_level(slot) as f32;

        scores[slot] = cosine as f64
            + overlap as f64 * 0.1
            + degree_term as f64 * 0.05
            + age_term
            + hamming_term
            + stability as f64 * 0.01
            + salience as f64 * 0.01;
    }

    // Sparse BM25 pass: posting-lookup over the fixed query token set,
    // ported constants (BM25_K1/BM25_B) from the crate's own postings
    // builder so the term cost matches the production formula shape.
    let n_docs = inner.n_docs() as f64;
    let avg_len = inner.avg_len();
    for word in QUERY_WORDS.get_or_init(build_query_words).iter() {
        if let Some(token_id) = inner.token_id(word) {
            let slots = inner.posting_slots(token_id);
            let tfs = inner.posting_tfs(token_id);
            let doc_freq = slots.len() as f64;
            let idf = ((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0).ln();
            for (i, &slot) in slots.iter().enumerate() {
                let tf = tfs[i] as f64;
                let dl = inner.doc_len(slot as usize) as f64;
                let denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / avg_len);
                let bm25 = idf * (tf * (BM25_K1 + 1.0)) / denom;
                scores[slot as usize] += bm25;
            }
        }
    }

    scores.iter().sum()
}

static QUERY_WORDS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();

/// Resolves the `MAX_QUERY_TOKENS` query words from the same vocab table
/// slot 0 was forced to contain, by re-deriving them the same way
/// `build_vocab` does (deterministic given the same seed/id, independent of
/// the vocab size passed in).
fn build_query_words() -> Vec<String> {
    build_vocab(MAX_QUERY_TOKENS as u32, 0x9E37_79B9)
}

fn main() {
    let corpus = build_corpus();
    let mut rng: u32 = 0xC0FF_EE11;
    let query_vector = synthetic_vector(&mut rng);
    let query_aaak_ids = pick_distinct_ids(&mut rng, 200, 3, &[]);
    let aaak_vocab = build_vocab(200, 0x517C_C1B7);
    let query_aaak: HashSet<String> = query_aaak_ids
        .iter()
        .map(|&id| aaak_vocab[id as usize].clone())
        .collect();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    let record_norms: Vec<f32> = (0..corpus.inner.len())
        .map(|slot| {
            corpus
                .inner
                .vector_row(slot)
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt()
        })
        .collect();

    // Warm-up pass, discarded -- first pass pays allocator/cache warm-up
    // the repeated warm iterations below should not be charged for.
    std::hint::black_box(fused_scan(&corpus, &query_vector, &record_norms, &query_aaak, now));

    let mut samples = Vec::with_capacity(WARM_ITERS);
    for _ in 0..WARM_ITERS {
        let start = Instant::now();
        let total = fused_scan(&corpus, &query_vector, &record_norms, &query_aaak, now);
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(total);
    }

    report("fused-scan (full term set, warm)", &samples);
}

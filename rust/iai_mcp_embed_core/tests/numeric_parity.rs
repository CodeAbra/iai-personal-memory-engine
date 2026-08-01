//! Embedder acceptance gate — per-text cosine ≥ 0.9999 between BertEmbedder
//! output and the frozen PyTorch baseline at `bench/embedder_baseline/`.
//!
//! Run: `cd rust && cargo nextest run -p iai_mcp_embed_core --test numeric_parity`
//!
//! Closes SPEC.md R2 (per-text cosine threshold) + R3 (tokenizer determinism)
//! at the Rust forward-pass level. The Python-side PyO3 wrapper parity test
//! lives in a later wave.
//!
//! This file is the strongest correctness gate for the BertEmbedder forward
//! pass: if any of the three RESEARCH.md pitfalls (wrong pooling, wrong GELU,
//! wrong LayerNorm eps) regresses, cosine drops below 0.9999 and this test
//! reports the failing indices.

mod baseline_loader;

use baseline_loader::load_npy_f32;
use iai_mcp_embed_core::bert::BertEmbedder;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

/// `CARGO_MANIFEST_DIR` = `rust/iai_mcp_embed_core`; two `..` reach repo root.
fn baseline_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../bench/embedder_baseline")
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "cosine_sim: length mismatch");
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn hf_cache_present() -> bool {
    dirs::home_dir()
        .map(|h| {
            h.join(".cache/huggingface/hub/models--BAAI--bge-small-en-v1.5")
                .join("snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/model.safetensors")
                .exists()
        })
        .unwrap_or(false)
}

#[test]
fn baseline_sha256_matches_frozen_value() {
    let path = baseline_dir().join("vectors.npy");
    // nosemgrep
    let bytes = std::fs::read(&path).expect("read vectors.npy");
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let actual = format!("{:x}", hasher.finalize());
    // SHA256 of the full on-disk vectors.npy file (magic + NPY header + raw bytes).
    // Note: this differs from `hashlib.sha256(np_array.tobytes())` by ~80 bytes
    // of NPY header. The capture script (bench/embedder_baseline.py) was patched
    // post-49-03 to hash the file after np.save, not vectors.tobytes() before.
    let expected = "31cc9bb0643835b872dbd21e0553b898e3de79cc28aebcf8c27814363ec5432b";
    assert_eq!(
        actual, expected,
        "vectors.npy SHA256 drift — frozen baseline may have been overwritten (expected {expected}, got {actual})"
    );
}

#[test]
fn baseline_loader_yields_expected_shape() {
    let (data, shape) = load_npy_f32(baseline_dir().join("vectors.npy")).expect("load_npy");
    assert_eq!(shape, vec![100, 384], "baseline shape drift");
    assert_eq!(data.len(), 38400, "baseline element count drift");
}

#[test]
fn test_all_baseline_texts() {
    if !hf_cache_present() {
        eprintln!(
            "HF cache absent — skipping numeric parity (Wave 1 / Wave 3 will trigger download)"
        );
        return;
    }

    let baseline_dir = baseline_dir();
    let texts: Vec<String> = serde_json::from_str(
        &std::fs::read_to_string(baseline_dir.join("texts.json")).expect("read texts.json"),
    )
    .expect("parse texts.json");
    assert_eq!(texts.len(), 100, "baseline texts.json length drift");

    let (baseline_vectors, baseline_shape) =
        load_npy_f32(baseline_dir.join("vectors.npy")).expect("load vectors.npy");
    assert_eq!(baseline_shape, vec![100, 384], "baseline shape drift");

    let embedder = BertEmbedder::load().expect("BertEmbedder::load");

    let mut failures: Vec<(usize, f32, String)> = Vec::new();

    for (i, text) in texts.iter().enumerate() {
        let got = embedder.encode(text).expect("encode");
        assert_eq!(got.len(), 384, "text[{i}] vector len drift");
        let exp = &baseline_vectors[i * 384..(i + 1) * 384];
        let cos = cosine_sim(&got, exp);
        if cos < 0.9999 {
            let preview: String = text.chars().take(60).collect();
            failures.push((i, cos, preview));
        }
    }

    if !failures.is_empty() {
        let summary: String = failures
            .iter()
            .take(10)
            .map(|(i, c, p)| format!("  text[{i}] cosine={c:.6} preview={p:?}"))
            .collect::<Vec<_>>()
            .join("\n");
        panic!(
            "{} of 100 baseline texts FAILED cosine ≥ 0.9999 threshold:\n{}\n(first 10 shown)",
            failures.len(),
            summary
        );
    }
}

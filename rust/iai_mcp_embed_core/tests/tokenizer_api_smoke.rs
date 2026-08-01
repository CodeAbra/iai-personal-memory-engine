//! Tokenizer-API smoke verification.
//!
//! The `with_truncation` method signature on `tokenizers::Tokenizer` 0.23.1
//! may differ from the HF Python API it mirrors. This smoke compiles and
//! invokes the exact code path `bert.rs` hard-codes, so a signature mismatch
//! fails here before it can break the embedder.

use tokenizers::{Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};

#[test]
fn truncation_api_compiles_and_runs() {
    // Build the simplest possible tokenizer from a known-good HF cache file.
    // The bge-small-en-v1.5 tokenizer.json is expected on disk (snapshot
    // SHA 5c38ec7c40...). If absent the test returns early (reports as
    // PASS) — first embedder use
    // triggers the lazy hf-hub download.
    let cache_root = dirs::home_dir().unwrap().join(".cache/huggingface/hub");
    let candidate = cache_root
        .join("models--BAAI--bge-small-en-v1.5")
        .join("snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/tokenizer.json");
    if !candidate.exists() {
        eprintln!(
            "tokenizer.json absent at {} — skipping smoke (lazy fetch happens on first embedder use)",
            candidate.display()
        );
        return;
    }

    let mut tokenizer = Tokenizer::from_file(&candidate).expect("from_file");

    // The API under test — the exact call signature
    // bert.rs::BertEmbedder::load() uses. If the field or method names
    // differ, this fails at compile or at runtime.
    let trunc = TruncationParams {
        max_length: 512,
        strategy: TruncationStrategy::LongestFirst,
        stride: 0,
        direction: TruncationDirection::Right,
    };
    tokenizer
        .with_truncation(Some(trunc))
        .expect("with_truncation should accept Some(TruncationParams)");

    // Encode a known >512 token text to verify truncation actually trims.
    let long_text = "word ".repeat(800);
    let encoding = tokenizer.encode(long_text.as_str(), true).expect("encode");
    assert!(
        encoding.get_ids().len() <= 512,
        "truncation did NOT trim to 512 tokens (got {})",
        encoding.get_ids().len()
    );
}

//! Tokenizer-API smoke verification.
//!
//! RESEARCH.md Assumption A2 (LOW confidence): the `with_truncation` method
//! signature on `tokenizers::Tokenizer` 0.23.1 may not match what we assumed
//! from the HF Python API. This smoke compiles + invokes the actual code path
//! that Wave 1 will hard-code in `bert.rs`, so if A2 is wrong the failure
//! surfaces here BEFORE Wave 1 commits any production logic to it.

use tokenizers::{Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};

#[test]
fn truncation_api_compiles_and_runs() {
    // Build the simplest possible tokenizer from a known-good HF cache file.
    // The bge-small-en-v1.5 tokenizer.json is already on disk per SPEC.md
    // background (snapshot SHA 5c38ec7c40...). If absent, the test xfails
    // (treated as PASS — Wave 1 will trigger the lazy hf-hub download).
    let cache_root = dirs::home_dir().unwrap().join(".cache/huggingface/hub");
    let candidate = cache_root
        .join("models--BAAI--bge-small-en-v1.5")
        .join("snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/tokenizer.json");
    if !candidate.exists() {
        eprintln!(
            "tokenizer.json absent at {} — skipping smoke (Wave 1 will trigger lazy fetch)",
            candidate.display()
        );
        return;
    }

    let mut tokenizer = Tokenizer::from_file(&candidate).expect("from_file");

    // The API under test — this is the exact call signature Wave 1 will use
    // in bert.rs::BertEmbedder::load(). If the field names or method names
    // differ from RESEARCH.md Pattern 9, this fails at compile or at runtime.
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

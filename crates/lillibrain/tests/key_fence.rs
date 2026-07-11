//! Guard test: the storage engine carries no cryptographic surface.
//!
//! The engine stores opaque blobs and is deliberately blind to their meaning.
//! All sealing happens above the storage seam in the host; no key, cipher, or
//! transform lives in this crate. This test scans the crate source for any
//! cryptographic identifier and fails on a single hit — mirroring the host's
//! residual-grep release gate, so a sealing symbol can never silently land here.

use std::fs;
use std::path::{Path, PathBuf};

/// Cryptographic tokens that must never appear in the crate source. The bare
/// word "key" is intentionally excluded — it is a legitimate B-tree term
/// (record keys, key ordering); only cryptographic compounds and primitives
/// are forbidden.
const FORBIDDEN: &[&str] = &[
    "aes",
    "gcm",
    "cipher",
    "encrypt",
    "decrypt",
    "crypto",
    "nonce",
    "key_material",
    "aes_key",
    "keymaterial",
];

/// Recursively collect every `.rs` file under `dir`.
fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("source dir is readable") {
        let path = entry.expect("dir entry is readable").path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn crate_source_has_no_crypto_symbol() {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_sources(&src, &mut files);
    assert!(!files.is_empty(), "expected at least one source file under src/");

    let mut hits: Vec<String> = Vec::new();
    for file in &files {
        let text = fs::read_to_string(file).expect("source file is valid UTF-8");
        for (lineno, line) in text.lines().enumerate() {
            let lowered = line.to_ascii_lowercase();
            for token in FORBIDDEN {
                if lowered.contains(token) {
                    hits.push(format!(
                        "{}:{}: forbidden crypto token '{}' -> {}",
                        file.display(),
                        lineno + 1,
                        token,
                        line.trim()
                    ));
                }
            }
        }
    }

    assert!(
        hits.is_empty(),
        "the storage engine must carry no cryptographic surface; found:\n{}",
        hits.join("\n")
    );
}

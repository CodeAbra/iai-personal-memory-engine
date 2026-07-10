//! Golden-fixture loader.
//!
//! A golden is a `<stem>.json` manifest plus a sibling `<stem>.bin` raw payload.
//! [`load_golden`] reads both, then enforces the format contract: the `.bin`
//! length must equal `product(shape) * sizeof(dtype)`, its sha256 must match the
//! manifest, and the byte order must be little-endian. Any mismatch — a tampered
//! or corrupted `.bin` — errors instead of returning silently wrong bytes.

use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

/// Manifest describing a golden payload.
///
/// `format` selects the payload shape: `raw_bytes` (the implemented path, a raw
/// little-endian array in the `.bin`) or `json` (reserved — structured data
/// carried inline in `payload`).
#[derive(Debug, Clone, Deserialize)]
pub struct GoldenManifest {
    pub name: String,
    pub subsystem: String,
    pub format: String,
    #[serde(default)]
    pub payload: Option<serde_json::Value>,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub byte_order: String,
    pub sha256: String,
    pub source: String,
    pub generated_by: String,
}

/// A loaded golden: the verified raw bytes plus its manifest.
#[derive(Debug, Clone)]
pub struct GoldenBytes {
    pub bytes: Vec<u8>,
    pub manifest: GoldenManifest,
}

/// Errors raised while loading or validating a golden.
#[derive(Debug, thiserror::Error)]
pub enum GoldenError {
    #[error("failed to read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse manifest {path}: {source}")]
    Manifest {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("unsupported golden format {format:?} (only \"raw_bytes\" is implemented)")]
    UnsupportedFormat { format: String },

    #[error("byte order must be \"le\", got {byte_order:?}")]
    ByteOrder { byte_order: String },

    #[error("unknown dtype {dtype:?}")]
    UnknownDtype { dtype: String },

    #[error("degenerate shape {shape:?}: every dimension must be >= 1 and the shape must be non-empty")]
    DegenerateShape { shape: Vec<usize> },

    #[error("length mismatch: .bin is {actual} bytes, manifest implies {expected}")]
    Length { actual: usize, expected: usize },

    #[error("sha256 mismatch: .bin hashes to {actual}, manifest says {expected}")]
    Sha256 { actual: String, expected: String },
}

fn dtype_size(dtype: &str) -> Option<usize> {
    match dtype {
        "u8" | "i8" => Some(1),
        "f16" | "u16" | "i16" => Some(2),
        "f32" | "u32" | "i32" => Some(4),
        "f64" | "u64" | "i64" => Some(8),
        _ => None,
    }
}

/// Load and validate a golden from its manifest path.
///
/// Reads the sibling `.bin`, then asserts length, sha256, and byte order against
/// the manifest before returning the bytes.
pub fn load_golden(manifest_path: impl AsRef<Path>) -> Result<GoldenBytes, GoldenError> {
    let manifest_path = manifest_path.as_ref();

    let manifest_text = std::fs::read_to_string(manifest_path).map_err(|e| GoldenError::Io {
        path: manifest_path.to_path_buf(),
        source: e,
    })?;
    let manifest: GoldenManifest =
        serde_json::from_str(&manifest_text).map_err(|e| GoldenError::Manifest {
            path: manifest_path.to_path_buf(),
            source: e,
        })?;

    if manifest.format != "raw_bytes" {
        return Err(GoldenError::UnsupportedFormat {
            format: manifest.format,
        });
    }
    if manifest.byte_order != "le" {
        return Err(GoldenError::ByteOrder {
            byte_order: manifest.byte_order,
        });
    }

    let elem_size = dtype_size(&manifest.dtype).ok_or_else(|| GoldenError::UnknownDtype {
        dtype: manifest.dtype.clone(),
    })?;

    // Reject a degenerate shape before computing the expected length. An empty
    // shape makes `product()` return 1 (the multiplicative identity), and a zero
    // dimension makes it return 0 — both turn the length assertion into a no-op
    // that would pass an arbitrarily-sized `.bin`. A real golden has a concrete,
    // non-empty extent in every dimension.
    if manifest.shape.is_empty() || manifest.shape.iter().any(|&d| d == 0) {
        return Err(GoldenError::DegenerateShape {
            shape: manifest.shape.clone(),
        });
    }
    let expected_len = manifest.shape.iter().product::<usize>() * elem_size;

    let bin_path = manifest_path.with_extension("bin");
    let bytes = std::fs::read(&bin_path).map_err(|e| GoldenError::Io {
        path: bin_path.clone(),
        source: e,
    })?;

    if bytes.len() != expected_len {
        return Err(GoldenError::Length {
            actual: bytes.len(),
            expected: expected_len,
        });
    }

    let digest = format!("{:x}", Sha256::digest(&bytes));
    if digest != manifest.sha256 {
        return Err(GoldenError::Sha256 {
            actual: digest,
            expected: manifest.sha256,
        });
    }

    Ok(GoldenBytes { bytes, manifest })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a manifest + sibling `.bin` into a fresh temp dir and return the
    /// manifest path. The caller owns the returned temp dir (kept alive via the
    /// returned guard) for the duration of the test.
    fn write_pair(shape_json: &str, bin: &[u8]) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("temp dir");
        let stem = dir.path().join("degenerate");
        let manifest_path = stem.with_extension("json");
        let bin_path = stem.with_extension("bin");

        let digest = format!("{:x}", Sha256::digest(bin));
        let manifest = format!(
            r#"{{
                "name": "degenerate",
                "subsystem": "test",
                "format": "raw_bytes",
                "dtype": "u8",
                "shape": {shape_json},
                "byte_order": "le",
                "sha256": "{digest}",
                "source": "synthetic",
                "generated_by": "test"
            }}"#
        );
        std::fs::write(&manifest_path, manifest).expect("write manifest");
        std::fs::write(&bin_path, bin).expect("write bin");
        (dir, manifest_path)
    }

    #[test]
    fn empty_shape_is_rejected() {
        // shape: [] makes product() == 1; without the guard a 1-byte .bin would
        // pass the length check as a no-op. It must be rejected outright.
        let (_dir, manifest_path) = write_pair("[]", &[0u8]);
        let err = load_golden(&manifest_path).expect_err("empty shape must error");
        assert!(
            matches!(err, GoldenError::DegenerateShape { .. }),
            "expected DegenerateShape, got {err:?}"
        );
    }

    #[test]
    fn zero_dimension_is_rejected() {
        // shape: [4, 0] makes product() == 0; the length guard would pass an
        // empty .bin. It must be rejected outright.
        let (_dir, manifest_path) = write_pair("[4, 0]", &[]);
        let err = load_golden(&manifest_path).expect_err("zero dimension must error");
        assert!(
            matches!(err, GoldenError::DegenerateShape { .. }),
            "expected DegenerateShape, got {err:?}"
        );
    }
}

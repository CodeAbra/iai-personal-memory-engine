//! Parity harness: golden fixtures plus a co-scan and bench gate that prove the
//! Rust components match the reference implementation row-for-row and meet the
//! performance bar before any component is adopted.

pub mod golden;
pub mod thresholds;
pub mod verifier;

pub use golden::{load_golden, GoldenBytes, GoldenError, GoldenManifest};
pub use verifier::{DimStatus, DimensionResult, StoreVerifier, VerifyReport};

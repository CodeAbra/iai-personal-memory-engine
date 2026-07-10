//! Store-parity verifier: a six-dimension co-scan that proves a candidate store
//! reproduces the reference store exactly before it is adopted.
//!
//! The dimensions and their durable contracts (these outlast the skeleton — the
//! real bodies fill in later, but the shape and the rules are fixed here):
//!
//! - **A counts** — per-table row counts over the fixed table allow-list.
//! - **B bytes** — ciphertext + embedding byte-equality.
//! - **C decrypt** — decrypt parity.
//! - **D recall** — recall hit-id + score parity.
//! - **E temporal** — temporal-recall parity.
//! - **F codec** — versioned codec round-trip (the tier marker and the structured
//!   payload).
//!
//! ## Read discipline
//!
//! Every dimension reads the source through a **throwaway snapshot copy** of the
//! store directory — including the write-ahead and shared-memory sidecars — never
//! the live directory. The verifier must never mutate or contend with a live
//! store.
//!
//! ## Scan discipline
//!
//! A, B, C, and F are **bulk ordered co-scans**: index the destination once into
//! an id-keyed map, then stream the source in id order. They must never issue a
//! per-id lookup inside a row loop — a per-id probe full-scans the un-indexed
//! event log and collapses at scale.
//!
//! ## Recall discipline
//!
//! D and E observe **real on-open recall** and must never force an index rebuild.
//! A forced rebuild masks the empty-index failure where a freshly opened store
//! answers every query with nothing while the reference answers normally: if the
//! candidate returns no hits where the source returns hits, that is RED. Recall
//! that blows up is caught and reported RED, never propagated.
//!
//! ## Crypto seam
//!
//! C is the only dimension that touches the encryption key, and it does so above
//! the engine seam. The storage engine stays crypto-blind: it sees ciphertext
//! blobs only and never the key.
//!
//! [`StoreVerifier::verify`] runs all six and reports `ok` as the strict AND of
//! every dimension — a single RED fails the whole report.

use std::collections::BTreeMap;

use serde::Serialize;

/// Status of one verification dimension.
///
/// `Pending` is structurally distinct from `Pass`: a dimension that has not been
/// implemented (or could not run) is **not** a verified dimension, and the strict
/// gate must never treat it as one. This is the load-bearing invariant of the
/// whole harness — the verifier exists because a candidate store once passed every
/// dimension it *did* check yet broke on one it never checked (recall=0). A
/// default-passing skip is exactly that false-pass shape, so `Pending` can never
/// green the gate. Only `Pass` does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DimStatus {
    /// The dimension ran and matched the reference.
    Pass,
    /// The dimension ran and diverged from the reference.
    Fail,
    /// The dimension has not been implemented or could not run. NON-passing.
    Pending,
}

/// Outcome of one verification dimension.
#[derive(Debug, Clone, Serialize)]
pub struct DimensionResult {
    pub status: DimStatus,
    pub reason: String,
    pub counts: BTreeMap<String, i64>,
}

impl DimensionResult {
    /// A passing result with no extra counts.
    pub fn green(reason: impl Into<String>) -> Self {
        Self {
            status: DimStatus::Pass,
            reason: reason.into(),
            counts: BTreeMap::new(),
        }
    }

    /// A failing result.
    pub fn red(reason: impl Into<String>) -> Self {
        Self {
            status: DimStatus::Fail,
            reason: reason.into(),
            counts: BTreeMap::new(),
        }
    }

    /// A not-yet-implemented dimension.
    ///
    /// Returns [`DimStatus::Pending`], which the strict gate treats as NON-passing
    /// — a pending dimension must never green the report. A real production
    /// verifier path must replace every `pending()` body with [`green`](Self::green)
    /// or [`red`](Self::red): a skipped dimension is not a verified dimension, and
    /// leaving one pending is the exact false-pass the harness exists to catch.
    pub fn pending() -> Self {
        Self {
            status: DimStatus::Pending,
            reason: "not implemented".to_string(),
            counts: BTreeMap::new(),
        }
    }

    /// Whether this dimension counts as passing the strict gate. Only
    /// [`DimStatus::Pass`] passes; both `Fail` and `Pending` are non-passing.
    pub fn is_pass(&self) -> bool {
        self.status == DimStatus::Pass
    }
}

/// Aggregate verification report across all dimensions.
#[derive(Debug, Clone, Serialize)]
pub struct VerifyReport {
    pub ok: bool,
    pub dimensions: BTreeMap<String, DimensionResult>,
}

/// A store-parity verifier. Implementors provide each dimension; the default
/// [`verify`](StoreVerifier::verify) aggregates them under stable keys with a
/// strict-AND `ok`.
pub trait StoreVerifier {
    /// Per-table row counts over the fixed table allow-list. Bulk co-scan.
    fn dim_a_counts(&self) -> DimensionResult;

    /// Ciphertext + embedding byte-equality. Bulk ordered co-scan — index the
    /// destination once, stream the source in id order; never a per-id probe.
    fn dim_b_bytes(&self) -> DimensionResult;

    /// Decrypt parity. The only key-touching dimension, above the engine seam.
    fn dim_c_decrypt(&self) -> DimensionResult;

    /// Recall hit-id + score parity over real on-open recall. Must not force an
    /// index rebuild; candidate-empty while source-nonempty is RED.
    fn dim_d_recall(&self) -> DimensionResult;

    /// Temporal-recall parity. Same snapshot + no-rebuild discipline as D.
    fn dim_e_temporal(&self) -> DimensionResult;

    /// Versioned codec round-trip via the co-scan.
    fn dim_f_codec(&self) -> DimensionResult;

    /// Run every dimension and aggregate. `ok` is the strict AND of all six.
    fn verify(&self) -> VerifyReport {
        let mut dimensions = BTreeMap::new();
        dimensions.insert("A_counts".to_string(), self.dim_a_counts());
        dimensions.insert("B_bytes".to_string(), self.dim_b_bytes());
        dimensions.insert("C_decrypt".to_string(), self.dim_c_decrypt());
        dimensions.insert("D_recall".to_string(), self.dim_d_recall());
        dimensions.insert("E_temporal".to_string(), self.dim_e_temporal());
        dimensions.insert("F_codec".to_string(), self.dim_f_codec());

        // Strict AND: only a `Pass` counts. A `Pending` (unimplemented) dimension
        // is non-passing, so a partial harness can never green the gate.
        let ok = dimensions.values().all(|d| d.is_pass());
        VerifyReport { ok, dimensions }
    }
}

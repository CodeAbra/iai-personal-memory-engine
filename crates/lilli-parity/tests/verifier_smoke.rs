//! Smoke tests for the StoreVerifier skeleton: a pending (unimplemented)
//! dimension must NEVER pass the strict gate, dimension A green and red, and
//! report serialization.

use lilli_parity::{DimStatus, DimensionResult, StoreVerifier};

/// An in-memory store standing in for a real candidate/reference pair. Only
/// dimension A (row counts of two tables) is implemented; B–F are pending stubs.
struct FixtureStore {
    src_count: i64,
    dst_count: i64,
}

impl StoreVerifier for FixtureStore {
    fn dim_a_counts(&self) -> DimensionResult {
        if self.src_count == self.dst_count {
            DimensionResult::green(format!("counts match ({})", self.src_count))
        } else {
            DimensionResult::red(format!(
                "count mismatch: src={} dst={}",
                self.src_count, self.dst_count
            ))
        }
    }

    fn dim_b_bytes(&self) -> DimensionResult {
        DimensionResult::pending()
    }

    fn dim_c_decrypt(&self) -> DimensionResult {
        DimensionResult::pending()
    }

    fn dim_d_recall(&self) -> DimensionResult {
        DimensionResult::pending()
    }

    fn dim_e_temporal(&self) -> DimensionResult {
        DimensionResult::pending()
    }

    fn dim_f_codec(&self) -> DimensionResult {
        DimensionResult::pending()
    }
}

/// A store where every dimension is implemented and green. Only such a report
/// may pass the strict gate.
struct AllGreenStore;

impl StoreVerifier for AllGreenStore {
    fn dim_a_counts(&self) -> DimensionResult {
        DimensionResult::green("counts match")
    }
    fn dim_b_bytes(&self) -> DimensionResult {
        DimensionResult::green("bytes equal")
    }
    fn dim_c_decrypt(&self) -> DimensionResult {
        DimensionResult::green("decrypt parity")
    }
    fn dim_d_recall(&self) -> DimensionResult {
        DimensionResult::green("recall parity")
    }
    fn dim_e_temporal(&self) -> DimensionResult {
        DimensionResult::green("temporal parity")
    }
    fn dim_f_codec(&self) -> DimensionResult {
        DimensionResult::green("codec round-trip")
    }
}

#[test]
fn all_stub_report_does_not_pass() {
    // Dimension A is green, but B–F are pending. A pending (unimplemented)
    // dimension is NOT a verified dimension, so the strict gate must stay red.
    // This is the silent false-pass shape the verifier exists to prevent:
    // a correct-but-incomplete run must never green the gate.
    let store = FixtureStore {
        src_count: 42,
        dst_count: 42,
    };
    let report = store.verify();
    assert!(
        !report.ok,
        "a report with any pending dimension must NOT pass the strict gate"
    );
    assert!(report.dimensions["A_counts"].is_pass(), "dimension A green");
    assert_eq!(
        report.dimensions["B_bytes"].status,
        DimStatus::Pending,
        "B is a pending stub"
    );
    assert!(
        !report.dimensions["B_bytes"].is_pass(),
        "a pending dimension is non-passing"
    );
}

#[test]
fn report_passes_only_when_every_dimension_is_implemented_pass() {
    let report = AllGreenStore.verify();
    assert!(
        report.ok,
        "only an all-implemented-Pass report passes the strict gate"
    );
    for (name, dim) in &report.dimensions {
        assert_eq!(dim.status, DimStatus::Pass, "{name} must be Pass");
    }

    // The report must serialize without error.
    let json = serde_json::to_string(&report).expect("report serializes");
    assert!(json.contains("A_counts"));
}

#[test]
fn mismatched_fixture_fails_strict_and() {
    let store = FixtureStore {
        src_count: 42,
        dst_count: 41,
    };
    let report = store.verify();
    assert!(!report.ok, "a single RED must fail the strict-AND report");
    assert_eq!(
        report.dimensions["A_counts"].status,
        DimStatus::Fail,
        "dimension A red"
    );
}

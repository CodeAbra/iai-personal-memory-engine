//! Smoke tests for the golden loader against the committed fixtures.

use std::path::PathBuf;

use lilli_parity::load_golden;

/// Fixtures live at the repo root, two levels up from this crate.
fn fixtures_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/golden")
}

const PROJECTION_SHA256: &str =
    "df97cc72a960567da17edbba16107881340349bd47b69f9b58d3091d96eb4e4e";

#[test]
fn rails_round_trip_loads_and_verifies() {
    let manifest = fixtures_root().join("_rails/round_trip.json");
    let golden = load_golden(&manifest).expect("rails golden loads (length + sha256 checked)");

    let expected: Vec<u8> = (0u8..16).collect();
    assert_eq!(golden.bytes, expected, "rails bytes must be 0..16");
    assert_eq!(golden.manifest.dtype, "u8");
    assert_eq!(golden.manifest.shape, vec![16]);
}

#[test]
fn projection_golden_matches_locked_hash() {
    let manifest = fixtures_root().join("hdc/projection.json");
    let golden =
        load_golden(&manifest).expect("projection golden loads (length + sha256 checked)");

    assert_eq!(
        golden.bytes.len(),
        15_360_000,
        "projection golden must be 384 * 10000 * 4 bytes"
    );
    assert_eq!(
        golden.manifest.sha256, PROJECTION_SHA256,
        "projection manifest must carry the locked hash"
    );
    assert_eq!(golden.manifest.dtype, "f32");
    assert_eq!(golden.manifest.shape, vec![384, 10000]);
    assert_eq!(golden.manifest.byte_order, "le");
}

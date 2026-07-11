//! Committed insert-scaling gates.
//!
//! These constants live in library code (not in a bench or test file) so they
//! are compiled and exercised by `cargo test`, and the bench plus the host-side
//! linearity test reference the same numbers — the thresholds live in exactly
//! one place. They encode the proven post-fix insert profile: throughput is
//! linear in the row count and the per-insert page reads track tree height, not
//! the number of rows.

/// Sustained insert throughput floor, in rows per second.
///
/// The proven post-fix rate; a regression to a whole-tree sweep per insert
/// drops this by more than an order of magnitude.
pub const INSERT_ROWS_PER_S_FLOOR: f64 = 1500.0;

/// Upper bound on the wall-time growth factor when the row count doubles.
///
/// Linear insert cost makes doubling the rows roughly double the time; the
/// slack above 2.0 absorbs constant overhead and measurement noise. A quadratic
/// regression would push this factor toward 4.0.
pub const DOUBLING_TIME_RATIO_MAX: f64 = 2.2;

/// Upper bound on the average page reads per insert.
///
/// Insert descends the tree once and rewrites a constant number of pages, so
/// the per-insert read count is bounded by tree height — a small constant at
/// these corpus sizes, independent of the row count. A whole-tree sweep would
/// make this grow with the row count.
pub const READS_PER_INSERT_MAX: f64 = 64.0;

/// Returns true when `measured` throughput meets or beats the floor.
pub fn throughput_ok(measured_rows_per_s: f64) -> bool {
    measured_rows_per_s >= INSERT_ROWS_PER_S_FLOOR
}

/// Returns true when the wall-time ratio for a doubled row count stays linear.
pub fn doubling_ratio_ok(ratio: f64) -> bool {
    ratio <= DOUBLING_TIME_RATIO_MAX
}

/// Returns true when the average page reads per insert stay bounded by height.
pub fn reads_per_insert_ok(reads_per_insert: f64) -> bool {
    reads_per_insert <= READS_PER_INSERT_MAX
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn throughput_floor_direction() {
        assert!(throughput_ok(1510.0));
        assert!(throughput_ok(1500.0));
        assert!(!throughput_ok(28.0));
    }

    #[test]
    fn doubling_ratio_direction() {
        // The proven post-fix shape: 20k took 2.01x the 10k time.
        assert!(doubling_ratio_ok(2.01));
        assert!(doubling_ratio_ok(2.2));
        // A quadratic regression doubles work per doubled row count.
        assert!(!doubling_ratio_ok(3.9));
    }

    #[test]
    fn reads_per_insert_direction() {
        // A few-level tree: a handful of reads per insert.
        assert!(reads_per_insert_ok(6.0));
        assert!(reads_per_insert_ok(64.0));
        // Reads scaling with the row count is the regression signature.
        assert!(!reads_per_insert_ok(5000.0));
    }
}

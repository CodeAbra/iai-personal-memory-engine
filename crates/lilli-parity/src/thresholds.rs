//! Committed performance thresholds and the match-or-beat assertion helper.
//!
//! These constants live in library code (not in a bench file) so they are
//! exercised by `cargo test`. The bench references them so the numbers live in
//! exactly one place.

/// Recall p95 ceiling at 100 records, in milliseconds.
pub const RECALL_P95_N100_MS: f64 = 77.0;

/// Recall p95 ceiling at 1k records, in milliseconds.
pub const RECALL_P95_N1K_MS: f64 = 105.0;

/// Recall p95 ceiling at 10k records, in milliseconds. The hard gate.
pub const RECALL_P95_N10K_MS: f64 = 368.0;

/// Verbatim-recall target (exact reproduction).
pub const VERBATIM_RECALL: f64 = 1.0;

/// Verbatim-recall floor under drift.
pub const VERBATIM_FLOOR: f64 = 0.993;

/// Resident-set ceiling at 10k records, in megabytes.
pub const RSS_N10K_MB: f64 = 589.0;

/// Migration throughput floor, in rows per second (linear).
pub const MIGRATION_ROWS_PER_S: f64 = 1500.0;

/// Minimum HDC-kernel speedup over the pure-Python reference.
pub const HDC_SPEEDUP_MIN: f64 = 10.0;

/// Target HDC-kernel speedup over the pure-Python reference.
pub const HDC_SPEEDUP_TARGET: f64 = 100.0;

/// Steady-state session token budget.
pub const SESSION_TOKEN_BUDGET: f64 = 3000.0;

/// Returns true when `measured` meets or beats `threshold`.
///
/// For latency, resident memory, and token cost, lower is better
/// (`lower_is_better = true`). For speedup and verbatim-recall, higher is better
/// (`lower_is_better = false`).
pub fn meets_or_beats(measured: f64, threshold: f64, lower_is_better: bool) -> bool {
    if lower_is_better {
        measured <= threshold
    } else {
        measured >= threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lower_is_better_direction() {
        // Latency: 360ms beats the 368ms gate; 400ms fails.
        assert!(meets_or_beats(360.0, RECALL_P95_N10K_MS, true));
        assert!(!meets_or_beats(400.0, RECALL_P95_N10K_MS, true));

        // RSS: under the ceiling passes, over fails.
        assert!(meets_or_beats(500.0, RSS_N10K_MB, true));
        assert!(!meets_or_beats(700.0, RSS_N10K_MB, true));

        // Session cost: 2800 beats the 3000 budget; 3200 fails.
        assert!(meets_or_beats(2800.0, SESSION_TOKEN_BUDGET, true));
        assert!(!meets_or_beats(3200.0, SESSION_TOKEN_BUDGET, true));

        // Boundary: equal to the threshold meets it.
        assert!(meets_or_beats(368.0, RECALL_P95_N10K_MS, true));
    }

    #[test]
    fn higher_is_better_direction() {
        // Speedup: 15x beats the 10x floor; 8x fails.
        assert!(meets_or_beats(15.0, HDC_SPEEDUP_MIN, false));
        assert!(!meets_or_beats(8.0, HDC_SPEEDUP_MIN, false));

        // Verbatim recall: at floor passes, below fails.
        assert!(meets_or_beats(VERBATIM_RECALL, VERBATIM_FLOOR, false));
        assert!(!meets_or_beats(0.99, VERBATIM_FLOOR, false));

        // Boundary: equal to the floor meets it.
        assert!(meets_or_beats(10.0, HDC_SPEEDUP_MIN, false));
    }

    #[test]
    fn constants_hold_committed_values() {
        assert_eq!(RECALL_P95_N100_MS, 77.0);
        assert_eq!(RECALL_P95_N1K_MS, 105.0);
        assert_eq!(RECALL_P95_N10K_MS, 368.0);
        assert_eq!(VERBATIM_RECALL, 1.0);
        assert_eq!(VERBATIM_FLOOR, 0.993);
        assert_eq!(RSS_N10K_MB, 589.0);
        assert_eq!(MIGRATION_ROWS_PER_S, 1500.0);
        assert_eq!(HDC_SPEEDUP_MIN, 10.0);
        assert_eq!(HDC_SPEEDUP_TARGET, 100.0);
        assert_eq!(SESSION_TOKEN_BUDGET, 3000.0);
    }
}

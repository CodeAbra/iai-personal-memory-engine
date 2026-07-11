//! Error taxonomy for the storage engine.
//!
//! Every variant carries enough context to localize a fault on read or commit.
//! The engine fails loud: corruption, freelist cycles, and out-of-bounds page
//! references are surfaced as typed errors, never silently swallowed.

use thiserror::Error;

/// A storage-engine error.
#[derive(Debug, Error)]
pub enum StoreError {
    /// An underlying I/O operation failed.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// A page's stored checksum did not match its recomputed checksum.
    #[error("checksum mismatch on page {page_no} (expected {expected:#010x}, got {actual:#010x})")]
    CrcMismatch {
        /// The 1-indexed page number that failed verification.
        page_no: u32,
        /// The checksum stored in the page.
        expected: u32,
        /// The checksum recomputed over the page contents.
        actual: u32,
    },

    /// The freelist trunk chain is internally inconsistent (cycle or an entry
    /// beyond the current file size).
    #[error("freelist corruption: {detail}")]
    FreelistCorruption {
        /// A human-readable description of the inconsistency.
        detail: String,
    },

    /// A structural invariant was violated (the on-disk state is internally
    /// inconsistent in a way that is neither a checksum nor a freelist fault).
    #[error("integrity violation: {detail}")]
    Integrity {
        /// A human-readable description of the violation.
        detail: String,
    },

    /// A page number was referenced that lies outside the current file extent.
    #[error("page {page_no} out of bounds (db_size={db_size})")]
    PageOutOfBounds {
        /// The offending 1-indexed page number.
        page_no: u32,
        /// The current total page count.
        db_size: u32,
    },
}

/// Result alias for storage-engine operations.
pub type Result<T> = std::result::Result<T, StoreError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_crc_mismatch_carries_page_and_values() {
        let e = StoreError::CrcMismatch {
            page_no: 7,
            expected: 0xAABB_CCDD,
            actual: 0x0011_2233,
        };
        let s = format!("{e}");
        assert!(s.contains("page 7"));
        assert!(s.contains("aabbccdd"));
        assert!(s.contains("00112233"));
    }

    #[test]
    fn error_freelist_corruption_carries_detail() {
        let e = StoreError::FreelistCorruption {
            detail: "cycle at page 5".to_string(),
        };
        assert!(format!("{e}").contains("cycle at page 5"));
    }

    #[test]
    fn error_page_out_of_bounds_carries_context() {
        let e = StoreError::PageOutOfBounds {
            page_no: 99,
            db_size: 4,
        };
        let s = format!("{e}");
        assert!(s.contains("99"));
        assert!(s.contains("db_size=4"));
    }

    #[test]
    fn error_io_converts_from_std() {
        let io = std::io::Error::other("boom");
        let e: StoreError = io.into();
        assert!(matches!(e, StoreError::Io(_)));
    }
}

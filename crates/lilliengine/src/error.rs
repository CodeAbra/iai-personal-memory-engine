//! Error taxonomy for the SQL engine frontend.
//!
//! Every variant carries the verbatim user-facing message text the reference
//! engine raises, so callers (and the parity oracle) observe byte-identical SQL
//! error strings. The engine fails loud on any statement shape outside the
//! supported subset rather than silently mis-parsing it.

use thiserror::Error;

/// The class of an unrecoverable storage-layer corruption, preserved across the
/// crate boundary so a caller can branch on the fault (quarantine, rebuild from
/// bank) instead of treating it as a routine SQL error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorruptionKind {
    /// A page's stored checksum did not match its recomputed checksum.
    ChecksumMismatch,
    /// The freelist trunk chain is internally inconsistent.
    Freelist,
    /// A structural on-disk invariant was violated.
    Integrity,
    /// A page number was referenced outside the current file extent.
    PageOutOfBounds,
}

/// A SQL engine error.
///
/// `ParseError` and `UnsupportedStatement` both carry an already-formatted
/// message string. They are kept as distinct variants so a caller can tell a
/// malformed-but-in-subset statement (`ParseError`) apart from a well-formed
/// statement whose shape is deliberately outside the supported grammar
/// (`UnsupportedStatement`); the Display text is the carried string in both
/// cases. `ProgrammingError` mirrors the `sqlite3.ProgrammingError` raised when
/// the supplied bind count does not match the statement's placeholder count.
/// `Corruption` and `StorageIo` preserve the storage layer's fault class rather
/// than flattening it to a parse error — the boundary raises a dedicated Python
/// exception for corruption so a caller never mistakes disk damage for a routine
/// operational error.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum EngineError {
    /// A syntax or grammar error within the supported subset.
    #[error("{0}")]
    ParseError(String),

    /// A well-formed statement whose shape is outside the supported subset.
    #[error("{0}")]
    UnsupportedStatement(String),

    /// A bind-count or other DB-API programming fault.
    #[error("{0}")]
    ProgrammingError(String),

    /// Unrecoverable storage corruption — a distinct class the boundary raises as
    /// a dedicated Python exception so the caller can quarantine rather than
    /// silently degrade.
    #[error("storage corruption: {detail}")]
    Corruption {
        /// The class of corruption detected.
        kind: CorruptionKind,
        /// The underlying storage error's message.
        detail: String,
    },

    /// A transient storage I/O fault — recoverable, kept distinct from corruption.
    #[error("storage io error: {0}")]
    StorageIo(String),

    /// A read-only snapshot fence tripped under a concurrent checkpoint —
    /// transient and retryable. The boundary raises a dedicated OperationalError
    /// subclass so the reader's fence-retry loop catches it structurally; it must
    /// never be classified as corruption.
    #[error("{0}")]
    SnapshotFence(String),
}

impl EngineError {
    /// Construct a [`EngineError::ParseError`] from anything string-like.
    pub fn parse(msg: impl Into<String>) -> Self {
        EngineError::ParseError(msg.into())
    }

    /// Construct a [`EngineError::UnsupportedStatement`] from anything
    /// string-like.
    pub fn unsupported(msg: impl Into<String>) -> Self {
        EngineError::UnsupportedStatement(msg.into())
    }
}

/// Result alias for SQL engine operations.
pub type Result<T> = std::result::Result<T, EngineError>;

impl From<lillibrain::StoreError> for EngineError {
    fn from(e: lillibrain::StoreError) -> Self {
        use lillibrain::StoreError as S;
        let detail = e.to_string();
        match e {
            S::Io(_) => EngineError::StorageIo(detail),
            S::SnapshotFence { .. } => EngineError::SnapshotFence(detail),
            S::CrcMismatch { .. } => EngineError::Corruption {
                kind: CorruptionKind::ChecksumMismatch,
                detail,
            },
            S::FreelistCorruption { .. } => EngineError::Corruption {
                kind: CorruptionKind::Freelist,
                detail,
            },
            S::Integrity { .. } => EngineError::Corruption {
                kind: CorruptionKind::Integrity,
                detail,
            },
            S::PageOutOfBounds { .. } => EngineError::Corruption {
                kind: CorruptionKind::PageOutOfBounds,
                detail,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_error_displays_carried_string() {
        let e = EngineError::parse("expected 'FROM', got 'AS' at position 5");
        assert_eq!(format!("{e}"), "expected 'FROM', got 'AS' at position 5");
    }

    #[test]
    fn unsupported_displays_carried_string() {
        let e = EngineError::unsupported("unsupported statement type: 'DROP'");
        assert_eq!(format!("{e}"), "unsupported statement type: 'DROP'");
    }

    #[test]
    fn programming_error_displays_carried_string() {
        let e = EngineError::ProgrammingError(
            "Incorrect number of bindings supplied.".to_string(),
        );
        assert_eq!(format!("{e}"), "Incorrect number of bindings supplied.");
    }

    #[test]
    fn store_corruption_classes_are_preserved_across_the_boundary() {
        use lillibrain::StoreError;
        let cases = [
            (
                StoreError::CrcMismatch {
                    page_no: 7,
                    expected: 0xAABB_CCDD,
                    actual: 0x0011_2233,
                },
                CorruptionKind::ChecksumMismatch,
            ),
            (
                StoreError::FreelistCorruption {
                    detail: "cycle at page 5".to_string(),
                },
                CorruptionKind::Freelist,
            ),
            (
                StoreError::Integrity {
                    detail: "separator out of range".to_string(),
                },
                CorruptionKind::Integrity,
            ),
            (
                StoreError::PageOutOfBounds {
                    page_no: 99,
                    db_size: 4,
                },
                CorruptionKind::PageOutOfBounds,
            ),
        ];
        for (store_err, expected_kind) in cases {
            match EngineError::from(store_err) {
                EngineError::Corruption { kind, .. } => assert_eq!(kind, expected_kind),
                other => panic!("corruption must not flatten to {other:?}"),
            }
        }
    }

    #[test]
    fn store_io_is_recoverable_not_corruption() {
        use lillibrain::StoreError;
        let e = EngineError::from(StoreError::Io(std::io::Error::other("disk hiccup")));
        assert!(
            matches!(e, EngineError::StorageIo(_)),
            "transient io must not be classified as corruption"
        );
    }
}

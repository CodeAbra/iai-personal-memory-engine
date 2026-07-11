//! Error taxonomy for the SQL engine frontend.
//!
//! Every variant carries the verbatim user-facing message text the reference
//! engine raises, so callers (and the parity oracle) observe byte-identical SQL
//! error strings. The engine fails loud on any statement shape outside the
//! supported subset rather than silently mis-parsing it.

use thiserror::Error;

/// A SQL engine error.
///
/// `ParseError` and `UnsupportedStatement` both carry an already-formatted
/// message string. They are kept as distinct variants so a caller can tell a
/// malformed-but-in-subset statement (`ParseError`) apart from a well-formed
/// statement whose shape is deliberately outside the supported grammar
/// (`UnsupportedStatement`); the Display text is the carried string in both
/// cases. `ProgrammingError` mirrors the `sqlite3.ProgrammingError` raised when
/// the supplied bind count does not match the statement's placeholder count.
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
        EngineError::parse(format!("storage error: {e}"))
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
}

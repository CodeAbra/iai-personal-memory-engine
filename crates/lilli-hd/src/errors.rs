//! Typed errors for the hypervector kernels.
//!
//! These cover input-shape and finiteness violations on the numeric surface.
//! The bundle-capacity / saturation contract is NOT modelled here: the cap
//! check, telemetry emit, and raise stay in the host, which owns the secret
//! material for event protection. This crate is purely numeric — bytes and
//! floats in, bytes and floats out.

use std::fmt;

/// Error raised by the kernels on invalid numeric input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HvError {
    /// An input vector did not have the required length.
    Length { expected: usize, actual: usize },
    /// Two operands that must match in length did not.
    LengthMismatch { left: usize, right: usize },
    /// An embedding carried a non-finite value (NaN or infinity).
    NonFinite,
    /// A role name was not present in the role vocabulary.
    UnknownRole { role: String },
    /// A dimension had no frozen role codebook.
    UnsupportedDim { dim: usize },
}

impl fmt::Display for HvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HvError::Length { expected, actual } => {
                write!(f, "length mismatch: expected {expected}, got {actual}")
            }
            HvError::LengthMismatch { left, right } => {
                write!(f, "operands differ in length: {left} vs {right}")
            }
            HvError::NonFinite => write!(f, "input contains a non-finite value"),
            HvError::UnknownRole { role } => write!(f, "unknown role {role:?}"),
            HvError::UnsupportedDim { dim } => {
                write!(f, "no role codebook for dimension {dim}")
            }
        }
    }
}

impl std::error::Error for HvError {}

impl From<HvError> for pyo3::PyErr {
    fn from(e: HvError) -> Self {
        pyo3::exceptions::PyValueError::new_err(e.to_string())
    }
}

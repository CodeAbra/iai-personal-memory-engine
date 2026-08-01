//! Typed errors for the graph algorithm layer.
//!
//! Each variant maps to a deliberate Python exception via the
//! `impl From<GraphError> for pyo3::PyErr` block — invalid-input cases
//! become `ValueError`, missing-precondition cases become `RuntimeError`.
//! Future algorithm waves extend the enum but keep the same translation
//! contract so callers see consistent Python-side error types.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum GraphError {
    #[error("graph is empty")]
    EmptyGraph,

    #[error("graph has {component_count} disconnected components; expected 1")]
    DisconnectedComponent { component_count: usize },

    #[error("invalid node id: {0}")]
    InvalidNodeId(String),

    #[error("pyo3 conversion: {0}")]
    Pyo3Conversion(#[from] pyo3::PyErr),
}

impl From<GraphError> for pyo3::PyErr {
    fn from(err: GraphError) -> Self {
        match err {
            GraphError::EmptyGraph => pyo3::exceptions::PyRuntimeError::new_err(err.to_string()),
            GraphError::DisconnectedComponent { .. } => {
                pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
            }
            GraphError::InvalidNodeId(_) => {
                pyo3::exceptions::PyValueError::new_err(err.to_string())
            }
            GraphError::Pyo3Conversion(err) => err,
        }
    }
}

//! Public-API contract test for the shared CSR validator. Every graph kernel
//! entry point runs `validate_csr` before dereferencing neighbor ids as array
//! indices, so a malformed CSR must be rejected as a recoverable error rather
//! than panicking across the PyO3 boundary or silently adding a phantom node.

use iai_mcp_graph_core::validate_csr;

/// Triangle 0-1-2 with sorted undirected adjacency: the canonical well-formed
/// CSR the kernels expect.
fn triangle() -> ([i64; 4], [i64; 6]) {
    ([0, 2, 4, 6], [1, 2, 0, 2, 0, 1])
}

#[test]
fn accepts_well_formed_csr() {
    let (indptr, indices) = triangle();
    assert!(validate_csr(&indptr, &indices, 3).is_ok());
}

#[test]
fn accepts_empty_graph() {
    let indptr = [0i64];
    let indices: [i64; 0] = [];
    assert!(validate_csr(&indptr, &indices, 0).is_ok());
}

#[test]
fn rejects_wrong_indptr_length() {
    let (_, indices) = triangle();
    let indptr = [0i64, 2, 4]; // len 3, needs n_nodes + 1 == 4
    assert!(validate_csr(&indptr, &indices, 3).is_err());
}

#[test]
fn rejects_non_monotonic_indptr() {
    let (_, indices) = triangle();
    let indptr = [0i64, 4, 2, 6]; // decreases 4 -> 2
    assert!(validate_csr(&indptr, &indices, 3).is_err());
}

#[test]
fn rejects_tail_not_matching_indices_len() {
    let (_, indices) = triangle();
    let indptr = [0i64, 2, 4, 5]; // tail 5 != indices.len() == 6
    assert!(validate_csr(&indptr, &indices, 3).is_err());
}

#[test]
fn rejects_out_of_range_neighbor() {
    let (indptr, _) = triangle();
    let indices = [1i64, 9, 0, 2, 0, 1]; // 9 >= n_nodes == 3
    assert!(validate_csr(&indptr, &indices, 3).is_err());
}

#[test]
fn rejects_negative_neighbor() {
    let (indptr, _) = triangle();
    let indices = [1i64, -1, 0, 2, 0, 1];
    assert!(validate_csr(&indptr, &indices, 3).is_err());
}

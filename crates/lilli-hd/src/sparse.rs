//! Sparse vector-symbolic arithmetic kernel (procedural tier).
//!
//! A hypervector is a sorted list of active indices in `[0, 2048)`, capped at
//! K = 20. Binding unions, unbinding subtracts, bundling keeps the most
//! frequent indices, permuting shifts modulo the dimension. Similarity is
//! Jaccard.
//!
//! The pad path that backfills an under-K result is seeded deterministically
//! from a SHA256 of the operand index lists (never a process-salted hash), so
//! it is reproducible across processes and languages. No frozen golden and no
//! persisted data exercises the pad path in this layer; the host side is aligned
//! to this same scheme separately so a differential can cover it.

use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::codebook::seed_from_str;

/// Sparse dimension: active indices live in `[0, SPARSE_DIM)`.
pub const SPARSE_DIM: u16 = 2048;
/// Active bits per hypervector.
pub const SPARSE_K: usize = 20;

/// SHA256-seeded deterministic backfill to K active indices.
///
/// Reproducible across processes and languages. The seed is derived from the
/// sorted operand lists via the shared `seed_from_str`; a SplitMix64 walk over
/// that seed proposes candidate indices until K distinct ones are present.
pub fn pad_to_k(indices: &[u16], seed_a: &[u16], seed_b: &[u16]) -> Vec<u16> {
    use std::collections::BTreeSet;
    let mut existing: BTreeSet<u16> = indices.iter().copied().collect();
    if existing.len() >= SPARSE_K {
        return existing.into_iter().take(SPARSE_K).collect();
    }

    let mut sa = seed_a.to_vec();
    sa.sort_unstable();
    let mut sb = seed_b.to_vec();
    sb.sort_unstable();
    let operand_repr = format!("{sa:?}|{sb:?}");
    let mut state = seed_from_str("sparse-pad", &operand_repr);

    while existing.len() < SPARSE_K {
        // SplitMix64 step — deterministic stream from the SHA256 seed.
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let candidate = (z % SPARSE_DIM as u64) as u16;
        existing.insert(candidate);
    }
    existing.into_iter().take(SPARSE_K).collect()
}

fn sorted_unique(values: impl IntoIterator<Item = u16>) -> Vec<u16> {
    use std::collections::BTreeSet;
    values.into_iter().collect::<BTreeSet<_>>().into_iter().collect()
}

/// Union-bind two index lists, capped at K. Backfills if the union underflows.
pub fn bind_impl(a: &[u16], b: &[u16]) -> Vec<u16> {
    let merged = sorted_unique(a.iter().copied().chain(b.iter().copied()));
    if merged.len() >= SPARSE_K {
        return merged[..SPARSE_K].to_vec();
    }
    pad_to_k(&merged, a, b)
}

/// Set-difference unbind. Backfills if the remainder underflows.
pub fn unbind_impl(bound: &[u16], key: &[u16]) -> Vec<u16> {
    use std::collections::BTreeSet;
    let key_set: BTreeSet<u16> = key.iter().copied().collect();
    let remainder: Vec<u16> = sorted_unique(bound.iter().copied())
        .into_iter()
        .filter(|v| !key_set.contains(v))
        .collect();
    if remainder.len() >= SPARSE_K {
        return remainder[..SPARSE_K].to_vec();
    }
    pad_to_k(&remainder, bound, key)
}

/// Frequency-rank bundle: keep the K most frequent indices, ties broken by
/// ascending index, then sort the result.
pub fn bundle_impl(hvs: &[Vec<u16>]) -> Vec<u16> {
    use std::collections::BTreeMap;
    if hvs.is_empty() {
        return Vec::new();
    }
    let mut counts: BTreeMap<u16, usize> = BTreeMap::new();
    for hv in hvs {
        for &idx in hv {
            *counts.entry(idx).or_insert(0) += 1;
        }
    }
    // Rank by (-count, idx). BTreeMap iterates idx-ascending already, so a
    // stable sort by descending count preserves the index tie-break.
    let mut ranked: Vec<(u16, usize)> = counts.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut top: Vec<u16> = ranked.into_iter().take(SPARSE_K).map(|(i, _)| i).collect();
    top.sort_unstable();
    top
}

/// Circular index shift modulo the dimension, returned sorted.
pub fn permute_impl(hv: &[u16], shift: i64) -> Vec<u16> {
    let d = SPARSE_DIM as i64;
    // Normalize the shift into [0, d) first so an extreme i64 shift cannot
    // overflow the per-index `i + shift`. The result is unchanged for in-range
    // shifts.
    let s = ((shift % d) + d) % d;
    let mut out: Vec<u16> = hv
        .iter()
        .map(|&i| ((i as i64 + s) % d) as u16)
        .collect();
    out.sort_unstable();
    out
}

/// Jaccard similarity over the two index sets.
pub fn similarity_impl(a: &[u16], b: &[u16]) -> f64 {
    use std::collections::BTreeSet;
    let sa: BTreeSet<u16> = a.iter().copied().collect();
    let sb: BTreeSet<u16> = b.iter().copied().collect();
    let union = sa.union(&sb).count();
    if union == 0 {
        return 0.0;
    }
    let inter = sa.intersection(&sb).count();
    inter as f64 / union as f64
}

// --- PyO3 surface -----------------------------------------------------------

fn extract_indices(list: &Bound<'_, PyList>) -> PyResult<Vec<u16>> {
    list.iter().map(|item| item.extract::<u16>()).collect()
}

#[pyfunction]
#[pyo3(name = "sparse_bind")]
pub fn bind(a: &Bound<'_, PyList>, b: &Bound<'_, PyList>) -> PyResult<Vec<u16>> {
    Ok(bind_impl(&extract_indices(a)?, &extract_indices(b)?))
}

#[pyfunction]
#[pyo3(name = "sparse_unbind")]
pub fn unbind(
    bound: &Bound<'_, PyList>,
    key: &Bound<'_, PyList>,
) -> PyResult<Vec<u16>> {
    Ok(unbind_impl(&extract_indices(bound)?, &extract_indices(key)?))
}

#[pyfunction]
#[pyo3(name = "sparse_bundle")]
pub fn bundle(hvs: &Bound<'_, PyList>) -> PyResult<Vec<u16>> {
    let lists: Vec<Vec<u16>> = hvs
        .iter()
        .map(|item| {
            let inner = item.downcast::<PyList>()?;
            extract_indices(inner)
        })
        .collect::<PyResult<_>>()?;
    Ok(bundle_impl(&lists))
}

#[pyfunction]
#[pyo3(name = "sparse_permute")]
pub fn permute(hv: &Bound<'_, PyList>, shift: i64) -> PyResult<Vec<u16>> {
    Ok(permute_impl(&extract_indices(hv)?, shift))
}

#[pyfunction]
#[pyo3(name = "sparse_similarity")]
pub fn similarity(a: &Bound<'_, PyList>, b: &Bound<'_, PyList>) -> PyResult<f64> {
    Ok(similarity_impl(&extract_indices(a)?, &extract_indices(b)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_to_k_is_deterministic_and_complete() {
        // A union below K forces the backfill path (never hit by the ops golden,
        // whose role vectors always union to >= K).
        let indices = vec![1u16, 5, 17];
        let a = vec![1u16, 5];
        let b = vec![17u16];
        let out1 = pad_to_k(&indices, &a, &b);
        let out2 = pad_to_k(&indices, &a, &b);
        assert_eq!(out1, out2, "pad must be deterministic across calls");
        assert_eq!(out1.len(), SPARSE_K, "pad must reach exactly K indices");
        assert!(
            out1.windows(2).all(|w| w[0] < w[1]),
            "pad output must be sorted and distinct"
        );
        for i in &indices {
            assert!(out1.contains(i), "pad must retain the original indices");
        }
    }

    #[test]
    fn pad_backfill_bind_underflow_reaches_k() {
        // bind of two short lists whose union is < K must backfill to K.
        let out = bind_impl(&[1u16, 2, 3], &[3u16, 4]);
        assert_eq!(out.len(), SPARSE_K);
    }

    #[test]
    fn debug_repr_matches_python_str_list() {
        // The pad seed derives from the Debug repr of the sorted operand lists;
        // it must stay byte-identical to Python's str(sorted(list)) — brackets
        // and ", " separators — or the two backends reseed differently and the
        // procedural tier diverges silently.
        assert_eq!(format!("{:?}", Vec::<u16>::new()), "[]");
        assert_eq!(format!("{:?}", vec![5u16]), "[5]");
        assert_eq!(format!("{:?}", vec![1u16, 5, 17]), "[1, 5, 17]");
    }
}

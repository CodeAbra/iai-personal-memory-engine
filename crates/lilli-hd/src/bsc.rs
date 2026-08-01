//! Binary spatter code kernel (episodic tier).
//!
//! Bytes carry packed bits, MSB-first within each byte (the convention numpy's
//! `packbits` / `unpackbits` use). Every operation here is integer-exact and
//! matches the Python reference bit for bit.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};

use crate::errors::HvError;

/// Unpack packed bytes into a bit vector, MSB-first within each byte.
pub fn unpack_bits_msb(hv: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(hv.len() * 8);
    for &byte in hv {
        for shift in (0..8).rev() {
            out.push((byte >> shift) & 1);
        }
    }
    out
}

/// Pack a bit vector (length a multiple of 8 not required; trailing bits pad
/// with zero, matching numpy `packbits`) into bytes, MSB-first within a byte.
pub fn pack_bits_msb(bits: &[u8]) -> Vec<u8> {
    let n_bytes = bits.len().div_ceil(8);
    let mut out = vec![0u8; n_bytes];
    for (i, &bit) in bits.iter().enumerate() {
        if bit & 1 != 0 {
            out[i / 8] |= 1 << (7 - (i % 8));
        }
    }
    out
}

/// XOR two equal-length packed hypervectors. Bind and unbind are the same op
/// (XOR is its own inverse).
pub fn bind_impl(a: &[u8], b: &[u8]) -> Result<Vec<u8>, HvError> {
    if a.len() != b.len() {
        return Err(HvError::LengthMismatch {
            left: a.len(),
            right: b.len(),
        });
    }
    Ok(a.iter().zip(b).map(|(x, y)| x ^ y).collect())
}

/// Majority-vote bundle over already-bound packed hypervectors.
///
/// `d` is the bit dimension. Empty input returns `d / 8` zero bytes. The tie
/// rule is `sums * 2 >= n` — load-bearing for even `n`. This does ONLY the
/// numeric vote; the capacity cap, telemetry, and raise live in the host.
///
/// The per-bit set-count is accumulated bit-sliced over 64-bit words: each
/// input hypervector contributes its bits to per-bit-position counters held in
/// a SWAR ladder of `u64` planes (a binary-counter network). One popcount-free
/// word-wide add folds 64 bit-positions at once, so the inner loop touches the
/// packed bytes as `u64` words with no per-bit shifting. The bytes are read
/// big-endian per word so bit position `j` within a word keeps its MSB-first
/// meaning, matching the packed layout. Output is packed MSB-first in place,
/// bit-identical to the scalar per-bit vote.
pub fn bundle_impl(bound_hvs: &[Vec<u8>], d: usize) -> Result<Vec<u8>, HvError> {
    if d % 8 != 0 {
        return Err(HvError::Length {
            expected: d.next_multiple_of(8),
            actual: d,
        });
    }
    let n_bytes = d / 8;
    // Every input hypervector must be exactly n_bytes long: the word and tail
    // loops index each hv by byte offset, so a short or mismatched hv would read
    // out of bounds. Validate before any indexing.
    for hv in bound_hvs {
        if hv.len() != n_bytes {
            return Err(HvError::Length {
                expected: n_bytes,
                actual: hv.len(),
            });
        }
    }
    if bound_hvs.is_empty() {
        return Ok(vec![0u8; n_bytes]);
    }
    let n = bound_hvs.len() as u32;
    let n_words = n_bytes / 8;
    let tail_bytes = n_bytes % 8;
    let mut out = vec![0u8; n_bytes];

    // Bits needed to hold a count up to n (n itself, so width = floor(log2 n)+1).
    let bit_width = (u32::BITS - n.leading_zeros()).max(1) as usize;

    for w in 0..n_words {
        let off = w * 8;
        // planes[k] = bit k of the per-position count, across all 64 positions
        // in this word — a bit-sliced binary-counter network.
        let mut planes = vec![0u64; bit_width];
        for hv in bound_hvs {
            // Big-endian load keeps bit-position j MSB-first within the word.
            let word = u64::from_be_bytes(hv[off..off + 8].try_into().unwrap());
            // Ripple-add the 0/1 mask `word` into the counter network.
            let mut carry = word;
            for plane in planes.iter_mut() {
                let new = *plane ^ carry;
                carry &= *plane;
                *plane = new;
                if carry == 0 {
                    break;
                }
            }
        }
        let result = majority_from_planes(&planes, n);
        out[off..off + 8].copy_from_slice(&result.to_be_bytes());
    }

    // Tail bytes (when n_bytes is not a multiple of 8) use the scalar per-bit
    // vote — exact, and rarely hit (4096/8 = 512 is word-aligned).
    if tail_bytes != 0 {
        let base_byte = n_words * 8;
        for tb in 0..tail_bytes {
            let byte_idx = base_byte + tb;
            for shift in 0..8u32 {
                let mut s = 0u32;
                for hv in bound_hvs {
                    s += ((hv[byte_idx] >> (7 - shift)) & 1) as u32;
                }
                if s * 2 >= n {
                    out[byte_idx] |= 1 << (7 - shift);
                }
            }
        }
    }

    Ok(out)
}

/// Given the bit-sliced per-position counter planes (`planes[k]` = bit `k` of
/// the count for each of the 64 positions) and the total `n`, return a 64-bit
/// word whose position `j` is set iff `2 * count[j] >= n`, i.e.
/// `count[j] >= ceil(n/2)`. The unsigned `count >= T` test is evaluated
/// bit-serially from the MSB down, SWAR across all 64 lanes at once.
#[inline]
fn majority_from_planes(planes: &[u64], n: u32) -> u64 {
    // 2 * count >= n  <=>  count >= ceil(n / 2).
    let t = n.div_ceil(2);
    // gt: positions already decided count > T on the high bits inspected.
    // eq: positions still exactly equal to T on the bits inspected so far.
    let mut gt = 0u64;
    let mut eq = !0u64;
    for k in (0..planes.len()).rev() {
        let cbit = planes[k];
        let tbit_set = (t >> k) & 1 != 0;
        if tbit_set {
            // t-bit = 1: positions whose count-bit is 0 here become strictly less.
            eq &= cbit;
        } else {
            // t-bit = 0: positions whose count-bit is 1 here become strictly greater.
            gt |= eq & cbit;
            eq &= !cbit;
        }
    }
    // count >= T  <=>  decided-greater OR still-equal.
    gt | eq
}

/// Circular bit-roll. Positive `shift` moves bits toward higher index, exactly
/// like `np.roll` over the unpacked bit array. The bit length is `8 * hv.len()`.
pub fn permute_impl(hv: &[u8], shift: i64) -> Vec<u8> {
    let bits = unpack_bits_msb(hv);
    let n = bits.len();
    if n == 0 {
        return Vec::new();
    }
    let n_i = n as i64;
    // np.roll: out[i] = bits[(i - shift) mod n]
    let s = ((shift % n_i) + n_i) % n_i;
    let mut rolled = vec![0u8; n];
    for (i, dst) in rolled.iter_mut().enumerate() {
        let src = ((i as i64 - s) % n_i + n_i) % n_i;
        *dst = bits[src as usize];
    }
    pack_bits_msb(&rolled)
}

/// Hamming distance in bits via popcount of the XOR. `u64::count_ones` lowers
/// to a hardware popcount.
pub fn hamming_bits(a: &[u8], b: &[u8]) -> u32 {
    let mut acc = 0u32;
    let mut ca = a.chunks_exact(8);
    let mut cb = b.chunks_exact(8);
    for (x, y) in ca.by_ref().zip(cb.by_ref()) {
        let xa = u64::from_le_bytes(x.try_into().unwrap());
        let yb = u64::from_le_bytes(y.try_into().unwrap());
        acc += (xa ^ yb).count_ones();
    }
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        acc += (x ^ y).count_ones();
    }
    acc
}

/// Normalised Hamming distance: `ham_bits / (len * 8)`. Returns 1.0 on a length
/// mismatch and 0.0 on empty input (matching the Python reference).
pub fn hamming_impl(a: &[u8], b: &[u8]) -> f64 {
    if a.len() != b.len() {
        return 1.0;
    }
    if a.is_empty() {
        return 0.0;
    }
    let ham = hamming_bits(a, b) as f64;
    ham / (a.len() as f64 * 8.0)
}

/// Packed cosine: `(D - 2 * ham_bits) / D`, integer-exact (no float unpack).
/// Returns 0.0 on a length mismatch or empty input.
pub fn cosine_packed_impl(a: &[u8], b: &[u8]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let d = (a.len() * 8) as f64;
    let ham = hamming_bits(a, b) as f64;
    (d - 2.0 * ham) / d
}

// --- PyO3 surface -----------------------------------------------------------

#[pyfunction]
#[pyo3(name = "bsc_bind")]
pub fn bind<'py>(py: Python<'py>, a: &[u8], b: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    let out = bind_impl(a, b)?;
    Ok(PyBytes::new_bound(py, &out))
}

#[pyfunction]
#[pyo3(name = "bsc_bundle")]
pub fn bundle<'py>(
    py: Python<'py>,
    bound_hvs: &Bound<'py, PyList>,
    d: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let hvs: Vec<Vec<u8>> = bound_hvs
        .iter()
        .map(|item| item.extract::<Vec<u8>>())
        .collect::<PyResult<_>>()?;
    let out = py.allow_threads(|| bundle_impl(&hvs, d))?;
    Ok(PyBytes::new_bound(py, &out))
}

#[pyfunction]
#[pyo3(name = "bsc_permute")]
pub fn permute<'py>(py: Python<'py>, hv: &[u8], shift: i64) -> PyResult<Bound<'py, PyBytes>> {
    let out = permute_impl(hv, shift);
    Ok(PyBytes::new_bound(py, &out))
}

#[pyfunction]
#[pyo3(name = "bsc_hamming")]
pub fn hamming(a: &[u8], b: &[u8]) -> f64 {
    hamming_impl(a, b)
}

#[pyfunction]
#[pyo3(name = "bsc_cosine_packed")]
pub fn cosine_packed(a: &[u8], b: &[u8]) -> f64 {
    cosine_packed_impl(a, b)
}

//! Role / filler codebook access, backed by the frozen goldens.
//!
//! Role vocabularies are fixed and finite, so their hypervectors are frozen as
//! checked-in goldens and looked up by role index — never re-derived from an
//! RNG. Open-vocabulary filler hypervectors are produced by NumPy's PCG64
//! streams (`integers` / `choice`); reproducing those draw orders bit-for-bit
//! in Rust is not tractable, so filler generation stays in the host. The
//! ×10–100 speedup lives in the bind/bundle/permute/similarity kernels, not in
//! per-string RNG. The frozen filler-sample goldens still serve as a parity
//! oracle for any future host-side change.

use std::path::PathBuf;
use std::sync::OnceLock;

use sha2::{Digest, Sha256};

use crate::errors::HvError;

/// The fixed BSC role vocabulary, in golden-row order.
pub const BSC_ROLE_VOCABULARY: [&str; 18] = [
    "WHEN",
    "WHERE",
    "ROLE",
    "PROJECT",
    "COMMUNITY_ID",
    "TEMPORAL_POSITION",
    "ACTOR",
    "OBJECT",
    "INTENT",
    "MODALITY",
    "LANG",
    "SESSION_ID",
    "TIER",
    "VALENCE",
    "CERTAINTY",
    "SOURCE",
    "TOPIC",
    "PARENT_ID",
];

/// First 8 bytes of `sha256("{prefix}:{value}")` as a big-endian u64.
///
/// This mirrors the host seed derivation exactly and is fully deterministic
/// (no process-salted hashing).
pub fn seed_from_str(prefix: &str, value: &str) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(format!("{prefix}:{value}").as_bytes());
    let digest = hasher.finalize();
    let mut head = [0u8; 8];
    head.copy_from_slice(&digest[..8]);
    u64::from_be_bytes(head)
}

fn hdc_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("IAI_MCP_HDC_FIXTURES") {
        return PathBuf::from(dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/golden/hdc")
}

/// Load a `[rows, width]` golden once, asserting its shape, and return it split
/// into fixed-width rows.
fn load_role_table(
    cache: &'static OnceLock<Vec<Vec<u8>>>,
    name: &str,
    rows: usize,
    width: usize,
) -> &'static [Vec<u8>] {
    cache.get_or_init(|| {
        let manifest = hdc_dir().join(format!("{name}.json"));
        let g = lilli_parity::load_golden(&manifest)
            .unwrap_or_else(|e| panic!("load {name} golden: {e}"));
        assert_eq!(
            g.manifest.shape,
            vec![rows, width],
            "{name} golden shape must be [{rows}, {width}]"
        );
        g.bytes.chunks_exact(width).map(|c| c.to_vec()).collect()
    })
}

fn role_index(role: &str) -> Result<usize, HvError> {
    BSC_ROLE_VOCABULARY
        .iter()
        .position(|&r| r == role)
        .ok_or_else(|| HvError::UnknownRole {
            role: role.to_string(),
        })
}

static BSC_ROLES_4096: OnceLock<Vec<Vec<u8>>> = OnceLock::new();
static BSC_ROLES_10000: OnceLock<Vec<Vec<u8>>> = OnceLock::new();
static FHRR_ROLES: OnceLock<Vec<Vec<u8>>> = OnceLock::new();
static SPARSE_ROLES: OnceLock<Vec<Vec<u8>>> = OnceLock::new();

/// BSC role hypervector for `role` at dimension `d`, selected per-dimension
/// from the frozen golden and returned as a fixed-width row.
pub fn bsc_role_hv(role: &str, d: usize) -> Result<Vec<u8>, HvError> {
    let idx = role_index(role)?;
    let table = match d {
        4096 => load_role_table(&BSC_ROLES_4096, "bsc_roles_4096", 18, 512),
        10000 => load_role_table(&BSC_ROLES_10000, "bsc_roles_10000", 18, 1250),
        _ => return Err(HvError::UnsupportedDim { dim: d }),
    };
    Ok(table[idx].clone())
}

/// FHRR role hypervector (10000 uint8 phases) for `role`, from the frozen
/// golden.
pub fn fhrr_role_hv(role: &str) -> Result<Vec<u8>, HvError> {
    let idx = role_index(role)?;
    let table = load_role_table(&FHRR_ROLES, "fhrr_roles", 18, 10000);
    Ok(table[idx].clone())
}

/// Sparse role active-index list for `role`, decoded from the frozen packed
/// golden (40-byte little-endian u16 records, sentinel-stripped).
pub fn sparse_role_indices(role: &str) -> Result<Vec<u16>, HvError> {
    let idx = role_index(role)?;
    let table = load_role_table(&SPARSE_ROLES, "sparse_roles", 18, 40);
    Ok(unpack_sparse(&table[idx]))
}

/// Sentinel for the packed sparse layout: pad slots carry `0xFFFF`.
pub const SPARSE_SENTINEL: u16 = 0xFFFF;

/// Decode a 40-byte packed sparse row into its active-index list, dropping the
/// `0xFFFF` padding sentinels.
pub fn unpack_sparse(packed: &[u8]) -> Vec<u16> {
    packed
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .filter(|&v| v != SPARSE_SENTINEL)
        .collect()
}

/// Encode an active-index list into the 40-byte packed layout (20 little-endian
/// u16 slots, padded with the sentinel).
pub fn pack_sparse(indices: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(40);
    for i in 0..20 {
        let v = indices.get(i).copied().unwrap_or(SPARSE_SENTINEL);
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

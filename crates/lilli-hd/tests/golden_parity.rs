//! Bit-parity tests: each Rust kernel must reproduce its frozen golden exactly.
//!
//! The goldens are the parity oracle dumped from the Python reference. A
//! mismatch means the Rust kernel is wrong — never regenerate a golden to make
//! a test pass. All tests run with plain `cargo test`: the kernels exercised
//! here are the pure-Rust inner functions; no Python, no maturin.

use std::path::PathBuf;

use lilli_parity::load_golden;

fn hdc_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/golden/hdc")
}

/// Load a `[rows, width]` byte golden, asserting its shape, split into rows.
fn load_rows(name: &str, rows: usize, width: usize) -> Vec<Vec<u8>> {
    let g = load_golden(hdc_dir().join(format!("{name}.json")))
        .unwrap_or_else(|e| panic!("load {name}: {e}"));
    assert_eq!(g.manifest.shape, vec![rows, width], "{name} shape");
    g.bytes.chunks_exact(width).map(|c| c.to_vec()).collect()
}

/// Load a 1-D f64 golden, asserting its length.
fn load_f64(name: &str, len: usize) -> Vec<f64> {
    let g = load_golden(hdc_dir().join(format!("{name}.json")))
        .unwrap_or_else(|e| panic!("load {name}: {e}"));
    assert_eq!(g.manifest.shape, vec![len], "{name} shape");
    g.bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

// --- projection -------------------------------------------------------------

#[test]
fn projection_p_self_check() {
    // load_p panics if the locked sha256 drifts; reaching the assert proves it.
    let p = lilli_hd::projection::load_p();
    assert_eq!(p.len(), 384 * 10000, "P is [384,10000] f32");
}

#[test]
fn projection_embeds_the_frozen_matrix_with_no_filesystem_dependency() {
    use sha2::{Digest, Sha256};

    // The frozen matrix ships inside the binary: the embedded bytes are exactly
    // the 15,360,000-byte raw [384, 10000] f32 array, and their sha256 equals the
    // locked value. No filesystem read is involved — these bytes are compiled in.
    let bytes = lilli_hd::projection::embedded_matrix_bytes();
    assert_eq!(
        bytes.len(),
        384 * 10000 * 4,
        "embedded matrix is the exact 15,360,000-byte frozen artifact"
    );
    let digest = format!("{:x}", Sha256::digest(bytes));
    assert_eq!(
        digest,
        lilli_hd::projection::P_SHA256,
        "embedded matrix sha256 equals the locked value"
    );

    // load_p decodes the embedded bytes (default path, no override set) into the
    // full f32 matrix, self-checking the same sha256 and length on the way.
    let p = lilli_hd::projection::load_p();
    assert_eq!(
        p.len(),
        384 * 10000,
        "embedded projection decodes to [384, 10000]"
    );
}

#[test]
fn projection_bit_identical_on_100_embeddings() {
    let g_in = load_golden(hdc_dir().join("proj_100_in.json")).expect("proj_100_in");
    assert_eq!(g_in.manifest.shape, vec![100, 384]);
    let inputs: Vec<Vec<f32>> = g_in
        .bytes
        .chunks_exact(384 * 4)
        .map(|row| {
            row.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        })
        .collect();

    let expected = load_rows("proj_100", 100, 1250);

    for (i, (emb, want)) in inputs.iter().zip(&expected).enumerate() {
        let got = lilli_hd::projection::project_impl(emb)
            .unwrap_or_else(|e| panic!("project row {i}: {e}"));
        assert_eq!(&got, want, "projection mismatch on embedding {i}");
    }
}

// --- input-shape guards -----------------------------------------------------

#[test]
fn bsc_bundle_rejects_a_short_hypervector() {
    use lilli_hd::bsc;
    use lilli_hd::errors::HvError;
    // A hypervector shorter than d/8 bytes must return a typed length error, not
    // index out of bounds. d = 4096 expects 512-byte hvs; pass a 4-byte one.
    let short = vec![0u8; 4];
    let err = bsc::bundle_impl(&[short], 4096).expect_err("short hv must error");
    assert!(
        matches!(
            err,
            HvError::Length {
                expected: 512,
                actual: 4
            }
        ),
        "got {err:?}"
    );
}

#[test]
fn bsc_bundle_rejects_a_non_byte_aligned_dimension() {
    use lilli_hd::bsc;
    use lilli_hd::errors::HvError;
    // d must be a whole number of bytes (d % 8 == 0).
    let hv = vec![0u8; 1];
    let err = bsc::bundle_impl(&[hv], 7).expect_err("non-aligned d must error");
    assert!(matches!(err, HvError::Length { .. }), "got {err:?}");
}

#[test]
fn fhrr_bundle_rejects_a_short_hypervector() {
    use lilli_hd::errors::HvError;
    use lilli_hd::fhrr;
    // FHRR hvs are FHRR_DIM bytes; a short one must return a typed length error.
    let short = vec![0u8; 4];
    let full = vec![0u8; fhrr::FHRR_DIM];
    let err = fhrr::bundle_impl(&[full, short]).expect_err("short hv must error");
    assert!(
        matches!(err, HvError::Length { expected, actual: 4 } if expected == fhrr::FHRR_DIM),
        "got {err:?}"
    );
}

#[test]
fn fhrr_permute_normalizes_an_extreme_shift() {
    use lilli_hd::fhrr;
    // An extreme i64 shift must not overflow; the result equals the permute with
    // the shift reduced into [0, n) by Euclidean remainder.
    let hv: Vec<u8> = (0..64u16).map(|i| i as u8).collect();
    let n = hv.len() as i64;
    for &shift in &[i64::MIN, i64::MAX] {
        let got = fhrr::permute_impl(&hv, shift);
        let want = fhrr::permute_impl(&hv, shift.rem_euclid(n));
        assert_eq!(got, want, "fhrr permute extreme shift {shift}");
    }
}

#[test]
fn sparse_permute_normalizes_an_extreme_shift() {
    use lilli_hd::sparse;
    // The sparse dimension, not the hv length, is the modulus here.
    let hv: Vec<u16> = vec![0, 5, 17, 100, 2047];
    let d = sparse::SPARSE_DIM as i64;
    for &shift in &[i64::MIN, i64::MAX] {
        let got = sparse::permute_impl(&hv, shift);
        let want = sparse::permute_impl(&hv, shift.rem_euclid(d));
        assert_eq!(got, want, "sparse permute extreme shift {shift}");
    }
}

// --- codebooks --------------------------------------------------------------

#[test]
fn bsc_role_hv_per_dim_identical() {
    use lilli_hd::codebook::{bsc_role_hv, BSC_ROLE_VOCABULARY};

    let roles_4096 = load_rows("bsc_roles_4096", 18, 512);
    let roles_10000 = load_rows("bsc_roles_10000", 18, 1250);

    for (idx, role) in BSC_ROLE_VOCABULARY.iter().enumerate() {
        assert_eq!(
            bsc_role_hv(role, 4096).unwrap(),
            roles_4096[idx],
            "{role}@4096"
        );
        assert_eq!(
            bsc_role_hv(role, 10000).unwrap(),
            roles_10000[idx],
            "{role}@10000"
        );
    }
}

#[test]
fn fhrr_role_hv_identical() {
    use lilli_hd::codebook::{fhrr_role_hv, BSC_ROLE_VOCABULARY};
    let roles = load_rows("fhrr_roles", 18, 10000);
    for (idx, role) in BSC_ROLE_VOCABULARY.iter().enumerate() {
        assert_eq!(fhrr_role_hv(role).unwrap(), roles[idx], "fhrr role {role}");
    }
}

#[test]
fn sparse_role_indices_identical() {
    use lilli_hd::codebook::{sparse_role_indices, unpack_sparse, BSC_ROLE_VOCABULARY};
    let roles = load_rows("sparse_roles", 18, 40);
    for (idx, role) in BSC_ROLE_VOCABULARY.iter().enumerate() {
        let want = unpack_sparse(&roles[idx]);
        assert_eq!(
            sparse_role_indices(role).unwrap(),
            want,
            "sparse role {role}"
        );
    }
}

// --- BSC ops ----------------------------------------------------------------

fn bsc_bind_rows() -> Vec<Vec<u8>> {
    use lilli_hd::bsc;
    use lilli_hd::codebook::{bsc_role_hv, BSC_ROLE_VOCABULARY};
    let fillers = load_rows("bsc_fillers_sample", 510, 512);
    (0..18)
        .map(|i| {
            let role = bsc_role_hv(BSC_ROLE_VOCABULARY[i], 4096).unwrap();
            bsc::bind_impl(&role, &fillers[i % 20]).unwrap()
        })
        .collect()
}

#[test]
fn bsc_ops_byte_identical() {
    use lilli_hd::bsc;
    let ops = load_rows("bsc_ops", 25, 512);
    let binds = bsc_bind_rows();

    for i in 0..18 {
        assert_eq!(binds[i], ops[i], "bsc bind row {i}");
    }
    for (k, &n) in [3usize, 7, 10].iter().enumerate() {
        assert_eq!(
            bsc::bundle_impl(&binds[..n], 4096).unwrap(),
            ops[18 + k],
            "bsc bundle n={n}"
        );
    }
    for (k, &shift) in [1i64, 7, 33, 4095].iter().enumerate() {
        assert_eq!(
            bsc::permute_impl(&binds[0], shift),
            ops[21 + k],
            "bsc permute {shift}"
        );
    }
}

#[test]
fn bsc_bundle_d10000_tail_byte_identical() {
    // D=10000 => 1250 bytes = 156 u64 words + a 2-byte tail. This is the only
    // bundle parity case that drives the scalar tail-byte vote of the SWAR
    // kernel; the D=4096 table above is word-aligned (tail empty). A drift in
    // the tail branch fails here and nowhere else.
    use lilli_hd::bsc;
    use lilli_hd::codebook::{bsc_role_hv, BSC_ROLE_VOCABULARY};

    let want = load_rows("bsc_bundle_10000", 3, 1250);
    let n_roles = BSC_ROLE_VOCABULARY.len();
    // bound_i = bind(role_i, role_{i+1}) at D=10000 — distinct 1250-byte HVs.
    let binds: Vec<Vec<u8>> = (0..n_roles)
        .map(|i| {
            let a = bsc_role_hv(BSC_ROLE_VOCABULARY[i], 10000).unwrap();
            let b = bsc_role_hv(BSC_ROLE_VOCABULARY[(i + 1) % n_roles], 10000).unwrap();
            bsc::bind_impl(&a, &b).unwrap()
        })
        .collect();

    for (k, &n) in [3usize, 7, 10].iter().enumerate() {
        assert_eq!(
            bsc::bundle_impl(&binds[..n], 10000).unwrap(),
            want[k],
            "bsc bundle D=10000 tail vote n={n}"
        );
    }
}

#[test]
fn bsc_ops_sims_integer_exact() {
    use lilli_hd::bsc;
    let sims = load_f64("bsc_ops_sims", 34);
    let binds = bsc_bind_rows();
    for i in 0..17 {
        let ham = bsc::hamming_impl(&binds[i], &binds[i + 1]);
        let cos = bsc::cosine_packed_impl(&binds[i], &binds[i + 1]);
        assert_eq!(ham, sims[2 * i], "bsc hamming i={i}");
        assert_eq!(cos, sims[2 * i + 1], "bsc cosine i={i}");
    }
}

// --- FHRR ops ---------------------------------------------------------------

fn fhrr_bind_rows() -> Vec<Vec<u8>> {
    use lilli_hd::codebook::{fhrr_role_hv, BSC_ROLE_VOCABULARY};
    use lilli_hd::fhrr;
    let fillers = load_rows("fhrr_fillers_sample", 510, 10000);
    (0..10)
        .map(|i| {
            let role = fhrr_role_hv(BSC_ROLE_VOCABULARY[i]).unwrap();
            fhrr::bind_impl(&role, &fillers[i]).unwrap()
        })
        .collect()
}

#[test]
fn fhrr_ops_byte_identical() {
    use lilli_hd::codebook::{fhrr_role_hv, BSC_ROLE_VOCABULARY};
    use lilli_hd::fhrr;
    let ops = load_rows("fhrr_ops", 27, 10000);
    let bound = fhrr_bind_rows();

    // rows[0:10] = bind(role_i, filler_i)
    for i in 0..10 {
        assert_eq!(bound[i], ops[i], "fhrr bind row {i}");
    }
    // rows[10:20] = unbind(bound_i, role_i)
    for i in 0..10 {
        let role = fhrr_role_hv(BSC_ROLE_VOCABULARY[i]).unwrap();
        assert_eq!(
            fhrr::unbind_impl(&bound[i], &role).unwrap(),
            ops[10 + i],
            "fhrr unbind {i}"
        );
    }
    // row[20] = bundle([bound_0]) passthrough
    assert_eq!(
        fhrr::bundle_impl(&bound[..1]).unwrap(),
        ops[20],
        "fhrr single passthrough"
    );
    // rows[21:24] = bundle(bound[:n]) for n in (2,3,5)
    for (k, &n) in [2usize, 3, 5].iter().enumerate() {
        assert_eq!(
            fhrr::bundle_impl(&bound[..n]).unwrap(),
            ops[21 + k],
            "fhrr bundle n={n}"
        );
    }
    // rows[24:27] = permute(bound_0, shift) for shift in (1,13,9999)
    for (k, &shift) in [1i64, 13, 9999].iter().enumerate() {
        assert_eq!(
            fhrr::permute_impl(&bound[0], shift),
            ops[24 + k],
            "fhrr permute {shift}"
        );
    }
}

#[test]
fn fhrr_ops_sims_within_tolerance() {
    use lilli_hd::fhrr;
    let sims = load_f64("fhrr_ops_sims", 9);
    let bound = fhrr_bind_rows();
    for i in 0..9 {
        let got = fhrr::similarity_impl(&bound[i], &bound[i + 1]);
        assert!(
            (got - sims[i]).abs() < 1e-9,
            "fhrr similarity i={i}: got {got}, want {}",
            sims[i]
        );
    }
}

// --- sparse ops -------------------------------------------------------------

#[test]
fn sparse_ops_byte_identical() {
    use lilli_hd::codebook::{
        pack_sparse, sparse_role_indices, unpack_sparse, BSC_ROLE_VOCABULARY,
    };
    use lilli_hd::sparse;

    let ops = load_rows("sparse_ops", 13, 40);
    let fillers_g = load_rows("sparse_fillers_sample", 510, 40);

    let a = sparse_role_indices(BSC_ROLE_VOCABULARY[0]).unwrap();
    let b = sparse_role_indices(BSC_ROLE_VOCABULARY[1]).unwrap();
    let c = sparse_role_indices(BSC_ROLE_VOCABULARY[2]).unwrap();
    let fillers: Vec<Vec<u16>> = (0..6).map(|i| unpack_sparse(&fillers_g[i])).collect();

    // rows[0:4] = bind[(a,b),(a,b),(a,c),(b,c)]
    let binds = [(&a, &b), (&a, &b), (&a, &c), (&b, &c)];
    for (i, (x, y)) in binds.iter().enumerate() {
        let r = sparse::bind_impl(x, y);
        assert_eq!(pack_sparse(&r), ops[i], "sparse bind row {i}");
    }

    // rows[4:7] = unbind(union(a,b,c), key) for key in (a,b,c)
    let mut union3: Vec<u16> = a.iter().chain(&b).chain(&c).copied().collect();
    union3.sort_unstable();
    union3.dedup();
    for (k, key) in [&a, &b, &c].iter().enumerate() {
        let u = sparse::unbind_impl(&union3, key);
        assert_eq!(pack_sparse(&u), ops[4 + k], "sparse unbind key {k}");
    }

    // rows[7:10] = bundle([a,b,c,fillers][:n]) for n in (2,4,6)
    let mut pool: Vec<Vec<u16>> = vec![a.clone(), b.clone(), c.clone()];
    pool.extend(fillers.iter().cloned());
    for (k, &n) in [2usize, 4, 6].iter().enumerate() {
        let r = sparse::bundle_impl(&pool[..n]);
        assert_eq!(pack_sparse(&r), ops[7 + k], "sparse bundle n={n}");
    }

    // rows[10:13] = permute(a, shift) for shift in (1,17,2047)
    for (k, &shift) in [1i64, 17, 2047].iter().enumerate() {
        let r = sparse::permute_impl(&a, shift);
        assert_eq!(pack_sparse(&r), ops[10 + k], "sparse permute {shift}");
    }
}

#[test]
fn sparse_ops_sims_exact() {
    use lilli_hd::codebook::{sparse_role_indices, BSC_ROLE_VOCABULARY};
    use lilli_hd::sparse;
    let sims = load_f64("sparse_ops_sims", 4);
    let a = sparse_role_indices(BSC_ROLE_VOCABULARY[0]).unwrap();
    let b = sparse_role_indices(BSC_ROLE_VOCABULARY[1]).unwrap();
    let c = sparse_role_indices(BSC_ROLE_VOCABULARY[2]).unwrap();
    // pairs [(a,b),(a,c),(b,c),(a,a)]
    let pairs = [(&a, &b), (&a, &c), (&b, &c), (&a, &a)];
    for (i, (x, y)) in pairs.iter().enumerate() {
        let got = sparse::similarity_impl(x, y);
        assert_eq!(got, sims[i], "sparse jaccard pair {i}");
    }
}

// --- algebraic laws (cheap fuzz guards) -------------------------------------

mod laws {
    use lilli_hd::{bsc, fhrr, sparse};
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn bsc_bind_self_inverse(
            a in proptest::collection::vec(any::<u8>(), 64),
            b in proptest::collection::vec(any::<u8>(), 64),
        ) {
            let bound = bsc::bind_impl(&a, &b).unwrap();
            let back = bsc::bind_impl(&bound, &b).unwrap();
            prop_assert_eq!(back, a);
        }

        #[test]
        fn bsc_permute_round_trip(
            hv in proptest::collection::vec(any::<u8>(), 64),
            shift in 1i64..512,
        ) {
            let p = bsc::permute_impl(&hv, shift);
            let back = bsc::permute_impl(&p, -shift);
            prop_assert_eq!(back, hv);
        }

        #[test]
        fn fhrr_bind_unbind_round_trip(
            a in proptest::collection::vec(any::<u8>(), 32),
            b in proptest::collection::vec(any::<u8>(), 32),
        ) {
            let bound = fhrr::bind_impl(&a, &b).unwrap();
            let back = fhrr::unbind_impl(&bound, &b).unwrap();
            prop_assert_eq!(back, a);
        }

        #[test]
        fn sparse_permute_round_trip(
            idxs in proptest::collection::hash_set(0u16..2048, 1..20),
            shift in 1i64..2048,
        ) {
            let mut hv: Vec<u16> = idxs.into_iter().collect();
            hv.sort_unstable();
            let p = sparse::permute_impl(&hv, shift);
            let back = sparse::permute_impl(&p, -shift);
            prop_assert_eq!(back, hv);
        }
    }
}

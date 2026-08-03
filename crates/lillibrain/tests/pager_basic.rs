//! Black-box integration checks for the pager and freelist public surface.

use lillibrain::consts::PAGE_SIZE;
use lillibrain::freelist::{alloc_page, free_page, freelist_pages};
use lillibrain::pager::Pager;

fn temp_path() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("brain.lilli");
    (dir, path)
}

#[test]
fn header_roundtrips_across_reopen() {
    let (_d, path) = temp_path();
    {
        let p = Pager::open(&path).unwrap();
        assert_eq!(p.db_size().unwrap(), 3);
    }
    let p = Pager::open(&path).unwrap();
    assert_eq!(p.db_size().unwrap(), 3);
    assert_eq!(p.page_size(), PAGE_SIZE);
}

#[test]
fn page_bytes_roundtrip_after_flush_and_reopen() {
    let (_d, path) = temp_path();
    let pno;
    let payload: Vec<u8> = (0..PAGE_SIZE).map(|i| (i % 97) as u8).collect();
    {
        let p = Pager::open(&path).unwrap();
        pno = p.extend_file().unwrap();
        p.write_page(pno, &payload).unwrap();
        p.flush().unwrap();
    }
    let p = Pager::open(&path).unwrap();
    let got = p.read_page(pno).unwrap();
    // The body before the checksum slot must round-trip byte-for-byte.
    assert_eq!(&got[..PAGE_SIZE - 4], &payload[..PAGE_SIZE - 4]);
}

#[test]
fn freelist_conserves_page_count_across_churn() {
    let (_d, path) = temp_path();
    let p = Pager::open(&path).unwrap();
    let pages: Vec<u32> = (0..8).map(|_| alloc_page(&p).unwrap()).collect();
    let stable = p.db_size().unwrap();
    for &pno in &pages {
        free_page(&p, pno).unwrap();
    }
    assert_eq!(p.freelist_count().unwrap(), 8);

    let reused: std::collections::HashSet<u32> = (0..8).map(|_| alloc_page(&p).unwrap()).collect();
    let original: std::collections::HashSet<u32> = pages.into_iter().collect();
    assert_eq!(reused, original, "no leak, no double-free");
    assert_eq!(p.db_size().unwrap(), stable, "file did not grow");
}

#[test]
fn freelist_chain_walk_is_bounded() {
    let (_d, path) = temp_path();
    let p = Pager::open(&path).unwrap();
    for _ in 0..4 {
        let pno = alloc_page(&p).unwrap();
        free_page(&p, pno).unwrap();
    }
    // A healthy chain walks without error and reports the freed pages.
    let listed = freelist_pages(&p).unwrap();
    assert_eq!(listed.len(), p.freelist_count().unwrap() as usize);
}

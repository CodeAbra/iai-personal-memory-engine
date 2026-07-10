//! Trunk-page freelist over the pager.
//!
//! Freed pages are recorded on a chain of trunk pages so they can be reused
//! before the file is extended. A trunk page stores a pointer to the next
//! trunk, a count of free leaf page numbers, and the leaf page numbers
//! themselves. Allocation prefers a freed page over extending the file; the
//! trunk chain is walked with a cycle guard bounded by the file size so a
//! corrupted chain is rejected rather than looped.
//!
//! Trunk layout (big-endian, one page):
//!   `[0:4]`  next trunk page number (0 = end of chain)
//!   `[4:8]`  count of leaf page numbers stored here
//!   `[8:]`   `count` leaf page numbers, 4 bytes each

use crate::consts::{
    HDR_FIRST_FREELIST_OFFSET, HDR_FREELIST_COUNT_OFFSET, PAGE_CRC_COVERED,
};
use crate::error::{Result, StoreError};
use crate::pager::Pager;

const TRUNK_FIXED_SIZE: usize = 8;
const TRUNK_ENTRY_SIZE: usize = 4;

/// Decoded contents of a trunk page.
struct TrunkPage {
    next_trunk: u32,
    leaf_pages: Vec<u32>,
}

impl TrunkPage {
    fn decode(data: &[u8]) -> Result<TrunkPage> {
        let next_trunk = u32::from_be_bytes(data[0..4].try_into().unwrap());
        let leaf_count = u32::from_be_bytes(data[4..8].try_into().unwrap()) as usize;
        let cap = max_leaves(data.len());
        if leaf_count > cap {
            return Err(StoreError::FreelistCorruption {
                detail: format!("trunk leaf_count {leaf_count} exceeds capacity {cap}"),
            });
        }
        let mut leaf_pages = Vec::with_capacity(leaf_count);
        for i in 0..leaf_count {
            let off = TRUNK_FIXED_SIZE + i * TRUNK_ENTRY_SIZE;
            leaf_pages.push(u32::from_be_bytes(data[off..off + 4].try_into().unwrap()));
        }
        Ok(TrunkPage {
            next_trunk,
            leaf_pages,
        })
    }

    fn encode(&self, page_size: usize) -> Vec<u8> {
        let mut buf = vec![0u8; page_size];
        buf[0..4].copy_from_slice(&self.next_trunk.to_be_bytes());
        buf[4..8].copy_from_slice(&(self.leaf_pages.len() as u32).to_be_bytes());
        for (i, pno) in self.leaf_pages.iter().enumerate() {
            let off = TRUNK_FIXED_SIZE + i * TRUNK_ENTRY_SIZE;
            buf[off..off + 4].copy_from_slice(&pno.to_be_bytes());
        }
        buf
    }
}

/// Maximum leaf entries a trunk page can hold without colliding with the
/// per-page checksum slot.
fn max_leaves(page_size: usize) -> usize {
    (PAGE_CRC_COVERED.min(page_size) - TRUNK_FIXED_SIZE) / TRUNK_ENTRY_SIZE
}

/// Allocate one page number, preferring the freelist over file extension.
///
/// When the freelist is empty the file is extended. Otherwise a leaf page is
/// popped from the head trunk; when the head trunk's leaf list is empty the
/// trunk page itself is recycled and the chain advances.
pub fn alloc_page(pager: &Pager) -> Result<u32> {
    let first_trunk = pager.first_freelist_trunk()?;
    if first_trunk == 0 {
        return pager.extend_file();
    }

    let trunk_data = pager.read_page(first_trunk)?;
    let mut trunk = TrunkPage::decode(&trunk_data)?;

    if let Some(allocated) = trunk.leaf_pages.pop() {
        pager.write_page(first_trunk, &trunk.encode(pager.page_size()))?;
        let count = pager.freelist_count()?;
        pager.update_header_u32(HDR_FREELIST_COUNT_OFFSET, count.saturating_sub(1))?;
        Ok(allocated)
    } else {
        // The head trunk holds no leaves; recycle the trunk page itself.
        pager.update_header_u32(HDR_FIRST_FREELIST_OFFSET, trunk.next_trunk)?;
        let count = pager.freelist_count()?;
        pager.update_header_u32(HDR_FREELIST_COUNT_OFFSET, count.saturating_sub(1))?;
        Ok(first_trunk)
    }
}

/// Push `page_no` onto the freelist.
///
/// The page is appended to the head trunk's leaf list. When the head trunk is
/// full (or none exists) `page_no` becomes a new trunk pointing at the old head.
pub fn free_page(pager: &Pager, page_no: u32) -> Result<()> {
    let first_trunk = pager.first_freelist_trunk()?;
    let page_size = pager.page_size();
    let cap = max_leaves(page_size);

    if first_trunk == 0 {
        let new_trunk = TrunkPage {
            next_trunk: 0,
            leaf_pages: Vec::new(),
        };
        pager.write_page(page_no, &new_trunk.encode(page_size))?;
        pager.update_header_u32(HDR_FIRST_FREELIST_OFFSET, page_no)?;
        let count = pager.freelist_count()?;
        pager.update_header_u32(HDR_FREELIST_COUNT_OFFSET, count + 1)?;
        return Ok(());
    }

    let trunk_data = pager.read_page(first_trunk)?;
    let mut trunk = TrunkPage::decode(&trunk_data)?;

    if trunk.leaf_pages.len() < cap {
        trunk.leaf_pages.push(page_no);
        pager.write_page(first_trunk, &trunk.encode(page_size))?;
        let count = pager.freelist_count()?;
        pager.update_header_u32(HDR_FREELIST_COUNT_OFFSET, count + 1)?;
    } else {
        let new_trunk = TrunkPage {
            next_trunk: first_trunk,
            leaf_pages: Vec::new(),
        };
        pager.write_page(page_no, &new_trunk.encode(page_size))?;
        pager.update_header_u32(HDR_FIRST_FREELIST_OFFSET, page_no)?;
        let count = pager.freelist_count()?;
        pager.update_header_u32(HDR_FREELIST_COUNT_OFFSET, count + 1)?;
    }
    Ok(())
}

/// Collect every page number currently on the freelist chain.
///
/// Walks the trunk chain from the header's first-trunk pointer, collecting each
/// trunk page and each leaf page. The walk is bounded by the file size: a
/// repeated page or a page number beyond the file extent is rejected as
/// [`StoreError::FreelistCorruption`].
pub fn freelist_pages(pager: &Pager) -> Result<std::collections::HashSet<u32>> {
    let mut result = std::collections::HashSet::new();
    let db_size = pager.db_size()?;
    let mut trunk_no = pager.first_freelist_trunk()?;
    let mut seen = std::collections::HashSet::new();

    while trunk_no != 0 {
        if !seen.insert(trunk_no) {
            return Err(StoreError::FreelistCorruption {
                detail: format!("chain cycle: page {trunk_no} repeated"),
            });
        }
        if trunk_no > db_size {
            return Err(StoreError::FreelistCorruption {
                detail: format!("trunk page {trunk_no} exceeds db_size={db_size}"),
            });
        }
        result.insert(trunk_no);

        let trunk_data = pager.read_page(trunk_no)?;
        let trunk = TrunkPage::decode(&trunk_data)?;
        for leaf in &trunk.leaf_pages {
            if !seen.insert(*leaf) {
                return Err(StoreError::FreelistCorruption {
                    detail: format!("chain cycle: leaf page {leaf} repeated"),
                });
            }
            result.insert(*leaf);
        }
        trunk_no = trunk.next_trunk;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_db() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("brain.lilli");
        (dir, path)
    }

    #[test]
    fn freelist_empty_alloc_extends_file() {
        let (_d, path) = temp_db();
        let p = Pager::open(&path).unwrap();
        let before = p.db_size().unwrap();
        let pno = alloc_page(&p).unwrap();
        assert_eq!(pno, before + 1);
        assert_eq!(p.db_size().unwrap(), before + 1);
    }

    #[test]
    fn freelist_free_then_alloc_reuses_page_without_growth() {
        let (_d, path) = temp_db();
        let p = Pager::open(&path).unwrap();
        // Allocate a page by extending, then free it.
        let pno = alloc_page(&p).unwrap();
        let size_after_alloc = p.db_size().unwrap();
        free_page(&p, pno).unwrap();
        assert_eq!(p.freelist_count().unwrap(), 1);

        // Re-allocating must return the freed page and not grow the file.
        let reused = alloc_page(&p).unwrap();
        assert_eq!(reused, pno, "freelist must be preferred over extend");
        assert_eq!(p.db_size().unwrap(), size_after_alloc);
        assert_eq!(p.freelist_count().unwrap(), 0);
    }

    #[test]
    fn freelist_three_free_three_alloc_no_leak() {
        let (_d, path) = temp_db();
        let p = Pager::open(&path).unwrap();
        let a = alloc_page(&p).unwrap();
        let b = alloc_page(&p).unwrap();
        let c = alloc_page(&p).unwrap();
        let stable = p.db_size().unwrap();

        free_page(&p, a).unwrap();
        free_page(&p, b).unwrap();
        free_page(&p, c).unwrap();
        assert_eq!(p.freelist_count().unwrap(), 3);

        let mut got = std::collections::HashSet::new();
        got.insert(alloc_page(&p).unwrap());
        got.insert(alloc_page(&p).unwrap());
        got.insert(alloc_page(&p).unwrap());

        // All three freed pages came back, none leaked, file did not grow.
        let mut want = std::collections::HashSet::new();
        want.insert(a);
        want.insert(b);
        want.insert(c);
        assert_eq!(got, want);
        assert_eq!(p.db_size().unwrap(), stable);
        assert_eq!(p.freelist_count().unwrap(), 0);
    }

    #[test]
    fn freelist_cycle_is_detected() {
        let (_d, path) = temp_db();
        let p = Pager::open(&path).unwrap();
        // Build a self-referential trunk: a page whose next_trunk points to
        // itself, set as the head of the chain.
        let trunk_no = alloc_page(&p).unwrap();
        let self_cycle = TrunkPage {
            next_trunk: trunk_no,
            leaf_pages: Vec::new(),
        };
        p.write_page(trunk_no, &self_cycle.encode(p.page_size()))
            .unwrap();
        p.update_header_u32(HDR_FIRST_FREELIST_OFFSET, trunk_no)
            .unwrap();

        let err = freelist_pages(&p).unwrap_err();
        assert!(matches!(err, StoreError::FreelistCorruption { .. }));
    }

    #[test]
    fn trunk_decode_rejects_out_of_range_leaf_count() {
        let (_d, path) = temp_db();
        let p = Pager::open(&path).unwrap();
        // Stamp a trunk page whose declared leaf_count exceeds the page capacity.
        // A naive decoder slice-indexes past the buffer and panics; decode must
        // surface a typed FreelistCorruption instead.
        let trunk_no = alloc_page(&p).unwrap();
        let mut page = vec![0u8; p.page_size()];
        let bogus = (max_leaves(p.page_size()) + 1) as u32;
        page[4..8].copy_from_slice(&bogus.to_be_bytes());
        p.write_page(trunk_no, &page).unwrap();
        p.update_header_u32(HDR_FIRST_FREELIST_OFFSET, trunk_no)
            .unwrap();

        let err = freelist_pages(&p).unwrap_err();
        assert!(matches!(err, StoreError::FreelistCorruption { .. }));
    }
}

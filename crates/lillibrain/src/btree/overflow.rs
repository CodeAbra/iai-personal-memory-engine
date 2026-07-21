//! Overflow page chains for payloads larger than a single page.
//!
//! When a payload exceeds the spill threshold, its entire content moves to a
//! chain of freelist-allocated pages and the on-page leaf cell keeps only the
//! key, the total length, and a four-byte pointer to the first overflow page.
//!
//! Overflow page layout:
//!   `[0:4]`         next overflow page (big-endian u32; 0 = last page)
//!   `[4:USABLE_END]` payload bytes (the trailing four bytes hold the per-page
//!                    checksum and are never payload)

use crate::btree::page::USABLE_END;
use crate::consts::{OVERFLOW_DATA_OFFSET, OVERFLOW_NEXT_OFFSET, PAGE_SIZE};
use crate::error::{Result, StoreError};
use crate::freelist::{alloc_page, free_page};
use crate::pager::Pager;

/// Usable payload bytes per overflow page (the rest is the next-pointer and the
/// trailing checksum slot).
pub const OVERFLOW_USABLE: usize = USABLE_END - OVERFLOW_DATA_OFFSET;

/// Allocate an overflow chain for `payload` and return the first page number.
///
/// Allocates `ceil(len / OVERFLOW_USABLE)` pages from the freelist, writing each
/// with its next-pointer and payload slice. Fails on an empty payload.
pub fn write_overflow_chain(pager: &Pager, payload: &[u8]) -> Result<u32> {
    if payload.is_empty() {
        return Err(StoreError::Integrity {
            detail: "overflow: payload must be non-empty".to_string(),
        });
    }
    let page_count = payload.len().div_ceil(OVERFLOW_USABLE);

    // Allocate all pages first so each page can record the next page number.
    let mut page_nos: Vec<u32> = Vec::with_capacity(page_count);
    for _ in 0..page_count {
        page_nos.push(alloc_page(pager)?);
    }

    for (i, &pno) in page_nos.iter().enumerate() {
        let next = if i + 1 < page_count { page_nos[i + 1] } else { 0 };
        let start = i * OVERFLOW_USABLE;
        let chunk = &payload[start..(start + OVERFLOW_USABLE).min(payload.len())];
        let mut buf = vec![0u8; PAGE_SIZE];
        buf[OVERFLOW_NEXT_OFFSET..OVERFLOW_NEXT_OFFSET + 4].copy_from_slice(&next.to_be_bytes());
        buf[OVERFLOW_DATA_OFFSET..OVERFLOW_DATA_OFFSET + chunk.len()].copy_from_slice(chunk);
        pager.write_page(pno, &buf)?;
    }
    Ok(page_nos[0])
}

/// Reassemble the payload from an overflow chain.
///
/// Follows next-pointers from `first_page` until a zero terminator, slicing the
/// last page to exactly `total_len` bytes. Fails on an out-of-range page number,
/// a chain that ends short of `total_len`, or a cycle.
pub fn read_overflow_chain(pager: &Pager, first_page: u32, total_len: usize) -> Result<Vec<u8>> {
    let db_size = pager.db_size()?;
    let mut out = Vec::with_capacity(total_len);
    let mut current = first_page;
    let mut seen = std::collections::HashSet::new();

    while current != 0 {
        if !seen.insert(current) {
            return Err(StoreError::Integrity {
                detail: format!("overflow chain cycle at page {current}"),
            });
        }
        if current < 1 || current > db_size {
            return Err(StoreError::PageOutOfBounds {
                page_no: current,
                db_size,
            });
        }
        let page = pager.read_page(current)?;
        let next = u32::from_be_bytes(
            page[OVERFLOW_NEXT_OFFSET..OVERFLOW_NEXT_OFFSET + 4]
                .try_into()
                .unwrap(),
        );
        let remaining = total_len.saturating_sub(out.len());
        let avail = &page[OVERFLOW_DATA_OFFSET..USABLE_END];
        let take = if next == 0 { remaining.min(avail.len()) } else { avail.len() };
        // A non-last page that would carry the running total past total_len means
        // the chain is longer than the cell's declared payload length — a
        // structural mismatch. Fail loud instead of underflowing the subtraction
        // above or returning a silently truncated prefix.
        if next != 0 && out.len() + take > total_len {
            return Err(StoreError::Integrity {
                detail: format!(
                    "overflow chain from page {first_page}: exceeds declared {total_len} bytes"
                ),
            });
        }
        out.extend_from_slice(&avail[..take]);
        current = next;
    }

    if out.len() < total_len {
        return Err(StoreError::Integrity {
            detail: format!(
                "overflow chain from page {first_page}: collected {} of {total_len} bytes",
                out.len()
            ),
        });
    }
    out.truncate(total_len);
    Ok(out)
}

/// Free every page in the overflow chain rooted at `first_page`.
///
/// Reads each page's next-pointer, returns the page to the freelist, then
/// advances. Fails on a cycle.
pub fn free_overflow_chain(pager: &Pager, first_page: u32) -> Result<()> {
    let mut current = first_page;
    let mut seen = std::collections::HashSet::new();
    while current != 0 {
        if !seen.insert(current) {
            return Err(StoreError::Integrity {
                detail: format!("overflow chain cycle during free at page {current}"),
            });
        }
        let page = pager.read_page(current)?;
        let next = u32::from_be_bytes(
            page[OVERFLOW_NEXT_OFFSET..OVERFLOW_NEXT_OFFSET + 4]
                .try_into()
                .unwrap(),
        );
        free_page(pager, current)?;
        current = next;
    }
    Ok(())
}

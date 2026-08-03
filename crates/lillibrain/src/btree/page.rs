//! B-tree page binary layout: header encode/decode, cell-pointer array, cell I/O.
//!
//! All multi-byte on-disk fields are big-endian. Cell content is packed from the
//! end of the usable page area downward; the per-page checksum occupies the final
//! four bytes of every page, so the usable area ends at [`USABLE_END`] rather than
//! at the raw page size.
//!
//! Leaf page (8-byte header):
//!   `[0]`     page-type byte = leaf
//!   `[1:3]`   first freeblock (unused, zero)
//!   `[3:5]`   cell count
//!   `[5:7]`   cell content start
//!   `[7]`     fragmented bytes (unused, zero)
//!   `[8:12]`  next-leaf sibling page number (0 = no sibling)
//!   `[12:]`   cell-pointer array (cell count * 2 bytes, ascending key order)
//!   Cell: `varint(key) + varint(payload_len) + payload`
//!
//! Interior page (12-byte header):
//!   `[0]`     page-type byte = interior
//!   `[1:3]`   first freeblock (unused, zero)
//!   `[3:5]`   cell count
//!   `[5:7]`   cell content start
//!   `[7]`     fragmented bytes (unused, zero)
//!   `[8:12]`  rightmost-child page number (in the header, not a cell)
//!   `[12:]`   cell-pointer array
//!   Cell: `u32(left_child) + varint(key)`

use crate::codec::{decode_varint, encode_varint};
use crate::consts::{
    INTERIOR_HEADER_SIZE, LEAF_HEADER_SIZE, PAGE_CRC_OFFSET, PAGE_INTERIOR_TABLE, PAGE_LEAF_TABLE,
    PAGE_SIZE,
};
use crate::error::{Result, StoreError};

/// The first byte beyond the usable page area: cell content is packed downward
/// from here so it never overlaps the trailing per-page checksum slot.
pub const USABLE_END: usize = PAGE_CRC_OFFSET;

/// Byte offset of a leaf page's cell count (a big-endian u16).
pub const LEAF_NUM_CELLS_OFFSET: usize = 3;
/// Byte offset of the leaf next-sibling link (a big-endian u32, 0 = no sibling).
pub const LEAF_SIBLING_OFFSET: usize = LEAF_HEADER_SIZE;
/// Byte offset where a leaf page's cell-pointer array begins.
pub const LEAF_PTR_ARRAY_START: usize = LEAF_HEADER_SIZE + 4;
/// Byte offset where an interior page's cell-pointer array begins.
pub const INTERIOR_PTR_ARRAY_START: usize = INTERIOR_HEADER_SIZE;

const INTERIOR_CHILD_SIZE: usize = 4;

/// Which kind of B-tree page a buffer holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageKind {
    /// A leaf table page.
    Leaf,
    /// An interior table page.
    Interior,
}

/// Return the page kind from the first byte of a page buffer.
pub fn page_kind(page: &[u8]) -> Result<PageKind> {
    match page[0] {
        PAGE_LEAF_TABLE => Ok(PageKind::Leaf),
        PAGE_INTERIOR_TABLE => Ok(PageKind::Interior),
        other => Err(StoreError::Integrity {
            detail: format!("unknown page-type byte {other:#04x}"),
        }),
    }
}

fn read_u16(page: &[u8], at: usize) -> u16 {
    u16::from_be_bytes(page[at..at + 2].try_into().unwrap())
}

/// Bounds-checked big-endian u16 read. The cell-pointer array is indexed by a
/// cell count read from the page header; a malformed-but-CRC-valid header can
/// claim more cells than the page can hold, marching the pointer offset past the
/// buffer. Reading through such an offset must surface a typed error rather than
/// panic on an out-of-range slice.
fn checked_u16(page: &[u8], at: usize) -> Result<u16> {
    let bytes = page.get(at..at + 2).ok_or_else(|| StoreError::Integrity {
        detail: format!("page read of 2 bytes at {at} is out of bounds"),
    })?;
    Ok(u16::from_be_bytes(bytes.try_into().unwrap()))
}

fn read_u32(page: &[u8], at: usize) -> u32 {
    u32::from_be_bytes(page[at..at + 4].try_into().unwrap())
}

/// Bounds-checked big-endian u32 read. A cell pointer recovered from a
/// malformed-but-CRC-valid page can point past the buffer; reading through it
/// must surface a typed error rather than panic on an out-of-range slice.
fn checked_u32(page: &[u8], at: usize) -> Result<u32> {
    let bytes = page.get(at..at + 4).ok_or_else(|| StoreError::Integrity {
        detail: format!("page read of 4 bytes at {at} is out of bounds"),
    })?;
    Ok(u32::from_be_bytes(bytes.try_into().unwrap()))
}

/// Read the four-byte overflow-chain pointer of a spilled leaf cell, bounds-checked.
///
/// A spilled cell stores the first overflow page number in its four payload
/// bytes. On a corrupt page the recovered `payload_start` may point past the
/// usable area; this returns a typed error instead of panicking.
pub fn read_spilled_pointer(page: &[u8], payload_start: usize) -> Result<u32> {
    checked_u32(page, payload_start)
}

// ---------------------------------------------------------------------------
// Leaf header
// ---------------------------------------------------------------------------

/// Decoded leaf header: cell count and the lowest byte offset cell content occupies.
#[derive(Debug, Clone, Copy)]
pub struct LeafHeader {
    /// Number of cells on the page.
    pub num_cells: usize,
    /// Lowest byte offset used by cell content.
    pub cell_content_start: usize,
}

/// Read the leaf header fields from a page.
pub fn read_leaf_header(page: &[u8]) -> LeafHeader {
    LeafHeader {
        num_cells: read_u16(page, LEAF_NUM_CELLS_OFFSET) as usize,
        cell_content_start: read_u16(page, 5) as usize,
    }
}

fn write_leaf_header(page: &mut [u8], num_cells: usize, cell_content_start: usize) {
    page[0] = PAGE_LEAF_TABLE;
    page[1..3].copy_from_slice(&0u16.to_be_bytes());
    page[3..5].copy_from_slice(&(num_cells as u16).to_be_bytes());
    page[5..7].copy_from_slice(&(cell_content_start as u16).to_be_bytes());
    page[7] = 0;
}

/// Read the next-leaf sibling page number (0 = no sibling).
pub fn get_sibling(page: &[u8]) -> u32 {
    read_u32(page, LEAF_SIBLING_OFFSET)
}

/// Write the next-leaf sibling page number into a page buffer.
pub fn set_sibling(page: &mut [u8], sibling: u32) {
    page[LEAF_SIBLING_OFFSET..LEAF_SIBLING_OFFSET + 4].copy_from_slice(&sibling.to_be_bytes());
}

/// Read the cell-pointer offset for the leaf cell at `idx`, validating that it
/// lands inside the usable cell-content area. A pointer outside
/// `[LEAF_PTR_ARRAY_START, USABLE_END)` is a malformed page and is rejected with
/// a typed error rather than dereferenced into a panic.
pub fn leaf_cell_ptr(page: &[u8], idx: usize) -> Result<usize> {
    let at = LEAF_PTR_ARRAY_START
        .checked_add(idx.checked_mul(2).ok_or_else(|| StoreError::Integrity {
            detail: format!("leaf cell index {idx} pointer offset overflow"),
        })?)
        .ok_or_else(|| StoreError::Integrity {
            detail: format!("leaf cell index {idx} pointer offset overflow"),
        })?;
    let ptr = checked_u16(page, at)? as usize;
    if ptr < LEAF_PTR_ARRAY_START || ptr >= USABLE_END {
        return Err(StoreError::Integrity {
            detail: format!("leaf cell {idx} pointer {ptr} outside usable area"),
        });
    }
    Ok(ptr)
}

// ---------------------------------------------------------------------------
// Interior header
// ---------------------------------------------------------------------------

/// Decoded interior header: cell count and the rightmost-child page number.
#[derive(Debug, Clone, Copy)]
pub struct InteriorHeader {
    /// Number of separator-key cells on the page.
    pub num_cells: usize,
    /// The rightmost child page number, stored in the header.
    pub rightmost_child: u32,
}

/// Read the interior header fields from a page.
pub fn read_interior_header(page: &[u8]) -> InteriorHeader {
    InteriorHeader {
        num_cells: read_u16(page, 3) as usize,
        rightmost_child: read_u32(page, 8),
    }
}

fn write_interior_header(
    page: &mut [u8],
    num_cells: usize,
    cell_content_start: usize,
    rightmost_child: u32,
) {
    page[0] = PAGE_INTERIOR_TABLE;
    page[1..3].copy_from_slice(&0u16.to_be_bytes());
    page[3..5].copy_from_slice(&(num_cells as u16).to_be_bytes());
    page[5..7].copy_from_slice(&(cell_content_start as u16).to_be_bytes());
    page[7] = 0;
    page[8..12].copy_from_slice(&rightmost_child.to_be_bytes());
}

// ---------------------------------------------------------------------------
// Leaf cell I/O
// ---------------------------------------------------------------------------

/// An on-page leaf cell as decoded from the binary.
///
/// `payload_len` is the declared full payload length: for a spilled cell this is
/// the total length even though only a four-byte overflow pointer sits on the
/// page. `spilled` distinguishes the two storage strategies. `on_page` is the
/// exact on-page byte slice of the cell (inline payload or four-byte pointer).
#[derive(Debug, Clone)]
pub struct LeafCellRaw {
    /// The cell key.
    pub key: i64,
    /// The declared full payload length.
    pub payload_len: usize,
    /// Whether the payload spilled to an overflow chain.
    pub spilled: bool,
    /// Offset within the page where the inline payload or overflow pointer begins.
    pub payload_start: usize,
}

/// Build the on-page bytes for an inline leaf cell.
pub fn encode_inline_cell(key: i64, payload: &[u8]) -> Vec<u8> {
    let mut cell = encode_varint(key);
    cell.extend_from_slice(&encode_varint(payload.len() as i64));
    cell.extend_from_slice(payload);
    cell
}

/// Build the on-page bytes for a spilled leaf cell pointing at an overflow chain.
pub fn encode_spilled_cell(key: i64, total_len: usize, first_ovf: u32) -> Vec<u8> {
    let mut cell = encode_varint(key);
    cell.extend_from_slice(&encode_varint(total_len as i64));
    cell.extend_from_slice(&first_ovf.to_be_bytes());
    cell
}

/// Decode the framing of the leaf cell at `idx` without copying the payload.
pub fn read_leaf_cell_raw(page: &[u8], idx: usize, threshold: usize) -> Result<LeafCellRaw> {
    let ptr = leaf_cell_ptr(page, idx)?;
    let (key, kn) = decode_varint(page, ptr)?;
    let (payload_len, vn) = decode_varint(page, ptr + kn)?;
    let payload_len = payload_len as usize;
    let spilled = payload_len > threshold;
    Ok(LeafCellRaw {
        key,
        payload_len,
        spilled,
        payload_start: ptr + kn + vn,
    })
}

/// Return the exact on-page byte slice of the leaf cell at `idx`.
pub fn raw_leaf_cell_bytes(page: &[u8], idx: usize, threshold: usize) -> Result<Vec<u8>> {
    let ptr = leaf_cell_ptr(page, idx)?;
    let raw = read_leaf_cell_raw(page, idx, threshold)?;
    let on_page = if raw.spilled { 4 } else { raw.payload_len };
    let end = raw.payload_start + on_page;
    let slice = page.get(ptr..end).ok_or_else(|| StoreError::Integrity {
        detail: format!("leaf cell {idx} spans [{ptr}, {end}) past the page"),
    })?;
    Ok(slice.to_vec())
}

/// Write a leaf page from a list of pre-built raw cell byte runs (already in key
/// order), preserving the supplied sibling link. Cells are packed downward from
/// [`USABLE_END`]; the pointer array grows up from [`LEAF_PTR_ARRAY_START`].
///
/// Fails if the cells plus pointer array do not fit the usable page area.
pub fn write_leaf_from_raw(cells: &[Vec<u8>], sibling: u32) -> Result<Vec<u8>> {
    let mut page = vec![0u8; PAGE_SIZE];
    let ptr_area_end = LEAF_PTR_ARRAY_START + cells.len() * 2;
    let mut write_pos = USABLE_END;
    let mut ptrs: Vec<usize> = Vec::with_capacity(cells.len());
    for raw in cells {
        write_pos = write_pos
            .checked_sub(raw.len())
            .ok_or_else(|| StoreError::Integrity {
                detail: "leaf page overflow while packing cells".to_string(),
            })?;
        if write_pos < ptr_area_end {
            return Err(StoreError::Integrity {
                detail: "leaf page overflow while packing cells".to_string(),
            });
        }
        page[write_pos..write_pos + raw.len()].copy_from_slice(raw);
        ptrs.push(write_pos);
    }
    for (i, p) in ptrs.iter().enumerate() {
        page[LEAF_PTR_ARRAY_START + i * 2..LEAF_PTR_ARRAY_START + i * 2 + 2]
            .copy_from_slice(&(*p as u16).to_be_bytes());
    }
    let ccs = if cells.is_empty() {
        USABLE_END
    } else {
        write_pos
    };
    write_leaf_header(&mut page, cells.len(), ccs);
    set_sibling(&mut page, sibling);
    Ok(page)
}

/// Read every leaf cell key in key order.
pub fn read_leaf_keys(page: &[u8], threshold: usize) -> Result<Vec<i64>> {
    let hdr = read_leaf_header(page);
    let mut keys = Vec::with_capacity(hdr.num_cells);
    for i in 0..hdr.num_cells {
        keys.push(read_leaf_cell_raw(page, i, threshold)?.key);
    }
    Ok(keys)
}

/// On-page byte cost of a leaf cell given its key and declared payload length.
pub fn on_page_cell_size(key: i64, declared_len: usize, threshold: usize) -> usize {
    let payload_bytes = if declared_len > threshold {
        4
    } else {
        declared_len
    };
    encode_varint(key).len() + encode_varint(declared_len as i64).len() + payload_bytes
}

// ---------------------------------------------------------------------------
// Interior node I/O
// ---------------------------------------------------------------------------

/// Decoded interior node: separator keys and the child pointers between them.
///
/// `children.len() == keys.len() + 1`; the last child is the rightmost child
/// taken from the header.
#[derive(Debug, Clone)]
pub struct InteriorNode {
    /// Separator keys in ascending order.
    pub keys: Vec<i64>,
    /// Child page pointers; one more than the number of keys.
    pub children: Vec<u32>,
}

/// Read an interior node's keys and children.
pub fn read_interior_node(page: &[u8]) -> Result<InteriorNode> {
    let hdr = read_interior_header(page);
    let mut keys = Vec::with_capacity(hdr.num_cells);
    let mut children = Vec::with_capacity(hdr.num_cells + 1);
    for i in 0..hdr.num_cells {
        let at = INTERIOR_PTR_ARRAY_START
            .checked_add(i.checked_mul(2).ok_or_else(|| StoreError::Integrity {
                detail: format!("interior cell index {i} pointer offset overflow"),
            })?)
            .ok_or_else(|| StoreError::Integrity {
                detail: format!("interior cell index {i} pointer offset overflow"),
            })?;
        let ptr = checked_u16(page, at)? as usize;
        if ptr < INTERIOR_PTR_ARRAY_START || ptr + INTERIOR_CHILD_SIZE > USABLE_END {
            return Err(StoreError::Integrity {
                detail: format!("interior cell {i} pointer {ptr} outside usable area"),
            });
        }
        let left_child = checked_u32(page, ptr)?;
        let (key, _n) = decode_varint(page, ptr + INTERIOR_CHILD_SIZE)?;
        children.push(left_child);
        keys.push(key);
    }
    children.push(hdr.rightmost_child);
    Ok(InteriorNode { keys, children })
}

/// Build an interior page from keys and children. Cells pack downward from
/// [`USABLE_END`]; the rightmost child goes into the header.
///
/// Requires `children.len() == keys.len() + 1`.
pub fn write_interior_node(keys: &[i64], children: &[u32]) -> Result<Vec<u8>> {
    if children.len() != keys.len() + 1 {
        return Err(StoreError::Integrity {
            detail: format!(
                "interior node: {} children for {} keys",
                children.len(),
                keys.len()
            ),
        });
    }
    let mut page = vec![0u8; PAGE_SIZE];
    let rightmost = *children.last().unwrap();
    let ptr_area_end = INTERIOR_PTR_ARRAY_START + keys.len() * 2;
    let mut write_pos = USABLE_END;
    let mut ptrs: Vec<usize> = Vec::with_capacity(keys.len());
    for (i, key) in keys.iter().enumerate() {
        let mut cell = children[i].to_be_bytes().to_vec();
        cell.extend_from_slice(&encode_varint(*key));
        write_pos = write_pos
            .checked_sub(cell.len())
            .ok_or_else(|| StoreError::Integrity {
                detail: "interior page overflow while packing cells".to_string(),
            })?;
        if write_pos < ptr_area_end {
            return Err(StoreError::Integrity {
                detail: "interior page overflow while packing cells".to_string(),
            });
        }
        page[write_pos..write_pos + cell.len()].copy_from_slice(&cell);
        ptrs.push(write_pos);
    }
    for (i, p) in ptrs.iter().enumerate() {
        page[INTERIOR_PTR_ARRAY_START + i * 2..INTERIOR_PTR_ARRAY_START + i * 2 + 2]
            .copy_from_slice(&(*p as u16).to_be_bytes());
    }
    let ccs = if keys.is_empty() {
        USABLE_END
    } else {
        write_pos
    };
    write_interior_header(&mut page, keys.len(), ccs, rightmost);
    Ok(page)
}

/// Packed on-page byte footprint of an interior node with these keys.
pub fn interior_used_bytes(keys: &[i64]) -> usize {
    let ptr_area = INTERIOR_HEADER_SIZE + keys.len() * 2;
    let cell_bytes: usize = keys.iter().map(|k| 4 + encode_varint(*k).len()).sum();
    ptr_area + cell_bytes
}

/// Whether an interior node with these keys fits one usable page.
pub fn interior_fits(keys: &[i64]) -> bool {
    interior_used_bytes(keys) <= USABLE_END
}

/// Build an empty leaf page with no sibling.
pub fn empty_leaf() -> Vec<u8> {
    write_leaf_from_raw(&[], 0).unwrap()
}

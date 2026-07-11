//! On-disk geometry, magic, and header field offsets for the storage engine.
//!
//! All multi-byte on-disk fields are big-endian. The values here define the
//! file format: the magic that identifies a store file, the page size, the
//! 100-byte database header on page 1, the B+tree page-type bytes, the overflow
//! spill threshold, and the byte offsets of every header field.

/// File magic at offset 0 of page 1. Exactly 16 bytes. This format is its own
/// and is not byte-compatible with any other database file.
pub const DB_MAGIC: [u8; 16] = *b"LILLI storage 1\x00";

/// Bytes per page. Payloads whose length exceeds the overflow threshold spill
/// into a chain of overflow pages.
pub const PAGE_SIZE: usize = 8192;

/// Bytes reserved for the database header on page 1.
pub const HEADER_SIZE: usize = 100;

// ---------------------------------------------------------------------------
// Page-type bytes
// ---------------------------------------------------------------------------

/// Leaf table page type byte.
pub const PAGE_LEAF_TABLE: u8 = 0x0D;
/// Interior table page type byte.
pub const PAGE_INTERIOR_TABLE: u8 = 0x05;

// ---------------------------------------------------------------------------
// Page header sizes
// ---------------------------------------------------------------------------

/// Leaf page header size in bytes.
pub const LEAF_HEADER_SIZE: usize = 8;
/// Interior page header size in bytes (includes the 4-byte rightmost child pointer).
pub const INTERIOR_HEADER_SIZE: usize = 12;

// ---------------------------------------------------------------------------
// Database header field byte offsets (big-endian throughout)
// ---------------------------------------------------------------------------

/// Offset of the 16-byte magic.
pub const HDR_MAGIC_OFFSET: usize = 0;
/// Offset of the 2-byte page-size field.
pub const HDR_PAGE_SIZE_OFFSET: usize = 16;
/// Offset of the 1-byte write-version field (1 = rollback journal, 2 = write-ahead log).
pub const HDR_WRITE_VERSION_OFFSET: usize = 18;
/// Offset of the 1-byte read-version field.
pub const HDR_READ_VERSION_OFFSET: usize = 19;
/// Offset of the 4-byte total-page-count field.
pub const HDR_DB_SIZE_OFFSET: usize = 28;
/// Offset of the 4-byte first-freelist-trunk field (0 = empty).
pub const HDR_FIRST_FREELIST_OFFSET: usize = 32;
/// Offset of the 4-byte total-free-page-count field.
pub const HDR_FREELIST_COUNT_OFFSET: usize = 36;
/// Offset of the 4-byte schema-cookie field.
pub const HDR_SCHEMA_COOKIE_OFFSET: usize = 40;

// ---------------------------------------------------------------------------
// Write-version byte values
// ---------------------------------------------------------------------------

/// Write-version byte value for rollback-journal mode (the fresh-file default).
pub const WRITE_VERSION_JOURNAL: u8 = 1;
/// Write-version byte value for write-ahead-log mode.
pub const WRITE_VERSION_WAL: u8 = 2;

// ---------------------------------------------------------------------------
// Write-ahead log format
// ---------------------------------------------------------------------------

/// Big-endian checksum-variant magic in the first 4 bytes of a log file.
pub const WAL_MAGIC_BE: u32 = 0x377F_0683;
/// Format version stored in the log header.
pub const WAL_FORMAT_VERSION: u32 = 3_007_000;
/// Fixed log file header size in bytes.
pub const WAL_HEADER_SIZE: usize = 32;
/// Per-frame header size in bytes (page number, commit size, checksum pair).
pub const WAL_FRAME_HEADER_SIZE: usize = 24;
/// Total bytes per log frame: frame header plus one page.
pub const WAL_FRAME_SIZE: usize = WAL_FRAME_HEADER_SIZE + PAGE_SIZE;
/// Committed-frame count threshold that triggers an automatic checkpoint.
pub const WAL_AUTOCHECKPOINT_FRAMES: u32 = 1_000;

// ---------------------------------------------------------------------------
// Overflow page constants
// ---------------------------------------------------------------------------

/// Spill threshold: a payload longer than this moves entirely to an overflow
/// chain, leaving zero payload bytes on the leaf cell.
pub const OVERFLOW_THRESHOLD: usize = PAGE_SIZE - 35;
/// Byte offset within an overflow page of the next-page pointer (0 terminates).
pub const OVERFLOW_NEXT_OFFSET: usize = 0;
/// Byte offset within an overflow page where payload bytes begin.
pub const OVERFLOW_DATA_OFFSET: usize = 4;

// ---------------------------------------------------------------------------
// Per-page checksum
// ---------------------------------------------------------------------------

/// Width in bytes of the per-page checksum.
pub const PAGE_CRC_SIZE: usize = 4;

/// Number of leading page bytes the per-page checksum covers. The checksum
/// itself occupies the final [`PAGE_CRC_SIZE`] bytes of every page and is the
/// CRC32 of the preceding bytes; on read the trailing checksum is recomputed
/// over this many bytes and compared.
pub const PAGE_CRC_COVERED: usize = PAGE_SIZE - PAGE_CRC_SIZE;

/// Byte offset within a page where the per-page checksum is stored.
pub const PAGE_CRC_OFFSET: usize = PAGE_CRC_COVERED;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consts_magic_is_sixteen_bytes() {
        assert_eq!(DB_MAGIC.len(), 16);
        assert_eq!(&DB_MAGIC, b"LILLI storage 1\x00");
    }

    #[test]
    fn consts_page_geometry() {
        assert_eq!(PAGE_SIZE, 8192);
        assert_eq!(HEADER_SIZE, 100);
        assert_eq!(OVERFLOW_THRESHOLD, 8192 - 35);
    }

    #[test]
    fn consts_page_type_bytes() {
        assert_eq!(PAGE_LEAF_TABLE, 0x0D);
        assert_eq!(PAGE_INTERIOR_TABLE, 0x05);
        assert_eq!(LEAF_HEADER_SIZE, 8);
        assert_eq!(INTERIOR_HEADER_SIZE, 12);
    }

    #[test]
    fn consts_header_offsets_match_reference() {
        assert_eq!(HDR_MAGIC_OFFSET, 0);
        assert_eq!(HDR_PAGE_SIZE_OFFSET, 16);
        assert_eq!(HDR_WRITE_VERSION_OFFSET, 18);
        assert_eq!(HDR_READ_VERSION_OFFSET, 19);
        assert_eq!(HDR_DB_SIZE_OFFSET, 28);
        assert_eq!(HDR_FIRST_FREELIST_OFFSET, 32);
        assert_eq!(HDR_FREELIST_COUNT_OFFSET, 36);
        assert_eq!(HDR_SCHEMA_COOKIE_OFFSET, 40);
    }

    #[test]
    fn consts_crc_layout_is_trailing_four_bytes() {
        assert_eq!(PAGE_CRC_SIZE, 4);
        assert_eq!(PAGE_CRC_COVERED, PAGE_SIZE - 4);
        assert_eq!(PAGE_CRC_OFFSET, PAGE_SIZE - 4);
    }

    #[test]
    fn consts_wal_frame_geometry() {
        assert_eq!(WAL_FRAME_SIZE, WAL_FRAME_HEADER_SIZE + PAGE_SIZE);
    }

    #[test]
    fn consts_write_version_bytes() {
        assert_eq!(WRITE_VERSION_JOURNAL, 1);
        assert_eq!(WRITE_VERSION_WAL, 2);
    }
}

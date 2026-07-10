//! B-tree index over the paged file.
//!
//! The key->value index keyed by `i64`, storing opaque byte payloads. Layered
//! as page binary format, overflow chains for large values, a cursor for descent
//! and scans, and the split and delete restructuring code.

pub mod cursor;
pub mod delete;
pub mod overflow;
pub mod page;
pub mod split;

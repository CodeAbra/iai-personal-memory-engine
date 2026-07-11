//! Durable record store built on a hand-rolled, ACID embedded B-tree.
//!
//! Owns the on-disk substrate end to end: a paged file, a B-tree index with
//! tree-height insert cost, a write-ahead log for crash atomicity, and the
//! record codec that serializes stored rows. No third-party database library
//! is used; the pager, tree, log, and codec are implemented in this crate.
//!
//! Stored blobs are opaque: the engine never inspects, transforms, or
//! interprets the bytes it holds, and it holds no secret material of any kind.

pub mod btree;
pub mod check;
pub mod codec;
pub mod consts;
pub mod error;
pub mod freelist;
pub mod gates;
pub mod io;
pub mod journal;
pub mod pager;
pub mod py;
pub mod store;
pub mod wal;

pub use codec::{decode_record, decode_record_columns, encode_record, Value};
pub use error::{Result, StoreError};
pub use pager::Pager;
pub use py::register;
pub use store::{Store, Tree};

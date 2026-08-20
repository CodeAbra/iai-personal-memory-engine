//! Emits the `.pyi` stubs for the native wrapper module.
//!
//! Invocation:
//!   cargo run --bin stub_gen -p iai_mcp_native
//!
//! Output:
//!   rust/iai_mcp_native/stubs/iai_mcp_native/{__init__,embed/__init__,graph/__init__}.pyi
//!
//! Generated stubs are staged here, not written to the live installed
//! package dir; flatten them into the tracked `.pyi` copies by hand.

use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    iai_mcp_native::stub_info_staged()?.generate()?;
    Ok(())
}

//! Emits the `.pyi` stub for the native wrapper module.
//!
//! Invocation:
//!   cargo run --bin stub_gen -p iai_mcp_native
//!
//! Output:
//!   rust/iai_mcp_native/iai_mcp_native.pyi
//!
//! The generated stub is the source of truth for `mypy --strict` checks
//! against the Rust-facing Python surface. `maturin build` bundles it
//! into the wheel.

use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    let stub = iai_mcp_native::stub_info()?;
    stub.generate()?;
    Ok(())
}

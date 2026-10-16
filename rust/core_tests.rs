//! Tests of symjit-core's modules that run compiled kernels, so they need the JIT (moved from
//! the modules' `#[cfg(test)]` blocks).

mod defuns;
#[cfg(feature = "math")]
mod libm;
mod strings;
mod utils;

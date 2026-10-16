//! Typed builder for `Composer` kernels (cargo feature `composer`).
//!
//! A kernel is written as ordinary Rust over the handle types [`Real`], [`Cplx`] and [`Mask`].
//! Arithmetic operators and methods on the handles append instructions to a [`Translator`]
//! instead of computing values, so the Rust code runs once, while the kernel is built:
//! - temporaries and constants are allocated automatically;
//! - the handle types decide the real/complex flags of each instruction;
//! - comparisons return a [`Mask`], which selects values ([`Mask::select`]), branches
//!   ([`KernelBuilder::if_else`]) or converts to 1.0/0.0 ([`Mask::to_real`]);
//! - Rust loops and helper functions are unrolled and inlined into the kernel;
//! - [`KernelBuilder::recurse`] calls the kernel itself (`@self`), [`KernelBuilder::call`]
//!   another kernel (an [`Applet`](crate::Applet) of the configuration's `Defuns`), and
//!   [`KernelBuilder::callback`] a Rust function ([`add_callback`], [`add_callback2`]).
//!
//! ```
//! use symjit::kernel::{KernelBuilder, KernelCompile, Real};
//! use symjit::Config;
//!
//! let k = KernelBuilder::new(Config::default());
//! let (x, y): (Real, Real) = k.params();
//! let r = (x * x + y * y).sqrt();
//! let app = k.compile(r.lt(1.0).select(r, 1.0))?.seal()?;
//! assert_eq!(app.evaluate_single(&[0.3, 0.4]), 0.5);
//! assert_eq!(app.evaluate_single(&[3.0, 4.0]), 1.0);
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! Operators cannot return errors, so the first error is recorded and returned by
//! [`KernelBuilder::compile`].
//!
//! The attribute macro [`kernel`] (crate `symjit-macros`) writes this code from an ordinary
//! Rust function: it rewrites comparisons, `&&`/`||`, `if`/`else`, recursion and helper calls
//! into the functions of [`ops`], and keeps the function itself as plain Rust:
//!
//! ```
//! use symjit::kernel::kernel;
//!
//! #[kernel]
//! fn clamp_norm(x: f64, y: f64) -> f64 {
//!     let r = (x * x + y * y).sqrt();
//!     if r < 1.0 { r } else { 1.0 }
//! }
//!
//! let app = clamp_norm::compile(symjit::Config::default())?.seal()?;
//! assert_eq!(app.evaluate_single(&[0.3, 0.4]), clamp_norm(0.3, 0.4));
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! A kernel can call other kernels and Rust functions (`#[kernel(callback)]`); `compile`
//! compiles the kernels it calls and adds them to its configuration:
//!
//! ```
//! use symjit::kernel::kernel;
//!
//! #[kernel(callback)]
//! fn weight(x: f64) -> f64 {
//!     1.0 / (1.0 + x * x)
//! }
//!
//! #[kernel]
//! fn integrand(x: f64) -> f64 {
//!     x.cos() * weight(x)
//! }
//!
//! /// The midpoint rule with `n` points: a kernel loop that calls a kernel.
//! #[kernel]
//! fn integrate(a: f64, b: f64, n: f64) -> f64 {
//!     let h = (b - a) / n;
//!     let mut sum = 0.0;
//!     let mut i = 0.0;
//!     while i < n {
//!         sum += integrand(a + (i + 0.5) * h);
//!         i += 1.0;
//!     }
//!     sum * h
//! }
//!
//! # fn main() -> anyhow::Result<()> {
//! let app = integrate::compile(symjit::Config::default())?.seal()?;
//! assert_eq!(app.evaluate_single(&[0.0, 2.0, 50.0]), integrate(0.0, 2.0, 50.0));
//! # Ok(())
//! # }
//! ```
//!
//! Complex kernels (`Config::set_complex(true)`) hold every value as a complex number; a
//! [`Real`] is one whose imaginary part is zero. Real parameters take the real part of the
//! argument. Real functions are the complex functions applied to real arguments, and the
//! ones that leave the real domain (e.g. `ln` of a negative number, `asin` outside [-1, 1])
//! return the real part of the complex result instead of NaN; `sqrt` is NaN for negative
//! numbers in both. Complex arithmetic also changes some real
//! results: division (and `recip`) is `x conj(y) / |y|^2`, within an ulp or two of the real
//! quotient and NaN for a zero divisor, and the sign of a zero result can be lost.

// The builder is symjit-core's (`symjit_core::kernel`); this module adds what makes an
// application of it (`KernelCompile`, `compile`, `ops::add_kernel`) and the `#[kernel]` macro.

pub use symjit_core::kernel::*;

/// The attribute macro that turns a Rust function into a kernel; see [`ops`] and
/// `macros/src/lib.rs`.
pub use symjit_macros::kernel;

use crate::Application;

/// `compile` of a [`KernelBuilder`]: writes the outputs and compiles the kernel into an
/// application (`KernelBuilder::finish`, then `Application::from_finished`).
pub trait KernelCompile {
    fn compile<'k, V: Value<'k>>(&'k self, outs: V) -> Result<Application>;
}

impl KernelCompile for KernelBuilder {
    fn compile<'k, V: Value<'k>>(&'k self, outs: V) -> Result<Application> {
        let salt = self.debug_salt();
        Application::from_finished(self.finish(outs)?, salt.as_deref())
    }
}

/// `KernelBuilder::compile` as a function (the `#[kernel]` macro calls it).
pub fn compile<'k, V: Value<'k>>(k: &'k KernelBuilder, outs: V) -> Result<Application> {
    k.compile(outs)
}

/// The operations the `#[kernel]` macro rewrites Rust syntax into (`symjit_core::kernel::ops`),
/// and `add_kernel`, which compiles a called kernel.
pub mod ops {
    pub use symjit_core::kernel::ops::*;

    use anyhow::anyhow;

    use super::Result;
    use crate::{Application, Config, Defuns};

    /// Compiles the kernel `name` (`build(config, complex)`, the `__build` of its
    /// `#[kernel]` module) for a caller compiled with `config`, and adds it to `df` (once).
    /// A real kernel is compiled as a complex one when its caller is complex.
    pub fn add_kernel(
        config: &Config,
        df: &mut Defuns,
        name: &str,
        complex: bool,
        build: fn(Config, bool) -> Result<Application>,
    ) -> Result<()> {
        if df.funcs.contains_key(name) {
            return Ok(());
        }
        if complex && !config.is_complex() {
            return Err(anyhow!(
                "the complex kernel `{name}` is called from a real kernel; make the caller \
                 complex (`#[kernel(complex)]`)"
            ));
        }
        let app = build(config.clone(), config.is_complex())?.seal()?;
        df.add_applet(name, app);
        Ok(())
    }
}

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
//! use symjit::kernel::{KernelBuilder, Real};
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
use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::{
    Add, AddAssign, BitAnd, BitOr, BitXor, Div, DivAssign, Mul, MulAssign, Neg, Not, Sub, SubAssign,
};

use anyhow::anyhow;
use num_complex::Complex;

use crate::code::{BinaryFunc, Func, UnaryFunc};
use crate::{Application, Composer, Config, Defuns, Slot, Translator};

/// The result type of [`KernelBuilder::compile`] (`anyhow::Result`).
pub type Result<T> = anyhow::Result<T>;

/// The attribute macro that turns a Rust function into a kernel; see [`ops`] and
/// `macros/src/lib.rs`.
pub use symjit_macros::kernel;

struct Inner {
    translator: Translator,
    complex: bool,
    direct: bool,
    next_temp: usize,
    next_label: usize,
    num_params: usize,
    consts: HashMap<(u64, u64), usize>,
    /// the kernels of the configuration's `Defuns`
    kernels: HashMap<String, Callee>,
    /// the callbacks of the configuration's `Defuns`: number of parameters
    callbacks: HashMap<String, usize>,
    error: Option<anyhow::Error>,
}

/// The prefix of the names of callbacks in `Defuns`: the translators pass the arguments of
/// `composer_*` functions in registers, as callbacks expect them (other names get them in the
/// `__Arg` slots of kernel calls).
const CALLBACK_PREFIX: &str = "composer_";

/// Registers the Rust function `f` as the callback `name` of [`KernelBuilder::callback`].
pub fn add_callback(df: &mut Defuns, name: &str, f: UnaryFunc) {
    df.add_unary(&format!("{CALLBACK_PREFIX}{name}"), f);
}

/// Registers the Rust function `f` of two arguments as the callback `name` of
/// [`KernelBuilder::callback`].
pub fn add_callback2(df: &mut Defuns, name: &str, f: BinaryFunc) {
    df.add_binary(&format!("{CALLBACK_PREFIX}{name}"), f);
}

/// Adds the functions of `df` to those of `config` (`Config::set_defuns` can only be called
/// once); a function of `df` replaces one of the same name.
pub fn add_defuns(config: &mut Config, mut df: Defuns) {
    if df.is_empty() {
        return;
    }
    if let Some(old) = config.df.take() {
        for (name, f) in old.funcs.iter() {
            df.funcs.entry(name.clone()).or_insert_with(|| f.clone());
        }
        // the closures of `add_sliced_func` belong to `old`
        df.parents.push(old);
    }
    config.set_defuns(df);
}

/// A kernel of the configuration's `Defuns`.
#[derive(Clone, Copy)]
struct Callee {
    num_params: usize,
    complex: bool,
    /// made by `Compiler::compile`: its inputs are states, which a call does not set
    has_states: bool,
}

/// The kernels and callbacks of the configuration's `Defuns`.
fn callables(config: &Config) -> (HashMap<String, Callee>, HashMap<String, usize>) {
    let mut kernels = HashMap::new();
    let mut callbacks = HashMap::new();
    if let Some(df) = &config.df {
        for (name, f) in df.funcs.iter() {
            match (f, name.strip_prefix(CALLBACK_PREFIX)) {
                (Func::App(app), _) => {
                    // the count is of f64s, two per complex number
                    let complex = app.config.is_complex();
                    let callee = Callee {
                        num_params: app.count_params / if complex { 2 } else { 1 },
                        complex,
                        has_states: app.count_states > 0,
                    };
                    kernels.insert(name.clone(), callee);
                }
                (Func::Unary(_), Some(short)) => {
                    callbacks.insert(short.to_string(), 1);
                }
                (Func::Binary(_), Some(short)) => {
                    callbacks.insert(short.to_string(), 2);
                }
                _ => {}
            }
        }
    }
    (kernels, callbacks)
}

/// Builds one kernel; see the module documentation.
pub struct KernelBuilder {
    inner: RefCell<Inner>,
}

/// A real value of the kernel being built.
#[derive(Clone, Copy)]
pub struct Real<'k> {
    k: &'k KernelBuilder,
    slot: Slot,
}

/// A complex value of the kernel being built (complex kernels only).
#[derive(Clone, Copy)]
pub struct Cplx<'k> {
    k: &'k KernelBuilder,
    slot: Slot,
}

/// The result of a comparison: all bits set for true, zero for false.
#[derive(Clone, Copy)]
pub struct Mask<'k> {
    k: &'k KernelBuilder,
    slot: Slot,
}

impl KernelBuilder {
    /// A builder for a kernel compiled with `config`. The kernel is complex if
    /// `config.is_complex()`, and uses the direct translator if `config.direct()`.
    pub fn new(config: Config) -> KernelBuilder {
        let complex = config.is_complex();
        let direct = config.direct();
        let (kernels, callbacks) = callables(&config);
        KernelBuilder {
            inner: RefCell::new(Inner {
                translator: Translator::new(config),
                complex,
                direct,
                next_temp: 0,
                next_label: 0,
                num_params: 0,
                consts: HashMap::new(),
                kernels,
                callbacks,
                error: None,
            }),
        }
    }

    pub fn is_complex(&self) -> bool {
        self.inner.borrow().complex
    }

    /// The parameters of the kernel, in order: a `Real`, a `Cplx`, or a tuple or array of
    /// them. Calling it again declares further parameters.
    pub fn params<'k, V: Value<'k>>(&'k self) -> V {
        V::from_params(self)
    }

    /// A real parameter: the next one in order.
    pub fn param(&self) -> Real<'_> {
        self.params()
    }

    /// A complex parameter: the next one in order.
    pub fn param_complex(&self) -> Cplx<'_> {
        self.params()
    }

    /// A real constant.
    pub fn constant(&self, val: f64) -> Real<'_> {
        Real {
            k: self,
            slot: self.const_slot(Complex::new(val, 0.0)),
        }
    }

    /// A complex constant.
    pub fn constant_complex(&self, val: Complex<f64>) -> Cplx<'_> {
        self.check_complex("a complex constant");
        Cplx {
            k: self,
            slot: self.const_slot(val),
        }
    }

    /// A mask constant: all bits set for `true`, zero for `false`.
    pub fn mask(&self, val: bool) -> Mask<'_> {
        let zero = self.constant(0.0);
        if val {
            zero.eq(0.0)
        } else {
            zero.ne(0.0)
        }
    }

    /// `re + i im`.
    pub fn complex<'k>(&'k self, re: impl IntoReal<'k>, im: impl IntoReal<'k>) -> Cplx<'k> {
        self.check_complex("`complex`");
        let (re, im) = (re.into_real(self).slot, im.into_real(self).slot);
        Cplx {
            k: self,
            slot: self.fun("complex", &[re, im]),
        }
    }

    /// Runs `then` or `otherwise` depending on `cond`, and returns the value of the branch
    /// that ran. Only the instructions of that branch are executed, so a branch can recurse.
    pub fn if_else<'k, V: Value<'k>>(
        &'k self,
        cond: Mask<'k>,
        then: impl FnOnce() -> V,
        otherwise: impl FnOnce() -> V,
    ) -> V {
        let label_else = self.new_label();
        let label_end = self.new_label();

        self.emit(|c| c.append_if_else(&cond.slot, label_else));
        let t = self.copy_slots(&then());
        self.emit(|c| c.append_goto(label_end));
        self.emit(|c| c.append_label(label_else));
        let e = self.copy_slots(&otherwise());
        self.emit(|c| c.append_label(label_end));

        // a Φ-function per component of the value
        let joined: Vec<Slot> = t
            .iter()
            .zip(e.iter())
            .map(|(t, e)| {
                let dst = self.new_temp();
                self.emit(|c| c.append_join(&dst, &cond.slot, t, e));
                dst
            })
            .collect();

        V::from_slots(self, &mut joined.into_iter())
    }

    /// A loop of the kernel: `state` starts as `init` and is replaced by `body(state)` while
    /// `cond(state)` is true; returns the final state. The condition is computed before the
    /// first iteration and after each one (so the code of `cond` appears twice).
    ///
    /// The state lives in fixed temporaries, which only the direct translator supports
    /// (`Config::set_direct(true)`): the indirect one renames every assignment.
    pub fn while_loop<'k, V: Value<'k>>(
        &'k self,
        init: V,
        cond: impl Fn(V) -> Mask<'k>,
        body: impl Fn(V) -> V,
    ) -> V {
        if !self.inner.borrow().direct {
            self.fail(
                "a kernel loop needs the direct translator (Config::set_direct(true))".to_string(),
            );
        }

        // the state in fixed temporaries
        let fixed = self.copy_slots(&init);
        let state = || V::from_slots(self, &mut fixed.clone().into_iter());

        let label_top = self.new_label();
        let label_end = self.new_label();

        let c = cond(state());
        self.emit(|t| t.append_if_else(&c.slot, label_end));
        self.emit(|t| t.append_label(label_top));

        // the new state is copied before it is stored, since it can read the old one
        // (e.g. `(a, b) = (b, a + b)`)
        let next = self.copy_slots(&body(state()));
        for (dst, src) in fixed.iter().zip(next.iter()) {
            self.emit(|t| t.append_assign(dst, src));
        }

        // `append_if_else` jumps when its condition is zero: back to the top while `cond`
        let c = !cond(state());
        self.emit(|t| t.append_if_else(&c.slot, label_top));
        self.emit(|t| t.append_label(label_end));

        state()
    }

    /// Calls the kernel itself with `args`, which must match its parameters, and returns
    /// its (first) output. Native recursion is not depth-checked: a recursion that does
    /// not end overflows the machine stack.
    pub fn recurse<'k, A: Value<'k>, R: Scalar<'k>>(&'k self, args: A) -> R {
        let mut slots = Vec::new();
        args.slots(&mut slots);
        R::from_slot(self, self.fun("@self", &slots))
    }

    /// Calls the kernel `name`, an [`Applet`](crate::Applet) of the configuration's `Defuns`
    /// (`Defuns::add_applet`), with `args`, which must match its parameters, and returns its
    /// (first) output. Both kernels must be real or both complex. The SIMD kernel calls the
    /// SIMD kernel of the callee; if the callee has none (e.g. compiled without SIMD), the
    /// caller is compiled without SIMD too.
    pub fn call<'k, A: Value<'k>, R: Value<'k>>(&'k self, name: &str, args: A) -> R {
        let mut slots = Vec::new();
        args.slots(&mut slots);
        let found = self.inner.borrow().kernels.get(name).copied();
        match found {
            None => self.fail(format!(
                "`{name}` is not a kernel of the configuration (`Defuns::add_applet`)"
            )),
            Some(c) if c.has_states => self.fail(format!(
                "`{name}` is an applet of `Compiler::compile`, whose inputs are states; only \
                 kernels with parameters (`Translator`, `KernelBuilder`) can be called"
            )),
            Some(c) if c.complex != self.is_complex() => self.fail(format!(
                "the {} kernel `{name}` is called from a {} kernel; both must be real or both \
                 complex",
                if c.complex { "complex" } else { "real" },
                if c.complex { "real" } else { "complex" },
            )),
            Some(c) if c.num_params != slots.len() => self.fail(format!(
                "the kernel `{name}` takes {} arguments, not {}",
                c.num_params,
                slots.len()
            )),
            _ => {}
        }
        let dst = self.fun(name, &slots);
        self.one_value(name, dst)
    }

    /// Calls the Rust function registered as `name` by [`add_callback`] or [`add_callback2`]
    /// with one or two real arguments (`x` or `(x, y)`). In complex kernels it receives the
    /// real parts.
    pub fn callback<'k, A: RealArgs<'k>>(&'k self, name: &str, args: A) -> Real<'k> {
        let mut slots = Vec::new();
        args.slots(&mut slots);
        let found = self.inner.borrow().callbacks.get(name).copied();
        match found {
            None => self.fail(format!(
                "`{name}` is not a callback of the configuration (`kernel::add_callback`)"
            )),
            Some(n) if n != slots.len() => self.fail(format!(
                "the callback `{name}` takes {n} arguments, not {}",
                slots.len()
            )),
            _ => {}
        }
        let dst = self.new_temp();
        let name = format!("{CALLBACK_PREFIX}{name}");
        self.emit(|c| c.append_fun(&dst, &name, &slots, false));
        Real { k: self, slot: dst }
    }

    /// The result of the call of `name` in `dst` as an `R`, which must be a single value
    /// (a call returns the first output of the kernel).
    fn one_value<'k, R: Value<'k>>(&'k self, name: &str, dst: Slot) -> R {
        let r = R::from_slots(self, &mut std::iter::repeat(dst));
        let mut slots = Vec::new();
        r.slots(&mut slots);
        if slots.len() != 1 {
            self.fail(format!(
                "a call of the kernel `{name}` returns one value, not {}",
                slots.len()
            ));
        }
        r
    }

    /// Writes `outs` (a value, tuple or array) to the outputs of the kernel and compiles it.
    pub fn compile<'k, V: Value<'k>>(&'k self, outs: V) -> Result<Application> {
        let mut slots = Vec::new();
        outs.slots(&mut slots);
        for (i, s) in slots.iter().enumerate() {
            self.emit(|c| c.append_assign(&Slot::Out(i), s));
        }

        let mut inner = self.inner.borrow_mut();
        if let Some(err) = inner.error.take() {
            return Err(err);
        }
        let num_params = inner.num_params;
        inner.translator.set_num_params(num_params);
        inner.translator.compile()
    }

    // the plumbing

    fn emit(&self, f: impl FnOnce(&mut Translator) -> Result<()>) {
        let mut inner = self.inner.borrow_mut();
        if inner.error.is_none() {
            if let Err(err) = f(&mut inner.translator) {
                inner.error = Some(err);
            }
        }
    }

    fn fail(&self, msg: String) {
        let mut inner = self.inner.borrow_mut();
        if inner.error.is_none() {
            inner.error = Some(anyhow!(msg));
        }
    }

    fn check_complex(&self, what: &str) {
        if !self.is_complex() {
            self.fail(format!(
                "{what} needs a complex kernel (Config::set_complex)"
            ));
        }
    }

    fn check_same(&self, other: &KernelBuilder) {
        if !std::ptr::eq(self, other) {
            self.fail("a value of another kernel is used".to_string());
        }
    }

    fn new_temp(&self) -> Slot {
        let mut inner = self.inner.borrow_mut();
        inner.next_temp += 1;
        Slot::Temp(inner.next_temp - 1)
    }

    fn new_label(&self) -> usize {
        let mut inner = self.inner.borrow_mut();
        inner.next_label += 1;
        inner.next_label
    }

    fn new_param(&self) -> Slot {
        let mut inner = self.inner.borrow_mut();
        inner.num_params += 1;
        Slot::Param(inner.num_params - 1)
    }

    fn const_slot(&self, val: Complex<f64>) -> Slot {
        let key = (val.re.to_bits(), val.im.to_bits());
        let found = self.inner.borrow().consts.get(&key).copied();
        let idx = match found {
            Some(idx) => idx,
            None => {
                let mut inner = self.inner.borrow_mut();
                match inner.translator.append_constant(val) {
                    Ok(idx) => {
                        inner.consts.insert(key, idx);
                        idx
                    }
                    Err(err) => {
                        inner.error.get_or_insert(err);
                        0
                    }
                }
            }
        };
        Slot::Const(idx)
    }

    /// `op(args)` into a new temporary. The real flag is never set: it would wrap the
    /// arguments in `real()` (indirect translator) and mark parameters as real for the
    /// whole kernel; real values are complex numbers with a zero imaginary part instead.
    fn fun(&self, op: &str, args: &[Slot]) -> Slot {
        let dst = self.new_temp();
        self.emit(|c| c.append_fun(&dst, op, args, false));
        dst
    }

    /// Like `fun`, but for a real function that can leave the real domain: in complex
    /// kernels, the real part of the result.
    fn real_fun(&self, op: &str, args: &[Slot]) -> Slot {
        let s = self.fun(op, args);
        if self.is_complex() {
            self.fun("real", &[s])
        } else {
            s
        }
    }

    fn add(&self, a: Slot, b: Slot) -> Slot {
        let dst = self.new_temp();
        self.emit(|c| c.append_add(&dst, &[a, b], 0));
        dst
    }

    fn mul(&self, a: Slot, b: Slot) -> Slot {
        let dst = self.new_temp();
        self.emit(|c| c.append_mul(&dst, &[a, b], 0));
        dst
    }

    fn powi(&self, a: Slot, p: i64) -> Slot {
        let dst = self.new_temp();
        self.emit(|c| c.append_pow(&dst, &a, p, false));
        dst
    }

    fn powf(&self, a: Slot, p: Slot) -> Slot {
        let dst = self.new_temp();
        self.emit(|c| c.append_powf(&dst, &a, &p, false));
        dst
    }

    /// Copies the slots of `v` into new temporaries (the values of a branch, so that the
    /// join reads values set inside it).
    fn copy_slots<'k, V: Value<'k>>(&'k self, v: &V) -> Vec<Slot> {
        let mut slots = Vec::new();
        v.slots(&mut slots);
        slots
            .into_iter()
            .map(|s| {
                let dst = self.new_temp();
                self.emit(|c| c.append_assign(&dst, &s));
                dst
            })
            .collect()
    }
}

/// A value of the kernel made of slots: `Real`, `Cplx`, `Mask`, and tuples and arrays of
/// them. Used for parameters, outputs, branch values and the arguments of `recurse`.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a kernel value",
    note = "kernel loops, kernel branches and outputs hold `Real`, `Cplx` and `Mask` values \
            (`f64`, `Complex<f64>`, `bool` in a `#[kernel]` function), or tuples or arrays of \
            them; a counter updated in a kernel loop must be a float (`let mut i = 0.0`)"
)]
pub trait Value<'k>: Sized {
    fn slots(&self, out: &mut Vec<Slot>);
    fn from_slots(k: &'k KernelBuilder, slots: &mut dyn Iterator<Item = Slot>) -> Self;
    fn from_params(k: &'k KernelBuilder) -> Self;
}

/// A single value: `Real` or `Cplx`, what a (recursive) kernel call returns.
pub trait Scalar<'k>: Value<'k> {
    fn from_slot(k: &'k KernelBuilder, slot: Slot) -> Self;
}

impl<'k> Value<'k> for Real<'k> {
    fn slots(&self, out: &mut Vec<Slot>) {
        out.push(self.slot);
    }

    fn from_slots(k: &'k KernelBuilder, slots: &mut dyn Iterator<Item = Slot>) -> Self {
        Real {
            k,
            slot: slots.next().expect("too few slots"),
        }
    }

    fn from_params(k: &'k KernelBuilder) -> Self {
        let p = k.new_param();
        let slot = if k.is_complex() {
            k.fun("real", &[p])
        } else {
            p
        };
        Real { k, slot }
    }
}

impl<'k> Scalar<'k> for Real<'k> {
    fn from_slot(k: &'k KernelBuilder, slot: Slot) -> Self {
        Real { k, slot }
    }
}

impl<'k> Value<'k> for Cplx<'k> {
    fn slots(&self, out: &mut Vec<Slot>) {
        out.push(self.slot);
    }

    fn from_slots(k: &'k KernelBuilder, slots: &mut dyn Iterator<Item = Slot>) -> Self {
        Cplx {
            k,
            slot: slots.next().expect("too few slots"),
        }
    }

    fn from_params(k: &'k KernelBuilder) -> Self {
        k.check_complex("a complex parameter");
        Cplx {
            k,
            slot: k.new_param(),
        }
    }
}

impl<'k> Scalar<'k> for Cplx<'k> {
    fn from_slot(k: &'k KernelBuilder, slot: Slot) -> Self {
        k.check_complex("a complex value");
        Cplx { k, slot }
    }
}

impl<'k> Value<'k> for () {
    fn slots(&self, _: &mut Vec<Slot>) {}

    fn from_slots(_: &'k KernelBuilder, _: &mut dyn Iterator<Item = Slot>) -> Self {}

    fn from_params(_: &'k KernelBuilder) -> Self {}
}

impl<'k> Value<'k> for Mask<'k> {
    fn slots(&self, out: &mut Vec<Slot>) {
        out.push(self.slot);
    }

    fn from_slots(k: &'k KernelBuilder, slots: &mut dyn Iterator<Item = Slot>) -> Self {
        Mask {
            k,
            slot: slots.next().expect("too few slots"),
        }
    }

    fn from_params(k: &'k KernelBuilder) -> Self {
        k.fail("a mask cannot be a parameter".to_string());
        Mask {
            k,
            slot: k.new_param(),
        }
    }
}

impl<'k, V: Value<'k>, const N: usize> Value<'k> for [V; N] {
    fn slots(&self, out: &mut Vec<Slot>) {
        for v in self.iter() {
            v.slots(out);
        }
    }

    fn from_slots(k: &'k KernelBuilder, slots: &mut dyn Iterator<Item = Slot>) -> Self {
        std::array::from_fn(|_| V::from_slots(k, slots))
    }

    fn from_params(k: &'k KernelBuilder) -> Self {
        std::array::from_fn(|_| V::from_params(k))
    }
}

macro_rules! impl_value_tuple {
    ($($t:ident),+) => {
        impl<'k, $($t: Value<'k>),+> Value<'k> for ($($t,)+) {
            #[allow(non_snake_case)]
            fn slots(&self, out: &mut Vec<Slot>) {
                let ($($t,)+) = self;
                $($t.slots(out);)+
            }

            fn from_slots(k: &'k KernelBuilder, slots: &mut dyn Iterator<Item = Slot>) -> Self {
                ($($t::from_slots(k, slots),)+)
            }

            fn from_params(k: &'k KernelBuilder) -> Self {
                ($($t::from_params(k),)+)
            }
        }
    };
}

impl_value_tuple!(A);
impl_value_tuple!(A, B);
impl_value_tuple!(A, B, C);
impl_value_tuple!(A, B, C, D);
impl_value_tuple!(A, B, C, D, E);
impl_value_tuple!(A, B, C, D, E, F);
impl_value_tuple!(A, B, C, D, E, F, G);
impl_value_tuple!(A, B, C, D, E, F, G, H);

/// The arguments of a callback: one or two `Real`s (`x`, `(x,)` or `(x, y)`).
pub trait RealArgs<'k> {
    fn slots(&self, out: &mut Vec<Slot>);
}

impl<'k> RealArgs<'k> for Real<'k> {
    fn slots(&self, out: &mut Vec<Slot>) {
        out.push(self.slot);
    }
}

impl<'k> RealArgs<'k> for (Real<'k>,) {
    fn slots(&self, out: &mut Vec<Slot>) {
        out.push(self.0.slot);
    }
}

impl<'k> RealArgs<'k> for (Real<'k>, Real<'k>) {
    fn slots(&self, out: &mut Vec<Slot>) {
        out.push(self.0.slot);
        out.push(self.1.slot);
    }
}

/// A real operand: a `Real` or a number (a constant of the kernel).
pub trait IntoReal<'k> {
    fn into_real(self, k: &'k KernelBuilder) -> Real<'k>;
}

impl<'k> IntoReal<'k> for Real<'k> {
    fn into_real(self, k: &'k KernelBuilder) -> Real<'k> {
        k.check_same(self.k);
        self
    }
}

impl<'k> IntoReal<'k> for f64 {
    fn into_real(self, k: &'k KernelBuilder) -> Real<'k> {
        k.constant(self)
    }
}

/// A complex operand: a `Cplx`, a `Real`, or a number (a constant of the kernel).
pub trait IntoCplx<'k> {
    fn into_cplx(self, k: &'k KernelBuilder) -> Cplx<'k>;
}

impl<'k> IntoCplx<'k> for Cplx<'k> {
    fn into_cplx(self, k: &'k KernelBuilder) -> Cplx<'k> {
        k.check_same(self.k);
        self
    }
}

impl<'k> IntoCplx<'k> for Real<'k> {
    fn into_cplx(self, k: &'k KernelBuilder) -> Cplx<'k> {
        k.check_same(self.k);
        self.to_complex()
    }
}

impl<'k> IntoCplx<'k> for f64 {
    fn into_cplx(self, k: &'k KernelBuilder) -> Cplx<'k> {
        k.constant(self).to_complex()
    }
}

impl<'k> IntoCplx<'k> for Complex<f64> {
    fn into_cplx(self, k: &'k KernelBuilder) -> Cplx<'k> {
        k.constant_complex(self)
    }
}

macro_rules! real_unary {
    ($($(#[$doc:meta])* $name:ident => $op:literal),* $(,)?) => {
        $(
            $(#[$doc])*
            pub fn $name(self) -> Real<'k> {
                Real { k: self.k, slot: self.k.fun($op, &[self.slot]) }
            }
        )*
    };
}

macro_rules! real_unary_domain {
    ($($(#[$doc:meta])* $name:ident => $op:literal),* $(,)?) => {
        $(
            $(#[$doc])*
            pub fn $name(self) -> Real<'k> {
                Real { k: self.k, slot: self.k.real_fun($op, &[self.slot]) }
            }
        )*
    };
}

impl<'k> Real<'k> {
    /// The value as a complex number (complex kernels only).
    pub fn to_complex(self) -> Cplx<'k> {
        self.k.check_complex("a complex value");
        Cplx {
            k: self.k,
            slot: self.slot,
        }
    }

    real_unary! {
        abs => "abs",
        recip => "recip",
        floor => "floor",
        ceil => "ceiling",
        round_ties_even => "round",
        trunc => "trunc",
        fract => "frac",
        /// 2^self for an integer self in [-1022, 1023], built from the bits of a double
        /// (`Expr::pow2i`).
        pow2i => "pow2i",
        /// The exponent of self: floor(log2(|self|)) for normal numbers (`Expr::getexp`).
        getexp => "getexp",
        /// The mantissa of self in [1, 2), with the sign of self (`Expr::getmant`).
        getmant => "getmant",
        exp => "exp",
        exp2 => "exp2",
        exp_m1 => "expm1",
        sin => "sin",
        cos => "cos",
        tan => "tan",
        atan => "arctan",
        sinh => "sinh",
        cosh => "cosh",
        tanh => "tanh",
        asinh => "arcsinh",
        erf => "erf",
        erfc => "erfc",
    }

    real_unary_domain! {
        ln => "ln",
        log10 => "log10",
        log2 => "log2",
        ln_1p => "log1p",
        asin => "arcsin",
        acos => "arccos",
        acosh => "arccosh",
        atanh => "arctanh",
    }

    /// The real cube root (in complex kernels, of `|x|` with the sign of `x`: the principal
    /// complex root of a negative number is not real).
    pub fn cbrt(self) -> Real<'k> {
        if self.k.is_complex() {
            let c = Real {
                k: self.k,
                slot: self.k.real_fun("cbrt", &[self.abs().slot]),
            };
            self.lt(0.0).select(-c, c)
        } else {
            Real {
                k: self.k,
                slot: self.k.fun("cbrt", &[self.slot]),
            }
        }
    }

    /// Rounds half-way cases away from zero, like `f64::round` (symjit's `round` is
    /// `round_ties_even`).
    pub fn round(self) -> Real<'k> {
        let t = self.trunc();
        let step = self.lt(0.0).select(-1.0, 1.0);
        (self - t).abs().ge(0.5).select(t + step, t)
    }

    /// The square root (NaN for negative numbers, also in complex kernels).
    pub fn sqrt(self) -> Real<'k> {
        Real {
            k: self.k,
            slot: self.k.fun("real_root", &[self.slot]),
        }
    }

    pub fn powi(self, p: i32) -> Real<'k> {
        Real {
            k: self.k,
            slot: self.k.powi(self.slot, p as i64),
        }
    }

    pub fn powf(self, p: impl IntoReal<'k>) -> Real<'k> {
        let p = p.into_real(self.k).slot;
        let s = self.k.powf(self.slot, p);
        let slot = if self.k.is_complex() {
            self.k.fun("real", &[s])
        } else {
            s
        };
        Real { k: self.k, slot }
    }

    /// `self - floor(self / y) * y`, as symjit computes it: the sign of `y`, like Python's
    /// `%`. Rust's `%` (an exact remainder with the sign of `self`) has no counterpart, so
    /// `Real` does not implement `Rem`.
    pub fn modulo(self, y: impl IntoReal<'k>) -> Real<'k> {
        let y = y.into_real(self.k).slot;
        Real {
            k: self.k,
            slot: self.k.fun("rem", &[self.slot, y]),
        }
    }

    pub fn min(self, other: impl IntoReal<'k>) -> Real<'k> {
        let other = other.into_real(self.k);
        self.lt(other).select(self, other)
    }

    pub fn max(self, other: impl IntoReal<'k>) -> Real<'k> {
        let other = other.into_real(self.k);
        self.gt(other).select(self, other)
    }

    fn compare(self, op: &str, other: impl IntoReal<'k>) -> Mask<'k> {
        let other = other.into_real(self.k).slot;
        Mask {
            k: self.k,
            slot: self.k.fun(op, &[self.slot, other]),
        }
    }

    pub fn lt(self, other: impl IntoReal<'k>) -> Mask<'k> {
        self.compare("lt", other)
    }

    pub fn le(self, other: impl IntoReal<'k>) -> Mask<'k> {
        self.compare("leq", other)
    }

    pub fn gt(self, other: impl IntoReal<'k>) -> Mask<'k> {
        self.compare("gt", other)
    }

    pub fn ge(self, other: impl IntoReal<'k>) -> Mask<'k> {
        self.compare("geq", other)
    }

    pub fn eq(self, other: impl IntoReal<'k>) -> Mask<'k> {
        self.compare("eq", other)
    }

    pub fn ne(self, other: impl IntoReal<'k>) -> Mask<'k> {
        self.compare("neq", other)
    }
}

macro_rules! cplx_unary {
    ($($name:ident => $op:literal),* $(,)?) => {
        $(
            pub fn $name(self) -> Cplx<'k> {
                Cplx { k: self.k, slot: self.k.fun($op, &[self.slot]) }
            }
        )*
    };
}

impl<'k> Cplx<'k> {
    /// The real part.
    pub fn re(self) -> Real<'k> {
        Real {
            k: self.k,
            slot: self.k.fun("real", &[self.slot]),
        }
    }

    /// The imaginary part.
    pub fn im(self) -> Real<'k> {
        Real {
            k: self.k,
            slot: self.k.fun("imaginary", &[self.slot]),
        }
    }

    /// `|z|^2`.
    pub fn norm_sqr(self) -> Real<'k> {
        Real {
            k: self.k,
            slot: self.k.fun("abs2", &[self.slot]),
        }
    }

    /// `|z|`.
    pub fn norm(self) -> Real<'k> {
        Real {
            k: self.k,
            slot: self.k.fun("abs", &[self.slot]),
        }
    }

    cplx_unary! {
        conj => "conjugate",
        recip => "recip",
        sqrt => "root",
        exp => "exp",
        ln => "ln",
        sin => "sin",
        cos => "cos",
        tan => "tan",
        sinh => "sinh",
        cosh => "cosh",
        tanh => "tanh",
        asin => "arcsin",
        acos => "arccos",
        atan => "arctan",
    }

    pub fn powi(self, p: i32) -> Cplx<'k> {
        Cplx {
            k: self.k,
            slot: self.k.powi(self.slot, p as i64),
        }
    }

    pub fn powc(self, p: impl IntoCplx<'k>) -> Cplx<'k> {
        let p = p.into_cplx(self.k).slot;
        Cplx {
            k: self.k,
            slot: self.k.powf(self.slot, p),
        }
    }
}

impl<'k> Mask<'k> {
    /// `a` where the mask is true, else `b`; both are computed.
    pub fn select(self, a: impl IntoReal<'k>, b: impl IntoReal<'k>) -> Real<'k> {
        let (a, b) = (a.into_real(self.k).slot, b.into_real(self.k).slot);
        Real {
            k: self.k,
            slot: self.join(a, b),
        }
    }

    /// `a` where the mask is true, else `b`; both are computed.
    pub fn select_complex(self, a: impl IntoCplx<'k>, b: impl IntoCplx<'k>) -> Cplx<'k> {
        let (a, b) = (a.into_cplx(self.k).slot, b.into_cplx(self.k).slot);
        Cplx {
            k: self.k,
            slot: self.join(a, b),
        }
    }

    /// 1.0 for true and 0.0 for false.
    pub fn to_real(self) -> Real<'k> {
        let one = self.k.constant(1.0).slot;
        Real {
            k: self.k,
            slot: self.k.fun("and", &[self.slot, one]),
        }
    }

    fn join(self, a: Slot, b: Slot) -> Slot {
        let dst = self.k.new_temp();
        self.k.emit(|c| c.append_join(&dst, &self.slot, &a, &b));
        dst
    }
}

// Operators. `a - b` is `minus` and `a / b` is `divide`, so the kernel rounds like Rust.

macro_rules! binary_ops {
    ($ty:ident, $conv:ident, [$($rhs:ty),*]) => {
        $(
            impl<'k> Add<$rhs> for $ty<'k> {
                type Output = $ty<'k>;
                fn add(self, rhs: $rhs) -> $ty<'k> {
                    let r = rhs.$conv(self.k).slot;
                    $ty { k: self.k, slot: self.k.add(self.slot, r) }
                }
            }

            impl<'k> Sub<$rhs> for $ty<'k> {
                type Output = $ty<'k>;
                fn sub(self, rhs: $rhs) -> $ty<'k> {
                    let r = rhs.$conv(self.k).slot;
                    $ty { k: self.k, slot: self.k.fun("minus", &[self.slot, r]) }
                }
            }

            impl<'k> Mul<$rhs> for $ty<'k> {
                type Output = $ty<'k>;
                fn mul(self, rhs: $rhs) -> $ty<'k> {
                    let r = rhs.$conv(self.k).slot;
                    $ty { k: self.k, slot: self.k.mul(self.slot, r) }
                }
            }

            impl<'k> Div<$rhs> for $ty<'k> {
                type Output = $ty<'k>;
                fn div(self, rhs: $rhs) -> $ty<'k> {
                    let r = rhs.$conv(self.k).slot;
                    $ty { k: self.k, slot: self.k.fun("divide", &[self.slot, r]) }
                }
            }

            impl<'k> AddAssign<$rhs> for $ty<'k> {
                fn add_assign(&mut self, rhs: $rhs) {
                    *self = *self + rhs;
                }
            }

            impl<'k> SubAssign<$rhs> for $ty<'k> {
                fn sub_assign(&mut self, rhs: $rhs) {
                    *self = *self - rhs;
                }
            }

            impl<'k> MulAssign<$rhs> for $ty<'k> {
                fn mul_assign(&mut self, rhs: $rhs) {
                    *self = *self * rhs;
                }
            }

            impl<'k> DivAssign<$rhs> for $ty<'k> {
                fn div_assign(&mut self, rhs: $rhs) {
                    *self = *self / rhs;
                }
            }
        )*
    };
}

binary_ops!(Real, into_real, [Real<'k>, f64]);
binary_ops!(Cplx, into_cplx, [Cplx<'k>, Real<'k>, f64, Complex<f64>]);

// a number or a `Real` on the left of a `Cplx`, a number on the left of a `Real`

macro_rules! left_ops {
    ($lhs:ty, $ty:ident, $conv:ident) => {
        impl<'k> Add<$ty<'k>> for $lhs {
            type Output = $ty<'k>;
            fn add(self, rhs: $ty<'k>) -> $ty<'k> {
                self.$conv(rhs.k) + rhs
            }
        }

        impl<'k> Sub<$ty<'k>> for $lhs {
            type Output = $ty<'k>;
            fn sub(self, rhs: $ty<'k>) -> $ty<'k> {
                self.$conv(rhs.k) - rhs
            }
        }

        impl<'k> Mul<$ty<'k>> for $lhs {
            type Output = $ty<'k>;
            fn mul(self, rhs: $ty<'k>) -> $ty<'k> {
                self.$conv(rhs.k) * rhs
            }
        }

        impl<'k> Div<$ty<'k>> for $lhs {
            type Output = $ty<'k>;
            fn div(self, rhs: $ty<'k>) -> $ty<'k> {
                self.$conv(rhs.k) / rhs
            }
        }
    };
}

left_ops!(f64, Real, into_real);
left_ops!(f64, Cplx, into_cplx);
left_ops!(Real<'k>, Cplx, into_cplx);
left_ops!(Complex<f64>, Cplx, into_cplx);

impl<'k> Neg for Real<'k> {
    type Output = Real<'k>;
    fn neg(self) -> Real<'k> {
        Real {
            k: self.k,
            slot: self.k.fun("neg", &[self.slot]),
        }
    }
}

impl<'k> Neg for Cplx<'k> {
    type Output = Cplx<'k>;
    fn neg(self) -> Cplx<'k> {
        Cplx {
            k: self.k,
            slot: self.k.fun("neg", &[self.slot]),
        }
    }
}

macro_rules! mask_ops {
    ($($tr:ident, $method:ident => $op:literal);* $(;)?) => {
        $(
            impl<'k> $tr for Mask<'k> {
                type Output = Mask<'k>;
                fn $method(self, rhs: Mask<'k>) -> Mask<'k> {
                    self.k.check_same(rhs.k);
                    Mask { k: self.k, slot: self.k.fun($op, &[self.slot, rhs.slot]) }
                }
            }
        )*
    };
}

mask_ops! {
    BitAnd, bitand => "and";
    BitOr, bitor => "or";
    BitXor, bitxor => "xor";
}

impl<'k> Not for Mask<'k> {
    type Output = Mask<'k>;
    fn not(self) -> Mask<'k> {
        Mask {
            k: self.k,
            slot: self.k.fun("not", &[self.slot]),
        }
    }
}

/// The operations that the [`kernel`] attribute macro rewrites Rust syntax into.
///
/// The macro sees syntax, not types, so each operation also accepts plain Rust values and then
/// behaves like Rust: `a < b` becomes [`lt`]`(a, b)`, a `Mask` for kernel values and a `bool`
/// for numbers, and `if c { .. } else { .. }` becomes [`branch`], a branch of the kernel when
/// `c` is a `Mask` and an ordinary `if` when it is a `bool`. So Rust loops, helper variables
/// and conditions on loop counters still run while the kernel is built.
pub mod ops {
    use super::*;

    thread_local! {
        /// The kernels being built on this thread, outermost first.
        static BUILDING: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    }

    /// Marks the kernel `name` as being built until it is dropped.
    pub struct Building(());

    impl Drop for Building {
        fn drop(&mut self) {
            BUILDING.with(|b| b.borrow_mut().pop());
        }
    }

    /// Starts building the kernel `name`; an error if it is being built already, i.e. if
    /// kernels call each other in a cycle (a kernel calling itself is a recursion instead).
    pub fn building(name: &str) -> Result<Building> {
        BUILDING.with(|b| {
            let mut b = b.borrow_mut();
            if let Some(i) = b.iter().position(|n| n == name) {
                let mut cycle: Vec<&str> = b[i..].iter().map(|n| n.as_str()).collect();
                cycle.push(name);
                return Err(anyhow!(
                    "the kernels call each other in a cycle ({}); a kernel can only call \
                     itself",
                    cycle.join(" -> ")
                ));
            }
            b.push(name.to_string());
            Ok(Building(()))
        })
    }

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

    /// Adds the Rust function `f` (the `extern "C"` shim of a `#[kernel(callback)]`) to `df`.
    pub fn add_callback(df: &mut Defuns, name: &str, f: UnaryFunc) {
        super::add_callback(df, name, f)
    }

    /// Adds the Rust function `f` of two arguments to `df`.
    pub fn add_callback2(df: &mut Defuns, name: &str, f: BinaryFunc) {
        super::add_callback2(df, name, f)
    }

    /// Converts a value into the handle type `T`: numbers become constants of the kernel.
    pub trait Lift<'k, T> {
        fn lift(self, k: &'k KernelBuilder) -> T;
    }

    pub fn lift<'k, T, A: Lift<'k, T>>(k: &'k KernelBuilder, a: A) -> T {
        a.lift(k)
    }

    macro_rules! lift_scalar {
        ($($from:ty => $to:ident by |$k:ident, $a:ident| $e:expr;)*) => {
            $(
                impl<'k> Lift<'k, $to<'k>> for $from {
                    fn lift(self, $k: &'k KernelBuilder) -> $to<'k> {
                        let $a = self;
                        $e
                    }
                }
            )*
        };
    }

    lift_scalar! {
        Real<'k> => Real by |k, a| a.into_real(k);
        f64 => Real by |k, a| k.constant(a);
        Cplx<'k> => Cplx by |k, a| a.into_cplx(k);
        Real<'k> => Cplx by |k, a| a.into_cplx(k);
        f64 => Cplx by |k, a| a.into_cplx(k);
        Complex<f64> => Cplx by |k, a| a.into_cplx(k);
        Mask<'k> => Mask by |k, a| { k.check_same(a.k); a };
        bool => Mask by |k, a| k.mask(a);
    }

    /// A number among numbers: the literals of an array of numbers that is not a kernel
    /// value (the macro lifts the literals of arrays that mix them with other values).
    impl<'k> Lift<'k, f64> for f64 {
        fn lift(self, _: &'k KernelBuilder) -> f64 {
            self
        }
    }

    impl<'k, A: Lift<'k, T>, T, const N: usize> Lift<'k, [T; N]> for [A; N] {
        fn lift(self, k: &'k KernelBuilder) -> [T; N] {
            self.map(|a| a.lift(k))
        }
    }

    /// The type both branches of a kernel `if` are converted to (`Self` is the type of the
    /// first branch, `B` of the second): `Real` for `Real`/`f64`, `Cplx` if either is complex,
    /// `Mask` for `Mask`/`bool`, and element by element for tuples and arrays. (A trait on
    /// the pair `(A, B)` makes the trait solver overflow on the tuple impls.)
    #[diagnostic::on_unimplemented(
        message = "the branches of a kernel `if` give `{Self}` and `{B}`",
        note = "both branches must give kernel values of the same shape (numbers, `Real`, \
                `Cplx`, `Mask`, or tuples or arrays of them); a variable assigned in a kernel \
                branch must be a float, not an integer (`let mut k = 0.0`)"
    )]
    pub trait Unify<'k, B> {
        type Out: Value<'k>;
        fn left(self, k: &'k KernelBuilder) -> Self::Out;
        fn right(b: B, k: &'k KernelBuilder) -> Self::Out;
    }

    macro_rules! unify {
        ($out:ident: $(($a:ty, $b:ty)),*) => {
            $(
                impl<'k> Unify<'k, $b> for $a {
                    type Out = $out<'k>;
                    fn left(self, k: &'k KernelBuilder) -> $out<'k> {
                        self.lift(k)
                    }
                    fn right(b: $b, k: &'k KernelBuilder) -> $out<'k> {
                        b.lift(k)
                    }
                }
            )*
        };
    }

    unify!(Real: (Real<'k>, Real<'k>), (Real<'k>, f64), (f64, Real<'k>), (f64, f64));
    unify!(Cplx: (Cplx<'k>, Cplx<'k>), (Cplx<'k>, Real<'k>), (Cplx<'k>, f64),
        (Cplx<'k>, Complex<f64>), (Real<'k>, Cplx<'k>), (f64, Cplx<'k>),
        (Complex<f64>, Cplx<'k>), (Complex<f64>, Complex<f64>), (Complex<f64>, Real<'k>),
        (Complex<f64>, f64), (Real<'k>, Complex<f64>), (f64, Complex<f64>));
    unify!(Mask: (Mask<'k>, Mask<'k>), (Mask<'k>, bool), (bool, Mask<'k>), (bool, bool));

    // a branch that only assigns variables
    impl<'k> Unify<'k, ()> for () {
        type Out = ();
        fn left(self, _: &'k KernelBuilder) {}
        fn right(_: (), _: &'k KernelBuilder) {}
    }

    impl<'k, A: Unify<'k, B>, B, const N: usize> Unify<'k, [B; N]> for [A; N] {
        type Out = [A::Out; N];
        fn left(self, k: &'k KernelBuilder) -> Self::Out {
            self.map(|a| a.left(k))
        }
        fn right(b: [B; N], k: &'k KernelBuilder) -> Self::Out {
            b.map(|b| A::right(b, k))
        }
    }

    macro_rules! tuples {
        ($(($($a:ident $b:ident $t:ident $i:tt),+)),*) => {
            $(
                impl<'k, $($a: Lift<'k, $t>, $t),+> Lift<'k, ($($t,)+)> for ($($a,)+) {
                    fn lift(self, k: &'k KernelBuilder) -> ($($t,)+) {
                        ($(self.$i.lift(k),)+)
                    }
                }

                impl<'k, $($a: Unify<'k, $b>, $b),+> Unify<'k, ($($b,)+)> for ($($a,)+) {
                    type Out = ($($a::Out,)+);
                    fn left(self, k: &'k KernelBuilder) -> Self::Out {
                        ($(self.$i.left(k),)+)
                    }
                    fn right(b: ($($b,)+), k: &'k KernelBuilder) -> Self::Out {
                        ($($a::right(b.$i, k),)+)
                    }
                }
            )*
        };
    }

    tuples! {
        (A0 B0 T0 0),
        (A0 B0 T0 0, A1 B1 T1 1),
        (A0 B0 T0 0, A1 B1 T1 1, A2 B2 T2 2),
        (A0 B0 T0 0, A1 B1 T1 1, A2 B2 T2 2, A3 B3 T3 3),
        (A0 B0 T0 0, A1 B1 T1 1, A2 B2 T2 2, A3 B3 T3 3, A4 B4 T4 4),
        (A0 B0 T0 0, A1 B1 T1 1, A2 B2 T2 2, A3 B3 T3 3, A4 B4 T4 4, A5 B5 T5 5)
    }

    /// `if c { t } else { e }`: a kernel branch for a `Mask`, an `if` for a `bool`.
    #[diagnostic::on_unimplemented(
        message = "cannot branch on `{Self}` with these branches",
        note = "a branch on a kernel value (`Mask`) must return the same kind of value from both \
                branches (numbers, `Real`, `Cplx`, `Mask`, or tuples or arrays of them)",
        note = "and its branches cannot assign variables declared outside them: both branches \
                run while the kernel is built"
    )]
    pub trait Branch<'k, T, E> {
        type Output;
        fn branch(self, k: &'k KernelBuilder, t: T, e: E) -> Self::Output;
    }

    impl<'k, V, T: FnOnce() -> V, E: FnOnce() -> V> Branch<'k, T, E> for bool {
        type Output = V;
        fn branch(self, _: &'k KernelBuilder, t: T, e: E) -> V {
            if self {
                t()
            } else {
                e()
            }
        }
    }

    // `Fn`, not `FnOnce`: a closure that assigns a captured variable is only `FnMut`, and
    // both branches run while the kernel is built
    impl<'k, A: Unify<'k, B>, B, T: Fn() -> A, E: Fn() -> B> Branch<'k, T, E> for Mask<'k> {
        type Output = A::Out;
        fn branch(self, k: &'k KernelBuilder, t: T, e: E) -> A::Out {
            k.check_same(self.k);
            k.if_else(self, || t().left(k), || A::right(e(), k))
        }
    }

    pub fn branch<'k, C: Branch<'k, T, E>, T, E>(
        k: &'k KernelBuilder,
        cond: C,
        t: T,
        e: E,
    ) -> C::Output {
        cond.branch(k, t, e)
    }

    /// `if c { .. }` without `else`: only for a `bool`.
    #[diagnostic::on_unimplemented(
        message = "an `if` without `else` cannot branch on a kernel value (`{Self}`)",
        note = "give it an `else` branch, and return the values from both branches"
    )]
    pub trait When {
        fn when(self, f: impl FnOnce());
    }

    impl When for bool {
        fn when(self, f: impl FnOnce()) {
            if self {
                f()
            }
        }
    }

    pub fn when(cond: impl When, f: impl FnOnce()) {
        cond.when(f)
    }

    /// `while c(state) { state = body(state) }`: a kernel loop if the condition is a `Mask`, a
    /// Rust loop (while the kernel is built) if it is a `bool`.
    #[diagnostic::on_unimplemented(
        message = "cannot loop on a condition of type `{Self}`",
        note = "a kernel loop (on a `Mask`) can only update kernel values (`Real`, `Cplx`, \
                `Mask`, or tuples or arrays of them)"
    )]
    pub trait LoopCond<'k, V>: Sized {
        fn run(k: &'k KernelBuilder, init: V, cond: impl Fn(V) -> Self, body: impl Fn(V) -> V)
            -> V;
    }

    impl<'k, V: Clone> LoopCond<'k, V> for bool {
        fn run(
            _: &'k KernelBuilder,
            init: V,
            cond: impl Fn(V) -> bool,
            body: impl Fn(V) -> V,
        ) -> V {
            let mut state = init;
            while cond(state.clone()) {
                state = body(state);
            }
            state
        }
    }

    impl<'k, V: Value<'k>> LoopCond<'k, V> for Mask<'k> {
        fn run(
            k: &'k KernelBuilder,
            init: V,
            cond: impl Fn(V) -> Mask<'k>,
            body: impl Fn(V) -> V,
        ) -> V {
            k.while_loop(init, cond, body)
        }
    }

    pub fn while_loop<'k, V, C: LoopCond<'k, V>>(
        k: &'k KernelBuilder,
        init: V,
        cond: impl Fn(V) -> C,
        body: impl Fn(V) -> V,
    ) -> V {
        C::run(k, init, cond, body)
    }

    /// The right side of an assignment, converted to the type of the variable (`x = 1.0`
    /// for a `Real` `x`).
    pub trait Coerce<'k, R> {
        fn coerce(k: &'k KernelBuilder, r: R) -> Self;
    }

    impl<'k, T> Coerce<'k, T> for T {
        fn coerce(_: &'k KernelBuilder, r: T) -> T {
            r
        }
    }

    macro_rules! coerce {
        ($($from:ty => $to:ty),*) => {
            $(
                impl<'k> Coerce<'k, $from> for $to {
                    fn coerce(k: &'k KernelBuilder, r: $from) -> $to {
                        r.lift(k)
                    }
                }
            )*
        };
    }

    coerce!(f64 => Real<'k>, f64 => Cplx<'k>, Real<'k> => Cplx<'k>, Complex<f64> => Cplx<'k>,
        bool => Mask<'k>);

    pub fn coerce<'k, L: Coerce<'k, R>, R>(k: &'k KernelBuilder, _like: &L, r: R) -> L {
        L::coerce(k, r)
    }

    /// `<`, `<=`, `>`, `>=`: a `Mask` for kernel values, a `bool` for Rust values.
    pub trait Compare<Rhs> {
        type Output;
        fn lt(self, rhs: Rhs) -> Self::Output;
        fn le(self, rhs: Rhs) -> Self::Output;
        fn gt(self, rhs: Rhs) -> Self::Output;
        fn ge(self, rhs: Rhs) -> Self::Output;
    }

    /// `==`, `!=`: a `Mask` for kernel values, a `bool` for Rust values.
    pub trait Equal<Rhs> {
        type Output;
        fn eq(self, rhs: Rhs) -> Self::Output;
        fn ne(self, rhs: Rhs) -> Self::Output;
    }

    impl<T: PartialOrd> Compare<T> for T {
        type Output = bool;
        fn lt(self, rhs: T) -> bool {
            self < rhs
        }
        fn le(self, rhs: T) -> bool {
            self <= rhs
        }
        fn gt(self, rhs: T) -> bool {
            self > rhs
        }
        fn ge(self, rhs: T) -> bool {
            self >= rhs
        }
    }

    impl<T: PartialEq> Equal<T> for T {
        type Output = bool;
        fn eq(self, rhs: T) -> bool {
            self == rhs
        }
        fn ne(self, rhs: T) -> bool {
            self != rhs
        }
    }

    macro_rules! compare_real {
        ($($lhs:ty, $rhs:ty, |$a:ident, $b:ident| $real:expr);*) => {
            $(
                impl<'k> Compare<$rhs> for $lhs {
                    type Output = Mask<'k>;
                    fn lt(self, rhs: $rhs) -> Mask<'k> {
                        let ($a, $b) = (self, rhs);
                        let (x, y): (Real<'k>, Real<'k>) = $real;
                        x.lt(y)
                    }
                    fn le(self, rhs: $rhs) -> Mask<'k> {
                        let ($a, $b) = (self, rhs);
                        let (x, y): (Real<'k>, Real<'k>) = $real;
                        x.le(y)
                    }
                    fn gt(self, rhs: $rhs) -> Mask<'k> {
                        let ($a, $b) = (self, rhs);
                        let (x, y): (Real<'k>, Real<'k>) = $real;
                        x.gt(y)
                    }
                    fn ge(self, rhs: $rhs) -> Mask<'k> {
                        let ($a, $b) = (self, rhs);
                        let (x, y): (Real<'k>, Real<'k>) = $real;
                        x.ge(y)
                    }
                }

                impl<'k> Equal<$rhs> for $lhs {
                    type Output = Mask<'k>;
                    fn eq(self, rhs: $rhs) -> Mask<'k> {
                        let ($a, $b) = (self, rhs);
                        let (x, y): (Real<'k>, Real<'k>) = $real;
                        x.eq(y)
                    }
                    fn ne(self, rhs: $rhs) -> Mask<'k> {
                        let ($a, $b) = (self, rhs);
                        let (x, y): (Real<'k>, Real<'k>) = $real;
                        x.ne(y)
                    }
                }
            )*
        };
    }

    compare_real! {
        Real<'k>, Real<'k>, |a, b| (a, b.into_real(a.k));
        Real<'k>, f64, |a, b| (a, b.into_real(a.k));
        f64, Real<'k>, |a, b| (a.into_real(b.k), b)
    }

    pub fn lt<A: Compare<B>, B>(a: A, b: B) -> A::Output {
        a.lt(b)
    }

    pub fn le<A: Compare<B>, B>(a: A, b: B) -> A::Output {
        a.le(b)
    }

    pub fn gt<A: Compare<B>, B>(a: A, b: B) -> A::Output {
        a.gt(b)
    }

    pub fn ge<A: Compare<B>, B>(a: A, b: B) -> A::Output {
        a.ge(b)
    }

    pub fn eq<A: Equal<B>, B>(a: A, b: B) -> A::Output {
        a.eq(b)
    }

    pub fn ne<A: Equal<B>, B>(a: A, b: B) -> A::Output {
        a.ne(b)
    }

    /// `&&` and `||`: both sides are evaluated (kernel masks cannot short-circuit).
    pub trait Logic<Rhs> {
        type Output;
        fn and(self, rhs: Rhs) -> Self::Output;
        fn or(self, rhs: Rhs) -> Self::Output;
    }

    impl Logic<bool> for bool {
        type Output = bool;
        fn and(self, rhs: bool) -> bool {
            self && rhs
        }
        fn or(self, rhs: bool) -> bool {
            self || rhs
        }
    }

    impl<'k> Logic<Mask<'k>> for Mask<'k> {
        type Output = Mask<'k>;
        fn and(self, rhs: Mask<'k>) -> Mask<'k> {
            self & rhs
        }
        fn or(self, rhs: Mask<'k>) -> Mask<'k> {
            self | rhs
        }
    }

    impl<'k> Logic<bool> for Mask<'k> {
        type Output = Mask<'k>;
        fn and(self, rhs: bool) -> Mask<'k> {
            self & self.k.mask(rhs)
        }
        fn or(self, rhs: bool) -> Mask<'k> {
            self | self.k.mask(rhs)
        }
    }

    impl<'k> Logic<Mask<'k>> for bool {
        type Output = Mask<'k>;
        fn and(self, rhs: Mask<'k>) -> Mask<'k> {
            rhs.k.mask(self) & rhs
        }
        fn or(self, rhs: Mask<'k>) -> Mask<'k> {
            rhs.k.mask(self) | rhs
        }
    }

    pub fn and<A: Logic<B>, B>(a: A, b: B) -> A::Output {
        a.and(b)
    }

    pub fn or<A: Logic<B>, B>(a: A, b: B) -> A::Output {
        a.or(b)
    }

    /// The fields `z.re` and `z.im` of a `Complex<f64>`, and the parts of a `Cplx`.
    pub trait Parts {
        type Part;
        fn re(self) -> Self::Part;
        fn im(self) -> Self::Part;
    }

    impl Parts for Complex<f64> {
        type Part = f64;
        fn re(self) -> f64 {
            self.re
        }
        fn im(self) -> f64 {
            self.im
        }
    }

    impl<'k> Parts for Cplx<'k> {
        type Part = Real<'k>;
        fn re(self) -> Real<'k> {
            Cplx::re(self)
        }
        fn im(self) -> Real<'k> {
            Cplx::im(self)
        }
    }

    pub fn re<Z: Parts>(z: Z) -> Z::Part {
        z.re()
    }

    pub fn im<Z: Parts>(z: Z) -> Z::Part {
        z.im()
    }

    /// `Complex::new(re, im)`: a `Complex<f64>` from numbers, a `Cplx` otherwise.
    pub trait MakeComplex<'k, Im> {
        type Output;
        fn complex(self, k: &'k KernelBuilder, im: Im) -> Self::Output;
    }

    impl<'k> MakeComplex<'k, f64> for f64 {
        type Output = Complex<f64>;
        fn complex(self, _: &'k KernelBuilder, im: f64) -> Complex<f64> {
            Complex::new(self, im)
        }
    }

    macro_rules! make_complex {
        ($(($re:ty, $im:ty)),*) => {
            $(
                impl<'k> MakeComplex<'k, $im> for $re {
                    type Output = Cplx<'k>;
                    fn complex(self, k: &'k KernelBuilder, im: $im) -> Cplx<'k> {
                        k.complex(self, im)
                    }
                }
            )*
        };
    }

    make_complex!((Real<'k>, Real<'k>), (Real<'k>, f64), (f64, Real<'k>));

    pub fn complex<'k, A: MakeComplex<'k, B>, B>(k: &'k KernelBuilder, re: A, im: B) -> A::Output {
        re.complex(k, im)
    }
}

// The functions of symjit's math library (`libm.rs`) build kernels too: `libm::exp(x)` with a
// `Real` adds its operations to the kernel.
#[cfg(feature = "math")]
impl<'k> crate::libm::MathOps for Real<'k> {
    type Mask = Mask<'k>;

    fn lit(self, v: f64) -> Real<'k> {
        self.k.constant(v)
    }

    // as `v < self`: `gt` is true for NaN on x86 (see `MathOps::gt`)
    fn gt(self, v: f64) -> Mask<'k> {
        self.k.constant(v).lt(self)
    }

    fn lt(self, v: f64) -> Mask<'k> {
        Real::lt(self, v)
    }

    fn select(m: Mask<'k>, a: Real<'k>, b: Real<'k>) -> Real<'k> {
        m.select(a, b)
    }

    fn abs(self) -> Real<'k> {
        Real::abs(self)
    }

    fn pow2i(self) -> Real<'k> {
        Real::pow2i(self)
    }

    fn getexp(self) -> Real<'k> {
        Real::getexp(self)
    }

    fn getmant(self) -> Real<'k> {
        Real::getmant(self)
    }
}

# Kernels in Rust

The `composer` cargo feature lets Rust programs write symjit kernels as ordinary Rust
functions. The `#[kernel]` attribute compiles a function into a kernel and keeps the function
itself, so the two can be compared:

```rust
use num_complex::Complex;
use symjit::kernel::kernel;
use symjit::Config;

/// The escape time of the Mandelbrot iteration (at most `limit` steps).
#[kernel]
fn escape(c: Complex<f64>, limit: f64) -> f64 {
    let mut z = Complex::new(0.0, 0.0);
    let mut n = 0.0;
    while n < limit && z.norm_sqr() <= 4.0 {
        z = z * z + c;
        n += 1.0;
    }
    n
}

fn main() -> anyhow::Result<()> {
    let app = escape::compile(Config::default())?.seal()?;
    let c = Complex::new(-0.75, 0.1);
    let n = app.evaluate_single(&[c, Complex::new(100.0, 0.0)]);
    assert_eq!(n.re, escape(c, 100.0));    // the kernel and the Rust function agree
    Ok(())
}
```

`escape::compile(config)` returns an `Application`, like the `Translator` of the
[Composer](COMPOSER.md) interface that it uses. `seal()` makes it an `Applet`, which evaluates
points (`evaluate`, `evaluate_single`) and matrices of points (`evaluate_matrix`, with SIMD
and threads).

The macro is built on a typed builder, `symjit::kernel::KernelBuilder`. The builder can also
be used directly (see [The builder](#the-builder)). The command-line compiler
[sjc](../../sjc/docs/SJC.md#rust-input) compiles the kernels of a Rust file into object files or WebAssembly
modules without a Rust compiler.

## Enabling it

The feature is off by default:

```toml
[dependencies]
symjit = { version = "2.28", features = ["composer"] }
```

It adds the module `symjit::kernel` and the crate `symjit-macros`. It does not change the
code symjit generates.

## How it works

The body of a kernel runs once, when the kernel is compiled. Its values are handles to
values of the kernel:

| in the function   | while compiling             |
|-------------------|-----------------------------|
| `f64`             | `Real`                      |
| `Complex<f64>`    | `Cplx` (complex kernels)    |
| `bool`            | `Mask` (a comparison)       |

Arithmetic and methods on handles (`x * y`, `x.sin()`, `z.conj()`) add instructions to the
kernel. Everything else is plain Rust that runs once, while the kernel is built:

- `for` loops are unrolled, and integer arithmetic is done at build time. A loop over an
  array of kernel values builds code for every element.
- Helper functions with `#[kernel(inline)]` are inlined. They can take any parameters,
  e.g. an `i32` exponent or a `[f64; 4]` of coefficients.
- Closures and local variables work as usual.

The macro rewrites only the syntax Rust cannot overload:

- **Comparisons and `&&`/`||`** become masks. Both sides of `&&` and `||` are always
  evaluated.
- **`if c { .. } else { .. }`** is a kernel branch when `c` is a kernel value, and an
  ordinary `if` when it is a `bool`. Only the instructions of the branch taken are executed,
  so a branch can recurse.
- **`while c { .. }`** is a kernel loop when `c` is a kernel value.
- **Variables assigned in a branch or loop** (`x = ..`, `x += ..`, `(a, b) = ..`) are passed
  through it. Each side of a branch works on copies and returns their new values.
- **Calls** of the kernel itself, and of functions with the attribute, are rewritten (see
  [Calls](#calls)).
- **`let mut x = 0.0`** and **`let mut z = Complex::new(0.0, 0.0)`** (literals in a `let mut`)
  make kernel values, so that they can accumulate kernel values. A `let` with a type,
  `let x: f64 = ..`, makes a kernel value too.
- **`[1.0, x]`**: float literals in an array take the type of the other elements.
- **`z.re`, `z.im` and `Complex::new(re, im)`** work on kernel values.

A kernel is complex if a parameter or the result is complex, or with `#[kernel(complex)]`.
Its parameters must be `f64` or `Complex<f64>`, or tuples or arrays of them. It returns
`f64`, `Complex<f64>` or `bool`, or a tuple or array of them, one output each. A `bool`
output is a mask: all bits set for true (a NaN as an `f64`), zero for false. Return
`if c { 1.0 } else { 0.0 }` for a number.

### Loops

A kernel loop runs at run time. Its condition can depend on the arguments:

```rust
#[kernel]
fn newton_sqrt(a: f64) -> (f64, f64) {
    let mut x = if a > 1.0 { a } else { 1.0 };
    let mut steps = 0.0;
    while (x * x - a).abs() > 1e-12 * a {
        x = 0.5 * (x + a / x);
        steps += 1.0;
    }
    (x, steps)
}
```

- The variables a loop or branch updates must be kernel values. A counter is `0.0`, not
  `0`.
- Kernel loops need the direct translator. `compile` selects it for a kernel with a `while`.
- A `while` with a `bool` condition is an ordinary Rust loop, run while the kernel is built.

### SIMD

`evaluate_matrix` runs the SIMD kernel on groups of rows (4 with AVX, 8 with AVX-512, 2 on
ARM64). When the rows of a group take different sides of a kernel branch, or leave a loop at
different times, the group is computed again by the scalar kernel, one row at a time. This
also covers branches in the kernels a kernel calls. The results are the same; only the
speed differs.

## Calls

A kernel calls a function named by a single identifier. The function must have the
attribute too:

```rust
#[kernel(callback)]
fn weight(x: f64) -> f64 {
    1.0 / (1.0 + x * x)          // ordinary Rust, run when the kernel runs
}

#[kernel(inline)]
fn midpoint(a: f64, h: f64, i: f64) -> f64 {
    a + (i + 0.5) * h            // inlined into the kernels that call it
}

#[kernel]
fn integrand(x: f64) -> f64 {
    x.cos() * weight(x)          // a kernel of its own
}

#[kernel]
fn integrate(a: f64, b: f64, n: f64) -> f64 {
    let h = (b - a) / n;
    let mut sum = 0.0;
    let mut i = 0.0;
    while i < n {
        sum += integrand(midpoint(a, h, i));   // a call in a kernel loop
        i += 1.0;
    }
    sum * h
}
```

- **`#[kernel(inline)]` helpers** are inlined. They cannot call themselves.
- **`#[kernel]` kernels** are compiled separately and called.
  - `compile` compiles the kernels a kernel calls, also through its inline helpers and
    their calls. It adds them to the `Defuns` of its configuration, merged with any it
    already has.
  - A called kernel returns one value.
  - A real kernel called from a complex one is compiled as a complex kernel. A real kernel
    cannot call a complex one: make the caller `#[kernel(complex)]`.
  - Kernels cannot call each other in a cycle. Only a kernel itself can recurse.
- **Recursion**: a kernel calls itself. The machine code does not check the depth, so a
  recursion that does not end overflows the stack and crashes the process.
- **`#[kernel(callback)]` functions** take one or two `f64` and return an `f64`. They run
  as Rust when the kernel runs. In complex kernels they receive the real parts. A panic in
  one aborts the process (it unwinds into machine code).

Call other functions with a path (`f64::max(a, b)`, `crate::f(x)`) or through a local
variable. They then run at build time.

## Real values in complex kernels

A complex kernel holds every value as a complex number; a `Real` is one whose imaginary
part is zero. Real parameters take the real part of their argument. This changes a few
results:

- Real functions are the complex functions applied to real arguments, within a few ulps of
  the real ones. The ones that leave the real domain (`ln` of a negative number, `asin`
  outside [-1, 1]) return the real part of the complex result instead of NaN. `sqrt` is
  NaN for negative numbers in both.
- Division and `recip` are complex divisions: within an ulp or two of the real quotient,
  and NaN for a zero divisor.
- The sign of a zero result can be lost.

## Differences from Rust

- `f64::round` rounds halves away from zero, as in Rust. `Real::round_ties_even` is
  symjit's `round`.
- There is no `%`. Write `x - (x / y).floor() * y`, or use `Real::modulo` (floored) in the
  builder.
- With `fastmath` (the default), multiplications and additions can be fused, which changes
  the last bit of some results.

## Limitations

These are errors at compile time, unless noted:

| what                                                   | message, or effect                                     |
|--------------------------------------------------------|--------------------------------------------------------|
| `return`                                               | "a kernel cannot `return`"                             |
| `break`/`continue` out of a kernel `if`                | "a kernel `if` cannot `break` or `continue` ..."       |
| an `if` without `else` on a kernel value, that assigns nothing | "an `if` without `else` cannot branch on a kernel value" |
| an integer updated in a kernel loop or branch          | "`{integer}` is not a kernel value" / "the branches of a kernel `if` give `{integer}` ..." |
| branches of different shapes                           | "`if` and `else` have incompatible types"              |
| a closure in a kernel `if`/`while` that assigns an outer variable | "a closure in a kernel `if` or `while` cannot assign ..." |
| a parameter that is not `f64` or `Complex<f64>`        | "the parameters of a kernel must be ..."              |
| a call of a function without the attribute             | "cannot find module or crate `f`": give it `#[kernel(inline)]`, or call it with a path |
| a callback that is not `fn(f64[, f64]) -> f64`         | "a callback takes one or two `f64` and returns `f64`"  |
| an inline helper that calls itself                     | "an inline helper cannot call itself"                  |
| a kernel defined inside a function body                | "cannot find ... in module `super`": kernels must be items of a module |
| mutation by a method (`v.push(x)`) in a kernel branch or loop | not passed through the branch or loop (no error)  |

The calls report their errors when the kernel is compiled (an `Err` from `compile`): a real
kernel calling a complex one, a cycle of calls, a called kernel with several outputs.

## The builder

`KernelBuilder` is what the macro uses. It can build kernels from data, e.g. a kernel per
model read at run time:

```rust
use symjit::kernel::{KernelBuilder, Real};
use symjit::prelude::*;     // `compile` (KernelCompile)
use symjit::Config;

let k = KernelBuilder::new(Config::default());
let (x, y): (Real, Real) = k.params();
let r = (x * x + y * y).sqrt();
let app = k.compile(r.lt(1.0).select(r, 1.0))?.seal()?;
assert_eq!(app.evaluate_single(&[0.3, 0.4]), 0.5);
```

- `k.params()` declares parameters (`Real`, `Cplx`, tuples, arrays).
- `k.constant(v)` and `k.constant_complex(z)` make constants.
- Comparisons (`x.lt(y)`, ...) give a `Mask`. `mask.select(a, b)` picks a value, and
  `mask.to_real()` gives 1.0/0.0.
- `k.if_else(cond, || then, || otherwise)` is a kernel branch. `k.while_loop(init,
  |state| cond, |state| next)` is a kernel loop; it needs `Config::set_direct(true)`.
- `k.recurse(args)` calls the kernel itself.
- `k.call(name, args)` calls a kernel added with `Defuns::add_applet(name, applet)`. The
  two must both be real or both complex. Applets of `Compiler::compile` cannot be called:
  their inputs are states.
- `k.callback(name, x)` and `k.callback(name, (x, y))` call a Rust `extern "C"` function
  added with `kernel::add_callback(&mut defuns, name, f)` (or `add_callback2`).
- `kernel::add_defuns(&mut config, defuns)` adds functions to a configuration that has some
  already (`Config::set_defuns` can be called once).
- `k.compile(outs)` (the `KernelCompile` trait, in `symjit::prelude`) writes the outputs and
  compiles; `k.finish(outs)` gives the program and its MIR instead (`symjit::Compilation`, what
  `sjc` compiles). Operators cannot return errors, so the
  first error is kept and returned here.

## Examples

`examples/claude/rk4_bessel.rs` solves Bessel's equation with a Runge-Kutta kernel that calls
the equation's kernel in a loop, and compares the solution with the Bessel functions:

```bash
cargo run --release --features composer --example rk4_bessel
```

`examples/claude/pi_hypersphere.rs` computes pi from the volume of the four-dimensional unit
ball with a recursive complex kernel (a port of the Python `Composer` example of the same
name), and checks it bit for bit against plain Rust:

```bash
cargo run --release --features composer --example pi_hypersphere -- 7
```

## Tests

| file                       | what                                                          |
|----------------------------|---------------------------------------------------------------|
| `tests/kernel_builder.rs`  | the builder's operations against `f64` and `Complex<f64>`     |
| `tests/kernel_macro.rs`    | the macro: branches, loops, recursion, build-time code        |
| `tests/kernel_calls.rs`    | calls of kernels and callbacks, SIMD groups that diverge      |
| `tests/kernel_diff.rs`     | each kernel against its Rust function at random points, in every mode |
| `tests/kernel_ui.rs`       | the compile errors of `tests/ui/*.rs` (with `trybuild`)       |

```bash
cargo test --features composer --test kernel_macro      # one file
SYMJIT_DIFF_POINTS=5000 SYMJIT_DIFF_SEED=7 SYMJIT_DIFF_VERBOSE=1 \
    cargo test --features composer --test kernel_diff -- --nocapture
TRYBUILD=overwrite cargo test --features composer --test kernel_ui   # after changing a message
```

`kernel_diff` runs each kernel with the direct and indirect translators, without SIMD, with
SIMD and with AVX-512 SIMD, one point at a time and as a matrix, and real kernels also as
complex ones. The compile-error messages in `tests/ui/*.stderr` depend on the compiler's
wording, and may need `TRYBUILD=overwrite` after a Rust update.

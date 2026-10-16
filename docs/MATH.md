# symjit's own math library

The `math` cargo feature adds `rust/libm.rs`, a math library written in generic Rust whose
functions the SIMD kernels evaluate inline, on all lanes at once. Without it, a SIMD kernel
calls the scalar C function once per lane. The library has `exp`, `ln` (and `log`, an alias), `log10`, `sin`,
`cos` and the six hyperbolic functions `sinh`, `cosh`, `tanh`, `coth`, `sech` and `csch`, and
complex versions of `exp`, `sin`, `cos` and the hyperbolic functions.

```toml
[dependencies]
symjit = { version = "2.28", features = ["math"] }
```

The feature is off by default. It does not change the code symjit generates unless the
`inline_math` option is set. (It is unrelated to the `libm` feature, which links the C
library's complex functions.)

## Inlining

```python
f = compile_func([x], [sinh(x)], ty="native;inline_math=true")
```

```rust
config.set_inline_math(true)?;          // or config.set_option("inline_math", "true")
```

- **Real kernels**: a call of a library function becomes a call of a funclet, `@libm_sinh`,
  appended to the kernel. The SIMD kernels (x86 AVX2 and AVX-512, ARM64) run it; the scalar
  kernel (single points, the points after the last full SIMD row, kernels without SIMD), the
  bytecode interpreter and the WebAssembly module call the C function as before
  (`Generator::inlines_math`).
- **Complex kernels**, `exp`, `sin`, `cos` and the hyperbolic functions: `complexify.rs` calls
  the library's complex function (`@libm_cplx_sinh`, ...) when the argument is in its domain
  (`libm::complex_domain`: |re| < 709 for exp and 710 for the hyperbolic functions, |re| < 2^20
  π/2 for sin and cos, |im| < 2^20 π/2, or 710 for sin and cos; false for NaN and infinities),
  and the complex C function otherwise; a SIMD row with both is computed again by
  the scalar kernel. Real arguments take the complex function too (their imaginary part is a
  correctly signed zero), so that rows of real and complex arguments stay in the SIMD kernel;
  an argument known to be real at compile time gets the real function.
- **Complex kernels**, `ln`, `log` and `log10`: `complexify.rs` tests the argument at run time.
  When it is a positive real (zero imaginary part), the real function is called (the funclet
  in SIMD kernels), otherwise the complex C function. A SIMD row whose lanes disagree is
  computed again by the scalar kernel. The result's imaginary part is then +0.
- **`log`** is an alias of `ln`, the natural logarithm, in real and complex kernels (and in
  object files and WebAssembly modules); `log10` is the decimal logarithm. (Before patch
  v228-0021, `log` was log10 in real kernels.)
- **Domains**: `sin` and `cos` are inlined for |x| < 2^20 π/2 ≈ 1.65e6
  (`libm::TRIG_LIMIT`). The call site tests the argument; larger arguments, NaN and infinities
  go to the C function, and a SIMD row with both kinds of arguments is computed by the scalar
  kernel. `ln` and `log10` handle every argument (0, negative numbers, infinities, NaN,
  subnormals) and need no test in real kernels.

Measured on this machine, single-threaded, 10^6 points (ms):

| function | AVX2, C per lane | AVX2, inlined | AVX-512, C per lane | AVX-512, inlined |
|----------|------------------|---------------|---------------------|------------------|
| exp      | 3.57             | 1.46          | 2.74                | 0.93             |
| sinh     | 11.72            | 2.11          | 11.28               | 1.24             |
| cosh     | 4.76             | 1.69          | 4.63                | 1.03             |
| tanh     | 10.35            | 2.56          | 9.95                | 1.51             |
| coth     | 11.52            | 2.56          | 11.19               | 1.50             |
| sech     | 5.48             | 1.72          | 5.25                | 1.05             |
| csch     | 12.34            | 2.27          | 12.10               | 1.32             |
| sin      | 12.57            | 2.09          | 12.25               | 1.20             |
| cos      | 12.53            | 2.01          | 12.11               | 1.13             |
| ln       | 3.28             | 1.92          | 2.68                | 1.21             |
| log10    | 5.55             | 1.99          | 5.15                | 1.19             |

Complex kernels, 10^6 points (ms; complex arguments, |x| < 5, |y| < 3; real arguments take
the same time):

| function | AVX2, C per lane | AVX2, inlined | AVX-512, C per lane | AVX-512, inlined |
|----------|------------------|---------------|---------------------|------------------|
| sinh     | 32.4             | 8.7           | 32.7                | 7.4              |
| cosh     | 32.4             | 9.0           | 33.4                | 7.2              |
| tanh     | 35.6             | 9.7           | 36.9                | 7.6              |
| coth     | 37.0             | 9.8           | 38.3                | 7.6              |
| sech     | 37.4             | 10.0          | 36.7                | 7.7              |
| csch     | 35.3             | 9.9           | 34.5                | 7.5              |
| exp      | 20.8             | 8.0           | 20.9                | 6.6              |
| sin      | 36.9             | 8.8           | 36.6                | 7.0              |
| cos      | 37.0             | 8.8           | 36.0                | 7.1              |

## Accuracy

Maximum error against mpmath, in ulps (33,000 points for `exp`, 13,000 for the others; with
`fastmath`, the default):

| function | C library | inlined |
|----------|-----------|---------|
| exp      | 0.50      | 0.80    |
| sinh     | 1.61      | 1.51    |
| cosh     | 0.99      | 1.18    |
| tanh     | 1.98      | 2.71    |
| coth     | 2.54      | 2.53    |
| sech     | 1.97      | 1.88    |
| csch     | 2.39      | 2.17    |
| sin      | 0.50      | 0.72    |
| cos      | 0.50      | 0.68    |
| ln       | 0.50      | 0.74    |
| log10    | 1.50      | 1.50    |

(`sin` and `cos`: 12,000 points in [-10, 10], [-π/4, π/4], [-1.6e6, 1.6e6] and within 1e-9 of
multiples of π/2.)

Special values follow C: `exp(-inf) = 0`, `sinh(-0) = -0`, `tanh(±inf) = ±1`,
`coth(-0) = -inf`, `csch(-inf) = -0`, NaN gives NaN, `exp` and `sinh`/`cosh` overflow where the
results do (709.78 and 710.48). `sech` and `csch` keep their subnormal results
(`sech(720) ≈ 4.1e-313`), where the C library's `1 / cosh(x)` is 0.

`sin` and `cos`: x = n π/2 + r with π/2 in four parts (fdlibm's, the first three of 33 bits, so
the products with n < 2^20 are exact); r is kept as a sum of two doubles and passed to musl's
minimax polynomials with their corrections for the second part, and the quadrant n mod 4 is
chosen with selects. `sin(x) = x` for |x| < 2^-26, which keeps `sin(-0) = -0`.

How the hyperbolic functions are computed: for |x| < 1, the Taylor series of sinh and cosh
(no cancellation near 0); otherwise sinh and cosh from h = e^|x| / 2, scaled inside `exp` so
that it does not overflow before sinh and cosh do, and the others from e = e^-|x|, which does
not overflow at all: tanh = (1 - e²) / (1 + e²), sech = 2e / (1 + e²), and so on.

The complex hyperbolic functions of z = x + iy, normwise against mpmath (the largest error in a
component in ulps of |f(z)|; 3,000 points each in |x|, |y| < 3, |x| < 30 with |y| < 100,
|x| < 705 with |y| < 10, and within 1e-3 of the poles i k π/2):

| function | C path | inlined |
|----------|--------|---------|
| sinh     | 2.2    | 2.0     |
| cosh     | 1.8    | 2.0     |
| tanh     | 3.8    | 3.4     |
| coth     | 5.5    | 3.9     |
| sech     | 3.6    | 3.8     |
| csch     | 3.2    | 4.1     |
| exp      | 1.4    | 1.5     |
| sin      | 1.9    | 1.9     |
| cos      | 1.8    | 2.1     |

(Before patch v228-0020, the C path's tanh, `tanh_c` in code.rs, divided sinh 2x by
cosh 2x + cos 2y, which cancels near its poles: 8.6e7 ulp for tanh and 1.4e8 for coth, its
reciprocal; it now uses the library's formula.) They are sinh z = sinh x cos y + i cosh x sin y,
cosh z = cosh x cos y + i sinh x sin y, and the others as conj(numerator) / |denominator|²,
whose denominators are sums of squares, e.g. tanh z = (sinh x cosh x + i sin y cos y) /
(sinh² x + cos² y); for |x| > 22 their limits, which do not overflow (sech and csch keep their
subnormal results). sinh x and cosh x share one exponential and sin y and cos y one reduction.

The complex exp, sin and cos are e^x cos y + i e^x sin y, sin x cosh y + i cos x sinh y and
cos x cosh y - i sin x sinh y.

`ln` and `log10` (their errors above: 12,000 points in (0, 4], up to e^±700, near 1, and
subnormal): x = 2^k m with m in [√2/2, √2) (`getexp`, `getmant`; subnormals are scaled by 2^54 first), and
fdlibm's polynomial for log(m); special values by selects (ln(±0) = -inf, ln(x < 0) = NaN,
ln(inf) = inf).

## Writing a function

A function is generic over `libm::MathOps`, the operations it may use: `+`, `-`, `*`, `/` (also
with an `f64`), `abs`, constants (`x.lit(v)`), comparisons with a constant (`x.gt(v)`,
`x.lt(v)`), selects (`T::select(mask, a, b)`, both computed), and the exponent-field
operations `pow2i`, `getexp` and `getmant`. It has no branches. The same source then gives:

- **`f64`**: the reference implementation, `libm::sinh(1.5f64)`.
- **`kernel::Real`** (with the `composer` feature): `libm::sinh(x)` adds the operations to a
  kernel being built with `KernelBuilder`.
- **traced values** (`libm::Val`): the inlined MIR. The trace records the operations once,
  and `Trace::emit` writes them with its own register allocation and, with `fastmath`, fused
  multiply-adds for products used once. The funclets are appended after the MIR optimizations
  (`Builder::optimize_mir`), because the peephole fusions assume that every value is used
  once, and use no stack slots, because the compactor could give one of the caller's slots,
  live across the call, to the funclet.

`x > c` is computed as `c < x` (the x86 `gt` is true for NaN), and selects use `xor` and `and`
(the ARM `andnot` takes its operands the other way round).

To add a function, write it in `libm.rs` and add it to `LIBRARY` (and to `c_function` and
`library_function` in the tests); a function that is only accurate below some |x| gets it in
`domain_limit`. A function must fit in 12 registers (`library_fits_in_registers` prints what
each needs: `sin` 10, the complex `tanh` 12). Complex functions go in `COMPLEX_LIBRARY`, as
functions of (x, y) giving (real part, imaginary part); their funclets take and return
`Ret`/`Temp`. Then add it to `libm::tests`
(`cargo test --lib --features math,composer libm`) and `python/tests/test_math.py`.

## The exponent-field operations

These are in the default build, as model functions (`pow2i`, `getexp`, `getmant`) and as
methods of `Expr`, `kernel::Real` and the Python `Composer`. They are defined bit for bit
(`utils.rs`), so every backend gives the same results:

| operation    | result                                                            |
|--------------|-------------------------------------------------------------------|
| `pow2i(x)`   | 2^x for an integer x in [-1022, 1023]: the low 12 bits of round(x + 1023) become the sign and the exponent |
| `getexp(x)`  | the exponent field minus 1023: floor(log2(abs(x))) for normal numbers, -1023 for zeros and subnormals, 1024 for infinities and NaN |
| `getmant(x)` | x with the exponent of 1.0: the mantissa in [1, 2), with the sign of x |

Complex kernels apply them to the real part and return a real.

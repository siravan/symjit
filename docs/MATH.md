# symjit's own math library

The `math` cargo feature adds `rust/libm.rs`, a math library written in generic Rust whose
functions the SIMD kernels evaluate inline, on all lanes at once. Without it, a SIMD kernel
calls the scalar C function once per lane. The library has `exp` and the six hyperbolic
functions `sinh`, `cosh`, `tanh`, `coth`, `sech` and `csch`.

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
- **Complex kernels**: `complexify.rs` tests the imaginary part of the argument at run time.
  When it is zero, the real function is called (the funclet in SIMD kernels), otherwise the
  complex C function. A SIMD row whose lanes disagree is computed again by the scalar kernel.
  The test is skipped when the argument is known to be real. A zero imaginary part of the
  argument gives +0 as the imaginary part of the result (the complex C functions give a signed
  zero), and `coth`/`csch` of a real 0 give an infinity.

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

Complex kernels (AVX2, `sinh`): real arguments 25.3 ms → 7.1 ms; complex arguments 33.1 ms,
unchanged; 1% of the arguments complex 25.4 ms → 8.5 ms.

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

Special values follow C: `exp(-inf) = 0`, `sinh(-0) = -0`, `tanh(±inf) = ±1`,
`coth(-0) = -inf`, `csch(-inf) = -0`, NaN gives NaN, `exp` and `sinh`/`cosh` overflow where the
results do (709.78 and 710.48). `sech` and `csch` keep their subnormal results
(`sech(720) ≈ 4.1e-313`), where the C library's `1 / cosh(x)` is 0.

How the hyperbolic functions are computed: for |x| < 1, the Taylor series of sinh and cosh
(no cancellation near 0); otherwise sinh and cosh from h = e^|x| / 2, scaled inside `exp` so
that it does not overflow before sinh and cosh do, and the others from e = e^-|x|, which does
not overflow at all: tanh = (1 - e²) / (1 + e²), sech = 2e / (1 + e²), and so on.

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
`library_function` in the tests). Then add it to `libm::tests`
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

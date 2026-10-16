# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Symjit (2.28.0) is a JIT compiler that turns SymPy/Symbolica expressions into machine code (AMD64 with
SSE/AVX/AVX2/AVX-512, ARM64/NEON, RISC-V64) without LLVM. It is a Rust `cdylib`/`rlib` (`rust/lib.rs`, crate
`symjit`) wrapped by a thin Python package (`python/symjit`) that talks to it over a C ABI with `ctypes`. The
crate also has a public Rust API (`Compiler`, `Translator`, `Composer`, `Expr`, `Applet`, `Application`,
`Config`, `Defuns`, ...; see the `pub use` list in `lib.rs`). The repository is a Cargo workspace whose other
member, `macros/`, is the `symjit-macros` proc-macro crate (`#[kernel]`). Object files, WebAssembly modules
and the `sjc` compiler are the separate crate `../sjc`. Prose docs are in `docs/` (KERNELS, MATH; OBJ, WASM
and SJC are in `../sjc/docs`).

## Repository layout

This tree is `dev/` inside `symjit-playground/`, next to the published version and the patch series:

| Path (from `symjit-playground/`) | What it is |
|---|---|
| `core/` | **Crate `symjit-core`** (since the split, `plans/split-crates/PLAN.md`): the IR (`mir.rs`, `serializer.rs`, ...), the front ends (`Translator`, `Composer`, `KernelBuilder`, `Builder`, `Compilation`/`Model`, the `.sjb` format) and the code generators (`amd/`, `arm/`, `riscv64/`); nothing that runs code. Its own workspace and `Cargo.lock`; `dev/` depends on it by path (`../core`). See `core/CLAUDE.md`. |
| `dev/` | **This tree, the main working copy** (the JIT, crate `symjit`, on top of `core/`): 2.28.0, the published 2.27 plus the patches `v228-0001`..`0029` plus the crate split (no patches since: archived with tar) (complex division, the `composer` and `math` features, the `sympy` feature gating the C interface, compression-mode AVX-512 work, the exponent-field operations, `log` = `ln`, ...). All new work goes here. |
| `sjc/` | **Crate `sjc`** (since the split): the compiler driver (`.sjb` or `#[kernel]` Rust source → object file + C header, or WebAssembly module), with the object writer (`src/linker/`) and the wasm backend (`src/wasm/`); on `core/` only (symjit is a dev-dependency of its tests). Its own workspace. See `sjc/CLAUDE.md`. |
| `v227/` | **The published version, 2.27.1**, with `rust/mir.rs` split into `rust/mir/` and the 2.27 bug-fix backport applied (`v227-backport-bug-fixes-split-mir.patch`). Only for 2.27 maintenance. It has no `Cargo.lock` and no `target/`: build it with `cargo build --release --lib --offline` and an outside `CARGO_TARGET_DIR`, and delete the `Cargo.lock` that creates. |
| `patches/` | `v228-NNNN-<topic>.patch`: the series that turns 2.27 (after the `v227-*` patches below) into `dev/`, in order; each change to `dev/` adds the next one. `v227-*.patch`: the patches applied to the published 2.27 before `v228-0001` (`five-fixes`, `tests-fixes`, `recursion-constants-fixes`, `pi-test`, `composer-complex-recursion`, `split-mir`). `v227-backport-bug-fixes.patch` (fixes 1–10, for 2.27 after `tests-fixes`, with or without `split-mir`) and `v227-backport-bug-fixes-split-mir.patch` (fixes 3–10, for the tree just before `v228-0001`): bug fixes of `dev/` that are independent of the four new features. |
| `plans/` | Plans not implemented yet, e.g. `plans/cuda-2.29/PLAN.md` (a PTX/CUDA backend for 2.29). |
| `rust/`, `python/`, `examples/`, `CLAUDE.md`, ... at the root | **Archived**: the earlier development line (its own `experimental` and `async` features, GL297 and integration studies). Read-only reference; do not port changes to it. Its data is still used, e.g. `examples/symbolica/*.txt` (the 1-, 2-, 3-loop and benchmark evaluators) and `GL297_study/`. |

Changes are recorded as patch files, not commits: after a change to `dev/`, write the next
`patches/v228-NNNN-<topic>.patch` with `diff -ruN` against a copy of `dev/` at the previous patch, and check that
`patch -p1` on that copy reproduces `dev/`.

## Build, test, run

`dev/` holds the JIT and the Python package; the IR, front ends and code generators are in `../core`
(crate `symjit-core`), which `rust/lib.rs` re-exports under the old module names (`pub(crate) use
symjit_core::{mir, config, ...}`), so `super::mir::Mir` still works in dev's modules. The front ends end in
`finish()` (a `Compilation`); dev adds `compile()` through extension traits: `symjit::CompileExt` (every
`Composer`, e.g. `Translator`) and `symjit::kernel::KernelCompile` (`KernelBuilder`), both in
`symjit::prelude` — Rust code calling `.compile()` on them needs `use symjit::prelude::*;`. The `#[kernel]`
macro calls `symjit::kernel::compile(&builder, outs)`; `symjit::kernel` re-exports core's builder and adds
`compile`, `ops::add_kernel` and the macro (`rust/kernel.rs`). Tests of core modules that run kernels are in
`rust/core_tests/`. Core's own tests: `cd ../core && cargo test --release --offline` (~2600, the encoding
tests among them).

```bash
cargo build --release --lib              # -> target/release/libsymjit.so
cargo test --lib                         # Rust unit tests (~2400, < 1 s)
cargo test --lib amd::tests              # one module; add a test name to run a single test
cargo test --release                     # unit + integration tests (tests/*.rs)
RUST_MIN_STACK=67108864 cargo test       # debug builds: tests/funcs.rs `test_args` overflows the default stack
cargo test --features composer           # adds the kernel builder/macro tests (tests/kernel_*.rs) and doc tests
cargo test --lib --features math,composer   # adds the libm::* tests (own math library, docs/MATH.md)
cargo run --release --features composer --example rk4_bessel       # examples/claude/*.rs (Rust)
cargo run --release --features composer --example pi_hypersphere -- 7
```

Object files, wasm modules and sjc: `cd ../sjc && cargo build --release && cargo test --release` (see
`../sjc/CLAUDE.md`; its tests use this crate, with `composer`, as the reference).

Cargo features (`Cargo.toml`): `async` (**default**: the lockstep/coroutine prototype, `rust/lockstep.rs`),
`sympy` (**default**: the C interface the Python package calls, `rust/sympy.rs`; without it the library
exports no C functions and has only the Rust API, so keep it on for any build the Python package loads),
`composer` (`rust/kernel.rs` + `macros/`), `math` (`symjit_core::libm`, symjit's own math library;
not to be confused with `libm`, which links the C library's complex functions), `symbolica`, `libm`,
`selinux-fix`. Build with
`--lib`, or the `profiler` bin (`profile/profiler.rs`, a Samply driver that reads `3loop_instructions_2.txt`
from the working directory) is built too. `--all-features` does not compile (`selinux-fix`).

Integration tests take their options from `tests/symjit.toml` through `tests/common/mod.rs` `base_config()`
(it sets `SYMJIT_TOML` once); every `Config` in `tests/` must come from it, so that a `symjit.toml` in the
working directory cannot change the tests. List every key in that file: a missing one reads as 0/false.
`tests/kernels.rs` writes `test.bin` (a kernel dump) into the working directory.

Cross-testing under qemu-user (installed: the two Rust targets, `gcc-aarch64-linux-gnu`,
`gcc-riscv64-linux-gnu`, `qemu-user`):

```bash
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER="qemu-aarch64 -cpu max -L /usr/aarch64-linux-gnu" \
    cargo test --lib --target aarch64-unknown-linux-gnu          # also `-cpu cortex-a57` (no FCMA)
CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_LINKER=riscv64-linux-gnu-gcc \
CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_RUNNER="qemu-riscv64 -L /usr/riscv64-linux-gnu" \
    cargo test --lib --target riscv64gc-unknown-linux-gnu
```

RISC-V has only the scalar kernel: tests that need SIMD kernels are
`#[cfg_attr(target_arch = "riscv64", ignore = ...)]`.

### Python

The package does **not** build the library on import: `python/symjit/engine.py` loads the
`python/symjit/_lib.*.so` whose name matches the platform (`x86_64-linux`, `aarch64-linux`, ...). After changing
Rust code, rebuild and copy the library over it (or `pip install -e .`, setuptools-rust, `pyproject.toml`), or
test a scratch copy of the package without touching the checked-in one:

```bash
cp target/release/libsymjit.so python/symjit/_lib.cpython-312-x86_64-linux-gnu.so
python -m unittest discover -s python/tests                  # all Python tests
python python/tests/test_regressions.py Regressions.test_real_kernel_times_imaginary_unit
SYMJIT_PYTHON_PATH=/path/to/pkg python -m unittest discover -s python/tests   # another build of the package
SYMJIT_FUZZ_SEEDS=1500 SYMJIT_FUZZ_START=5000 python python/tests/test_fuzz.py   # bigger fuzz run
```

A stale `_lib*.so` is the most common reason for Python tests failing after a Rust fix. Tests that use the
`symbolica` package abort inside a sandbox ("Could not acquire Symbolica thread permit"); run them unsandboxed.

`python/tests`: `test_regressions.py` (a minimal repro per fixed bug; `expectedFailure` = known open bug),
`test_fuzz.py` (random expressions × ~10 option sets against numpy; `SYMJIT_FUZZ_VERBOSE=1` prints each case,
needed to find a process-aborting panic), `test_composer_fuzz.py` (the `Composer` interface against a numpy
interpreter, one subprocess per family), `test_accuracy.py`/`accuracy_cases.py` (ulp errors against mpmath),
`test_special_values.py` (NaN/Inf/±0/subnormals), `test_exponent_ops.py` (`pow2i`/`getexp`/`getmant` bit for
bit), `test_math.py` (feature `math`; skips without it), `test_disasm.py`, `test_api.py` (the public API
against independent references: `compile_ode`/`compile_jac` with `solve_ivp`, CellML models, save/load,
`callable_quad`, `defuns`, `Sum`/`Product`, evaluators, `coroutines`), `test_properties.py` (every execution
path and option gives the bytecode interpreter's bits with `fastmath=False`; identities in ulps) and
`test_emulated.py` (opt-in, `SYMJIT_EMULATED=1`: cross-compiles the library for aarch64 and riscv64 and runs
~180 models through `emulated_driver.c`, the C ABI as `engine.py` calls it, under qemu-user, compared with this
machine's bytecode interpreter; needs `qemu-user`, `gcc-aarch64-linux-gnu`, `gcc-riscv64-linux-gnu` and the
two Rust targets; builds its option words with `engine.pack_options`). `test_api` (CellML, `Lockstep`),
`test_emulated` (the 1- and 2-loop evaluators) and one regression test read `examples/cellml` and
`examples/symbolica/*loop_instructions_2.txt` and skip (or leave those cases out) while they are missing.
The tests of object files and wasm modules made from saved functions (`test_obj.py`, `test_wasm.py`,
`test_sjc.py`) are in `../sjc/tests/python` and import `test_fuzz`, `test_composer_fuzz` and `test_regressions` from here.

Config defaults come from `Config::default()` in `rust/config.rs`, overridden by a `symjit.toml` in the working
directory or the file named by `SYMJIT_TOML`. Python options (`opt_level`, `use_simd`, `enable_simd512`, `cse`,
`fastmath`, `fast_complex`, `compress`, `compact`, `direct`, `coroutines`, ...) are packed into a bit field in
`RustyCompiler.__init__` (`engine.py`) that must match the constants in `rust/config.rs`. `ty` strings:
`native`, `amd`, `amd-avx`, `amd-sse`, `arm`, `riscv`, `bytecode`, `debug`, `wasm` (runs on bytecode here; a
model to save for sjc's wasm modules: `compress` is off), or a `.toml` path;
`Config::from_name` also takes `"native;key=value;..."` (applied with `set_option`). `coroutines=True`
(bit `0x4000_0000`) turns on lockstep evaluation with fixed `yield_every` 10000 and `lockstep` 256.
`measure(...)` keys: `ker-scalar-size`, `ker-simd-size`, `ker-fast-size`, `mir-size`, `mir-bytes`,
`registers`, `stack-size`, `async`, `version`. Saved `.sjb` files are version 4 (`load_with_version` reads 3 and 4).

## Architecture (the pipeline spans many files)

1. **Python front end** (`structure.py`): converts SymPy expressions to a JSON tree (`Tree`/`Const`/`Var` nodes,
   `ifelse`, `Sum`/`Product` loops). `expr()` does not memoize, so shared subexpressions are expanded in the JSON
   and re-shared later by the Rust CSE. `func.py` (`Func`, `FuncComplex`, `SymbolicaFunc`) wrap a compiled
   application; `ode.py`/`composer.py` add ODE right-hand sides and the low-level `Composer` builder;
   `disasm.py` disassembles dumped kernels. `compile_evaluator` takes Symbolica evaluators or their instruction
   text (`([instructions], temp_count, [constants])`, parsed by `parser.rs`/`instruction.rs`).
2. **FFI and model** (`sympy.rs`, `model.rs`, `parser.rs`, `applet.rs`): `compile`/`translate`/`dump`/`measure`
   are `extern "C"` (feature `sympy`, re-exported at the crate root by `lib.rs`); a Rust panic here aborts the
   Python process. Two translators turn instructions/Composer
   calls into a program: `IndirectTranslator` (`compiler.rs`; builds a `Node` tree through `builder.rs`, which
   also rewrites some patterns, e.g. `x**1.5`, `x * i` → `times_i` in complex mode) and `DirectTranslator`
   (`composer.rs`; emits MIR directly; needed for kernel loops).
3. **MIR** (`mir.rs`, `serializer.rs`, `block.rs`): a linear register IR with `Reg::Gen(n)`, `Reg::Static`, and
   two reserved scratch registers, `Reg::Ret` (call results, first argument) and `Reg::Temp` (imaginary part of
   call arguments/results, constants loaded by fused ops). `mir/peephole.rs` (`optimize_peephole`, FMA/recip/
   save-load/times2 fusions) runs at opt level ≥ 1; `allocator.rs` (levels 2–3) and `compactor.rs` (reuses
   `Stack[..]` slots) follow; `mir/interpreter.rs` is the bytecode interpreter. Complex numbers: `complexify.rs`
   lowers complex MIR to real MIR, while "fast complex" uses packed-`f64x2` generators (`amd/complex.rs`,
   `arm/complex.rs`).
4. **Code generation**: the `Generator` trait (`generator.rs`) per architecture and kernel flavor:
   `amd/{scalar,sse,vector(f64x4),vector_f64x8,complex}.rs`, `arm/{scalar,vector,complex}.rs`, `riscv64/mod.rs`.
   Instructions go through `amd::asm::Amd` (byte-level x86 encoder; VEX, and EVEX in `f64x8.rs` `Prefix`, also
   used by `asm.rs` for xmm16–xmm31) or the `arm!{}`/`rvv!{}` macros. `assembler.rs` holds the byte buffer,
   labels and jump patching; `machine.rs` maps it executable; `runnable.rs` (`Application`) owns the kernels.
   `Mir::rerun` replays the MIR into a generator. Object mode (`Config::is_relocatable`, direct calls to libm,
   `generator::object_symbol`, `VirtualTable::c_name`, `Complexifier::for_object`) is in core (feature `obj`);
   the object writer, the wasm backend and sjc's Rust input are in `../sjc` (`sjc/CLAUDE.md`). sjc's Rust input
   interprets `#[kernel]` functions mirroring the macro (`macros/src/lib.rs`) and the `ops` of
   `symjit_core::kernel`: a change to the macro's semantics must be made in both (`sjc/tests/source.rs`
   compiles the same corpus both ways and requires identical MIR).
   **Layering (`plans/split-crates/PLAN.md`; these modules are in `core/`):** front ends end in
   `compilation.rs` `Compilation { prog, reals, mir }` (`Composer::finish`, `KernelBuilder::finish`,
   `Compilation::from_program`); `Model::new` is the machine-code-free half of building an application
   (complexify, `original`, counts, `can_fast`), and `write_sjb`/`read_sjb`/`check_registers` the `.sjb`
   format; `Application::from_model` adds the machine code. `Translator`/`IndirectTranslator` are in
   `translator.rs`; `Compiler`/`FastFunc` stay in `compiler.rs`. The runtime-only pieces: `machine.rs`
   (`MachineCode`, the `Compiled` trait), `bytecode.rs` (`CompiledMir`), `applet.rs`, `lockstep.rs`
   (registers `_yield_` with `code::register_yield_point`). Kernels callable from kernels are
   `code::Callee` trait objects (`Func::App(Arc<dyn Callee>)`, implemented by `Applet`).
5. **Kernels**: each function is compiled up to three times — `scalar` (one point), `simd` (vectorized calls,
   rayon threads over chunks), and `fast` (register arguments, few inputs). `ty="bytecode"` interprets the MIR.
   Machine code is followed by a constant pool (constants and function pointers), so raw dumps do not
   disassemble cleanly to the end. A SIMD kernel with diverging lanes (`simd_branch=false`) returns non-zero
   and `evaluate_matrix` recomputes those rows with the scalar kernel.
6. **Compression mode** (`compress=True`, `topology.rs`, `statement.rs`): repeated expression shapes become
   funclets. Arguments go to the funclet's spill area (`Config::location`) through registers: the plain entry
   loads argument *i* into vector register *i* (16 registers, 32 with AVX-512: `vector_f64x8`); with
   `Config::use_xmm16_31()` at opt level 3 (`wide_args`; below it every call site loads the arguments,
   and EVEX loads of xmm16–xmm31 are a byte longer, so it keeps xmm*i*), `amd/complex.rs` passes
   arguments 0–15 in xmm16–xmm31 and 16–31 in xmm0–xmm15 (`funclet_reg`), and a body without calls
   (`Mir::funclet_calls`, `Generator::begin_funclet`) keeps arguments 0–15 there: `load_stack`/
   `save_stack` of their spill slots become register moves (`kept_arg`). Nothing else in that kernel
   uses xmm16–xmm31, but calls clobber them, so a body with calls saves every argument. The `_ultra` entry (opt level 3, all
   arguments stack slots) gets their slot numbers packed 4 × 16 bits per GPR (`pack_locs`,
   `SUBROUTINE_ARGS` in `amd/mod.rs`: 8 registers on Unix, 6 on Windows where rsi/rdi are non-volatile).
   The `_ultra` entry is emitted for every funclet at every opt level.
7. **Kernel builder and `#[kernel]`** (feature `composer`, `docs/KERNELS.md`): `rust/kernel.rs` is a typed
   builder (`KernelBuilder`, `Real`/`Cplx`/`Mask` handles, `ops` module for the macro) over `Translator`;
   `macros/src/lib.rs` rewrites what Rust cannot overload (`if`/`while` on kernel values, comparisons,
   assignments passed through branches, calls of other kernels/callbacks). Kernel loops need the direct
   translator (set automatically). Calls between kernels use `Defuns::add_applet`; Rust callbacks are
   registered as `composer_<name>` (an unprefixed user function gets its arguments in `__Arg` slots and returns
   garbage).
8. **Exponent-field operations** (`UniOp::Pow2i`/`GetExp`/`GetMant`; model functions `pow2i`, `getexp`,
   `getmant`, also `Expr`, `Real` and the Python `Composer`): defined bit for bit by `utils::pow2i`/`getexp`/
   `getmant` (the interpreter calls them), so every backend gives the same bits for every input. x86 uses
   `vpsllq`/`vpsrlq` and RIP-relative constants that `predefined_consts` emits only when referenced
   (`Assembler::is_referenced`, so other kernels do not change); ARM builds constants in x9 and uses T2 (v31);
   RISC-V uses t0/t1/ft11; complex generators apply them to the real part and return a real. Tests:
   `utils::exponent_tests` (also under qemu), `python/tests/test_exponent_ops.py`, sjc's `test_wasm.py`.
9. **Own math library** (feature `math`, `rust/libm.rs`, `docs/MATH.md`): `exp`, `ln` (`log` is an alias
   of `ln` everywhere since v228-0021; `log10` is decimal), `log10`, the six hyperbolic functions, `sin` and `cos` (inlined for |x| < `libm::TRIG_LIMIT` only: `libm::domain_limit`, a test at
   the call site and in `complexify`), and complex exp/sin/cos/hyperbolics (`COMPLEX_LIBRARY`, funclets
   `@libm_cplx_<name>` of `Ret`/`Temp`, `Complexifier::call_complex_inline` with a domain test; scalar
   kernels map them back to the complex C function, `mir::libm_c_function`), written once as generic Rust over `libm::MathOps` (branch-free): `f64` (reference),
   `kernel::Real` (builder kernels) and traced values (`Trace`, emitted as MIR with its own register
   allocation and fused multiply-adds, no stack slots). With `Config::inline_math`
   (`ty="native;inline_math=true"`), `libm::inline_calls` (end of `Builder::optimize_mir`, after the
   peephole pass, whose fusions assume single-use values) turns real calls into calls of funclets
   `@libm_<name>` (between `Branch`/`Label` `mir::LIBM_END`). Only SIMD generators run them
   (`Generator::inlines_math`); `Mir::rerun` and the interpreter turn the calls back into C calls for the
   others. Complex kernels: `Complexifier::call_real_or_complex` tests im == 0 at run time (`branch_if`
   with is_else = true: SIMD rows with mixed lanes bail out to the scalar kernel, now also in
   `Application::exec_vectorized_simd`) and appends the funclets itself (`libm::append_funclets`; their
   constants are copied to the builder in `Application::with_mir`). Selects use `xor`/`and` and `x > c`
   is `c < x`: ARM's `andnot` is s1 & !s2 (others !s1 & s2), and x86 `gt`/`geq` are unordered (true for
   NaN). ARM/RISC-V keep the return address around `@libm_` calls (LR2/t6).

## Things that bite

- **Logarithm names**: symjit's `ln` and `log` are the natural logarithm (C's `log`), `log10` the decimal one
  (`VirtualTable::from_str`/`c_name`, `cplx_log`, sjc's `wasm_runner.mjs`); `Expr::log10`, `Real::log10` and the Python
  `Composer.log10` emit `log10`. Before v228-0021, real kernels' `log` was log10.
- **`Reg::Temp`/`Reg::Ret` aliasing**: many generator methods load a constant into `Temp` and then read `s1`; if
  `s1 == Temp` the source is destroyed. Peephole fusions must not fuse when an operand is `Temp`. Prefer
  RIP-relative operand forms (`*_label` assembler methods) over scratch registers.
- **`branch_if(cond, label, is_else)`** jumps when `cond` is *not* zero if `is_else` is true, and when it is zero
  otherwise — the opposite of what the name suggests.
- **Kernel-convention calls** (`Config::is_kernel_func`: recursive functions, `@self`, applets) take their
  arguments in the `__Arg` stack slots, not in `Left`/`Right`. Native recursion has no depth check.
- **Real kernels and complex-only operations**: the real generators leave `times_i`/`times_neg_i` (and similar
  complex-only methods) empty; `Mir::times_i` emits `xor` for real kernels instead. A generator method that
  emits nothing leaves its destination register holding a stale value — results then depend on register
  allocation and differ between option sets.
- **Stack slots**: a peephole that removes a `Save` to `Stack[k]` must know `k` is never loaded again; the
  compactor panics with `cannot find Stack[..]` otherwise.
- **`amd/mod.rs` register choices**: `MEM=R13`, `IDX=R12`, `STATES=R14`, `PARAMS=R15`, `STACK=RBX`, RBP is the
  frame pointer. The assembler handles the RSP/R12 (SIB) and RBP/R13 (mod=00) cases; EVEX uses disp8*N
  (`Prefix::modrm_mem_n`, which always emits a disp8 for offset 0).
- **AVX-512 `xmm16`–`xmm31`** need AVX512VL (`Config::use_xmm16_31`), not just AVX512F (`use_simd512`).
- **Non-reproducible machine code**: compressed kernels (`Topology::subs` is a `HashMap`) and kernels calling
  math functions (function addresses, table order) differ from run to run; compare their results or sizes.
- **Encoding tests** (core's `src/{amd,arm,riscv64}/tests.rs`, sjc's `src/wasm/tests.rs`) pin instructions
  against an external assembler (GNU `as`, Keystone, or Capstone-checked encoders; `gen_tests.py` with wabt). Keep them generated,
  not hand-edited; record intentional deviations as `NOTE:` comments or `#[ignore = "reason"]`.
- **`tests/ui/*.stderr`** (trybuild, `tests/kernel_ui.rs`) hold rustc's own diagnostics and were written with
  rustc 1.99; after a toolchain change, `TRYBUILD=overwrite cargo test --features composer --test kernel_ui`.
- **Never run rustfmt on `lib.rs` or a `mod.rs`**: it recurses into every module (several files are not
  rustfmt-clean). Format single leaf files, or check with `rustfmt --edition 2024 --check <file>`. The crates
  are edition 2024, formatted in the 2021 style (`rustfmt.toml`: `style_edition = "2021"`, also in `../core`
  and `../sjc`; the 2024 style would reformat most files).
- **Known open defects**: RISC-V `round`/`trunc`/`floor`/`ceiling` go through `fcvt.l.d` (sign of zero lost,
  wrong for |x| ≥ 2^63, ±inf, NaN; `tests/kernel_builder.rs` `real_arithmetic`/`real_functions` fail on
  RISC-V); Symbolica evaluators with `ty="bytecode"` return zeros; malformed evaluator constants panic in
  `compiler.rs` instead of returning an error; the two `kernels.rs` coef tests are ignored
  (`direct_arena_identity_output(false)` is not implemented).
- `python/symjit/pyengine` is the obsolete pure-Python backend, and `ty="amd-sse"` (`amd/sse.rs`) is obsolete;
  don't extend either.

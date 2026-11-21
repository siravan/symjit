# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Symjit (2.28) is a JIT compiler that turns SymPy/Symbolica
expressions into machine code (AMD64 with SSE/AVX/AVX2/AVX-512, ARM64/NEON, RISC-V64) without LLVM. It is a
Rust `cdylib`/`rlib` (`rust/lib.rs`, crate `symjit`) wrapped by a thin Python package (`python/symjit`) that
talks to it over a C ABI with `ctypes`. The crate also has a public Rust API (`Compiler`, `Translator`,
`Composer`, `Expr`, `Applet`, `Application`, `Config`, `Defuns`, `Matrix`, ...; see the `pub use` list in
`lib.rs`), and `symjit::backend` re-exports the internals an outside code generator needs (`Generator`,
`Mir`, `Builder`, `Complexifier`, `Reg`, `Loc`, ...; used by
[symjit-utils](https://github.com/siravan/symjit-utils)). Prose docs are in `docs/`
(COMPILATION, COMPOSER, FUNCTIONS, INSPECTION, OPTIMIZATION, SYMBOLICA, COMPANIONS).

Branches: `v228` is the 2.28.x line (2.28.0, 2.28.1, ...), `v227` the 2.27.x line; each minor version
gets its own branch (`v229` for 2.29).

## Build, test, run

```bash
cargo build --release --lib                  # -> target/release/libsymjit.so
cargo test --release                         # ~2200 unit + integration tests
cargo test --lib amd::tests                  # one module; add a test name to run a single test
cargo test --release --test call_capacity    # one integration test file
```

`tests/kernels.rs` writes a `test.bin` in the working directory; neither it nor `Cargo.lock` is part of the
repository.

Build with `--lib`, or the `profiler` bin (`profile/profiler.rs`, a Samply driver that reads
`3loop_instructions_2.txt` from the working directory) is built too. Cargo features: `async` (**default**,
the lockstep/coroutine evaluation, `rust/lockstep.rs`, corosensei), `sympy` (**default**, the C interface the
Python package calls, `rust/sympy.rs`; without it the library has only the Rust API), `symbolica`, `libm`
(links the C library's complex functions instead of num_complex's formulas), `selinux-fix`.

Integration tests take their options from `tests/symjit.toml` through `tests/common/mod.rs` `base_config()`
(it sets `SYMJIT_TOML` once); every `Config` in `tests/` must come from it, so that a `symjit.toml` in the
working directory cannot change the tests. List every key in that file: a missing one reads as 0/false.
Note that it sets `symbolica = true` (inputs are parameters): tests of `Compiler`/`Expr` models call
`config.set_symbolica(false)`.

Cross-testing under qemu-user (needs the two Rust targets, `gcc-aarch64-linux-gnu`,
`gcc-riscv64-linux-gnu`, `qemu-user`):

```bash
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER="qemu-aarch64 -cpu max -L /usr/aarch64-linux-gnu" \
    cargo test --release --target aarch64-unknown-linux-gnu      # also `-cpu cortex-a57` (no FCMA)
CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_LINKER=riscv64-linux-gnu-gcc \
CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_RUNNER="qemu-riscv64 -L /usr/riscv64-linux-gnu" \
    cargo test --release --target riscv64gc-unknown-linux-gnu
```

RISC-V has only the scalar kernel: tests that need SIMD kernels are
`#[cfg_attr(target_arch = "riscv64", ignore = ...)]`; tests that run x86 code (`tests/sse_complexify.rs`)
are `#![cfg(target_arch = "x86_64")]`, `tests/bug_fp_preserve.rs` is aarch64-only, and
`tests/win64_nonvolatile.rs` is Windows x64-only.

On Windows the library is `symjit.dll` and the Python package loads `python/symjit/_lib.cp312-win_amd64.pyd`.
Linux and macOS cannot catch Win64 calling-convention bugs (see "Win64 non-volatile registers" below), so
run the tests on a Windows x64 machine after changing the AMD prologues, epilogues or register saves.

### Python

The package does **not** build the library on import: `python/symjit/engine.py` loads the
`python/symjit/_lib.*.so` (`_lib.*.pyd` on Windows) whose name matches the platform. After changing Rust code, copy the fresh library
over it (a stale `_lib*.so` is the most common reason for Python tests failing after a Rust fix):

```bash
cp target/release/libsymjit.so python/symjit/_lib.cpython-312-x86_64-linux-gnu.so
PYTHONPATH=python python -B -m unittest discover -s python/tests      # all Python tests
SYMJIT_FUZZ_SEEDS=1500 SYMJIT_FUZZ_START=5000 python python/tests/test_fuzz.py   # bigger fuzz run
```

Tests that use the `symbolica` package are skipped without it. Unlicensed (restricted mode), Symbolica
locks a thread permit in `$XDG_RUNTIME_DIR`, which must be writable ("Could not acquire Symbolica thread
permit" otherwise).

`python/tests`: `test_regressions.py` (a minimal repro per fixed bug; `expectedFailure` = known open bug),
`test_fuzz.py` (random expressions × ~10 option sets against numpy; `SYMJIT_FUZZ_VERBOSE=1` prints each case,
needed to find a process-aborting panic), `test_composer_fuzz.py` (the `Composer` interface against a numpy
interpreter, one subprocess per family), `test_accuracy.py`/`accuracy_cases.py` (ulp errors against mpmath),
`test_special_values.py` (NaN/Inf/±0/subnormals), `test_stable_math.py`, `test_disasm.py`.

Config defaults come from `Config::default()` in `rust/config.rs`, overridden by a `symjit.toml` in the working
directory or the file named by `SYMJIT_TOML`. Python options (`opt_level`, `use_simd`, `enable_simd512`, `cse`,
`fastmath`, `fast_complex`, `compress`, `compact`, `direct`, `stable_math`, `coroutines`, ...) are packed into a
bit field in `RustyCompiler.__init__` (`engine.py`) that must match the constants at the top of
`rust/config.rs` (`stable_math` is `0x0800_0000`, `coroutines` `0x4000_0000`, `ultra` `0x8000_0000`, the last bit). `ty` strings: `native`, `amd`,
`amd-avx`, `amd-sse`, `arm`, `riscv`, `bytecode`, `debug`, or a `.toml` path; `Config::from_name` also takes
`"native;key=value;..."`. `coroutines=True` turns on lockstep evaluation with fixed `yield_every` 10000 and
`lockstep` 256. `measure(...)` keys: `ker-scalar-size`, `ker-simd-size`, `ker-fast-size`, `mir-size`,
`mir-bytes`, `stack-size`, `async`, `version`. Saved `.sjb` files are version 4 (2.27 on: the `Config`
holds the stack limit and argument count); version 3 (2.22–2.26, e.g. pyAmpliCol's kernels, symjit
2.26.4) has neither, and loads with the defaults since 2.28.1 (`tests/sjb_version3.rs`; before, version-3
files failed with `invalid magic number (Program)`). `Application::read_sjb` reads one without compiling it.

Examples (`examples/*.py`) double as smoke tests and take options through `examples/util.py`;
`examples/claude/` holds larger studies; `bench/run.py` (`bench/README.md`) times workloads against a
per-host baseline (run by hand, not in the test suite).

## Architecture (the pipeline spans many files)

1. **Python front end** (`structure.py`): converts SymPy expressions to a JSON tree (`Tree`/`Const`/`Var` nodes,
   `ifelse`, `Sum`/`Product` loops). `expr()` does not memoize, so shared subexpressions are expanded in the JSON
   and re-shared later by the Rust CSE. `func.py` (`Func`, `FuncComplex`, `SymbolicaFunc`) wrap a compiled
   application; `ode.py`/`composer.py` add ODE right-hand sides and the low-level `Composer` builder;
   `disasm.py` disassembles dumped kernels (`disassemble`/`disassembly` with objdump, capstone or
   `tool="symjit"`, the `*-asm-hex` dump; `assembly` returns the `*-asm` text). `compile_evaluator` takes Symbolica evaluators or their instruction
   text (`([instructions], temp_count, [constants])`, parsed by `parser.rs`/`instruction.rs`).
2. **FFI and model** (`sympy.rs`, `model.rs`, `parser.rs`, `applet.rs`): `compile`/`translate`/`dump`/`measure`
   are `extern "C"` (feature `sympy`); a Rust panic here aborts the Python process. Two translators turn
   instructions/Composer calls into a program: `IndirectTranslator` (`compiler.rs`; builds a `Node` tree
   through `builder.rs`, which also rewrites some patterns, e.g. `x**1.5`, `x * i` → `times_i` in complex
   mode) and `DirectTranslator` (`composer.rs`; emits MIR directly; needed for kernel loops).
3. **MIR** (`mir.rs`, `serializer.rs`, `block.rs`): a linear register IR with `Reg::Gen(n)`, `Reg::Static`, and
   two reserved scratch registers, `Reg::Ret` (call results, first argument) and `Reg::Temp` (imaginary part of
   call arguments/results, constants loaded by fused ops). `mir/peephole.rs` (`optimize_peephole`, FMA/recip/
   save-load/times2 fusions) runs at opt level ≥ 1; `allocator.rs` (levels 2–3) and `compactor.rs` (reuses
   `Stack[..]` slots) follow; `mir/interpreter.rs` is the bytecode interpreter. Complex numbers:
   `complexify.rs` lowers complex MIR to real MIR, while "fast complex" uses packed-`f64x2` generators
   (`amd/complex.rs`, `arm/complex.rs`).
4. **Code generation**: the `Generator` trait (`generator.rs`) per architecture and kernel flavor:
   `amd/{scalar,sse,vector(f64x4),vector_f64x8,complex}.rs`, `arm/{scalar,vector,complex}.rs`,
   `riscv64/mod.rs`. Instructions go through `amd::asm::Amd` (byte-level x86 encoder; VEX, and EVEX in
   `f64x8.rs` `Prefix`) or the `arm!{}`/`rvv!{}` macros. `assembler.rs` holds the byte buffer, labels and
   jump patching; `machine.rs` maps it executable; `runnable.rs` (`Application`) owns the kernels.
   `Mir::rerun` replays the MIR into a generator.
5. **Kernels**: each function is compiled up to three times — `scalar` (one point), `simd` (vectorized calls,
   rayon threads over chunks), and `fast` (register arguments, few inputs). `ty="bytecode"` interprets the
   MIR. Machine code is followed by a constant pool (constants and function pointers), so raw dumps do not
   disassemble cleanly to the end. `dump(name, "scalar-asm")` (`"simd-asm"`, `"fast-asm"`) writes a
   kernel as GNU `as` input that assembles back to the same bytes (local labels, `symjit_<kind>` entry;
   spellings chosen for the encoding, e.g. RISC-V `add rd, rs, zero` rather than `mv`); `"scalar-asm-hex"`
   (...) is an objdump-like listing with offsets and bytes. `rust/disasm/` decodes only what symjit's
   encoders emit (RISC-V, AArch64, x86-64). On x86-64 the text carries the pseudo-prefixes that make
   `as` pick symjit's encoding (`{load}`/`{store}`, `{disp8}`/`{disp32}`, `{vex3}`, `{evex}`, `rex.w`),
   and the forms no prefix gives (scalar VEX with L=1: FMA `sd`, `vucomisd`, `vroundsd`; `81` immediates
   that fit a byte or go to `rax`) are `.insn` lines with the instruction in a comment. `MachineCode` keeps the assembler's labels
   (`Generator::take_labels`), and `Generator::mark_data` sets the label `@data` where the constant pool
   starts (no bytes; both default to nothing, for generators outside the crate). The decoder tests are
   generated from the encoding tests (`rust/disasm/gen_tests.py riscv|arm|x86`; the x86 test's code is
   `x86_tests.tail.rs`, and it also assembles the 1156 decoded lines with `as` and compares the bytes); `tests/disasm_listing.rs`
   assembles listings with the cross binutils, when installed, and compares the bytes. A SIMD kernel with diverging lanes (`simd_branch=false`) returns non-zero
   and `evaluate_matrix` recomputes those rows with the scalar kernel.
6. **Compression mode** (`compress=True`, `topology.rs`, `statement.rs`): repeated expression shapes become
   funclets, called with their arguments in the funclet's spill area; the `_ultra` entry (all arguments
   stack slots) gets their slot numbers packed into GPRs (`SUBROUTINE_ARGS` in `amd/mod.rs`).
7. **Stable math** (`stable_math`, `complexify.rs` `stable_*`): range-safe complex division, `1/z`, `abs` and
   `sqrt`, and negative integer powers, inlined in the complexified MIR with power-of-two scaling (exact, so
   in the normal range the bits equal the default code's). `stable_divide` puts the scaled numerator in
   `Gen(0)`, `Gen(1)` and the divisor in `dst`, except for a destination that is not a `Gen` pair (`Left`,
   whose parts are `Ret` and `Temp`: a quotient passed straight to a call by the indirect translator at
   opt level ≥ 1): then the numerator goes to `dst` and the divisor to `Gen(0)`, `Gen(1)`, because the
   generators' `divide_complex` uses `Reg::Temp` as scratch (tests/stable_divide_call_argument.rs).
8. **Lockstep** (feature `async`, `lockstep.rs`): `_yield_` calls inserted every `yield_every` MIR nodes let
   `evaluate_matrix` run `lockstep` points as corosensei coroutines, so each code segment stays in cache.

## Things that bite

- **Logarithm names**: symjit's `ln` is the natural logarithm and the real `log` the decimal one (C's
  `log10`, `Expr::log10`); the complex `cplx_log` is the natural logarithm.
- **`Reg::Temp`/`Reg::Ret` aliasing**: many generator methods load a constant into `Temp` and then read
  `s1`; if `s1 == Temp` the source is destroyed. Peephole fusions must not fuse when an operand is `Temp`.
  Prefer RIP-relative operand forms (`*_label` assembler methods) over scratch registers.
- **`branch_if(cond, label, is_else)`** jumps when `cond` is *not* zero if `is_else` is true, and when it is
  zero otherwise — the opposite of what the name suggests.
- **Kernel-convention calls** (`Config::is_kernel_func`: recursive functions, `@self`, applets) take their
  arguments in the `__Arg` stack slots, not in `Left`/`Right`. Native recursion has no depth check.
- **Real kernels and complex-only operations**: the real generators leave `times_i`/`times_neg_i` empty;
  `Mir::times_i` emits `xor` for real kernels instead. A generator method that emits nothing leaves its
  destination register holding a stale value.
- **Stack slots**: a peephole that removes a `Save` to `Stack[k]` must know `k` is never loaded again; the
  compactor panics with `cannot find Stack[..]` otherwise.
- **Translators**: kernel loops need the direct translator (`config.set_direct(true)`); the indirect one does
  not reassign temporaries, and the loop never ends. The direct translator panics on the function names
  `"minus"`/`"divide"` (`operation.rs`): use `append_div` and an added negation.
- **`Application::call`** returns an empty vector for `ty="bytecode"` (it runs only compiled kernels).
- **ODE models have no vectorized form** (JSON models with `odes`, `count_diffs > 0`): the indirect kernels'
  stack frame holds only the states and observables, and the derivatives were written past it (stack
  corruption). `execute_vectorized`/`execute_matrix` refuse them (Python: a `ValueError`), and
  `Application::exec_vectorized` asserts. Call such models one point at a time, or rewrite the JSON so the
  derivatives are observables.
- **RISC-V immediates**: the `rvv!` macro masks immediates to their field width without a range check; a
  value that may not fit (e.g. an argument count up to `SLICE_CAP` = 2048) must go through `li`, and offsets
  through the `load_*`/`save_*` helpers, which switch to `li` + `add` above 2047.
- **`Complexifier::generic_complex_divide`** (the `Mir::rerun` fallback when a generator's `divide_complex`
  returns false) uses `re(Temp)` = `Gen(0)` as scratch, which holds live values in complexified MIR (e.g.
  `stable_divide`'s scaled numerator). All built-in generators implement `divide_complex`, so it is unused
  by symjit itself; outside generators should implement `divide_complex`.
- **`amd/mod.rs` register choices**: `MEM=R13`, `IDX=R12`, `STATES=R14`, `PARAMS=R15`, `STACK=RBX`, RBP is the
  frame pointer. The assembler handles the RSP/R12 (SIB) and RBP/R13 (mod=00) cases.
- **Win64 non-volatile registers**: Windows x64 requires a function to preserve the low 128 bits of
  xmm6–xmm15 (System V has no callee-saved xmm registers). On Windows, the helpers `save_used_registers`/
  `load_used_registers` in `amd/mod.rs` (called by `amd/{scalar,complex,sse}.rs`) keep xmm6–xmm10 16 bytes
  wide in the spill slots `Stack[6..15]` (slots 0–3 are the callees' home area, 4–5 the complex call
  arguments) and xmm11–xmm15 at `[RBP+48, RBP+128)`. The helpers take the generator's own `ϕ`: `complex.rs`
  maps `Gen(n)` to xmm(n+4), not xmm(n+2), and the wrong map leaves registers unsaved. `save_nonvolatile_regs`
  and, for the fast kernels, `save_nonvolatile_regs_fast` reserve those 80 bytes (`volatile_area_size`); a
  frame without them stores the registers in the caller's frame. The SIMD generators' slots are already
  register-wide. `vzeroupper` (AVX) goes in the epilogues of the AVX generators (`scalar.rs`, `complex.rs`,
  `vector.rs`, `vector_f64x8.rs`), not in the shared `load_nonvolatile_regs(_fast)`, which `sse.rs` (CPUs
  without AVX) also uses. Keep `SPILL_AREA`
  equal on all platforms: `.sjb` files store MIR with absolute stack slots (`__Arg k` is
  `Stack[SPILL_AREA + k]`), and the ARM/RISC-V generators save d8–d15/fs0–fs11 in slots 4–15. A clobbered
  register shows up as corrupted values in the *calling* Rust code, depending on its register allocation, so
  failures move around with unrelated edits; `tests/win64_nonvolatile.rs` calls every kernel kind with
  sentinels in the non-volatile registers and in the caller's frame above the home area.
- **AVX-512 `xmm16`–`xmm31`** need AVX512VL (`Config::use_xmm16_31`), not just AVX512F (`use_simd512`).
- **Non-reproducible machine code**: compressed kernels (`Topology::subs` is a `HashMap`) and kernels calling
  math functions (function addresses, table order) differ from run to run; compare their results or sizes.
- **SIMD vs scalar**: complex SIMD kernels can differ from the scalar kernel in the last bit (different
  formulas); bit-for-bit comparisons use `use_simd=False` or the scalar kernel.
- **Encoding tests** (`rust/{amd,arm,riscv64}/tests.rs`) pin instructions against an external assembler
  (GNU `as`, Keystone, or Capstone-checked encoders). Keep them generated, not hand-edited; record
  intentional deviations as `NOTE:` comments or `#[ignore = "reason"]`.
- **Never run rustfmt on `lib.rs` or a `mod.rs`**: it recurses into every module (several files are not
  rustfmt-clean). Format single leaf files, or check with `rustfmt --edition 2021 --check <file>`.
- **Known open defects**: RISC-V `round`/`trunc`/`floor`/`ceiling` go through `fcvt.l.d` (sign of zero lost,
  wrong for |x| ≥ 2^63, ±inf, NaN); Cephes' `gamma(±0)` is NaN and its `loggamma` is inaccurate near 1 and 2
  and infinite for subnormal arguments; Symbolica evaluators with `ty="bytecode"` return zeros; malformed
  evaluator constants panic in `compiler.rs` instead of returning an error; the two `kernels.rs` coef tests
  are ignored (`direct_arena_identity_output(false)` is not implemented).
- `python/symjit/pyengine` is the obsolete pure-Python backend, and `ty="amd-sse"` (`amd/sse.rs`) is
  obsolete; don't extend either.

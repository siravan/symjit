# Version 2.28.3

* Complex SIMD kernels (AVX, AVX-512, NEON) and the ARM fast-complex kernel (FCMA): a complex product
  passed straight to a function call (opt level >= 1) could overwrite its first factor and give a
  wrong result (#30).
* CSE with conditionals (`IfElse`/`Goto`/`Label`/`Join`): a subexpression shared by a branch and
  the code after it could be read from a temporary set only in the branch, giving wrong values when
  the branch was not taken (#31). With `simd_branch`, nested or consecutive if/else blocks had the
  same defect between exclusive branches: the statements inside branches and loops are now found
  from the jumps and their labels rather than by counting them.
* With `simd_branch`, the stack-slot compactor took every if/else for the start of a loop and
  reused no slot after the first one; loops are now found from the jumps back to their labels.
* With `simd_branch`, the else part of every if/else ran on every evaluation, in the scalar and the
  SIMD kernels, even when no point needed it (a Symbolica program whose else parts held most of its
  work ran 7-20x slower). The end of a then part now jumps over the else part when no lane has the
  condition zero, in both translators.
* The x86-64 and NEON generators assert that FMA, `recip`, `half` and `sign` get no destination that
  shares a machine register with a source (`Left`/`Ret`, `Temp`/`Right`), which would compute a
  wrong value; no program is known to produce one.

# Version 2.28.2

* `ultra` option (`Config::set_ultra`, bit `0x8000_0000`, Python `ultra=True`): subroutine calls pass
  their first arguments packed. It was implied by `opt_level >= 3` and is now separate (off by default).
* The warning about the SSE fallback is written once per session.
* `sign` is supported (`Expr::sign`): 1, -1, 0 (for +-0) or NaN; `z / |z|` for complex.
* `And`/`Or`/`Xor` with more than two operands (sympy flattens nested ones) compile.
* A `Piecewise` whose last condition is not `True` is NaN outside its conditions, as in sympy.
* `CLAUDE.md` describes the repository only; local notes go in `CLAUDE.local.md` (ignored).

# Version 2.28.1

* `.sjb` files version 3 (2.22-2.26) can be read.
* x86-64: the AVX code is chosen only on CPUs that also have FMA3 (#24).
* `Config::default`: an unreadable or malformed `symjit.toml` is reported and ignored (#25).
* `Application::save` writes the same bytes for the same program (#26).
* `Expr::si`/`ci`/`shi`/`chi` emit the function names symjit knows (`Si`, `Ci`, `Shi`, `Chi`).

# Version 2.28.0

* Code cleanup and testing.
* Bug fixes: RISC-V calls with 2048 arguments, `Expr::si`/`ci`/`shi`/`chi`, complex loops with complex
  conditions, SSE complex arithmetic.
* Vectorized calls of JSON ODE models raise an error (they corrupted the stack).

# Version 2.27.4

* Stable math. `Config.set_stable_math` is added.

# Version 2.27.3

* Adding `Config.set_num_args`.

# Version 2.27.2

* backporting bug fixes from the dev branch.

# Version 2.27.1

* Repo re-organization with moving tests from `symjit-crate`.
* Backporting recursive CSE and async mode (co-routines for single-threaded vectorized kernels).
* Various tests and examples (domino folder).

# Version 2.27.0

* The number of arguments of an inner/external function call is configurable (`num_args`
  option).
* Support for Symbolica v3 evaluators.
* New `libm` cargo feature: complex trigonometric and hyperbolic functions use the C library.
* aarch64: `fcmla` is used for complex multiplication whenever the CPU supports FCMA..
* Fixing recursive calls.

# Version 2.26.2

* Raising the function call argument limit to 1024 with the `symbolica` feature.
* Fixes for bugs found by fuzzing.

# Version 2.26.1

* Recursive calls: a `defuns` entry set to `None` calls the function being compiled.
* Querying the library version (`"version"`).

# Version 2.25.0

* Adding inner functions and compression modes to riscv and aarch64.
* Binary release for Linux arch64 (Raspberry Pi) and riscv64.
* Alias for `conj` and `imag`.

# Version 2.24.1

* Fixing compression mode in Windows. 

# Version 2.24.0

* Inner function calls (`add_applet`).
* Adding type hints to `compile_*` functions.

# Version 2.23.0

* New compression mode (`compress = True`).
* Ultramode (`compress = True` and `opt_level = 3`).

# Version 2.22.1

* `IndirectTranslator` cache changed to `Vec`.
* Trimming binary operators is symmetrical. 

# Version 2.22.0

* Modified kernel prologue/epilogue.
* `IndirectTranslator` changed from `Expr` output to `Node` output.
* `Operation` added.
* Arena infrastructure. 

# Version 2.21.0

* New stable release.
* `set_option` signature changed.

# Version 2.20.4

* aarch64 paired load/save improved.

# Version 2.20.3

* Stack limit added.

# Version 2.20.2

* `load/save_registers` updated for 32-reg allocations.

# Version 2.20.1

* Peephone fuse_recip3 bug fixed.
* `fuse_sincos` removed.

# Version 2.20.0

* AVX512 f64x8 added.
* x64 backend updated.

# Version 2.19.2

* Real log/ln bug fixed.
* Deprecation warnings added.
* Risc V branch bug fixed.

# Version 2.19.0

* Composer interface added.
* Documentation improvement.

# Version 2.18.0

* `parallel_mul` option added.
* Improved peephole optimization.

# Version 2.17.3

* `huge` option to use 2 MB pages added.
* `compress` mode added.
* Bug fixes to arm64 code generator.

# Version 2.17.0

* Using SIMD instructions for scalar complex functions.

# Version 2.16.0

* Code cleanup and infra-structure for MIR save/load.

# Version 2.15.3

* Vectorized complex load/save.
* Optimized complex multiplication and division (permissive mode).

# Version 2.15.2

* Direct translation from Symbolica.

# Version 2.15.1

* Fused load/math operations (both constant and memory access) are implemented.
* Debugging complex codegen.

# Version 2.15.0

* Code refactoring to better support multi-threading through `Applet`.
* `Pool` is removed as `opt_level = 2` is the default.

# Version 2.14.*

* Improved inter-operability with Symbolica, especially adding external functions.
* Addition of `Applet`.

# Version 2.13.5

* `symjit-crate` Symbolica dependency is removed.
* `Complex` is re-exported.

# Version 2.14.4

* Parser complex constants parsing is fixed.

# Version 2.13.3

* `get_instructions` parser added.
* Complex logical/boolean instructions changed to real only.

# Version 2.13.2

* `Bridge` encoder bugs fixed.

# Version 2.13.1

* `order=fortran` added to `compile_evaluator`.
* Matrix uses fat pointers now.

# Version 2.12.3

* Mixed complex-real computation to align with Symbolica 1.3.
* Symbolica Python interface improvements. 

# Version 2.12.0

* `Compiler` modified to access the output of Symbolica `export_instructions`.
* `Translator` object implements a two-pass translation of `export_instructions` outputs. 
* Adding `compile_evaluator` to the Python interface.
* Symbolica interface supports SIMD (`Application.evaluate*` functions). 
* `opt_level` default increased to 2.

# Version 2.11.0

* Adding support for complex numbers.
* Adding `dtype` to `compile` functions.
* Removing explicit independent variable.
* Fixing and consolidation stack frames between regular and fast functions.

# Version 2.10.0

* Adding support for NEON instructions on aarch64 (128-bit f64x2).
* Adding hidden observables (named observables prefixed with `__`).
* `compile_json` accepts regular functions in addition to ODE systems.
* `sanitize` switch is deprecated.

# Version 2.9.3

* Adding support for aarch64 code segments > 1 MB (aarch64 adrp instruction).

# Version 2.9.2

* Can support >2^16 variables on aarch64.
* Code-factoring, especially separating `Expr`.
* Symbolica support added to both Python and Rust. 
* Memory leak for normal compiled code fixed (fast functions still intentionally leak memory).

# Version 2.8.0

* Explicit looping constructs `Sum` and `Product` implemented.
* `defuns` keyword added, allowing for calling Python and other Symjit functions.
* Better testing (runtests.py).

# Version 2.7.1

* An `IfElse` level-2 bug fixed.

# Version 2.7.0

* RISC-V machine code is supported (64-bit only).
* `IfElse` explicitely supported in the low-level code genetators.
* Improvement in register usage in the Arm64 backend.
* A bug in Windows fast-func fixed.

# Version 2.6.0

* `flush()` removed and replaced with `used_registers` in the preamble code.
* Alignment fixed before writing constants.
* Level-2 optimization with improved graph coloring algorithm added.
* `opt_level` argument added to compile functions.

# Version 2.5.1

* Sympy constants accepted (is_number).
* RISC-V compilation (bytecode only).

# Version 2.5.0

* New Intemediate-Representation (MIR).
* Peephole optimization on MIR.
* Optional fused-mul-add instructions (fastmath option).
* Adding `apply` function to Python `Func`.
* Adding callable functions to Python `Func`.

# Version 2.4.2

* Large stack frame support in x64 (chkstk functionality).
* Debug mode fixed.

# Version 2.4.1

* Arm64 (aarch64) large stack frame.
* Consolidated tests suite (plots.py).
* New intrinsic operators (Min, Max, Heaviside).
* Switch to spec_math::cephes64 and addition of new numerical functions (erf, gamma, ...).

# Version 2.4 Release Note

* Common-subexpression elimination (keywork `cse`) implementede.
* Reg and Block classes added.
* Examples and tests updated.
* F32 support added to low-level codegen (not exposed to the API yet).

# Version 2.3 Release Note

* Multi-threading support added (keywork `use_threads`).
* Runnable class extensively rewritten to support parallelization.

# Version 2.2 Release Note

* Matrix class added to support parallelization.

# Version 2.1 Release Note

* Precise transcendental functions (expm1, log1p, exp2, and log2) added.
* Sub-expression rewriting rules.
* Debug mode added.
* A subtle bug in arm code generator fixed.

# Version 2.0 Release Note

* The rust backend is extensively updated. Specially, a new register allocator improves register use.
* The new backend uses a tree-based Intemediate Representation instead of the three-address form in version 1.
* The default instruction set on AMD64 systems is AVX instead of SSE2 (SSE2 can still be generated if needed).
* Improved and unified stack frame structure. The stack frame size is significantly reduced compared to version 1.
* Fast functions (exposed rust function pointers) are introduced to reduce function call overhead of small functions.
* Rounding functions `floor` and `ceil` are added.
* Special fast code generation for exponentiation to an integer power (e.g., `x**100`).
* Special fast code generation for modular exponentiation (e.g., `x**100 % 65537`).
* Less overhead in calling standard transcendental functions.
* Fixing a bug in comparison operators (swaping the meaning of strict vs non-strict comparisons).
* Addition of extensive testing examples.

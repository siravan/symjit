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

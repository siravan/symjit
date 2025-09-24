# Version 2.5.1

* Sympy constants accepted (is_number).
* Riscv compilation (bytecode only).

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

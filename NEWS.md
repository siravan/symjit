# Version 2 Release Note

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

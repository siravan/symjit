# Object files

`write_obj` writes a compiled function as a relocatable object file and a C header, so
that it can be linked into C and C++ programs:

```python
from sympy import symbols, sin, exp
from symjit import compile_func

x, y = symbols("x y")
f = compile_func([x, y], [sin(x) * y + exp(-x * x)])
f.write_obj("model")       # writes model.o and model.h
```

```c
#include <stdio.h>
#include "model.h"

int main(void) {
    double mem[MODEL_MEM_SIZE] = {0.5, 2.0};               /* the states, then the outputs */
    model(mem, NULL, 0, NULL);
    printf("%g %g\n", mem[MODEL_COUNT_STATES], model_fast(0.5, 2.0));
    return 0;
}
```

```bash
cc main.c model.o -lm -o main     # or c++; the header has extern "C"
```

The kernels are the machine code symjit runs itself, so they give the same results.
A complete example, which computes pi and checks it, is in
[examples/obj](../examples/obj).

## Enabling it

Object files are behind the `obj` cargo feature, which is off by default (also for the
Python package). Without it, `write_obj` fails with "symjit was built without the `obj`
feature". To enable it:

```bash
cargo build --release --features obj
cp target/release/libsymjit.so python/symjit/_lib.cpython-312-x86_64-linux-gnu.so     # Linux
cp target/release/libsymjit.dylib python/symjit/_lib.cpython-312-darwin.so           # macOS
cargo test --lib --features obj       # includes the linker tests (links C programs with `cc`)
```

or, for `pip install`, list it in `pyproject.toml` (`features = ["obj"]`). On macOS the
package loads any `_lib*` file in `python/symjit` whose name contains `darwin`, so remove
older ones.

## API

`f.write_obj(name, format=None)` is available on the functions returned by
`compile_func`, `compile_ode` (and their Jacobians), `compile_composer` and
`compile_evaluator` (for the last two, `dtype="float64"` or `"complex128"` selects the
kernel). It writes `name.o` and `name.h`; `name` may include a directory, and the kernels
are named after its last component, which must be a C identifier. `format` is `None`
(the host's: ELF on Linux, Mach-O on macOS), `"elf"` or `"macho"`. The architecture is the
one the function is compiled for: the host's, or e.g. AArch64 with `ty="arm"` (so ARM64
objects for an Apple silicon Mac can be written on an x86-64 Linux machine). From Rust:
`Application::write_obj(name)` and `write_obj_for(name, target)`.

## The object and the header

The object holds:

* `int name(double *mem, const symjit_slice *states, size_t idx, const double *params)`,
  the kernel, with two calling modes:
  * direct: `name(mem, NULL, 0, params)` evaluates one point; `mem` holds the states
    (`NAME_COUNT_STATES` values), then the outputs (`NAME_COUNT_OBS`), then the
    derivatives of an ODE (`NAME_COUNT_DIFFS`); `NAME_MEM_SIZE` doubles in all.
  * indirect: `name(NULL, states, i, params)`; `states` points to one `symjit_slice`
    (`{double *ptr; size_t len;}`) per state and then per output; the kernel reads element
    `i` of each state array and writes element `i` of each output array.

  `params` points to the parameters (`NAME_COUNT_PARAMS`), or is NULL if there are
  none. It returns 0. For `compile_composer`/`compile_evaluator` functions the inputs
  are the parameters and the outputs are written to `mem`: `name(outs, NULL, 0, inputs)`.
  Complex values are (re, im) pairs, and the counts are in doubles.
* `double name_fast(double x0, ...)`, for functions with a single output, no parameters
  and at most a few states: the output at one point.

The header defines the sizes, `symjit_slice`, and the prototypes, and says which C math
library functions the kernels call and which instructions they use.

## Limitations

* x86-64 and AArch64 (ARM64), as ELF (Linux) and Mach-O (macOS); the code is
  position-independent and links into executables (PIE or not) and shared libraries.
  Math functions are called with `call`/`bl` relocations; on AArch64 constants are
  loaded PC-relative, which limits a kernel to 1 MB.
* The kernels use the instruction set of the CPU that compiled them: AVX and FMA
  (x86-64-v3) on current x86-64 CPUs (`ty="amd-sse"` gives SSE2 code); ARMv8-A on AArch64
  (ARMv8.3-A with FCMA for complex kernels compiled on CPUs that have it). The header
  states it.
* The only functions the kernels can call are those of the C math library (link with
  `-lm`): `sin`, `cos`, `tan`, their inverses and hyperbolic versions, `exp`, `expm1`,
  `exp2`, `log` (symjit's `ln`), `log10` (symjit's `log`), `log1p`, `log2`, `cbrt`,
  `erf`, `erfc`, `tgamma` (`gamma`), `lgamma` (`loggamma`), `pow` and `atan2`.
  `write_obj` rejects functions that call anything else: `csc`, `sec`, `cot` and their
  hyperbolic versions, `sinc`, complex functions (`sin(z)` with `dtype="complex128"`;
  complex arithmetic and `sqrt` are fine), and user-defined functions (`defuns`).
* Only the scalar and fast kernels are written, not the SIMD kernels.

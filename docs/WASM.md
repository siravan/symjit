# WebAssembly backend

`ty="wasm"` compiles a model into a self-contained WebAssembly module that can be
shipped to and run by any WebAssembly host (a browser, Node, wasmtime, ...):

```python
from sympy import symbols, sin, cos
from symjit import compile_func

x, y = symbols("x y")
f = compile_func([x, y], [sin(x), cos(y)], ty="wasm")
f.dump("model.wasm", "wasm")   # write the module
f(0.5, 1.0)                    # still callable from Python (bytecode interpreter)
```

symjit does not execute the module itself: Python calls of a `ty="wasm"` function
run on the bytecode interpreter, and the module is only generated and exported.
Saving with `f.save(...)` and `load_func` keeps the type, and the loaded function
produces the same module.

## Enabling the backend

The backend is behind the `wasm` cargo feature, which is off by default (also for the
Python package). Without it, `ty="wasm"` fails with "symjit was built without the
`wasm` feature". 

## Status

The backend is being built in phases. **Phase 2** (current) compiles:

* loading and storing states, observables, parameters and stack slots,
* constants and calls to imported functions: math functions, user-defined Python
  functions passed in `defuns` (imported under their names), and complex functions,
* arithmetic (`+ - * /`, powers, `sqrt`, `Abs`, `floor`, `ceiling`, ...),
* comparisons, `And`/`Or`/`Not`, `Min`/`Max` and `Piecewise` (as branch-free
  selects),
* complex arithmetic and functions (`dtype="complex128"`), including complex `sqrt`,
* control flow: `Sum`/`Product` loops and the if/else blocks, gotos and loops of
  Symbolica evaluators (`compile_evaluator`) and `Composer` programs
  (`compile_composer`),
* recursive calls (`Composer.call(None, ...)`, `defuns` entries set to `None`), with
  the native backends' convention; a recursion too deep for the stack region makes
  `run` return 1 (native code has no such check and crashes),
* the `fast` export for models without parameters and with one output.

Not supported yet: SIMD kernels; models whose code exceeds the JavaScript API's
limit of 7,654,321 bytes per function (e.g. the largest Symbolica evaluators; the
compiler reports it); and functions registered from Rust with
`Defuns::add_sliced_func` or `add_applet`. Compiling a model that needs them fails with
`the WebAssembly backend does not support ... yet`. `compress=True` is ignored for
this target.

Calls from Python of a `compile_func` model with `ty="wasm"` run on the bytecode
interpreter. Functions of the Symbolica bridge (`compile_evaluator`,
`compile_composer`) cannot be called from Python with `ty="wasm"` (it raises); export
their modules with `f.dump(path, "wasm")`.

Numerically, the module computes what the bytecode interpreter computes, with two
differences from native code: fused multiply-adds are evaluated as a multiply
followed by an add (core WebAssembly has no FMA instruction), so results can differ
from the native kernels in the last bit; and `round` rounds halves to even (like the
aarch64 and riscv64 backends; the interpreter rounds them away from zero). Imported
functions are as accurate as the host's implementation (e.g. JavaScript's `Math`).

## Module interface

```wat
(import "symjit" "<name>" (func (param f64) (result f64)))      ;; unary functions
(import "symjit" "<name>" (func (param f64 f64) (result f64)))  ;; binary functions
(import "symjit" "cplx_<name>" (func (param f64 f64) (result f64 f64)))          ;; complex
(import "symjit" "cplx_<name>" (func (param f64 f64 f64 f64) (result f64 f64)))  ;; complex
(memory (export "memory") n)
(global (export "__heap_base") i32)
(func (export "run") (param $mem i32) (param $states i32) (param $idx i32)
                     (param $params i32) (result i32))
(func (export "fast") (param $x0 f64) ... (result f64))   ;; some models, see below
```

* **Imports.** Only the functions the model uses are imported, all from module
  `"symjit"` under symjit's function names, e.g. `sin`, `cos`, `exp`, `ln`, `log`,
  `power` (binary), `atan2` (binary), `arcsin`, `arcsinh`, `sec`, `cbrt`, `erf`,
  `gamma`. Complex functions are named `cplx_<name>` (`cplx_sin`, `cplx_exp`,
  `cplx_power`, ...) and take and return complex numbers as (re, im) pairs, using
  multiple return values (a JavaScript import returns `[re, im]`). User-defined
  functions from `defuns` keep their names (`ext(x)` imports `"ext"`).
  `WebAssembly.Module.imports(module)` lists them.
* **Memory.** Addresses below `__heap_base` hold the stack used by `run`; the host
  places its data at or above `__heap_base`. The memory starts with one free 64 KiB
  page above `__heap_base`; a host that needs more calls `memory.grow`.
* **`run`** returns 0 on success and 1 if the stack region is exhausted. All
  pointers are byte offsets into the exported memory; values are `f64`.

`run` has the same two calling modes as the native kernels:

* **Direct:** `run(mem, 0, 0, params)`. `mem` points to the model memory: the
  states (`count_states` values) followed by the observables (`count_obs` values;
  then the diffs for ODE models). The host writes the states, calls `run` and
  reads the observables.
* **Indirect:** `run(0, states, idx, params)`. `states` points to a table of
  `count_states + count_obs` entries of 8 bytes, each an `(i32 ptr, i32 len)` pair
  pointing to an `f64` array (the wasm32 layout of `&mut [f64]`): first the state
  arrays, then the observable arrays. `run` reads element `idx` of every state
  array and writes element `idx` of every observable array, so calling it for
  `idx = 0..n-1` evaluates the model over `n` points.

`params` points to the parameter values (`params=[...]` of `compile_func`), in order.

Modules from the Symbolica bridge (`compile_evaluator`, `compile_composer`) take their
inputs as parameters and write their outputs to `mem`: call `run(outs, 0, 0, inputs)`
with `outs` pointing to room for the outputs (complex values as (re, im) pairs).

**`fast`** is exported for models without parameters, with a single output and at
most 1000 states (the JavaScript API's limit on function parameters):
`fast(x0, x1, ...)` takes the states as arguments and returns the output (NaN if the
stack is exhausted). It allocates the model memory on the module's stack and calls
`run` in direct mode, so it needs no memory management by the host:

```js
const { fast } = new WebAssembly.Instance(module, { symjit: { sin: Math.sin } }).exports;
fast(0.5, 2.0);
```

## Minimal JavaScript host

```js
import { readFileSync } from "node:fs";

const module = new WebAssembly.Module(readFileSync("model.wasm"));
const { run, memory, __heap_base } = new WebAssembly.Instance(module, {
    symjit: { sin: Math.sin, cos: Math.cos },   // one entry per import
}).exports;

const f64 = new Float64Array(memory.buffer);
const mem = __heap_base.value;           // states, then observables
const params = mem + 8 * 16;

f64.set([0.5, 1.0], mem / 8);            // x, y
run(mem, 0, 0, params);
console.log(f64[mem / 8 + 2], f64[mem / 8 + 3]);   // sin(x), cos(y)
```

`examples/wasm/` has runnable examples and a small reusable host
(`host.mjs`: `evaluate`, `evaluateMany`, `evaluateInputs`, `fast`).
`python/tests/wasm_runner.mjs` is a fuller host: it maps symjit's function names to
JavaScript (`Math` lacks `erf`, `gamma` and a few others; a host must supply them),
implements the complex functions, lays out the memory and calls `run` in all modes.
Mind the differences between JavaScript and C math: `Math.pow(1, Infinity)` and
`Math.pow(-1, Infinity)` are NaN where C's `pow` (and symjit's native code) return 1,
so a host should wrap `power` as the runner does.

Hosts must respect the JavaScript API's limits: modules whose `run` function is
larger than 7,654,321 bytes are rejected at compile time by symjit, and `fast` is
only exported for at most 1000 states.

## Inspecting a module

The modules carry a `name` section, so wabt's tools show readable output:

```bash
wasm-validate model.wasm
wasm2wat model.wasm          # text form, with local names (ret, temp, r0, ...)
wasm-objdump -x model.wasm   # sections, imports and exports
```

## Implementation

`rust/wasm/encoder.rs` is the binary encoder (instructions, LEB128, sections);
its encodings are pinned against `wat2wasm` by `rust/wasm/tests.rs`, generated with
`python3 rust/wasm/gen_tests.py > rust/wasm/tests.rs`. `rust/wasm/mod.rs`
(`WasmGenerator`) implements the `Generator` trait: MIR registers become locals
(`Ret`, `Temp`, `r0`...), stack slots live in linear memory below `__heap_base`,
constants become `f64.const` immediates, and the body is buffered until `seal`,
when constants and import indices are resolved and the module is assembled.
Labels and jumps are buffered as markers and structured by `rust/wasm/flow.rs`: the
code is split into basic blocks placed in nested `block`s inside a `loop`; forward
jumps are plain `br`s, backward jumps and subroutine returns go through a `br_table`
dispatcher on a block-number local. `python/tests/test_wasm.py` validates the modules
with `wasm-validate` and runs them in Node against numpy, Symbolica and the fuzz
programs of `test_fuzz.py` and `test_composer_fuzz.py` (skipped if `node` or wabt is
missing).

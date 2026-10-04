# WebAssembly examples

Models compiled with `ty="wasm"` (see [docs/WASM.md](../../../docs/WASM.md)), written
to `out/` and evaluated in Node with a small JavaScript host. Each script checks the
results against numpy (or symjit's native compiler) and prints `ok!`.

| file                 | shows |
|----------------------|-------|
| `basic.py`           | one point at a time, many points over arrays, and the `fast` export |
| `piecewise_loops.py` | `Piecewise`, `Min`/`Max`, and `Sum`/`Product` loops |
| `composer.py`        | a `Composer` program with an if/else block and a counted loop (Newton's method) |
| `complex_numbers.py` | a complex model; complex functions are imported as `cplx_<name>` |
| `defuns.py`          | Python functions in `defuns` become imports the host implements (here in JavaScript) |
| `recursion.py`       | a recursive `Composer` function (Fibonacci) and what happens when the recursion is too deep |

```
cd examples/claude/wasm
python basic.py
```

Node must be on `PATH` to run the modules (without it the scripts only write them).
`out/` holds the generated `.wasm` files and their `.json` layouts and can be deleted.

## The host

`host.mjs` is a small reference host using only the standard WebAssembly JavaScript
API (it has been tested in Node only):

```js
import { load } from "./host.mjs";

const bytes = readFileSync("out/basic.wasm");                       // or: await (await fetch(url)).arrayBuffer()
const layout = JSON.parse(readFileSync("out/basic.json", "utf8"));
const model = await load(bytes, layout);

model.evaluate([0.5, 2.0]);                                         // one point -> outputs
model.evaluateMany([xs, ys]);                                        // Float64Arrays, one per state -> one per output
model.evaluateInputs([a]);                                           // Composer/Symbolica models (inputs -> outputs)
model.fast?.(0.5, 2.0);                                              // single-output models without parameters
```

`load` supplies the imports: symjit's real functions (`MATH`, with C's `pow`
semantics where JavaScript's `Math.pow` differs), the common complex functions
(`COMPLEX`), and anything passed in its third argument (user-defined functions).
The layout (`count_states`, `count_obs`, `count_params`, `mem_size`) is written next to
each module by `wasm_util.export`; complex models count real and imaginary parts
separately.

`run.mjs` is the command-line driver the Python scripts use (`wasm_util.node`).

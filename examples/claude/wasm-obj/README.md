# Object files two ways: `write_obj` vs WebAssembly + wasm2c

`run.py` turns each of six models into two relocatable object files with the same kernels,
links both into one generated C program, checks that they agree, and times them:

| object | how | code generator |
|---|---|---|
| **A** `out/<model>_a.o` | `f.write_obj("out/<model>_a")` (see [docs/OBJ.md](../../../docs/OBJ.md)) | symjit's native backend |
| **B** `out/<model>_b.o` | `ty="wasm"` → `f.dump("out/<model>_b.wasm", "wasm")` → `wasm2c` → `cc -O2 -c` | symjit's WebAssembly backend ([docs/WASM.md](../../../docs/WASM.md)), then wabt's wasm2c and the C compiler |

B's object also defines the module's imports (`w2c_symjit_sin` → `sin`, `w2c_symjit_ln` →
`log`, `w2c_symjit_power` → `pow`, ...: `out/<model>_b_all.c`), so both objects call the same C
math library functions, and B is linked with wasm2c's runtime (`wasm-rt-impl.c`) as A is linked
with libm. The driver (`out/<model>_bench.c`) evaluates N points in indirect mode (one call
per point over arrays: `<model>_a(NULL, slices, i, params)` vs
`w2c_<model>wasm_run(instance, 0, table, i, params)`, with B's arrays in the module's memory)
and with the `fast` entry points (single-output models), and writes the outputs, which
`run.py` compares with symjit's JIT.

```
python run.py                       # all models, 200000 points
python run.py -n 1000000 nbody      # some models (--list)
python run.py --no-fastmath         # A without FMA contraction: A, B and the JIT agree bit for bit
CC=clang python run.py              # another C compiler
CFLAGS=-march=native python run.py  # extra flags for B (and the driver)
```

Needs a symjit library built with both the `wasm` and `obj` features
(`cargo build --release --features wasm,obj`, then copy it over `python/symjit/_lib*.so`),
wabt's `wasm2c` with its runtime sources (Debian/Ubuntu package `wabt`; found next to the
`wasm2c` executable or in `/usr/share/wabt/wasm2c`), and a C compiler. The generated files go
to `out/`, which can be deleted.

## Results

AMD Ryzen 9 9900X, gcc 13 `-O2`, 200000 points, nanoseconds per point (best of 5 passes):

```
model            in out    A: run   B: run    B/A   A: fast  B: fast    B/A     A vs B   build A           build B
------------------------------------------------------------------------------------------------------------------
polynomial        1   1      6.86     5.59  0.81x      6.49     6.58  1.01x  identical    0.1 ms         0+3+29 ms
rational          2   1      2.40     2.12  0.88x      1.87     1.75  0.94x    3.7e-12    0.1 ms         0+2+31 ms
transcendental    2   1     31.58    34.81  1.10x     31.55    34.77  1.10x    7.0e-11    0.1 ms         0+2+37 ms
piecewise         2   1     10.54    13.43  1.27x      9.80    13.03  1.33x  identical    0.1 ms         0+2+34 ms
loop              2   1    226.20   239.53  1.06x    227.78   239.00  1.05x  identical    0.1 ms         0+2+36 ms
nbody            15  15    112.59   112.01  0.99x         -        -      -    5.4e-11    0.2 ms         0+3+69 ms
```

(`build B` = symjit to `.wasm` + `wasm2c` + C compiler.)

* **Arithmetic** (`polynomial`, `rational`, `nbody`): B is as fast as A or up to 20% faster; the
  C compiler schedules the straight-line code at least as well as symjit. With
  `CFLAGS=-march=native`, gcc contracts B's multiply-adds into FMAs and B is twice as fast on the
  degree-20 polynomial: symjit's own kernel has no FMA there although `fastmath` is on (the
  load/constant fusions of the peephole optimizer take the operands of `x * c + d` first, and it
  reloads `x` from memory at every step).
* **Math functions and branches** (`transcendental`, `loop`, `piecewise`): A is 5–30% faster.
  B's selects are compiled from WebAssembly's `select`/bit operations, and every memory access
  of the wasm2c code goes through the module's linear memory.
* **Build time**: `write_obj` takes a fraction of a millisecond; B is dominated by compiling the
  wasm2c output (30–70 ms).
* **Results**: without fastmath (`--no-fastmath`) A, B and the JIT agree bit for bit; with it they
  differ in the last bits where A uses FMAs (core WebAssembly has none). gcc's default
  `-ffp-contract=fast` lets it contract B's code too when the target has FMA (`-march=native`),
  so B then no longer computes exactly what the WebAssembly module computes.

wasm2c's runtime uses guard pages for the memory bounds checks on 64-bit hosts, so B pays for
no explicit checks; it does keep the module's own stack and its address arithmetic.

## The n-loop Symbolica evaluators: `nloop.py`

`nloop.py` does the same for the evaluators of `examples/symbolica/run_nloop.py`: it reads the
instruction streams `examples/symbolica/<n>loop_instructions_2.txt` (no Symbolica needed),
compiles them as complex evaluators with the options of `examples/symbolica/symjit.toml` (as
`run_nloop.py` does), and builds A (`f.write_obj(..., dtype="complex128")`) and B
(`ty="wasm"` → wasm2c → C compiler). The driver evaluates the same random complex points with
both, one call per point (`loop<n>_a(outs, NULL, 0, inputs)` and
`w2c_loop<n>wasm_run(instance, outs, 0, 0, inputs)`), and checks them against the JIT.

```
python nloop.py 1          # 1-loop evaluator
python nloop.py 1 2 3 -m 1000
```

```
loops inputs  JIT (batch)         A         B    B/A    A vs B A vs JIT     A .o    B .o   build A              build B
-----------------------------------------------------------------------------------------------------------------------
    1    223       0.32us    0.90us    1.82us  2.01x   3.9e-16  8.9e-16    0.0MB   0.2MB      0 ms     0.01+0.02+1.32 s
    2    259       3.40us   11.67us   44.42us  3.81x   2.2e-16  1.2e-15    0.2MB   1.9MB      2 ms    0.08+0.15+55.83 s
    3    301     201.03us  201.01us         -      -         - identical    1.7MB       -     17 ms                    -
```

(`inputs` = complex inputs; `JIT (batch)` = `evaluate_complex` on all points in one call from
Python, per point; differences are relative to the largest output.)

* Unlike the small models of `run.py`, B is 2x (1-loop) to 3.8x (2-loop) slower than A. These
  are long straight-line programs of complex arithmetic with thousands of temporaries: A
  computes on packed (re, im) pairs (`fast_complex`), while the WebAssembly module works on
  separate real and imaginary parts, and its spilled temporaries go through the module's
  linear memory.
* Building B is expensive: the 2-loop module is 19 MB of C in a single function, which gcc
  `-O2` compiles in about a minute with about 3 GB of memory (`CFLAGS=-O1` is faster);
  `write_obj` takes 2 ms.
* The 3-loop evaluator has no B: its code (9.9 MB) is above the 7,654,321-byte limit that
  symjit enforces for WebAssembly functions (the limit of JavaScript hosts; wasm2c itself has
  none). A works and matches the JIT exactly.
* The batch JIT is faster than A for 1 and 2 loops because it evaluates several points at once
  with the SIMD kernels; A (and B) contain only the scalar kernel.

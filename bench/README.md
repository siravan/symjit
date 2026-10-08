# Benchmarks

`run.py` times symjit on 16 workloads, compares each with a reference implementation, and
compares the results with a baseline recorded on the same machine. It is run by hand, not by
the test suite, because timings are only meaningful on a quiet machine.

```
python bench/run.py --list            # the workloads
python bench/run.py                   # all workloads (about a minute), compared with the baseline
python bench/run.py --quick           # smaller problems, separate baseline
python bench/run.py eos nbody         # some workloads
python bench/run.py --save            # record this host's baseline
SYMJIT_PYTHON_PATH=/path/to/pkg python bench/run.py   # another build of the package
```

| workload | what | reference |
|---|---|---|
| `eos`, `isoneutral`, `tke` | the pyhpc benchmarks (`examples/claude/pyhpc`) on 2^18–2^20 grid points | numpy |
| `transcendental`, `-1thread` | `sin`, `exp`, `log`, `atan`, `sqrt`, `tanh` on 2^20 points, with and without threads | numpy |
| `horner` | a polynomial of degree 24 in Horner form | `np.polyval` |
| `piecewise` | three branches with `Min`/`Max` | `np.select` |
| `complex` | complex arithmetic, `exp`, `sqrt` | numpy |
| `mandelbrot` | 20 iterations of z² + c, one symjit call per iteration (`examples/mandelbrot.py`) | numpy complex |
| `nbody` | gravitational accelerations, N = 8, 20000 systems (`examples/claude/nbody.py`) | numpy |
| `fast-call` | 200000 calls of the fast kernel from Python (ctypes overhead) | Python `math` |
| `quad` | 500 integrals with `callable_quad` | `scipy.integrate.quad` with a Python function |
| `ode-lorenz` | Lorenz system with `solve_ivp` (DOP853) | `lambdify` right-hand side |
| `cellml-ohara` | right-hand side of the O'Hara-Rudy model (49 states), 20000 calls | bytecode interpreter |
| `compile-large` | compile time of the N = 16 n-body accelerations | – |
| `symbolica-poly` | a sparse polynomial in 30 variables, complex (skipped without symbolica) | Symbolica's evaluator |

Each workload first checks that symjit and the reference agree. A time is the minimum over
7 samples of about 0.2 s each; on a quiet machine repeated runs agree within about 5%.
The table shows compile time, both run times, speedup, throughput and the size of the
generated machine code (scalar + SIMD + fast kernels).

## Baselines

`baselines/<host>.json` (`<host>` = OS, architecture and CPU model, e.g.
`linux-x86-64-amd-ryzen-9-9900x-12-core-processor`; `-quick` for `--quick`) holds the results of
`--save`, with the options and date. A run is a regression, and exits with status 1, if a
workload is more than `--threshold` (20%) slower, its code more than 10% larger, or its compile
time more than 2x longer (for compile times above 5 ms). Re-record the baseline (`--save`) after
an intended change, or on a new machine, e.g. an Apple M4.

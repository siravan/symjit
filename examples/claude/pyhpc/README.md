# pyhpc-benchmarks: numpy vs. symjit

Ports of the three benchmarks of [pyhpc-benchmarks](https://github.com/dionhaefner/pyhpc-benchmarks)
(public domain) that compare the performance of `numpy` and `symjit`. All three come from the
[Veros](https://github.com/team-ocean/veros) ocean model.

| file            | benchmark                | symjit usage |
|-----------------|--------------------------|--------------|
| `eos.py`        | equation of state        | point-wise formula (dHdT of TEOS-10), compiled into one fused kernel |
| `isoneutral.py` | isoneutral mixing        | stencil; the arithmetic of each stencil is a fused kernel, slicing/assignments stay in numpy |
| `tke.py`        | turbulent kinetic energy | superbee advection (about half of the numpy time) is a fused kernel; the LAPACK tridiagonal solve and the bookkeeping stay in numpy |
| `stencil.py`    | helper                   | runs point-wise kernels on shifted operands without copying them |
| `run.py`        | runner                   | checks the results, times numpy and symjit |

```
python run.py                            # all benchmarks
python run.py eos -s 65536 -s 1048576    # one benchmark, chosen sizes
python run.py isoneutral tke --simd512 --no-threads
```

Each module has `generate_inputs(size)`, `run_numpy(*inputs)` (the pyhpc numpy code), and
`setup_symjit(**options)`, which compiles the kernels and returns a function with the same
signature as `run_numpy`. `run.py` compares the outputs of both (`TOLERANCE` in each module)
before timing them.

## How stencils are ported

symjit kernels are point-wise: `compile_func` builds a scalar function of symbols that is
applied element by element to arrays of the same shape. The numpy codes are stencils, which
read shifted slices such as `a[1:-2, 2:-2, 1:]`. Passing such non-contiguous views makes symjit
copy them, and for memory-bound kernels the copies cost more than the arithmetic.
`stencil.Grid` avoids them: in row-major order, a shift is a slice of the *flattened* array (by
1, `nz`, or `ny*nz` elements along the three axes), and a slice of a flat array is contiguous.
The kernel is evaluated on the flat range spanning the output box, and the points of the range
outside the box are discarded.

## Results

AMD Ryzen 9 9900X (12 cores, 24 threads), numpy 2.4 (single-threaded), symjit with default
options (multi-threaded). Mean time of one call, in ms:

| benchmark | size 2^16 | size 2^18 | size 2^20 |
|---|---|---|---|
| equation of state         | 8.91 vs 0.52 (17x)  | 36.2 vs 1.53 (24x)   | 130 vs 5.79 (22x) |
| isoneutral mixing         | 9.30 vs 8.81 (1.1x) | 52.3 vs 35.9 (1.5x)  | 214 vs 142 (1.5x) |
| turbulent kinetic energy  | 6.00 vs 4.51 (1.3x) | 25.7 vs 26.9 (0.96x) | 141 vs 90.7 (1.6x) |

Single-threaded symjit (`--no-threads`) at size 2^20: equation of state 2.0x, turbulent kinetic
energy 1.3x, and isoneutral mixing 0.62x (slower than numpy).

Notes:

* The equation of state is the ideal case: no data movement and a long chain of arithmetic, so
  the fused kernel does one pass over the data instead of hundreds of numpy temporaries.
* Isoneutral mixing and TKE are dominated by array bookkeeping that is identical in both
  versions, so the gain is limited to the fused parts. For small grids the Python overhead of
  the many kernel calls (66 per isoneutral call) dominates and numpy wins. Multi-threading
  matters for isoneutral mixing, whose kernels are small and memory-bound.
* The results agree with numpy to about 1e-6 relative accuracy (isoneutral mixing) or better;
  the kernels reorder the arithmetic (and use `fastmath` by default), which matters where the
  formulas cancel.

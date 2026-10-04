# Object file example: pi in C

`pi.py` (modeled on `examples/pi.py`) compiles three functions that compute pi, writes
each as a relocatable object file with a C header (`f.write_obj(...)`, see
[docs/OBJ.md](../../../docs/OBJ.md)), and builds and runs `test_pi.c`, a C program that
calls them and checks the results against the known value of pi:

| kernel      | computes | calls |
|-------------|----------|-------|
| `pi_series` | Machin's formula, 4 (4 arctan(1/5) - arctan(1/239)), with arctan as its Taylor series | nothing |
| `pi_atan`   | Machin's formula with the C library's `atan` | `atan` |
| `pi_bbp`    | the Bailey-Borwein-Plouffe series up to k = n (a `Sum`, a loop in the kernel) | `pow` |

`test_pi.c` uses the kernels in both calling modes: one point at a time (`pi_series` in
direct mode, and the `_fast` kernels), and over an array (`pi_bbp` in indirect mode, which
evaluates the BBP series for n = 0, 2, ..., 12 and shows it converge).

```
cd examples/claude/obj
python pi.py
```

prints

```
Machin, arctan series (pi_series)            3.1415926535897931  (0 ulp)
Machin, arctan series (pi_series_fast)       3.1415926535897931  (0 ulp)
Machin, atan from libm (pi_atan_fast)        3.1415926535897936  (1 ulp)

BBP series:
  n =  0   3.1333333333333333   (1.85983e+13 ulp)
  ...
  n = 10   3.1415926535897931   (0 ulp)
  n = 12   3.1415926535897931   (0 ulp)
BBP, n = 12 (pi_bbp)                         3.1415926535897931  (0 ulp)

ok!
```

Requirements: a symjit library built with the `obj` cargo feature (off by default:
`cargo build --release --features obj`, then copy `target/release/libsymjit.so` over
`python/symjit/_lib*.so`), x86-64 Linux or macOS, and a C compiler (`cc`, or `$CC`).
The objects and headers go to `out/`, which can be deleted. To build the C program by
hand:

```
cc -Wall -O2 -I out test_pi.c out/pi_series.o out/pi_atan.o out/pi_bbp.o -lm -o out/test_pi
```

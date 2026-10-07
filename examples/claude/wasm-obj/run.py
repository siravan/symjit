"""Two ways from symjit to a C program: object files from `write_obj` vs WebAssembly + wasm2c.

For each model, symjit produces two relocatable object files with the same kernels:

    A   f.write_obj("out/<model>_a")                symjit's native code generator
    B   ty="wasm" -> out/<model>_b.wasm             symjit's WebAssembly backend,
        -> wasm2c -> out/<model>_b.c                translated to C by wasm2c (wabt)
        -> cc -O2 -c -> out/<model>_b.o             and compiled by the C compiler

B's object also holds the module's imports (`w2c_symjit_sin` -> `sin`, `w2c_symjit_ln` ->
`log`, ...), so A and B call the same C math library functions. B is linked with the wasm2c
runtime (wasm-rt-impl.c), as A is linked with libm.

A generated C driver (out/<model>_bench.c) evaluates both objects over the same N points in
indirect mode (one call per point over arrays: `<model>_a(NULL, slices, i, params)` and
`w2c_<model>wasm_run(instance, 0, table, i, params)`), and with the `fast` entry points when
the model has them. It checks that A, B and symjit's JIT agree and reports nanoseconds per
point (the best of several timed passes).

    python run.py                      # all models, 200000 points
    python run.py -n 1000000 nbody     # selected models (see --list)
    python run.py --no-fastmath        # without FMA contraction in A: A and B agree bit for bit
    CC=clang python run.py             # another C compiler (also for the wasm2c output)
    CFLAGS=-march=native python run.py # extra flags for the C compiler (B and the driver)

Needs a symjit library built with both the `wasm` and `obj` cargo features
(cargo build --release --features wasm,obj), wabt's wasm2c with its runtime
(wasm-rt.h, wasm-rt-impl.c; e.g. Debian/Ubuntu's `wabt` package), and a C compiler.
x86-64 or ARM64, Linux or macOS.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import sympy as sp
from sympy import Max, Min, Piecewise, Sum, atan, exp, log, sin, sqrt, symbols, tanh

from symjit import compile_func

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
CC = os.environ.get("CC", "cc")
CFLAGS = ["-O2", "-Wall"] + os.environ.get("CFLAGS", "").split()

# symjit's names of the imported functions -> C math library (as VirtualTable::c_name)
C_NAMES = {
    "sin": "sin", "cos": "cos", "tan": "tan", "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
    "arcsin": "asin", "arccos": "acos", "arctan": "atan", "arcsinh": "asinh", "arccosh": "acosh",
    "arctanh": "atanh", "exp": "exp", "expm1": "expm1", "exp2": "exp2", "ln": "log", "log": "log10",
    "log1p": "log1p", "log2": "log2", "cbrt": "cbrt", "erf": "erf", "erfc": "erfc", "gamma": "tgamma",
    "loggamma": "lgamma", "power": "pow", "atan2": "atan2",
}
BINARY = {"power", "atan2"}


# ------------------------------------------------------------------ the models
def nbody(N, eps2=0.01):
    """accelerations of N bodies (masses are parameters); full of shared subexpressions"""
    pos = [symbols(f"x{i} y{i} z{i}") for i in range(N)]
    m = list(symbols(f"m0:{N}"))
    acc = [[0, 0, 0] for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            d = [pos[j][c] - pos[i][c] for c in range(3)]
            w = (d[0] ** 2 + d[1] ** 2 + d[2] ** 2 + eps2) ** sp.Rational(-3, 2)
            for c in range(3):
                acc[i][c] += m[j] * w * d[c]
                acc[j][c] -= m[i] * w * d[c]
    states = [c for p in pos for c in p]
    return states, [a for row in acc for a in row], m


x, y, z, n, k = symbols("x y z n k")
COEF = np.random.default_rng(2).uniform(-1, 1, 21) / np.arange(1, 22)

MODELS = {
    # name: (states, outputs, params, domains of the states, parameter values)
    "polynomial": ([x], [sp.horner(sum(float(c) * x**i for i, c in enumerate(COEF)))], [], [(-1, 1)], []),
    "rational": ([x, y], [(x * x - y) / (1 + x * x + y * y) + x * y / (2 + x**4)], [], [(-2, 2)] * 2, []),
    "transcendental": ([x, y], [sin(x) * exp(-y * y) + log(1 + x * x) + atan(x * y) + sqrt(1 + y * y) * tanh(x)],
                       [], [(-3, 3)] * 2, []),
    "piecewise": ([x, y], [Piecewise((x * y, x < -1), (sin(y) + x, x < 1), (Max(x, y) - Min(x * y, 1), True))],
                  [], [(-2, 2)] * 2, []),
    "loop": ([x, n], [Sum(x**k / (k + 1), (k, 0, n))], [], [(-0.9, 0.9), (20, 40)], []),
    "nbody": (*nbody(5), [(-1, 1)] * 15, list(np.linspace(0.5, 1.5, 5))),
}


def run(cmd, **kw):
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if p.returncode != 0:
        sys.exit(f"{' '.join(cmd)} failed:\n{p.stdout}{p.stderr}")
    return p.stdout


def wasm2c_runtime():
    """the directory of wasm-rt-impl.c and the include directory of wasm-rt.h"""
    exe = shutil.which("wasm2c")
    if exe is None:
        sys.exit("wasm2c not found (install wabt)")
    prefix = os.path.dirname(os.path.dirname(os.path.realpath(exe)))
    for d in [os.path.join(prefix, "share", "wabt", "wasm2c"), "/usr/share/wabt/wasm2c", "/usr/src/wasm2c",
              "/opt/homebrew/share/wabt/wasm2c", "/usr/local/share/wabt/wasm2c"]:
        if os.path.exists(os.path.join(d, "wasm-rt-impl.c")):
            inc = d if os.path.exists(os.path.join(d, "wasm-rt.h")) else os.path.join(prefix, "include")
            return d, inc
    sys.exit("cannot find wasm2c's runtime (wasm-rt-impl.c)")


# ------------------------------------------------------------------ building A and B
def build_a(name, f):
    t0 = time.perf_counter()
    f.write_obj(os.path.join(OUT, f"{name}_a"))
    ms = 1e3 * (time.perf_counter() - t0)
    with open(os.path.join(OUT, f"{name}_a.h")) as fd:
        has_fast = f"{name}_a_fast(" in fd.read()
    return ms, has_fast


def build_b(name, g, include):
    base = os.path.join(OUT, f"{name}_b")
    t0 = time.perf_counter()
    g.dump(base + ".wasm", "wasm")
    t1 = time.perf_counter()
    # the module name has no underscores: wasm2c escapes "_" as "__" in identifiers
    run(["wasm2c", base + ".wasm", "-o", base + ".c", "-n", f"{name}wasm"])
    t2 = time.perf_counter()
    with open(base + ".h") as fd:
        header = fd.read()
    imports = re.findall(r"/\* import: 'symjit' '(\w+)' \*/", header)
    glue = [f'/* {name}_b.o: the wasm2c translation of {name}_b.wasm and its imports */',
            "#include <math.h>", f'#include "{name}_b.c"', ""]
    for imp in imports:
        if imp not in C_NAMES:
            sys.exit(f"{name}: no C function for the import {imp!r}")
        if imp in BINARY:
            glue.append(f"f64 w2c_symjit_{imp}(struct w2c_symjit *e, f64 a, f64 b) "
                        f"{{ (void)e; return {C_NAMES[imp]}(a, b); }}")
        else:
            glue.append(f"f64 w2c_symjit_{imp}(struct w2c_symjit *e, f64 a) {{ (void)e; return {C_NAMES[imp]}(a); }}")
    with open(base + "_all.c", "w") as fd:
        fd.write("\n".join(glue) + "\n")
    run([CC, *CFLAGS, "-Wno-unused-function", "-I", include, "-c", base + "_all.c", "-o", base + ".o"])
    t3 = time.perf_counter()
    has_fast = f"w2c_{name}wasm_fast(" in header
    return (1e3 * (t1 - t0), 1e3 * (t2 - t1), 1e3 * (t3 - t2)), has_fast, imports


DRIVER = r"""/* generated by run.py: evaluates {name}_a.o and {name}_b.o over the same points */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "{name}_a.h"
#include "{name}_b.h"

#define NS {ns}
#define NO {no}
#define NP {np}

static double now(void) {{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}}

static size_t n;
static double *in[NS + 1], *out_a[NO + 1], *fast_a, *fast_b, params[NP + 1];
static symjit_slice slices[NS + NO];
static w2c_{name}wasm inst;
static u32 table, wparams;

static void pass_a(void) {{
    for (size_t i = 0; i < n; i++) {name}_a(NULL, slices, i, params);
}}

static void pass_b(void) {{
    for (size_t i = 0; i < n; i++)
        if (w2c_{name}wasm_run(&inst, 0, table, (u32)i, wparams) != 0) {{
            fprintf(stderr, "run failed\n");
            exit(1);
        }}
}}

#if {fast}
static void pass_fast_a(void) {{
    for (size_t i = 0; i < n; i++) fast_a[i] = {name}_a_fast({args_a});
}}

static void pass_fast_b(void) {{
    for (size_t i = 0; i < n; i++) fast_b[i] = w2c_{name}wasm_fast(&inst, {args_a});
}}
#endif

/* seconds per pass: the best of `samples` samples of at least `target` seconds */
static double best(void (*pass)(void), int samples, double target) {{
    pass();
    double b = 1e300;
    for (int s = 0; s < samples; s++) {{
        int reps = 0;
        double t0 = now(), t;
        do {{
            pass();
            reps++;
            t = now() - t0;
        }} while (t < target);
        if (t / reps < b) b = t / reps;
    }}
    return b;
}}

static void save(const char *path, double **arrays, int count) {{
    FILE *fd = fopen(path, "wb");
    for (int j = 0; j < count; j++) fwrite(arrays[j], sizeof(double), n, fd);
    fclose(fd);
}}

int main(int argc, char **argv) {{
    if (argc < 3) return 2;
    n = (size_t)atol(argv[1]);
    FILE *fd = fopen(argv[2], "rb");
    for (int j = 0; j < NS; j++) {{
        in[j] = malloc(n * sizeof(double));
        if (fread(in[j], sizeof(double), n, fd) != n) return 3;
    }}
    if (NP > 0 && fread(params, sizeof(double), NP, fd) != NP) return 3;
    fclose(fd);

    /* A: the arrays are passed as slices */
    for (int j = 0; j < NS; j++) slices[j] = (symjit_slice){{in[j], n}};
    for (int j = 0; j < NO; j++) {{
        out_a[j] = calloc(n, sizeof(double));
        slices[NS + j] = (symjit_slice){{out_a[j], n}};
    }}

    /* B: the arrays live in the module's memory, above __heap_base; the table holds
       (u32 ptr, u32 len) pairs */
    wasm_rt_init();
    wasm2c_{name}wasm_instantiate(&inst{imports_arg});
    wasm_rt_memory_t *mem = w2c_{name}wasm_memory(&inst);
    u32 base = *w2c_{name}wasm_0x5F_heap_base(&inst);
    table = base;
    wparams = table + 8 * (NS + NO);
    u32 data = (wparams + 8 * (NP + 1) + 15) & ~15u;
    uint64_t need = (uint64_t)data + 8ull * n * (NS + NO);
    if (need > mem->size) wasm_rt_grow_memory(mem, (need - mem->size + 65535) / 65536);
    for (int j = 0; j < NS + NO; j++) {{
        u32 entry[2] = {{data + (u32)(8 * n * j), (u32)n}};
        memcpy(mem->data + table + 8 * j, entry, 8);
        if (j < NS) memcpy(mem->data + entry[0], in[j], 8 * n);
    }}
    if (NP > 0) memcpy(mem->data + wparams, params, 8 * NP);

    fast_a = calloc(n, sizeof(double));
    fast_b = calloc(n, sizeof(double));
    int samples = 5;
    double target = 0.1;
    printf("indirect_a %.6g\n", 1e9 * best(pass_a, samples, target) / n);
    printf("indirect_b %.6g\n", 1e9 * best(pass_b, samples, target) / n);
#if {fast}
    printf("fast_a %.6g\n", 1e9 * best(pass_fast_a, samples, target) / n);
    printf("fast_b %.6g\n", 1e9 * best(pass_fast_b, samples, target) / n);
#endif

    double *out_b[NO + 1];
    for (int j = 0; j < NO; j++) out_b[j] = (double *)(mem->data + data + 8 * n * (NS + j));
    char path[4096];
    snprintf(path, sizeof path, "%s.a", argv[2]);
    save(path, out_a, NO);
    snprintf(path, sizeof path, "%s.b", argv[2]);
    save(path, out_b, NO);
#if {fast}
    snprintf(path, sizeof path, "%s.fa", argv[2]);
    save(path, &fast_a, 1);
    snprintf(path, sizeof path, "%s.fb", argv[2]);
    save(path, &fast_b, 1);
#endif
    wasm2c_{name}wasm_free(&inst);
    wasm_rt_free();
    return 0;
}}
"""


def compare(got, want):
    """max relative difference, or 0.0 if bitwise equal (NaNs equal)"""
    same = (got == want) | (np.isnan(got) & np.isnan(want))
    if same.all():
        return 0.0
    with np.errstate(all="ignore"):
        rel = np.abs(got - want) / np.maximum(np.abs(want), 1e-300)
    return float(np.nanmax(rel[~same]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", help="models to run (default: all)")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("-n", type=int, default=200_000, help="points (default 200000)")
    parser.add_argument("--no-fastmath", action="store_true", help="compile A without FMA contraction")
    args = parser.parse_args()
    if args.list:
        print("\n".join(MODELS))
        return
    names = args.models or list(MODELS)
    for m in names:
        if m not in MODELS:
            parser.error(f"unknown model {m}; see --list")

    runtime, include = wasm2c_runtime()
    os.makedirs(OUT, exist_ok=True)
    rt = os.path.join(OUT, "wasm-rt-impl.o")
    run([CC, *CFLAGS, "-I", include, "-c", os.path.join(runtime, "wasm-rt-impl.c"), "-o", rt])

    rng = np.random.default_rng(1)
    rows = []
    for name in names:
        states, outputs, params, domains, pvals = MODELS[name]
        opts = dict(params=params or None, fastmath=not args.no_fastmath)
        f = compile_func(states, outputs, **opts)  # A, and the JIT reference
        g = compile_func(states, outputs, ty="wasm", **opts)  # B
        ms_a, fast_a = build_a(name, f)
        ms_b, fast_b, imports = build_b(name, g, include)
        fast = fast_a and fast_b

        cols = [rng.uniform(lo, hi, args.n) for lo, hi in domains]
        if name == "loop":
            cols[1] = np.floor(cols[1])
        inputs = os.path.join(OUT, f"{name}.in")
        with open(inputs, "wb") as fd:
            for c in cols:
                fd.write(c.astype(np.float64).tobytes())
            fd.write(np.asarray(pvals, dtype=np.float64).tobytes())

        src = os.path.join(OUT, f"{name}_bench.c")
        args_a = ", ".join(f"in[{j}][i]" for j in range(len(states)))
        with open(src, "w") as fd:
            fd.write(DRIVER.format(name=name, ns=len(states), no=len(outputs), np=len(params), fast=int(fast),
                                   args_a=args_a, imports_arg=", NULL" if imports else ""))
        exe = os.path.join(OUT, f"{name}_bench")
        run([CC, *CFLAGS, "-I", OUT, "-I", include, src, os.path.join(OUT, f"{name}_a.o"),
             os.path.join(OUT, f"{name}_b.o"), rt, "-lm", "-o", exe])
        times = dict(line.split() for line in run([exe, str(args.n), inputs]).splitlines())

        jit = np.array(f(*cols, *pvals) if pvals else f(*cols))
        a = np.fromfile(inputs + ".a").reshape(len(outputs), args.n)
        b = np.fromfile(inputs + ".b").reshape(len(outputs), args.n)
        diff_a, diff_b, diff_ab = compare(a, jit), compare(b, jit), compare(a, b)
        tol = 1e-9  # fused multiply-adds in A (and the JIT), not in B
        ok = diff_a == 0.0 and max(diff_b, diff_ab) <= tol
        if fast:
            fa, fb = np.fromfile(inputs + ".fa"), np.fromfile(inputs + ".fb")
            ok = ok and compare(fa, a[0]) <= tol and compare(fb, b[0]) == 0.0
        rows.append((name, len(states), len(outputs), imports, ms_a, ms_b, times, diff_ab, ok))

    print(f"\n{CC} {' '.join(CFLAGS)}, {args.n} points, A {'without' if args.no_fastmath else 'with'} fastmath; "
          f"times in ns per point (best of 5)\n")
    head = (f"{'model':15s} {'in':>3s} {'out':>3s}  {'A: run':>8s} {'B: run':>8s} {'B/A':>6s}  {'A: fast':>8s} "
            f"{'B: fast':>8s} {'B/A':>6s}  {'A vs B':>9s}  {'build A':>8s} {'build B':>17s}")
    print(head)
    print("-" * len(head))
    for name, ns, no, imports, ms_a, ms_b, t, diff, ok in rows:
        ia, ib = float(t["indirect_a"]), float(t["indirect_b"])
        if "fast_a" in t:
            fa, fb = float(t["fast_a"]), float(t["fast_b"])
            fast = f"{fa:8.2f} {fb:8.2f} {fb / fa:5.2f}x"
        else:
            fast = f"{'-':>8s} {'-':>8s} {'-':>6s}"
        agree = "identical" if diff == 0 else f"{diff:.1e}"
        steps_b = "+".join(f"{v:.0f}" for v in ms_b) + " ms"
        print(f"{name:15s} {ns:3d} {no:3d}  {ia:8.2f} {ib:8.2f} {ib / ia:5.2f}x  {fast}  {agree:>9s}  "
              f"{ms_a:5.1f} ms {steps_b:>17s}{'' if ok else '  RESULTS DIFFER'}")
    print("\nbuild B = symjit to .wasm + wasm2c + C compiler; A vs B = max relative difference of the outputs")
    if not all(r[-1] for r in rows):
        sys.exit("some results differ from the JIT")


if __name__ == "__main__":
    main()

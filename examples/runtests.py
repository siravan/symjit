import math
import platform
import sys
import time
from statistics import mean, stdev

import numpy as np

# import pandas as pd
from scipy import integrate
from symjit import compile_func
from sympy import (
    Function,
    Piecewise,
    Product,
    Sum,
    cos,
    sin,
    sqrt,
    symbols,
)

L = 1000
use_complex = len(sys.argv) > 1 and sys.argv[1] == "complex"


def arch():
    if platform.machine() in ["x86_64", "AMD64"]:
        return "amd"
    elif platform.machine() in ["arm64", "aarch64"]:
        return "arm"
    elif platform.machine() == "riscv64":
        return "riscv"
    else:
        return None


x, y, z, a, b = symbols("x y z a b")


def func(states, p, **args):
    t0 = time.perf_counter_ns()
    f = compile_func(states, p, **args)
    t1 = time.perf_counter_ns()
    # print(f"{(t1 - t0) * 1e-6:.1f} ms\t", end="")
    return f


###################################################################


def mandelbrot(**args):
    A, B = np.meshgrid(np.arange(-2, 1, 0.002), np.arange(-1.5, 1.5, 0.002))

    try:
        if False and args["dtype"] == "complex128":
            A = A + 0j
            B = B + 0j
    except KeyError:
        pass

    f = func([a, b, x, y], [x**2 - y**2 + a, 2 * x * y + b], **args)
    X = np.zeros_like(A)
    Y = np.zeros_like(A)

    t0 = time.perf_counter_ns()

    for i in range(5):
        X, Y = f(A, B, X, Y)

    t1 = time.perf_counter_ns()

    return X + Y, t1 - t0


def mandelbrot2(**args):
    A, B = np.meshgrid(np.arange(-2, 1, 0.002), np.arange(-1.5, 1.5, 0.002))

    def quad_map(x, y, a, b):
        return (x**2 - y**2 + a, 2 * x * y + b)

    X = 0
    Y = 0

    for i in range(5):
        X, Y = quad_map(X, Y, a, b)

    f = func([a, b], [X, Y], **args)

    t0 = time.perf_counter_ns()
    X, Y = f(A, B)
    t1 = time.perf_counter_ns()

    return X + Y, t1 - t0


def mandelbrot3(**args):
    A, B = np.meshgrid(np.arange(-2, 1, 0.002), np.arange(-1.5, 1.5, 0.002))
    f = func([x, y, a, b], [x**2 - y**2 + a, 2 * x * y + b], **args)

    t0 = time.perf_counter_ns()

    if args["backend"] == "sympy":
        X = np.zeros_like(A)
        Y = np.zeros_like(A)
        for i in range(5):
            X, Y = f(X, Y, A, B)
    else:
        n = A.shape[0] * A.shape[1]
        buf = np.zeros((4, n), dtype="double")
        buf[2, :] = A.ravel()
        buf[3, :] = B.ravel()
        for i in range(5):
            f.execute_vectorized(buf)
        X = np.reshape(buf[0, :], A.shape)
        Y = np.reshape(buf[1, :], A.shape)

    t1 = time.perf_counter_ns()

    return X + Y, t1 - t0


def pi(**args):
    N = 25

    def arctan_series(x):
        s = x
        for i in range(1, N):
            coef = -(1 + 2 * i) if (i & 1 == 1) else 1 + 2 * i
            s += x ** abs(coef) / coef
        return s

    p = 4 * (4 * arctan_series(x) - arctan_series(y))
    f = func([x, y], p, **args)

    t0 = time.perf_counter_ns()
    u = [f(1 / 5, 1 / 239) for _ in range(L)]
    t1 = time.perf_counter_ns()

    return np.sum(u), t1 - t0


def viete_expr(x, n):
    p = 1

    for i in range(n):
        t = x
        for j in range(i):
            t = x + x * sqrt(t)
        p *= sqrt(t)

    return p


def viete(**args):
    p = viete_expr(x, 21)
    f = func([x], [2 / p], **args)

    t0 = time.perf_counter_ns()
    u = [f(1 / 2) for _ in range(L)]
    t1 = time.perf_counter_ns()

    return np.sum(u), t1 - t0


def lemniscate(**args):
    p = 1

    for i in range(21):
        t = x
        for j in range(i):
            t = x + x / sqrt(t)
        p *= sqrt(t)

    f = func([x], [2 / p], **args)

    t0 = time.perf_counter_ns()
    u = [f(1 / 2) for _ in range(L)]
    t1 = time.perf_counter_ns()

    return np.sum(u), t1 - t0


def binom(**args):
    N = 15
    K = 8

    def binom(x, y, n, k):
        if k == 0 or k == n:
            return 1.0
        else:
            return binom(x, y, n - 1, k) * x + binom(x, y, n - 1, k - 1) * y

    f = func([x, y], binom(x, y, N, K), **args)

    t0 = time.perf_counter_ns()
    u = [f(1, 1) for _ in range(L)]
    t1 = time.perf_counter_ns()

    return np.sum(u), t1 - t0


def stress(**args):
    e = x**2 + x

    for _ in range(15):
        e = e**2 + e
        ed = e.diff(x)

    f = func([x], [ed], **args)

    t0 = time.perf_counter_ns()
    u = [f(0.001) for _ in range(L)]
    t1 = time.perf_counter_ns()

    return np.sum(u), t1 - t0


def power(**args):
    N = 150

    p = 0
    x0 = math.exp(math.log(N) / N)

    for i in range(-N, N + 1):
        p += sin(1 + x**i) ** 2

    f = func([x], [p], **args)

    t0 = time.perf_counter_ns()
    u = [f(x0) for _ in range(L)]
    t1 = time.perf_counter_ns()

    return np.sum(u), t1 - t0


def powi_mod(**args):
    def binom(x, y, n, k):
        if k == 0 or k == n:
            return 1.0
        else:
            return binom(x, y, n - 1, k) * x + binom(x, y, n - 1, k - 1) * y

    p = binom(x, y, 7, 4) ** 5 % 65537 + binom(x, y, 8, 5) ** (4**x) % 65537

    f = func([x, y], [p], **args)

    t0 = time.perf_counter_ns()
    u = [f(1, 1) for _ in range(L)]
    t1 = time.perf_counter_ns()

    return np.sum(u), t1 - t0


def fact(**args):
    def factorial(x, n):
        if n == 0:
            return 1
        else:
            return Piecewise([n, x >= n], [1, True]) * factorial(x, n - 1)

    p = factorial(x, 20)
    f = func([x], [p], **args)

    t0 = time.perf_counter_ns()
    u = [f(18) for _ in range(L)]
    t1 = time.perf_counter_ns()

    return np.sum(u), t1 - t0


def sumprod(**args):
    if args["backend"] == "sympy":
        return sum([math.exp(i / L) for i in range(L)]), 0

    fact = compile_func([z], Product(y, (y, 1, z)), **args)
    F = Function("F")
    f = func([x], Sum(x**z / F(z), (z, 0, 50)), **args, defuns={F: fact})

    t0 = time.perf_counter_ns()
    u = [f(i / L) for i in range(L)]
    t1 = time.perf_counter_ns()

    return np.sum(u), t1 - t0


def triple(**args):
    p = 1 / (1 - cos(x) * cos(y) * cos(z))
    f = func([x, y, z], p, **args)

    t0 = time.perf_counter_ns()
    u = integrate.tplquad(
        lambda x, y, z: f(x, y, z), 0, math.pi, 0, math.pi, 0, math.pi
    )[0]
    t1 = time.perf_counter_ns()

    return u, t1 - t0


def triple_fast(**args):
    p = 1 / (1 - cos(x) * cos(y) * cos(z))

    f = func([x, y, z], p, **args)

    if hasattr(f, "fast_func"):
        f = f.fast_func()

    t0 = time.perf_counter_ns()
    u = integrate.tplquad(
        lambda x, y, z: f(x, y, z), 0, math.pi, 0, math.pi, 0, math.pi
    )[0]
    t1 = time.perf_counter_ns()

    return u, t1 - t0


def triple_callable(**args):
    p = 1 / (1 - cos(x) * cos(y) * cos(z))

    f = func([x, y, z], p, **args)

    if args["backend"] == "sympy":
        h = lambda x, y, z: f(x, y, z)
    else:
        h = f.callable_quad()

    t0 = time.perf_counter_ns()
    u = integrate.tplquad(h, 0, math.pi, 0, math.pi, 0, math.pi)[0]
    t1 = time.perf_counter_ns()
    return u, t1 - t0


#############################################################################

def Ω(b):
    if b:
        return "T"
    else:
        return "F"


def abbr_ty(ty):
    return ty[0]


def abbr_dtype(dtype):
    if dtype == "complex128":
        return "C"
    else:
        return "R"


def cases():
    cases = []

    dtypes = ["float64"]
    if use_complex:
        dtypes.append("complex128")

    if arch() == "amd":
        for dtype in dtypes:
            # for ty in ["native", "amd-sse", "bytecode", "debug"]:
            for ty in ["native", "bytecode", "debug"]:
                for use_simd in ["f64", "f64x4", "f64x8"]:
                    for use_threads in [False, True]:
                        for cse in [False, True]:
                            for fastmath in [False, True]:
                                for fast_complex in [False, True]:
                                    for opt_level in [0, 1, 2, 3]:
                                        args = {
                                            "backend": "rust",
                                            "ty": ty,
                                            "use_simd": use_simd != "f64",
                                            "enable_simd512": use_simd == "f64x8",
                                            "use_threads": use_threads,
                                            "cse": cse,
                                            "fastmath": fastmath,
                                            "fast_complex": fast_complex,
                                            "opt_level": opt_level,
                                            "dtype": dtype,
                                        }
                                        s = f"d={abbr_dtype(dtype)},y={abbr_ty(ty)}:s={use_simd}:t={Ω(use_threads)}:c={Ω(cse)}:f={Ω(fastmath)}:x={Ω(fast_complex)},O={opt_level}"
                                        cases.append((s, args))
    else:
        for dtype in dtypes:
            for ty in ["native", "bytecode", "debug"]:
                for use_simd in [False, True]:
                    for use_threads in [False, True]:
                        for cse in [False, True]:
                            for fastmath in [False, True]:
                                for fast_complex in [False, True]:
                                    for opt_level in [0, 1, 2, 3]:
                                        args = {
                                            "backend": "rust",
                                            "ty": ty,
                                            "use_simd": use_simd,
                                            "use_threads": use_threads,
                                            "cse": cse,
                                            "fastmath": fastmath,
                                            "fast_complex": fast_complex,
                                            "opt_level": opt_level,
                                            "dtype": dtype,
                                        }
                                        s = f"d={abbr_dtype(dtype)},y={abbr_ty(ty)}:s={Ω(use_simd)}:t={Ω(use_threads)}:c={Ω(cse)}:f={Ω(fastmath)}:x={Ω(fast_complex)},O={opt_level}"
                                        cases.append((s, args))
    return cases


def test_model(f, label, pyback=True, bytecode=False, may_complex=True):
    print(f"testing {label}")
    # print("\td: dtype\t\t(R=float64, C=complex128)")
    # print("\ty: ty\t\t(n: native, a: amd-sse, b: bytecode, d: debug)")
    # print("\ts: simd\t\t(True/False) or (none/avx256/avx512)")
    # print("\tt: threads\t(True/False)")
    # print("\tc: cse\t\t(True/False)")
    # print("\tf: fastmath\t(True/False)")
    # print("\tx: fast_complex\t(True/False)")
    # print("\tO: opt_level\t(0/1/2/3)")

    print("\tlambdify.......\t", end="")
    X0, dt0 = f(backend="sympy")
    print(f"\tdone in {1e-6 * dt0:.3f} ms")

    table = []

    for abbr, args in cases():
        if (args["ty"] not in ["bytecode", "debug"] or bytecode) and (
            args["dtype"] == "float64" or may_complex
        ):
            # print(f"{abbr}\t", end="")
            X, dt = f(**args)
            try:
                np.testing.assert_array_almost_equal(X0, X.real)
                # print(f"\tpass in {1e-6 * dt:.3f} ms")
            except AssertionError:
                # print(f"\t\033[31mfail\033[0m in {1e-6 * dt:.3f} ms")
                print(f"{abbr} \033[31mfails\033[0m in {1e-6 * dt:.3f} ms")

            table.append((args, dt * 1e-6))

            if args["opt_level"] == 3:
                pass
                # print()

    # print(f"\t\033[92mspeed-up ratio {dt0 / dt:.1f}\033[0m")

    print_stats(table)

    if pyback and arch() != "riscv":
        print("\tpython.........", end="")
        X, dt = f(backend="python")
        np.testing.assert_array_almost_equal(X0, X)
        print(f"\tpass in {1e-6 * dt:.3f} ms")


def tobulate(table, col, val, dtype):
    x = [dt for (args, dt) in table if args[col] == val and args["dtype"] == dtype]
    if len(x) == 0:
        return math.nan
    else:
        return f"{mean(x):.2f}+/-{stdev(x):.2f}"

def is_complex(table):
    return any(args["dtype"] == "complex128" for (args, _) in table)

def print_stats(table):
    print("dtype = 'float64':")
    print(f"\tsimd:         {tobulate(table, "use_simd", False, "float64")} vs {tobulate(table, "use_simd", True, "float64")}")
    print(f"\tthreads:      {tobulate(table, "use_threads", False, "float64")} vs {tobulate(table, "use_threads", True, "float64")}")
    print(f"\tcse:          {tobulate(table, "cse", False, "float64")} vs {tobulate(table, "cse", True, "float64")}")
    print(f"\tfastmath:     {tobulate(table, "fastmath", False, "float64")} vs {tobulate(table, "fastmath", True, "float64")}")
    print(f"\tfast_complex: {tobulate(table, "fast_complex", False, "float64")} vs {tobulate(table, "fast_complex", True, "float64")}")
    print(f"""\topt_level:
        \t0: {tobulate(table, "opt_level", 0, "float64")}
        \t1: {tobulate(table, "opt_level", 1, "float64")}
        \t2: {tobulate(table, "opt_level", 2, "float64")}
        \t3: {tobulate(table, "opt_level", 3, "float64")}""")

    if is_complex(table):
        print()
        print("dtype = 'complex128':")
        print(f"\tsimd:         {tobulate(table, "use_simd", False, "complex128")} vs {tobulate(table, "use_simd", True, "complex128")}")
        print(f"\tthreads:      {tobulate(table, "use_threads", False, "complex128")} vs {tobulate(table, "use_threads", True, "complex128")}")
        print(f"\tcse:          {tobulate(table, "cse", False, "complex128")} vs {tobulate(table, "cse", True, "complex128")}")
        print(f"\tfastmath:     {tobulate(table, "fastmath", False, "complex128")} vs {tobulate(table, "fastmath", True, "complex128")}")
        print(f"\tfast_complex: {tobulate(table, "fast_complex", False, "complex128")} vs {tobulate(table, "fast_complex", True, "complex128")}")
        print(f"""\topt_level:
            \t0: {tobulate(table, "opt_level", 0, "complex128")}
            \t1: {tobulate(table, "opt_level", 1, "complex128")}
            \t2: {tobulate(table, "opt_level", 2, "complex128")}
            \t3: {tobulate(table, "opt_level", 3, "complex128")}""")


################################################################

test_model(mandelbrot, "mandelbrot")
test_model(mandelbrot2, "mandelbrot2")
test_model(mandelbrot3, "mandelbrot3", may_complex=False)
test_model(pi, "pi")
test_model(viete, "pi-viete")
test_model(lemniscate, "lemniscate")
test_model(binom, "binom")
test_model(binom, "stress")
test_model(power, "power")
test_model(powi_mod, "powi_mod", pyback=False)
test_model(fact, "fact")

if not use_complex:
    test_model(sumprod, "sumprod", pyback=False)
    test_model(triple_callable, "triple_callable", pyback=False, bytecode=False)

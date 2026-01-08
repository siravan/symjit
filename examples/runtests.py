import math
import platform
import sys
import time

import numpy as np
import pandas as pd
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
    print(f"{(t1 - t0) * 1e-6:.1f} ms\t", end="")
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
            X, Y = f(X, Y, A, BytesPath)
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
            for ty in ["native", "amd-sse", "bytecode", "debug"]:
                for use_simd in [False, True]:
                    for use_threads in [False, True]:
                        for cse in [False, True]:
                            for fastmath in [False, True]:
                                for opt_level in [0, 1, 2, 3]:
                                    args = {
                                        "backend": "rust",
                                        "ty": ty,
                                        "use_simd": use_simd,
                                        "use_threads": use_threads,
                                        "cse": cse,
                                        "fastmath": fastmath,
                                        "opt_level": opt_level,
                                        "sanitize": False,
                                        "dtype": dtype,
                                    }
                                    s = f"d={abbr_dtype(dtype)},y={abbr_ty(ty)}:s={Ω(use_simd)}:t={Ω(use_threads)}:c={Ω(cse)}:f={Ω(fastmath)},O={opt_level}"
                                    cases.append((s, args))
    else:
        for dtype in dtypes:
            for ty in ["native", "bytecode", "debug"]:
                for use_threads in [False, True]:
                    for cse in [False, True]:
                        for fastmath in [False, True]:
                            for opt_level in [0, 1, 2, 3]:
                                args = {
                                    "backend": "rust",
                                    "ty": ty,
                                    "use_simd": False,
                                    "use_threads": use_threads,
                                    "cse": cse,
                                    "fastmath": fastmath,
                                    "opt_level": opt_level,
                                    "sanitize": False,
                                }
                                s = f"d={abbr_dtype(dtype)},y={abbr_ty(ty)}:s=F:t={Ω(use_threads)}:c={Ω(cse)}:f={Ω(fastmath)},O={opt_level}"
                                cases.append((s, args))
    return cases


def test_model(f, label, log, pyback=True, bytecode=False):
    print(f"testing {label}")
    print("\td: dtype\t\t(R=float64, C=complex128)")
    print("\ty: ty\t\t(n: native, a: amd-sse, b: bytecode, d: debug)")
    print("\ts: simd\t\t(True/False)")
    print("\tt: threads\t(True/False)")
    print("\tc: cse\t\t(True/False)")
    print("\tf: fastmath\t(True/False)")
    print("\tO: opt_level\t(0/1/2/3)")

    print("\tlambdify.......\t", end="")
    X0, dt0 = f(backend="sympy")
    print(f"\tdone in {1e-6 * dt0:.3f} ms")

    log.append(
        {
            "backend": "sympy",
            "ty": "sympy",
            "use_simd": False,
            "use_threads": False,
            "cse": False,
            "fastmath": False,
            "opt_level": 0,
            "sanitize": False,
            "dt": dt0 * 1e-6,
        }
    )

    for abbr, args in cases():
        if args["ty"] not in ["bytecode", "debug"] or bytecode:
            print(f"{abbr}\t", end="")
            X, dt = f(**args)
            try:
                np.testing.assert_array_almost_equal(X0, X.real)
                print(f"\tpass in {1e-6 * dt:.3f} ms")
            except AssertionError:
                print(f"\t\033[31mfail\033[0m in {1e-6 * dt:.3f} ms")

        a = args.copy()
        a["dt"] = dt * 1e-6
        log.append(a)

    print(f"\t\033[92mspeed-up ratio {dt0 / dt:.1f}\033[0m")

    if pyback and arch() != "riscv":
        print("\tpython.........", end="")
        X, dt = f(backend="python")
        np.testing.assert_array_almost_equal(X0, X)
        print(f"\tpass in {1e-6 * dt:.3f} ms")

        log.append(
            {
                "backend": "python",
                "ty": "",
                "use_simd": False,
                "use_threads": False,
                "cse": False,
                "fastmath": False,
                "opt_level": 0,
                "sanitize": False,
                "dt": dt * 1e-6,
            }
        )


################################################################


log = []
test_model(mandelbrot, "mandelbrot", log)
test_model(mandelbrot2, "mandelbrot2", log)
test_model(mandelbrot2, "mandelbrot3", log)
test_model(pi, "pi", log)
test_model(viete, "pi-viete", log)
test_model(lemniscate, "lemniscate", log)
test_model(binom, "binom", log)
test_model(binom, "stress", log)
test_model(power, "power", log)
test_model(powi_mod, "powi_mod", log, pyback=False)
test_model(fact, "fact", log)

if not use_complex:
    test_model(sumprod, "sumprod", log, pyback=False)
    test_model(triple_callable, "triple_callable", log, pyback=False, bytecode=False)

df = pd.DataFrame(log)
df.to_csv("runtests.csv")
print("timing information saved as `runtests.csv`")

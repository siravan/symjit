"""The benchmark workloads.

Each workload is a function `setup(quick, options)` that compiles its symjit functions and
returns a `Case`: the symjit callable, a reference callable (numpy, scipy with a Python
function, the bytecode interpreter, or Symbolica's own evaluator) that computes the same
thing, a check that the two agree, the compiled functions (for code sizes), and the unit
of work (for throughput: elements, calls, integrals, ...).
"""

import contextlib
import importlib.util
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import sympy as sp
from sympy import Max, Min, Piecewise, atan, cos, exp, log, sin, sqrt, symbols, tanh

import symjit
from symjit import compile_func, compile_json, compile_ode

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PYHPC = os.path.join(ROOT, "examples", "claude", "pyhpc")
CELLML = os.path.join(ROOT, "examples", "cellml")


@dataclass
class Case:
    run: Callable  # the symjit version
    reference: Callable | None  # the same computation without symjit
    ref_name: str  # what `reference` is
    check: Callable | None  # check(symjit_result, reference_result) raises if they disagree
    funcs: list = field(default_factory=list)  # compiled Func objects (code sizes)
    compile_ms: float = 0.0
    work: float = 1.0  # units of work per call of `run`
    unit: str = "call"


def compile_timed(*args, **kwargs):
    t0 = time.perf_counter()
    f = compile_func(*args, **kwargs)
    return f, 1e3 * (time.perf_counter() - t0)


def allclose(rtol, atol=0.0):
    def check(got, want):
        for g, w in zip(_tuple(got), _tuple(want)):
            np.testing.assert_allclose(g, w, rtol=rtol, atol=atol)

    return check


def _tuple(v):
    return tuple(v) if isinstance(v, (tuple, list)) else (v,)


@contextlib.contextmanager
def recording(module):
    """Records the Func objects (and the compile time) of a module that calls compile_func."""
    funcs, total = [], [0.0]
    original = module.compile_func

    def wrapper(*args, **kwargs):
        f, ms = compile_timed(*args, **kwargs)
        funcs.append(f)
        total[0] += ms
        return f

    module.compile_func = wrapper
    try:
        yield funcs, total
    finally:
        module.compile_func = original


# ------------------------------------------------------------------ pyhpc (vectorized, large arrays)
def _pyhpc(name, size, quick_size):
    def setup(quick, options):
        sys.path.insert(0, PYHPC)
        module = __import__(name)
        n = quick_size if quick else size
        with recording(module) as (funcs, ms):
            run = module.setup_symjit(**options)
        inputs = module.generate_inputs(n)
        if name == "tke":
            # the kernels update some inputs in place: every call gets fresh copies (both sides
            # pay for the copies)
            fresh = lambda: [np.copy(v) for v in inputs]
            run_sj, run_np = lambda: run(*fresh()), lambda: module.run_numpy(*fresh())
        else:
            run_sj, run_np = lambda: run(*inputs), lambda: module.run_numpy(*inputs)
        return Case(
            run=run_sj,
            reference=run_np,
            ref_name="numpy",
            check=allclose(**module.TOLERANCE),
            funcs=funcs,
            compile_ms=ms[0],
            work=inputs[0].size,
            unit="point",
        )

    return setup


# ------------------------------------------------------------------ element-wise kernels
x, y, z, a, b, t = symbols("x y z a b t")


def transcendental(quick, options, threads=True):
    n = 2**16 if quick else 2**20
    rng = np.random.default_rng(1)
    X, Y = rng.uniform(-3, 3, n), rng.uniform(-3, 3, n)
    e = sin(x) * exp(-y * y) + log(1 + x * x) + atan(x * y) + sqrt(1 + y * y) * tanh(x)
    f, ms = compile_timed([x, y], [e], **(options | dict(use_threads=threads)))
    return Case(
        run=lambda: f(X, Y)[0],
        reference=lambda: np.sin(X) * np.exp(-Y * Y) + np.log(1 + X * X) + np.arctan(X * Y) + np.sqrt(1 + Y * Y) * np.tanh(X),
        ref_name="numpy",
        check=allclose(1e-13, 1e-15),
        funcs=[f],
        compile_ms=ms,
        work=n,
        unit="point",
    )


def transcendental_1thread(quick, options):
    return transcendental(quick, options, threads=False)


def horner(quick, options):
    n = 2**16 if quick else 2**20
    # degree 24: expressions nested deeper than about 64 levels fail to compile (the JSON
    # parser's recursion limit; see test_api.Limits), and Horner form nests 2 per degree
    coef = np.random.default_rng(2).uniform(-1, 1, 25) / np.arange(1, 26)
    poly = sum(float(c) * x**i for i, c in enumerate(coef[::-1]))
    f, ms = compile_timed([x], [sp.horner(poly)], **options)
    X = np.random.default_rng(3).uniform(-1, 1, n)
    return Case(
        run=lambda: f(X)[0],
        reference=lambda: np.polyval(coef, X),
        ref_name="numpy",
        check=allclose(1e-12, 1e-14),
        funcs=[f],
        compile_ms=ms,
        work=n,
        unit="point",
    )


def piecewise(quick, options):
    n = 2**16 if quick else 2**20
    rng = np.random.default_rng(4)
    X, Y = rng.uniform(-2, 2, n), rng.uniform(-2, 2, n)
    e = Piecewise((x * y, x < -1), (sin(y) + x, x < 1), (Max(x, y) - Min(x * y, 1), True))
    f, ms = compile_timed([x, y], [e], **options)

    def ref():
        return np.select([X < -1, X < 1], [X * Y, np.sin(Y) + X], np.maximum(X, Y) - np.minimum(X * Y, 1))

    return Case(lambda: f(X, Y)[0], ref, "numpy", allclose(1e-15), [f], ms, n, "point")


def complex_arith(quick, options):
    n = 2**15 if quick else 2**18
    rng = np.random.default_rng(5)
    X = rng.normal(size=n) + 1j * rng.normal(size=n)
    Y = rng.normal(size=n) + 1j * rng.normal(size=n)
    e = x * y + exp(x) / (y * y + 1) + x**3 - sqrt(y)
    f, ms = compile_timed([x, y], [e], dtype="complex128", **options)
    return Case(
        run=lambda: f(X, Y)[0],
        reference=lambda: X * Y + np.exp(X) / (Y * Y + 1) + X**3 - np.sqrt(Y),
        ref_name="numpy",
        check=allclose(1e-12),
        funcs=[f],
        compile_ms=ms,
        work=n,
        unit="point",
    )


def mandelbrot(quick, options):
    # 20 iterations of z -> z^2 + c on a grid, as in examples/mandelbrot.py
    h = 0.006 if quick else 0.002
    A, B = np.meshgrid(np.arange(-2, 1, h), np.arange(-1.5, 1.5, h))
    f, ms = compile_timed([a, b, x, y], [x**2 - y**2 + a, 2 * x * y + b], **options)
    C = A + 1j * B

    def run():
        X, Y = np.zeros_like(A), np.zeros_like(A)
        for _ in range(20):
            X, Y = f(A, B, X, Y)
        return (np.abs(X) < 2) & (np.abs(Y) < 2)

    def ref():
        Z = np.zeros_like(C)
        with np.errstate(all="ignore"):
            for _ in range(20):
                Z = Z * Z + C
        return (np.abs(Z.real) < 2) & (np.abs(Z.imag) < 2)

    def check(got, want):
        # orbits that escape differ in the last bits; the sets must agree almost everywhere
        assert np.mean(got != want) < 1e-3, np.mean(got != want)

    return Case(run, ref, "numpy", check, [f], ms, 20 * A.size, "point-iteration")


def nbody(quick, options):
    sys.path.insert(0, os.path.join(ROOT, "examples", "claude"))
    import nbody as nb

    N, M, eps2 = 8, (2000 if quick else 20000), 0.01
    timing = {}
    f = nb.compile_accelerations(N, eps2, timing=timing, **options)
    xs, m = nb.random_systems(M, N, np.random.default_rng(0))
    cols = list(xs.reshape(M, -1).T.copy())

    def run():
        return np.stack(f(*cols, *m)).T.reshape(M, N, 3)

    return Case(run, lambda: nb.acceleration_numpy(xs, m, eps2), "numpy", allclose(1e-11, 1e-12), [f],
                timing["compile"], M * N * (N - 1) / 2, "pair")


# ------------------------------------------------------------------ scalar calls
def fast_call(quick, options):
    # call overhead of the register-argument kernel from Python
    f, ms = compile_timed([x, y], x * y + sin(x), **options)
    g = f.fast_func()
    py = lambda u, v: u * v + math.sin(u)
    calls = 20_000 if quick else 200_000
    pts = [(0.001 * i, 1.5) for i in range(calls)]

    def run():
        return sum(g(u, v) for u, v in pts)

    def ref():
        return sum(py(u, v) for u, v in pts)

    return Case(run, ref, "Python math", allclose(1e-12), [f], ms, calls, "call")


def quad(quick, options):
    import scipy.integrate

    e = sp.atan(sqrt(2 + x**2)) / ((1 + x**2) * sqrt(2 + x**2))  # Ahmed's integral
    f, ms = compile_timed([x], e, **options)
    lc = f.callable_quad()
    py = lambda u: math.atan(math.sqrt(2 + u * u)) / ((1 + u * u) * math.sqrt(2 + u * u))
    reps = 50 if quick else 500

    def run():
        return [scipy.integrate.quad(lc, 0, 1 + 1e-3 * i)[0] for i in range(reps)]

    def ref():
        return [scipy.integrate.quad(py, 0, 1 + 1e-3 * i)[0] for i in range(reps)]

    return Case(run, ref, "scipy + Python", allclose(1e-14), [f], ms, reps, "integral")


def ode_lorenz(quick, options):
    import scipy.integrate

    rhs = [10 * (y - x), x * (28 - z) - y, x * y - sp.Rational(8, 3) * z]
    t0 = time.perf_counter()
    f = compile_ode(t, [x, y, z], rhs, **options)
    ms = 1e3 * (time.perf_counter() - t0)
    g = sp.lambdify([x, y, z], rhs)
    T = 5.0 if quick else 50.0
    opts = dict(method="DOP853", rtol=1e-10, atol=1e-10)

    def run():
        s = scipy.integrate.solve_ivp(f, (0, T), [1.0, 1.0, 1.0], **opts)
        return s.y[:, -1]

    def ref():
        s = scipy.integrate.solve_ivp(lambda tt, u: g(*u), (0, T), [1.0, 1.0, 1.0], **opts)
        return s.y[:, -1]

    def check(got, want):
        # chaotic: only a short horizon is reproducible, so compare the solver's first steps
        if T <= 5.0:
            np.testing.assert_allclose(got, want, rtol=1e-6)

    return Case(run, ref, "lambdify", check, [], ms, 1, "solve")


def cellml(quick, options):
    # right-hand side of the O'Hara-Rudy ventricular model (49 states), called from Python
    with open(os.path.join(CELLML, "ohara.json")) as fd:
        model = fd.read()
    t0 = time.perf_counter()
    f = compile_json(model, **options)
    ms = 1e3 * (time.perf_counter() - t0)
    g = compile_json(model, ty="bytecode", **options)
    u0, p = np.array(f.get_u0()), f.get_p()
    calls = 2000 if quick else 20000

    def loop(h):
        def run():
            for i in range(calls):
                r = h(0.001 * i, u0, *p)
            return r

        return run

    return Case(loop(f), loop(g), "bytecode", allclose(1e-10, 1e-15), [f.compiler], ms, calls, "call")


def compile_large(quick, options):
    # compilation time of a large expression: nbody accelerations with N = 16 (or 10)
    sys.path.insert(0, os.path.join(ROOT, "examples", "claude"))
    import nbody as nb

    N = 10 if quick else 16
    pos, _, mass = nb.symbols(N)
    flat = [c for body in pos for c in body]
    exprs = nb.accelerations(pos, mass, 0.01)
    holder = []

    def run():
        holder[:] = [compile_func(flat, exprs, params=mass, **options)]
        return 0

    run()
    return Case(run, None, "", None, holder, 0.0, 1, "compile")


def symbolica_poly(quick, options):
    # a random sparse polynomial in 30 variables (examples/symbolica/benchmarks.py), complex inputs
    import random

    from symbolica import E

    P, terms = 30, (300 if quick else 3000)
    random.seed(1349)
    vs = [E(f"x_{i}") for i in range(P)]
    expr = math.prod(vs)
    for _ in range(terms):
        random.shuffle(vs)
        expr += random.random() * math.prod(vs[:8])
    vs = [E(f"x_{i}") for i in range(P)]
    ev = expr.evaluator(vs, jit_compile=False, cpe_iterations=0, iterations=0)
    t0 = time.perf_counter()
    f = symjit.compile_evaluator(ev, dtype="complex128", **options)
    f.evaluate_complex(np.zeros((1, P), dtype=complex))  # compiles lazily
    ms = 1e3 * (time.perf_counter() - t0)
    rng = np.random.default_rng(7)
    n = 2000 if quick else 10000
    X = rng.random((n, P)) + 1j * rng.random((n, P)) - (0.5 + 0.5j)
    return Case(lambda: f.evaluate_complex(X), lambda: ev.evaluate_complex(X), "Symbolica",
                allclose(1e-11, 1e-14), [f.complex_compiler], ms, n, "point")


HAVE_SYMBOLICA = importlib.util.find_spec("symbolica") is not None

WORKLOADS = {
    "eos": _pyhpc("eos", 2**20, 2**16),
    "isoneutral": _pyhpc("isoneutral", 2**18, 2**14),
    "tke": _pyhpc("tke", 2**18, 2**14),
    "transcendental": transcendental,
    "transcendental-1thread": transcendental_1thread,
    "horner": horner,
    "piecewise": piecewise,
    "complex": complex_arith,
    "mandelbrot": mandelbrot,
    "nbody": nbody,
    "fast-call": fast_call,
    "quad": quad,
    "ode-lorenz": ode_lorenz,
    "cellml-ohara": cellml,
    "compile-large": compile_large,
}
if HAVE_SYMBOLICA:
    WORKLOADS["symbolica-poly"] = symbolica_poly

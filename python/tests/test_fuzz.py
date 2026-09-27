"""Differential fuzz tests for symjit.

Random expressions are compiled with many option combinations and compared, in scalar
and vectorized calls, against a numpy reference produced by sympy's `lambdify`.

Families:
    real       random real expressions (arithmetic, transcendental functions, Min/Max/Abs)
    shared     several outputs that share function calls and subexpressions
    piecewise  conditionals with two and three branches
    complex    complex arithmetic and functions
    lengths    vectorized calls with awkward array lengths (SIMD tails, threads)

Run from the repository root with

    python -m unittest discover -s python/tests -v

Environment variables:
    SYMJIT_FUZZ_SEEDS   number of random cases per family (default 100)
    SYMJIT_FUZZ_START   first seed (default 0)
    SYMJIT_FUZZ_VERBOSE print every case before compiling it (to locate a crash)

A failing case reports its seed and expressions; rerun it with
SYMJIT_FUZZ_START=<seed> SYMJIT_FUZZ_SEEDS=1. Note that a bug that aborts the process (a panic
in the compiler) kills the whole run; use SYMJIT_FUZZ_VERBOSE=1 to see the offending case.
"""

import os
import random
import sys
import unittest
import warnings

import numpy as np
import sympy as sp
from sympy import And, Abs, Max, Min, Not, Or, Piecewise, atan, cos, exp, floor, log, sin, sqrt, symbols, tanh

sys.path.insert(0, os.environ.get("SYMJIT_PYTHON_PATH", os.path.join(os.path.dirname(__file__), "..")))

from symjit import compile_func

SEEDS = int(os.environ.get("SYMJIT_FUZZ_SEEDS", "100"))
START = int(os.environ.get("SYMJIT_FUZZ_START", "0"))
VERBOSE = bool(os.environ.get("SYMJIT_FUZZ_VERBOSE"))

x, y, z = symbols("x y z")
V = [x, y, z]
NPOINTS = 65

# the option combinations every case is compiled with
OPTIONS = [
    dict(opt_level=0),
    dict(opt_level=1),
    dict(opt_level=2),
    dict(opt_level=3, cse=False),
    dict(opt_level=2, fastmath=False),
    dict(opt_level=2, use_simd=False),
    dict(opt_level=3, enable_simd512=True),
    dict(opt_level=2, compress=True),
    dict(opt_level=2, compact=False),
    dict(opt_level=2, use_threads=False),
]

COMPLEX_OPTIONS = [
    dict(opt_level=0),
    dict(opt_level=2),
    dict(opt_level=3),
    dict(opt_level=2, fast_complex=False),
    dict(opt_level=2, use_simd=False),
    dict(opt_level=3, enable_simd512=True, fast_complex=False),
    dict(opt_level=2, parallel_mul=False),
    dict(opt_level=1, compress=True),
]


# ------------------------------------------------------------------ random expressions
UNARY = [
    sin, cos, lambda a: exp(-a * a), lambda a: log(1 + a * a), lambda a: sqrt(1 + a * a), Abs, tanh, atan, floor,
    lambda a: -a, lambda a: a**2, lambda a: a**3, lambda a: 1 / (1 + a * a),
]
BINARY = [
    lambda a, b: a + b, lambda a, b: a - b, lambda a, b: a * b, lambda a, b: a / (1 + b * b), Min, Max,
]


def random_real(rng, depth):
    if depth == 0 or rng.random() < 0.15:
        return rng.choice(V) if rng.random() < 0.7 else sp.Float(round(rng.uniform(-3, 3), 3))
    if rng.random() < 0.4:
        return rng.choice(UNARY)(random_real(rng, depth - 1))
    return rng.choice(BINARY)(random_real(rng, depth - 1), random_real(rng, depth - 1))


def case_real(rng):
    return [random_real(rng, rng.randint(2, 6)) for _ in range(rng.randint(1, 3))]


SHARED_FUNCS = [sin, cos, exp, lambda a: log(1 + a * a), tanh, atan, lambda a: sqrt(1 + a * a)]


def case_shared(rng):
    """Outputs that reuse the same calls, as outputs and inside other outputs"""
    base = [rng.choice(SHARED_FUNCS)(rng.choice([x, y, z, x * y, x + z, y - z])) for _ in range(rng.randint(2, 4))]
    outs = []
    for _ in range(rng.randint(2, 5)):
        a, b = rng.choice(base), rng.choice(base)
        outs.append(
            rng.choice(
                [a, a * b, a + b * rng.choice(V), a / (2 + b * b), a - rng.choice(V), a * a + b, rng.choice(SHARED_FUNCS)(a)]
            )
        )
    return outs


def _value(rng, depth):
    if depth == 0 or rng.random() < 0.25:
        return rng.choice(V + [sp.Float(round(rng.uniform(-2, 2), 2))])
    a, b = _value(rng, depth - 1), _value(rng, depth - 1)
    return rng.choice([a + b, a - b, a * b, sin(a), Abs(a), Max(a, b)])


def _condition(rng, depth):
    a, b = _value(rng, 1), _value(rng, 1)
    c = rng.choice([a < b, a <= b, a > b, a >= b])
    if depth > 0 and rng.random() < 0.4:
        # a simple second operand keeps And/Or binary (sympy flattens nested ones, and
        # symjit does not yet accept And/Or with more than two operands; see test_regressions)
        c2 = _condition(rng, 0)
        return rng.choice([And(c, c2), Or(c, c2), Not(c)])
    return c


def _piecewise(rng, depth):
    if depth == 0 or rng.random() < 0.2:
        return _value(rng, 1)
    branches = rng.randint(1, 2)  # one or two conditions, then the final `True` branch
    pw = Piecewise(*[(_piecewise(rng, depth - 1), _condition(rng, 1)) for _ in range(branches)],
                   (_piecewise(rng, depth - 1), True))
    return pw + (_value(rng, 1) if rng.random() < 0.5 else 0)


def case_piecewise(rng):
    return [_piecewise(rng, rng.randint(1, 2)) for _ in range(rng.randint(1, 2))]


COMPLEX_UNARY = [
    sin, cos, lambda a: exp(a / 4), lambda a: sqrt(1 + a * a), lambda a: -a, lambda a: a**2, lambda a: a**3,
    lambda a: 1 / (1 + a * a), lambda a: a / 2, lambda a: a / 3.5, lambda a: sqrt(1 + a * a) ** 3,
]
COMPLEX_BINARY = [lambda a, b: a + b, lambda a, b: a - b, lambda a, b: a * b, lambda a, b: a / (2 + b * b)]


def random_complex(rng, depth):
    if depth == 0 or rng.random() < 0.15:
        return rng.choice(V) if rng.random() < 0.75 else sp.Float(round(rng.uniform(-3, 3), 2))
    if rng.random() < 0.4:
        return rng.choice(COMPLEX_UNARY)(random_complex(rng, depth - 1))
    return rng.choice(COMPLEX_BINARY)(random_complex(rng, depth - 1), random_complex(rng, depth - 1))


def case_complex(rng):
    return [random_complex(rng, rng.randint(2, 5)) for _ in range(rng.randint(1, 3))]


# ------------------------------------------------------------------ checking
def inputs(n, seed, dtype):
    r = np.random.default_rng(seed)
    if dtype == complex:
        return r.uniform(-1.5, 1.5, (3, n)) + 1j * r.uniform(-1.5, 1.5, (3, n))
    return r.uniform(-2, 2, (3, n))


def reference(exprs, X, dtype):
    n = X.shape[1]
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = sp.lambdify(V, exprs, "numpy")
        return np.array([np.broadcast_to(np.asarray(r, dtype=dtype), (n,)) for r in f(*X)])


def build_case(family, gen, seed):
    """The expressions of a case, or None if sympy cannot build or evaluate them"""
    rng = random.Random(f"{family}-{seed}")
    try:
        exprs = gen(rng)
    except RecursionError:
        return None

    # sympy merges nested Piecewise into And/Or with more than two operands, which symjit
    # cannot compile yet (a known bug, see test_regressions.test_nary_boolean): skip those
    if any(len(b.args) > 2 for e in exprs for b in e.atoms(And, Or)):
        return None
    return exprs


class FuzzBase(unittest.TestCase):
    family = None
    dtype = float
    options = OPTIONS
    generator = None
    tol = 1e-8

    def check(self, seed, exprs, options, X, ref):
        n = X.shape[1]
        kwargs = dict(options)
        if self.dtype == complex:
            kwargs["dtype"] = "complex128"

        if VERBOSE:
            print(f"[{self.family}] seed {seed} {options} {exprs}", flush=True)

        f = compile_func(V, exprs, **kwargs)
        vec = np.array(f(*X)).reshape(len(exprs), n)
        scalar = np.array([np.ravel(f(*X[:, i])) for i in range(n)]).T

        for kind, got in (("vectorized", vec), ("scalar", scalar)):
            err = np.abs(got - ref) / (1 + np.abs(ref))
            if not np.all(err < self.tol):
                k = np.unravel_index(np.argmax(err), err.shape)
                self.fail(
                    f"{self.family} seed {seed}, options {options}, {kind}: relative error {err.max():.3g} "
                    f"in output {k[0]} at point {k[1]}\nexpressions: {exprs}"
                )

    def run_family(self):
        X = inputs(NPOINTS, 7, self.dtype)
        for seed in range(START, START + SEEDS):
            exprs = build_case(self.family, self.generator, seed)
            if exprs is None:
                continue
            try:
                ref = reference(exprs, X, self.dtype)
            except Exception:  # sympy could not lambdify/evaluate the case
                continue
            if not np.all(np.isfinite(ref)):
                continue
            for options in self.options:
                with self.subTest(seed=seed, options=options):
                    self.check(seed, exprs, options, X, ref)


class RealFuzz(FuzzBase):
    family = "real"
    generator = staticmethod(case_real)

    def test_real(self):
        self.run_family()


class SharedFuzz(FuzzBase):
    family = "shared"
    generator = staticmethod(case_shared)

    def test_shared_subexpressions(self):
        self.run_family()


class PiecewiseFuzz(FuzzBase):
    family = "piecewise"
    generator = staticmethod(case_piecewise)
    options = OPTIONS[:6] + OPTIONS[7:8]

    def test_piecewise(self):
        self.run_family()


class ComplexFuzz(FuzzBase):
    family = "complex"
    dtype = complex
    options = COMPLEX_OPTIONS
    generator = staticmethod(case_complex)

    def test_complex(self):
        self.run_family()


class LengthFuzz(unittest.TestCase):
    """Vectorized calls with lengths that are not multiples of the SIMD width or the thread chunks"""

    LENGTHS = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 63, 65, 100, 1001]
    OPTIONS = [dict(opt_level=2), dict(opt_level=2, enable_simd512=True), dict(opt_level=2, use_threads=False),
               dict(opt_level=3, compress=True)]

    def test_lengths(self):
        for seed in range(START, START + max(1, SEEDS // 2)):
            exprs = build_case("real", case_real, seed + 100000)
            if exprs is None:
                continue
            try:
                lam = sp.lambdify(V, exprs, "numpy")
            except Exception:
                continue
            for options in self.OPTIONS:
                f = compile_func(V, exprs, **options)
                for n in self.LENGTHS:
                    with self.subTest(seed=seed, options=options, n=n):
                        X = inputs(n, n, float)
                        ref = reference(exprs, X, float)
                        if not np.all(np.isfinite(ref)):
                            continue
                        got = np.array(f(*X)).reshape(len(exprs), n)
                        err = np.abs(got - ref) / (1 + np.abs(ref))
                        self.assertTrue(np.all(err < 1e-8), f"seed {seed}, n={n}: error {err.max():.3g}; {exprs}")


if __name__ == "__main__":
    unittest.main()

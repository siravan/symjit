"""Property tests: consistency between the ways symjit can run a function, and
mathematical identities.

Consistency: without fastmath (no FMA contraction) every execution path on a host must give
the same bits as the bytecode interpreter for real functions: scalar calls, the fast
kernel, `apply`, vectorized calls (SIMD kernels, threads), every optimization level and
option. Complex functions are compared with a tolerance, because the SIMD kernels use
different multiplication/division sequences than the scalar ones. With fastmath, results are
compared with a tolerance.

Identities: sin^2 + cos^2 = 1, exp(log x) = x, symmetries (sin(-x) = -sin(x), exact), and
others, checked at many random points with every option set, in ulps where possible.

    python -m unittest python/tests/test_properties.py -v
"""

import os
import platform
import sys
import unittest
import warnings
import zlib

import numpy as np
import sympy as sp
from sympy import (
    Abs, And, Heaviside, Max, Min, Or, Piecewise, Product, Sum, acos, asin, asinh, atan, atan2, cbrt, ceiling, cos,
    cosh, erf, exp, floor, gamma, log, sin, sinh, sqrt, symbols, tan, tanh,
)

sys.path.insert(0, os.environ.get("SYMJIT_PYTHON_PATH", os.path.join(os.path.dirname(__file__), "..")))

from symjit import compile_func

x, y, k, n = symbols("x y k n")
HOST_X86 = platform.machine().lower() in ("x86_64", "amd64")
NPTS = 1031  # not a multiple of any SIMD width
RNG = np.random.default_rng(2027)

# a bank of real expressions covering the instruction selection and register allocation
REAL = {
    "transcendental": [sin(x) * exp(-y * y) + log(1 + x * x), atan(x / (1 + y * y)) + tanh(y) * x**3],
    "piecewise": [Piecewise((sqrt(Abs(x * x + y)), x > y), (cos(x * y), True)),
                  Piecewise((x, x < -1), (y * y, x < 1), (x - y, True))],
    "min_max_abs": [Max(x, y) - Min(x * y, 1) + Abs(x - y), Min(x, y, 0.5) * Max(x * x, y, -1)],
    "powers": [(x * x + 1) ** 1.5 + y**7 + 1 / (x * x + 0.5), Abs(x) ** 0.5 * y**-2 + (y * y + 2) ** sp.Rational(1, 3)],
    "rounding": [floor(x * 3) - ceiling(y / 2), x - floor(x), Heaviside(x - y) * y],
    "logic": [Piecewise((1.0, And(x > 0, y < 0.5)), (2.0, Or(x < -1, y > 1)), (3.0, True))],
    "shared": [sin(x * y) + cos(x * y), sin(x * y) * cos(x * y) - exp(sin(x * y)), (sin(x * y) + y) ** 2],
    # more live values than registers: spills
    "pressure": [sum(sin(x + i) * cos(y - i) for i in range(24)),
                 sp.prod([(x + i * y) for i in range(1, 12)]) / sp.prod([(y * y + i) for i in range(1, 12)])],
    "division": [x / y - y / x, (x - y) / (x * x + y * y + 1), 1 / (1 + exp(-x))],
}

LOOPS = [Sum(x**k / (k + 1), (k, 0, n)), Product(1 + x / (k + 1), (k, 0, n))]

COMPLEX = [x * y + x / y, exp(x) * sin(y) - sqrt(x), log(x * y + 1) + cosh(x) / y, x**3 - y**2 * x + 1 / (x + y)]

# options that must not change the bits of a real function (with fastmath=False)
EXACT_OPTIONS = [
    dict(),
    dict(opt_level=0),
    dict(opt_level=1),
    dict(opt_level=3),
    dict(cse=False),
    dict(compact=False),
    dict(compress=True),
    dict(opt_level=3, compress=True),
    dict(use_simd=False),
    dict(enable_simd512=True),
    dict(use_threads=False),
]


def points(lo=-3.0, hi=3.0):
    return RNG.uniform(lo, hi, NPTS), RNG.uniform(lo, hi, NPTS)


def same_bits(a, b):
    """Equal, with NaNs equal to each other (not bitwise for NaN payloads)."""
    a, b = np.asarray(a), np.asarray(b)
    return (a == b) | (np.isnan(a) & np.isnan(b))


def compile_quiet(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compile_func(*args, **kwargs)


class Consistency(unittest.TestCase):
    def check_paths(self, exprs, X, Y, options, tag):
        """Every path of every option set gives the bits of the bytecode interpreter."""
        ref = compile_quiet([x, y], exprs, ty="bytecode", fastmath=False)
        R = ref(X, Y)
        idx = range(0, NPTS, 37)
        for kw in options:
            f = compile_quiet([x, y], exprs, fastmath=False, **kw)
            V = f(X, Y)
            for j, e in enumerate(exprs):
                bad = np.flatnonzero(~same_bits(V[j], R[j]))
                self.assertEqual(len(bad), 0,
                                 f"{tag} {kw} vectorized, {e} at {[(X[i], Y[i]) for i in bad[:3]]}: "
                                 f"{V[j][bad[:3]]} vs {R[j][bad[:3]]}")
            for i in idx:
                s = f(X[i], Y[i])
                self.assertTrue(all(same_bits(s, [r[i] for r in R])), f"{tag} {kw} scalar at {(X[i], Y[i])}: {s}")
                self.assertTrue(all(same_bits(f.apply([X[i], Y[i]]), s)), f"{tag} {kw} apply")
                one = f(X[i : i + 1], Y[i : i + 1])  # a one-element array (all SIMD tail)
                self.assertTrue(all(same_bits([o[0] for o in one], s)), f"{tag} {kw} length-1 array")

    def test_real_paths_are_bit_identical(self):
        for family, exprs in REAL.items():
            with self.subTest(family=family):
                X, Y = points()
                self.check_paths(exprs, X, Y, EXACT_OPTIONS, family)

    def test_fast_kernel(self):
        # single output, few inputs: scalar calls go through the register-argument kernel
        for family, exprs in REAL.items():
            for e in exprs:
                with self.subTest(expr=str(e)[:60]):
                    f = compile_quiet([x, y], e, fastmath=False)
                    g = compile_quiet([x, y], e, ty="bytecode", fastmath=False)
                    self.assertIsNotNone(f.fast_func())
                    X, Y = points()
                    for i in range(0, NPTS, 53):
                        self.assertTrue(same_bits(f(X[i], Y[i]), g(X[i], Y[i])), f"{e} at {(X[i], Y[i])}")

    def test_loops(self):
        f_ref = compile_quiet([x, n], LOOPS, ty="bytecode", fastmath=False)
        X, N = RNG.uniform(-1, 1, 101), RNG.integers(1, 40, 101).astype(float)
        R = f_ref(X, N)
        for kw in EXACT_OPTIONS:
            with self.subTest(**kw):
                V = compile_quiet([x, n], LOOPS, fastmath=False, **kw)(X, N)
                for v, r in zip(V, R):
                    np.testing.assert_array_equal(v, r)

    @unittest.skipUnless(HOST_X86, "x86-64 only")
    @unittest.expectedFailure
    def test_sse(self):
        # BUG: AmdSSEGenerator::shrink (rust/amd/sse.rs) handles `dst = s1 op s2` with
        # dst == s2 and a non-commutative op by swapping s1 and s2 (fxchg), which clobbers
        # s1. `ifelse` with dst == false_val reads Reg::Temp after such an `andnot`, so it
        # combines the false value with the bits of the true value instead of the mask:
        # Min(x*y, 1) + x returns x when x*y >= 2 (and garbage for 1 <= x*y < 2); Piecewise
        # and logical expressions are wrong at a third or more of the points.
        for family, exprs in REAL.items():
            with self.subTest(family=family):
                X, Y = points()
                self.check_paths(exprs, X, Y, [dict(ty="amd-sse"), dict(ty="amd-sse", use_simd=False)], family)

    @unittest.skipUnless(HOST_X86, "x86-64 only")
    def test_sse_without_branches(self):
        # the families without ifelse (Piecewise, Min/Max, Heaviside, logic); see test_sse
        families = {k: v for k, v in REAL.items() if k in ("transcendental", "powers", "shared", "pressure", "division")}
        for family, exprs in families.items():
            with self.subTest(family=family):
                X, Y = points()
                self.check_paths(exprs, X, Y, [dict(ty="amd-sse", use_simd=False)], family)

    def test_fastmath_is_close(self):
        for family, exprs in REAL.items():
            with self.subTest(family=family):
                X, Y = points()
                R = compile_quiet([x, y], exprs, ty="bytecode", fastmath=False)(X, Y)
                for kw in [dict(), dict(opt_level=3), dict(use_simd=False), dict(enable_simd512=True)]:
                    V = compile_quiet([x, y], exprs, fastmath=True, **kw)(X, Y)
                    for v, r in zip(V, R):
                        ok = same_bits(v, r) | np.isclose(v, r, rtol=1e-9, atol=1e-12)
                        if family == "rounding":
                            # floor/ceiling of a contracted product may land on the other integer
                            ok |= np.abs(v - r) <= 1
                        self.assertTrue(ok.all(), f"{family} {kw}: {v[~ok][:3]} vs {r[~ok][:3]}")

    def test_complex_paths(self):
        X = RNG.normal(size=257) + 1j * RNG.normal(size=257)
        Y = RNG.normal(size=257) + 1j * RNG.normal(size=257)
        R = compile_quiet([x, y], COMPLEX, ty="bytecode", fastmath=False, dtype="complex128")(X, Y)
        options = [dict(), dict(fast_complex=False), dict(parallel_mul=False), dict(opt_level=0), dict(opt_level=3),
                   dict(use_simd=False), dict(enable_simd512=True), dict(compress=True), dict(fastmath=True)]
        for kw in options:
            with self.subTest(**kw):
                kw = dict(fastmath=False) | kw
                f = compile_quiet([x, y], COMPLEX, dtype="complex128", **kw)
                V = f(X, Y)
                for v, r in zip(V, R):
                    np.testing.assert_allclose(v, r, rtol=1e-14)
                for i in range(0, 257, 17):
                    np.testing.assert_allclose(f(X[i], Y[i]), [r[i] for r in R], rtol=1e-14)

    def test_debug_mode(self):
        # ty="debug" runs native and bytecode code and raises if they differ
        X, Y = points()
        exprs = REAL["transcendental"] + REAL["piecewise"] + REAL["powers"]
        f = compile_quiet([x, y], exprs, ty="debug", fastmath=False)
        R = compile_quiet([x, y], exprs, fastmath=False)(X, Y)
        for v, r in zip(f(X, Y), R):
            self.assertTrue(same_bits(v, r).all())


# ------------------------------------------------------------------ identities
# (name, lhs, rhs, domain of x, tolerance in ulps of max(|lhs|, |rhs|, scale), scale);
# the tolerances are about 1.5 times the worst error seen over 40 x 1031 points
IDENTITIES = [
    ("sin^2 + cos^2", sin(x) ** 2 + cos(x) ** 2, 1, (-100, 100), 2, 1),
    ("cosh^2 - sinh^2", cosh(x) ** 2 - sinh(x) ** 2, 1, (-3, 3), 8, 100),
    ("tan", tan(x), sin(x) / cos(x), (-1.5, 1.5), 4, 1),
    ("tanh", tanh(x), sinh(x) / cosh(x), (-20, 20), 4, 1),
    ("exp(log x)", exp(log(x)), x, (1e-3, 1e3), 2, 0),
    ("log(exp x)", log(exp(x)), x, (-700, 700), 2, 1),
    ("exp(a+b)", exp(x + 0.75), exp(x) * exp(0.75), (-300, 300), 4, 0),
    ("log(ab)", log(x * 3.5), log(x) + log(3.5), (1e-3, 1e3), 4, 1),
    ("atan(tan x)", atan(tan(x)), x, (-1.5, 1.5), 2, 0),
    # asin and acos are ill-conditioned near |sin x| = 1 and |cos x| = 1
    ("asin(sin x)", asin(sin(x)), x, (-1.2, 1.2), 2, 0),
    ("acos(cos x)", acos(cos(x)), x, (0.4, 2.7), 4, 0),
    ("asinh(sinh x)", asinh(sinh(x)), x, (-20, 20), 3, 0),
    ("sqrt^2", sqrt(x) ** 2, x, (0, 1e6), 1, 0),
    ("cbrt^3", cbrt(x) ** 3, x, (1e-6, 1e6), 3, 0),
    # x + 1 is exact for these x (multiples of 2^-40): gamma's condition number x psi(x)
    # would turn its rounding into tens of ulps
    ("gamma(x+1)", gamma(x + 1), x * gamma(x), (0.1, 20), 12, 0),
    ("atan2", atan2(x, 2.0), atan(x / 2.0), (-1e3, 1e3), 2, 0),
    ("double angle", sin(2 * x), 2 * sin(x) * cos(x), (-10, 10), 4, 1),
    ("logistic", 1 / (1 + exp(-x)) + 1 / (1 + exp(x)), 1, (-30, 30), 2, 1),
    ("erf odd", erf(x) + erf(-x), 0, (-5, 5), 0, 1),
]

# exact (bitwise) symmetries and IEEE facts
EXACT = [
    ("sin odd", sin(-x), -sin(x)),
    ("cos even", cos(-x), cos(x)),
    ("tan odd", tan(-x), -tan(x)),
    ("tanh odd", tanh(-x), -tanh(x)),
    ("atan odd", atan(-x), -atan(x)),
    ("sinh odd", sinh(-x), -sinh(x)),
    ("cosh even", cosh(-x), cosh(x)),
    ("erf odd", erf(-x), -erf(x)),
    ("sqrt(x^2) = |x|", sqrt(x * x), Abs(x)),  # without over/underflow of x*x
    ("x^2 = x*x", x**2, x * x),
    ("|x|^2 = x*x", Abs(x) ** 2, x * x),
    ("floor(-x) = -ceil(x)", floor(-x), -ceiling(x)),
    ("min/max", Min(x, 0.25) + Max(x, 0.25), x + 0.25),
]

IDENTITY_OPTIONS = [dict(), dict(fastmath=False), dict(use_simd=False), dict(opt_level=0), dict(enable_simd512=True)]


class Identities(unittest.TestCase):
    def test_identities(self):
        for name, lhs, rhs, (lo, hi), ulps, scale in IDENTITIES:
            rng = np.random.default_rng(zlib.crc32(name.encode()))  # independent of test order
            X = np.round(rng.uniform(lo, hi, NPTS) * 2.0**40) / 2.0**40
            for kw in IDENTITY_OPTIONS:
                with self.subTest(identity=name, **kw):
                    f = compile_quiet([x], [lhs, sp.sympify(rhs) + 0 * x], **kw)
                    L, R = f(X)
                    s = f(float(X[0]))
                    self.assertEqual(s, [L[0], R[0]])  # scalar = vectorized
                    tol = (ulps + 0.5) * np.spacing(np.maximum(np.maximum(np.abs(L), np.abs(R)), scale))
                    err = np.abs(L - R)
                    i = int(np.argmax(err - tol))
                    self.assertTrue((err <= tol).all(),
                                    f"{name} {kw}: x = {X[i]!r}: {L[i]!r} vs {R[i]!r} "
                                    f"({err[i] / np.spacing(max(abs(L[i]), abs(R[i]), scale)):.1f} ulp)")

    def test_exact_symmetries(self):
        X = np.concatenate([RNG.uniform(-10, 10, NPTS), [0.0, -0.0, 1e-300, 5e-324, 1e300, 0.5, 2.5, -2.5]])
        for name, lhs, rhs in EXACT:
            if name.startswith("sqrt"):
                X = X[(np.abs(X) > 1e-150) & (np.abs(X) < 1e150)]
            for kw in IDENTITY_OPTIONS:
                with self.subTest(identity=name, **kw):
                    L, R = compile_quiet([x], [lhs, rhs], **kw)(X)
                    ok = same_bits(L, R)
                    self.assertTrue(ok.all(), f"{name} {kw}: x = {X[~ok][:3]}: {L[~ok][:3]} vs {R[~ok][:3]}")

    def test_complex_identities(self):
        Z = (RNG.normal(size=257) + 1j * RNG.normal(size=257)) * 2
        cases = [
            ("exp(z) exp(-z)", exp(x) * exp(-x), 1),
            ("sin^2 + cos^2", sin(x) ** 2 + cos(x) ** 2, 1),
            ("sqrt(z)^2", sqrt(x) ** 2 - x, 0),
            ("log(exp z)", log(exp(x)) - x, 0),  # |Im z| < pi for these points
            ("cosh^2 - sinh^2", cosh(x) ** 2 - sinh(x) ** 2, 1),
        ]
        Z = Z[np.abs(Z.imag) < 3]
        for name, e, want in cases:
            for kw in [dict(), dict(fast_complex=False), dict(use_simd=False)]:
                with self.subTest(identity=name, **kw):
                    v = compile_quiet([x], [e], dtype="complex128", **kw)(Z)[0]
                    # cancellation: the terms are as large as exp(2 |z|)
                    scale = np.exp(2 * np.abs(Z))
                    err = np.abs(v - want) / scale
                    self.assertLess(err.max(), 1e-14, f"{name} {kw}: z = {Z[np.argmax(err)]}")

    def test_complex_conjugate_symmetry(self):
        # f(conj z) = conj f(z) for functions real on the real axis (off their branch cuts)
        Z = RNG.normal(size=257) + 1j * RNG.uniform(0.1, 2, 257)
        for e in [exp(x), sin(x), cos(x), sinh(x), tanh(x), log(x), sqrt(x), x**3 + 2 * x, 1 / (x * x + 1)]:
            for kw in [dict(), dict(fast_complex=False)]:
                with self.subTest(f=str(e), **kw):
                    f = compile_quiet([x], [e], dtype="complex128", **kw)
                    np.testing.assert_allclose(f(np.conj(Z))[0], np.conj(f(Z)[0]), rtol=4e-16, atol=1e-300)


if __name__ == "__main__":
    unittest.main()

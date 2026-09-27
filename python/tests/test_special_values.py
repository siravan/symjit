"""Special values: NaN, +-Inf, +-0, subnormal and huge arguments.

Every compiled function is compared, scalar and vectorized and for several option sets, with
numpy/scipy (which follow C99 Annex F): NaN matches NaN, infinities match exactly, zeros must have
the same sign, and finite values agree to 1e-13 relative.

Deviations that are known and open are recorded as `expectedFailure` tests that state the
difference; when one is fixed the test starts to pass ("unexpected success") and the marker
should be removed.

Set SYMJIT_PYTHON_PATH to test another build of the package.
"""

import itertools
import os
import sys
import unittest
import warnings

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("SYMJIT_PYTHON_PATH", os.path.join(HERE, "..")))
sys.path.insert(0, HERE)

from accuracy_cases import special_unary  # noqa: E402
from symjit import compile_func  # noqa: E402

x, y = sp.symbols("x y")
NAN, INF = np.nan, np.inf

UNARY_POINTS = np.array([
    NAN, INF, -INF, 0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.5, -3.5, 5e-324, -5e-324,
    2.2250738585072014e-308, 1.7976931348623157e308, -1.7976931348623157e308, 1e-300, 1e300, 709.0, 710.0,
    -745.0, -746.0, 1e22, np.pi / 2, np.pi, 100.0, 1e-10, 0.9999999999999999, 1.0000000000000002,
])

BINARY_POINTS = np.array([
    NAN, INF, -INF, 0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.0, 1e308, -1e308, 5e-324, 1e-300, 7.5, -7.5,
])

OPTIONS = [dict(), dict(fastmath=False), dict(use_simd=False, opt_level=0), dict(opt_level=3, compress=True)]


def mismatches(got, want, rtol=1e-13, signed_zero=True):
    """Indices where `got` differs from `want` (NaN == NaN, signed zeros distinguished if asked)"""
    bad = []
    for i, (g, w) in enumerate(zip(np.asarray(got, float), np.asarray(want, float))):
        if np.isnan(g) and np.isnan(w):
            continue
        if np.isnan(g) or np.isnan(w) or np.isinf(g) or np.isinf(w):
            if not (g == w):
                bad.append(i)
        elif g == 0.0 and w == 0.0:
            if signed_zero and np.signbit(g) != np.signbit(w):
                bad.append(i)
        elif abs(g - w) > rtol * max(abs(g), abs(w)):
            bad.append(i)
    return bad


def describe(bad, args, got, want, limit=4):
    lines = []
    for i in bad[:limit]:
        a = ", ".join(repr(float(c[i])) for c in args)
        lines.append(f"f({a}) = {got[i]!r}, expected {want[i]!r}")
    return "; ".join(lines) + (f" (+{len(bad) - limit} more)" if len(bad) > limit else "")


class SpecialValues(unittest.TestCase):
    def check_unary(self, name, sym, ref, points=UNARY_POINTS):
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            want = ref(points)
        for options in OPTIONS:
            f = compile_func([x], [sym(x)], **options)
            vec = np.ravel(f(points))
            scalar = np.array([np.ravel(f(float(p)))[0] for p in points])
            for kind, got in (("vectorized", vec), ("scalar", scalar)):
                with self.subTest(function=name, options=options, kind=kind):
                    bad = mismatches(got, want)
                    self.assertFalse(bad, f"{name} {options} {kind}: " + describe(bad, [points], got, want))

    def check_binary(self, name, expr, ref, points=BINARY_POINTS, signed_zero=True):
        a, b = (t.ravel() for t in np.meshgrid(points, points))
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            want = ref(a, b)
        for options in OPTIONS:
            f = compile_func([x, y], [expr], **options)
            vec = np.ravel(f(a, b))
            scalar = np.array([np.ravel(f(float(p), float(q)))[0] for p, q in zip(a, b)])
            for kind, got in (("vectorized", vec), ("scalar", scalar)):
                with self.subTest(function=name, options=options, kind=kind):
                    bad = mismatches(got, want, signed_zero=signed_zero)
                    self.assertFalse(bad, f"{name} {options} {kind}: " + describe(bad, [a, b], got, want))

    # ---------------------------------------------------------------- unary functions
    def test_unary_functions(self):
        for name, (sym, ref) in special_unary().items():
            if name == "gamma":
                continue  # see test_gamma_at_zero
            self.check_unary(name, sym, ref)

    def test_gamma(self):
        pts = UNARY_POINTS[~np.isin(UNARY_POINTS, [0.0])]  # +-0 are in test_gamma_at_zero
        pts = pts[(pts != 0.0)]
        sym, ref = special_unary()["gamma"]
        self.check_unary("gamma", sym, ref, pts)

    @unittest.expectedFailure
    def test_gamma_at_zero(self):
        # KNOWN DEVIATION: gamma(+-0) is NaN (C99 tgamma(+-0) = +-inf)
        sym, ref = special_unary()["gamma"]
        self.check_unary("gamma", sym, ref, np.array([0.0, -0.0]))

    # ---------------------------------------------------------------- arithmetic and comparisons
    def test_arithmetic(self):
        self.check_binary("x+y", x + y, np.add)
        self.check_binary("x-y", x - y, np.subtract)
        self.check_binary("x*y", x * y, np.multiply)
        self.check_binary("x/y", x / y, np.divide)
        self.check_binary("x**y", x**y, np.power)
        self.check_binary("atan2", sp.atan2(x, y), np.arctan2)

    def test_ordered_comparisons(self):
        # Piecewise((1, cond), (0, True)) == the IEEE comparison; both `<` and `<=` are correct
        for name, cond, ref in (("<", x < y, lambda a, b: (a < b) * 1.0), ("<=", x <= y, lambda a, b: (a <= b) * 1.0),
                                ("==", sp.Eq(x, y), lambda a, b: (a == b) * 1.0),
                                ("!=", sp.Ne(x, y), lambda a, b: (a != b) * 1.0)):
            self.check_binary(name, sp.Piecewise((1.0, cond), (0.0, True)), ref)

    @unittest.expectedFailure
    def test_greater_than_with_nan(self):
        # KNOWN DEVIATION: `x > y` and `x >= y` are true when an operand is NaN (IEEE: false)
        for name, cond, ref in ((">", x > y, lambda a, b: (a > b) * 1.0), (">=", x >= y, lambda a, b: (a >= b) * 1.0)):
            self.check_binary(name, sp.Piecewise((1.0, cond), (0.0, True)), ref)

    def test_greater_than_without_nan(self):
        finite = BINARY_POINTS[~np.isnan(BINARY_POINTS)]
        for name, cond, ref in ((">", x > y, lambda a, b: (a > b) * 1.0), (">=", x >= y, lambda a, b: (a >= b) * 1.0)):
            self.check_binary(name, sp.Piecewise((1.0, cond), (0.0, True)), ref, finite)

    # ---------------------------------------------------------------- Min and Max
    def test_min_max_without_nan(self):
        finite = BINARY_POINTS[~np.isnan(BINARY_POINTS)]
        # the sign of a zero result is not checked: Min(0.0, -0.0) may return either
        self.check_binary("Min", sp.Min(x, y), np.minimum, finite, signed_zero=False)
        self.check_binary("Max", sp.Max(x, y), np.maximum, finite, signed_zero=False)

    def test_min_max_nan_operand(self):
        # DOCUMENTED BEHAVIOR: when an operand is NaN, Min returns its second operand and Max its
        # first one, so the result depends on the operand order (sympy also sorts the arguments of
        # Min/Max). It is neither numpy's `minimum`/`maximum` (NaN propagates) nor `fmin`/`fmax`
        # (NaN is ignored).
        pts = np.array([NAN, 1.0, -2.0, INF])
        for expr, name, pick in ((sp.Min(x, y), "Min", lambda a, b: b), (sp.Max(x, y), "Max", lambda a, b: a)):
            f = compile_func([x, y], [expr])
            for a, b in itertools.product(pts, pts):
                if np.isnan(a) or np.isnan(b):
                    got = float(np.ravel(f(float(a), float(b)))[0])
                    want = float(pick(a, b))
                    self.assertTrue(got == want or (np.isnan(got) and np.isnan(want)), f"{name}({a}, {b}) = {got}")

    # ---------------------------------------------------------------- Mod and Heaviside
    def test_mod_finite(self):
        # sympy's Mod is x - y*floor(x/y) and must equal Python's `%` for ordinary values
        pts = np.array([7.5, -7.5, 2.0, -2.0, 3.0, 0.5, -0.5, 1000.0, 123.456, -0.75])
        # the sign of a zero result is not checked: Python gives -0.0 for x % -x, symjit +0.0
        self.check_binary("Mod", sp.Mod(x, y), np.mod, pts, signed_zero=False)

    @unittest.expectedFailure
    def test_mod_large_quotients(self):
        # KNOWN DEVIATION: x - y*floor(x/y) breaks down when |x/y| > 2^52: Mod(1e20, 3) = 4096
        # (numpy: 1.0), and the result is outside [0, y)
        pts = np.array([1e16, 1e20, 1e308])
        a, b = np.meshgrid(pts, np.array([3.0, 7.0]))
        self.check_binary("Mod", sp.Mod(x, y), np.mod, np.array([1e20, 1e308, 3.0, 7.0]))

    @unittest.expectedFailure
    def test_mod_result_range(self):
        # KNOWN DEVIATION: the result can have the wrong sign: Mod(1, 0.1) = -5.55e-17
        f = compile_func([x, y], [sp.Mod(x, y)])
        got = float(np.ravel(f(1.0, 0.1))[0])
        self.assertGreaterEqual(got, 0.0)

    @unittest.expectedFailure
    def test_mod_infinite_divisor(self):
        # KNOWN DEVIATION: Mod(1, inf) is NaN (Python: 1.0)
        self.check_binary("Mod", sp.Mod(x, y), np.mod, np.array([1.0, 0.5, 3.0, INF]))

    @unittest.expectedFailure
    def test_heaviside_nan(self):
        # KNOWN DEVIATION: Heaviside(NaN) = 1 (numpy: NaN)
        self.check_unary("Heaviside", sp.Heaviside, lambda a: np.heaviside(a, 0.5), UNARY_POINTS)

    def test_heaviside(self):
        pts = UNARY_POINTS[~np.isnan(UNARY_POINTS)]
        self.check_unary("Heaviside", sp.Heaviside, lambda a: np.heaviside(a, 0.5), pts)

    # ---------------------------------------------------------------- unsupported
    @unittest.expectedFailure
    def test_sign_is_supported(self):
        # KNOWN LIMITATION: sympy.sign is not compilable ("op_code sign is not found")
        compile_func([x], [sp.sign(x)])


if __name__ == "__main__":
    unittest.main()

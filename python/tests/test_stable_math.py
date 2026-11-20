"""`stable_math=True`: range-safe complex division, abs, sqrt and negative powers.

The Rust test `tests/stable_math.rs` covers the native kernels in detail; this test adds the
Python API, `ty="bytecode"`, `ty="amd-sse"` and AVX-512 (`enable_simd512`), each compared with
exact results and, in the normal range, with the default code bit for bit.
"""

import cmath
import math
import os
import platform
import sys
import unittest
from fractions import Fraction

import numpy as np
import sympy as sp

sys.path.insert(0, os.environ.get("SYMJIT_PYTHON_PATH", os.path.join(os.path.dirname(__file__), "..")))

from symjit import compile_func

x, y = sp.symbols("x y")
EQS = [x / y, 1 / y, sp.Abs(x), sp.sqrt(x), y**-2]

TINY = 5e-324
EXTREME = [
    (1e308 + TINY * 1j, 1 + 0j),
    (3e200 + 4e200j, 1e200 - 2e200j),
    (1e-200 + 1e200j, 1e200 + 1e-200j),
    (TINY + TINY * 1j, 1e308 + 1e308j),
    (3e-320 + 4e-320j, 1e-320 - 2e-320j),
    (-4 + 1e-300j, 1e160 + 1e160j),
    (1e300 + 1e-300j, 1e-160 - 1e-160j),
]


def exact_div(a, b):
    p, q, r, s = map(Fraction, (a.real, a.imag, b.real, b.imag))
    den = r * r + s * s
    return (p * r + q * s) / den, (q * r - p * s) / den


def to_float(f):
    try:
        return float(f)
    except OverflowError:
        return math.inf if f > 0 else -math.inf


def close(got, re, im, ulps):
    got = complex(got)
    mag = max(abs(re), abs(im))
    if math.isinf(mag):  # an overflowing result: the infinite components must match
        return all(g == e for g, e in [(got.real, re), (got.imag, im)] if math.isinf(e))
    tol = ulps * (mag * 2.0**-52 if mag >= 2.0**-1022 else 0.0) + 4 * TINY
    return abs(got.real - re) <= tol and abs(got.imag - im) <= tol


def reference_sqrt(z):
    e = math.frexp(max(abs(z.real), abs(z.imag)))[1] // 2
    zs = complex(math.ldexp(z.real, -2 * e), math.ldexp(z.imag, -2 * e))
    w = cmath.sqrt(zs)
    t = math.ldexp(abs(w.real) if z.real >= 0 else abs(w.imag), e)
    u = z.imag / (2 * t)
    return complex(t, u) if z.real >= 0 else complex(abs(u), math.copysign(t, z.imag))


class StableMath(unittest.TestCase):
    def kernels(self):
        yield "native", dict(ty="native")
        yield "native simd512", dict(ty="native", enable_simd512=True)
        yield "bytecode", dict(ty="bytecode")
        if platform.machine() in ("x86_64", "AMD64"):   # x86 code runs on x86 only
            yield "amd-sse", dict(ty="amd-sse")

    def evaluate(self, options, points, stable, fast_complex, vectorized):
        f = compile_func(
            [x, y], EQS, dtype="complex128", fastmath=False, stable_math=stable, fast_complex=fast_complex, **options
        )
        if vectorized:
            X = np.array([a for a, _ in points])
            Y = np.array([b for _, b in points])
            return np.array(f(X, Y)).T
        return np.array([f(a, b) for a, b in points])

    def test_extreme_values(self):
        for name, options in self.kernels():
            for fast_complex in [True, False]:
                for vectorized in [False, True]:
                    res = self.evaluate(options, EXTREME, True, fast_complex, vectorized)
                    for (a, b), outs in zip(EXTREME, res):
                        what = f"{name} fast_complex={fast_complex} vectorized={vectorized} x={a} y={b}"
                        q = [to_float(v) for v in exact_div(a, b)]
                        r = [to_float(v) for v in exact_div(1 + 0j, b)]
                        self.assertTrue(close(outs[0], *q, 4), f"{what}: x/y = {outs[0]}, exact {q}")
                        self.assertTrue(close(outs[1], *r, 4), f"{what}: 1/y = {outs[1]}, exact {r}")
                        self.assertTrue(close(outs[2], math.hypot(a.real, a.imag), 0.0, 2), f"{what}: |x| = {outs[2]}")
                        w = reference_sqrt(a)
                        self.assertTrue(close(outs[3], w.real, w.imag, 4), f"{what}: sqrt(x) = {outs[3]}, ref {w}")

    def test_normal_range_bits(self):
        # not `amd-sse`: its default code uses formulas without FMA, the stable code the
        # FMA formulas of the AVX kernels (SSE is deprecated)
        rng = np.random.default_rng(7)
        n = 300
        X = (rng.normal(size=n) + 1j * rng.normal(size=n)) * 10.0 ** rng.uniform(-30, 30, n)
        Y = (rng.normal(size=n) + 1j * rng.normal(size=n)) * 10.0 ** rng.uniform(-30, 30, n)
        points = list(zip(X, Y))
        for name, options in self.kernels():
            if name == "amd-sse":
                continue
            for fast_complex in [True, False]:
                for vectorized in [False, True]:
                    a = self.evaluate(options, points, False, fast_complex, vectorized)
                    b = self.evaluate(options, points, True, fast_complex, vectorized)
                    # y**-2 takes the reciprocal first, which changes its rounding
                    self.assertTrue(
                        np.array_equal(a[:, :4], b[:, :4], equal_nan=True),
                        f"{name} fast_complex={fast_complex} vectorized={vectorized}",
                    )

    def test_negative_powers(self):
        # |y|^2 overflows (or underflows) while y^-2 is representable
        for name, options in self.kernels():
            for dtype, ys in [("float64", [1e155, -3e160, 1e-155]), ("complex128", [1e155 + 1e155j, 8e-155 + 8e-155j])]:
                f = compile_func([y], [y**-2, y**-3], dtype=dtype, stable_math=True, fastmath=False, **options)
                for v in ys:
                    got = f(v)
                    for k, g in zip([2, 3], got):
                        # exact: (1/v)^k
                        re, im = exact_div(1 + 0j, complex(v))
                        p, q = Fraction(1), Fraction(0)
                        for _ in range(k):
                            p, q = p * re - q * im, p * im + q * re
                        e = complex(to_float(p), to_float(q))
                        self.assertTrue(close(g, e.real, e.imag, 16), f"{name} {dtype}: {v}**-{k} = {g}, exact {e}")


if __name__ == "__main__":
    unittest.main()

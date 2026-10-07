"""Tests of the public Python API against independent references.

Covers what the fuzzers do not exercise: the calling conventions of `Func` (output formats,
parameters, `apply`, arrays of any shape), `compile_ode` and `compile_jac` with scipy's
`solve_ivp`, `compile_json` (CellML models), `save`/`load_func`, the scipy `LowLevelCallable`s
(`callable_quad`, `callable_filter`), user-defined functions (`defuns`), `Sum`/`Product`
loops, the Symbolica bridge (`compile_evaluator`) and input checking.

Cases that crash the process are run in a subprocess. Known bugs are `expectedFailure`
tests that state the defect.

    python -m unittest python/tests/test_api.py -v
"""

import importlib.util
import math
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
import warnings

import numpy as np
import sympy as sp
from sympy import Function, Max, Min, Piecewise, Product, Sum, cos, exp, log, sin, sqrt, symbols, tanh

PYTHON_PATH = os.environ.get("SYMJIT_PYTHON_PATH", os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PYTHON_PATH)

from symjit import compile_func, compile_jac, compile_json, compile_ode, load_func

try:
    import scipy.integrate
    import scipy.ndimage

    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

HAVE_SYMBOLICA = importlib.util.find_spec("symbolica") is not None
CELLML = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "cellml")

x, y, z, a, b, k, n, t = symbols("x y z a b k n t")
RNG = np.random.default_rng(11)


def run_isolated(code, timeout=120):
    """Runs `code` in a fresh interpreter (for cases that may abort the process) and
    returns (returncode, output)."""
    prelude = f"import sys; sys.path.insert(0, {PYTHON_PATH!r})\n"
    p = subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return p.returncode, p.stdout + p.stderr


# ------------------------------------------------------------------ calling conventions
class Calls(unittest.TestCase):
    def test_output_formats(self):
        # a list gives a list, a tuple a tuple, a single expression a float
        self.assertEqual(compile_func([x, y], [x + y, x * y])(1, 2), [3.0, 2.0])
        self.assertEqual(compile_func([x, y], (x + y, x * y))(1, 2), (3.0, 2.0))
        self.assertEqual(compile_func([x, y], x * y)(2, 3), 6.0)
        self.assertEqual(compile_func(x, x + 1)(2), 3.0)

    def test_fast_func(self):
        f = compile_func([x, y], x * y + sin(x))
        g = f.fast_func()
        self.assertIsNotNone(g)
        self.assertEqual(g(0.5, 3.0), 0.5 * 3.0 + math.sin(0.5))

    def test_params(self):
        f = compile_func([x], [a * x + b], params=[a, b])
        self.assertEqual(f(2.0, 3.0, 1.0), [7.0])
        # parameters persist between calls
        self.assertEqual(f(4.0), [13.0])
        X = np.linspace(-1, 1, 9)
        np.testing.assert_array_equal(f(X, 0.5, -1.0)[0], 0.5 * X - 1.0)

    def test_apply(self):
        f = compile_func([x, y], [x - y, x * a], params=[a])
        np.testing.assert_array_equal(f.apply([3.0, 1.0], [2.0]), [2.0, 6.0])
        np.testing.assert_array_equal(f.apply(np.array([1.0, 1.0])), [0.0, 2.0])

    def test_array_shapes(self):
        f = compile_func([x, y], [x * y, x - y])
        for shape in [(1,), (7,), (3, 4), (2, 3, 5), (1000,), (4097,)]:
            X, Y = RNG.normal(size=shape), RNG.normal(size=shape)
            with self.subTest(shape=shape):
                u, v = f(X, Y)
                self.assertEqual(u.shape, shape)
                np.testing.assert_array_equal(u, X * Y)
                np.testing.assert_array_equal(v, X - Y)

    def test_array_kinds(self):
        f = compile_func([x, y], [x * y + 1])
        base = np.arange(40.0)
        X = base[::2]  # strided
        Y = np.asfortranarray(np.arange(20.0).reshape(4, 5)).ravel(order="F")
        np.testing.assert_array_equal(f(X, Y)[0], X * Y + 1)
        I = np.arange(10)  # integers are converted
        np.testing.assert_array_equal(f(I, I)[0], (I * I + 1).astype(float))
        T = np.arange(12.0).reshape(3, 4).T  # transposed, not contiguous
        np.testing.assert_array_equal(f(T, T)[0], T * T + 1)

    def test_vectorized_matches_scalar(self):
        exprs = [sin(x) * exp(-y), sqrt(x * x + y * y), Piecewise((x, x > y), (y, True)), tanh(x - y)]
        f = compile_func([x, y], exprs)
        X, Y = RNG.uniform(-3, 3, 257), RNG.uniform(-3, 3, 257)
        V = f(X, Y)
        for i in range(0, 257, 16):
            np.testing.assert_array_equal([v[i] for v in V], f(X[i], Y[i]))

    def test_threads_do_not_change_results(self):
        e = [sin(x) * cos(y) + exp(-x * y)]
        X, Y = RNG.normal(size=100_003), RNG.normal(size=100_003)
        u = compile_func([x, y], e, use_threads=True)(X, Y)[0]
        v = compile_func([x, y], e, use_threads=False)(X, Y)[0]
        np.testing.assert_array_equal(u, v)

    def test_many_states(self):
        X = symbols("X[0:3000]")
        f = compile_func(list(X), sum(X[i] * (i % 7) for i in range(3000)))
        vals = RNG.normal(size=3000)
        self.assertAlmostEqual(f(*vals), sum(vals[i] * (i % 7) for i in range(3000)), places=9)

    def test_complex_calls(self):
        f = compile_func([x, y], [x * y, x / y + 1], dtype="complex128")
        u, v = 1 + 2j, 3 - 1j
        np.testing.assert_allclose(f(u, v), [u * v, u / v + 1], rtol=1e-15)
        U, V = RNG.normal(size=33) + 1j * RNG.normal(size=33), RNG.normal(size=33) + 1j * RNG.normal(size=33)
        np.testing.assert_allclose(f(U, V)[0], U * V, rtol=1e-15)

    @unittest.expectedFailure
    def test_missing_argument_is_an_error(self):
        # BUG: a call with fewer arguments than states broadcasts the given ones
        # (f(1.0) with two states computes f(1.0, 1.0)) instead of raising
        f = compile_func([x, y], [x + y, x - y])
        with self.assertRaises((ValueError, TypeError, AssertionError)):
            f(1.0)

    def test_mismatched_arrays_are_an_error(self):
        f = compile_func([x, y], [x + y])
        with self.assertRaises(AssertionError):
            f(np.arange(4.0), np.arange(5.0))


# ------------------------------------------------------------------ loops
class Loops(unittest.TestCase):
    def test_sum(self):
        f = compile_func([n], Sum(1 / k**2, (k, 1, n)))
        for m in [1, 2, 10, 1000]:
            self.assertAlmostEqual(f(float(m)), sum(1 / j**2 for j in range(1, m + 1)), places=13)

    def test_sum_fixed_bounds(self):
        f = compile_func([x], Sum(x**k / sp.gamma(k + 1), (k, 0, 20)))
        self.assertAlmostEqual(f(1.0), math.e, places=14)
        f = compile_func([x, n], Sum(x**k, (k, 0, n - 1)))
        self.assertAlmostEqual(f(0.5, 4.0), 1.875, places=15)

    def test_product(self):
        f = compile_func([n], Product(1 + 1 / k, (k, 1, n)))
        for m in [1, 5, 50]:
            self.assertAlmostEqual(f(float(m)), m + 1.0, places=11)

    def test_sum_vectorized(self):
        f = compile_func([x, n], Sum(x / k, (k, 1, n)))
        X, N = RNG.uniform(-1, 1, 37), RNG.integers(1, 30, 37).astype(float)
        want = [xx * sum(1 / j for j in range(1, int(m) + 1)) for xx, m in zip(X, N)]
        np.testing.assert_allclose(f(X, N), want, rtol=1e-14)

    def test_sum_two_independent_loops(self):
        f = compile_func([n], Sum(k, (k, 1, n)) * Product(2, (k, 1, 3)))
        self.assertEqual(f(10.0), 55.0 * 8)

    @unittest.expectedFailure
    def test_empty_sum(self):
        # BUG: loops run their body at least once: Sum(k=1..0) returns the first term
        # instead of 0 (native and bytecode alike)
        for ty in ["native", "bytecode"]:
            self.assertEqual(compile_func([n], Sum(1 / k**2, (k, 1, n)), ty=ty)(0.0), 0.0)

    @unittest.expectedFailure
    def test_empty_product(self):
        # BUG: as test_empty_sum; Product(k=1..0) should be 1
        self.assertEqual(compile_func([n], Product(2, (k, 1, n)))(0.0), 1.0)

    @unittest.expectedFailure
    def test_nested_sum_with_dependent_bound(self):
        # LIMITATION: an inner loop bound that depends on the outer index fails to compile
        # ("variable k not found")
        j = symbols("j")
        f = compile_func([n], Sum(Sum(j * k, (j, 1, k)), (k, 1, n)))
        self.assertEqual(f(4.0), sum(kk * jj for kk in range(1, 5) for jj in range(1, kk + 1)))


# ------------------------------------------------------------------ ODEs and Jacobians
@unittest.skipUnless(HAVE_SCIPY, "needs scipy")
class Odes(unittest.TestCase):
    def test_harmonic_oscillator(self):
        f = compile_ode(t, [x, y], [y, -x])
        sol = scipy.integrate.solve_ivp(f, (0, 10), [0.0, 1.0], rtol=1e-11, atol=1e-12, dense_output=True)
        T = np.linspace(0, 10, 50)
        np.testing.assert_allclose(sol.sol(T)[0], np.sin(T), atol=1e-8)

    def test_params_and_time(self):
        # x' = -a x + t, x(0) = 1: x = t/a - 1/a^2 + (1 + 1/a^2) exp(-a t)
        f = compile_ode(t, [x], [-a * x + t], params=[a])
        A = 2.0
        sol = scipy.integrate.solve_ivp(f, (0, 3), [1.0], args=(A,), rtol=1e-11, atol=1e-12)
        T = sol.t[-1]
        self.assertAlmostEqual(sol.y[0, -1], T / A - 1 / A**2 + (1 + 1 / A**2) * math.exp(-A * T), places=8)

    def test_rhs_matches_sympy(self):
        rhs = [10 * (y - x), x * (28 - z) - y, x * y - sp.Rational(8, 3) * z]
        f = compile_ode(t, [x, y, z], rhs)
        g = sp.lambdify([x, y, z], rhs)
        for _ in range(20):
            u = RNG.normal(size=3) * 10
            np.testing.assert_allclose(f(0.0, u), g(*u), rtol=1e-15)

    def test_returns_a_copy(self):
        f = compile_ode(t, [x], [-x])
        u = f(0.0, [1.0])
        f(0.0, [5.0])
        self.assertEqual(u[0], -1.0)

    def test_jacobian(self):
        rhs = [10 * (y - x), x * (28 - z) - y, x * y - sp.Rational(8, 3) * z]
        J = compile_jac(t, [x, y, z], rhs)
        Jref = sp.lambdify([x, y, z], sp.Matrix(rhs).jacobian([x, y, z]))
        for _ in range(10):
            u = RNG.normal(size=3) * 5
            np.testing.assert_allclose(J(0.0, u), Jref(*u), rtol=1e-15, atol=1e-15)

    def test_jacobian_with_params(self):
        rhs = [a * x * y, -b * y**2 + sin(x)]
        J = compile_jac(t, [x, y], rhs, params=[a, b])
        Jref = sp.lambdify([x, y, a, b], sp.Matrix(rhs).jacobian([x, y]))
        np.testing.assert_allclose(J(0.0, [0.3, -1.2], 2.0, 0.5), Jref(0.3, -1.2, 2.0, 0.5), rtol=1e-15)

    def test_stiff_van_der_pol_with_jacobian(self):
        mu = 1000.0
        rhs = [y, a * (1 - x**2) * y - x]
        f = compile_ode(t, [x, y], rhs, params=[a])
        J = compile_jac(t, [x, y], rhs, params=[a])
        g = sp.lambdify([x, y, a], rhs)
        opts = dict(method="Radau", rtol=1e-6, atol=1e-8, args=(mu,))
        s1 = scipy.integrate.solve_ivp(f, (0, 3000), [2.0, 0.0], jac=J, **opts)
        s2 = scipy.integrate.solve_ivp(lambda tt, u, m: g(*u, m), (0, 3000), [2.0, 0.0], **opts)
        self.assertTrue(s1.success and s2.success)
        self.assertGreater(s1.njev, 0)
        np.testing.assert_allclose(s1.y[:, -1], s2.y[:, -1], rtol=1e-3, atol=1e-6)

    def test_complex_ode(self):
        # z' = w z with w = i: z = exp(i t)
        w = symbols("w")
        f = compile_ode(t, [x], [w * x], params=[w], dtype="complex128")
        np.testing.assert_allclose(f(0j, [1 + 2j], 1j), [1j * (1 + 2j)])
        sol = scipy.integrate.solve_ivp(lambda tt, u: f(tt + 0j, u, 1j), (0, 1), [1 + 0j], rtol=1e-11, atol=1e-12)
        self.assertLess(abs(sol.y[0, -1] - np.exp(1j)), 1e-9)

    def test_complex_jacobian(self):
        J = compile_jac(t, [x, y], [x * y, x**2], dtype="complex128")
        u = [1 + 1j, 2 - 1j]
        np.testing.assert_allclose(J(0j, u), [[u[1], u[0]], [2 * u[0], 0]], rtol=1e-15)


@unittest.skipUnless(HAVE_SCIPY and os.path.isdir(CELLML), "needs scipy and examples/cellml")
class CellML(unittest.TestCase):
    def load(self, name, **kw):
        with open(os.path.join(CELLML, name)) as fd:
            return compile_json(fd.read(), **kw)

    def test_native_matches_bytecode(self):
        # without fastmath (no FMA contraction) native code and the bytecode interpreter
        # agree bit for bit, also on the large O'Hara-Rudy model
        for name in ["lorenz.json", "beeler.json", "tentusscher.json", "ohara.json"]:
            with self.subTest(model=name):
                f = self.load(name, fastmath=False)
                g = self.load(name, ty="bytecode", fastmath=False)
                h = self.load(name)
                u0, p = np.array(f.get_u0()), f.get_p()
                for s in range(5):
                    u = u0 * (1 + 0.01 * s)
                    want = g(0.5 * s, u, *p)
                    np.testing.assert_array_equal(f(0.5 * s, u, *p), want)
                    np.testing.assert_allclose(h(0.5 * s, u, *p), want, rtol=1e-10, atol=1e-15)

    def test_beeler_reuter_action_potential(self):
        f = self.load("beeler.json")
        sol = scipy.integrate.solve_ivp(f, (0, 500), f.get_u0(), args=f.get_p(), method="BDF", max_step=0.5)
        self.assertTrue(sol.success)
        v = sol.y[6]  # membrane potential
        # a stimulated cardiac action potential: depolarizes above 0 mV and recovers below -80 mV
        self.assertGreater(v.max(), 0.0)
        self.assertLess(v[-1], -80.0)


# ------------------------------------------------------------------ save and load
class SaveLoad(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.d.name, "f.sjb")

    def tearDown(self):
        self.d.cleanup()

    def roundtrip(self, f, eqs=[]):
        f.save(self.path)
        return load_func(self.path, eqs=eqs)

    def test_real(self):
        f = compile_func([x, y], [sin(x) * y, x / y, Piecewise((x, x > y), (y * y, True))])
        g = self.roundtrip(f)
        X, Y = RNG.normal(size=101), RNG.normal(size=101)
        self.assertEqual(f(0.3, 1.7), g(0.3, 1.7))
        for u, v in zip(f(X, Y), g(X, Y)):
            np.testing.assert_array_equal(u, v)

    def test_output_format_from_eqs(self):
        g = self.roundtrip(compile_func([x, y], [x + y, x * y]), eqs=())
        self.assertEqual(g(1.0, 2.0), (3.0, 2.0))

    def test_params(self):
        g = self.roundtrip(compile_func([x], [a * x], params=[a]))
        self.assertEqual(g(3.0, 2.0), [6.0])

    def test_complex(self):
        f = compile_func([x, y], [x * y + sp.exp(x)], dtype="complex128")
        g = self.roundtrip(f)
        self.assertEqual(f(1 + 1j, 2 - 1j), g(1 + 1j, 2 - 1j))

    def test_options_are_kept(self):
        for kw in [dict(opt_level=0), dict(opt_level=3, compress=True), dict(use_simd=False), dict(ty="bytecode")]:
            with self.subTest(**kw):
                f = compile_func([x, y], [exp(x) * cos(y)], **kw)
                g = self.roundtrip(f)
                X = RNG.normal(size=19)
                np.testing.assert_array_equal(f(X, X)[0], g(X, X)[0])

    def test_defuns(self):
        F = Function("F")
        defuns = {F: lambda u: 3 * u}
        f = compile_func([x], [F(x) + 1], defuns=defuns)
        f.save(self.path)
        g = load_func(self.path, defuns=defuns)
        self.assertEqual(g(2.0), [7.0])

    def test_bad_files(self):
        with open(self.path, "wb") as fd:
            fd.write(b"garbage" * 10)
        with self.assertRaises(ValueError):
            load_func(self.path)
        compile_func([x], [x]).save(self.path)
        with open(self.path, "rb") as fd:
            head = fd.read()[:50]
        with open(self.path, "wb") as fd:
            fd.write(head)
        with self.assertRaises(ValueError):
            load_func(self.path)
        with self.assertRaises(ValueError):
            load_func(os.path.join(self.d.name, "missing.sjb"))


# ------------------------------------------------------------------ lockstep options
class Lockstep(unittest.TestCase):
    # `compile_evaluator(..., yield_every=, lockstep=)`: yield points in the kernels, and batch
    # calls that run `lockstep` points in lockstep (cargo feature `async`; other builds keep the
    # options but ignore them). The results must be bit-identical to plain batch calls.
    PATH = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "symbolica", "1loop_instructions_2.txt")

    def setUp(self):
        if not os.path.exists(self.PATH):
            self.skipTest("examples/symbolica/1loop_instructions_2.txt not found")
        with open(self.PATH, encoding="utf-8") as fd:
            self.ev = fd.read()
        rng = np.random.default_rng(5)
        # 37 points: SIMD rows of 4 or 8 points and a scalar tail
        self.X = rng.random((37, 223)) + 1j * rng.random((37, 223))

    def compile(self, **kw):
        from symjit import compile_evaluator

        return compile_evaluator(self.ev, dtype="complex128", num_params=223, use_threads=False, **kw)

    def test_results_are_identical(self):
        # Lockstep runs the same kernel, so it gives the bits of sequential calls of it. A yield
        # point is a statement boundary, which (with fastmath) can keep a multiplication and an
        # addition from fusing into an FMA, so kernels with yield points may differ from kernels
        # without them by an ulp; with fastmath=False the scalar kernel is bit-identical (the
        # complex SIMD kernels still differ by up to 2 ulp there).
        for simd in [dict(use_simd=False), dict(use_simd=True), dict(use_simd=True, enable_simd512=True)]:
            plain = self.compile(**simd).evaluate_complex(self.X)
            for opts in [dict(yield_every=50, lockstep=16), dict(yield_every=1000, lockstep=5), dict(lockstep=8)]:
                with self.subTest(**simd, **opts):
                    f = self.compile(**simd, **opts)
                    self.assertEqual(f.measure("yield-every"), opts.get("yield_every", 0))
                    self.assertEqual(f.measure("lockstep"), opts["lockstep"])
                    sequential = self.compile(**simd, yield_every=opts.get("yield_every", 0))
                    got = f.evaluate_complex(self.X)
                    np.testing.assert_array_equal(got, sequential.evaluate_complex(self.X))
                    np.testing.assert_allclose(got, plain, rtol=1e-14)
                    if not simd["use_simd"]:
                        exact = self.compile(**simd, **opts, fastmath=False).evaluate_complex(self.X)
                        want = self.compile(**simd, fastmath=False).evaluate_complex(self.X)
                        np.testing.assert_array_equal(exact, want)

    def test_yield_points_are_inserted(self):
        f = self.compile(use_simd=False, yield_every=50)
        if f.measure("async") == 0:
            self.skipTest("yield points need the cargo feature `async`")
        plain = self.compile(use_simd=False)
        self.assertGreater(f.measure("mir-size"), plain.measure("mir-size"))

    def test_options_are_saved(self):
        f = self.compile(use_simd=False, yield_every=500, lockstep=32)
        want = f.evaluate_complex(self.X)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ev.sjb")
            f.save(path)
            g = load_func(path)
        self.assertEqual(g.measure("yield-every"), 500)
        self.assertEqual(g.measure("lockstep"), 32)
        np.testing.assert_array_equal(g.evaluate_complex(self.X), want)

    def test_invalid_values(self):
        for opts in [dict(lockstep=-1), dict(yield_every=1.5), dict(lockstep=True), dict(yield_every="10")]:
            with self.subTest(**opts), self.assertRaises(ValueError):
                self.compile(use_simd=False, **opts)


# ------------------------------------------------------------------ scipy callables
@unittest.skipUnless(HAVE_SCIPY, "needs scipy")
class Callables(unittest.TestCase):
    def test_quad(self):
        f = compile_func([x], exp(-(x**2)))
        for use_fast in [True, False]:
            with self.subTest(use_fast=use_fast):
                v, _ = scipy.integrate.quad(f.callable_quad(use_fast=use_fast), -10, 10)
                self.assertAlmostEqual(v, math.sqrt(math.pi), places=12)

    def test_quad_extra_args(self):
        # quad passes its `args` after x
        f = compile_func([x, a], exp(-a * x**2))
        v, _ = scipy.integrate.quad(f.callable_quad(), -10, 10, args=(2.0,))
        self.assertAlmostEqual(v, math.sqrt(math.pi / 2), places=12)

    def test_quad_ahmed_integral(self):
        f = compile_func([x], sp.atan(sqrt(2 + x**2)) / ((1 + x**2) * sqrt(2 + x**2)))
        v, _ = scipy.integrate.quad(f.callable_quad(), 0, 1)
        self.assertAlmostEqual(v, 5 * math.pi**2 / 96, places=13)

    def test_nquad(self):
        f = compile_func([x, y], exp(-(x**2) - y**2))
        v, _ = scipy.integrate.nquad(f.callable_quad(), [[-8, 8], [-8, 8]])
        self.assertAlmostEqual(v, math.pi, places=8)

    def test_generic_filter(self):
        X = symbols("X[0:9]")
        f = compile_func(list(X), Max(*X) - Min(*X))
        img = RNG.random((40, 50))
        got = scipy.ndimage.generic_filter(img, f.callable_filter(), [3, 3])
        want = scipy.ndimage.generic_filter(img, lambda v: v.max() - v.min(), [3, 3])
        np.testing.assert_array_equal(got, want)

    def test_generic_filter_weighted(self):
        X = symbols("X[0:5]")
        w = [1, -2, 3, -4, 5]
        f = compile_func(list(X), sum(wi * xi for wi, xi in zip(w, X)))
        sig = RNG.normal(size=200)
        got = scipy.ndimage.generic_filter(sig, f.callable_filter(), 5, mode="reflect")
        want = scipy.ndimage.generic_filter(sig, lambda v: np.dot(w, v), 5, mode="reflect")
        np.testing.assert_allclose(got, want, rtol=1e-14, atol=1e-14)


# ------------------------------------------------------------------ user functions
class Defuns(unittest.TestCase):
    def test_python_functions(self):
        F, G = Function("F"), Function("G")
        f = compile_func([x], F(x) + G(x, 2 * x), defuns={F: lambda u: u * u, G: lambda u, v: u - v})
        self.assertEqual(f(3.0), 6.0)
        np.testing.assert_array_equal(f(np.array([1.0, 2.0])), [0.0, 2.0])

    def test_string_keys(self):
        F = Function("F")
        self.assertEqual(compile_func([x], F(x), defuns={"F": lambda u: u + 1})(1.0), 2.0)

    def test_symjit_function(self):
        H = Function("H")
        inner = compile_func([x], sin(x) * 2)
        f = compile_func([x], H(x) + 1, defuns={H: inner})
        self.assertEqual(f(0.5), 2 * math.sin(0.5) + 1)

    def test_too_many_arguments(self):
        F = Function("F")
        with self.assertRaisesRegex(ValueError, "1 or 2 arguments"):
            compile_func([x], F(x, x, x), defuns={F: lambda p, q, r: p})

    def test_undefined_function(self):
        with self.assertRaisesRegex(ValueError, "op_code F is not found"):
            compile_func([x], Function("F")(x))

    @unittest.expectedFailure
    def test_recursion_with_function_key(self):
        # BUG: a `None` entry (recursion) must be keyed by a string; a sympy Function key,
        # as for the other entries, raises AttributeError ('R' has no attribute 'encode')
        R = Function("R")
        compile_func([n, x], [n + R(n - 1, x) * 0], defuns={R: None})

    @unittest.expectedFailure
    def test_recursion_terminates(self):
        # BUG: Piecewise compiles to a branchless select that evaluates both branches, so a
        # recursive call in one of them never terminates: even fib(1) overflows the stack
        # (segfault). With a single state, building the fast kernel also panics
        # ("label @self not found").
        code, out = run_isolated(
            """
            from sympy import Function, Piecewise, symbols
            from symjit import compile_func
            n = symbols("n"); R = Function("R")
            fib = compile_func([n], [Piecewise((n, n < 2), (R(n - 1) + R(n - 2), True)), n], defuns={"R": None})
            print(fib(1.0)[0], fib(20.0)[0])
            """
        )
        self.assertEqual((code, out.split()[-2:]), (0, ["1.0", "6765.0"]), out[-500:])


# ------------------------------------------------------------------ input checking
class Errors(unittest.TestCase):
    def test_bad_options(self):
        with self.assertRaisesRegex(ValueError, "Config error"):
            compile_func([x], x, ty="nonsense")
        with self.assertRaisesRegex(ValueError, "dtype"):
            compile_func([x], x, dtype="float32")

    def test_unsupported_function(self):
        with self.assertRaisesRegex(ValueError, "besselj is not found"):
            compile_func([x], sp.besselj(0, x))

    def test_unknown_variable(self):
        with self.assertRaisesRegex(ValueError, "variable y not found"):
            compile_func([x], x + y)

    def test_complex_constant_message(self):
        # complex constants are not supported by the sympy front end; the error should at
        # least be a Python exception, not a crash
        with self.assertRaises((TypeError, ValueError)):
            compile_func([x], sp.I * x, dtype="complex128")


class Limits(unittest.TestCase):
    def test_nesting_depth_60(self):
        e = x
        for _ in range(60):
            e = sin(e)
        v = 0.5
        for _ in range(60):
            v = math.sin(v)
        self.assertAlmostEqual(compile_func([x], [e])(0.5)[0], v, places=14)

    @unittest.expectedFailure
    def test_deep_nesting(self):
        # LIMITATION: expressions nested more than about 64 levels deep (sin(sin(...)) of
        # depth 65, a polynomial of degree 31 in Horner form) fail with "Cannot parse JSON:
        # recursion limit exceeded": serde_json's default limit of 128 levels when the model
        # is parsed (rust/model.rs), two levels per node
        e = sp.horner(sum((i + 1) * x**i for i in range(40)))
        self.assertEqual(compile_func([x], [e])(1.0)[0], sum(range(1, 41)))


# ------------------------------------------------------------------ Symbolica bridge
@unittest.skipUnless(HAVE_SYMBOLICA, "needs symbolica")
class Evaluators(unittest.TestCase):
    # symbolica is imported lazily: its unlicensed version aborts a second process that
    # imports it concurrently (see test_obj)
    def setUp(self):
        from symbolica import E, Expression, S

        self.E, self.S, self.Expression = E, S, Expression
        self.vars = [S("x"), S("y")]

    def test_real(self):
        from symjit import compile_evaluator

        ev = self.E("x + y^2 + sin(x)*y + exp(-x*y)").evaluator(self.vars)
        X = RNG.uniform(-2, 2, (1003, 2))
        want = np.asarray(ev.evaluate(X))
        for kw in [dict(), dict(use_simd=False), dict(opt_level=0), dict(enable_simd512=True), dict(compress=True)]:
            with self.subTest(**kw):
                np.testing.assert_allclose(compile_evaluator(ev, **kw).evaluate(X), want, rtol=1e-14)

    def test_complex(self):
        from symjit import compile_evaluator

        ev = self.E("x*y + x^3 - y/x").evaluator(self.vars)
        Z = RNG.normal(size=(50, 2)) + 1j * RNG.normal(size=(50, 2))
        want = np.asarray(ev.evaluate_complex(Z))
        for kw in [dict(), dict(fast_complex=False)]:
            with self.subTest(**kw):
                f = compile_evaluator(ev, dtype="complex128", **kw)
                np.testing.assert_allclose(f.evaluate_complex(Z), want, rtol=1e-14)

    def test_multiple_outputs(self):
        from symjit import compile_evaluator

        ev = self.Expression.evaluator_multiple([self.E("x+y"), self.E("x*y"), self.E("exp(x)/y")], self.vars)
        X = RNG.uniform(0.5, 2, (17, 2))
        np.testing.assert_allclose(compile_evaluator(ev).evaluate(X), ev.evaluate(X), rtol=1e-15)

    def test_save_load_explicit_dtype(self):
        from symjit import compile_evaluator

        ev = self.E("x + y^2").evaluator(self.vars)
        f = compile_evaluator(ev)
        X = RNG.normal(size=(5, 2))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ev.sjb")
            f.save(path, dtype="float64")
            np.testing.assert_array_equal(load_func(path).evaluate(X), f.evaluate(X))

    @unittest.expectedFailure
    def test_save_load_default(self):
        # BUG: SymbolicaFunc.save defaults to dtype="complex128", so saving a real evaluator
        # writes (and first compiles) its complex kernel; the loaded function then has no
        # real kernel and `evaluate` fails ("Cannot parse JSON")
        from symjit import compile_evaluator

        ev = self.E("x + y^2").evaluator(self.vars)
        f = compile_evaluator(ev)
        X = RNG.normal(size=(5, 2))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ev.sjb")
            f.save(path)
            np.testing.assert_array_equal(load_func(path).evaluate(X), f.evaluate(X))


if __name__ == "__main__":
    unittest.main()

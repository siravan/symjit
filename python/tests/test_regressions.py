"""Regression tests for bugs found by fuzzing and review.

Each test compiles a minimal expression that used to give wrong results (or abort the
process) and compares it with numpy.
"""

import os
import sys
import unittest
import warnings

import numpy as np
import sympy as sp
from sympy import And, Max, Min, Or, Piecewise, Rational, cos, exp, log, sin, sqrt, symbols

sys.path.insert(0, os.environ.get("SYMJIT_PYTHON_PATH", os.path.join(os.path.dirname(__file__), "..")))

from symjit import Composer, compile_composer, compile_func

x, y, z = symbols("x y z")

R = np.random.default_rng(3)
XR = R.uniform(-2, 2, (3, 33))
XC = R.uniform(-1.5, 1.5, (3, 33)) + 1j * R.uniform(-1.5, 1.5, (3, 33))
# the last points lie in every quadrant of the complex plane (branch cuts)
ZC = np.array([0.5 + 0.3j, -1.2 + 0.4j, 2 + 0j, -2 + 0j, 0.1 - 0.9j, -0.5 - 0.5j, -0.3 - 2j, 1e-3 + 1j])


def reference(exprs, X):
    """See test_fuzz.reference: falls back to pointwise evaluation for an expression whose
    vectorized numpy code raises a ragged-array ValueError (some sympy/numpy version
    combinations, e.g. on macOS/arm64, hit this for expressions such as `Min(0, -z)`; not
    reproducible with the versions pinned here)."""
    n = X.shape[1]
    rows = []
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for e in exprs:
            fi = sp.lambdify([x, y, z], e, "numpy")
            try:
                rows.append(np.broadcast_to(np.asarray(fi(*X)), (n,)))
            except ValueError:
                rows.append(np.array([fi(*pt) for pt in X.T]))
        return np.array(rows)


OPTS = [dict(opt_level=0), dict(opt_level=1), dict(opt_level=2), dict(opt_level=3), dict(opt_level=2, compact=False)]


def recursive_composer(kind):
    """factorial or fibonacci with a recursive call (`call(None, ...)`) in a block"""
    cp = Composer(1, 1)
    n, one, two = cp.arg(0), cp.constant(1.0), cp.constant(2.0)
    cond = cp.leq(n, one) if kind == "factorial" else cp.lt(n, two)
    base, rec = cp.new_block(), cp.new_block()
    r1 = base.fadd(one, cp.constant(0.0)) if kind == "factorial" else base.fadd(n, cp.constant(0.0))
    if kind == "factorial":
        r2 = rec.fmul(n, rec.call(None, rec.fsub(n, one)))
    else:
        r2 = rec.fadd(rec.call(None, rec.fsub(n, one)), rec.call(None, rec.fsub(n, two)))
    cp.append_if_else(cond, base, rec)
    cp.assign(cp.out(0), cp.join(cond, r1, r2))
    return cp


class Regressions(unittest.TestCase):
    def check(self, exprs, X, options=None, complex_=False, tol=1e-9):
        exprs = list(exprs)
        ref = reference(exprs, X)
        for o in options or OPTS:
            kw = dict(o)
            if complex_:
                kw["dtype"] = "complex128"
            f = compile_func([x, y, z], exprs, **kw)
            n = X.shape[1]
            vec = np.array(f(*X)).reshape(len(exprs), n)
            sc = np.array([np.ravel(f(*X[:, i])) for i in range(n)]).T
            for kind, got in (("vectorized", vec), ("scalar", sc)):
                with self.subTest(options=o, kind=kind, exprs=str(exprs)):
                    np.testing.assert_allclose(got, ref, rtol=tol, atol=tol)

    def test_shared_call_result(self):
        # fuse_save3 dropped the stack save of a call result that was used again;
        # the compactor panicked ("cannot find Stack[48]") and aborted the process
        self.check([sin(y), sin(y) * x], XR)
        self.check([sin(y), sin(y) + cos(y) * sin(y)], XR)
        self.check([exp(x), exp(x) / (1 + y * y)], XR)
        self.check([sin(y), z, sin(y) * z], XR)
        self.check([sin(y), Piecewise((z, sin(y) < x), (x, True))], XR)

    def test_fma_load_minus_with_temp(self):
        # fuse_fma3 fused a Load+Minus while a Times operand lived in the Temp register
        self.check([-(z - y) ** 3 + log(1 + z * z)], XR)
        self.check([-(-(y**3) - y + z) ** 3 + log(Min(0, -z) ** 2 + 1)], XR)
        self.check([-(z - y) ** 3 + Min(0, -z)], XR)

    def test_complex_neg_of_temp(self):
        # neg/abs/sign used the Temp register as scratch, so a Temp operand became zero
        self.check([1.135 - y**3], XC, complex_=True)
        self.check([2.0 - y**2, x - y**3], XC, complex_=True)
        self.check([cos(1 / z)], XC, complex_=True)
        self.check([sin(1 / (z**2 + 1))], XC, complex_=True)
        self.check([cos(1 / z)], XC, complex_=True, options=[dict(opt_level=2, fast_complex=False)])

    def test_complex_fractional_power(self):
        # x**1.5 was rewritten as sqrt(x**3), which is the wrong branch for complex x
        Z = np.tile(ZC, 5)[:33]
        X = np.stack([Z, Z, Z])
        for e in (z ** Rational(3, 2), z ** Rational(5, 2), z ** Rational(-3, 2), z ** Rational(1, 3), sqrt(z)):
            self.check([e], X, complex_=True, options=[dict(opt_level=0), dict(opt_level=2), dict(opt_level=2, fast_complex=False)])

    def test_complex_sech_csch_sign(self):
        # the libm build computed sech/csch as sqrt(1 - tanh**2), which has the wrong sign
        # wherever Re(cosh(z)) < 0 (|Im z| > pi/2), e.g. sech(3j) = -1.0101
        Z = np.array([0.5 + 0.5j, 3j, 0.5 + 2.5j, -1 + 3j, 2 - 4j, -0.2 - 2j] * 6)[:33]
        X = np.stack([Z, Z, Z])
        self.check([sp.sech(z), sp.csch(z)], X, complex_=True, options=[dict(opt_level=2)])

    def test_complex_reciprocal_functions(self):
        # csc, sec, cot, csch, sech and coth computed 1/w as conj(w)/|w|^2, which
        # overflows for |w| > 1.3e154: 0 instead of tiny values (e.g. sech(700) = 2e-304),
        # and NaN+NaNj when the function itself overflows (sech(800+0.5j) = 0); a partial
        # fix in the libm build returned 0+NaNj there and turned NaN arguments into 0+NaNj.
        # Also tanh and tan, which were NaN for |Re z| (|Im z|) > ~355 without libm.
        import cmath
        import mpmath

        mpmath.mp.dps = 40
        funcs = [(sp.csc, mpmath.csc), (sp.sec, mpmath.sec), (sp.cot, mpmath.cot), (sp.tan, mpmath.tan),
                 (sp.csch, mpmath.csch), (sp.sech, mpmath.sech), (sp.coth, mpmath.coth), (sp.tanh, mpmath.tanh)]
        f = compile_func([z], [g(z) for g, _ in funcs], dtype="complex128")

        hyperbolic = [0.5 + 0.5j, 3j, 0.5 + 2.5j, -1 + 3j, 2 - 4j, 400 + 0.25j, 700, -700 + 1j, 800, -800 + 2j,
                      800 - 3j, 0.25 + 800j]
        for w in hyperbolic:
            # the trigonometric functions overflow along the imaginary axis instead
            for k, point in enumerate([1j * w] * 4 + [w] * 4):
                with self.subTest(func=funcs[k][0].__name__, z=point):
                    got = complex(f(point)[k])
                    want = complex(funcs[k][1](mpmath.mpc(point.real, point.imag)))
                    self.assertTrue(cmath.isfinite(got), got)
                    self.assertLessEqual(abs(got - want), 1e-13 * abs(want) + 1e-320, (got, want))

        inf, nan = float("inf"), float("nan")
        for point in (complex(inf, 0), complex(inf, 1), complex(-inf, -2)):
            got = [complex(v) for v in f(point)][4:]  # csch, sech, coth, tanh
            self.assertEqual(got[:2], [0, 0], point)
            self.assertEqual(got[3].real, 1 if point.real > 0 else -1, point)
        for point in (complex(nan, 0), complex(1, nan)):
            for v in f(point):
                self.assertTrue(cmath.isnan(v), (point, v))  # NaN propagates

    def test_bytecode_complex_sqrt(self):
        # the bytecode interpreter did not implement subroutine calls (complexify's
        # `@complex_root`) and panicked, aborting the process; Python calls of
        # `ty="wasm"` functions run on it
        Z = np.tile(ZC, 5)[:33]
        X = np.stack([Z, Z, Z])
        ref = reference([sqrt(z * y + 1)], X)
        for ty in ("bytecode", "wasm"):
            try:
                f = compile_func([x, y, z], [sqrt(z * y + 1)], dtype="complex128", ty=ty)
            except ValueError as e:
                if ty == "wasm" and "without the `wasm` feature" in str(e):
                    continue  # the library was built without the wasm backend
                raise
            got = np.array([np.ravel(f(*X[:, i])) for i in range(X.shape[1])]).T
            np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-9, err_msg=ty)

    def test_bytecode_array_calls(self):
        # calls with arrays of a function without machine code (ty="bytecode", "wasm")
        # silently returned zeros
        f = compile_func([x, y, z], [x * y + z, sin(x) - z], ty="bytecode")
        got = np.array(f(*XR))
        np.testing.assert_allclose(got, reference([x * y + z, sin(x) - z], XR), rtol=1e-12, atol=1e-12)

    def test_composer_recursion(self):
        # a recursive call passed its argument in a register while the callee (a whole
        # kernel) reads the `__Arg` slots, so factorial(n) gave n; and a call inside a
        # block was registered on the block, which is not compiled
        want = {"factorial": [1, 1, 2, 120, 3628800], "fibonacci": [0, 1, 1, 55, 6765]}
        ks = {"factorial": [0, 1, 2, 5, 10], "fibonacci": [0, 1, 2, 10, 20]}
        for kind in ("factorial", "fibonacci"):
            for options in (dict(), dict(direct=False), dict(opt_level=0)):
                with self.subTest(kind=kind, options=options):
                    f = compile_composer(recursive_composer(kind), **options)
                    self.assertEqual([float(f(float(k))[0][0]) for k in ks[kind]], want[kind])

    def test_composer_callback_in_block(self):
        cp = Composer(1, 1)
        x = cp.arg(0)
        cond = cp.lt(x, cp.constant(0.0))
        neg, pos = cp.new_block(), cp.new_block()
        r1 = neg.call(lambda a: -a, x)
        r2 = pos.fadd(x, cp.constant(0.0))
        cp.append_if_else(cond, neg, pos)
        cp.assign(cp.out(0), cp.join(cond, r1, r2))
        f = compile_composer(cp)
        self.assertEqual([float(f(v)[0][0]) for v in (-3.0, 4.0)], [3.0, 4.0])

    def test_piecewise_many_branches(self):
        # Piecewise with three or more branches raised an AttributeError in structure.py
        self.check([Piecewise((1, x < 0), (2, x < 1), (3, True))], XR)
        self.check([Piecewise((x, x < y), (y, y < z), (z, True))], XR)
        self.check([Piecewise((sin(x), x < -1), (x * y, x < 0), (z, x < 1), (y - z, True))], XR)

    @unittest.expectedFailure
    def test_nary_boolean(self):
        # KNOWN BUG (open): sympy flattens nested And/Or into one operation with more than two
        # operands, and compilation fails with `missing poly op: and` / `missing poly op: or`
        a, b, c = x < y, y < z, z < 0.5
        self.check([Piecewise((x, Or(a, b, c)), (z, True))], XR, options=[dict(opt_level=2)])
        self.check([Piecewise((x, And(a, b, c)), (z, True))], XR, options=[dict(opt_level=2)])

    @unittest.expectedFailure
    def test_piecewise_without_final_true(self):
        # KNOWN BUG (open): a Piecewise whose last condition is not `True` is undefined (nan)
        # outside its conditions in sympy, but symjit treats the last branch as the else branch
        X = np.array([[-1.0, 0.5, 5.0]] * 3)
        self.check([Piecewise((1, x < 0), (2, x < 1))], X, options=[dict(opt_level=2)])

    def test_cse_of_nested_subexpressions(self):
        # Block::eliminate assigned a common subexpression without rewriting the shared
        # subexpressions nested in it, and ran two passes: each pass shared one more level,
        # so the code of deeply nested expressions (examples/stress.py) grew exponentially
        # (128363 MIR instructions at depth 13, now 336). Complex code lost sharing too, because
        # Block::trim split the expressions into temporaries (complex mode has fewer scratch
        # registers) before the elimination ran; trim now runs after it (573 -> 295 at depth 10).
        e = x**2 + x
        for _ in range(10):
            e = e**2 + e
        ed = e.diff(x)
        want = float(ed.subs(x, sp.Rational(1, 10000)))
        for dtype, limit in [("float64", 400), ("complex128", 400)]:
            f = compile_func([x], [ed], dtype=dtype)
            self.assertLess(f.measure("mir-size"), limit, dtype)
            self.assertAlmostEqual(complex(f(1e-4)[0]).real, want, places=12)

    def test_compressed_stack_slots_are_reused(self):
        # Compactor (cargo feature `experimental`): the stack slots passed to the funclets of
        # compression mode (LoadArgs) were never returned to the pool, so the frame of a
        # compressed evaluator was up to four times larger than the uncompressed one (GL297:
        # 25600 vs 6750 slots; this 1-loop evaluator: 1134 vs 646). A slot passed twice
        # (x * x) is released once.
        from symjit import compile_evaluator

        if compile_func([x], [x]).measure("experimental") == 0:
            self.skipTest("the compactor fix is in the `experimental` cargo feature")
        path = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "symbolica", "1loop_instructions_2.txt")
        if not os.path.exists(path):
            self.skipTest("examples/symbolica/1loop_instructions_2.txt not found")
        with open(path, encoding="utf-8") as fd:
            ev = fd.read()
        rng = np.random.default_rng(1)
        X = rng.random((3, 223)) + 1j * rng.random((3, 223))
        opts = dict(dtype="complex128", num_params=223, fast_complex=True, use_simd=False, use_threads=False)
        plain = compile_evaluator(ev, compress=False, **opts)
        want = plain.evaluate_complex(X)
        for level in [2, 3]:
            f = compile_evaluator(ev, compress=True, opt_level=level, **opts)
            self.assertLessEqual(f.measure("stack-size"), plain.measure("stack-size") * 1.1, level)
            np.testing.assert_allclose(f.evaluate_complex(X), want, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()

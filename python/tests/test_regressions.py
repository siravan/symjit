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

from symjit import compile_func

x, y, z = symbols("x y z")

R = np.random.default_rng(3)
XR = R.uniform(-2, 2, (3, 33))
XC = R.uniform(-1.5, 1.5, (3, 33)) + 1j * R.uniform(-1.5, 1.5, (3, 33))
# the last points lie in every quadrant of the complex plane (branch cuts)
ZC = np.array([0.5 + 0.3j, -1.2 + 0.4j, 2 + 0j, -2 + 0j, 0.1 - 0.9j, -0.5 - 0.5j, -0.3 - 2j, 1e-3 + 1j])


def reference(exprs, X):
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n = X.shape[1]
        return np.array([np.broadcast_to(np.asarray(r), (n,)) for r in sp.lambdify([x, y, z], exprs, "numpy")(*X)])


OPTS = [dict(opt_level=0), dict(opt_level=1), dict(opt_level=2), dict(opt_level=3), dict(opt_level=2, compact=False)]


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


if __name__ == "__main__":
    unittest.main()

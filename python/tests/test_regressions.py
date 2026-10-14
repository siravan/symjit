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

    def test_composer_complex_divide_by_constant(self):
        # x86 fast complex: a constant divisor is loaded into Reg::Temp, which `divide` used as
        # scratch, so z / 4 gave (0.5-0.5j) for z = 0.5 with the indirect translator (which
        # turns `fdiv`, z * recip(4), into a division; found by the Rust kernel builder)
        cp = Composer(1, 2)
        z = cp.arg(0)
        cp.assign(cp.out(0), cp.fdiv(z, cp.constant(4.0)))
        cp.assign(cp.out(1), cp.fdiv(z, cp.constant(4.0 + 1.0j)))
        for options in (dict(), dict(direct=False), dict(direct=False, fast_complex=False)):
            with self.subTest(options=options):
                f = compile_composer(cp, dtype="complex128", **options)
                for v in (0.5, 0.5 - 1.5j):
                    got = f.evaluate_complex([[v]])[0]
                    self.assertAlmostEqual(complex(got[0]), v / 4, delta=1e-15)
                    self.assertAlmostEqual(complex(got[1]), v / (4 + 1j), delta=1e-15)

    def test_composer_complex_recursion(self):
        # `Composer.call` marked every call `is_real`, so with the indirect translator a
        # recursive call took the real parts of its arguments: f(a, b) below, which recurses
        # once and returns b, gave 3 instead of 3+4j
        cp = Composer(2, 1)
        a, b = cp.arg(0), cp.arg(1)
        stop = cp.lt(cp.real(a), cp.constant(0.5))
        base, rec = cp.new_block(), cp.new_block()
        r = rec.call(None, rec.complex(cp.constant(0.0), cp.imag(a)), b)
        cp.append_if_else(stop, base, rec)
        cp.assign(cp.out(0), cp.join(stop, b, r))
        for options in (dict(), dict(direct=False), dict(direct=False, fast_complex=False)):
            with self.subTest(options=options):
                f = compile_composer(cp, dtype="complex128", **options)
                self.assertEqual(complex(f.evaluate_complex([[1 + 2j, 3 + 4j]])[0, 0]), 3 + 4j)

    def test_compressed_funclets_with_many_arguments(self):
        # Compression mode passes the first arguments of a funclet in registers, or (opt_level
        # 3, "ultra" entry) packs their stack slot numbers into general registers, 4 per
        # register. The real AVX-512 SIMD kernel asked for 32 such arguments, but there were 4
        # registers (16 arguments): a funclet with more than 16 arguments aborted the process
        # (index out of bounds), at any opt_level. There are now 8 registers on Unix (6 on
        # Windows). With AVX-512 at opt_level 3, the complex scalar kernel passes arguments 0-15
        # in xmm16-xmm31 and 16-31 in xmm0-xmm15, and a funclet body without calls reads arguments 0-15 from
        # xmm16-xmm31 instead of saving them. The results must equal the uncompressed ones.
        from symjit import compile_evaluator

        def funclets(n_terms, temps):
            # 4 outputs, each a sum of `n_terms` products of two parameters, or of temps that
            # copy the parameters (stack slots, the packed entry): one funclet with
            # 2 * n_terms arguments (real) or n_terms arguments (complex, split by trim)
            ir, t = [], 0
            if temps:
                ir = [("add", ("temp", i), [("param", i), ("const", 0)], 0) for i in range(64)]
                t = 64
            leaf = ("temp" if temps else "param", None)
            sums = []
            for k in range(4):
                prods = []
                for i in range(n_terms):
                    a, b = (2 * i + k) % 64, (2 * i + 3 * k + 1) % 64
                    ir.append(("mul", ("temp", t), [(leaf[0], a), (leaf[0], b)], 0))
                    prods.append(t)
                    t += 1
                ir.append(("add", ("temp", t), [("temp", j) for j in prods], 0))
                sums.append(t)
                t += 1
            ir += [("mul", ("out", k), [("const", 0), ("temp", s)], 0) for k, s in enumerate(sums)]
            return str((ir, t, [1]))

        rng = np.random.default_rng(5)
        X = rng.uniform(0.5, 1.5, (37, 64))
        cases = [(n, temps, dt) for n in [8, 12, 16, 20] for temps in [False, True] for dt in ["float64", "complex128"]]
        cases += [(n, True, "complex128") for n in [24, 30, 40]]
        for n_terms, temps, dtype in cases:
            ev = funclets(n_terms, temps)
            base = dict(dtype=dtype, num_params=64, use_threads=False)
            if dtype == "complex128":
                base["fast_complex"] = True
                args = X + 0.5j * X[::-1]
            else:
                args = X

            def run(**kw):
                f = compile_evaluator(ev, **base, **kw)
                if dtype == "complex128":
                    return np.vstack([f.evaluate_complex(args[:1]), f.evaluate_complex(args)])
                return np.vstack([f.evaluate(args[:1]), f.evaluate(args)])

            want = run(compress=False, use_simd=False)
            for level in [2, 3]:
                for simd, simd512 in [(False, False), (True, False), (True, True)]:
                    with self.subTest(n_terms=n_terms, temps=temps, dtype=dtype, level=level, simd=simd, simd512=simd512):
                        got = run(compress=True, opt_level=level, use_simd=simd, enable_simd512=simd512)
                        np.testing.assert_allclose(got, want, rtol=1e-13)

    def test_real_kernel_times_imaginary_unit(self):
        # The builder rewrote `x * i` (and `x * -i`) as `times_i(x)` in real kernels too, where
        # the native generators emit nothing for it: the result was whatever the register held
        # (often `x`, or a leftover value), so real Symbolica evaluators with imaginary
        # constants were wrong, more so in compression mode (GL297: 1e3 to 1e63 relative
        # error). A real kernel holds the real part of `i * x`, which is 0.
        from symjit import Composer, compile_composer, compile_evaluator

        rng = np.random.default_rng(9)
        X = rng.uniform(0.5, 1.5, (37, 8))
        # constants: 0: 1, 1: i, 2: -i
        ir = [
            ("mul", ("temp", 0), [("param", 0), ("const", 1)], 0),  # x0 * i -> 0
            ("mul", ("temp", 1), [("const", 2), ("param", 1)], 0),  # -i * x1 -> 0
            ("add", ("temp", 2), [("param", 2), ("temp", 0)], 0),  # x2
            ("mul", ("temp", 3), [("temp", 2), ("param", 3), ("const", 1)], 0),  # 0
            ("add", ("out", 0), [("temp", 1), ("param", 4)], 0),  # x4
            ("add", ("out", 1), [("temp", 2), ("temp", 3), ("const", 0)], 0),  # x2 + 1
        ]
        # the same shape 4 times with different parameters: a funclet in compression mode
        for k in range(4):
            ir.append(("mul", ("temp", 4 + k), [("param", k), ("param", k + 4), ("param", (k + 1) % 8), ("const", 1)], 0))
        ir.append(("add", ("out", 2), [("temp", 4 + k) for k in range(4)] + [("param", 5)], 0))  # x5
        ev = f"({ir}, 8, [1, 1𝑖, -1𝑖])"
        want = np.stack([X[:, 4], X[:, 2] + 1, X[:, 5]], axis=1)
        for kw in [dict(compress=False), dict(compress=False, opt_level=0), dict(compress=True, opt_level=2),
                   dict(compress=True, opt_level=3), dict(use_simd=False), dict(enable_simd512=True)]:
            with self.subTest(**kw):
                f = compile_evaluator(ev, dtype="float64", num_params=8, use_threads=False, **kw)
                np.testing.assert_array_equal(np.vstack([f.evaluate(X[:1]), f.evaluate(X)]), np.vstack([want[:1], want]))

        # the indirect translator of the Composer (`direct=False`) goes through the same builder
        cp = Composer(2, 2)
        cp.assign(cp.out(0), cp.fadd(cp.fmul(cp.arg(0), cp.constant(1j)), cp.arg(1)))
        cp.assign(cp.out(1), cp.fadd(cp.fmul(cp.constant(-1j), cp.arg(0)), cp.arg(1)))
        for direct in [True, False]:
            with self.subTest(direct=direct):
                f = compile_composer(cp, direct=direct, use_threads=False)
                np.testing.assert_array_equal(f.evaluate(X[:, :2]), np.stack([X[:, 1], X[:, 1]], axis=1))


if __name__ == "__main__":
    unittest.main()

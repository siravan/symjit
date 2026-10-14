"""symjit's own math library (cargo feature `math`, `rust/libm.rs`).

With `ty="native;inline_math=true"`, the SIMD kernels evaluate the library's functions (`exp`,
the six hyperbolic functions, `sin` and `cos`) inline, on all lanes at once; the scalar kernel (single points,
and the rows after the last full SIMD row) and the bytecode interpreter keep calling the C
library (sin and cos also for |x| >= 2^20 pi/2, NaN and infinities: a SIMD row with such an
argument is computed by the scalar kernel). In complex kernels, the hyperbolic functions have
complex versions (for |re| < 710 and |im| < 2^20 pi/2); exp, sin and cos call the real function when the imaginary part of the argument is zero
(a SIMD row with real and complex arguments is computed by the scalar kernel). The Rust tests
(`libm::tests`) check the SIMD rows bit for bit against the Rust version of the library, also on
ARM64 and RISC-V under qemu.

Skips when the library is built without the feature. Set SYMJIT_PYTHON_PATH to test another
build of the package.
"""

import os
import sys
import unittest

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("SYMJIT_PYTHON_PATH", os.path.join(HERE, "..")))

from symjit import compile_evaluator, compile_func  # noqa: E402
from symjit.disasm import disassembly  # noqa: E402

x = sp.Symbol("x")
INLINE = "native;inline_math=true"


def has_math():
    try:
        compile_func([x], [x], ty=INLINE)
        return True
    except Exception:  # noqa: BLE001
        return False


def ulps(a, b):
    return np.abs(a.view(np.int64) - b.view(np.int64))


def inputs():
    rng = np.random.default_rng(3)
    special = [0.0, -0.0, np.inf, -np.inf, np.nan, 709.78, 709.79, -745.13, -745.14, 1e-300, -1e-300,
               1e300, -1e300, 0.5 * np.log(2), -0.5 * np.log(2)]
    return np.concatenate([special, rng.uniform(-745, 710, 20000), rng.uniform(-1, 1, 5000),
                           rng.normal(0, 10, 5000)])


# name: (sympy function, numpy reference, max ulp against it); the references of coth, sech and
# csch are 1 / tanh, cosh, sinh, an ulp or so off themselves
FUNCTIONS = {
    "exp": (sp.exp, np.exp, 1),
    "sinh": (sp.sinh, np.sinh, 4),
    "cosh": (sp.cosh, np.cosh, 4),
    "tanh": (sp.tanh, np.tanh, 4),
    "coth": (sp.coth, lambda z: 1 / np.tanh(z), 6),
    "sech": (sp.sech, lambda z: 1 / np.cosh(z), 6),
    "csch": (sp.csch, lambda z: 1 / np.sinh(z), 6),
    "sin": (sp.sin, np.sin, 2),
    "cos": (sp.cos, np.cos, 2),
    "ln": (sp.log, np.log, 2),
    "log": (sp.Function("log"), np.log10, 3),  # symjit's log is log10
}

HYPERBOLIC = ["sinh", "cosh", "tanh", "coth", "sech", "csch"]
# the functions with complex versions in the library (complex kernels, complex arguments)
COMPLEX_INLINED = HYPERBOLIC + ["exp", "sin", "cos"]
# the bounds of |re| and |im| of their arguments (`libm::complex_domain`)
COMPLEX_DOMAIN = {"exp": (709, 2**20 * np.pi / 2), "sin": (2**20 * np.pi / 2, 710), "cos": (2**20 * np.pi / 2, 710)}

OPTIONS = [dict(), dict(fastmath=False), dict(opt_level=0), dict(opt_level=1), dict(opt_level=3),
           dict(use_simd=False), dict(enable_simd512=True), dict(enable_simd512=True, fastmath=False),
           dict(cse=False)]


@unittest.skipUnless(has_math(), "needs the `math` feature (cargo build --features math)")
class InlineMath(unittest.TestCase):
    def test_against_numpy(self):
        xs = inputs()
        for name, (f_sp, f_np, bound) in FUNCTIONS.items():
            with np.errstate(all="ignore"):
                ref = f_np(xs)
            # where the references are accurate: normal results (1 / cosh(x) is 0 for large |x|)
            normal = np.isfinite(ref) & (np.abs(ref) >= 2.2250738585072014e-308) & (np.abs(xs) < 700)
            special = np.isnan(xs) | np.isinf(xs) | (xs == 0)
            for options in OPTIONS:
                with self.subTest(name=name, options=options):
                    f = compile_func([x], [f_sp(x)], ty=INLINE, **options)
                    v = np.ravel(f(xs))
                    self.assertLessEqual(ulps(v[normal], ref[normal]).max(), bound)
                    # nan, infinities and signed zeros (also of the results: coth(-0) = -inf)
                    np.testing.assert_array_equal(v[special], ref[special])
                    signed = special & ~np.isnan(ref)  # (the sign of a NaN has no meaning)
                    np.testing.assert_array_equal(np.signbit(v[signed]), np.signbit(ref[signed]))

    def test_scalar_kernel_calls_c(self):
        # single points (scalar kernel) and the interpreter give the C library's bits
        xs = inputs()[:1500]
        for name, (f_sp, _, _) in FUNCTIONS.items():
            for options in [dict(), dict(fastmath=False), dict(enable_simd512=True), dict(use_simd=False)]:
                with self.subTest(name=name, options=options):
                    want = np.ravel([compile_func([x], [f_sp(x)], **options)(t) for t in xs])
                    for ty in [INLINE, "bytecode;inline_math=true"]:
                        f = compile_func([x], [f_sp(x)], ty=ty, **options)
                        got = np.ravel([f(t) for t in xs])
                        same = (got.view(np.uint64) == want.view(np.uint64)) | (np.isnan(got) & np.isnan(want))
                        self.assertTrue(same.all(), f"{ty}: x = {xs[~same][:4]}")

    def test_simd_rows_use_the_library(self):
        # the library differs from the C library by an ulp or so at some points
        xs = inputs()[:4096]
        for name, (f_sp, _, _) in FUNCTIONS.items():
            for options in [dict(), dict(enable_simd512=True)]:
                with self.subTest(name=name, options=options):
                    inline = np.ravel(compile_func([x], [f_sp(x)], ty=INLINE, **options)(xs))
                    c_lib = np.ravel(compile_func([x], [f_sp(x)], **options)(xs))
                    self.assertGreater(np.sum(inline != c_lib), 0)

    def test_trigonometric_domain(self):
        # |x| >= 2^20 pi/2, NaN and infinities: the C functions; SIMD rows with such arguments
        # are computed by the scalar kernel, the other rows by the library
        rng = np.random.default_rng(6)
        xs = rng.uniform(-10, 10, 4096)
        xs[::37] = rng.uniform(1.7e6, 1e300, len(xs[::37]))
        xs[5] = np.inf
        xs[6] = np.nan
        for name, f_np in [("sin", np.sin), ("cos", np.cos)]:
            for options in [dict(), dict(enable_simd512=True)]:
                with self.subTest(name=name, options=options):
                    f = compile_func([x], [getattr(sp, name)(x)], ty=INLINE, **options)
                    got = np.ravel(f(xs))
                    with np.errstate(invalid="ignore"):
                        ref = f_np(xs)
                    np.testing.assert_allclose(got, ref, rtol=1e-15, atol=1e-15)

    def test_subnormal_results(self):
        # sech and csch keep subnormal results, where 1 / cosh(x) and 1 / sinh(x) are 0
        xs = np.array([720.0, -720.0, 740.0, 744.0] * 4)
        for name, want in [("sech", 2 * np.exp(-np.abs(xs))), ("csch", 2 * np.sign(xs) * np.exp(-np.abs(xs)))]:
            f = compile_func([x], [getattr(sp, name)(x)], ty=INLINE)
            got = np.ravel(f(xs))
            np.testing.assert_allclose(got, want, rtol=1e-10, atol=0, err_msg=name)

    def test_in_expressions(self):
        y = sp.Symbol("y")
        xs = np.random.default_rng(4).uniform(-20, 20, (2, 5000))
        e = [sp.exp(x) * sp.exp(-y / 3) + sp.exp(x * y / 50), sp.sin(sp.exp(x / 10)) + y]
        ref = sp.lambdify([x, y], e, "numpy")(*xs)
        for options in [dict(), dict(enable_simd512=True), dict(use_simd=False), dict(opt_level=3)]:
            with self.subTest(options=options):
                f = compile_func([x, y], e, ty=INLINE, **options)
                np.testing.assert_allclose(np.asarray(f(*xs)), ref, rtol=1e-14)

    def test_simd_kernel_is_inlined(self):
        f = compile_func([x], [sp.exp(x)], ty=INLINE)
        simd = disassembly(f, kernel="simd")
        self.assertIn("vpsllq", simd)  # pow2i
        self.assertEqual(sum("call" in line for line in simd.splitlines()), 1)  # the funclet
        plain = disassembly(compile_func([x], [sp.exp(x)]), kernel="simd")
        self.assertGreater(sum("call" in line for line in plain.splitlines()), 1)  # one per lane
        scalar = disassembly(f, kernel="scalar")
        self.assertNotIn("vpsllq", scalar)  # the scalar kernel calls the C library

    def test_complex_kernels(self):
        # the hyperbolic functions: the library's complex versions; exp, sin and cos: real
        # arguments (zero imaginary part) get the real function, the others the complex C
        # function (SIMD rows with both are computed by the scalar kernel)
        rng = np.random.default_rng(2)
        n = 4096
        re_ = rng.uniform(-20, 20, n)
        cases = {"real": re_ + 0j, "complex": re_ + 1j * rng.uniform(-3, 3, n),
                 "mixed": re_ + 1j * np.where(rng.random(n) < 0.5, 0.0, rng.uniform(-3, 3, n))}
        for name, (f_sp, f_np, _) in FUNCTIONS.items():
            if name == "log":
                # a complex kernel's log is the natural logarithm (code.rs cplx_log), not the
                # library's log10: always the C function
                for z in cases.values():
                    f = compile_func([x], [f_sp(x)], ty=INLINE, dtype="complex128")
                    c_lib = compile_func([x], [f_sp(x)], dtype="complex128")
                    np.testing.assert_array_equal(np.ravel(f(z)), np.ravel(c_lib(z)))
                continue
            for options in [dict(), dict(fast_complex=False), dict(enable_simd512=True), dict(use_simd=False),
                            dict(ty="bytecode;inline_math=true")]:
                options = dict(options)
                ty = options.pop("ty", INLINE)
                f = compile_func([x], [f_sp(x)], ty=ty, dtype="complex128", **options)
                c_lib = compile_func([x], [f_sp(x)], dtype="complex128", **options)
                for case, z in cases.items():
                    with self.subTest(name=name, ty=ty, options=options, case=case):
                        got = np.ravel(f(z))
                        np.testing.assert_allclose(got, f_np(z), rtol=1e-13)
                        real = z.imag == 0
                        if name not in ("ln", "log"):  # complex for negative reals
                            self.assertTrue(np.all(got[real].imag == 0))
                        want = np.ravel(c_lib(z))
                        if name in COMPLEX_INLINED:
                            # the library's: an ulp or so off the C function at some points
                            if case != "real" and ty == INLINE and options.get("use_simd", True):
                                self.assertGreater(np.sum(got != want), 0)
                        else:
                            # complex arguments: the complex C function's bits
                            np.testing.assert_array_equal(got[~real], want[~real])

    def test_complex_functions(self):
        # accuracy (normwise) against mpmath, also near the poles of tanh and coth (z = i pi/2
        # and i pi), where the C path's tanh cancelled before 0020; and the
        # domain: |re| >= 710, |im| >= 2^20 pi/2, NaN and infinities get the C function
        import mpmath as mp

        rng = np.random.default_rng(8)
        n = 800
        z = np.concatenate([rng.uniform(-3, 3, n) + 1j * rng.uniform(-3, 3, n),
                            rng.uniform(-700, 700, n) + 1j * rng.uniform(-50, 50, n),
                            rng.uniform(-1e-3, 1e-3, n) + 1j * (np.pi / 2 * rng.integers(-9, 9, n)
                                                               + rng.uniform(-1e-3, 1e-3, n))])
        for name in COMPLEX_INLINED:
            f = compile_func([x], [getattr(sp, name)(x)], ty=INLINE, dtype="complex128")
            got = np.ravel(f(z))
            with mp.workprec(200):
                ref = [getattr(mp, name)(mp.mpc(c.real, c.imag)) for c in z]
            worst = 0.0
            for a, r in zip(got, ref):  # (r is a 200-bit mpc)
                m = abs(r)
                if 1e-300 < m < 1e300:
                    e = max(abs(mp.mpf(a.real) - r.real), abs(mp.mpf(a.imag) - r.imag)) / np.spacing(float(m))
                    worst = max(worst, float(e))
            self.assertLess(worst, 6, name)

        for name in COMPLEX_INLINED:
            lx, ly = COMPLEX_DOMAIN.get(name, (710, 2**20 * np.pi / 2))
            outside = np.array([lx * 1.1 + 1j, -lx * 1.5 + 0.5j, 1 + ly * 1.2j, complex(np.nan, 1),
                                complex(1, np.inf)] * 4)
            f = compile_func([x], [getattr(sp, name)(x)], ty=INLINE, dtype="complex128")
            c_lib = compile_func([x], [getattr(sp, name)(x)], dtype="complex128")
            np.testing.assert_array_equal(np.ravel(f(outside)), np.ravel(c_lib(outside)), err_msg=name)

    def test_evaluator(self):
        # a Symbolica evaluator with exp statements and sums of products of their results, also
        # with `compress` (too small to form funclets; compression-mode kernels with inlined
        # functions were checked on the GL297 evaluators with `sqrt` replaced by `exp`)
        ir = [("fun", ("temp", i), "exp", [], [("param", i)], False) for i in range(16)]
        t = 16
        for k in range(4):
            prods = []
            for i in range(10):
                ir.append(("mul", ("temp", t), [("temp", (2 * i + k) % 16), ("temp", (3 * i + k + 1) % 16)], 0))
                prods.append(t)
                t += 1
            ir.append(("add", ("temp", t), [("temp", j) for j in prods], 0))
            ir.append(("fun", ("temp", t + 1), "exp", [], [("temp", t)], False))
            ir.append(("mul", ("out", k), [("const", 0), ("temp", t + 1)], 0))
            t += 2
        ev = str((ir, t, ["1/1000"])).replace("'exp'", "exp").replace("'1/1000'", "1/1000")
        X = np.random.default_rng(5).uniform(-1, 1, (37, 16))
        want = compile_evaluator(ev, num_params=16, use_threads=False, compress=False).evaluate(X)
        for level in [2, 3]:
            for simd, simd512 in [(False, False), (True, False), (True, True)]:
                with self.subTest(level=level, simd=simd, simd512=simd512):
                    f = compile_evaluator(ev, num_params=16, use_threads=False, compress=True, opt_level=level,
                                          use_simd=simd, enable_simd512=simd512, ty=INLINE)
                    np.testing.assert_allclose(f.evaluate(X), want, rtol=1e-13)


if __name__ == "__main__":
    unittest.main()

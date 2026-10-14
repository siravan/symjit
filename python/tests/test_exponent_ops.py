"""The exponent-field operations `pow2i`, `getexp` and `getmant`.

They are defined bit for bit (`rust/utils.rs`):

    pow2i(x)   = bits(x + 0x1.8p52 + 1023) << 52      (2**x for an integer x in [-1022, 1023])
    getexp(x)  = ((bits(x) << 1) >> 53) - 1023         (floor(log2|x|) for normal x)
    getmant(x) = bits(x) & 0x800fffffffffffff | bits(1.0)   (the mantissa in [1, 2), with the sign)

so every kernel (scalar, SIMD, AVX-512, the bytecode interpreter, complex kernels) must give
exactly these bits, also for zeros, subnormals, infinities, NaN and values outside the useful
range. SymPy reaches them as undefined functions (`sympy.Function("pow2i")`), the Composer as
methods. The Rust tests (`utils::exponent_tests`) cover the same on ARM64 and RISC-V under qemu.

Set SYMJIT_PYTHON_PATH to test another build of the package.
"""

import os
import sys
import unittest

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("SYMJIT_PYTHON_PATH", os.path.join(HERE, "..")))

from symjit import Composer, compile_composer, compile_func  # noqa: E402

U = np.uint64


def reference(name, x):
    x = np.asarray(x, dtype=np.float64)
    b = x.view(U)
    if name == "pow2i":
        return ((x + U(0x43380000000003FF).view(np.float64)).view(U) << U(52)).view(np.float64)
    if name == "getexp":
        return ((b << U(1)) >> U(53)).astype(np.float64) - 1023.0
    return ((b & U(0x800FFFFFFFFFFFFF)) | U(0x3FF0000000000000)).view(np.float64)


def inputs():
    rng = np.random.default_rng(1)
    special = [0.0, -0.0, np.inf, -np.inf, np.nan, 5e-324, -2.2e-308, 1e300, -1e300, 0.5, 1.5, 2.5,
               -0.5, -1.5, 1023, 1024, -1022, -1023, -1024, 1025, 2047, 2048, 4096, -4096, 3.0e15]
    return np.concatenate([special, np.arange(-1100, 1101, dtype=float), rng.normal(0, 1e3, 300),
                           np.exp(rng.uniform(-700, 700, 300)) * rng.choice([-1, 1], 300)])


OPTIONS = [dict(), dict(opt_level=0), dict(opt_level=3), dict(fastmath=False), dict(use_simd=False),
           dict(enable_simd512=True), dict(enable_simd512=True, opt_level=3), dict(ty="bytecode"),
           dict(cse=False)]
NAMES = ["pow2i", "getexp", "getmant"]


class ExponentOps(unittest.TestCase):
    def assertBits(self, got, want, msg):
        got, want = np.ravel(got), np.ravel(want)
        bad = np.flatnonzero(got.view(U) != want.view(U))
        self.assertEqual(len(bad), 0, f"{msg}: x = {inputs()[bad[:4]]}, got {got[bad[:4]]}, want {want[bad[:4]]}")

    def test_sympy_functions(self):
        x = sp.Symbol("x")
        xs = inputs()
        for name in NAMES:
            want = reference(name, xs)
            for options in OPTIONS:
                with self.subTest(name=name, options=options):
                    f = compile_func([x], [sp.Function(name)(x)], **options)
                    self.assertBits(f(xs), want, f"{name} {options} vectorized")
                    self.assertBits([f(t) for t in xs[:300]], want[:300], f"{name} {options} scalar")

    def test_in_expressions(self):
        # operands in other registers than the argument, results used by arithmetic
        x = sp.Symbol("x")
        xs = inputs()
        e = sp.Function("pow2i")(sp.Function("getexp")(x)) * sp.Function("getmant")(x)
        want = reference("pow2i", reference("getexp", xs)) * reference("getmant", xs)
        normal = np.isfinite(xs) & (np.abs(xs) >= 2.2250738585072014e-308)
        np.testing.assert_array_equal(want[normal], xs[normal])  # x = 2**getexp(x) * getmant(x)
        for options in OPTIONS:
            with self.subTest(options=options):
                f = compile_func([x], [e], **options)
                self.assertBits(f(xs), want, f"{options}")

    def test_complex_kernels(self):
        # the operations apply to the real part and return a real
        x = sp.Symbol("x")
        xs = inputs()[:500] + 1j * np.linspace(-7, 7, 500)
        for name in NAMES:
            want = reference(name, xs.real)
            for options in [dict(), dict(fast_complex=False), dict(enable_simd512=True), dict(ty="bytecode")]:
                with self.subTest(name=name, options=options):
                    f = compile_func([x], [sp.Function(name)(x)], dtype="complex128", **options)
                    got = np.ravel(f(xs))
                    self.assertBits(got.real.copy(), want, f"{name} {options}")
                    self.assertTrue(np.all(got.imag == 0), f"{name} {options}: imaginary part")

    def test_composer(self):
        xs = inputs()
        cp = Composer(1, 3)
        a = cp.arg(0)
        for i, name in enumerate(NAMES):
            cp.assign(cp.out(i), getattr(cp, name)(a))
        for options in [dict(), dict(use_simd=False), dict(enable_simd512=True)]:
            with self.subTest(options=options):
                f = compile_composer(cp, **options)
                got = np.array(f.evaluate(xs.reshape(-1, 1).copy())).reshape(len(xs), -1).T
                for i, name in enumerate(NAMES):
                    self.assertBits(got[i], reference(name, xs), f"{name} {options}")


if __name__ == "__main__":
    unittest.main()

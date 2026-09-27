"""
Accuracy report for symjit's elementary functions and compound expressions.

    python accuracy.py                 # the tables below
    python accuracy.py -n 2000         # more samples per function

1. Elementary functions: the error in ulps (units in the last place, 0.5 = correctly rounded) of
   the scalar kernel, the vectorized (SIMD) kernel, and the kernel compiled with `fastmath=False`,
   measured against mpmath at 60 digits. numpy (or scipy) is shown for comparison.
2. Compound expressions where rounding and cancellation matter (Horner polynomials, sums with
   many terms, differences of nearly equal numbers). The error is relative to the exact value; the
   reference column is numpy evaluating the *same formula* in double precision, so symjit should be
   about as accurate (FMA, used by `fastmath`, usually makes it slightly better).

The pass/fail version of these checks is python/tests/test_accuracy.py (and test_special_values.py
for NaN, Inf, zeros, and subnormals).
"""

import argparse
import os
import sys
import warnings

import numpy as np
import scipy.special as ss
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python", "tests"))

from accuracy_cases import CASES, abs_error, mp, ulp_error, x  # noqa: E402
from symjit import compile_func  # noqa: E402

warnings.simplefilter("ignore")

NUMPY = {
    "sin": np.sin, "sin (large)": np.sin, "cos": np.cos, "cos (large)": np.cos, "tan": np.tan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan, "sinh": np.sinh, "cosh": np.cosh,
    "tanh": np.tanh, "asinh": np.arcsinh, "acosh": np.arccosh, "atanh": np.arctanh, "exp": np.exp,
    "exp (wide)": np.exp, "log": np.log, "log (near 1)": np.log, "sqrt": np.sqrt, "cbrt": np.cbrt,
    "x**2.7": lambda a: a**2.7, "x**-1.3": lambda a: a**-1.3, "erf": ss.erf, "erfc": ss.erfc,
    "gamma": ss.gamma, "loggamma": ss.gammaln,
}


def errors(values, exact, metric):
    fn = ulp_error if metric == "ulp" else abs_error
    e = np.array([fn(g, r) for g, r in zip(values, exact)])
    return e.max(), e.mean()


def functions(n):
    print(f"Elementary functions, {n} samples each (errors in ulps; metric 'abs' = 2^-52 relative to max(|y|, 1))\n")
    print(f"{'function':14s} {'metric':>6s} {'symjit max':>11s} {'mean':>7s} {'scalar=vec':>10s} {'fastmath off':>13s}"
          f" {'numpy/scipy max':>16s}   note")
    for name, (sym, ref, sampler, bound, metric, note) in CASES.items():
        pts = np.array(sorted(set(sampler(n, np.random.default_rng(abs(hash(name)) % 1000 + 1)).tolist())))
        exact = [ref(mp.mpf(float(p))) for p in pts]

        f = compile_func([x], [sym(x)])
        vec = np.ravel(f(pts))
        scalar = np.array([np.ravel(f(float(p)))[0] for p in pts])
        exact_off = np.ravel(compile_func([x], [sym(x)], fastmath=False)(pts))

        mx, mean = errors(vec, exact, metric)
        mx_off, _ = errors(exact_off, exact, metric)
        mx_np, _ = errors(NUMPY[name](pts), exact, metric)
        same = "yes" if np.array_equal(scalar, vec) else "NO"
        print(f"{name:14s} {metric:>6s} {mx:11.2f} {mean:7.2f} {same:>10s} {mx_off:13.2f} {mx_np:16.2f}   {note}")


def compound(n):
    print("\nCompound expressions (relative error; the reference column is numpy evaluating the same formula)\n")
    rng = np.random.default_rng(0)
    coefficients = [float(c) for c in rng.uniform(-1, 1, 21)]
    i = sp.Symbol("i")

    def horner(t, c):
        r = c[-1]
        for a in reversed(c[:-1]):
            r = a + t * r
        return r

    cases = [
        ("Horner polynomial, degree 20", lambda t: horner(t, coefficients), lambda t: horner(t, [mp.mpf(c) for c in coefficients]),
         rng.uniform(-1, 1, n)),
        ("(exp(x) - 1) / x near 0", lambda t: (sp.exp(t) - 1) / t, lambda t: (mp.exp(t) - 1) / t, rng.uniform(1e-6, 1e-3, n)),
        ("sqrt(x + 1) - sqrt(x), x large", lambda t: sp.sqrt(t + 1) - sp.sqrt(t), lambda t: mp.sqrt(t + 1) - mp.sqrt(t),
         np.exp(rng.uniform(5, 30, n))),
        ("1 - cos(x) near 0", lambda t: 1 - sp.cos(t), lambda t: 1 - mp.cos(t), rng.uniform(1e-4, 1e-2, n)),
        ("x**3 - 3 x**2 + 3 x - 1 near 1", lambda t: t**3 - 3 * t**2 + 3 * t - 1, lambda t: t**3 - 3 * t**2 + 3 * t - 1,
         1 + rng.uniform(-1e-3, 1e-3, n)),
        ("sum_{k=1}^{1000} 1/(k + x)^2", lambda t: sp.Sum(1 / (i + t) ** 2, (i, 1, 1000)),
         lambda t: mp.fsum(1 / (mp.mpf(k) + t) ** 2 for k in range(1, 1001)), rng.uniform(0, 1, min(n, 300))),
        ("product_{k=1}^{50} (1 + x/k)", lambda t: sp.Product(1 + t / i, (i, 1, 50)),
         lambda t: mp.fprod(1 + t / mp.mpf(k) for k in range(1, 51)), rng.uniform(0, 2, min(n, 300))),
    ]

    print(f"{'expression':38s} {'numpy (same formula)':>21s} {'symjit':>10s} {'fastmath off':>13s}   (max relative error)")
    for name, sym, ref, pts in cases:
        exact = [ref(mp.mpf(float(p))) for p in pts]
        expr = sym(x)
        lam = sp.lambdify(x, expr.doit() if hasattr(expr, "doit") else expr, "numpy")
        with np.errstate(all="ignore"):
            if isinstance(expr, (sp.Sum, sp.Product)):
                k = np.arange(1, 1001 if isinstance(expr, sp.Sum) else 51, dtype=float)[:, None]
                base = pts[None, :]
                ref_np = np.zeros(len(pts))
                if isinstance(expr, sp.Sum):
                    acc = np.zeros(len(pts))
                    for row in 1 / (k + base) ** 2:  # sequential accumulation, like the compiled loop
                        acc = acc + row
                    ref_np = acc
                else:
                    acc = np.ones(len(pts))
                    for row in 1 + base / k:
                        acc = acc * row
                    ref_np = acc
            else:
                ref_np = np.asarray(lam(pts), float)

        def rel(values):
            return max(float(abs(mp.mpf(float(g)) - e) / abs(e)) for g, e in zip(values, exact))

        sj = np.ravel(compile_func([x], [expr])(pts))
        sj_off = np.ravel(compile_func([x], [expr], fastmath=False)(pts))
        print(f"{name:38s} {rel(ref_np):21.2e} {rel(sj):10.2e} {rel(sj_off):13.2e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", type=int, default=600, help="samples per function")
    args = parser.parse_args()
    functions(args.n)
    compound(args.n)


if __name__ == "__main__":
    main()

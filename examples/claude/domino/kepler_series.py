# This file is generated using claude code
"""Kepler's equation E - e sin E = M solved by power series in the eccentricity e.

Two classical series give the eccentric anomaly E(e, M) of an elliptic orbit:

    Lagrange (1770):   E = M + sum_n e^n Q_n(M),
                       Q_n(M) = 1/(2^(n-1) n!) sum_k (-1)^k C(n, k) (n - 2k)^(n-1) sin((n - 2k) M)

    Bessel (1824):     E = M + sum_j (2/j) J_j(j e) sin(j M),
                       J_j(x) = sum_s (-1)^s (x/2)^(j+2s) / (s! (j+s)!)

Truncated at the same order K in e, they are the same polynomial, but they add the terms in a
different order (by powers of e and by harmonics of M). At order 250 each has about 15,700 terms
with coefficients up to 1e44 (machine code in megabytes). Both converge only for e below the
Laplace limit 0.6627434... Checks:

1. The two forms agree (rounding differences only, for e up to 0.6).
2. Against Newton's method on Kepler's equation: the error falls with the order below the
   Laplace limit (to rounding at e = 0.5 for order 250), and grows with it above (the series
   diverges at e = 0.8).
3. The Laplace limit from the growth of the coefficients, max_M |Q_n(M)| ~ n^(-3/2) rho^(-n):
   rho from the ratio of consecutive terms, against the root of x exp(sqrt(1+x^2)) = 1 + sqrt(1+x^2).
4. Vectorized calls (SIMD) on a grid of (e, M) points with e <= 0.5, against Newton's method
   (within the truncation error of the order).

    python kepler_series.py           # order 250
    python kepler_series.py -K 100    # another order
"""

import argparse
import sys
import time

import util

opts = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False)
opts.add_argument("-K", type=int, default=250, help="the order in e (default 250)")
our, rest = opts.parse_known_args()
if "-h" in rest or "--help" in rest:
    opts.print_help()
    sys.exit(0)
sys.argv = sys.argv[:1] + rest
args = util.process_argv()

import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402

K = our.K
e, M = sp.symbols("e M")


def Q(n):
    """the coefficient of e^n in Lagrange's series, a sum of sines"""
    c = sp.Rational(1, 2 ** (n - 1) * sp.factorial(n))
    return sp.Add(*[c * (-1) ** k * sp.binomial(n, k) * (n - 2 * k) ** (n - 1) * sp.sin((n - 2 * k) * M)
                    for k in range(n // 2 + 1) if n != 2 * k])


def P(j, K):
    """(2/j) J_j(j e) up to e^K: the coefficient of sin(j M) in Bessel's series, a polynomial in e"""
    return sp.Add(*[sp.Rational(2, j) * (-1) ** s * sp.Rational(j, 2) ** (j + 2 * s)
                    / (sp.factorial(s) * sp.factorial(j + s)) * e ** (j + 2 * s) for s in range((K - j) // 2 + 1)])


def newton(ecc, mean):
    E = mean + ecc * np.sin(mean)
    for _ in range(100):
        E = E - (E - ecc * np.sin(E) - mean) / (1 - ecc * np.cos(E))
    return E


# Mul(..., evaluate=False) keeps the grouping, so that the two forms are not merged into one sum
def build():
    Qs = [Q(n) for n in range(1, K + 1)]
    lagrange = sp.Add(M, *[sp.Mul(e ** n, q, evaluate=False) for n, q in enumerate(Qs, 1)], evaluate=False)
    bessel = sp.Add(M, *[sp.Mul(sp.sin(j * M), P(j, K), evaluate=False) for j in range(1, K + 1)], evaluate=False)
    return [([e, M], [lagrange, bessel]), ([M], Qs)], dict(terms=sum(len(x.args) for x in Qs))


# compile_cached caches the expressions' JSON models (which keep the grouping) when
# SYMJIT_EXAMPLE_CACHE is set (run_physics.py)
(f, q), info, t_build, t_compile, cached = util.compile_cached("kepler_series", (K,), build, **args)
terms = info["terms"]
print(f"order {K}: {terms:,} terms in each form; sympy {t_build:.1f} s{' (cached)' if cached else ''}, "
      f"symjit compilation {t_compile:.1f} s, "
      f"{f.measure('ker-scalar-size') / 2**20:.2f} MiB of machine code (+ {q.measure('ker-scalar-size') / 2**20:.2f} MiB "
      f"for the {K} coefficients Q_n)\n")

ok = True


def report(name, passed, detail):
    global ok
    ok &= passed
    print(f"{'ok  ' if passed else 'FAIL'} {name}: {detail}")


Ms = np.linspace(0.05, 2 * np.pi - 0.05, 200)

# 1. the two forms
worst = 0.0
for ecc in (0.1, 0.3, 0.5, 0.6):
    r = np.array([f(ecc, m) for m in Ms])
    worst = max(worst, np.max(np.abs(r[:, 0] - r[:, 1])))
report("two forms", worst < 1e-12, f"max |Lagrange - Bessel| for e <= 0.6: {worst:.1e}")

# 2. convergence below the Laplace limit, divergence above (partial sums from the Q_n)
Qv = np.array([q(m) for m in Ms]).reshape(len(Ms), K)
orders = [K // 4, K // 2, K]
rows = []
for ecc in (0.3, 0.5, 0.6, 0.65, 0.7, 0.8):
    ref = newton(ecc, Ms)
    errs = [np.max(np.abs(Ms + Qv[:, :k] @ ecc ** np.arange(1, k + 1) - ref)) for k in orders]
    rows.append((ecc, errs))
conv = all(errs[2] <= errs[1] <= errs[0] for ecc, errs in rows if ecc <= 0.65)
div = all(errs[2] > errs[1] > errs[0] for ecc, errs in rows if ecc >= 0.8)  # near the limit it falls first
truncation = rows[1][1][2]  # the error of order K at e = 0.5
report("convergence", conv and div, "max |E - Newton| at orders " + ", ".join(map(str, orders))
       + " (falls with the order below the Laplace limit, grows above it):")
for ecc, errs in rows:
    print(f"       e = {ecc:<4}: " + "  ".join(f"{x:9.1e}" for x in errs))

# 3. the Laplace limit
# max over M of |Q_n|, n = 1 ... K, on a fine grid (the peaks sharpen and move with n)
fine = np.linspace(0, 2 * np.pi, 20001)
peak = np.max(np.abs(np.array([np.asarray(v).ravel() for v in q(fine)])), axis=1)
n = K - 1 if K % 2 == 0 else K  # Q_n of the same parity: n and n - 2
ratio = peak[n - 1] / peak[n - 3]  # ~ rho^-2 ((n - 2)/n)^(3/2)
rho = np.sqrt(((n - 2) / n) ** 1.5 / ratio)
x = 0.66
for _ in range(50):  # the root of g(x) = x exp(sqrt(1 + x^2)) - 1 - sqrt(1 + x^2)
    s = np.sqrt(1 + x * x)
    x -= (x * np.exp(s) - 1 - s) / (np.exp(s) * (1 + x * x / s) - x / s)
report("Laplace limit", abs(rho / x - 1) < 0.01,
       f"from |Q_{n}| / |Q_{n - 2}|: {rho:.5f}; exact {x:.10f} (relative difference {abs(rho / x - 1):.1e})")

# 4. vectorized on a grid
E, MM = np.meshgrid(np.linspace(0, 0.5, 100), Ms)
f(np.array([0.1] * 8), np.array([1.0] * 8))  # the first call may prepare the SIMD kernel
t0 = time.perf_counter()
L, B = (np.asarray(v).ravel() for v in f(E.ravel(), MM.ravel()))
t1 = time.perf_counter()
ref = newton(E.ravel(), MM.ravel())
t2 = time.perf_counter()
d = max(np.max(np.abs(L - ref)), np.max(np.abs(B - ref)))
report("vectorized", d < 2 * truncation + 1e-13,
       f"{E.size:,} points with e <= 0.5: {1e3 * (t1 - t0):.0f} ms (both forms), Newton in numpy {1e3 * (t2 - t1):.0f} ms; "
       f"max error {d:.1e} (the truncation error at e = 0.5: {truncation:.1e})")

print(f"\nmachine code: {f.measure('ker-scalar-size') / 2**20:.2f} MiB (scalar), "
      f"{f.measure('ker-simd-size') / 2**20:.2f} MiB (SIMD)")
print("all checks passed" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)

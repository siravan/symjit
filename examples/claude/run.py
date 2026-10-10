"""Runs the symbolic-numeric integrator of sni.py on the 170 test integrals of cases.py with
each backend, checks every answer independently, and compares the times.

    python run.py                          # backends symjit and lambdify
    python run.py --hyint ~/src/hyint      # also the original hyint (a clone of github.com/siravan/hyint)
    python run.py --show                   # print every integral and its answer
    python run.py --repeat 3               # best of 3 runs per backend

An answer is accepted if d/dx(answer) equals the integrand at four fixed complex points
(30-digit evaluation by sympy; points where the integrand is huge or tiny are avoided),
independently of the numbers the integrator used. The
times of sni.py are split into building the numerical functions (differentiation and
symjit compilation or lambdify), evaluating them, and the linear algebra.
"""

import argparse
import os
import sys
import time

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sni  # noqa: E402
from cases import basic_integrals, x  # noqa: E402

CHECK_POINTS = [0.37 + 0.11j, 1.91 + 0.4j, -2.73 + 1.3j, 0.8 - 1.7j, -0.61 - 0.23j, 0.13 + 0.97j,
                -1.37 + 0.52j, 2.44 - 0.71j]


def check(eq, sol, var=x, num_points=4):
    """'ok' if d/dvar sol equals eq at the check points (relative error < 1e-8), 'approx' if
    it does to 1e-5 (floating-point coefficients of limited accuracy), else 'wrong'.

    Points where |eq| is above 1e10 or below 1e-10 are skipped while there are others: near
    a singularity (eq ~ 1e60), coefficients with float errors of 1e-12 leave large absolute
    errors in a correct answer. `num_points` points are used."""
    if sol is None or sol == 0 and eq != 0:
        return "wrong"
    d = sp.diff(sol, var) - eq
    worst = 0.0
    try:
        refs = []
        for z in CHECK_POINTS:
            ref = complex(eq.subs(var, z).evalf(30))
            if np.isfinite(ref):
                refs.append((not 1e-10 < abs(ref) < 1e10, z, ref))
        refs.sort(key=lambda t: t[0])  # well-scaled points first (stable)
        if not refs:
            return "wrong"
        for _, z, ref in refs[:num_points]:
            val = complex(d.subs(var, z).evalf(30))
            err = abs(val) / max(1.0, abs(ref))
            if not err < 1e-5:  # also catches nan
                return "wrong"
            worst = max(worst, err)
    except (TypeError, ValueError, ZeroDivisionError):
        return "wrong"
    return "ok" if worst < 1e-8 else "approx"


def correct(eq, sol, var=x):
    """True if d/dvar sol equals eq at the check points (relative error < 1e-8)"""
    return check(eq, sol, var) == "ok"


def run_sni(backend, seed):
    for k in sni.STATS:
        sni.STATS[k] = 0
    rng = np.random.default_rng(seed)
    answers, times = [], []
    for eq in basic_integrals:
        eq = sp.sympify(eq)
        t0 = time.perf_counter()
        try:
            sol = sni.integrate(eq, x, backend=backend, rng=rng)
        except Exception as e:  # report, keep going
            sol = e
        times.append(time.perf_counter() - t0)
        answers.append(sol)
    return answers, times, dict(sni.STATS)


def run_hyint(path, seed):
    sys.path.insert(0, path)
    from hyint import hyint

    np.random.seed(seed)
    answers, times = [], []
    for eq in basic_integrals:
        eq = sp.sympify(eq)
        t0 = time.perf_counter()
        try:
            sol = hyint.integrate(eq, x)
            sol = None if sol == 0 else sol
        except Exception as e:
            sol = e
        times.append(time.perf_counter() - t0)
        answers.append(sol)
    return answers, times, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backends", nargs="*", default=["symjit", "lambdify"], choices=["symjit", "lambdify"])
    parser.add_argument("--hyint", metavar="PATH",
                        help="also run hyint from this directory (a clone of its repository)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1, help="runs per backend; the best total time is kept")
    parser.add_argument("--show", action="store_true", help="print every integral and its answer")
    args = parser.parse_args()

    runs = [(b, lambda b=b: run_sni(b, args.seed)) for b in args.backends]
    if args.hyint:
        runs.append(("hyint", lambda: run_hyint(os.path.expanduser(args.hyint), args.seed)))

    results = {}
    for name, run in runs:
        best = None
        for _ in range(args.repeat):
            r = run()
            if best is None or sum(r[1]) < sum(best[1]):
                best = r
        answers, times, stats = best
        ok = [correct(sp.sympify(eq), a) if not isinstance(a, Exception) else False
              for eq, a in zip(basic_integrals, answers)]
        results[name] = (answers, times, stats, ok)

    n = len(basic_integrals)
    print(f"{n} integrals; times in seconds (best of {args.repeat})\n")
    print(f"{'':10s} {'correct':>8s} {'wrong':>6s} {'none':>5s} {'error':>6s} {'total':>7s} "
          f"{'build':>7s} {'eval':>7s} {'solve':>7s}  compiled/fallbacks")
    for name, (answers, times, stats, ok) in results.items():
        errors = sum(isinstance(a, Exception) for a in answers)
        none = sum(a is None for a in answers)
        wrong = n - sum(ok) - errors - none
        line = f"{name:10s} {sum(ok):8d} {wrong:6d} {none:5d} {errors:6d} {sum(times):7.2f}"
        if stats:
            line += (f" {stats['build']:7.2f} {stats['eval']:7.3f} {stats['solve']:7.2f}"
                     f"  {stats['compiled']}/{stats['fallbacks']}")
        print(line)

    names = list(results)
    if args.show or len(names) > 1:
        print()
    for i, eq in enumerate(basic_integrals):
        oks = [results[nm][3][i] for nm in names]
        if args.show or len(set(oks)) > 1:
            print(f"{str(eq):40s} " + "  ".join(f"{nm}: {'ok' if o else 'FAIL'}" for nm, o in zip(names, oks)))
            if args.show:
                for nm in names:
                    print(f"    {nm:9s} {results[nm][0][i]}")


if __name__ == "__main__":
    main()

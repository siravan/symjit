"""Shared definitions for the accuracy tests (`test_accuracy.py`) and the report
(`examples/claude/accuracy.py`): the functions, their sampling domains, and the reference
implementations (mpmath at 60 digits)."""

import numpy as np
import mpmath as mp
import sympy as sp

mp.mp.dps = 60
x = sp.Symbol("x")


def ulp_error(got, exact):
    """|got - exact| in units of the spacing of doubles at `exact` (0.5 is a correctly rounded result)"""
    g = float(got)
    e = float(exact)
    if not np.isfinite(g) or not np.isfinite(e):
        return np.inf if np.isfinite(e) else np.nan
    if e == 0.0:
        return 0.0 if g == 0.0 else np.inf
    return float(abs(mp.mpf(g) - exact) / mp.mpf(np.spacing(abs(e))))


def abs_error(got, exact, scale=1.0):
    """|got - exact| relative to max(|exact|, scale), in units of 2^-52"""
    return float(abs(mp.mpf(float(got)) - exact) / max(abs(exact), mp.mpf(scale)) / mp.mpf(2) ** -52)


def uni(a, b):
    return lambda n, r: r.uniform(a, b, n)


def logspace(lo, hi):
    return lambda n, r: np.exp(r.uniform(np.log(lo), np.log(hi), n))


def near(c, w):
    return lambda n, r: c + r.uniform(-w, w, n)


# name -> (sympy function, mpmath function, sampler, bound in ulp, metric, note)
#   metric "ulp": relative error in ulps; "abs": error in 2^-52 relative to max(|y|, 1)
CASES = {
    "sin": (sp.sin, mp.sin, uni(-10, 10), 1.0, "ulp", ""),
    "sin (large)": (sp.sin, mp.sin, uni(-1e15, 1e15), 1.0, "ulp", "argument reduction"),
    "cos": (sp.cos, mp.cos, uni(-10, 10), 1.0, "ulp", ""),
    "cos (large)": (sp.cos, mp.cos, uni(-1e15, 1e15), 1.0, "ulp", "argument reduction"),
    "tan": (sp.tan, mp.tan, uni(-1.5, 1.5), 1.0, "ulp", ""),
    "asin": (sp.asin, mp.asin, uni(-1, 1), 1.0, "ulp", ""),
    "acos": (sp.acos, mp.acos, uni(-1, 1), 1.0, "ulp", ""),
    "atan": (sp.atan, mp.atan, uni(-50, 50), 1.0, "ulp", ""),
    "sinh": (sp.sinh, mp.sinh, uni(-20, 20), 2.0, "ulp", ""),
    "cosh": (sp.cosh, mp.cosh, uni(-20, 20), 2.0, "ulp", ""),
    "tanh": (sp.tanh, mp.tanh, uni(-5, 5), 3.0, "ulp", ""),
    "asinh": (sp.asinh, mp.asinh, uni(-50, 50), 2.0, "ulp", ""),
    "acosh": (sp.acosh, mp.acosh, uni(1, 50), 2.0, "ulp", ""),
    "atanh": (sp.atanh, mp.atanh, uni(-0.999, 0.999), 200.0, "ulp",
              "loses accuracy as |x| -> 1 (about 120 ulp at 0.9999); a correctly rounded atanh is 1 ulp"),
    "exp": (sp.exp, mp.exp, uni(-30, 30), 1.0, "ulp", ""),
    "exp (wide)": (sp.exp, mp.exp, uni(-700, 700), 1.0, "ulp", ""),
    "log": (sp.log, mp.log, logspace(1e-30, 1e30), 1.0, "ulp", ""),
    "log (near 1)": (sp.log, mp.log, near(1.0, 1e-3), 1.0, "ulp", ""),
    "sqrt": (sp.sqrt, mp.sqrt, uni(0, 100), 1.0, "ulp", ""),
    "cbrt": (lambda a: sp.Pow(a, sp.Rational(1, 3)), mp.cbrt, uni(0, 100), 1.0, "ulp", ""),
    "x**2.7": (lambda a: a**2.7, lambda a: a ** mp.mpf(2.7), uni(0, 50), 1.0, "ulp", ""),
    "x**-1.3": (lambda a: a**-1.3, lambda a: a ** mp.mpf(-1.3), uni(0.01, 50), 1.0, "ulp", ""),
    "erf": (sp.erf, mp.erf, uni(-5, 5), 4.0, "ulp", ""),
    "erfc": (sp.erfc, mp.erfc, uni(-3, 10), 100.0, "ulp", "cephes; up to about 60 ulp for large x"),
    "gamma": (sp.gamma, mp.gamma, uni(0.1, 30), 10.0, "ulp", "cephes"),
    "loggamma": (sp.loggamma, mp.loggamma, uni(0.1, 100), 4.0, "abs",
                 "absolute accuracy: the relative error is unbounded near the roots x = 1, 2"),
}

# functions of the special-value tests: name -> (sympy expression builder, numpy/scipy reference)
def special_unary():
    import scipy.special as ss

    return {
        "sin": (sp.sin, np.sin), "cos": (sp.cos, np.cos), "tan": (sp.tan, np.tan),
        "asin": (sp.asin, np.arcsin), "acos": (sp.acos, np.arccos), "atan": (sp.atan, np.arctan),
        "sinh": (sp.sinh, np.sinh), "cosh": (sp.cosh, np.cosh), "tanh": (sp.tanh, np.tanh),
        "asinh": (sp.asinh, np.arcsinh), "acosh": (sp.acosh, np.arccosh), "atanh": (sp.atanh, np.arctanh),
        "exp": (sp.exp, np.exp), "log": (sp.log, np.log), "sqrt": (sp.sqrt, np.sqrt),
        "cbrt": (lambda a: sp.Pow(a, sp.Rational(1, 3)), np.cbrt),
        "abs": (sp.Abs, np.abs), "floor": (sp.floor, np.floor), "ceiling": (sp.ceiling, np.ceil),
        "erf": (sp.erf, ss.erf), "erfc": (sp.erfc, ss.erfc), "gamma": (sp.gamma, ss.gamma),
        "loggamma": (sp.loggamma, ss.gammaln),
        "neg": (lambda a: -a, np.negative), "recip": (lambda a: 1 / a, lambda a: 1 / a),
        "square": (lambda a: a**2, np.square), "cube": (lambda a: a**3, lambda a: a**3),
        "half": (lambda a: a / 2, lambda a: a / 2), "x**-2": (lambda a: a**-2, lambda a: a**-2.0),
    }

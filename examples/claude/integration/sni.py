"""Symbolic-numeric indefinite integration with symjit.

A compact rewrite of hyint (https://github.com/siravan/hyint, BSD-3-Clause), the Python
counterpart of SymbolicNumericIntegration.jl (https://github.com/SciML/SymbolicNumericIntegration.jl,
MIT; Iravanian et al., arXiv:2201.12468 and ISSAC'24). The method:

1. Candidate terms (the ansatz): walking the integrand, each part suggests terms its
   antiderivative may contain (e.g. sin(u) -> cos(u), Si(u); 1/p(x) -> log(p), p**-1 and the
   log/atan terms of the roots of p); constant factors are dropped.
2. A linear system: find q with sum_j q_j * b_j'(x) = f(x) at random points x of a disk in the
   complex plane; dividing each row by f(x) gives A q = 1.
3. Pruning and sparse regression: a QR decomposition drops dependent candidates, then a
   sequentially thresholded least-squares fit (STLSQ, as in SINDy) zeros small coefficients.
4. Coefficients are rounded to small fractions, and the result is checked on held-out points.
   On failure, new points are tried, then a larger candidate set.

Step 2 is where the time goes: every candidate set needs f and the derivative of every
candidate at many complex points. `backend="symjit"` compiles them as one complex function
(shared subexpressions are computed once) and evaluates all points in one vectorized call;
`backend="lambdify"` builds the same function with sympy's `lambdify` (numpy, cse=True).
Either way, the function is built once per candidate set and reused for all trials (hyint
lambdifies every term again for every trial).

Compared with hyint, the sample points cover the whole disk (hyint's lie within sqrt(radius)),
a few antiderivative rules are corrected (asinh, acsc, asech, atanh, acoth), the roots of
denominators are found by sympy.roots with a numerical fallback, and solutions are verified
on points not used for the fit.

    from sympy import Symbol, sin, exp
    from sni import integrate
    x = Symbol("x")
    integrate(x * exp(x) * sin(x), x)           # backend="symjit" (default) or "lambdify"
"""

import os
import sys
import time
from fractions import Fraction

import numpy as np
import sympy as sp
from sympy import (Add, Ci, Ei, Integer, Mul, Rational, Si, acos, acosh, acot,
                   acoth, acsc, acsch, asec, asech, asin, asinh, atan, atanh, cos, cosh, cot,
                   coth, csc, csch, diff, erfi, exp, expand, log, sec, sech, sin, sinh, sqrt,
                   tan, tanh)

sys.path.insert(0, os.environ.get("SYMJIT_PYTHON_PATH",
                                  os.path.join(os.path.dirname(__file__), "..", "..", "..", "python")))

from symjit import compile_func  # noqa: E402

# time spent building the numerical functions (incl. differentiation), evaluating them, and
# in the linear algebra; reset by the caller
STATS = dict(build=0.0, eval=0.0, solve=0.0, compiled=0, fallbacks=0)


def integrate(eq, x, backend="symjit", num_trials=10, depth=1, abstol=1e-6, radius=5.0,
              max_basis=100, rng=None, verbose=False):
    """The antiderivative of the univariate expression `eq` (numeric coefficients only), or
    None if none is found.

    Candidate sets are tried in turn, `num_trials` times each (new random points every time):
    the derivative closure (holonomic integrands only), the partial-integration rules of
    SymbolicNumericIntegration.jl (`homotopy_terms`), its kernel generator (`kernel_terms`)
    and hyint's ansatz (`ansatz_terms`); then, `depth` times, the expansions of the
    homotopy and ansatz sets (pruned to their independent candidates first). Sets larger than
    `max_basis` are skipped."""
    eq = sp.sympify(eq)
    if not eq.has(x):
        return eq * x
    rng = rng if rng is not None else np.random.default_rng()
    # the antiderivative of a real integrand may have at most two complex coefficients (as in
    # SymbolicNumericIntegration.jl): fits of many complex coefficients are spurious
    max_complex = 2 if is_real(eq, x) else None
    centers = linear_centers(eq, x, radius)
    opts = dict(abstol=abstol, radius=radius, max_complex=max_complex, centers=centers)

    generators = []
    if is_holonomic(eq, x):
        generators.append(("closure", lambda: closure(eq, x)))
    generators += [("homotopy", lambda: homotopy_terms(eq, x)), ("kernel", lambda: kernel_terms(eq, x)),
                   ("ansatz", lambda: ansatz_terms(eq, x))]

    tried = set()
    expandable = {}  # name -> (basis, fit, evaluator) of the sets to expand
    for step in range(depth + 1):
        if step == 0:
            stages = [(name, make()) for name, make in generators]
        else:
            stages = []
            for name, (basis, fit, evaluator) in expandable.items():
                # the candidates that were not fitted (e.g. Ei(u)/x) are kept: their multiples by x
                # in the expansion can be needed
                pruned = independent_subset(evaluator, fit, rng, **opts) + [b for b in basis if b not in fit]
                stages.append((name + " expanded", expand_basis(pruned, x) if pruned else None))
            expandable = {}
        for name, basis in stages:
            if basis is None:
                continue
            # simplest first: of linearly dependent candidates, the QR step keeps the first ones
            basis = sorted((b for b in dict.fromkeys(basis) if b.has(x)),
                           key=lambda b: (sp.count_ops(b), sp.default_sort_key(b)))
            key = frozenset(basis)
            if not basis or len(basis) > max_basis or key in tried:
                continue
            tried.add(key)
            fit, evaluator = prepare_basis(eq, x, basis, backend)
            if not fit:
                continue
            if name.split()[0] in ("homotopy", "ansatz"):
                expandable[name.split()[0]] = (basis, fit, evaluator)
            for trial in range(num_trials):
                sol = solve_sparse(evaluator, fit, rng, **opts)
                if sol is not None:
                    if verbose:
                        print(f"solved by the {name} candidates ({len(fit)} terms), trial {trial + 1}")
                    return sol
    return None


def prepare_basis(eq, x, basis, backend):
    """the candidates worth fitting and their Evaluator"""
    t0 = time.perf_counter()
    fit = basis
    derivs = [diff(b, x) for b in basis]
    if not eq.has(*SPECIAL):
        # a candidate whose derivative is not elementary (x*Ei(x)) cannot contribute to an
        # elementary integrand (unless several cancel); symjit cannot compile them
        pairs = [(b, d) for b, d in zip(basis, derivs) if not d.has(*SPECIAL)]
        fit, derivs = [b for b, _ in pairs], [d for _, d in pairs]
    STATS["build"] += time.perf_counter() - t0
    if not fit:
        return [], None
    return fit, Evaluator(eq, derivs, x, backend)


def independent_subset(evaluator, basis, rng, radius=5.0, centers=(), **_):
    """the linearly independent candidates (QR at random points), as SymbolicNumericIntegration.jl
    prunes a candidate set before expanding it"""
    n = len(basis)
    E = evaluator(sample_points(3 * n, radius, rng, centers))
    if E is None:
        return basis
    with np.errstate(all="ignore"):
        A = E[1:].T / E[0][:, None]
    A = A[np.isfinite(A).all(axis=1)][: 2 * n]
    if A.shape[0] < n:
        return basis
    try:
        _, R = np.linalg.qr(A)
    except np.linalg.LinAlgError:
        return basis
    return [b for b, keep in zip(basis, np.abs(np.diag(R)) > 1e-3) if keep]


def linear_centers(eq, x, radius):
    """(center, scale) of the region where each inner argument a*x + b of eq is small: the
    point -b/a and 3/|a|. Far from it, f(a*x + b) can be almost constant (tan, tanh, exp ...),
    and a fit there could accept a wrong antiderivative."""
    centers = set()
    for e in sp.preorder_traversal(eq):
        args = e.args if (e.is_Function or e.is_Pow) else ()
        for u in args:
            if u.has(x) and u.is_polynomial(x) and sp.degree(u, x) == 1:
                a, b = sp.Poly(u, x).all_coeffs()
                a, b = complex(a), complex(b)
                if abs(a) > 1 or abs(b) > 1:
                    centers.add((-b / a, min(radius, 3 / abs(a))))
    return sorted(centers, key=lambda c: (c[0].real, c[0].imag, c[1]))


def is_real(eq, x):
    """True if eq is real (and finite) at one of a few real points: a real integrand may be
    real only on part of the axis (e.g. asin(2*x) for |x| < 1/2)"""
    for v in (0.1234567, 0.6180339, 1.4142136, 2.7182818):
        try:
            z = complex(eq.subs(x, v).evalf())
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if np.isfinite(z) and abs(z.imag) < 1e-12 * max(1.0, abs(z)):
            return True
    return False


##################### numerical part ##########################################

# complex functions symjit compiles through their definitions
REWRITES = [(acsc, lambda u: asin(1 / u)), (asec, lambda u: acos(1 / u)), (acot, lambda u: atan(1 / u)),
            (acsch, lambda u: asinh(1 / u)), (asech, lambda u: acosh(1 / u)), (acoth, lambda u: atanh(1 / u))]


def for_symjit(e):
    for f, g in REWRITES:
        if e.has(f):
            e = e.replace(f, g)
    return e


SPECIAL = (Ei, Si, Ci, erfi, sp.li)


class Evaluator:
    """f and the derivatives of the candidate terms (`derivs`) as one function of x,
    evaluated at an array of complex points (rows: f, b_1', ..., b_n')."""

    def __init__(self, eq, derivs, x, backend):
        t0 = time.perf_counter()
        exprs = [eq] + list(derivs)
        self.f = None
        if backend == "symjit":
            try:
                # one thread: the calls have few points, and run.py/corpus.py run many in parallel
                self.f = compile_func([x], [for_symjit(e) for e in exprs], dtype="complex128", use_threads=False)
                self.symjit = True
                STATS["compiled"] += 1
            except Exception:
                # an operation symjit does not support: fall back to lambdify
                STATS["fallbacks"] += 1
        if self.f is None:
            self.f = sp.lambdify(x, exprs, ["scipy", "numpy"], cse=True)
            self.symjit = False
        STATS["build"] += time.perf_counter() - t0

    def __call__(self, points):
        """the values (rows: f, b_1', ...), or None if they cannot be computed (e.g. a function
        that neither symjit nor numpy/scipy implements)"""
        t0 = time.perf_counter()
        try:
            with np.errstate(all="ignore"):
                if self.symjit:
                    rows = self.f(points)
                else:
                    rows = [np.broadcast_to(np.asarray(r, dtype=complex), points.shape) for r in self.f(points)]
            return np.array(rows, dtype=complex)
        except Exception:
            return None
        finally:
            STATS["eval"] += time.perf_counter() - t0


def sample_points(n, radius, rng, centers=()):
    """n random points with |z - c| log-uniform in [s/20, s] and uniform angles, for the
    center c = 0 with s = radius, and, for half of the points, the (c, s) of `centers` (see
    linear_centers): points where the functions vary"""
    c = np.zeros(n, dtype=complex)
    s = np.full(n, radius)
    if centers:
        k = rng.integers(len(centers), size=n)
        near = rng.random(n) < 0.5
        c[near] = np.array([centers[i][0] for i in k])[near]
        s[near] = np.array([centers[i][1] for i in k])[near]
    r = s * np.exp(np.log(1 / 20) * rng.random(n))
    return c + r * np.exp(2j * np.pi * rng.random(n))


def stlsq(A, thresholds):
    """sequentially thresholded least squares for A q = 1: yields (active columns, q) per
    threshold"""
    b = np.ones(A.shape[0], dtype=complex)
    for thr in thresholds:
        active = np.ones(A.shape[1], dtype=bool)
        while True:
            q, *_ = np.linalg.lstsq(A[:, active], b, rcond=None)
            keep = np.abs(q) > thr
            if keep.all():
                break
            active[active] = keep
            if not active.any():
                break
        if active.any():
            yield active, q


def solve_sparse(evaluator, basis, rng, abstol=1e-6, radius=5.0, num_verify=8, max_complex=None, centers=()):
    """One trial: fits the candidates at new random points and returns the antiderivative if
    it passes the check at held-out points (and has at most `max_complex` complex
    coefficients), else None."""
    n = len(basis)
    m = 2 * n + num_verify
    E = evaluator(sample_points(m + n, radius, rng, centers))  # extra points replace improper ones
    if E is None:
        return None
    t0 = time.perf_counter()
    try:
        f = E[0]
        with np.errstate(all="ignore"):
            A = E[1:].T / f[:, None]
        ok = np.isfinite(A).all(axis=1) & (np.abs(f) > 1e-12)
        A = A[ok][:m]
        if A.shape[0] < m:
            return None
        A_fit, A_ver = A[: 2 * n], A[2 * n:]

        # a linearly independent subset of the candidates
        _, R = np.linalg.qr(A_fit)
        independent = np.abs(np.diag(R)) > 1e-3
        A_fit, A_ver = A_fit[:, independent], A_ver[:, independent]
        terms = [b for b, keep in zip(basis, independent) if keep]

        for active, q in stlsq(A_fit, np.exp(np.arange(-10, -4))):
            err = np.max(np.abs(A_ver[:, active] @ q - 1))
            if err > abstol:
                continue
            if max_complex is not None and np.sum(np.abs(q.imag) > 1e-8 * np.abs(q)) > max_complex:
                continue
            kernel = [b for b, a in zip(terms, active) if a]
            # rounded coefficients if the check is (almost) as good with them as without: a
            # coefficient like e**8/2 must not be rounded to a nearby fraction. If rounding all
            # of them fails, only those within 1e-10 of their rounded value are rounded.
            nice = [nice_number(c) for c in q]
            close = [abs(complex(r) - c) < 1e-10 * max(1.0, abs(c)) for r, c in zip(nice, q)]
            for choice in (nice, [r if ok else nice_float(c) for r, c, ok in zip(nice, q, close)]):
                q_choice = np.array([complex(c) for c in choice])
                if np.max(np.abs(A_ver[:, active] @ q_choice - 1)) <= max(10 * err, 1e-9):
                    return Add(*[c * b for c, b in zip(choice, kernel)])
            return Add(*[nice_float(c) * b for c, b in zip(q, kernel)])
        return None
    except np.linalg.LinAlgError:
        return None
    finally:
        STATS["solve"] += time.perf_counter() - t0


NICE_CONSTANTS = [sp.pi, sqrt(sp.pi), sqrt(2), sqrt(3), sqrt(5)]


def nice_number(c, max_den=100, tol=1e-7):
    """c as a sympy number: a small fraction, or a simple expression in pi, sqrt(pi), sqrt(2),
    sqrt(3) and sqrt(5) (e.g. sqrt(pi)/2), if c is that close to one (real and imaginary parts)"""
    def part(v):
        r = Fraction(v).limit_denominator(max_den)
        if abs(float(r) - v) < tol * max(1.0, abs(v)):
            return Rational(r.numerator, r.denominator)
        s = sp.nsimplify(v, NICE_CONSTANTS, tolerance=tol * max(1.0, abs(v)))
        if sp.count_ops(s) <= 6 and abs(float(s) - v) < tol * max(1.0, abs(v)):
            return s
        return sp.Float(v, 15)

    re, im = part(c.real), part(c.imag) if abs(c.imag) > tol * max(1.0, abs(c)) else 0
    return re + sp.I * im


def nice_float(c):
    return sp.Float(c.real, 15) if abs(c.imag) < 1e-9 * abs(c) else sp.Float(c.real, 15) + sp.I * sp.Float(c.imag, 15)


##################### candidate terms #########################################

def term_shape(y, x):
    """y without its constant factors, as a sum of shapes (2*x*sin(x) + 3 -> x*sin(x) + 1)"""
    y = expand(y)
    if y.is_Add:
        return Add(*[term_shape(t, x) for t in y.args])
    if y.is_Mul:
        return Mul(*[t for t in y.args if t.has(x)])
    if y.is_Number:
        return Integer(1)
    return y


def split_terms(y, x):
    y = term_shape(y, x)
    return list(y.args) if y.is_Add else [y]


def is_holonomic(y, x):
    """closed under differentiation (up to linear combinations): polynomials times sin, cos,
    sinh, cosh and exp of polynomials"""
    if y.is_Number or y == x:
        return True
    if y.is_Function:
        return y.func in (sin, cos, sinh, cosh, exp) and y.args[0].is_polynomial(x)
    if y.is_Pow:
        return is_holonomic(y.base, x) and y.exp.is_Integer and y.exp > 0
    if y.is_Add or y.is_Mul:
        return all(is_holonomic(t, x) for t in y.args)
    return False


def closure(y, x, n=5):
    """candidates of a holonomic integrand: its terms and their repeated derivatives"""
    basis = term_shape(y, x)
    for _ in range(n):
        new = basis + Add(*[term_shape(diff(t, x), x) for t in split_terms(basis, x)])
        new = term_shape(new, x)
        if new == basis:
            break
        basis = new
    return split_terms(expand(x + basis + x * basis), x)


def ansatz_terms(y, x):
    return split_terms(x + ansatz(y, x), x)


def expand_basis(basis, x, max_ops=1000):
    """a larger candidate set: the candidates, their derivatives and x times them; None if
    the candidates have more than `max_ops` operations (multiplying out their derivatives can
    take very long; SymbolicNumericIntegration.jl has the same limit)"""
    b = Add(*basis)
    if sp.count_ops(b) > max_ops:
        return None
    return split_terms(expand((1 + x) * (b + diff(b, x))), x)


def ansatz(y, x):
    if y.is_Add:
        return Add(*[ansatz(u, x) for u in y.args])
    if y.is_Mul:
        w = 0
        for u in y.args:
            q = ansatz(u, x) if (u.is_Pow or u.is_Function) else (expand((1 + x) * u) if u.has(x) else u)
            w += (q + 1) * (y / u + 1)  # the 1s keep the other factors and the constant
        return w
    if y.is_Pow:
        return ansatz_pow(y, x)
    if y.is_Function:
        return integrate_fun(y, x)
    if y.is_polynomial(x):
        return expand((1 + x) * y)
    return y


def ansatz_pow(y, x):
    p, k = y.base, y.exp
    if not k.is_number:  # e.g. x**x
        return expand((1 + x) * y)
    if p.is_Function:
        if k > 0:
            return integrate_fun(p, x) * (1 + p ** (k - 1))
        return integrate_fun_inv(p, x) * (1 + p ** (k + 1))
    if p.is_polynomial(x):
        if k.is_Rational and k.q == 2:
            return ansatz_sqrt(p, k, x)
        if k > 0:
            return expand((1 + x) * y)
        return ansatz_poly_inv(p, k, x)
    if p == sp.E:  # exp written as a power
        return integrate_fun(exp(k), x)
    if k > 0:
        return expand((1 + x) * y)
    return y + log(1 / y)


def poly_roots(p, x):
    """the roots of the univariate polynomial p (exact if sympy finds them in a small form, else
    numerical)"""
    poly = sp.Poly(p, x)
    r = sp.roots(poly)
    if sum(r.values()) == poly.degree() and all(sp.count_ops(z) <= 20 for z in r):
        return list(r)
    zs = np.roots([complex(c) for c in poly.all_coeffs()])
    return [sp.nsimplify(complex(z), rational=False, tolerance=1e-10) for z in zs]


def ansatz_poly_inv(p, k, x):
    """candidates for p**k, p a polynomial and k a negative integer: p**k ... x**i * p**k, log(p),
    and log(x - r) / the atan and log terms of each complex-conjugate pair of roots"""
    h = p ** k + log(p)
    for i in range(int(-sp.degree(p, x) * k)):
        h += x ** i * p ** k
    if p.free_symbols != {x}:
        return h
    for r in poly_roots(p, x):
        re, im = sp.re(r), sp.im(r)
        if abs(complex(im)) < 1e-12:
            h += log(x - r)
        elif float(im) > 0:  # one of each conjugate pair
            h += atan((x - re) / im) + log(x ** 2 - 2 * re * x + re ** 2 + im ** 2)
    return h


def ansatz_sqrt(p, k, x):
    """candidates for p**k, p a polynomial and k = n/2"""
    h = p ** k + p ** (k + 1)
    if sp.degree(p, x) == 2:
        h += log(diff(p, x) / 2 + sqrt(p))
        if p.free_symbols == {x}:
            roots = poly_roots(p, x)
            if len(roots) == 2 and sp.simplify(roots[0] + roots[1]) == 0:
                a = sp.Abs(roots[0])
                h += asinh(x / a) + asin(x / a) + acosh(x / a)
    return h


def integrate_fun(y, x):
    """F(u)/u' for y = f(u(x)), where F is (a guess of the parts of) the antiderivative of f"""
    op, u = y.func, y.args[0]
    du = diff(u, x)
    if du == 0:
        return 0
    rules = {
        exp: lambda: exp(u) + Ei(u),
        log: lambda: u + u * log(u) + (ansatz_poly_inv(u, -1, x) if u.is_polynomial(x) else 0),
        sin: lambda: cos(u) + Si(u),
        cos: lambda: sin(u) + Ci(u),
        tan: lambda: log(cos(u)) + u * tan(u),
        csc: lambda: log(csc(u) + cot(u)),
        sec: lambda: log(sec(u) + tan(u)),
        cot: lambda: log(sin(u)) + u * cot(u),
        sinh: lambda: cosh(u),
        cosh: lambda: sinh(u),
        tanh: lambda: log(cosh(u)),
        csch: lambda: log(tanh(u / 2)),
        sech: lambda: atan(sinh(u)),
        coth: lambda: log(sinh(u)),
        asin: lambda: u * asin(u) + sqrt(1 - u ** 2),
        acos: lambda: u * acos(u) + sqrt(1 - u ** 2),
        atan: lambda: u * atan(u) + log(1 + u ** 2),
        acsc: lambda: u * acsc(u) + acosh(u),
        asec: lambda: u * asec(u) + acosh(u),
        acot: lambda: u * acot(u) + log(1 + u ** 2),
        asinh: lambda: u * asinh(u) + sqrt(u ** 2 + 1),
        acosh: lambda: u * acosh(u) + sqrt(u ** 2 - 1) + sqrt(u - 1) * sqrt(u + 1),
        atanh: lambda: u * atanh(u) + log(1 - u ** 2),
        acsch: lambda: u * acsch(u) + asinh(u),
        asech: lambda: u * asech(u) + asin(u),
        acoth: lambda: u * acoth(u) + log(1 - u ** 2),
    }
    rule = rules.get(op)
    if rule is None:
        return expand((1 + x) * y)
    h = rule() / du
    if op == exp and u.is_polynomial(x) and sp.degree(u, x) == 2:
        # exp(a*(x + c)**2 + k) -> erfi(sqrt(a)*(x + c)) (erf for a < 0)
        a, b, _ = sp.Poly(u, x).all_coeffs()
        h += erfi(sqrt(a) * (x + b / (2 * a)))
    return h


def integrate_fun_inv(y, x):
    """F(u)/u' for y = 1/f(u(x)), where F is (a guess of the parts of) the antiderivative of 1/f"""
    op, u = y.func, y.args[0]
    du = diff(u, x)
    if du == 0:
        return 0
    rules = {
        log: lambda: log(log(u)) + Ei(log(u)),
        sin: lambda: log(cos(u) + 1) + log(cos(u) - 1) + log(sin(u)),
        cos: lambda: log(sin(u) + 1) + log(sin(u) - 1) + log(cos(u)),
        tan: lambda: log(sin(u)) + log(tan(u)),
        csc: lambda: cos(u) + log(csc(u)),
        sec: lambda: sin(u) + log(sec(u)),
        cot: lambda: log(cos(u)) + log(cot(u)),
        sinh: lambda: log(tanh(u / 2)) + log(sinh(u)),
        cosh: lambda: atan(sinh(u)) + log(cosh(u)),
        tanh: lambda: log(sinh(u)) + log(tanh(u)),
        csch: lambda: cosh(u) + log(csch(u)),
        sech: lambda: sinh(u) + log(sech(u)),
        coth: lambda: log(cosh(u)) + log(coth(u)),
    }
    rule = rules.get(op)
    return rule() / du if rule else y + log(y)


##################### the rules of SymbolicNumericIntegration.jl ###############
#
# Ported from src/homotopy.jl, src/candidates.jl and src/rules.jl, with corrections: the
# antiderivative guesses of acsc, asech, acsch, atanh and acoth, and erfi of any quadratic
# exponent (SNI: erfi(x) only).

def homotopy_terms(eq, x):
    """candidates from partial integration (SNI `generate_homotopy`): for each factor f**k of
    a term, with the guesses y for the antiderivative of f(u) and its inner argument u
    (`partial_int_rule`), the terms of (1 + y) * (1 + rest / u'), where rest is the term without
    f**j, j = 1 ... k"""
    eq = expand(eq)
    if eq.is_Add:
        out = []
        for t in eq.args:
            out += homotopy_terms(t, x)
        return list(dict.fromkeys(out))
    factors = transform(eq, x)
    q = Mul(*[f ** k for f, k in factors])
    S = 0
    for f, k in factors:
        y, u = partial_int_rule(f, x)
        du = diff(u, x)
        du = du if du != 0 else Integer(1)
        for j in range(1, k + 1):
            h = (q / f ** j) / du
            S += expand((1 + y) * (1 + h))
    return list(dict.fromkeys([x] + split_terms(S, x)))


def transform(eq, x):
    """the x-dependent factors of a term as (factor, k): negative powers become reciprocal
    factors and rational powers n/d roots, f**(n/d) -> (f**(1/d), n) (SNI `transformer`)"""
    out = []
    for t in (eq.args if eq.is_Mul else (eq,)):
        if not t.has(x):
            continue
        if t.is_Pow and t.exp.is_number:
            b, k = t.base, t.exp
            if k.is_Float:
                k = sp.nsimplify(k, rational=True, tolerance=1e-12)
            if k.is_Integer:
                out.append((b, int(k)) if k > 0 else (1 / b, int(-k)))
                continue
            if k.is_Rational:
                n, d = int(k.p), int(k.q)
                out.append((b ** Rational(1, d), n) if n > 0 else (b ** Rational(-1, d), -n))
                continue
        out.append((t, 1))
    return out


def is_univar_poly(p, x):
    return p.has(x) and p.is_polynomial(x) and p.free_symbols == {x}


def partial_int_rule(f, x):
    """(y, u): guesses y for the antiderivative of the factor f = g(u) with respect to u, and u
    (SNI `partial_int_rules`)"""
    if f.is_Function and len(f.args) == 1:
        u = f.args[0]
        rule = PARTIAL_RULES.get(f.func)
        if rule is not None:
            return rule(u, x), u
    if f.is_Pow:
        b, k = f.base, f.exp
        if k == -1 and b.is_Function and len(b.args) == 1:  # 1/g(u)
            u = b.args[0]
            rule = RECIPROCAL_RULES.get(b.func)
            if rule is not None:
                return rule(u, x), u
        if k in (Rational(1, 2), Rational(-1, 2)):
            return Add(*sqrt_rule(b, x, k)), b
        if k.is_Integer and k < 0 and is_univar_poly(b, x):
            return Add(*pow_minus_rule(b, x, int(k))), b
        if k == -1:
            return log(b) + b * log(b), b
        if k.is_Integer and k < 0:
            return Add(*[b ** i for i in range(int(k) + 1, 0)]), b
        if k.is_Integer and k > 0:
            return Add(*[b ** (i + 1) for i in range(1, int(k) + 2)]), b
    return f + f ** 2, f


def _exp_rule(u, x):
    h = exp(u) + Ei(u)
    if u.is_polynomial(x) and sp.degree(u, x) == 2:
        a, b, _ = sp.Poly(u, x).all_coeffs()
        h += erfi(sqrt(a) * (x + b / (2 * a)))
    return h


PARTIAL_RULES = {
    sin: lambda u, x: cos(u) + Si(u),
    cos: lambda u, x: sin(u) + Ci(u),
    tan: lambda u, x: log(cos(u)),
    csc: lambda u, x: log(csc(u) + cot(u)) + log(sin(u)),
    sec: lambda u, x: log(sec(u) + tan(u)) + log(cos(u)),
    cot: lambda u, x: log(sin(u)),
    sinh: lambda u, x: cosh(u),
    cosh: lambda u, x: sinh(u),
    tanh: lambda u, x: log(cosh(u)),
    csch: lambda u, x: log(tanh(u / 2)),
    sech: lambda u, x: atan(sinh(u)),
    coth: lambda u, x: log(sinh(u)),
    asin: lambda u, x: u * asin(u) + sqrt(1 - u * u),
    acos: lambda u, x: u * acos(u) + sqrt(1 - u * u),
    atan: lambda u, x: u * atan(u) + log(u * u + 1),
    acsc: lambda u, x: u * acsc(u) + acosh(u),
    asec: lambda u, x: u * asec(u) + acosh(u),
    acot: lambda u, x: u * acot(u) + log(u * u + 1),
    asinh: lambda u, x: u * asinh(u) + sqrt(u * u + 1),
    acosh: lambda u, x: u * acosh(u) + sqrt(u * u - 1),
    atanh: lambda u, x: u * atanh(u) + log(1 - u * u),
    acsch: lambda u, x: u * acsch(u) + asinh(u),
    asech: lambda u, x: u * asech(u) + asin(u),
    acoth: lambda u, x: u * acoth(u) + log(1 - u * u),
    log: lambda u, x: u + u * log(u) + Add(*pow_minus_rule(u, x, -1)),
    exp: _exp_rule,
}

RECIPROCAL_RULES = {
    sin: lambda u, x: log(csc(u) + cot(u)) + log(sin(u)),
    cos: lambda u, x: log(sec(u) + tan(u)) + log(cos(u)),
    tan: lambda u, x: log(sin(u)) + log(tan(u)),
    csc: lambda u, x: cos(u) + log(csc(u)),
    sec: lambda u, x: sin(u) + log(sec(u)),
    cot: lambda u, x: log(cos(u)) + log(cot(u)),
    sinh: lambda u, x: log(tanh(u / 2)) + log(sinh(u)),
    cosh: lambda u, x: atan(sinh(u)) + log(cosh(u)),
    tanh: lambda u, x: log(sinh(u)) + log(tanh(u)),
    csch: lambda u, x: cosh(u) + log(csch(u)),
    sech: lambda u, x: sinh(u) + log(sech(u)),
    coth: lambda u, x: log(cosh(u)) + log(coth(u)),
    log: lambda u, x: log(log(u)) + sp.li(u),
}


def pow_minus_rule(p, x, k):
    """guesses for the antiderivative of p**k, k a negative integer: p**k, p**(k+1) and, for a
    univariate polynomial p, log(x - r) for its real roots and atan and log terms for its
    complex-conjugate pairs (SNI `pow_minus_rule`)"""
    if not is_univar_poly(p, x):
        return [p ** k, p ** (k + 1), log(p), p * log(p)]
    q = []
    for r in poly_roots(p, x):
        re, im = sp.re(r), sp.im(r)
        if abs(complex(im)) < 1e-12:
            q.append(log(x - r))
        elif float(im) > 0:
            q += [atan((x - re) / im), (1 + x) * log(x ** 2 - 2 * re * x + re ** 2 + im ** 2)]
    q = list(dict.fromkeys(q))
    return [p ** k] + q if k == -1 else [p ** k, p ** (k + 1)] + q


def sqrt_rule(p, x, k):
    """guesses for the antiderivative of p**k, k = +-1/2 (SNI `sqrt_rule`)"""
    h = [p ** k, p ** (k + 1), log(diff(p, x) / 2 + sqrt(p))]
    if is_univar_poly(p, x) and sp.degree(p, x) == 2:
        r = poly_roots(p, x)
        lead = sp.Poly(p, x).LC()
        if len(r) == 2 and all(abs(complex(z).imag) < 1e-12 for z in r):
            if abs(complex(r[0] + r[1])) < 1e-12:
                a = sp.Abs(r[0])
                h.append(acosh(x / a) if lead > 0 else asin(x / a))
        elif len(r) == 2 and abs(complex(r[0]).real) < 1e-12:
            h.append(asinh(x / sp.Abs(sp.im(r[0]))))
    return h


def is_kernel_factor(f, x):
    """linear polynomials and sin, cos, sinh, cosh and exp of them, and positive integer
    powers of these (SNI `s_rules`)"""
    if f.is_Pow and f.exp.is_Integer and f.exp > 0:
        f = f.base
    if f.is_polynomial(x) and f.has(x):
        return sp.degree(f, x) == 1
    return f.func in (sin, cos, sinh, cosh, exp) and f.args[0].is_polynomial(x) and sp.degree(f.args[0], x) == 1


def closure_terms(f, x, max_terms=50):
    """f and its repeated derivatives (their terms, up to max_terms), and x times them
    (SNI `closure`)"""
    if not f.has(x):
        return [Integer(1)]
    seen = list(dict.fromkeys(split_terms(f, x)))
    queue = list(seen)
    while queue and len(seen) < max_terms:
        y = queue.pop(0)
        for t in split_terms(diff(y, x), x):
            if t.has(x) and t not in seen:
                seen.append(t)
                queue.append(t)
    return list(dict.fromkeys(seen + [x * s for s in seen]))


def kernel_terms(eq, x):
    """candidates of SNI's second generator (`generate_basis(eq, x, true)`): for each term,
    the derivative closure of its kernel (the product of its `is_kernel_factor` factors)
    times the partial-integration candidates of the rest"""
    S = []
    for t in split_terms(expand(eq), x):
        f = Mul(*[u for u in (t.args if t.is_Mul else (t,)) if is_kernel_factor(u, x)])
        p = t / f
        C2 = homotopy_terms(p, x) if p.has(x) else [Integer(1)]
        C1 = closure_terms(f, x)
        S += [c1 * c2 for c1 in C1 for c2 in C2]
    return split_terms(Add(*S), x)

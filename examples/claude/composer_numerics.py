"""
Numerical algorithms written with `Composer`, which have no closed-form SymPy expression.

The algorithms below iterate until they converge or escape, and the number of iterations depends
on the input. They are written with `Composer` loops and compiled to a function that is applied
to whole arrays (SIMD, multi-threaded):

  newton      Newton's iteration for sqrt(a) until |x^2 - a| <= 1e-12 a
  bisection   root of x^3 - x - c on [1, 3] with a fixed number of halvings (counted loop)
  collatz     the Collatz stopping time of each integer (up to a step limit)
  mandelbrot  the escape time of each point of the Mandelbrot set (real arithmetic)
  cfrac       tan(x) from its continued fraction, evaluated until the convergent stops changing

Vectorization makes the interesting part the control flow. When the code is compiled for SIMD,
the lanes of a vector do not finish together, so a branch is taken only if *all* lanes agree
(`branch_else` jumps only if every lane is false). A loop is therefore written as

    top:   cond = <loop condition>
           branch_else(cond, done)          # leave when no lane wants to continue
           new = <the loop body computed from the current state>
           state = join(cond, new, state)   # lanes that have finished keep their values
           branch(top)
    done:

(see `while_loop` below). `use_simd=False` compiles the scalar version of the same program, and each
result is compared with a plain numpy reference and with the scalar version.

Composer is compiled with `direct=True` (the default). The `direct=False` translation does not
support backward branches.
"""

import argparse
import time

import numpy as np
from symjit import Composer, compile_composer


def while_loop(cp, cond, body, state):
    """
    cond(): a slot, true while a lane should keep iterating.
    body(): a dictionary name -> slot with the new value of each state variable.
    state: a dictionary name -> temporary slot.
    """
    top, done = cp.new_label(), cp.new_label()
    cp.set_label(top)
    c = cond()
    cp.branch_else(c, done)
    new = body()
    for name, slot in state.items():
        cp.assign(slot, cp.join(c, new[name], slot))
    cp.branch(top)
    cp.set_label(done)


def variable(cp, value):
    t = cp.new_temp()
    cp.assign(t, value)
    return t


# ------------------------------------------------------------------------------ newton


def newton_sqrt(**options):
    cp = Composer(1, 2)
    a = cp.arg(0)
    x = variable(cp, cp.constant(1.0))
    steps = variable(cp, cp.constant(0.0))

    def cond():
        err = cp.abs(cp.fsub(cp.fmul(x, x), a))
        return cp.and_(cp.gt(err, cp.fmul(cp.constant(1e-12), a)), cp.lt(steps, cp.constant(200.0)))

    def body():
        return {
            "x": cp.fmul(cp.constant(0.5), cp.fadd(x, cp.fdiv(a, x))),
            "steps": cp.fadd(steps, cp.constant(1.0)),
        }

    while_loop(cp, cond, body, {"x": x, "steps": steps})
    cp.assign(cp.out(0), x)
    cp.assign(cp.out(1), steps)
    return compile_composer(cp, **options)


def newton_sqrt_numpy(a):
    x = np.ones_like(a)
    steps = np.zeros_like(a)
    while True:
        active = (np.abs(x * x - a) > 1e-12 * a) & (steps < 200)
        if not active.any():
            return x, steps
        x = np.where(active, 0.5 * (x + a / x), x)
        steps = np.where(active, steps + 1, steps)


# ------------------------------------------------------------------------------ bisection


def bisection(iterations=60, **options):
    """Solves x^3 - x - c = 0 on [1, 3] (c in [1, 24])"""
    cp = Composer(1, 1)
    c = cp.arg(0)
    lo = variable(cp, cp.constant(1.0))
    hi = variable(cp, cp.constant(3.0))

    f = lambda b, x: b.fsub(b.fsub(b.cube(x), x), c)

    body = cp.new_block()
    mid = body.fmul(body.constant(0.5), body.fadd(lo, hi))
    below = body.lt(f(body, mid), body.constant(0.0))  # f increases on [1, 3]: the root is above mid
    body.assign(lo, body.join(below, mid, lo))
    body.assign(hi, body.join(below, hi, mid))
    cp.append_for(cp.new_temp(), 0, iterations, body)

    cp.assign(cp.out(0), cp.fmul(cp.constant(0.5), cp.fadd(lo, hi)))
    return compile_composer(cp, **options)


def bisection_numpy(c, iterations=60):
    lo, hi = np.ones_like(c), np.full_like(c, 3.0)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        below = mid**3 - mid - c < 0
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------------------ collatz


def collatz(limit=1000, **options):
    cp = Composer(1, 1)
    n = variable(cp, cp.arg(0))
    steps = variable(cp, cp.constant(0.0))

    def cond():
        return cp.and_(cp.gt(n, cp.constant(1.0)), cp.lt(steps, cp.constant(float(limit))))

    def body():
        half = cp.fmul(n, cp.constant(0.5))
        even = cp.eq(cp.floor(half), half)
        nxt = cp.join(even, half, cp.fadd(cp.fmul(n, cp.constant(3.0)), cp.constant(1.0)))
        return {"n": nxt, "steps": cp.fadd(steps, cp.constant(1.0))}

    while_loop(cp, cond, body, {"n": n, "steps": steps})
    cp.assign(cp.out(0), steps)
    return compile_composer(cp, **options)


def collatz_numpy(n0, limit=1000):
    n = n0.copy()
    steps = np.zeros_like(n)
    while True:
        active = (n > 1) & (steps < limit)
        if not active.any():
            return steps
        half = n / 2
        n = np.where(active, np.where(np.floor(half) == half, half, 3 * n + 1), n)
        steps = np.where(active, steps + 1, steps)


# ------------------------------------------------------------------------------ mandelbrot


def mandelbrot(max_iter=200, **options):
    cp = Composer(2, 1)
    cx, cy = cp.arg(0), cp.arg(1)
    zx = variable(cp, cp.constant(0.0))
    zy = variable(cp, cp.constant(0.0))
    count = variable(cp, cp.constant(0.0))

    def cond():
        r2 = cp.fadd(cp.fmul(zx, zx), cp.fmul(zy, zy))
        return cp.and_(cp.leq(r2, cp.constant(4.0)), cp.lt(count, cp.constant(float(max_iter))))

    def body():
        return {
            "zx": cp.fadd(cp.fsub(cp.fmul(zx, zx), cp.fmul(zy, zy)), cx),
            "zy": cp.fadd(cp.fmul(cp.constant(2.0), cp.fmul(zx, zy)), cy),
            "count": cp.fadd(count, cp.constant(1.0)),
        }

    while_loop(cp, cond, body, {"zx": zx, "zy": zy, "count": count})
    cp.assign(cp.out(0), count)
    return compile_composer(cp, **options)


def mandelbrot_numpy(cx, cy, max_iter=200):
    zx, zy = np.zeros_like(cx), np.zeros_like(cx)
    count = np.zeros_like(cx)
    while True:
        active = (zx * zx + zy * zy <= 4.0) & (count < max_iter)
        if not active.any():
            return count
        zx, zy = np.where(active, zx * zx - zy * zy + cx, zx), np.where(active, 2 * zx * zy + cy, zy)
        count = np.where(active, count + 1, count)


# ------------------------------------------------------------------------------ continued fraction


def cfrac_tan(max_terms=60, **options):
    """tan(x) = x / (1 - x^2 / (3 - x^2 / (5 - ...))), from the recurrences of the convergents"""
    cp = Composer(1, 2)
    x = cp.arg(0)
    x2 = cp.fmul(x, x)
    a1, a0 = variable(cp, cp.constant(0.0)), variable(cp, x)  # numerators A_{n-1}, A_n
    b1, b0 = variable(cp, cp.constant(1.0)), variable(cp, cp.constant(1.0))  # denominators B_{n-1}, B_n
    n = variable(cp, cp.constant(1.0))
    change = variable(cp, cp.constant(1.0))

    # A_1 = x, B_1 = 1 (b_1 = 1, a_1 = x); the next terms are b_k = 2k - 1, a_k = -x^2
    def cond():
        return cp.and_(cp.gt(change, cp.constant(1e-15)), cp.lt(n, cp.constant(float(max_terms))))

    def body():
        k = cp.fadd(n, cp.constant(1.0))
        bk = cp.fsub(cp.fmul(cp.constant(2.0), k), cp.constant(1.0))
        na = cp.fsub(cp.fmul(bk, a0), cp.fmul(x2, a1))
        nb = cp.fsub(cp.fmul(bk, b0), cp.fmul(x2, b1))
        rel = cp.abs(cp.fsub(cp.fdiv(na, nb), cp.fdiv(a0, b0)))
        return {"a1": a0, "a0": na, "b1": b0, "b0": nb, "n": k, "change": rel}

    while_loop(cp, cond, body, {"a1": a1, "a0": a0, "b1": b1, "b0": b0, "n": n, "change": change})
    cp.assign(cp.out(0), cp.fdiv(a0, b0))
    cp.assign(cp.out(1), n)
    return compile_composer(cp, **options)


def cfrac_tan_numpy(x, max_terms=60):
    x2 = x * x
    a1, a0 = np.zeros_like(x), x.copy()
    b1, b0 = np.ones_like(x), np.ones_like(x)
    n = np.ones_like(x)
    change = np.ones_like(x)
    while True:
        active = (change > 1e-15) & (n < max_terms)
        if not active.any():
            return a0 / b0, n
        k = n + 1
        bk = 2 * k - 1
        na, nb = bk * a0 - x2 * a1, bk * b0 - x2 * b1
        rel = np.abs(na / nb - a0 / b0)
        a1, a0 = np.where(active, a0, a1), np.where(active, na, a0)
        b1, b0 = np.where(active, b0, b1), np.where(active, nb, b0)
        n, change = np.where(active, k, n), np.where(active, rel, change)


# ------------------------------------------------------------------------------ driver

CONFIGS = {
    "default": dict(),
    "no simd": dict(use_simd=False),
    "no branch simd": dict(simd_branch=False),
    "avx512": dict(enable_simd512=True),
}


def best(fn, repeat=5):
    fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return 1e3 * min(times)


def run(name, build, make_input, reference, compare, n, timing_reference=None):
    print(f"\n{name}")
    X = make_input(n)
    want = reference(X)

    results = {}
    for label, options in CONFIGS.items():
        f = build(**options)
        got = f.evaluate(X if X.ndim == 2 else X.reshape(-1, 1))
        got = np.asarray(got)
        results[label] = got
        compare(got, want)
    for label, got in results.items():  # the SIMD and the scalar programs must give the same numbers
        np.testing.assert_array_equal(got, results["no simd"], err_msg=f"{label} differs from the scalar code")

    f = build()
    arr = X if X.ndim == 2 else X.reshape(-1, 1)
    t_sj = best(lambda: f.evaluate(arr))
    t_np = best(lambda: (timing_reference or reference)(X))
    print(f"  all {len(CONFIGS)} option sets agree with numpy and with each other; "
          f"symjit {t_sj:.2f} ms, numpy {t_np:.2f} ms ({t_np / t_sj:.1f}x)   [n = {n}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", type=int, default=200000, help="number of points")
    args = parser.parse_args()
    n = args.n
    rng = np.random.default_rng(0)

    def check_newton(got, want):
        np.testing.assert_allclose(got[:, 0], want[0], rtol=1e-13)
        np.testing.assert_array_equal(got[:, 1], want[1])

    a = np.exp(rng.uniform(np.log(1e-6), np.log(1e12), n))
    run("Newton sqrt (per-lane convergence)", newton_sqrt, lambda n: a, lambda X: newton_sqrt_numpy(X), check_newton, n)
    print(f"  iterations per point: min {int(newton_sqrt_numpy(a)[1].min())}, max {int(newton_sqrt_numpy(a)[1].max())}")

    c = rng.uniform(1.0, 24.0, n)
    run("Bisection (60 halvings, counted loop)", bisection, lambda n: c, bisection_numpy,
        lambda got, want: np.testing.assert_allclose(got.ravel(), want, rtol=1e-14), n)
    root = np.asarray(bisection().evaluate(c.reshape(-1, 1))).ravel()
    print(f"  max |x^3 - x - c| at the roots: {np.max(np.abs(root**3 - root - c)):.1e}")

    m = np.arange(1, n + 1, dtype=float)
    run("Collatz stopping time", collatz, lambda n: m, collatz_numpy,
        lambda got, want: np.testing.assert_array_equal(got.ravel(), want), n)
    steps = collatz_numpy(m)
    print(f"  longest: n = {int(m[np.argmax(steps)])} takes {int(steps.max())} steps")

    side = int(np.sqrt(n))
    X, Y = np.meshgrid(np.linspace(-2.0, 0.6, side), np.linspace(-1.2, 1.2, side))
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    run("Mandelbrot escape time (real arithmetic)", mandelbrot, lambda n: pts,
        lambda P: mandelbrot_numpy(P[:, 0], P[:, 1]), lambda got, want: np.testing.assert_array_equal(got.ravel(), want), len(pts))
    counts = mandelbrot_numpy(pts[:, 0], pts[:, 1])
    print(f"  {np.mean(counts >= 200):.1%} of the points do not escape in 200 iterations")

    t = rng.uniform(-1.4, 1.4, n)

    def check_tan(got, want):
        np.testing.assert_allclose(got[:, 0], want, rtol=1e-13)

    # accuracy is checked against np.tan; the timing is compared with the same algorithm in numpy
    run("tan(x) from its continued fraction", cfrac_tan, lambda n: t, np.tan, check_tan, n,
        timing_reference=lambda X: cfrac_tan_numpy(X))
    terms = np.asarray(cfrac_tan().evaluate(t.reshape(-1, 1)))[:, 1]
    print(f"  terms per point: min {int(terms.min())}, max {int(terms.max())} (numpy version of the algorithm: "
          f"max {int(cfrac_tan_numpy(t)[1].max())}; the stopping test depends on rounding); np.tan takes "
          f"{best(lambda: np.tan(t)):.2f} ms")


if __name__ == "__main__":
    main()

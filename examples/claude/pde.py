"""
Method-of-lines PDEs: correctness and performance of symjit on large right-hand sides.

Three uses of symjit for the semi-discretized reaction-diffusion equations

    u_t = Du lap(u) - u v^2 + F (1 - u)          (Gray-Scott model)
    v_t = Dv lap(v) + u v^2 - (F + k) v

on a periodic grid:

  1. `heat1d`: the diffusion equation on a 1D grid, written as an unrolled ODE with one
     state per grid point (thousands of states) and compiled with `compile_ode`. The
     solution is compared with the exact solution of the *discrete* problem (a Fourier mode
     decaying at a known rate), and the conservation of the total heat is checked.
  2. `Gray-Scott, unrolled`: the same idea for the 2D model on a small grid (2 n^2 states).
  3. `Gray-Scott, kernel`: the 2D model on a large grid. The right-hand side of one cell is
     compiled into a point-wise kernel that is applied to the whole grid. Two ways of
     supplying the neighbors are compared: shifted copies (`np.roll`) and ghost cells with
     shifted views of the flattened arrays (no copies, see pyhpc/stencil.py).

Checks for the 2D model: every implementation gives the same right-hand side as numpy, the
homogeneous state (u, v) = (1, 0) is an exact fixed point, and a mirror-symmetric initial
condition stays symmetric during the integration (this catches wrong neighbor indexing).
"""

import argparse
import os
import sys
import time

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from symjit import compile_func, compile_ode

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pyhpc"))
from stencil import Grid  # noqa: E402

try:  # optional baseline
    import numba

    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

DU, DV, F, K = 0.16, 0.08, 0.035, 0.065  # Gray-Scott parameters (h = 1)


# ------------------------------------------------------------------------------
# 1. the heat equation as an unrolled ODE


def heat1d(n, options, D=1.0):
    h = 1.0 / n
    u = sp.symbols(f"u0:{n}")
    t = sp.Symbol("t")
    rhs = [D * (u[(i - 1) % n] - 2 * u[i] + u[(i + 1) % n]) / h**2 for i in range(n)]

    t0 = time.perf_counter()
    f = compile_ode(t, u, rhs, **options)
    compile_ms = 1e3 * (time.perf_counter() - t0)

    x = np.arange(n) * h
    y0 = np.sin(2 * np.pi * x) + 0.5 * np.sin(6 * np.pi * x)
    T = 0.02
    sol = solve_ivp(f, (0, T), y0, method="DOP853", rtol=1e-11, atol=1e-13)

    # each Fourier mode of the periodic 3-point Laplacian decays with its own exact rate
    lam = lambda m: D * 4 / h**2 * np.sin(np.pi * m * h) ** 2
    exact = np.exp(-lam(1) * T) * np.sin(2 * np.pi * x) + 0.5 * np.exp(-lam(3) * T) * np.sin(6 * np.pi * x)
    err = np.max(np.abs(sol.y[:, -1] - exact))
    heat = abs(np.sum(sol.y[:, -1]) - np.sum(y0))

    print(f"  n = {n:5d}: compiled in {compile_ms:6.0f} ms, error against the exact discrete solution {err:.1e}, "
          f"heat conservation error {heat:.1e}, {sol.nfev} evaluations")
    assert err < 1e-9
    assert heat < 1e-8


# ------------------------------------------------------------------------------
# 2 and 3. Gray-Scott

def rhs_numpy(u, v):
    lap = lambda a: np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1) - 4 * a
    uvv = u * v * v
    return DU * lap(u) - uvv + F * (1 - u), DV * lap(v) + uvv - (F + K) * v


def cell_kernel(options):
    """The right-hand side of one cell as a function of the cell and its four neighbors"""
    u, un, us, ue, uw, v, vn, vs, ve, vw = sp.symbols("u un us ue uw v vn vs ve vw")
    uvv = u * v * v
    du = DU * (un + us + ue + uw - 4 * u) - uvv + F * (1 - u)
    dv = DV * (vn + vs + ve + vw - 4 * v) + uvv - (F + K) * v
    return compile_func([u, un, us, ue, uw, v, vn, vs, ve, vw], [du, dv], **options)


def make_rhs_roll(kernel):
    """Neighbors as shifted copies"""

    def rhs(u, v):
        du, dv = kernel(u, np.roll(u, 1, 0), np.roll(u, -1, 0), np.roll(u, -1, 1), np.roll(u, 1, 1),
                        v, np.roll(v, 1, 0), np.roll(v, -1, 0), np.roll(v, -1, 1), np.roll(v, 1, 1))
        return du, dv

    return rhs


def make_rhs_flat(kernel, n):
    """Neighbors as shifted views of flattened arrays with ghost cells"""
    g = Grid((n + 2, n + 2, 1))
    box = ((1, n + 1), (1, n + 1), (0, 1))
    s = g.sx  # the neighbor along the first axis

    def rhs(u, v):
        up, vp = g.flat(np.pad(u, 1, mode="wrap")), g.flat(np.pad(v, 1, mode="wrap"))
        du, dv = g.apply(kernel, [(up, 0), (up, -s), (up, s), (up, 1), (up, -1),
                                  (vp, 0), (vp, -s), (vp, s), (vp, 1), (vp, -1)], box)
        return du[:, :, 0], dv[:, :, 0]

    return rhs


def make_rhs_numba():
    @numba.njit(fastmath=True, cache=False)
    def rhs_loop(u, v):
        n = u.shape[0]
        du = np.empty_like(u)
        dv = np.empty_like(v)
        for i in range(n):
            for j in range(n):
                ip, im, jp, jm = (i + 1) % n, (i - 1) % n, (j + 1) % n, (j - 1) % n
                uc, vc = u[i, j], v[i, j]
                uvv = uc * vc * vc
                du[i, j] = DU * (u[ip, j] + u[im, j] + u[i, jp] + u[i, jm] - 4 * uc) - uvv + F * (1 - uc)
                dv[i, j] = DV * (v[ip, j] + v[im, j] + v[i, jp] + v[i, jm] - 4 * vc) + uvv - (F + K) * vc
        return du, dv

    return rhs_loop


def initial_condition(n):
    """Mirror-symmetric (x -> -x, y -> -y, x <-> y) seed in the center of the grid"""
    u = np.ones((n, n))
    v = np.zeros((n, n))
    c = n // 2
    r = max(2, n // 10)
    for arr, val in ((u, 0.5), (v, 0.25)):
        arr[c - r : c + r + 1, c - r : c + r + 1] = val
    u[c - r // 2 : c + r // 2 + 1, c - r // 2 : c + r // 2 + 1] = 0.3
    return u, v


def unrolled_ode(n, options):
    """The 2D model with one state per grid cell (2 n^2 states)"""
    u = np.array(sp.symbols(f"u0:{n * n}")).reshape(n, n)
    v = np.array(sp.symbols(f"v0:{n * n}")).reshape(n, n)
    t = sp.Symbol("t")
    du, dv = [], []
    for i in range(n):
        for j in range(n):
            nb = lambda a: a[(i + 1) % n, j] + a[(i - 1) % n, j] + a[i, (j + 1) % n] + a[i, (j - 1) % n]
            uvv = u[i, j] * v[i, j] ** 2
            du.append(DU * (nb(u) - 4 * u[i, j]) - uvv + F * (1 - u[i, j]))
            dv.append(DV * (nb(v) - 4 * v[i, j]) + uvv - (F + K) * v[i, j])

    t0 = time.perf_counter()
    f = compile_ode(t, list(u.ravel()) + list(v.ravel()), du + dv, **options)
    return f, 1e3 * (time.perf_counter() - t0)


def check_gray_scott(n, options):
    kernel = cell_kernel(options)
    implementations = {"roll": make_rhs_roll(kernel), "flat": make_rhs_flat(kernel, n)}
    if HAVE_NUMBA:
        implementations["numba"] = make_rhs_numba()

    rng = np.random.default_rng(n)
    u, v = rng.uniform(0, 1, (n, n)), rng.uniform(0, 0.5, (n, n))
    want = rhs_numpy(u, v)
    for name, rhs in implementations.items():
        got = rhs(u.copy(), v.copy())
        for w, g in zip(want, got):
            np.testing.assert_allclose(g, w, rtol=1e-12, atol=1e-13, err_msg=name)

        # the homogeneous state is a fixed point (up to round-off: the kernel is free to
        # distribute the constants over the sums)
        z = rhs(np.ones((n, n)), np.zeros((n, n)))
        assert max(np.max(np.abs(z[0])), np.max(np.abs(z[1]))) < 1e-15, name

    # integrate with RK4 and compare the trajectories; the symmetric seed must stay symmetric
    u0, v0 = initial_condition(n)
    dt, steps = 1.0, 300

    def integrate(rhs):
        u, v = u0.copy(), v0.copy()
        for _ in range(steps):
            k1 = rhs(u, v)
            k2 = rhs(u + 0.5 * dt * k1[0], v + 0.5 * dt * k1[1])
            k3 = rhs(u + 0.5 * dt * k2[0], v + 0.5 * dt * k2[1])
            k4 = rhs(u + dt * k3[0], v + dt * k3[1])
            u = u + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            v = v + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        return u, v

    ref = integrate(rhs_numpy)
    print(f"  n = {n}: pattern u in [{ref[0].min():.3f}, {ref[0].max():.3f}] after {steps} RK4 steps")
    for name, rhs in implementations.items():
        u, v = integrate(rhs)
        diff = max(np.max(np.abs(u - ref[0])), np.max(np.abs(v - ref[1])))
        # the grid is symmetric about the cell n // 2 (indices i and n - i)
        us = np.roll(u[::-1, ::-1], 1, axis=(0, 1))
        sym = max(np.max(np.abs(u - us)), np.max(np.abs(u - u.T)))
        print(f"    {name:5s}: difference from numpy {diff:.1e}, asymmetry {sym:.1e}")
        assert diff < 1e-9 and sym < 1e-9


def check_unrolled(n, options):
    f, compile_ms = unrolled_ode(n, options)
    rng = np.random.default_rng(n)
    u, v = rng.uniform(0, 1, (n, n)), rng.uniform(0, 0.5, (n, n))
    y = np.concatenate([u.ravel(), v.ravel()])
    got = np.asarray(f(0.0, y))
    du, dv = rhs_numpy(u, v)
    want = np.concatenate([du.ravel(), dv.ravel()])
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13)

    t0 = time.perf_counter()
    for _ in range(200):
        f(0.0, y)
    per_call = 1e6 * (time.perf_counter() - t0) / 200
    t0 = time.perf_counter()
    for _ in range(200):
        rhs_numpy(u, v)
    per_np = 1e6 * (time.perf_counter() - t0) / 200
    print(f"  n = {n:3d} ({2 * n * n:5d} states): compiled in {compile_ms:6.0f} ms, agrees with numpy; "
          f"one evaluation {per_call:.0f} us (numpy: {per_np:.0f} us)")


def benchmark(sizes, options, repeat=20):
    kernel = cell_kernel(options)
    print(f"  {'n':>6s} {'numpy':>10s} {'symjit roll':>12s} {'symjit flat':>12s}" + (f" {'numba':>10s}" if HAVE_NUMBA else "")
          + "   (ms per right-hand side evaluation)")
    numba_rhs = make_rhs_numba() if HAVE_NUMBA else None

    for n in sizes:
        rng = np.random.default_rng(0)
        u, v = rng.uniform(0, 1, (n, n)), rng.uniform(0, 0.5, (n, n))
        rhs = {"numpy": rhs_numpy, "roll": make_rhs_roll(kernel), "flat": make_rhs_flat(kernel, n)}
        if HAVE_NUMBA:
            rhs["numba"] = numba_rhs

        row = []
        for name, fn in rhs.items():
            fn(u, v)
            reps = max(3, min(repeat, int(0.5 / max(1e-4, 1e-8 * n * n))))
            t0 = time.perf_counter()
            for _ in range(reps):
                fn(u, v)
            row.append(1e3 * (time.perf_counter() - t0) / reps)
        print(f"  {n:6d}" + "".join(f" {t:10.2f}" if i != 1 and i != 2 else f" {t:12.2f}" for i, t in enumerate(row)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-s", "--size", type=int, action="append", help="grid size n of the benchmark (repeatable)")
    parser.add_argument("--simd512", action="store_true")
    parser.add_argument("--no-threads", action="store_true")
    parser.add_argument("--opt-level", type=int, default=2)
    args = parser.parse_args()

    options = dict(enable_simd512=args.simd512, use_threads=not args.no_threads, opt_level=args.opt_level)
    print(f"symjit options: {options}; numba {'available' if HAVE_NUMBA else 'not installed'}\n")

    print("1D heat equation, one state per grid point")
    for n in (100, 400, 800):
        heat1d(n, options)

    print("\nGray-Scott, unrolled into 2 n^2 states")
    for n in (8, 16, 32):
        check_unrolled(n, options)

    print("\nGray-Scott, point-wise kernel on the whole grid")
    for n in (32, 128):
        check_gray_scott(n, options)

    print("\nperformance")
    benchmark(args.size or [64, 128, 256, 512, 1024, 2048], options)


if __name__ == "__main__":
    main()

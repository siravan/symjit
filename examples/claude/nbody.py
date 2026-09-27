"""
Gravitational N-body problem: correctness and performance of symjit on a force kernel.

The accelerations of N softened point masses,

    a_i = sum_{j != i} G m_j (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2),

are written as an unrolled symbolic sum over all pairs. The pair distances are shared
between the two bodies of a pair, so the expressions are full of common subexpressions,
which is a good stress test of the code generator (register allocation, spilling, and
common-subexpression elimination). The masses are parameters of the compiled function.

Checks:
  1. the compiled accelerations agree with a vectorized numpy reference on random systems;
  2. Newton's third law: sum_i m_i a_i = 0 (up to round-off) for every system;
  3. the figure-eight three-body orbit (Chenciner and Montgomery, 2000) returns to its
     initial state after one period, integrated with `compile_ode` and DOP853;
  4. a 16-body cluster integrated with velocity Verlet conserves energy and momentum;
  5. performance: accelerations for many independent systems (numpy reference vs symjit),
     for several N.
"""

import argparse
import time

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from symjit import compile_func, compile_ode


def symbols(N):
    pos = [sp.symbols(f"x{i} y{i} z{i}") for i in range(N)]
    vel = [sp.symbols(f"vx{i} vy{i} vz{i}") for i in range(N)]
    mass = list(sp.symbols(f"m0:{N}"))
    return pos, vel, mass


def accelerations(pos, mass, eps2):
    """Symbolic accelerations of all bodies (G = 1); each pair is computed once"""
    N = len(pos)
    terms = [[[] for _ in range(3)] for _ in range(N)]  # summing at once is much faster in sympy

    for i in range(N):
        for j in range(i + 1, N):
            d = [pos[j][k] - pos[i][k] for k in range(3)]
            r2 = d[0] ** 2 + d[1] ** 2 + d[2] ** 2 + eps2
            w = r2 ** sp.Rational(-3, 2)
            for k in range(3):
                terms[i][k].append(mass[j] * w * d[k])
                terms[j][k].append(-mass[i] * w * d[k])

    return [sp.Add(*terms[i][k]) for i in range(N) for k in range(3)]


def acceleration_numpy(x, m, eps2):
    """x: (M, N, 3) positions of M systems, m: (N,) masses; returns (M, N, 3)"""
    d = x[:, None, :, :] - x[:, :, None, :]  # d[s, i, j] = r_j - r_i
    r2 = np.sum(d * d, axis=-1) + eps2
    np.fill_diagonal(r2[0], 1.0)  # avoid 0 ** -1.5 on the diagonal (the terms are zero anyway)
    w = r2 ** -1.5
    w[:, np.arange(x.shape[1]), np.arange(x.shape[1])] = 0.0
    return np.einsum("sij,j,sijk->sik", w, m, d)


def random_systems(M, N, rng):
    x = rng.standard_normal((M, N, 3))
    m = rng.uniform(0.5, 2.0, N)
    return x, m


def compile_accelerations(N, eps2, timing=None, **options):
    pos, _, mass = symbols(N)
    flat = [c for body in pos for c in body]
    t0 = time.perf_counter()
    exprs = accelerations(pos, mass, eps2)
    t1 = time.perf_counter()
    f = compile_func(flat, exprs, params=mass, **options)
    if timing is not None:
        timing["build"], timing["compile"] = 1e3 * (t1 - t0), 1e3 * (time.perf_counter() - t1)
    return f


def total_energy(x, v, m, eps2):
    kinetic = 0.5 * np.sum(m[:, None] * v**2)
    d = x[:, None, :] - x[None, :, :]
    r = np.sqrt(np.sum(d * d, axis=-1) + eps2)
    iu = np.triu_indices(len(m), 1)
    potential = -np.sum((m[:, None] * m[None, :] / r)[iu])
    return kinetic + potential


def check_accelerations(N, eps2, options):
    rng = np.random.default_rng(N)
    f = compile_accelerations(N, eps2, **options)
    x, m = random_systems(1000, N, rng)

    got = np.array(f(*x.reshape(len(x), -1).T, *m)).T.reshape(x.shape)
    want = acceleration_numpy(x, m, eps2)
    err = np.max(np.abs(got - want)) / np.max(np.abs(want))
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10 * np.max(np.abs(want)))

    # Newton's third law: the total force vanishes
    net = np.max(np.abs(np.einsum("sik,i->sk", got, m)))
    scale = np.max(np.abs(got * m[None, :, None]))
    assert net < 1e-12 * scale * N, f"net force {net:.2e}"
    print(f"  N = {N:2d}: max relative error {err:.1e}, net force / typical force {net / scale:.1e}")


def figure_eight(options):
    p1 = np.array([-0.97000436, 0.24308753, 0.0])
    v3 = np.array([-0.93240737, -0.86473146, 0.0])  # velocity of body 3 (2 * v1 with the opposite sign)
    x0 = np.array([p1, [0.0, 0.0, 0.0], -p1])
    v0 = np.array([-v3 / 2, v3, -v3 / 2])
    period = 6.32591398

    pos, vel, mass = symbols(3)
    t = sp.Symbol("t")
    states = [c for body in pos for c in body] + [c for body in vel for c in body]
    acc = accelerations(pos, mass, 0)
    rhs = [c for body in vel for c in body] + acc
    f = compile_ode(t, states, rhs, params=mass, **options)

    y0 = np.concatenate([x0.ravel(), v0.ravel()])
    sol = solve_ivp(f, (0, period), y0, method="DOP853", rtol=1e-12, atol=1e-12, args=(1.0, 1.0, 1.0))
    err = np.max(np.abs(sol.y[:, -1] - y0))
    print(f"  figure-eight: state after one period differs from the initial state by {err:.1e} "
          f"({sol.nfev} right-hand side evaluations)")
    assert err < 1e-6  # the published initial conditions have 8 digits


def verlet_cluster(options, N=16, eps2=0.1**2, steps=2000, dt=5e-4):
    rng = np.random.default_rng(1)
    x0 = rng.standard_normal((N, 3))
    v0 = 0.3 * rng.standard_normal((N, 3))
    m = rng.uniform(0.5, 2.0, N)
    v0 -= np.sum(m[:, None] * v0, axis=0) / m.sum()  # zero total momentum

    f = compile_accelerations(N, eps2, **options)

    def a_symjit(x):
        return np.array(f(*x.ravel(), *m)).reshape(N, 3)

    def a_numpy(x):
        return acceleration_numpy(x[None], m, eps2)[0]

    def integrate(a):
        x, v = x0.copy(), v0.copy()
        e0 = total_energy(x, v, m, eps2)
        acc = a(x)
        worst = 0.0
        t0 = time.perf_counter()
        for _ in range(steps):
            v += 0.5 * dt * acc
            x += dt * v
            acc = a(x)
            v += 0.5 * dt * acc
            worst = max(worst, abs(total_energy(x, v, m, eps2) - e0) / abs(e0))
        return x, v, worst, time.perf_counter() - t0

    x, v, worst, elapsed = integrate(a_symjit)
    xn, vn, _, _ = integrate(a_numpy)

    momentum = np.max(np.abs(np.sum(m[:, None] * v, axis=0)))
    diff = np.max(np.abs(x - xn))
    print(f"  {N}-body cluster, {steps} velocity-Verlet steps: max relative energy error {worst:.1e}, "
          f"|total momentum| {momentum:.1e}, difference from the numpy-force trajectory {diff:.1e} "
          f"({1e6 * elapsed / steps:.0f} us/step)")
    assert worst < 1e-3  # a property of the integrator (second order) and the softening
    assert momentum < 1e-10  # Newton's third law holds to round-off in the compiled forces
    assert diff < 1e-8  # the forces agree to 1e-16, so the trajectories stay together


def benchmark(sizes, M, eps2, options, repeat=5):
    print(f"\n  accelerations of M = {M} independent systems")
    print(f"  {'N':>4s} {'sympy (ms)':>11s} {'compile (ms)':>13s} {'numpy (ms)':>12s} {'symjit (ms)':>12s} {'speedup':>8s}"
          f" {'ns/pair (symjit)':>17s}")
    rng = np.random.default_rng(0)

    for N in sizes:
        timing = {}
        f = compile_accelerations(N, eps2, timing=timing, **options)

        x, m = random_systems(M, N, rng)
        cols = list(x.reshape(M, -1).T.copy())

        def best(fn):
            fn()
            times = []
            for _ in range(repeat):
                t0 = time.perf_counter()
                fn()
                times.append(time.perf_counter() - t0)
            return 1e3 * min(times)

        t_np = best(lambda: acceleration_numpy(x, m, eps2))
        t_sj = best(lambda: f(*cols, *m))
        pairs = M * N * (N - 1) / 2
        print(f"  {N:4d} {timing['build']:11.0f} {timing['compile']:13.0f} {t_np:12.2f} {t_sj:12.2f} {t_np / t_sj:7.1f}x {1e6 * t_sj / pairs:17.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-M", type=int, default=20000, help="number of systems in the benchmark")
    parser.add_argument("-N", type=int, action="append", help="number of bodies (repeatable; default 4 8 16 32)")
    parser.add_argument("--simd512", action="store_true")
    parser.add_argument("--no-threads", action="store_true")
    parser.add_argument("--opt-level", type=int, default=2)
    args = parser.parse_args()

    options = dict(enable_simd512=args.simd512, use_threads=not args.no_threads, opt_level=args.opt_level)
    sizes = args.N or [4, 8, 16, 32]
    print(f"symjit options: {options}\n")

    print("accelerations against the numpy reference")
    for N in sizes:
        check_accelerations(N, 0.01, options)
    print("integrations")
    figure_eight(options)
    verlet_cluster(options)
    benchmark(sizes, args.M, 0.01, options)


if __name__ == "__main__":
    main()

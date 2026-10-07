"""A simplified examples/pendulum.py: checks symjit's mass matrix and forcing vector of an
n-link pendulum against sympy's lambdify.

examples/pendulum.py builds the equations of motion of an n-link pendulum (Kane's method),
compiles the mass matrix MM and the forcing vector with symjit, integrates them for 50 s and
reports the drift of the total energy. Here the same MM and forcing are compared directly with
lambdify at random states (the integration of a chaotic system amplifies any difference), and
a short integration reports the energy drift and the final state.

    python pendulum_check.py                 # n = 5 links
    python pendulum_check.py -n 25           # the size of examples/pendulum.py
    python pendulum_check.py --opt_level 0   # symjit options as in examples/util.py
"""

import argparse
import os
import sys

import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from scipy.integrate import solve_ivp

sys.path.insert(0, os.environ.get("SYMJIT_PYTHON_PATH", os.path.join(os.path.dirname(__file__), "..", "..", "python")))

from symjit import compile_func  # noqa: E402


def equations(n):
    """MM (2n x 2n) and forcing (2n) of the n-link pendulum, as in examples/pendulum.py"""
    m, g, iZZ, l, reibung = sm.symbols("m, g, iZZ, l, reibung")
    q = me.dynamicsymbols(f"q:{n}")
    u = me.dynamicsymbols(f"u:{n}")
    t = me.dynamicsymbols._t
    A = sm.symbols(f"A:{n}", cls=me.ReferenceFrame)
    Dmc = sm.symbols(f"Dmc:{n}", cls=me.Point)
    P = sm.symbols(f"P:{n}", cls=me.Point)
    O = me.ReferenceFrame("O")
    PO = me.Point("PO")
    PO.set_vel(O, 0)
    l1, l2 = l / n, l / (2 * n)
    for i in range(n):
        A[i].orient_axis(O, q[i], O.z)
        A[i].set_ang_vel(O, u[i] * O.z)
        base = PO if i == 0 else P[i - 1]
        Dmc[i].set_pos(base, l2 * A[i].x)
        Dmc[i].v2pt_theory(base, O, A[i])
        P[i].set_pos(base, l1 * A[i].x)
        P[i].v2pt_theory(base, O, A[i])
    bodies = [me.RigidBody(f"body{i}", Dmc[i], A[i], m, (me.inertia(A[i], 0.0, 0.0, iZZ), Dmc[i])) for i in range(n)]
    loads = [(Dmc[i], -m * g * O.y) for i in range(n)] + [(A[i], -u[i] * reibung * A[i].z) for i in range(n)]
    KM = me.KanesMethod(O, q_ind=q, u_ind=u, kd_eqs=[u[i] - q[i].diff(t) for i in range(n)])
    KM.kanes_equations(bodies, loads)
    w = sm.symbols(f"w:{n}")
    v = sm.symbols(f"v:{n}")
    subs_w = {q[i]: w[i] for i in range(n)}
    subs_v = {u[i]: v[i] for i in range(n)}
    MM = me.msubs(KM.mass_matrix_full, subs_w, subs_v)
    force = me.msubs(KM.forcing_full, subs_w, subs_v)
    kin = me.msubs(sum(b.kinetic_energy(O) for b in bodies), subs_w, subs_v)
    pot = me.msubs(sum(m * g * me.dot(p.pos_from(PO), O.y) for p in Dmc), subs_w, subs_v)
    return (*w, *v), (m, g, l, iZZ, reibung), MM, force, kin + pot


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", type=int, default=5, help="number of links (default 5)")
    parser.add_argument("--opt_level", type=int, default=2)
    parser.add_argument("--no-simd", action="store_true")
    parser.add_argument("--no-fastmath", action="store_true")
    parser.add_argument("--points", type=int, default=20, help="random states compared (default 20)")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds integrated (default 5)")
    args = parser.parse_args()
    n = args.n
    opts = dict(opt_level=args.opt_level, use_simd=not args.no_simd, fastmath=not args.no_fastmath)

    states, params, MM, force, energy = equations(n)
    MM_list = [MM[i, j] for i in range(MM.shape[0]) for j in range(MM.shape[1])]
    MM_jit = compile_func(states, MM_list, params=params, **opts)
    force_jit = compile_func(states, list(force), params=params, **opts)
    MM_lam = sm.lambdify(list(states) + list(params), MM, cse=True)
    force_lam = sm.lambdify(list(states) + list(params), force, cse=True)
    energy_lam = sm.lambdify(list(states) + list(params), energy, cse=True)

    l1, m1, g1 = 20.0, 1.0, 9.8
    p = [m1, g1, l1, m1 * (l1 / n) ** 2 / 12, 0.0]

    # 1. MM and forcing at random states
    rng = np.random.default_rng(0)
    err_mm = err_f = 0.0
    for _ in range(args.points):
        y = np.concatenate([rng.uniform(0, 2 * np.pi, n), rng.normal(0, 1, n)])
        mm_ref = np.asarray(MM_lam(*y, *p), dtype=float)
        f_ref = np.asarray(force_lam(*y, *p), dtype=float).ravel()
        mm = MM_jit.apply(y, p).reshape(2 * n, 2 * n)
        f = force_jit.apply(y, p)
        err_mm = max(err_mm, np.max(np.abs(mm - mm_ref)) / np.max(np.abs(mm_ref)))
        err_f = max(err_f, np.max(np.abs(f - f_ref)) / max(np.max(np.abs(f_ref)), 1e-300))
    print(f"n = {n}, {opts}")
    print(f"MM       max relative difference to lambdify: {err_mm:.2e}")
    print(f"forcing  max relative difference to lambdify: {err_f:.2e}")

    # 2. a short integration (as examples/pendulum.py)
    y0 = [3.0 * np.pi / 2.0 + np.pi * i / n for i in range(1, n + 1)] + [0.0] * n

    def gradient(t, y):
        return np.linalg.solve(MM_jit.apply(y, p).reshape(2 * n, 2 * n), force_jit.apply(y, p))

    sol = solve_ivp(gradient, (0.0, args.interval), y0, method="Radau", rtol=1e-8, atol=1e-10)
    e = [energy_lam(*sol.y[:, k], *p) for k in range(sol.y.shape[1])]
    drift = (max(e) - min(e)) / max(abs(v) for v in e)
    print(f"integration: {sol.message} nfev = {sol.nfev}, relative energy drift {drift:.3e}")
    print("final q[0:3] =", " ".join(f"{v:.12f}" for v in sol.y[:3, -1]))
    ok = err_mm < 1e-12 and err_f < 1e-10
    print("OK" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

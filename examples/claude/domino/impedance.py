# This file is generated using claude code
"""The impedance of an RLC grid at any frequency: resistors.py with complex numbers.

Kirchhoff's formula for the resistance between the nodes a and b of a network is algebraic, so
it holds as well for complex edge impedances z_e(omega) (R, i omega L, 1/(i omega C) or their
sums):

    Z_ab = sum over the 2-forests F separating a and b of  prod_{e not in F} z_e
           --------------------------------------------------------------
           sum over the spanning trees T              of  prod_{e not in T} z_e

This example compiles Z_ab for an m x n grid with dtype="complex128" (4 x 4, opposite corners:
100,352 spanning trees and 186,368 2-forests) and checks it at many frequencies:

1. Random RLC networks (R + i omega L + 1/(i omega C) on every edge) against complex nodal
   analysis (numpy).
2. An anisotropic LC grid, inductors L on the horizontal edges and capacitors C on the vertical
   ones: the grid's Laplacian still separates, with the eigenvalues
   lambda_jk = y_v (2 - 2 cos(pi j/m)) + y_h (2 - 2 cos(pi k/n)), y = 1/z, and the cosine
   eigenvectors psi_jk of resistors.py, so Z_ab = sum (psi_jk(a) - psi_jk(b))^2 / lambda_jk exactly.
   Near omega = 1/sqrt(L C), where z_v = -z_h, the terms of Kirchhoff's sums cancel (by 1e8 on the
   4 x 4 grid): the error is compared with the condition number of the sums.
3. Its resonances: Z_ab has poles where lambda_jk = 0, at omega^2 = (2 - 2 cos(pi k/n)) /
   (L C (2 - 2 cos(pi j/m))), unless psi_jk(a) = psi_jk(b). They are located by bisection on the
   compiled function and compared with the formula.
4. Foster's reactance theorem, on random lossless networks (each edge an inductor or a capacitor):
   Z(i omega) is purely imaginary, its reactance increases with omega between the poles, and the
   poles and zeros alternate.
5. Passivity: Re Z >= 0 for random RLC networks.
6. A vectorized frequency sweep (complex SIMD) against batched numpy solves.

    python impedance.py               # a 4 x 4 grid, between opposite corners
    python impedance.py -m 3 -n 4     # other grids
    python impedance.py --no-simd     # symjit options as in the other examples (util.py)
"""

import argparse
import sys
import time

import util

opts = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False)
opts.add_argument("-m", type=int, default=4, help="rows of nodes (default 4)")
opts.add_argument("-n", type=int, default=4, help="columns of nodes (default 4)")
opts.add_argument("--points", type=int, default=20000, help="frequencies of the vectorized sweep (default 20000)")
opts.add_argument("--force", action="store_true", help="allow grids with more than a million spanning trees")
our, rest = opts.parse_known_args()
if "-h" in rest or "--help" in rest:
    opts.print_help()
    sys.exit(0)
sys.argv = sys.argv[:1] + rest
args = util.process_argv()
args["dtype"] = "complex128"

import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402

m, n = our.m, our.n
node = lambda i, j: i * n + j  # noqa: E731
a, b = node(0, 0), node(m - 1, n - 1)
V = m * n
horizontal = [(node(i, j), node(i, j + 1)) for i in range(m) for j in range(n - 1)]
vertical = [(node(i, j), node(i + 1, j)) for i in range(m - 1) for j in range(n)]
edges = horizontal + vertical
H = len(horizontal)
z = [sp.Symbol(f"z{k}") for k in range(len(edges))]


# ------------------------------------------------------------------ the grid's eigenvectors
def path_eigen(p):
    """the eigenvalues 2 - 2 cos(pi j/p) and normalized eigenvectors of a path's Laplacian"""
    lam = np.array([2 - 2 * np.cos(np.pi * j / p) for j in range(p)])
    vec = [np.cos(np.pi * j * (np.arange(p) + 0.5) / p) for j in range(p)]
    return lam, [v / np.linalg.norm(v) for v in vec]


lm, um = path_eigen(m)  # along the columns (vertical edges)
ln, un = path_eigen(n)  # along the rows (horizontal edges)
modes = [(j, k, np.outer(um[j], un[k]).ravel()) for j in range(m) for k in range(n) if j or k]
weight = {(j, k): (psi[a] - psi[b]) ** 2 for j, k, psi in modes}  # the residue factors


def wu_anisotropic(yh, yv):
    """Z_ab of the grid with the admittance yh on every horizontal and yv on every vertical edge"""
    return sum(weight[j, k] / (yv * lm[j] + yh * ln[k]) for j, k, _ in modes)


tree_count = round(np.prod([lm[j] + ln[k] for j, k, _ in modes]) / V)
if tree_count > 1_000_000 and not our.force:
    sys.exit(f"the {m} x {n} grid has {tree_count:,} spanning trees; use --force to build an expression this large")


# ------------------------------------------------------------------ Kirchhoff's formula
def spanning_trees(num_nodes, edge_list):
    """all spanning trees (as sets of edge indices) of a multigraph (as in resistors.py)"""
    need = num_nodes - 1
    out = []
    comp = list(range(num_nodes))

    def walk(k, chosen):
        if len(chosen) == need:
            out.append(frozenset(chosen))
            return
        if len(edge_list) - k < need - len(chosen):
            return
        u, v = edge_list[k]
        cu, cv = comp[u], comp[v]
        if cu != cv:
            saved = list(comp)
            for i in range(num_nodes):
                if comp[i] == cv:
                    comp[i] = cu
            chosen.append(k)
            walk(k + 1, chosen)
            chosen.pop()
            comp[:] = saved
        walk(k + 1, chosen)

    walk(0, [])
    return out


def merged(u):
    return a if u == b else u


def complement_sum(subgraphs):
    return sp.Add(*[sp.Mul(*[z[e] for e in range(len(edges)) if e not in s]) for s in subgraphs])


def build():
    trees = spanning_trees(V, edges)
    relabel = {u: k for k, u in enumerate(sorted({merged(u) for u in range(V)}))}
    forests = spanning_trees(V - 1, [(relabel[merged(u)], relabel[merged(v)]) for u, v in edges])
    num, den = complement_sum(forests), complement_sum(trees)
    # the two sums as a second function, for their condition numbers: sum |terms| / |sum| is the
    # sum evaluated with |z_e|, divided by |sum| (compiled with Z, they would not be shared)
    return [(z, [num / den]), (z, [num, den])], dict(trees=len(trees), forests=len(forests))


# compile_cached caches the expressions' JSON models when SYMJIT_EXAMPLE_CACHE is set (run_physics.py)
(f, sums), info, t_build, t_compile, cached = util.compile_cached("impedance", (m, n), build, **args)
print(f"{m} x {n} grid, {len(edges)} impedances, Z between opposite corners: {info['trees']:,} spanning trees, "
      f"{info['forests']:,} 2-forests")
print(f"enumeration and sympy {t_build:.1f} s{' (cached)' if cached else ''}, symjit compilation (complex128) "
      f"{t_compile:.1f} s, {f.measure('ker-scalar-size') / 2**20:.2f} MiB of machine code\n")

ok = True


def report(name, passed, detail):
    global ok
    ok &= passed
    print(f"{'ok  ' if passed else 'FAIL'} {name}: {detail}")


def nodal(zs):
    """Z_ab by complex nodal analysis; zs: len(edges) impedances, or len(edges) x N"""
    zs = np.asarray(zs, dtype=complex)
    Y = np.zeros(zs.shape[1:] + (V, V), dtype=complex)
    for k, (u, v) in enumerate(edges):
        y = 1 / zs[k]
        Y[..., u, u] += y
        Y[..., v, v] += y
        Y[..., u, v] -= y
        Y[..., v, u] -= y
    keep = [i for i in range(V) if i != b]
    Yg = Y[..., keep, :][..., :, keep]
    rhs = np.zeros(Yg.shape[:-1], dtype=complex)
    rhs[..., keep.index(a)] = 1
    return np.linalg.solve(Yg, rhs[..., None])[..., keep.index(a), 0]


def Zf(zs):
    """the compiled Z for one set of impedances (scalar) or len(edges) x N (vectorized)"""
    zs = np.asarray(zs, dtype=complex)
    return np.asarray(f(*zs)[0]).ravel() if zs.ndim == 2 else complex(np.ravel(f(*zs))[0])


rng = np.random.default_rng(1)

# 1. random RLC networks at random frequencies
worst = 0.0
for _ in range(30):
    R, L, C, w = rng.uniform(0.2, 2, len(edges)), rng.uniform(0.2, 2, len(edges)), rng.uniform(0.2, 2, len(edges)), \
        rng.uniform(0.1, 10)
    zs = R + 1j * w * L + 1 / (1j * w * C)
    worst = max(worst, abs(Zf(zs) / nodal(zs) - 1))
report("random RLC", worst < 1e-11, f"max relative difference to complex nodal analysis over 30 networks {worst:.1e}")

# 2. the anisotropic LC grid against the eigenvalue formula. Near omega = 1/sqrt(L C), z_v = -z_h
# and the terms of both sums cancel: the condition number kappa = sum |terms| / |sum| reaches 1e8
# and any floating-point evaluation of the sums loses digits (nodal analysis does not); the
# error is compared with 1e-14 kappa.
def condition(zs):
    zs = np.asarray(zs, dtype=complex)
    s_abs = np.abs(np.ravel(sums(*np.abs(zs).astype(complex))))
    s = np.abs(np.ravel(sums(*zs)))
    return max(s_abs[0] / s[0], s_abs[1] / s[1])


Lh, Cv = 1.5, 0.8
ws = np.geomspace(0.05, 20, 400)
worst, kappa_max = 0.0, 0.0
for w in ws:
    zh, zv = 1j * w * Lh, 1 / (1j * w * Cv)
    zs = [zh] * H + [zv] * (len(edges) - H)
    exact = wu_anisotropic(1 / zh, 1 / zv)
    if abs(exact) < 1e6:  # not at a pole
        kappa = condition(zs)
        kappa_max = max(kappa_max, kappa)
        worst = max(worst, abs(Zf(zs) - exact) / abs(exact) / (1e-14 * kappa + 1e-12))
report("LC grid", worst < 1, f"against the eigenvalue formula at 400 frequencies: the largest error is {worst:.1e} "
       f"times 1e-14 kappa; the largest condition number {kappa_max:.1e} (at omega near 1/sqrt(L C) = "
       f"{1 / np.sqrt(Lh * Cv):.4f})")

# 3. the resonances of the LC grid
predicted = {}
for j, k, _ in modes:
    if j and k:
        wjk = np.sqrt(ln[k] / (Lh * Cv * lm[j]))
        predicted.setdefault(round(wjk, 9), 0.0)
        predicted[round(wjk, 9)] += weight[j, k]
poles = sorted(w for w, res in predicted.items() if res > 1e-12)


def reactance(w):
    return Zf([1j * w * Lh] * H + [1 / (1j * w * Cv)] * (len(edges) - H)).imag


grid = np.geomspace(poles[0] / 1.5, poles[-1] * 1.5, 20001)
X = np.array([reactance(w) for w in grid])
found = []
for i in np.nonzero((X[:-1] > 0) & (X[1:] < 0))[0]:  # + to -: a pole (the reactance increases elsewhere)
    lo, hi = grid[i], grid[i + 1]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if reactance(mid) > 0 else (lo, mid)
    found.append(0.5 * (lo + hi))
match = len(found) == len(poles) and np.allclose(found, poles, rtol=1e-8)
report("resonances", match, f"{len(found)} poles found by bisection, {len(poles)} predicted; max relative difference "
       + (f"{np.max(np.abs(np.array(found) / poles - 1)):.1e}" if len(found) == len(poles) else "-")
       + "; omega = " + ", ".join(f"{p:.6f}" for p in poles[:6]) + (" ..." if len(poles) > 6 else ""))

# 4. Foster's reactance theorem on random lossless networks
def foster_events(react, ws, depth=0):
    """the poles ('p') and zeros ('z') of a reactance function between the frequencies ws, and
    the number of places where it decreases other than across a pole. An interval where it seems
    to decrease may hide a zero and a pole close together: it is sampled again, more finely."""
    X = react(ws)
    events, bad = [], 0
    for i in range(len(ws) - 1):
        if X[i] > 0 > X[i + 1]:
            events.append("p")
        elif X[i] < 0 < X[i + 1]:
            events.append("z")
        elif X[i + 1] <= X[i]:
            if depth < 8:
                more, nbad = foster_events(react, np.linspace(ws[i], ws[i + 1], 101), depth + 1)
                events += more
                bad += nbad
            else:
                bad += 1
    return events, bad


bad, imag_worst, counts = 0, 0.0, []
for _ in range(5):
    kind = rng.random(len(edges)) < 0.5
    Ls, Cs = rng.uniform(0.3, 3, len(edges)), rng.uniform(0.3, 3, len(edges))

    def react(ws):
        Zs = Zf(np.where(kind[:, None], 1j * ws * Ls[:, None], 1 / (1j * ws * Cs[:, None])))
        global imag_worst
        imag_worst = max(imag_worst, np.max(np.abs(Zs.real) / np.abs(Zs)))
        return Zs.imag

    events, nbad = foster_events(react, np.geomspace(0.05, 20, 5001))
    bad += nbad + sum(e1 == e2 for e1, e2 in zip(events, events[1:]))  # two poles or two zeros in a row
    counts.append(len(events))
report("Foster", bad == 0 and imag_worst < 1e-9,
       f"5 random LC networks, 5,001+ frequencies each (refined where needed): |Re Z|/|Z| <= {imag_worst:.1e}, the "
       f"reactance increases between poles, {sum(counts)} poles and zeros alternate (violations: {bad})")

# 5. passivity of random RLC networks
ws = np.geomspace(0.01, 100, 2001)
lowest = np.inf
for _ in range(5):
    R, L, C = rng.uniform(0.01, 2, len(edges)), rng.uniform(0.2, 2, len(edges)), rng.uniform(0.2, 2, len(edges))
    zs = R[:, None] + 1j * ws * L[:, None] + 1 / (1j * ws * C[:, None])
    lowest = min(lowest, np.min(Zf(zs).real))
report("passivity", lowest >= 0, f"min Re Z over 5 RLC networks and 2,001 frequencies each: {lowest:.2e}")

# 6. vectorized sweep against batched solves
N = our.points
R, L, C = rng.uniform(0.2, 2, len(edges)), rng.uniform(0.2, 2, len(edges)), rng.uniform(0.2, 2, len(edges))
ws = np.geomspace(0.01, 100, N)
zs = R[:, None] + 1j * ws * L[:, None] + 1 / (1j * ws * C[:, None])
Zf(zs[:, :8])  # the first call may prepare the SIMD kernel
t0 = time.perf_counter()
Zs = Zf(zs)
t1 = time.perf_counter()
ref = nodal(zs)
t2 = time.perf_counter()
diff = np.max(np.abs(Zs / ref - 1))
report("vectorized", diff < 1e-11, f"{N:,} frequencies in {1e3 * (t1 - t0):.0f} ms, numpy solves {1e3 * (t2 - t1):.0f} ms; "
       f"max relative difference {diff:.1e}")

print(f"\nmachine code: {f.measure('ker-scalar-size') / 2**20:.2f} MiB (scalar), "
      f"{f.measure('ker-simd-size') / 2**20:.2f} MiB (SIMD)")
print("all checks passed" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)

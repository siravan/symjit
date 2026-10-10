# This file is generated using claude code
"""The effective resistance of a resistor grid with a symbolic resistance on every edge.

By Kirchhoff's theorem, the resistance between the nodes a and b of a network with the edge
resistances r_e is a ratio of two sums over subgraphs:

    R_ab = sum over the 2-forests F separating a and b of  prod_{e not in F} r_e
           -----------------------------------------------------------------
           sum over the spanning trees T              of  prod_{e not in T} r_e

(the 2-forests are the spanning trees of the network with a and b merged). This example
enumerates both for an m x n grid (4 x 4: 100,352 spanning trees and 2-forests, about two
million operations) and compiles R_ab with symjit. Checks:

1. Random resistances: against nodal analysis (one numpy linear solve).
2. Unit resistors: the numerator and the denominator count the 2-forests and the spanning trees,
   the latter given by Kirchhoff's eigenvalue formula, prod_{(j,k) != 0} lambda_jk / (m n); and R
   against Wu's formula (2004), sum_{(j,k) != 0} (psi_jk(a) - psi_jk(b))^2 / lambda_jk, with the
   eigenvalues lambda_jk = 4 - 2 cos(pi j/m) - 2 cos(pi k/n) and eigenvectors psi_jk of the
   grid's Laplacian.
3. Vectorized calls (SIMD) over many random resistance sets, against batched numpy solves.
4. Rayleigh's monotonicity law: raising any resistance never lowers R.

    python resistors.py               # a 4 x 4 grid, between opposite corners
    python resistors.py -m 3 -n 5     # other grids (4 x 5 has 557,568 spanning trees)
    python resistors.py --b 1,2       # another second node (row,column)
    python resistors.py --no-simd     # symjit options as in the other examples (util.py)
"""

import argparse
import sys
import time

import util

opts = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False)
opts.add_argument("-m", type=int, default=4, help="rows of nodes (default 4)")
opts.add_argument("-n", type=int, default=4, help="columns of nodes (default 4)")
opts.add_argument("--a", default="0,0", help="the first node, row,column (default 0,0)")
opts.add_argument("--b", default=None, help="the second node (default: the opposite corner)")
opts.add_argument("--points", type=int, default=20000, help="random resistance sets of the vectorized check")
opts.add_argument("--force", action="store_true", help="allow grids with more than a million spanning trees")
our, rest = opts.parse_known_args()
if "-h" in rest or "--help" in rest:
    opts.print_help()
    sys.exit(0)
sys.argv = sys.argv[:1] + rest
args = util.process_argv()

import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402

m, n = our.m, our.n
node = lambda i, j: i * n + j  # noqa: E731
a = node(*map(int, our.a.split(",")))
b = node(*map(int, our.b.split(","))) if our.b else node(m - 1, n - 1)
V = m * n
edges = [(node(i, j), node(i, j + 1)) for i in range(m) for j in range(n - 1)] + \
        [(node(i, j), node(i + 1, j)) for i in range(m - 1) for j in range(n)]
r = [sp.Symbol(f"r{k}") for k in range(len(edges))]


# ------------------------------------------------------------------ Kirchhoff's eigenvalues
def grid_eigen(m, n):
    """the eigenvalues and normalized eigenvectors of the grid's Laplacian (products of the
    eigenvectors of two paths)"""
    def path(p):
        lam = [2 - 2 * np.cos(np.pi * j / p) for j in range(p)]
        vec = [np.array([np.cos(np.pi * j * (x + 0.5) / p) for x in range(p)]) for j in range(p)]
        return lam, [v / np.linalg.norm(v) for v in vec]
    (lm, um), (ln, un) = path(m), path(n)
    return [(lm[j] + ln[k], np.outer(um[j], un[k]).ravel()) for j in range(m) for k in range(n) if j or k]


eigen = grid_eigen(m, n)
tree_count = round(np.prod([lam for lam, _ in eigen]) / V)
if tree_count > 1_000_000 and not our.force:
    sys.exit(f"the {m} x {n} grid has {tree_count:,} spanning trees; use --force to build an expression this large")


# ------------------------------------------------------------------ enumeration
def spanning_trees(num_nodes, edge_list):
    """all spanning trees (as sets of edge indices) of a multigraph; edges with equal
    endpoints (loops) are never used"""
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
        if cu != cv:  # take edge k: merge the components
            saved = list(comp)
            for i in range(num_nodes):
                if comp[i] == cv:
                    comp[i] = cu
            chosen.append(k)
            walk(k + 1, chosen)
            chosen.pop()
            comp[:] = saved
        walk(k + 1, chosen)  # leave edge k out

    walk(0, [])
    return out


def merged(u):
    """node u with b merged into a"""
    return a if u == b else u


def complement_sum(subgraphs):
    return sp.Add(*[sp.Mul(*[r[e] for e in range(len(edges)) if e not in s]) for s in subgraphs])


def build():
    trees = spanning_trees(V, edges)
    relabel = {u: k for k, u in enumerate(sorted({merged(u) for u in range(V)}))}
    forests = spanning_trees(V - 1, [(relabel[merged(u)], relabel[merged(v)]) for u, v in edges])
    num, den = complement_sum(forests), complement_sum(trees)
    # the two sums as a second function (compiled together with R, as three outputs, the shared
    # sums are not shared: 2.5x the code and 6x the memory)
    return [(r, [num / den]), (r, [num, den])], dict(trees=len(trees), forests=len(forests))


# compile_cached caches the expressions' JSON models when SYMJIT_EXAMPLE_CACHE is set (run_physics.py)
(f, counts), info, t_build, t_compile, cached = util.compile_cached("resistors", (m, n, a, b), build, **args)
num_trees, num_forests = info["trees"], info["forests"]

ops = num_trees * (len(edges) - V + 1) + num_forests * (len(edges) - V + 2)
print(f"{m} x {n} grid, {len(edges)} resistors, R between nodes {divmod(a, n)} and {divmod(b, n)}: "
      f"{num_trees:,} spanning trees, {num_forests:,} 2-forests")
print(f"enumeration and sympy expressions {t_build:.2f} s{' (cached)' if cached else ''}, symjit compilation "
      f"{t_compile:.2f} s; about {ops:,} operations, {f.measure('ker-scalar-size') / 2**20:.2f} MiB of machine code\n")

ok = True


def report(name, passed, detail):
    global ok
    ok &= passed
    print(f"{'ok  ' if passed else 'FAIL'} {name}: {detail}")


def laplacian(res):
    """the weighted Laplacian, for one resistance set (len(edges)) or a batch (len(edges) x N)"""
    res = np.asarray(res, dtype=float)
    Lap = np.zeros(res.shape[1:] + (V, V))
    for k, (u, v) in enumerate(edges):
        g = 1 / res[k]
        Lap[..., u, u] += g
        Lap[..., v, v] += g
        Lap[..., u, v] -= g
        Lap[..., v, u] -= g
    return Lap


def nodal(res):
    """R_ab by nodal analysis: ground b, inject a unit current at a, solve for the potentials"""
    keep = [i for i in range(V) if i != b]
    Lg = laplacian(res)[..., keep, :][..., :, keep]
    rhs = np.zeros(Lg.shape[:-1])
    rhs[..., keep.index(a)] = 1
    return np.linalg.solve(Lg, rhs[..., None])[..., keep.index(a), 0]


# 1. random resistances
rng = np.random.default_rng(1)
worst = 0.0
for _ in range(20):
    res = rng.uniform(0.5, 2.0, len(edges))
    worst = max(worst, abs(f(*res)[0] / nodal(res) - 1))
report("random resistances", worst < 1e-12, f"max relative difference to nodal analysis over 20 sets {worst:.1e}")

# 2. unit resistors: the counts and Wu's formula
R1 = f(*np.ones(len(edges)))[0]
forests1, trees1 = counts(*np.ones(len(edges)))
wu = sum((psi[a] - psi[b]) ** 2 / lam for lam, psi in eigen)
report("unit resistors", round(trees1) == tree_count == num_trees and round(forests1) == num_forests
       and abs(R1 / wu - 1) < 1e-12,
       f"R = {R1:.15f} (Wu's formula {wu:.15f}); {trees1:,.0f} spanning trees (Kirchhoff {tree_count:,}), "
       f"{forests1:,.0f} 2-forests")

# 3. vectorized calls against batched solves
N = our.points
Rs = rng.uniform(0.5, 2.0, (len(edges), N))
f(*Rs[:, :8])  # the first call may prepare the SIMD kernel
t0 = time.perf_counter()
z = np.asarray(f(*Rs)[0]).ravel()
t1 = time.perf_counter()
d = nodal(Rs)
t2 = time.perf_counter()
diff = np.max(np.abs(z / d - 1))
report("vectorized", diff < 1e-12,
       f"{N:,} resistance sets in {1e3 * (t1 - t0):.0f} ms, numpy solves {1e3 * (t2 - t1):.0f} ms; "
       f"max relative difference {diff:.1e}")

# 4. Rayleigh's monotonicity law
res = rng.uniform(0.5, 2.0, len(edges))
base = f(*res)[0]
up = []
for k in range(len(edges)):
    w = res.copy()
    w[k] *= 1.5
    up.append(f(*w)[0] - base)
report("monotonicity", min(up) >= -1e-14 * base,
       f"raising one resistance by 50% changes R by {min(up):.2e} ... {max(up):.2e} (never negative)")

print(f"\nmachine code: {f.measure('ker-scalar-size') / 2**20:.2f} MiB (scalar), "
      f"{f.measure('ker-simd-size') / 2**20:.2f} MiB (SIMD)")
if args.get("use_simd", True) and not args.get("ty", "native").startswith(("bytecode", "debug")) \
        and f.measure("ker-simd-size") == 0:
    print("note: no SIMD kernel: its stack frame exceeds symjit's stack limit (see domino.py)")
print("all checks passed" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)

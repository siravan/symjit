# This file is generated using claude code
"""The Lee-Yang circle theorem, checked on a megabyte-sized complex polynomial.

The Ising model on an m x n lattice (periodic boundaries by default), with a coupling J_e >= 0 on
every bond and a magnetic field h, has (up to a factor) the partition function

    P(z) = sum over the 2^N spin states of  prod_{unsatisfied bonds e} x_e  *  z^(number of down spins),

with the bond weights x_e = exp(-2 J_e) and the fugacity z = exp(-2 h): a polynomial of degree
N = m n in z. Lee and Yang (1952) proved that for ferromagnetic couplings (0 < x_e <= 1) all its
zeros lie on the unit circle |z| = 1. This example builds P as one sympy sum (4 x 4: 65,536
states, 59,101 terms in 33 variables) and compiles it with dtype="complex128". Checks:

1. Random couplings and complex z, against brute-force numpy over all spin states.
2. No coupling (x_e = 1): P(z) = (1 + z)^N.
3. The coefficients of P in z, from P at the roots of unity (one vectorized call and an FFT),
   against numpy; they are palindromic (flipping all spins).
4. Lee-Yang: for random ferromagnetic couplings, the roots of P lie on |z| = 1, and the argument
   principle (the winding of P(z) along a circle, evaluated by vectorized calls) counts no zeros
   inside |z| = 0.9 and all N inside |z| = 1.1.
5. The hypothesis matters: with antiferromagnetic bonds (x_e > 1) zeros leave the circle.
6. The Lee-Yang edge: for a uniform coupling J, the zero closest to z = 1 (h = 0) moves towards it
   as J grows (the phase transition of the infinite lattice).
7. Vectorized calls (complex SIMD) against numpy.

    python lee_yang.py                # a 4 x 4 periodic lattice
    python lee_yang.py -m 3 -n 5      # other lattices (up to 18 spins without --force)
    python lee_yang.py --open         # free boundaries
"""

import argparse
import sys
import time

import util

opts = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False)
opts.add_argument("-m", type=int, default=4, help="rows (default 4)")
opts.add_argument("-n", type=int, default=4, help="columns (default 4)")
opts.add_argument("--open", action="store_true", help="free instead of periodic boundaries")
opts.add_argument("--points", type=int, default=20000, help="random points of the vectorized check (default 20000)")
opts.add_argument("--force", action="store_true", help="allow more than 18 spins")
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
N = m * n
if N > 18 and not our.force:
    sys.exit(f"{N} spins: 2^{N} states; use --force to build an expression this large")
site = lambda i, j: i * n + j  # noqa: E731
bonds = set()
for i in range(m):
    for j in range(n):
        if j + 1 < n or (not our.open and n > 2):
            bonds.add(tuple(sorted((site(i, j), site(i, (j + 1) % n)))))
        if i + 1 < m or (not our.open and m > 2):
            bonds.add(tuple(sorted((site(i, j), site((i + 1) % m, j)))))
bonds = sorted(bonds)
B = len(bonds)
x = [sp.Symbol(f"x{k}") for k in range(B)]
zs = sp.Symbol("z")

# ------------------------------------------------------------------ the polynomial
states = (np.arange(2 ** N)[:, None] >> np.arange(N)) & 1  # 1 = down
unsat = np.array([states[:, u] != states[:, v] for u, v in bonds]).T
ndown = states.sum(axis=1)

def build():
    P = sp.Add(*[sp.Mul(*[x[k] for k in np.nonzero(unsat[s])[0]]) * zs ** int(ndown[s]) for s in range(2 ** N)])
    return [(x + [zs], [P])], dict(terms=len(P.args))


# compile_cached caches the expression's JSON model when SYMJIT_EXAMPLE_CACHE is set (run_physics.py)
(f,), info, t_build, t_compile, cached = util.compile_cached("lee_yang", (m, n, "open" if our.open else "periodic"),
                                                            build, **args)
print(f"{m} x {n} lattice ({'free' if our.open else 'periodic'} boundaries): {N} spins, {B} bonds, {2 ** N:,} states, "
      f"{info['terms']:,} terms")
print(f"sympy {t_build:.1f} s{' (cached)' if cached else ''}, symjit compilation (complex128) {t_compile:.1f} s, "
      f"{f.measure('ker-scalar-size') / 2**20:.2f} MiB of machine code\n")


def Pf(xs, z):
    """the compiled P: xs (B weights) and z scalars, or xs B x K and z of length K"""
    z = np.asarray(z, dtype=complex)
    if z.ndim == 0:
        return complex(np.ravel(f(*np.asarray(xs, dtype=complex), z))[0])
    xs = np.broadcast_to(np.asarray(xs, dtype=complex).reshape(B, -1), (B, z.size))
    return np.asarray(f(*xs, z)[0]).ravel()


def brute(xs, z):
    """P by numpy over all spin states"""
    w = np.prod(np.where(unsat, np.asarray(xs), 1.0), axis=1)
    return np.sum(w[:, None] * np.asarray(z, dtype=complex).ravel()[None, :] ** ndown[:, None], axis=0)


def coefficients(xs):
    """the coefficients c_0 ... c_N of P in z, from P at the 2N-th roots of unity w_k (one vectorized
    call): P(w_k) = sum_j c_j w_k^j, so c_j = (1/M) sum_k P(w_k) w_k^-j, a forward FFT"""
    M = 2 * N
    w = np.exp(2j * np.pi * np.arange(M) / M)
    return (np.fft.fft(Pf(np.repeat(np.asarray(xs)[:, None], M, axis=1), w)) / M)[:N + 1]


def winding(xs, r, K=8192):
    """the number of zeros inside |z| = r: the change of arg P along the circle / 2 pi"""
    w = r * np.exp(2j * np.pi * np.arange(K + 1) / K)
    phase = np.unwrap(np.angle(Pf(np.repeat(np.asarray(xs)[:, None], K + 1, axis=1), w)))
    return (phase[-1] - phase[0]) / (2 * np.pi)


ok = True


def report(name, passed, detail):
    global ok
    ok &= passed
    print(f"{'ok  ' if passed else 'FAIL'} {name}: {detail}")


rng = np.random.default_rng(1)

# 1. random couplings and complex z
worst = 0.0
for _ in range(20):
    xs = rng.uniform(0.05, 1, B)
    z = rng.uniform(0.3, 2) * np.exp(1j * rng.uniform(0, 2 * np.pi))
    ref = brute(xs, [z])[0]
    worst = max(worst, abs(Pf(xs, z) / ref - 1))
report("random points", worst < 1e-12, f"max relative difference to brute-force numpy over 20 points {worst:.1e}")

# 2. no coupling
pts = np.exp(1j * np.linspace(0.1, 6, 7)) * np.linspace(0.5, 1.5, 7)
d = max(abs(Pf(np.ones(B), z) - (1 + z) ** N) / (1 + abs(z)) ** N for z in pts)
report("no coupling", d < 1e-13, f"P(z) = (1 + z)^{N} within {d:.1e} (relative to (1 + |z|)^{N})")

# 3. coefficients
xs = rng.uniform(0.05, 1, B)
c = coefficients(xs)
ref = np.array([np.sum(np.prod(np.where(unsat[ndown == k], xs, 1.0), axis=1)) for k in range(N + 1)])
dc = np.max(np.abs(c - ref)) / np.max(np.abs(ref))
pal = np.max(np.abs(c - c[::-1])) / np.max(np.abs(c))
report("coefficients", dc < 1e-12 and pal < 1e-12,
       f"from {2 * N} values on the unit circle: max difference to numpy {dc:.1e}, palindromic within {pal:.1e}")

# 4. Lee-Yang: the zeros lie on the unit circle
off, wind = 0.0, []
for _ in range(10):
    xs = rng.uniform(0.05, 1, B)
    roots = np.roots(coefficients(xs).real[::-1])
    off = max(off, np.max(np.abs(np.abs(roots) - 1)))
    wind.append((winding(xs, 0.9), winding(xs, 1.1)))
inside = all(abs(a) < 1e-6 and abs(b - N) < 1e-6 for a, b in wind)
report("Lee-Yang", off < 1e-6 and inside,
       f"10 random ferromagnetic lattices: max ||z_k| - 1| = {off:.1e}; zeros inside |z| = 0.9: "
       f"{max(abs(a) for a, _ in wind):.0f}, inside |z| = 1.1: {np.mean([b for _, b in wind]):.0f}")

# 5. antiferromagnetic bonds break the theorem
xs = rng.uniform(0.05, 1, B)
xs[: B // 2] = rng.uniform(2, 20, B // 2)  # J < 0 on half the bonds
roots = np.roots(coefficients(xs).real[::-1])
spread = np.max(np.abs(np.abs(roots) - 1))
w9 = winding(xs, 0.9)
report("antiferromagnet", spread > 1e-3 and w9 > 0.5,
       f"with J < 0 on half the bonds the zeros leave the circle: max ||z_k| - 1| = {spread:.2f}, "
       f"{w9:.0f} of them inside |z| = 0.9")

# 6. the Lee-Yang edge: the zero closest to z = 1 approaches it as J grows
Js = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
gaps = []
for J in Js:
    roots = np.roots(coefficients(np.full(B, np.exp(-2 * J))).real[::-1])
    gaps.append(np.min(np.abs(np.angle(roots))))
report("Lee-Yang edge", all(g1 > g2 for g1, g2 in zip(gaps, gaps[1:])),
       "uniform coupling J = " + ", ".join(map(str, Js)) + ": the smallest |arg z_k| = "
       + ", ".join(f"{g:.3f}" for g in gaps))

# 7. vectorized calls against numpy
K = our.points
X = rng.uniform(0.05, 1, (B, K))
Z = rng.uniform(0.3, 2, K) * np.exp(1j * rng.uniform(0, 2 * np.pi, K))
Pf(X[:, :8], Z[:8])  # the first call may prepare the SIMD kernel
t0 = time.perf_counter()
got = Pf(X, Z)
t1 = time.perf_counter()
sub = slice(0, 2000)  # numpy over all states is slow: a subset
ref = np.array([brute(X[:, k], [Z[k]])[0] for k in range(sub.stop)])
t2 = time.perf_counter()
d = np.max(np.abs(got[sub] / ref - 1))
report("vectorized", d < 1e-12, f"{K:,} points in {1e3 * (t1 - t0):.0f} ms ({1e6 * (t1 - t0) / K:.1f} us each); "
       f"numpy {1e6 * (t2 - t1) / sub.stop:.0f} us each; max relative difference on {sub.stop:,} of them {d:.1e}")

print(f"\nmachine code: {f.measure('ker-scalar-size') / 2**20:.2f} MiB (scalar), "
      f"{f.measure('ker-simd-size') / 2**20:.2f} MiB (SIMD)")
if args.get("use_simd", True) and not args.get("ty", "native").startswith(("bytecode", "debug")) \
        and f.measure("ker-simd-size") == 0:
    print("note: no SIMD kernel: its stack frame exceeds symjit's stack limit, so the vectorized calls ran the "
          "scalar kernel (see domino.py, --stack-limit)")
print("all checks passed" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)

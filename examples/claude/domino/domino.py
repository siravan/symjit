# This file is generated using claude code
"""Domino tilings (the dimer model): a megabyte-sized expression with exact checks.

The partition function of the dimer model on an m x n board, with a weight w_e for every
place e a domino can lie (h_i_j: the squares (i, j) and (i, j+1); v_i_j: (i, j) and (i+1, j)), is

    Z(w) = sum over all domino tilings T of the board of  prod_{e in T} w_e.

This example enumerates the tilings, builds Z as one sympy sum of products (the 6 x 8 board:
167,089 tilings of 24 dominoes, 82 weights) and compiles it with symjit (6 MiB of machine
code). The result is checked in four ways:

1. With all weights 1, Z is the number of tilings, which Kasteleyn's formula gives in closed
   form: prod_{j <= m/2} prod_{k <= n/2} (4 cos^2(pi j/(m+1)) + 4 cos^2(pi k/(n+1))).
2. For random weights, Z = |det K|, where K is the Kasteleyn matrix: rows for the black squares,
   columns for the white ones, w_e for a horizontal domino and i*w_e for a vertical one.
3. Vectorized calls (the SIMD kernel) over many random weight sets, against numpy determinants.
4. The probability that a domino lies at e. Z is linear in each weight, so
   P(e) = w_e dZ/dw_e / Z = (Z - Z|_{w_e=0}) / Z, from the compiled function alone. Kenyon's
   formula gives it as |K(b, w) K^-1(w, b)| for e = (b, w); and at each square, the
   probabilities of the dominoes covering it add up to 1.

    python domino.py                  # the 6 x 8 board
    python domino.py -m 4 -n 6        # smaller boards (one side must be even)
    python domino.py --no-simd        # symjit options as in the other examples (util.py)
    python domino.py --stack-limit 4  # allow the SIMD kernel of the 6 x 8 board (a 1.5 MiB frame)
"""

import argparse
import sys
import time

import util

board = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False)
board.add_argument("-m", type=int, default=6, help="rows (default 6)")
board.add_argument("-n", type=int, default=8, help="columns (default 8)")
board.add_argument("--points", type=int, default=20000, help="random weight sets of the vectorized check (default 20000)")
board.add_argument("--force", action="store_true", help="allow boards with more than 2 million tilings")
board.add_argument("--stack-limit", type=float, default=0, metavar="MiB",
                   help="symjit's stack limit (default 1 MiB): the SIMD kernel is compiled only if its stack frame "
                        "fits; vectorized calls run it on worker threads with small stacks, so a frame larger than "
                        "about 2 MiB crashes them (use --no-threads then)")
board_args, rest = board.parse_known_args()
if "-h" in rest or "--help" in rest:
    board.print_help()
    sys.exit(0)
sys.argv = sys.argv[:1] + rest
args = util.process_argv()

import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402

if board_args.stack_limit:
    # numeric options travel in `ty` after the compiler type (Config::from_name)
    args["ty"] = args.get("ty", "native") + f";stack_limit={int(board_args.stack_limit * 2**20)}"

m, n = board_args.m, board_args.n
if (m * n) % 2:
    sys.exit("a board with an odd number of squares has no domino tilings")


# ------------------------------------------------------------------ the board and Kasteleyn
def kasteleyn_count(m, n):
    """the number of domino tilings of an m x n board (Kasteleyn, 1961)"""
    p = 1.0
    for j in range(1, (m + 1) // 2 + 1):
        for k in range(1, (n + 1) // 2 + 1):
            p *= 4 * np.cos(np.pi * j / (m + 1)) ** 2 + 4 * np.cos(np.pi * k / (n + 1)) ** 2
    return round(p)


expected = kasteleyn_count(m, n)
if expected > 2_000_000 and not board_args.force:
    sys.exit(f"the {m} x {n} board has {expected:,} tilings; use --force to build an expression this large")

edges = [("h", i, j) for i in range(m) for j in range(n - 1)] + [("v", i, j) for i in range(m - 1) for j in range(n)]
weights = {e: sp.Symbol(f"{e[0]}_{e[1]}_{e[2]}") for e in edges}
index = {e: k for k, e in enumerate(edges)}


def squares(e):
    """the two squares a domino at e covers"""
    t, i, j = e
    return ((i, j), (i, j + 1)) if t == "h" else ((i, j), (i + 1, j))


black = [(i, j) for i in range(m) for j in range(n) if (i + j) % 2 == 0]
white = [(i, j) for i in range(m) for j in range(n) if (i + j) % 2 == 1]
row = {s: k for k, s in enumerate(black)}
col = {s: k for k, s in enumerate(white)}


def kasteleyn_matrix(w):
    """K(b, w) = weight (horizontal) or i * weight (vertical); w is indexed like `edges`"""
    K = np.zeros((len(black), len(white)), dtype=complex)
    for e, k in index.items():
        a, b = squares(e)
        bl, wh = (a, b) if a in row else (b, a)
        K[row[bl], col[wh]] = w[k] * (1 if e[0] == "h" else 1j)
    return K


# ------------------------------------------------------------------ enumeration and compilation
def tilings(m, n):
    """all domino tilings of the board, as lists of edges (fill the first empty square)"""
    filled = [[False] * n for _ in range(m)]
    out, cur = [], []

    def place(pos):
        while pos < m * n and filled[pos // n][pos % n]:
            pos += 1
        if pos == m * n:
            out.append(list(cur))
            return
        i, j = divmod(pos, n)
        filled[i][j] = True
        for t, (a, b) in (("h", (i, j + 1)), ("v", (i + 1, j))):
            if a < m and b < n and not filled[a][b]:
                filled[a][b] = True
                cur.append((t, i, j))
                place(pos + 1)
                cur.pop()
                filled[a][b] = False
        filled[i][j] = False

    place(0)
    return out


symbols = [weights[e] for e in edges]


def build():
    T = tilings(m, n)
    return [(symbols, [sp.Add(*[sp.Mul(*[weights[e] for e in t]) for t in T])])], dict(tilings=len(T))


# compile_cached caches the expression's JSON model when SYMJIT_EXAMPLE_CACHE is set (run_physics.py)
(f,), info, t_build, t_compile, cached = util.compile_cached("domino", (m, n), build, **args)
num_tilings = info["tilings"]

print(f"{m} x {n} board: {num_tilings:,} tilings of {m * n // 2} dominoes, {len(edges)} weights")
print(f"enumeration and sympy expression {t_build:.2f} s{' (cached)' if cached else ''}, "
      f"symjit compilation {t_compile:.2f} s")
print(f"expression: {num_tilings * (m * n // 2 - 1) + num_tilings - 1:,} operations; scalar machine code "
      f"{f.measure('ker-scalar-size') / 2**20:.2f} MiB\n")

ok = True


def report(name, passed, detail):
    global ok
    ok &= passed
    print(f"{'ok  ' if passed else 'FAIL'} {name}: {detail}")


# 1. the number of tilings
count = f(*[1.0] * len(edges))[0]
report("count", round(count) == expected == num_tilings and abs(count - expected) < 1e-9 * expected,
       f"Z(1, ..., 1) = {count:,.0f}, Kasteleyn's formula {expected:,}, enumerated {num_tilings:,}")

# 2. random weights against |det K|
rng = np.random.default_rng(1)
worst = 0.0
for _ in range(20):
    w = rng.uniform(0.5, 2.0, len(edges))
    z = f(*w)[0]
    d = abs(np.linalg.det(kasteleyn_matrix(w)))
    worst = max(worst, abs(z - d) / d)
report("random weights", worst < 1e-12, f"max relative difference to |det K| over 20 weight sets {worst:.1e}")

# 3. vectorized calls against batched determinants
N = board_args.points
W = rng.uniform(0.5, 2.0, (len(edges), N))
z = np.asarray(f(*W)).ravel()  # the first call also prepares the SIMD kernel
t0 = time.perf_counter()
z = np.asarray(f(*W)).ravel()
t1 = time.perf_counter()
Ks = np.zeros((N, len(black), len(white)), dtype=complex)
for e, k in index.items():
    a, b = squares(e)
    bl, wh = (a, b) if a in row else (b, a)
    Ks[:, row[bl], col[wh]] = W[k] * (1 if e[0] == "h" else 1j)
d = np.abs(np.linalg.det(Ks))
t2 = time.perf_counter()
diff = np.max(np.abs(z - d) / d)
report("vectorized", diff < 1e-12,
       f"{N:,} weight sets in {1e3 * (t1 - t0):.0f} ms ({1e6 * (t1 - t0) / N:.2f} us each), "
       f"numpy determinants {1e3 * (t2 - t1):.0f} ms; max relative difference {diff:.1e}")

# 4. placement probabilities: (Z - Z|w_e=0) / Z against Kenyon's formula
ones = np.ones(len(edges))
Kinv = np.linalg.inv(kasteleyn_matrix(ones))
P = np.empty(len(edges))
worst = 0.0
for e, k in index.items():
    w = ones.copy()
    w[k] = 0.0
    P[k] = (count - f(*w)[0]) / count
    a, b = squares(e)
    bl, wh = (a, b) if a in row else (b, a)
    kenyon = abs((1 if e[0] == "h" else 1j) * Kinv[col[wh], row[bl]])
    worst = max(worst, abs(P[k] - kenyon))
cover = np.zeros((m, n))
for e, k in index.items():
    for s in squares(e):
        cover[s] += P[k]
report("probabilities", worst < 1e-12 and np.max(np.abs(cover - 1)) < 1e-12,
       f"max difference to Kenyon's formula {worst:.1e}; per square they sum to 1 within {np.max(np.abs(cover - 1)):.1e}")

print(f"\nmachine code: {f.measure('ker-scalar-size') / 2**20:.2f} MiB (scalar), "
      f"{f.measure('ker-simd-size') / 2**20:.2f} MiB (SIMD)")
if args.get("use_simd", True) and not args.get("ty", "native").startswith(("bytecode", "debug")) \
        and f.measure("ker-simd-size") == 0:
    frame = f.measure("stack-size") * (64 if args.get("enable_simd512") else 32)
    print(f"note: no SIMD kernel: its stack frame ({frame / 2**20:.2f} MiB) exceeds the stack limit; the vectorized "
          f"calls ran the scalar kernel (see --stack-limit)")
corner = P[index[("h", 0, 0)]]
print(f"P(domino at the corner, horizontal) = {corner:.6f}, vertical = {1 - corner:.6f}")
print("all checks passed" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)

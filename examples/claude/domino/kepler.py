# This file is generated using claude code
"""A Kepler orbit integrated by N unrolled leapfrog steps: megabytes of straight-line code.

The two-body problem (G M = 1) is integrated with the leapfrog (kick-drift-kick, Stormer-Verlet)
method, a symplectic integrator:

    v += h/2 a(x);   x += h v;   v += h/2 a(x),   with a(x) = -x / |x|^3.

The N steps are unrolled with the low-level `Composer` interface into one function
(x0, y0, vx0, vy0, h) -> (x, y, vx, vy) at t = N h: 50,000 steps are about 1.3 million
operations and 19 MiB of machine code, a long chain of dependent operations (a sympy expression
of it would have to be expanded into a tree, which grows exponentially with N). The same method
is also compiled as a Composer loop (a few hundred bytes of code) for comparison. Checks:

1. The unrolled function, the loop and a numpy implementation of the same steps agree.
2. The angular momentum x vy - y vx is conserved to rounding error (the leapfrog method keeps it
   exactly, as the force is central), and the energy error stays small (O(h^2), no drift).
3. Compared with the exact Kepler orbit (Kepler's equation solved by Newton's method), the
   position error at t = N h is O(h^2 t) = O(h^3 N): halving h divides it by about 8.

`--ty bytecode` is not supported: compile_composer runs native kernels only.
4. Vectorized calls (`evaluate`, SIMD) over a batch of orbits with eccentricities 0 ... 0.6,
   against numpy.

    python kepler.py                  # 50,000 steps
    python kepler.py -N 10000         # fewer steps
    python kepler.py --no-simd        # symjit options as in the other examples (util.py)
"""

import argparse
import sys
import time

import util

opts = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False)
opts.add_argument("-N", type=int, default=50000, help="leapfrog steps (default 50000)")
opts.add_argument("--h", type=float, default=0.0025, help="the step size of the checks (default 0.0025)")
opts.add_argument("--orbits", type=int, default=2000, help="orbits of the vectorized check (default 2000)")
our, rest = opts.parse_known_args()
if "-h" in rest or "--help" in rest:
    opts.print_help()
    sys.exit(0)
sys.argv = sys.argv[:1] + rest
args = util.process_argv()

import numpy as np  # noqa: E402
from symjit import Composer, compile_composer  # noqa: E402

N, H = our.N, our.h
if args.get("ty", "native").split(";")[0] in ("bytecode", "debug", "wasm"):
    # compile_composer goes through the Symbolica bridge, which runs native kernels only (and
    # returns zeros for these types instead of raising)
    sys.exit(f"compile_composer does not support ty={args['ty']}")


# ------------------------------------------------------------------ the two programs
def accel(cp, x, y):
    r2 = cp.fadd(cp.fmul(x, x), cp.fmul(y, y))
    r3 = cp.fmul(r2, cp.sqrt(r2))
    return cp.neg(cp.fdiv(x, r3)), cp.neg(cp.fdiv(y, r3))


def step(cp, s, h, half):
    """one kick-drift-kick step on the state s = (x, y, vx, vy, ax, ay); returns the new state"""
    x, y, vx, vy, ax, ay = s
    vx = cp.fadd(vx, cp.fmul(half, ax))
    vy = cp.fadd(vy, cp.fmul(half, ay))
    x = cp.fadd(x, cp.fmul(h, vx))
    y = cp.fadd(y, cp.fmul(h, vy))
    ax, ay = accel(cp, x, y)
    vx = cp.fadd(vx, cp.fmul(half, ax))
    vy = cp.fadd(vy, cp.fmul(half, ay))
    return x, y, vx, vy, ax, ay


def unrolled(n):
    cp = Composer(5, 4)
    x, y, vx, vy, h = (cp.arg(i) for i in range(5))
    half = cp.fmul(cp.constant(0.5), h)
    s = (x, y, vx, vy, *accel(cp, x, y))
    for _ in range(n):
        s = step(cp, s, h, half)
    for i in range(4):
        cp.assign(cp.out(i), s[i])
    return cp


def looped(n):
    cp = Composer(5, 4)
    h = cp.arg(4)
    half = cp.fmul(cp.constant(0.5), h)
    state = [cp.new_temp() for _ in range(6)]
    for t, v in zip(state, (cp.arg(0), cp.arg(1), cp.arg(2), cp.arg(3), *accel(cp, cp.arg(0), cp.arg(1)))):
        cp.assign(t, v)
    body = cp.new_block()
    for t, v in zip(state, step(body, state, h, half)):
        body.assign(t, v)
    cp.append_for(cp.new_temp(), 0, n, body)
    for i in range(4):
        cp.assign(cp.out(i), state[i])
    return cp


def leapfrog_numpy(x, y, vx, vy, h, n):
    """the same steps in numpy (works on arrays of orbits)"""
    half = 0.5 * h
    r2 = x * x + y * y
    r3 = r2 * np.sqrt(r2)
    ax, ay = -(x / r3), -(y / r3)
    for _ in range(n):
        vx = vx + half * ax
        vy = vy + half * ay
        x = x + h * vx
        y = y + h * vy
        r2 = x * x + y * y
        r3 = r2 * np.sqrt(r2)
        ax, ay = -(x / r3), -(y / r3)
        vx = vx + half * ax
        vy = vy + half * ay
    return x, y, vx, vy


def kepler_exact(x0, y0, vx0, vy0, t):
    """the position at time t on the exact Kepler orbit (elliptic, G M = 1)"""
    r0 = np.hypot(x0, y0)
    energy = (vx0 ** 2 + vy0 ** 2) / 2 - 1 / r0
    a = -1 / (2 * energy)
    L = x0 * vy0 - y0 * vx0
    ex, ey = vy0 * L - x0 / r0, -vx0 * L - y0 / r0  # the eccentricity vector
    e = np.hypot(ex, ey)
    E0 = np.arctan2((x0 * vx0 + y0 * vy0) / (e * np.sqrt(a)), (1 - r0 / a) / e)
    M = E0 - e * np.sin(E0) + t / a ** 1.5
    E = M
    for _ in range(50):  # Newton's method on Kepler's equation E - e sin E = M
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    xp, yp = a * (np.cos(E) - e), a * np.sqrt(1 - e * e) * np.sin(E) * np.sign(L)
    w = np.arctan2(ey, ex)
    return xp * np.cos(w) - yp * np.sin(w), xp * np.sin(w) + yp * np.cos(w)


def periapsis_start(e, theta):
    """an orbit with semi-major axis 1 and eccentricity e, starting at its periapsis, rotated by theta"""
    r, v = 1 - e, np.sqrt((1 + e) / (1 - e))
    c, s = np.cos(theta), np.sin(theta)
    return r * c, r * s, -v * s, v * c


# ------------------------------------------------------------------ compile
t0 = time.perf_counter()
cp = unrolled(N)
t1 = time.perf_counter()
f = compile_composer(cp, **args)
t2 = time.perf_counter()
g = compile_composer(looped(N), **args)
print(f"{N:,} leapfrog steps: Composer program {t1 - t0:.2f} s, symjit compilation {t2 - t1:.2f} s, "
      f"{f.measure('mir-size'):,} instructions, {f.measure('ker-scalar-size') / 2**20:.2f} MiB of machine code "
      f"(the loop: {g.measure('ker-scalar-size'):,} bytes)\n")

ok = True


def report(name, passed, detail):
    global ok
    ok &= passed
    print(f"{'ok  ' if passed else 'FAIL'} {name}: {detail}")


def call(fn, s, h):
    return np.ravel(fn(*s, h))[:4]


# 1. the unrolled code, the loop and numpy
s0 = periapsis_start(0.5, 0.3)
t0 = time.perf_counter()
u = call(f, s0, H)
t1 = time.perf_counter()
w = call(g, s0, H)
t2 = time.perf_counter()
z = np.array(leapfrog_numpy(*s0, H, N))
t3 = time.perf_counter()
d_loop = np.max(np.abs(u - w)) / np.max(np.abs(u))
d_numpy = np.max(np.abs(u - z)) / np.max(np.abs(u))
report("same steps", d_loop < 1e-10 and d_numpy < 1e-10,
       f"unrolled vs loop {d_loop:.1e}, vs numpy {d_numpy:.1e} (relative); one orbit: unrolled {1e3 * (t1 - t0):.1f} ms, "
       f"loop {1e3 * (t2 - t1):.1f} ms, numpy {1e3 * (t3 - t2):.0f} ms")

# 2. conserved quantities
L0, L = s0[0] * s0[3] - s0[1] * s0[2], u[0] * u[3] - u[1] * u[2]
E0 = (s0[2] ** 2 + s0[3] ** 2) / 2 - 1 / np.hypot(s0[0], s0[1])
E = (u[2] ** 2 + u[3] ** 2) / 2 - 1 / np.hypot(u[0], u[1])
report("conservation", abs(L - L0) < 1e-11 and abs((E - E0) / E0) < 1e-3,
       f"angular momentum changed by {abs(L - L0):.1e}, energy by {abs((E - E0) / E0):.1e} (relative) "
       f"over {N * H / (2 * np.pi):.1f} orbits")

# 3. the exact orbit: the error is O(h^3 N)
errs = []
for h in (2 * H, H, H / 2):
    x, y = call(f, s0, h)[:2]
    xe, ye = kepler_exact(*s0, N * h)
    errs.append(np.hypot(x - xe, y - ye))
# per halving of h, over both halvings (single ratios vary while the time is short)
rate = np.sqrt(errs[0] / errs[2])
report("exact orbit", errs[1] < 0.05 and errs[0] > errs[1] > errs[2] and 5 < rate < 12,
       f"position error at t = N h for h = {2 * H:g}, {H:g}, {H / 2:g}: "
       + ", ".join(f"{e:.2e}" for e in errs) + f"; it falls by {rate:.1f} per halving of h (about 8 expected)")

# 4. vectorized over a batch of orbits
rng = np.random.default_rng(1)
K = our.orbits
starts = np.array(periapsis_start(rng.uniform(0, 0.6, K), rng.uniform(0, 2 * np.pi, K)))
X = np.column_stack([*starts, np.full(K, H)])
f.evaluate(X[:8])  # the first call may prepare the SIMD kernel
t0 = time.perf_counter()
U = f.evaluate(X)
t1 = time.perf_counter()
V = g.evaluate(X)
t2 = time.perf_counter()
Z = np.array(leapfrog_numpy(*starts, H, N)).T
t3 = time.perf_counter()
d = max(np.max(np.abs(U - Z)), np.max(np.abs(V - Z))) / np.max(np.abs(Z))
dL = np.max(np.abs((U[:, 0] * U[:, 3] - U[:, 1] * U[:, 2]) - (starts[0] * starts[3] - starts[1] * starts[2])))
report("vectorized", d < 1e-10 and dL < 1e-11,
       f"{K:,} orbits: unrolled {1e3 * (t1 - t0):.0f} ms, loop {1e3 * (t2 - t1):.0f} ms, numpy {1e3 * (t3 - t2):.0f} ms; "
       f"max relative difference to numpy {d:.1e}, angular momentum within {dL:.1e}")

print(f"\nmachine code: {f.measure('ker-scalar-size') / 2**20:.2f} MiB (scalar), "
      f"{f.measure('ker-simd-size') / 2**20:.2f} MiB (SIMD)")
print("all checks passed" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)

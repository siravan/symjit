# This file is generated using claude code
"""A quantum particle on a lattice, N unrolled split-operator time steps: complex straight-line code.

The Schrodinger equation i dpsi/dt = H psi on a chain of S sites, with nearest-neighbor hopping
and a potential V_j on every site, H = -sum_j (|j><j+1| + |j+1><j|) + sum_j V_j |j><j|, is
integrated by the symmetric (Strang) split-operator method:

    psi <- e^{-i V dt/2} e^{-i T_even dt/2} e^{-i T_odd dt} e^{-i T_even dt/2} e^{-i V dt/2} psi,

where T_even and T_odd are the hopping terms of the even and odd bonds; each factor is a phase
or a set of independent 2 x 2 rotations, so every step is exactly unitary. The N steps are unrolled
with the `Composer` (dtype="complex128") into one function (psi_0, V, dt) -> psi(N dt): 20 sites
and 2,500 steps are about 600,000 complex operations and 20 MiB of machine code. Checks:

1. The unrolled function, the same steps as a Composer loop, and numpy agree.
2. The norm is conserved to rounding error (the steps are unitary).
3. Time reversal: N steps with dt and then N steps with -dt give back psi_0 (the symmetric
   splitting satisfies U(-dt) = U(dt)^-1).
4. Against the exact solution (numpy eigh of H): the error at t = N dt is O(dt^2 t) = O(dt^3 N),
   divided by about 8 when dt is halved.
5. A free particle starting at the middle site spreads as psi_j(t) = i^|j| J_|j|(2t) (Bessel
   functions) until it reaches the ends of the chain (t is chosen so that it has not).
6. The energy <psi|H|psi> changes only by O(dt^2) (no drift).
7. Vectorized calls (`evaluate_complex`, SIMD) over random states and potentials, against numpy.

    python quantum.py                 # 20 sites, 2,500 steps
    python quantum.py -S 12 -N 1000   # other sizes
    python quantum.py --no-simd       # symjit options as in the other examples (util.py)
"""

import argparse
import sys
import time

import util

opts = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False)
opts.add_argument("-S", type=int, default=20, help="lattice sites (default 20)")
opts.add_argument("-N", type=int, default=2500, help="time steps (default 2500)")
opts.add_argument("--dt", type=float, default=0.01, help="the step of the checks (default 0.01)")
opts.add_argument("--batch", type=int, default=1000, help="states of the vectorized check (default 1000)")
our, rest = opts.parse_known_args()
if "-h" in rest or "--help" in rest:
    opts.print_help()
    sys.exit(0)
sys.argv = sys.argv[:1] + rest
args = util.process_argv()
args["dtype"] = "complex128"
if args.get("ty", "native").split(";")[0] in ("bytecode", "debug", "wasm"):
    sys.exit(f"compile_composer does not support ty={args['ty']} (see kepler.py)")

import numpy as np  # noqa: E402
from scipy.special import jv  # noqa: E402
from symjit import Composer, compile_composer  # noqa: E402

S, N, DT = our.S, our.N, our.dt


# ------------------------------------------------------------------ the programs
def prepare(cp, V, dt):
    """the potential phases and the bond rotations' cosines and i*sines"""
    half = cp.fmul(cp.constant(0.5), dt)
    phases = [cp.exp(cp.fmul(cp.constant(-1j), cp.fmul(v, half))) for v in V]
    rot_half = (cp.cos(half), cp.fmul(cp.constant(1j), cp.sin(half)))
    rot_full = (cp.cos(dt), cp.fmul(cp.constant(1j), cp.sin(dt)))
    return phases, rot_half, rot_full


def step(cp, psi, phases, rot_half, rot_full):
    """one split-operator step on the list of amplitudes psi; returns the new list"""
    def bonds(psi, first, rot):
        c, s = rot
        psi = list(psi)
        for j in range(first, S - 1, 2):
            a, b = psi[j], psi[j + 1]
            psi[j] = cp.fadd(cp.fmul(c, a), cp.fmul(s, b))
            psi[j + 1] = cp.fadd(cp.fmul(s, a), cp.fmul(c, b))
        return psi

    psi = [cp.fmul(p, q) for p, q in zip(phases, psi)]
    psi = bonds(psi, 0, rot_half)
    psi = bonds(psi, 1, rot_full)
    psi = bonds(psi, 0, rot_half)
    return [cp.fmul(p, q) for p, q in zip(phases, psi)]


def unrolled(n):
    cp = Composer(2 * S + 1, S)
    consts = prepare(cp, [cp.arg(S + j) for j in range(S)], cp.arg(2 * S))
    psi = [cp.arg(j) for j in range(S)]
    for _ in range(n):
        psi = step(cp, psi, *consts)
    for j in range(S):
        cp.assign(cp.out(j), psi[j])
    return cp


def looped(n):
    cp = Composer(2 * S + 1, S)
    consts = prepare(cp, [cp.arg(S + j) for j in range(S)], cp.arg(2 * S))
    state = [cp.new_temp() for _ in range(S)]
    for j in range(S):
        cp.assign(state[j], cp.arg(j))
    body = cp.new_block()
    for t, v in zip(state, step(body, state, *consts)):
        body.assign(t, v)
    cp.append_for(cp.new_temp(), 0, n, body)
    for j in range(S):
        cp.assign(cp.out(j), state[j])
    return cp


def steps_numpy(psi, V, dt, n):
    """the same steps in numpy; psi and V are S x batch (or S) arrays"""
    psi = np.array(psi, dtype=complex)
    phases = np.exp(-1j * np.asarray(V) * dt / 2)

    def bonds(psi, first, th):
        c, s = np.cos(th), 1j * np.sin(th)
        a, b = psi[first:S - 1:2].copy(), psi[first + 1:S:2].copy()
        psi[first:S - 1:2], psi[first + 1:S:2] = c * a + s * b, s * a + c * b

    for _ in range(n):
        psi *= phases
        bonds(psi, 0, dt / 2)
        bonds(psi, 1, dt)
        bonds(psi, 0, dt / 2)
        psi *= phases
    return psi


def hamiltonian(V):
    return np.diag(V) - np.eye(S, k=1) - np.eye(S, k=-1)


def exact(psi, V, t):
    w, U = np.linalg.eigh(hamiltonian(V))
    return U @ (np.exp(-1j * w * t) * (U.conj().T @ psi))


t0 = time.perf_counter()
cp = unrolled(N)
t1 = time.perf_counter()
f = compile_composer(cp, **args)
t2 = time.perf_counter()
g = compile_composer(looped(N), **args)
print(f"{S} sites, {N:,} split-operator steps: Composer program {t1 - t0:.2f} s, symjit compilation (complex128) "
      f"{t2 - t1:.2f} s, {f.measure('mir-size'):,} instructions, {f.measure('ker-scalar-size') / 2**20:.2f} MiB of "
      f"machine code (the loop: {g.measure('ker-scalar-size'):,} bytes)\n")

ok = True


def report(name, passed, detail):
    global ok
    ok &= passed
    print(f"{'ok  ' if passed else 'FAIL'} {name}: {detail}")


def run(fn, psi, V, dt):
    return np.ravel(fn(*psi, *V, dt))[:S]


rng = np.random.default_rng(1)
psi0 = rng.normal(size=S) + 1j * rng.normal(size=S)
psi0 /= np.linalg.norm(psi0)
V = rng.uniform(-1, 1, S)

# 1. the unrolled code, the loop and numpy
t0 = time.perf_counter()
u = run(f, psi0, V, DT)
t1 = time.perf_counter()
w = run(g, psi0, V, DT)
t2 = time.perf_counter()
z = steps_numpy(psi0, V, DT, N)
t3 = time.perf_counter()
report("same steps", np.max(np.abs(u - w)) < 1e-12 and np.max(np.abs(u - z)) < 1e-12,
       f"unrolled vs loop {np.max(np.abs(u - w)):.1e}, vs numpy {np.max(np.abs(u - z)):.1e}; one state: unrolled "
       f"{1e3 * (t1 - t0):.1f} ms, loop {1e3 * (t2 - t1):.1f} ms, numpy {1e3 * (t3 - t2):.0f} ms")

# 2. the norm
report("norm", abs(np.linalg.norm(u) - 1) < 1e-12, f"|psi| - 1 = {np.linalg.norm(u) - 1:.1e} after {N:,} steps")

# 3. time reversal
back = run(f, u, V, -DT)
report("time reversal", np.max(np.abs(back - psi0)) < 1e-12,
       f"{N:,} steps with dt and {N:,} with -dt: max |psi - psi_0| = {np.max(np.abs(back - psi0)):.1e}")

# 4. the exact solution: the error is O(dt^3 N)
errs = [np.max(np.abs(run(f, psi0, V, dt) - exact(psi0, V, N * dt))) for dt in (2 * DT, DT, DT / 2)]
rate = np.sqrt(errs[0] / errs[2])
report("exact solution", errs[0] > errs[1] > errs[2] and 5 < rate < 12,
       f"max error at t = N dt for dt = {2 * DT:g}, {DT:g}, {DT / 2:g}: " + ", ".join(f"{e:.2e}" for e in errs)
       + f"; it falls by {rate:.1f} per halving of dt (about 8 expected)")

# 5. a free particle: Bessel functions
c = S // 2
start = np.zeros(S, dtype=complex)
start[c] = 1
# the longest time before the wave reaches the ends (edge amplitude below 1e-8)
T = max(t for t in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0) if abs(jv(c, 2 * t)) < 1e-8 or t == 0.1)
psi = run(f, start, np.zeros(S), T / N)
j = np.arange(S) - c
bessel = 1j ** np.abs(j) * jv(np.abs(j), 2 * T)
d = np.max(np.abs(psi - bessel))
report("free particle", d < 1e-7, f"from site {c} at t = {T:g}: max |psi_j - i^|j| J_|j|(2t)| = {d:.1e} "
       f"(the edges: |J_{c}(2t)| = {abs(jv(c, 2 * T)):.0e})")

# 6. the energy
E0, E = (np.real(np.vdot(p, hamiltonian(V) @ p)) for p in (psi0, u))
report("energy", abs(E - E0) < 10 * DT ** 2, f"<H> changed by {E - E0:.1e} over t = {N * DT:g} (dt^2 = {DT ** 2:g})")

# 7. vectorized over random states and potentials
K = our.batch
P0 = rng.normal(size=(S, K)) + 1j * rng.normal(size=(S, K))
P0 /= np.linalg.norm(P0, axis=0)
Vs = rng.uniform(-1, 1, (S, K))
X = np.vstack([P0, Vs, np.full((1, K), DT)]).T.astype(complex)
f.evaluate_complex(X[:8])  # the first call may prepare the SIMD kernel
t0 = time.perf_counter()
U = f.evaluate_complex(X)
t1 = time.perf_counter()
Z = steps_numpy(P0, Vs, DT, N).T
t2 = time.perf_counter()
d = np.max(np.abs(U - Z))
norms = np.max(np.abs(np.linalg.norm(U, axis=1) - 1))
report("vectorized", d < 1e-12 and norms < 1e-12,
       f"{K:,} states: unrolled {1e3 * (t1 - t0):.0f} ms, numpy {1e3 * (t2 - t1):.0f} ms; max difference {d:.1e}, "
       f"norms within {norms:.1e} of 1")

print(f"\nmachine code: {f.measure('ker-scalar-size') / 2**20:.2f} MiB (scalar), "
      f"{f.measure('ker-simd-size') / 2**20:.2f} MiB (SIMD)")
print("all checks passed" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)

# %%
import sympy as sm
import sympy.physics.mechanics as me
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, minimize, root
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import permutations

from symjit import compile_func

# %%
# This models a n link pendulum, same like the one in JM's lecture, only in 3D.
# $n$ balls of radius $r$ and mass $m$ are located at the center of each
# massless rod. An observer of mass $m_o$ may be attached to each ball,
# its distance rom the center of each ball is $\beta \cdot r$ perpendcular
# to the massless rod the ball is attached to.\
# Collisions between balls are handled with the **Hunt-Crossley** method.

# %%
start = time.time()

# here are n balls, and (n+1) frames, including the inertial frame N
# ===================
n = 2
# ===================
term_info = True
# ===================
frictionless = False
# ===================

if n < 2 or isinstance(n, int) == False:
    raise Exception("n must be an integer >= 2")

t = me.dynamicsymbols._t

aux0x, aux0y, aux0z, auxnx, auxny, auxnz = me.dynamicsymbols(
    "aux0x, aux0y, aux0z, auxnx, auxny, auxnz"
)
f0x, f0y, f0z, fnx, fny, fnz = me.dynamicsymbols("f0x, f0y, f0z, fnx, fny, fnz")
aux = [aux0x, aux0y, aux0z, auxnx, auxny, auxnz]
F = [f0x, f0y, f0z, fnx, fny, fnz]
rhs = [sm.symbols("rhs" + str(i)) for i in range(3 * n, 6 * n)]

N = me.ReferenceFrame("N")
P0 = me.Point("P0")
P0.set_vel(N, aux0x * N.x + aux0y * N.y + aux0z * N.z)

A = [N]
Dmc = []
Po = []
P = [P0]
q1 = []
u1 = []
l = []

for i in range(1, n + 1):
    P.append(me.Point("P" + str(i)))
    Dmc.append(me.Point("Dmc" + str(i)))
    Po.append(me.Point("Po" + str(i)))
    A.append(me.ReferenceFrame("A" + str(i)))

    q1.append([me.dynamicsymbols("q" + k + str(i)) for k in ("x", "y", "z")])
    u1.append([me.dynamicsymbols("u" + k + str(i)) for k in ("x", "y", "z")])

    l.append(sm.symbols("l" + str(i)))


m, mo, g, r, k, kb, d, l0, ln, beta, ctau, mu = sm.symbols(
    "m, mo, g, r, k, kb, d, l0, ln, beta, ctau, mu"
)
iXX, iYY, iZZ = sm.symbols("iXX, iYY, iZZ")
rhodtmax = [
    sm.symbols("rhodt_" + str(i) + str(j)) for i, j in permutations(range(n), r=2)
]

rot = []  # used for the kinematic equations below.
rot1 = []  # dto.

for i in range(1, n + 1):
    A[i].orient_body_fixed(A[0], (q1[i - 1][1], q1[i - 1][0], q1[i - 1][2]), "213")
    rot.append(A[i].ang_vel_in(A[0]))
    A[i].set_ang_vel(
        A[0], u1[i - 1][0] * A[i].x + u1[i - 1][1] * A[i].y + u1[i - 1][2] * A[i].z
    )
    rot1.append(A[i].ang_vel_in(A[0]))

    P[i].set_pos(P[i - 1], l[i - 1] * A[i].y)
    P[i].v2pt_theory(P[i - 1], A[0], A[i])

    Dmc[i - 1].set_pos(P[i - 1], l[i - 1] / 2.0 * A[i].y)
    Dmc[i - 1].v2pt_theory(P[i - 1], A[0], A[i])

    Po[i - 1].set_pos(Dmc[i - 1], beta * r * A[i].x)
    Po[i - 1].v2pt_theory(Dmc[i - 1], A[0], A[i])

Pn = me.Point("Pn")
Pn.set_pos(P[0], ln * A[0].x)
Pn.set_vel(A[0], auxnx * A[0].x + auxny * A[0].y + auxnz * A[0].z)


gleichung = []
for i in range(n):
    for uv in A[i + 1]:
        gleichung.append(me.dot(rot[i] - rot1[i], uv))
variablen = [sm.Derivative(j, t) for i in range(n) for j in q1[i]]

antwort = sm.solve(gleichung, variablen)
zaehler = 0.0
for zahl in variablen:
    zaehler += antwort[zahl].count_ops(visual=False)
print(f"antwort contains {zaehler:.0f} operations")

# %%
# This is needed to get initial generalized coordinates and speeds compatible
# with the configuration constraint and the resulting speed constraints.
# Here the dictionary *antwort* is needed.
# lnitial generalized coordinates and generalized speeds, which are compatible
# with the configuration constraint and the resulting speed constraints are
# found numerically before the integration starts.

# %%
distanz = [P[i + 1].pos_from(P[i]) for i in range(n)]
hol = sum(distanz) - ln * A[0].x
holm = me.dot(hol, hol)

holx = me.dot(hol, A[0].x)
holy = me.dot(hol, A[0].y)
holz = me.dot(hol, A[0].z)

hol_mat = sm.Matrix([holx, holy, holz])

holxdt = holx.diff(t).subs(antwort)
holydt = holy.diff(t).subs(antwort)
holzdt = holz.diff(t).subs(antwort)

speed_mat = sm.Matrix([holxdt, holydt, holzdt])

matrix_A = speed_mat.jacobian((u1[-2][2], u1[-1][0], u1[-1][2]))
vector_b = speed_mat.subs({u1[-2][2]: 0.0, u1[-1][0]: 0.0, u1[-1][2]: 0.0})
loesung = matrix_A.LUsolve(-vector_b)

if term_info == True:
    print(
        "loesung DS",
        set().union(*[me.find_dynamicsymbols(loesung[i]) for i in range(3)]),
    )
    print(
        f"loesung has {sm.count_ops(loesung):,} operations, "
        f"{sm.count_ops(sm.cse(loesung)):,} after cse"
    )

# %% [markdown]
# This function calculates the **forces** and the **torques** when two balls
# collide, according to Hunt-Crossley' theory.
# NOTE:
# As symjit does not accept Heaviside / Max functions, they are replaced
# by a smooth version / Piecewise((..), (..)) respectively.


# %%
def smooth_Heaviside(x, a, steep=20):
    """A smooth approximation of the Heaviside step function."""
    xx = steep * (x - a)
    return 0.5 * (1 + sm.tanh(xx))


def HC_disc(N, A1, A2, P1, P2, r, ctau, rhodtmax, k0):
    """
    This function returns the forces, torques on P2, when colliding with P1.
    """
    CP01 = me.Point("CP01")
    vektor = P2.pos_from(P1)
    richtung = vektor.normalize()
    abstand = vektor.magnitude()
    rho = 2.0 * r - abstand
    CP01.set_pos(P1, 0.5 * abstand * richtung)
    vCP01 = CP01.v2pt_theory(P1, N, A1)
    rhodt = me.dot(vCP01, richtung)
    # rho = sm.sqrt(rho**2) #sm.Max(rho, sm.S(0))
    rho = sm.Piecewise((rho, rho >= 0), (0, True))

    forcec = (
        k0
        * rho ** (3 / 2)
        * (1.0 + 3.0 / 2.0 * (1 - ctau) * rhodt / rhodtmax)
        * (richtung * smooth_Heaviside(2.0 * r - abstand, 0.0))
    )

    CP02 = me.Point("CP02")
    CP02.set_pos(P2, -0.5 * abstand * richtung)
    vCP02 = CP02.v2pt_theory(P2, N, A2)

    friction_force = forcec.magnitude() * mu * -(CP02.vel(N) - CP01.vel(N))
    hilfs = CP02.pos_from(P2)
    torque = hilfs.cross(friction_force) * smooth_Heaviside(2.0 * r - abstand, 0.0)
    forcef = (
        1.0
        / me.dot(hilfs, hilfs)
        * torque.cross(hilfs)
        * smooth_Heaviside(2.0 * r - abstand, 0.0)
    )

    return [forcec, forcef, torque]


# %%
# **Body list and list of forces for Kane's equations**

# %%
I = [me.inertia(A[i], iXX, iYY, iZZ) for i in range(1, n + 1)]

body = [
    me.RigidBody("body" + str(i + 1), Dmc[i], A[i + 1], m, (I[i], Dmc[i]))
    for i in range(0, n)
]
obs = [me.Particle("obs" + str(i + 1), Po[i], mo) for i in range(0, n)]
BODY = body + obs

# Gravitational forces
punkte = Dmc + Po
massen = [m] * n + [mo] * n
FL1 = [(punkt, -masse * g * A[0].y) for punkt, masse in zip(punkte, massen)]

# Reaction forces
FL_react = [
    (P[0], f0x * A[0].x + f0y * A[0].y + f0z * A[0].z),
    (Pn, fnx * A[0].x + fny * A[0].y + fnz * A[0].z),
]

# forces due to the balls hitting each other. My assumption is that never
# will 3 balls hit each other at the same time.
FB = []
distanz_Dmc = [
    Dmc[i].pos_from(Dmc[j]).magnitude() for i, j in permutations(range(n), r=2)
]
rhodtmax1 = [
    me.dot(
        Dmc[i].pos_from(Dmc[j]).diff(t, N).subs(antwort),
        Dmc[i].pos_from(Dmc[j]).normalize(),
    )
    for i, j in permutations(range(n), r=2)
]
print(
    "rhodtmax1 DS",
    set.union(
        *[
            me.find_dynamicsymbols(rhodtmax1[j])
            for j in range(len(list(permutations(range(n), r=2))))
        ]
    ),
)
zaehler = -1

for i, j in permutations(range(n), r=2):
    zaehler += 1
    FB.append(
        (
            Dmc[i],
            HC_disc(
                N, A[j + 1], A[i + 1], Dmc[j], Dmc[i], r, ctau, rhodtmax[zaehler], kb
            )[0],
        )
    )
    if frictionless == False:
        FB.append(
            (
                Dmc[i],
                HC_disc(
                    N,
                    A[j + 1],
                    A[i + 1],
                    Dmc[j],
                    Dmc[i],
                    r,
                    ctau,
                    rhodtmax[zaehler],
                    kb,
                )[1],
            )
        )
        FB.append(
            (
                A[i + 1],
                HC_disc(
                    N,
                    A[j + 1],
                    A[i + 1],
                    Dmc[j],
                    Dmc[i],
                    r,
                    ctau,
                    rhodtmax[zaehler],
                    kb,
                )[2],
            )
        )

# Forces due to the spring from P_n to Pn
PPF = Pn.pos_from(P[n])
PPFx = me.dot(PPF, A[0].x)
PPFy = me.dot(PPF, A[0].y)
PPFz = me.dot(PPF, A[0].z)
PPF_len = sm.sqrt(PPFx**2 + PPFy**2 + PPFz**2)

# to avoid divisions by zero
test = 1.0 / sm.Piecewise((PPF_len, PPF_len > 1.0e-20), (1.0e-20, True))

FL2 = [(P[n], -k * (l0 - PPF_len) * PPF * test), (Pn, k * (l0 - PPF_len) * PPF * test)]

FL = FL1 + FL2 + FL_react + FB

# %%
# **Kane's equations**
#
# Note, that I do not use velocity constraints, but use the spring to enforce
# the configuration constraint.
#

# q1, u1 need to be 'flattened' for use in Kane's set up.
q1h = [q1[i][j] for i in range(0, len(q1)) for j in range(3)]
u1h = [u1[i][j] for i in range(0, len(u1)) for j in range(3)]

q_ind = q1h
u_ind = u1h
kd = [me.dot(rot[i] - rot1[i], uv) for i in range(n) for uv in A[0]]
reaction_forces = [f0x, f0y, f0z, fnx, fny, fnz]

KM = me.KanesMethod(
    N,
    q_ind=q_ind,
    u_ind=u_ind,
    kd_eqs=kd,
    u_auxiliary=aux,
)
fr, frstar = KM.kanes_equations(BODY, FL)

MM = KM.mass_matrix_full
if term_info == True:
    print("MM DS", me.find_dynamicsymbols(MM))
    print("MM free symbols", MM.free_symbols)
    print(
        f"MM has {sm.count_ops(MM):,} operations, "
        f"{sm.count_ops(sm.cse(MM)):,} after cse",
        "\n",
    )

force = KM.forcing_full.subs({i: sm.S(0.0) for i in aux})
if term_info == True:
    print("force DS", me.find_dynamicsymbols(force))
    print("force free symbols", force.free_symbols)
    print(
        f"force has {sm.count_ops(force):,} operations, "
        f"{sm.count_ops(sm.cse(force)):,} after cse",
        "\n",
    )

print("it took {:.3f} sec to get Kanes equations".format(time.time() - start))

# %% [markdown]
# Convert the sympy functions to numpy functions to do numeric calculations.
# *cse=True* speeds up the numerical integration a lot

# %%
start1 = time.time()
qL = q1h + u1h
pL = [m, mo, g, r, k, kb, d, l0, ln, beta, mu, ctau] + l + [iXX, iYY, iZZ] + rhodtmax
pL1 = [m, mo, g, r, k, kb, d, l0, ln, beta, mu, ctau] + l + [iXX, iYY, iZZ]

# q_ind1, q_dep1, u_ind1, up_dep1 are needed further down in the lambdification
u_ind1 = deepcopy(u1h)  # deepcopy needed to avoid 'destroying' u1h
hilfs1 = u_ind1.pop(-1)
hilfs2 = u_ind1.pop(-2)
hilfs3 = u_ind1.pop(-2)
u_dep1 = [hilfs3, hilfs2, hilfs1]

q_ind1 = deepcopy(q1h)
hilfs1 = q_ind1.pop(-1)
hilfs2 = q_ind1.pop(-2)
hilfs3 = q_ind1.pop(-2)
q_dep1 = [hilfs3, hilfs2, hilfs1]

MM_lam = sm.lambdify(qL + pL, MM, cse=True)
force_lam = sm.lambdify(qL + pL, force, cse=True)

print(
    " it took {:.3f} sec for the lambdification of force and of MM".format(
        time.time() - start1
    )
)

# %% [markdown]
# Use symjit instead of lambdify

# %%
start1 = time.time()
qL = q1h + u1h
pL = [m, mo, g, r, k, kb, d, l0, ln, beta, mu, ctau] + l + [iXX, iYY, iZZ] + rhodtmax
pL1 = [m, mo, g, r, k, kb, d, l0, ln, beta, mu, ctau] + l + [iXX, iYY, iZZ]

# q_ind1, q_dep1, u_ind1, up_dep1 are needed further down in the lambdification
u_ind1 = deepcopy(u1h)  # deepcopy needed to avoid 'destroying' u1h
hilfs1 = u_ind1.pop(-1)
hilfs2 = u_ind1.pop(-2)
hilfs3 = u_ind1.pop(-2)
u_dep1 = [hilfs3, hilfs2, hilfs1]

q_ind1 = deepcopy(q1h)
hilfs1 = q_ind1.pop(-1)
hilfs2 = q_ind1.pop(-2)
hilfs3 = q_ind1.pop(-2)
q_dep1 = [hilfs3, hilfs2, hilfs1]

start3 = time.time()
w1 = sm.symbols(f"w:{len(q1h)}")
v1 = sm.symbols(f"v:{len(u1h)}")
dict_w = {q1h[i]: w1[i] for i in range(len(q1h))}
dict_v = {u1h[i]: v1[i] for i in range(len(u1h))}
MM1 = me.msubs(MM, dict_w, dict_v)
force1 = me.msubs(force, dict_w, dict_v)
MM1 = [MM1[i, j] for i in range(MM1.shape[0]) for j in range(MM1.shape[1])]
force1 = list(force1)

#############################################################
#############################################################
print("w1", w1)
print("v1", v1)
# print('MM1', MM1)
# print('force1', force1)

MM_jit = compile_func((*w1, *v1), MM1, params=pL, cse=True)
force_jit = compile_func((*w1, *v1), force1, params=pL, cse=True)

print(f"it took {time.time() - start3:.3f} sec to do compile_func")
#############################################################

# %%
start1 = time.time()

# find initial values of the generalized coordinates which satisfy the
# configuration constraint
holm_lam = sm.lambdify(q1h + pL, holm, cse=True)

# to get the initial values of the dependent speeds
speed_mat1_lam = sm.lambdify(u_dep1 + q1h + u_ind1 + pL, speed_mat, cse=True)
speed_mat11_lam = sm.lambdify(qL + pL, speed_mat, cse=True)

loesung_lam = sm.lambdify(q1h + u_ind1 + pL, loesung, cse=True)

loc_lam = sm.lambdify(q_dep1 + q_ind1 + pL, hol_mat, cse=True)

crash_lam = sm.lambdify(q1h + pL, distanz_Dmc, cse=True)

matrix_A_lam = sm.lambdify(q1h + u_ind1 + pL, matrix_A, cse=True)
# P4pos_lam = sm.lambdify(qL + pL, P4pos, cse=True)

rhodtmax1_lam = sm.lambdify(qL + pL1, rhodtmax1, cse=True)
distanz_Dmc_lam = sm.lambdify(qL + pL1, distanz_Dmc, cse=True)

print(f"it took {time.time() - start1:.3f} sec for this lambdification")

# %%
# **Initial values** for the subsequent **Numerical Integration**
#

# %%
# ============================
# Input variables
# the names are the same as in setting up Kane's equations, except
# that a "1" is appended.
# ============================
m1 = 1.0e0
mo1 = 1.0e-1
g1 = 9.8
r1 = 1.0
k1 = 1.0e7
kb1 = 1.0e4
d1 = 1.0e3
l01 = 1.0e-7

mu1 = 0.25
ctau1 = 0.8


l1 = [4.0 + 1.0 * (i + 1) for i in range(len(l))]
ln1 = np.sum(l1) / 2.5
print("length of the rods:", l1)
print("distance from P0 to Pn, that is ln:", ln1, "\n")
beta1 = 0.95

u11 = [(-1.0) ** i * 1.0 for i in range(len(u1h))]
rhodtmax1 = [1.0 for i, j in permutations(range(n), r=2)]
intervall = 1.0
# =====================================================================================
schritte = int(intervall * 3000)
times = np.linspace(0.0, intervall, schritte)

iXX1 = 2.0 / 5.0 * m1 * r1**2  # from the internet.
iYY1, iZZ1 = iXX1, iXX1

pL_vals = (
    [m1, mo1, g1, r1, k1, kb1, d1, l01, ln1, beta1, mu1, ctau1]
    + l1
    + [iXX1, iYY1, iZZ1]
    + rhodtmax1
)
pL1_vals = (
    [m1, mo1, g1, r1, k1, kb1, d1, l01, ln1, beta1, mu1, ctau1]
    + l1
    + [iXX1, iYY1, iZZ1]
)


def func(x0, args):
    return holm_lam(*x0, *args)


def func2(x0, args):
    return loc_lam(*x0, *args).reshape(3)


# find good consistent initial generalized coordinates
x0 = [1] * len(q1h)
for i in range(10):
    anfangs_q = minimize(func, x0, pL_vals)
    anfangs_q = list(anfangs_q.x)
    x0 = anfangs_q

print(
    f"config constraint violated by initial guess: "
    f"{f'{holm_lam(*anfangs_q, *pL_vals):0.3e}':>20}"
)

# improve the initial generalized coordinates.
# anfangs_q1 may be manipulated without affecting anfangs_q
anfangs_q1 = deepcopy(anfangs_q)  #
hilfs1 = anfangs_q1.pop(-1)
hilfs2 = anfangs_q1.pop(-2)
hilfs3 = anfangs_q1.pop(-2)

args = anfangs_q1 + pL_vals
x0 = [hilfs3, hilfs2, hilfs1]
AAA = fsolve(func2, x0, args)
anfangs_q[-4] = AAA[0]
anfangs_q[-3] = AAA[1]
anfangs_q[-1] = AAA[2]
print(
    f"after improvement, config constraint violated by: "
    f"{f'{holm_lam(*anfangs_q, *pL_vals):0.3e}':>15}",
    "\n",
)

# Setting the natural length of the spring a bit smaller that the
# initial error of the location of P_n from Pn may help.
pL_vals[7] = holm_lam(*anfangs_q, *pL_vals) / 2.0

# avoid initial conditions, where the balls are not separated
# from each other
zaehler = -1
for i, j in permutations(range(n), r=2):
    if i < j:
        zaehler += 1
        print(
            (
                f"initial distance Dmc_{i} from Dmc_{j} is "
                f"{f'{crash_lam(*anfangs_q, *pL_vals)[zaehler]:0.3f}':>5}"
            )
        )

        if np.min(crash_lam(*anfangs_q, *pL_vals)) <= 2.0 * r1:
            raise Exception("Balls are too close together")


def func1(x0, args):
    return speed_mat1_lam(*x0, *args).reshape(3)


anfangs_u_ind = deepcopy(u11)
anfangs_u_ind.pop(-1)
anfangs_u_ind.pop(-2)
anfangs_u_ind.pop(-2)

hilfs1, hilfs2, hilfs3 = loesung_lam(*anfangs_q, *anfangs_u_ind, *pL_vals)
hilfs1 = hilfs1[0]
hilfs2 = hilfs2[0]
hilfs3 = hilfs3[0]
anfangs_u_dep = [hilfs1, hilfs2, hilfs3]

args1 = anfangs_q + anfangs_u_ind + pL_vals
print(
    (
        "\n",
        "Violation of speed constraints:",
        [f"{speed_mat1_lam(*anfangs_u_dep, *args1)[i][0]:0.3e}" for i in range(3)],
        "\n",
    )
)

matrix_A1_det = np.linalg.det(matrix_A_lam(*anfangs_q, *anfangs_u_ind, *pL_vals))
print(f"Determinante of matrix_A, used to determine u_dep = {matrix_A1_det:.3f} \n")

u11[-4] = hilfs1
u11[-3] = hilfs2
u11[-1] = hilfs3
y0 = anfangs_q + u11
print(
    (
        "initial conditions are:",
        "\n",
        "y0 =",
        [f"{y0[i]:.3f}" for i in range(len(y0))],
        "\n",
    )
)

# %%
# Numerical integration using numpy functions.

start2 = time.time()
t_span = (0.0, intervall)
laenge = len(list(permutations(range(n), r=2)))


def gradient(t, y, args):
    """
    Here I try to find the speed just before a collission takes place.
    """
    zaehler = -1
    laenge = -len(list(permutations(range(n), r=2)))
    for _, _ in permutations(range(n), r=2):
        zaehler += 1
        if 0.0 < 2.0 * r1 - distanz_Dmc_lam(*y, *pL1_vals)[zaehler] < 0.001:
            args[laenge] = rhodtmax1_lam(*y, *pL1_vals)[zaehler]
        laenge += 1
    sol = np.linalg.solve(MM_lam(*y, *args), force_lam(*y, *args))
    return np.array(sol).T[0]


resultat1 = solve_ivp(
    gradient, t_span, y0, t_eval=times, args=(pL_vals,), atol=1.0e-9, rtol=1.0e-9
)
resultat = resultat1.y.T
print("resultat shape", resultat.shape, "\n")
print(resultat1.message, "\n")

print(
    "To numerically integrate an intervall of {} sec the routine cycled {} "
    "times and it took {:.3f} sec ".format(
        intervall, resultat1.nfev, time.time() - start2
    ),
    "\n",
)

# %%
# Plot some coordinates

schritte1 = resultat.shape[0]
times1 = times[0:schritte1]
bezeichnung = [str(i) for i in qL]

fig, ax = plt.subplots(figsize=(10, 5))
for i in (8, 9, 10):
    ax.plot(
        times1, [resultat[l, i] for l in range(resultat.shape[0])], label=bezeichnung[i]
    )
ax.legend()
ax.set_title(
    "Some generalized coordinates using numpy functions in solve_ivp (lambdify)"
)
_ = ax.set_xlabel("time (sec)")


# %%
# Use symjit
# In solve_ivp, if method='Radau' is used it raises an error,
# if no method is used it give an implausible solution.

start2 = time.time()
t_span = (0.0, intervall)
laenge = len(list(permutations(range(n), r=2)))


def gradient(t, y, args):
    """
    Here I try to find the speed just before a collission takes place.
    """
    zaehler = -1
    laenge = -len(list(permutations(range(n), r=2)))
    for _, _ in permutations(range(n), r=2):
        zaehler += 1
        if 0.0 < 2.0 * r1 - distanz_Dmc_lam(*y, *pL1_vals)[zaehler] < 0.001:
            args[laenge] = rhodtmax1_lam(*y, *pL1_vals)[zaehler]
        laenge += 1

    MM_matrix = np.array(MM_jit(*y, *args)).reshape(MM.shape)
    force_vector = np.array(force_jit(*y, *args)).reshape(MM.shape[0], 1)
    sol = np.linalg.solve(MM_matrix, force_vector)
    return np.array(sol).T[0]


resultat1 = solve_ivp(
    gradient, t_span, y0, t_eval=times, args=(pL_vals,), atol=1.0e-9, rtol=1.0e-9
)
resultat = resultat1.y.T
print("resultat shape", resultat.shape, "\n")
print(resultat1.message, "\n")

print(
    "To numerically integrate an intervall of {} sec the routine cycled {} "
    "times and it took {:.3f} sec ".format(
        intervall, resultat1.nfev, time.time() - start2
    ),
    "\n",
)

# %%
# Plot whichever generalized coordinates you want to see.

# %%
schritte1 = resultat.shape[0]
times1 = times[0:schritte1]
bezeichnung = [str(i) for i in qL]

fig, ax = plt.subplots(figsize=(10, 5))
for i in (8, 9, 10):
    ax.plot(
        times1, [resultat[l, i] for l in range(resultat.shape[0])], label=bezeichnung[i]
    )
ax.legend()
ax.set_title(
    "Some generalized coordinates using symjit functions in solve_ivp (symjit)"
)
_ = ax.set_xlabel("time (sec)")

plt.show()

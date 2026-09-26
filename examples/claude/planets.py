import util

args = util.process_argv()

import time

import numpy as np
from sympy import Float, atan, cos, floor, pi, sin, sqrt, symbols
from symjit import compile_func

# Approximate planetary ephemeris from Keplerian elements.
#
# The elements and their secular rates are the "Keplerian elements for
# approximate positions of the major planets" of E. M. Standish (JPL Solar System
# Dynamics), Table 2a, valid for 3000 BC -- 3000 AD:
#
#     a (au), e, I (deg), L (deg), varpi (deg), Omega (deg)  at J2000 and per century
#
# For Jupiter and the outer planets, the mean anomaly needs additional
# perturbation terms (mostly the great inequality of Jupiter/Saturn and the
# Uranus/Neptune resonance)
#
#     M = L - varpi + b T^2 + c cos(f T) + s sin(f T)
#
# with T in Julian centuries since J2000. The position then follows from
# Kepler's equation E - e sin(E) = M, which is solved with a fixed number of
# Newton iterations that are unrolled into the symbolic expression, so that
# symjit compiles the whole ephemeris (Kepler solve + rotations to the ecliptic)
# as a single function of T.

# fmt: off
#            a0           a1           e0          e1           I0          I1           L0           L1              varpi0       varpi1       Omega0       Omega1
ELEMENTS = {
    "Mercury": (0.38709843,  0.00000000, 0.20563661,  0.00002123,  7.00559432, -0.00590158,  252.25166724,  149472.67486623,   77.45771895,  0.15940013,   48.33961819, -0.12214182),
    "Venus":   (0.72332102, -0.00000026, 0.00676399, -0.00005107,  3.39777545,  0.00043494,  181.97970850,   58517.81560260,  131.76755713,  0.05679648,   76.67261496, -0.27274174),
    "Earth":   (1.00000018, -0.00000003, 0.01673163, -0.00003661, -0.00054346, -0.01337178,  100.46691572,   35999.37306329,  102.93005885,  0.31795260,   -5.11260389, -0.24123856),
    "Mars":    (1.52371243,  0.00000097, 0.09336511,  0.00009149,  1.85181869, -0.00724757,   -4.56813164,   19140.29934243,  -23.91744784,  0.45223625,   49.71320984, -0.26852431),
    "Jupiter": (5.20248019, -0.00002864, 0.04853590,  0.00018026,  1.29861416, -0.00322699,   34.33479152,    3034.90371757,   14.27495244,  0.18199196,  100.29282654,  0.13024619),
    "Saturn":  (9.54149883, -0.00003065, 0.05550825, -0.00032044,  2.49424102,  0.00451969,   50.07571329,    1222.11494724,   92.86136063,  0.54179478,  113.63998702, -0.25015002),
    "Uranus":  (19.18797948, -0.00020455, 0.04685740, -0.00001550,  0.77298127, -0.00180155,  314.20276625,     428.49512595,  172.43404441,  0.09266985,   73.96250215,  0.05739699),
    "Neptune": (30.06952752,  0.00006447, 0.00895439,  0.00000818,  1.77005520,  0.00022400,  304.22289287,     218.46515314,   46.68158724,  0.01009938,  131.78635853, -0.00606302),
    "Pluto":   (39.48686035,  0.00449751, 0.24885238,  0.00006016, 17.14104260,  0.00000501,  238.96535011,     145.18042903,  224.09702598, -0.00968827,  110.30167986, -0.00809981),
}

# extra terms (b, c, s, f) of the mean anomaly for Jupiter -- Pluto
PERTURBATIONS = {
    "Jupiter": (-0.00012452,  0.06064060, -0.35635438, 38.35125000),
    "Saturn":  ( 0.00025899, -0.13434469,  0.87320147, 38.35125000),
    "Uranus":  ( 0.00058331, -0.97731848,  0.17689245,  7.67025000),
    "Neptune": (-0.00041348,  0.68346318, -0.10162547,  7.67025000),
    "Pluto":   (-0.01262724,  0.0,         0.0,         0.0),
}
# fmt: on

BODIES = list(ELEMENTS)  # the "Earth" entry is the Earth-Moon barycenter
J2000 = 2451545.0
NEWTON_ITERATIONS = 4  # from M + e sin(M), this converges to machine precision for e < 0.3
OBLIQUITY = np.radians(23.43928)  # mean obliquity of the ecliptic at J2000


def angle(x):
    """Reduces an angle (in radians) to [-pi, pi)"""
    return x - 2 * pi * floor(x / (2 * pi) + 0.5)


def heliocentric_expressions(T, perturbed=True):
    """Symbolic heliocentric ecliptic (J2000) coordinates (x, y, z) in au of every body"""
    out = []

    for body in BODIES:
        a0, a1, e0, e1, I0, I1, L0, L1, w0, w1, N0, N1 = [Float(v, 17) for v in ELEMENTS[body]]
        a = a0 + a1 * T
        e = e0 + e1 * T
        I = (I0 + I1 * T) * pi / 180
        L = L0 + L1 * T
        varpi = w0 + w1 * T
        Omega = (N0 + N1 * T) * pi / 180

        M = L - varpi

        if perturbed and body in PERTURBATIONS:
            b, c, s, f = [Float(v, 17) for v in PERTURBATIONS[body]]
            M = M + b * T**2 + c * cos(f * T * pi / 180) + s * sin(f * T * pi / 180)

        M = angle(M * pi / 180)
        omega = varpi * pi / 180 - Omega  # argument of the perihelion

        # Kepler's equation: E - e sin(E) = M solved by Newton's method
        E = M + e * sin(M)
        for _ in range(NEWTON_ITERATIONS):
            E = E - (E - e * sin(E) - M) / (1 - e * cos(E))

        # coordinates in the orbital plane (x' axis toward the perihelion)
        xp = a * (cos(E) - e)
        yp = a * sqrt(1 - e * e) * sin(E)

        # rotation to the J2000 ecliptic frame
        co, so = cos(omega), sin(omega)
        cO, sO = cos(Omega), sin(Omega)
        cI, sI = cos(I), sin(I)

        x = (co * cO - so * sO * cI) * xp + (-so * cO - co * sO * cI) * yp
        y = (co * sO + so * cO * cI) * xp + (-so * sO + co * cO * cI) * yp
        z = (so * sI) * xp + (co * sI) * yp

        out += [x, y, z]

    return out


def heliocentric_numpy(T, perturbed=True):
    """The same model in plain numpy, solving Kepler's equation to full convergence"""
    T = np.asarray(T, dtype=float)
    out = []

    for body in BODIES:
        a0, a1, e0, e1, I0, I1, L0, L1, w0, w1, N0, N1 = ELEMENTS[body]
        a = a0 + a1 * T
        e = e0 + e1 * T
        I = np.radians(I0 + I1 * T)
        Omega = np.radians(N0 + N1 * T)
        varpi = w0 + w1 * T

        M = L0 + L1 * T - varpi
        if perturbed and body in PERTURBATIONS:
            b, c, s, f = PERTURBATIONS[body]
            M = M + b * T**2 + c * np.cos(np.radians(f * T)) + s * np.sin(np.radians(f * T))

        M = np.remainder(np.radians(M) + np.pi, 2 * np.pi) - np.pi
        omega = np.radians(varpi) - Omega

        E = M + e * np.sin(M)
        for _ in range(50):
            dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
            E = E - dE
            if np.max(np.abs(dE)) < 1e-15:
                break

        xp = a * (np.cos(E) - e)
        yp = a * np.sqrt(1 - e * e) * np.sin(E)

        co, so, cO, sO, cI, sI = np.cos(omega), np.sin(omega), np.cos(Omega), np.sin(Omega), np.cos(I), np.sin(I)
        out.append((co * cO - so * sO * cI) * xp + (-so * cO - co * sO * cI) * yp)
        out.append((co * sO + so * cO * cI) * xp + (-so * sO + co * cO * cI) * yp)
        out.append((so * sI) * xp + (co * sI) * yp)

    return np.array(out)


def to_equatorial(r):
    """Rotates ecliptic (x, y, z) to equatorial coordinates (J2000 mean equator)"""
    c, s = np.cos(OBLIQUITY), np.sin(OBLIQUITY)
    return np.array([r[0], c * r[1] - s * r[2], s * r[1] + c * r[2]])


def position(helio, body):
    """Heliocentric (x, y, z) of `body` from the (27, n) array of heliocentric coordinates"""
    i = BODIES.index(body)
    return helio[3 * i : 3 * i + 3]


def geocentric(helio, body):
    """Geocentric ecliptic vector of `body` from the (27, n) array of heliocentric coordinates"""
    return position(helio, body) - position(helio, "Earth")


def ra_dec(v):
    """Right ascension (hours), declination (degrees), and distance (au) from an equatorial vector"""
    ra = np.degrees(np.arctan2(v[1], v[0])) % 360.0 / 15.0
    r = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    dec = np.degrees(np.arcsin(v[2] / r))
    return ra, dec, r


def hms(h):
    m = (h - int(h)) * 60
    return f"{int(h):02d}h {int(m):02d}m {(m - int(m)) * 60:05.2f}s"


def dms(d):
    sign = "-" if d < 0 else "+"
    d = abs(d)
    m = (d - int(d)) * 60
    return f"{sign}{int(d):02d}° {int(m):02d}' {(m - int(m)) * 60:04.1f}\""


def julian_date(y, m, d):
    """Julian date of 0h of a Gregorian calendar day"""
    if m <= 2:
        y, m = y - 1, m + 12
    A = y // 100
    B = 2 - A + A // 4
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5


def angular_separation(u, v):
    cosine = np.sum(u * v, axis=0) / np.sqrt(np.sum(u * u, axis=0) * np.sum(v * v, axis=0))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


# ------------------------------------------------------------------------------
# compile the ephemeris

T = symbols("T")

t0 = time.time()
model = heliocentric_expressions(T)
t1 = time.time()
ephemeris = compile_func([T], model, **args)
t2 = time.time()

print(f"model built in {1000 * (t1 - t0):.1f} ms, compiled in {1000 * (t2 - t1):.1f} ms")
print(f"{len(model)} outputs ({len(BODIES)} bodies), {NEWTON_ITERATIONS} unrolled Newton iterations")

# ------------------------------------------------------------------------------
# 1. the compiled function reproduces the numpy reference (fully converged Kepler solver)

rng = np.random.default_rng(1)
Tv = rng.uniform(-30.0, 30.0, 100_000)  # 3000 BC to 3000 AD

got = np.asarray(ephemeris(Tv))
want = heliocentric_numpy(Tv)

err = np.max(np.abs(got - want))
print(f"max |symjit - numpy| over {len(Tv)} random epochs (x, y, z of all bodies): {err:.2e} au")
np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)

# a scalar call gives the same answer as the vectorized one
np.testing.assert_allclose(np.asarray(ephemeris(Tv[0])), got[:, 0], rtol=1e-12, atol=1e-12)

# ------------------------------------------------------------------------------
# 2. the ephemeris for a given date

# jd = julian_date(2025, 1, 1)

import pandas as pd
jd = pd.Timestamp.now(tz='UTC').to_julian_date()

Tn = (jd - J2000) / 36525.0
helio = np.asarray(ephemeris(Tn)).reshape(-1, 1)

print(f"\nGeocentric astrometric positions (J2000 equator), JD {jd}\n")
print(f"{'body':10s}{'RA':>14s}{'Dec':>16s}{'distance (au)':>16s}")

sun = to_equatorial(-position(helio, "Earth"))  # the Sun seen from the Earth
ra, dec, r = ra_dec(sun[:, 0])
print(f"{'Sun':10s}{hms(ra):>14s}{dms(dec):>16s}{r:>16.6f}")

for body in BODIES:
    if body == "Earth":
        continue
    v = to_equatorial(geocentric(helio, body))
    ra, dec, r = ra_dec(v[:, 0])
    print(f"{body:10s}{hms(ra):>14s}{dms(dec):>16s}{r:>16.6f}")

# ------------------------------------------------------------------------------
# 3. astronomical events found by scanning the compiled ephemeris

# great conjunction of Jupiter and Saturn: closest approach on 2020-12-21
days = np.arange(julian_date(2020, 12, 1), julian_date(2021, 1, 10), 0.01)
h = np.asarray(ephemeris((days - J2000) / 36525.0))
sep = angular_separation(geocentric(h, "Jupiter"), geocentric(h, "Saturn"))
k = np.argmin(sep)
print(f"\nJupiter-Saturn conjunction: JD {days[k]:.2f}, minimum separation {60 * sep[k]:.1f} arcmin (expected 2020-12-21)")
assert abs(days[k] - julian_date(2020, 12, 21)) < 2.0

# opposition of Mars: geocentric ecliptic longitude differs from the Sun's by 180 degrees on 2025-01-16
days = np.arange(julian_date(2025, 1, 1), julian_date(2025, 2, 1), 0.01)
h = np.asarray(ephemeris((days - J2000) / 36525.0))
mars = geocentric(h, "Mars")
sun_ecl = -position(h, "Earth")
dlon = (np.degrees(np.arctan2(mars[1], mars[0])) - np.degrees(np.arctan2(sun_ecl[1], sun_ecl[0]))) % 360.0 - 180.0
k = np.argmin(np.abs(dlon))
print(f"Mars opposition: JD {days[k]:.2f} (expected 2025-01-16 = JD {julian_date(2025, 1, 16):.1f})")
assert abs(days[k] - julian_date(2025, 1, 16)) < 2.0

# ------------------------------------------------------------------------------
# 4. the effect of the perturbation corrections, and accuracy against an
# independent theory (the Simon et al. series in ERFA), if pyerfa is installed.

try:
    import erfa
except ImportError:
    erfa = None
    print("\n(pyerfa is not installed: skipping the comparison with an independent ephemeris)")

if erfa is not None:
    T_check = np.linspace(-1.0, 1.0, 401)  # 1900 -- 2100
    with_terms = np.asarray(ephemeris(T_check))
    without_terms = np.asarray(compile_func([T], heliocentric_expressions(T, perturbed=False), **args)(T_check))

    print("\nHeliocentric direction error against ERFA plan94 (arcmin, max over 1900-2100)")
    print(f"{'body':10s}{'corrected':>12s}{'uncorrected':>14s}")

    # erfa.plan94 numbers: 1 Mercury, 2 Venus, 3 Earth-Moon barycenter, 4 Mars, ... 8 Neptune
    for n, body in enumerate(BODIES[:-1], start=1):
        jd_check = J2000 + 36525.0 * T_check
        pv = erfa.plan94(jd_check, 0.0, n)["p"]  # heliocentric, equatorial J2000, au
        ref = pv.T

        errors = []
        for h in (with_terms, without_terms):
            mine = to_equatorial(position(h, body))
            errors.append(60 * np.max(angular_separation(mine, ref)))

        print(f"{body:10s}{errors[0]:12.2f}{errors[1]:14.2f}")

        # Table 2a trades accuracy for its 6000-year validity: expect errors of up to ~20 arcmin
        assert errors[0] < 30.0
        if body in PERTURBATIONS:
            assert errors[0] <= errors[1] + 1e-9  # the corrections must not make things worse

# ------------------------------------------------------------------------------
# 5. speed: one compiled call versus the numpy reference on a large batch

Tv = np.linspace(-1.0, 1.0, 1_000_000)

t0 = time.time()
r1 = np.asarray(ephemeris(Tv))
t1 = time.time()
r2 = heliocentric_numpy(Tv)
t2 = time.time()

print(f"\n1M epochs: symjit {1000 * (t1 - t0):.1f} ms, numpy {1000 * (t2 - t1):.1f} ms")
np.testing.assert_allclose(r1, r2, rtol=1e-9, atol=1e-9)

print("ok!")

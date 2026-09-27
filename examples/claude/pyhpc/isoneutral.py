"""
Isoneutral mixing benchmark (pyhpc-benchmarks `isoneutral_mixing`).

Isopycnal diffusion for tracers following the functional formulation by Griffies et
al., adopted from MOM2.1 (as implemented in Veros). The computation is a stencil code:
it works on shifted slices of 3D arrays and writes the results into slices of the
output arrays.

symjit compiles point-wise kernels, so the port keeps the data movement (slicing,
shifting, broadcasting, and the in-place updates) in numpy (see `stencil.py` for how
shifted operands are passed without copies) and moves the arithmetic of each stencil
into fused kernels compiled with `compile_func`:

    rho    density derivatives at T cells
    grad   masked finite difference (tracer gradients)
    slope  isopycnal slopes, taper, and diffusivity at east/north faces (Ai_ez, Ai_nz)
    top    slopes, taper, and contributions to K_33 at top faces (Ai_bx, Ai_by)

Original: https://github.com/dionhaefner/pyhpc-benchmarks (public domain).
"""

import math

import numpy as np
import sympy as sp
from symjit import compile_func

from stencil import Grid

TOLERANCE = dict(rtol=1e-5, atol=1e-9)  # for comparing the symjit and numpy results
NAME = "isoneutral_mixing"

def generate_inputs(size):
    import numpy as np

    np.random.seed(17)

    shape = (
        math.ceil(2 * size ** (1 / 3)),
        math.ceil(2 * size ** (1 / 3)),
        math.ceil(0.25 * size ** (1 / 3)),
    )

    # masks
    maskT, maskU, maskV, maskW = (
        (np.random.rand(*shape) < 0.8).astype("float64") for _ in range(4)
    )

    # 1d arrays
    dxt, dxu = (np.random.randn(shape[0]) for _ in range(2))
    dyt, dyu = (np.random.randn(shape[1]) for _ in range(2))
    dzt, dzw, zt = (np.random.randn(shape[2]) for _ in range(3))
    cost, cosu = (np.random.randn(shape[1]) for _ in range(2))

    # 3d arrays
    K_iso, K_11, K_22, K_33 = (np.random.randn(*shape) for _ in range(4))

    # 4d arrays
    salt, temp = (np.random.randn(*shape, 3) for _ in range(2))

    # 5d arrays
    Ai_ez, Ai_nz, Ai_bx, Ai_by = (np.zeros((*shape, 2, 2)) for _ in range(4))

    return (
        maskT,
        maskU,
        maskV,
        maskW,
        dxt,
        dxu,
        dyt,
        dyu,
        dzt,
        dzw,
        cost,
        cosu,
        salt,
        temp,
        zt,
        K_iso,
        K_11,
        K_22,
        K_33,
        Ai_ez,
        Ai_nz,
        Ai_bx,
        Ai_by,
    )


# ------------------------------------------------------------------------------
# numpy version, from pyhpc-benchmarks

def _get_drhodT(salt, temp, p):
    rho0 = 1024.0
    z0 = 0.0
    theta0 = 283.0 - 273.15
    grav = 9.81
    betaT = 1.67e-4
    betaTs = 1e-5
    gammas = 1.1e-8

    zz = -p - z0
    thetas = temp - theta0
    return -(betaTs * thetas + betaT * (1 - gammas * grav * zz * rho0)) * rho0


def _get_drhodS(salt, temp, p):
    betaS = 0.78e-3
    rho0 = 1024.0
    return betaS * rho0 * np.ones_like(temp)


def isoneutral_diffusion_pre_numpy(
    maskT,
    maskU,
    maskV,
    maskW,
    dxt,
    dxu,
    dyt,
    dyu,
    dzt,
    dzw,
    cost,
    cosu,
    salt,
    temp,
    zt,
    K_iso,
    K_11,
    K_22,
    K_33,
    Ai_ez,
    Ai_nz,
    Ai_bx,
    Ai_by,
):
    """
    Isopycnal diffusion for tracer
    following functional formulation by Griffies et al
    Code adopted from MOM2.1
    """
    epsln = 1e-20
    iso_slopec = 1e-3
    iso_dslope = 1e-3
    K_iso_steep = 50.0
    tau = 0

    dTdx = np.zeros_like(K_11)
    dSdx = np.zeros_like(K_11)
    dTdy = np.zeros_like(K_11)
    dSdy = np.zeros_like(K_11)
    dTdz = np.zeros_like(K_11)
    dSdz = np.zeros_like(K_11)

    """
    drho_dt and drho_ds at centers of T cells
    """
    drdT = maskT * _get_drhodT(salt[:, :, :, tau], temp[:, :, :, tau], np.abs(zt))
    drdS = maskT * _get_drhodS(salt[:, :, :, tau], temp[:, :, :, tau], np.abs(zt))

    """
    gradients at top face of T cells
    """
    dTdz[:, :, :-1] = (
        maskW[:, :, :-1]
        * (temp[:, :, 1:, tau] - temp[:, :, :-1, tau])
        / dzw[np.newaxis, np.newaxis, :-1]
    )
    dSdz[:, :, :-1] = (
        maskW[:, :, :-1]
        * (salt[:, :, 1:, tau] - salt[:, :, :-1, tau])
        / dzw[np.newaxis, np.newaxis, :-1]
    )

    """
    gradients at eastern face of T cells
    """
    dTdx[:-1, :, :] = (
        maskU[:-1, :, :]
        * (temp[1:, :, :, tau] - temp[:-1, :, :, tau])
        / (dxu[:-1, np.newaxis, np.newaxis] * cost[np.newaxis, :, np.newaxis])
    )
    dSdx[:-1, :, :] = (
        maskU[:-1, :, :]
        * (salt[1:, :, :, tau] - salt[:-1, :, :, tau])
        / (dxu[:-1, np.newaxis, np.newaxis] * cost[np.newaxis, :, np.newaxis])
    )

    """
    gradients at northern face of T cells
    """
    dTdy[:, :-1, :] = (
        maskV[:, :-1, :]
        * (temp[:, 1:, :, tau] - temp[:, :-1, :, tau])
        / dyu[np.newaxis, :-1, np.newaxis]
    )
    dSdy[:, :-1, :] = (
        maskV[:, :-1, :]
        * (salt[:, 1:, :, tau] - salt[:, :-1, :, tau])
        / dyu[np.newaxis, :-1, np.newaxis]
    )

    def dm_taper(sx):
        """
        tapering function for isopycnal slopes
        """
        return 0.5 * (1.0 + np.tanh((-np.abs(sx) + iso_slopec) / iso_dslope))

    """
    Compute Ai_ez and K11 on center of east face of T cell.
    """
    diffloc = np.zeros_like(K_11)
    diffloc[1:-2, 2:-2, 1:] = 0.25 * (
        K_iso[1:-2, 2:-2, 1:]
        + K_iso[1:-2, 2:-2, :-1]
        + K_iso[2:-1, 2:-2, 1:]
        + K_iso[2:-1, 2:-2, :-1]
    )
    diffloc[1:-2, 2:-2, 0] = 0.5 * (K_iso[1:-2, 2:-2, 0] + K_iso[2:-1, 2:-2, 0])

    sumz = np.zeros_like(K_11)[1:-2, 2:-2]
    for kr in range(2):
        ki = 0 if kr == 1 else 1
        for ip in range(2):
            drodxe = (
                drdT[1 + ip : -2 + ip, 2:-2, ki:] * dTdx[1:-2, 2:-2, ki:]
                + drdS[1 + ip : -2 + ip, 2:-2, ki:] * dSdx[1:-2, 2:-2, ki:]
            )
            drodze = (
                drdT[1 + ip : -2 + ip, 2:-2, ki:]
                * dTdz[1 + ip : -2 + ip, 2:-2, : -1 + kr or None]
                + drdS[1 + ip : -2 + ip, 2:-2, ki:]
                * dSdz[1 + ip : -2 + ip, 2:-2, : -1 + kr or None]
            )
            sxe = -drodxe / (np.minimum(0.0, drodze) - epsln)
            taper = dm_taper(sxe)
            sumz[:, :, ki:] += (
                dzw[np.newaxis, np.newaxis, : -1 + kr or None]
                * maskU[1:-2, 2:-2, ki:]
                * np.maximum(K_iso_steep, diffloc[1:-2, 2:-2, ki:] * taper)
            )
            Ai_ez[1:-2, 2:-2, ki:, ip, kr] = taper * sxe * maskU[1:-2, 2:-2, ki:]
    K_11[1:-2, 2:-2, :] = sumz / (4.0 * dzt[np.newaxis, np.newaxis, :])

    """
    Compute Ai_nz and K_22 on center of north face of T cell.
    """
    diffloc[...] = 0
    diffloc[2:-2, 1:-2, 1:] = 0.25 * (
        K_iso[2:-2, 1:-2, 1:]
        + K_iso[2:-2, 1:-2, :-1]
        + K_iso[2:-2, 2:-1, 1:]
        + K_iso[2:-2, 2:-1, :-1]
    )
    diffloc[2:-2, 1:-2, 0] = 0.5 * (K_iso[2:-2, 1:-2, 0] + K_iso[2:-2, 2:-1, 0])

    sumz = np.zeros_like(K_11)[2:-2, 1:-2]
    for kr in range(2):
        ki = 0 if kr == 1 else 1
        for jp in range(2):
            drodyn = (
                drdT[2:-2, 1 + jp : -2 + jp, ki:] * dTdy[2:-2, 1:-2, ki:]
                + drdS[2:-2, 1 + jp : -2 + jp, ki:] * dSdy[2:-2, 1:-2, ki:]
            )
            drodzn = (
                drdT[2:-2, 1 + jp : -2 + jp, ki:]
                * dTdz[2:-2, 1 + jp : -2 + jp, : -1 + kr or None]
                + drdS[2:-2, 1 + jp : -2 + jp, ki:]
                * dSdz[2:-2, 1 + jp : -2 + jp, : -1 + kr or None]
            )
            syn = -drodyn / (np.minimum(0.0, drodzn) - epsln)
            taper = dm_taper(syn)
            sumz[:, :, ki:] += (
                dzw[np.newaxis, np.newaxis, : -1 + kr or None]
                * maskV[2:-2, 1:-2, ki:]
                * np.maximum(K_iso_steep, diffloc[2:-2, 1:-2, ki:] * taper)
            )
            Ai_nz[2:-2, 1:-2, ki:, jp, kr] = taper * syn * maskV[2:-2, 1:-2, ki:]
    K_22[2:-2, 1:-2, :] = sumz / (4.0 * dzt[np.newaxis, np.newaxis, :])

    """
    compute Ai_bx, Ai_by and K33 on top face of T cell.
    """
    sumx = np.zeros_like(K_11)[2:-2, 2:-2, :-1]
    sumy = np.zeros_like(K_11)[2:-2, 2:-2, :-1]

    for kr in range(2):
        drodzb = (
            drdT[2:-2, 2:-2, kr : -1 + kr or None] * dTdz[2:-2, 2:-2, :-1]
            + drdS[2:-2, 2:-2, kr : -1 + kr or None] * dSdz[2:-2, 2:-2, :-1]
        )

        # eastward slopes at the top of T cells
        for ip in range(2):
            drodxb = (
                drdT[2:-2, 2:-2, kr : -1 + kr or None]
                * dTdx[1 + ip : -3 + ip, 2:-2, kr : -1 + kr or None]
                + drdS[2:-2, 2:-2, kr : -1 + kr or None]
                * dSdx[1 + ip : -3 + ip, 2:-2, kr : -1 + kr or None]
            )
            sxb = -drodxb / (np.minimum(0.0, drodzb) - epsln)
            taper = dm_taper(sxb)
            sumx += (
                dxu[1 + ip : -3 + ip, np.newaxis, np.newaxis]
                * K_iso[2:-2, 2:-2, :-1]
                * taper
                * sxb ** 2
                * maskW[2:-2, 2:-2, :-1]
            )
            Ai_bx[2:-2, 2:-2, :-1, ip, kr] = taper * sxb * maskW[2:-2, 2:-2, :-1]

        # northward slopes at the top of T cells
        for jp in range(2):
            facty = cosu[1 + jp : -3 + jp] * dyu[1 + jp : -3 + jp]
            drodyb = (
                drdT[2:-2, 2:-2, kr : -1 + kr or None]
                * dTdy[2:-2, 1 + jp : -3 + jp, kr : -1 + kr or None]
                + drdS[2:-2, 2:-2, kr : -1 + kr or None]
                * dSdy[2:-2, 1 + jp : -3 + jp, kr : -1 + kr or None]
            )
            syb = -drodyb / (np.minimum(0.0, drodzb) - epsln)
            taper = dm_taper(syb)
            sumy += (
                facty[np.newaxis, :, np.newaxis]
                * K_iso[2:-2, 2:-2, :-1]
                * taper
                * syb ** 2
                * maskW[2:-2, 2:-2, :-1]
            )
            Ai_by[2:-2, 2:-2, :-1, jp, kr] = taper * syb * maskW[2:-2, 2:-2, :-1]

    K_33[2:-2, 2:-2, :-1] = sumx / (4 * dxt[2:-2, np.newaxis, np.newaxis]) + sumy / (
        4 * dyt[np.newaxis, 2:-2, np.newaxis] * cost[np.newaxis, 2:-2, np.newaxis]
    )
    K_33[2:-2, 2:-2, -1] = 0.0


def run_numpy(*inputs):
    isoneutral_diffusion_pre_numpy(*inputs)
    return inputs[-7:]


# ------------------------------------------------------------------------------
# symjit version

EPSLN = 1e-20
ISO_SLOPEC = 1e-3
ISO_DSLOPE = 1e-3
K_ISO_STEEP = 50.0


def _taper(sx):
    return 0.5 * (1.0 + sp.tanh((-sp.Abs(sx) + ISO_SLOPEC) / ISO_DSLOPE))


def setup_symjit(**options):
    """Compiles the point-wise kernels; returns a function with the signature of `run_numpy`"""
    rho0, z0, theta0, grav = 1024.0, 0.0, 283.0 - 273.15, 9.81
    betaT, betaTs, gammas, betaS = 1.67e-4, 1e-5, 1.1e-8, 0.78e-3

    # drho/dT and drho/dS at the center of T cells
    m, temp, pz = sp.symbols("m temp pz")
    zz = -pz - z0
    drdT = m * (-(betaTs * (temp - theta0) + betaT * (1 - gammas * grav * zz * rho0)) * rho0)
    drdS = m * (betaS * rho0)
    rho = compile_func([m, temp, pz], [drdT, drdS], **options)

    # masked finite difference: m * (a - b) / d
    m, a, b, d = sp.symbols("m a b d")
    grad = compile_func([m, a, b, d], [m * (a - b) / d], **options)

    # slopes and diffusivities at the east/north faces
    rT, rS, gT, gS, hT, hS, mk, diffloc, dz = sp.symbols("rT rS gT gS hT hS mk diffloc dz")
    drodx = rT * gT + rS * gS
    drodz = rT * hT + rS * hS
    sx = -drodx / (sp.Min(0.0, drodz) - EPSLN)
    taper = _taper(sx)
    slope = compile_func(
        [rT, rS, gT, gS, hT, hS, mk, diffloc, dz],
        [dz * mk * sp.Max(K_ISO_STEEP, diffloc * taper), taper * sx * mk],
        **options
    )

    # slopes and contributions to K_33 at the top faces
    rT, rS, gT, gS, hT, hS, fac, kiso, mk = sp.symbols("rT rS gT gS hT hS fac kiso mk")
    drodxb = rT * gT + rS * gS
    drodzb = rT * hT + rS * hS
    sb = -drodxb / (sp.Min(0.0, drodzb) - EPSLN)
    taper = _taper(sb)
    top = compile_func(
        [rT, rS, gT, gS, hT, hS, fac, kiso, mk],
        [fac * kiso * taper * sb**2 * mk, taper * sb * mk],
        **options
    )

    def pre(
        maskT, maskU, maskV, maskW, dxt, dxu, dyt, dyu, dzt, dzw, cost, cosu, salt, temp,
        zt, K_iso, K_11, K_22, K_33, Ai_ez, Ai_nz, Ai_bx, Ai_by,
    ):
        tau = 0
        shape = K_11.shape
        nx, ny, nz = shape

        g = Grid(shape)
        sx, sy = g.sx, g.sy
        flat, zvec, apply = g.flat, g.broadcast, g.apply

        def gradient(mask, var, d, box, off):
            out = np.zeros(shape)
            out[box[0][0] : box[0][1], box[1][0] : box[1][1], box[2][0] : box[2][1]] = apply(
                grad, [(mask, 0), (var, off), (var, 0), (d, 0)], box
            )[0]
            return out

        mT, mU, mV, mW = flat(maskT), flat(maskU), flat(maskV), flat(maskW)
        tempF, saltF = flat(temp[..., tau]), flat(salt[..., tau])

        # drho_dt and drho_ds at centers of T cells
        drdTF, drdSF = rho(mT, tempF, zvec(np.abs(zt)[np.newaxis, np.newaxis, :]))

        # gradients at top face of T cells
        dz = zvec(dzw[np.newaxis, np.newaxis, :])
        dTdz = gradient(mW, tempF, dz, ((0, nx), (0, ny), (0, nz - 1)), 1)
        dSdz = gradient(mW, saltF, dz, ((0, nx), (0, ny), (0, nz - 1)), 1)

        # gradients at eastern face of T cells
        dx = zvec((dxu[:, np.newaxis] * cost[np.newaxis, :])[:, :, np.newaxis])
        dTdx = gradient(mU, tempF, dx, ((0, nx - 1), (0, ny), (0, nz)), sx)
        dSdx = gradient(mU, saltF, dx, ((0, nx - 1), (0, ny), (0, nz)), sx)

        # gradients at northern face of T cells
        dy = zvec(dyu[np.newaxis, :, np.newaxis])
        dTdy = gradient(mV, tempF, dy, ((0, nx), (0, ny - 1), (0, nz)), sy)
        dSdy = gradient(mV, saltF, dy, ((0, nx), (0, ny - 1), (0, nz)), sy)

        dTdxF, dSdxF, dTdyF, dSdyF, dTdzF, dSdzF = (
            flat(a) for a in (dTdx, dSdx, dTdy, dSdy, dTdz, dSdz)
        )

        # Ai_ez and K11 on center of east face of T cell
        diffloc = np.zeros_like(K_11)
        diffloc[1:-2, 2:-2, 1:] = 0.25 * (
            K_iso[1:-2, 2:-2, 1:] + K_iso[1:-2, 2:-2, :-1] + K_iso[2:-1, 2:-2, 1:] + K_iso[2:-1, 2:-2, :-1]
        )
        diffloc[1:-2, 2:-2, 0] = 0.5 * (K_iso[1:-2, 2:-2, 0] + K_iso[2:-1, 2:-2, 0])
        diffF = flat(diffloc)

        sumz = np.zeros_like(K_11)[1:-2, 2:-2]
        for kr in range(2):
            ki = 0 if kr == 1 else 1
            dzk = np.zeros(nz)
            dzk[ki:] = dzw[: nz - ki]  # dzw[: -1 + kr or None] aligned with k >= ki
            dzk = zvec(dzk[np.newaxis, np.newaxis, :])
            for ip in range(2):
                term, ai = apply(
                    slope,
                    [
                        (drdTF, ip * sx), (drdSF, ip * sx), (dTdxF, 0), (dSdxF, 0),
                        (dTdzF, ip * sx - ki), (dSdzF, ip * sx - ki),
                        (mU, 0), (diffF, 0), (dzk, 0),
                    ],
                    ((1, nx - 2), (2, ny - 2), (ki, nz)),
                )
                sumz[:, :, ki:] += term
                Ai_ez[1:-2, 2:-2, ki:, ip, kr] = ai
        K_11[1:-2, 2:-2, :] = sumz / (4.0 * dzt[np.newaxis, np.newaxis, :])

        # Ai_nz and K_22 on center of north face of T cell
        diffloc[...] = 0
        diffloc[2:-2, 1:-2, 1:] = 0.25 * (
            K_iso[2:-2, 1:-2, 1:] + K_iso[2:-2, 1:-2, :-1] + K_iso[2:-2, 2:-1, 1:] + K_iso[2:-2, 2:-1, :-1]
        )
        diffloc[2:-2, 1:-2, 0] = 0.5 * (K_iso[2:-2, 1:-2, 0] + K_iso[2:-2, 2:-1, 0])
        diffF = flat(diffloc)

        sumz = np.zeros_like(K_11)[2:-2, 1:-2]
        for kr in range(2):
            ki = 0 if kr == 1 else 1
            dzk = np.zeros(nz)
            dzk[ki:] = dzw[: nz - ki]
            dzk = zvec(dzk[np.newaxis, np.newaxis, :])
            for jp in range(2):
                term, ai = apply(
                    slope,
                    [
                        (drdTF, jp * sy), (drdSF, jp * sy), (dTdyF, 0), (dSdyF, 0),
                        (dTdzF, jp * sy - ki), (dSdzF, jp * sy - ki),
                        (mV, 0), (diffF, 0), (dzk, 0),
                    ],
                    ((2, nx - 2), (1, ny - 2), (ki, nz)),
                )
                sumz[:, :, ki:] += term
                Ai_nz[2:-2, 1:-2, ki:, jp, kr] = ai
        K_22[2:-2, 1:-2, :] = sumz / (4.0 * dzt[np.newaxis, np.newaxis, :])

        # Ai_bx, Ai_by and K33 on top face of T cell
        sumx = np.zeros_like(K_11)[2:-2, 2:-2, :-1]
        sumy = np.zeros_like(K_11)[2:-2, 2:-2, :-1]
        kisoF = flat(K_iso)
        box = ((2, nx - 2), (2, ny - 2), (0, nz - 1))

        for kr in range(2):
            # eastward slopes at the top of T cells
            for ip in range(2):
                fac = np.zeros(nx)
                fac[2 : nx - 2] = dxu[1 + ip : -3 + ip]
                term, ai = apply(
                    top,
                    [
                        (drdTF, kr), (drdSF, kr),
                        (dTdxF, (ip - 1) * sx + kr), (dSdxF, (ip - 1) * sx + kr),
                        (dTdzF, 0), (dSdzF, 0),
                        (zvec(fac[:, np.newaxis, np.newaxis]), 0), (kisoF, 0), (mW, 0),
                    ],
                    box,
                )
                sumx += term
                Ai_bx[2:-2, 2:-2, :-1, ip, kr] = ai

            # northward slopes at the top of T cells
            for jp in range(2):
                fac = np.zeros(ny)
                fac[2 : ny - 2] = cosu[1 + jp : -3 + jp] * dyu[1 + jp : -3 + jp]
                term, ai = apply(
                    top,
                    [
                        (drdTF, kr), (drdSF, kr),
                        (dTdyF, (jp - 1) * sy + kr), (dSdyF, (jp - 1) * sy + kr),
                        (dTdzF, 0), (dSdzF, 0),
                        (zvec(fac[np.newaxis, :, np.newaxis]), 0), (kisoF, 0), (mW, 0),
                    ],
                    box,
                )
                sumy += term
                Ai_by[2:-2, 2:-2, :-1, jp, kr] = ai

        K_33[2:-2, 2:-2, :-1] = sumx / (4 * dxt[2:-2, np.newaxis, np.newaxis]) + sumy / (
            4 * dyt[np.newaxis, 2:-2, np.newaxis] * cost[np.newaxis, 2:-2, np.newaxis]
        )
        K_33[2:-2, 2:-2, -1] = 0.0

    def run(*inputs):
        pre(*inputs)
        return inputs[-7:]

    return run

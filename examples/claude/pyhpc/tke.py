"""
Turbulent kinetic energy benchmark (pyhpc-benchmarks `turbulent_kinetic_energy`).

Integrates the turbulent kinetic energy (TKE) equation of the Veros ocean model: implicit
vertical mixing and dissipation (a tridiagonal solve), lateral diffusion, advection with
the superbee flux limiter, and Adams-Bashforth time stepping.

The tridiagonal solve (LAPACK) and the array bookkeeping stay in numpy. The advection
of the TKE with the superbee scheme -- about half of the numpy time -- is a stencil with
a long chain of point-wise operations (masks, `where`s, and a flux limiter). symjit
compiles it into a single fused kernel (see `setup_symjit`), and the shifted operands of
the stencil are passed without copies (see `stencil.py`).

Original: https://github.com/dionhaefner/pyhpc-benchmarks (public domain).
"""

import math

import numpy as np
import sympy as sp
from scipy.linalg import lapack
from symjit import compile_func

from stencil import Grid

TOLERANCE = dict(rtol=1e-6, atol=1e-6)  # for comparing the symjit and numpy results
NAME = "turbulent_kinetic_energy"

def generate_inputs(size):
    import numpy as np

    np.random.seed(17)

    shape = (
        math.ceil(2 * size ** (1 / 3)),
        math.ceil(2 * size ** (1 / 3)),
        math.ceil(0.25 * size ** (1 / 3)),
    )

    # masks
    maskU, maskV, maskW = (
        (np.random.rand(*shape) < 0.8).astype("float64") for _ in range(3)
    )

    # 1d arrays
    dxt, dxu = (np.random.randn(shape[0]) for _ in range(2))
    dyt, dyu = (np.random.randn(shape[1]) for _ in range(2))
    dzt, dzw = (np.random.randn(shape[2]) for _ in range(2))
    cost, cosu = (np.random.randn(shape[1]) for _ in range(2))

    # 2d arrays
    kbot = np.random.randint(0, shape[2], size=shape[:2])
    forc_tke_surface = np.random.randn(*shape[:2])

    # 3d arrays
    kappaM, mxl, forc = (np.random.randn(*shape) for _ in range(3))

    # 4d arrays
    u, v, w, tke, dtke = (np.random.randn(*shape, 3) for _ in range(5))

    return (
        u,
        v,
        w,
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
        kbot,
        kappaM,
        mxl,
        forc,
        forc_tke_surface,
        tke,
        dtke,
    )


# ------------------------------------------------------------------------------
# numpy version, from pyhpc-benchmarks

def where(mask, a, b):
    return np.where(mask, a, b)


def solve_implicit(ks, a, b, c, d, b_edge=None, d_edge=None):
    land_mask = (ks >= 0)[:, :, np.newaxis]
    edge_mask = land_mask & (
        np.arange(a.shape[2])[np.newaxis, np.newaxis, :] == ks[:, :, np.newaxis]
    )
    water_mask = land_mask & (
        np.arange(a.shape[2])[np.newaxis, np.newaxis, :] >= ks[:, :, np.newaxis]
    )

    a_tri = water_mask * a * np.logical_not(edge_mask)
    b_tri = where(water_mask, b, 1.0)
    if b_edge is not None:
        b_tri = where(edge_mask, b_edge, b_tri)
    c_tri = water_mask * c
    d_tri = water_mask * d
    if d_edge is not None:
        d_tri = where(edge_mask, d_edge, d_tri)

    return solve_tridiag(a_tri, b_tri, c_tri, d_tri), water_mask


def solve_tridiag(a, b, c, d):
    """
    Solves a tridiagonal matrix system with diagonals a, b, c and RHS vector d.
    """
    assert a.shape == b.shape and a.shape == c.shape and a.shape == d.shape
    a[..., 0] = c[..., -1] = 0  # remove couplings between slices
    return lapack.dgtsv(a.flatten()[1:], b.flatten(), c.flatten()[:-1], d.flatten())[
        3
    ].reshape(a.shape)


def _calc_cr(rjp, rj, rjm, vel):
    """
    Calculates cr value used in superbee advection scheme
    """
    eps = 1e-20  # prevent division by 0
    return where(vel > 0.0, rjm, rjp) / where(np.abs(rj) < eps, eps, rj)


def pad_z_edges(arr):
    arr_shape = list(arr.shape)
    arr_shape[2] += 2
    out = np.zeros(arr_shape, arr.dtype)
    out[:, :, 1:-1] = arr
    return out


def limiter(cr):
    return np.maximum(0.0, np.maximum(np.minimum(1.0, 2 * cr), np.minimum(2.0, cr)))


def _adv_superbee(vel, var, mask, dx, axis, cost, cosu, dt_tracer):
    velfac = 1
    if axis == 0:
        sm1, s, sp1, sp2 = (
            (slice(1 + n, -2 + n or None), slice(2, -2), slice(None))
            for n in range(-1, 3)
        )
        dx = cost[np.newaxis, 2:-2, np.newaxis] * dx[1:-2, np.newaxis, np.newaxis]
    elif axis == 1:
        sm1, s, sp1, sp2 = (
            (slice(2, -2), slice(1 + n, -2 + n or None), slice(None))
            for n in range(-1, 3)
        )
        dx = (cost * dx)[np.newaxis, 1:-2, np.newaxis]
        velfac = cosu[np.newaxis, 1:-2, np.newaxis]
    elif axis == 2:
        vel, var, mask = (pad_z_edges(a) for a in (vel, var, mask))
        sm1, s, sp1, sp2 = (
            (slice(2, -2), slice(2, -2), slice(1 + n, -2 + n or None))
            for n in range(-1, 3)
        )
        dx = dx[np.newaxis, np.newaxis, :-1]
    else:
        raise ValueError("axis must be 0, 1, or 2")
    uCFL = np.abs(velfac * vel[s] * dt_tracer / dx)
    rjp = (var[sp2] - var[sp1]) * mask[sp1]
    rj = (var[sp1] - var[s]) * mask[s]
    rjm = (var[s] - var[sm1]) * mask[sm1]
    cr = limiter(_calc_cr(rjp, rj, rjm, vel[s]))
    return (
        velfac * vel[s] * (var[sp1] + var[s]) * 0.5
        - np.abs(velfac * vel[s]) * ((1.0 - cr) + uCFL * cr) * rj * 0.5
    )


def adv_flux_superbee_wgrid(
    adv_fe,
    adv_fn,
    adv_ft,
    var,
    u_wgrid,
    v_wgrid,
    w_wgrid,
    maskW,
    dxt,
    dyt,
    dzw,
    cost,
    cosu,
    dt_tracer,
):
    """
    Calculates advection of a tracer defined on Wgrid
    """
    maskUtr = np.zeros_like(maskW)
    maskUtr[:-1, :, :] = maskW[1:, :, :] * maskW[:-1, :, :]
    adv_fe[...] = 0.0
    adv_fe[1:-2, 2:-2, :] = _adv_superbee(
        u_wgrid, var, maskUtr, dxt, 0, cost, cosu, dt_tracer
    )

    maskVtr = np.zeros_like(maskW)
    maskVtr[:, :-1, :] = maskW[:, 1:, :] * maskW[:, :-1, :]
    adv_fn[...] = 0.0
    adv_fn[2:-2, 1:-2, :] = _adv_superbee(
        v_wgrid, var, maskVtr, dyt, 1, cost, cosu, dt_tracer
    )

    maskWtr = np.zeros_like(maskW)
    maskWtr[:, :, :-1] = maskW[:, :, 1:] * maskW[:, :, :-1]
    adv_ft[...] = 0.0
    adv_ft[2:-2, 2:-2, :-1] = _adv_superbee(
        w_wgrid, var, maskWtr, dzw, 2, cost, cosu, dt_tracer
    )


def integrate_tke(
    u,
    v,
    w,
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
    kbot,
    kappaM,
    mxl,
    forc,
    forc_tke_surface,
    tke,
    dtke,
    adv_flux=None,
):
    adv_flux = adv_flux or adv_flux_superbee_wgrid
    tau = 0
    taup1 = 1
    taum1 = 2

    dt_tracer = 1
    dt_mom = 1
    AB_eps = 0.1
    alpha_tke = 1.0
    c_eps = 0.7
    K_h_tke = 2000.0

    flux_east = np.zeros_like(maskU)
    flux_north = np.zeros_like(maskU)
    flux_top = np.zeros_like(maskU)

    sqrttke = np.sqrt(np.maximum(0.0, tke[:, :, :, tau]))

    """
    integrate Tke equation on W grid with surface flux boundary condition
    """
    dt_tke = dt_mom  # use momentum time step to prevent spurious oscillations

    """
    vertical mixing and dissipation of TKE
    """
    ks = kbot[2:-2, 2:-2] - 1

    a_tri = np.zeros_like(maskU[2:-2, 2:-2])
    b_tri = np.zeros_like(maskU[2:-2, 2:-2])
    c_tri = np.zeros_like(maskU[2:-2, 2:-2])
    d_tri = np.zeros_like(maskU[2:-2, 2:-2])
    delta = np.zeros_like(maskU[2:-2, 2:-2])

    delta[:, :, :-1] = (
        dt_tke
        / dzt[np.newaxis, np.newaxis, 1:]
        * alpha_tke
        * 0.5
        * (kappaM[2:-2, 2:-2, :-1] + kappaM[2:-2, 2:-2, 1:])
    )

    a_tri[:, :, 1:-1] = -delta[:, :, :-2] / dzw[np.newaxis, np.newaxis, 1:-1]
    a_tri[:, :, -1] = -delta[:, :, -2] / (0.5 * dzw[-1])

    b_tri[:, :, 1:-1] = (
        1
        + (delta[:, :, 1:-1] + delta[:, :, :-2]) / dzw[np.newaxis, np.newaxis, 1:-1]
        + dt_tke * c_eps * sqrttke[2:-2, 2:-2, 1:-1] / mxl[2:-2, 2:-2, 1:-1]
    )
    b_tri[:, :, -1] = (
        1
        + delta[:, :, -2] / (0.5 * dzw[-1])
        + dt_tke * c_eps / mxl[2:-2, 2:-2, -1] * sqrttke[2:-2, 2:-2, -1]
    )
    b_tri_edge = (
        1
        + delta / dzw[np.newaxis, np.newaxis, :]
        + dt_tke * c_eps / mxl[2:-2, 2:-2, :] * sqrttke[2:-2, 2:-2, :]
    )

    c_tri[:, :, :-1] = -delta[:, :, :-1] / dzw[np.newaxis, np.newaxis, :-1]

    d_tri[...] = tke[2:-2, 2:-2, :, tau] + dt_tke * forc[2:-2, 2:-2, :]
    d_tri[:, :, -1] += dt_tke * forc_tke_surface[2:-2, 2:-2] / (0.5 * dzw[-1])

    sol, water_mask = solve_implicit(ks, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge)
    tke[2:-2, 2:-2, :, taup1] = where(water_mask, sol, tke[2:-2, 2:-2, :, taup1])

    """
    Add TKE if surface density flux drains TKE in uppermost box
    """
    tke_surf_corr = np.zeros(maskU.shape[:2])
    mask = tke[2:-2, 2:-2, -1, taup1] < 0.0
    tke_surf_corr[2:-2, 2:-2] = where(
        mask, -tke[2:-2, 2:-2, -1, taup1] * 0.5 * dzw[-1] / dt_tke, 0.0
    )
    tke[2:-2, 2:-2, -1, taup1] = np.maximum(0.0, tke[2:-2, 2:-2, -1, taup1])

    """
    add tendency due to lateral diffusion
    """
    flux_east[:-1, :, :] = (
        K_h_tke
        * (tke[1:, :, :, tau] - tke[:-1, :, :, tau])
        / (cost[np.newaxis, :, np.newaxis] * dxu[:-1, np.newaxis, np.newaxis])
        * maskU[:-1, :, :]
    )
    flux_east[-1, :, :] = 0.0
    flux_north[:, :-1, :] = (
        K_h_tke
        * (tke[:, 1:, :, tau] - tke[:, :-1, :, tau])
        / dyu[np.newaxis, :-1, np.newaxis]
        * maskV[:, :-1, :]
        * cosu[np.newaxis, :-1, np.newaxis]
    )
    flux_north[:, -1, :] = 0.0
    tke[2:-2, 2:-2, :, taup1] += (
        dt_tke
        * maskW[2:-2, 2:-2, :]
        * (
            (flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
            / (cost[np.newaxis, 2:-2, np.newaxis] * dxt[2:-2, np.newaxis, np.newaxis])
            + (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
            / (cost[np.newaxis, 2:-2, np.newaxis] * dyt[np.newaxis, 2:-2, np.newaxis])
        )
    )

    """
    add tendency due to advection
    """
    adv_flux(
        flux_east,
        flux_north,
        flux_top,
        tke[:, :, :, tau],
        u[..., tau],
        v[..., tau],
        w[..., tau],
        maskW,
        dxt,
        dyt,
        dzw,
        cost,
        cosu,
        dt_tracer,
    )

    dtke[2:-2, 2:-2, :, tau] = maskW[2:-2, 2:-2, :] * (
        -(flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
        / (cost[np.newaxis, 2:-2, np.newaxis] * dxt[2:-2, np.newaxis, np.newaxis])
        - (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
        / (cost[np.newaxis, 2:-2, np.newaxis] * dyt[np.newaxis, 2:-2, np.newaxis])
    )
    dtke[:, :, 0, tau] += -flux_top[:, :, 0] / dzw[0]
    dtke[:, :, 1:-1, tau] += -(flux_top[:, :, 1:-1] - flux_top[:, :, :-2]) / dzw[1:-1]
    dtke[:, :, -1, tau] += -(flux_top[:, :, -1] - flux_top[:, :, -2]) / (0.5 * dzw[-1])

    """
    Adam Bashforth time stepping
    """
    tke[:, :, :, taup1] += dt_tracer * (
        (1.5 + AB_eps) * dtke[:, :, :, tau] - (0.5 + AB_eps) * dtke[:, :, :, taum1]
    )

    return tke, dtke, tke_surf_corr


def run_numpy(*inputs):
    return integrate_tke(*inputs)


# ------------------------------------------------------------------------------
# symjit version of the superbee advection


def _superbee_kernel(with_velfac, **options):
    """
    The point-wise part of `_adv_superbee`:

        vel        velocity at the face
        vm, v0, v1, v2   the tracer at the four points of the stencil
        mm, m0, m1       masks at the first three points
        dx         grid spacing at the face
        velfac     (optional) metric factor of the velocity
    """
    dt_tracer = 1.0
    eps = 1e-20

    vel, vm, v0, v1, v2, mm, m0, m1, dx, velfac = sp.symbols("vel vm v0 v1 v2 mm m0 m1 dx velfac")
    args = [vel, vm, v0, v1, v2, mm, m0, m1, dx]
    vf = 1.0
    if with_velfac:
        args.append(velfac)
        vf = velfac

    uCFL = sp.Abs(vf * vel * dt_tracer / dx)
    rjp = (v2 - v1) * m1
    rj = (v1 - v0) * m0
    rjm = (v0 - vm) * mm

    num = sp.Piecewise((rjm, vel > 0.0), (rjp, True))
    den = sp.Piecewise((eps, sp.Abs(rj) < eps), (rj, True))
    cr = num / den
    cr = sp.Max(0.0, sp.Max(sp.Min(1.0, 2 * cr), sp.Min(2.0, cr)))  # the limiter

    flux = vf * vel * (v1 + v0) * 0.5 - sp.Abs(vf * vel) * ((1.0 - cr) + uCFL * cr) * rj * 0.5
    return compile_func(args, [flux], **options)


def setup_symjit(**options):
    """Compiles the advection kernels; returns a function with the signature of `run_numpy`"""
    superbee = _superbee_kernel(False, **options)
    superbee_v = _superbee_kernel(True, **options)

    def adv_superbee(g, vel, var, mask, dx, axis, velfac, box, shifts):
        """
        vel, var, mask: flat arrays of the grid `g`; dx and velfac are flat arrays (or None)
        box: the output points; shifts: the flat offsets of the points sm1, s, sp1, and sp2
        """
        sm1, s, sp1, sp2 = shifts
        operands = [(vel, s), (var, sm1), (var, s), (var, sp1), (var, sp2),
                    (mask, sm1), (mask, s), (mask, sp1), (dx, 0)]
        if velfac is None:
            return g.apply(superbee, operands, box)[0]
        return g.apply(superbee_v, operands + [(velfac, 0)], box)[0]

    def adv_flux(adv_fe, adv_fn, adv_ft, var, u, v, w, maskW, dxt, dyt, dzw, cost, cosu, dt_tracer):
        assert dt_tracer == 1  # hard-wired in the kernels
        nx, ny, nz = maskW.shape
        g = Grid(maskW.shape)
        F = g.flat
        var, u, v, w, mask = F(var), F(u), F(v), F(w), F(maskW)

        # flux through the east faces
        maskUtr = np.zeros_like(maskW)
        maskUtr[:-1, :, :] = maskW[1:, :, :] * maskW[:-1, :, :]
        dx = g.broadcast((dxt[:, np.newaxis] * cost[np.newaxis, :])[:, :, np.newaxis])
        adv_fe[...] = 0.0
        adv_fe[1:-2, 2:-2, :] = adv_superbee(
            g, u, var, F(maskUtr), dx, 0, None, ((1, nx - 2), (2, ny - 2), (0, nz)),
            (-g.sx, 0, g.sx, 2 * g.sx),
        )

        # flux through the north faces
        maskVtr = np.zeros_like(maskW)
        maskVtr[:, :-1, :] = maskW[:, 1:, :] * maskW[:, :-1, :]
        dy = g.broadcast((cost * dyt)[np.newaxis, :, np.newaxis])
        velfac = g.broadcast(cosu[np.newaxis, :, np.newaxis])
        adv_fn[...] = 0.0
        adv_fn[2:-2, 1:-2, :] = adv_superbee(
            g, v, var, F(maskVtr), dy, 1, velfac, ((2, nx - 2), (1, ny - 2), (0, nz)),
            (-g.sy, 0, g.sy, 2 * g.sy),
        )

        # flux through the top faces: the z-axis is padded with a zero plane on each side
        def pad(a):
            out = np.zeros((nx, ny, nz + 2))
            out[:, :, 1:-1] = a
            return out

        gp = Grid((nx, ny, nz + 2))
        maskWtr = np.zeros_like(maskW)
        maskWtr[:, :, :-1] = maskW[:, :, 1:] * maskW[:, :, :-1]
        dz = np.zeros(nz + 2)
        dz[1:-1] = dzw  # the spacing at padded index k is dzw[k - 1]
        dz = gp.broadcast(dz[np.newaxis, np.newaxis, :])
        adv_ft[...] = 0.0
        adv_ft[2:-2, 2:-2, :-1] = adv_superbee(
            gp, gp.flat(pad(w.reshape(maskW.shape))), gp.flat(pad(var.reshape(maskW.shape))),
            gp.flat(pad(maskWtr)), dz, 2, None, ((2, nx - 2), (2, ny - 2), (1, nz)),
            (-1, 0, 1, 2),
        )

    def run(*inputs):
        return integrate_tke(*inputs, adv_flux=adv_flux)

    return run

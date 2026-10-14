//! symjit's own math library (cargo feature `math`).
//!
//! The functions are written once, as generic Rust over [`MathOps`], without branches (the
//! special cases are selects), so that the same source gives
//!
//! * a plain Rust function on `f64` (the reference implementation, `exp(1.5f64)`);
//! * a kernel of the kernel builder (`MathOps` for `kernel::Real`, features `math` and
//!   `composer`);
//! * MIR that is inlined in the kernels (`inline_calls`): a call of `exp` becomes a call of
//!   the funclet `@libm_exp`, which every backend compiles like the rest of the kernel, so the
//!   SIMD kernels evaluate it on all lanes at once instead of calling the scalar function once
//!   per lane. It is enabled by `Config::set_inline_math` (`ty = "native;inline_math=true"`).
//!
//! The functions use `+`, `-` and `*` but no explicit fused multiply-adds: with `fastmath`,
//! the peephole optimizer fuses them as it does any expression; without it, the kernels, the
//! bytecode interpreter and the `f64` version give the same bits.
//!
//! The exponent of a double is changed with `pow2i`, `getexp` and `getmant`, which are
//! defined bit for bit (`utils::pow2i`, ...).

use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Sub};

use anyhow::{anyhow, Result};

use super::mir::{BinOp, Instruction, Mir, UniOp, LIBM_COMPLEX, LIBM_END, LIBM_PREFIX};
use super::serializer::MirWriter;
use super::utils::{self, Reg};

/// The operations the functions of the library are written with.
pub trait MathOps:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
{
    /// The result of a comparison.
    type Mask: Copy;

    /// The constant `v`, in the same kernel as `self`.
    fn lit(self, v: f64) -> Self;
    /// self > v, false for NaN (as `v < self`: the x86 `gt` of the generators is true for
    /// NaN, an unordered comparison, while `lt` is ordered on every backend)
    fn gt(self, v: f64) -> Self::Mask;
    /// self < v, false for NaN
    fn lt(self, v: f64) -> Self::Mask;
    /// `a` where `m` is true, else `b` (both are computed).
    fn select(m: Self::Mask, a: Self, b: Self) -> Self;
    fn abs(self) -> Self;
    /// 2^self for an integer self in [-1022, 1023] (`utils::pow2i`).
    fn pow2i(self) -> Self;
    /// The exponent of self, floor(log2(|self|)) for normal numbers (`utils::getexp`).
    fn getexp(self) -> Self;
    /// The mantissa of self in [1, 2), with the sign of self (`utils::getmant`).
    fn getmant(self) -> Self;
}

impl MathOps for f64 {
    type Mask = bool;

    fn lit(self, v: f64) -> f64 {
        v
    }

    fn gt(self, v: f64) -> bool {
        self > v
    }

    fn lt(self, v: f64) -> bool {
        self < v
    }

    fn select(m: bool, a: f64, b: f64) -> f64 {
        if m {
            a
        } else {
            b
        }
    }

    fn abs(self) -> f64 {
        f64::abs(self)
    }

    fn pow2i(self) -> f64 {
        utils::pow2i(self)
    }

    fn getexp(self) -> f64 {
        utils::getexp(self)
    }

    fn getmant(self) -> f64 {
        utils::getmant(self)
    }
}

/***************************** the functions *****************************/

/// 1.5 * 2^52: `(x + ROUND) - ROUND` rounds x (|x| < 2^51) to an integer, ties to even.
const ROUND: f64 = 6755399441055744.0;

/// e^x, within an ulp or so (`fastmath` changes the rounding of a few results).
pub fn exp<T: MathOps>(x: T) -> T {
    exp_scaled(x, 0.0)
}

/// e^x 2^s for s = 0 or -1, without overflow when only e^x overflows.
///
/// x = k ln(2) + r with an integer k and |r| <= ln(2) / 2 (Cody and Waite: ln(2) in two
/// parts, the first with enough trailing zeros that k * LN2_HI is exact); e^r is the Taylor
/// polynomial of degree 13 (truncation error below 4e-18), and e^x 2^s = e^r 2^k1 2^(k2 + s)
/// with k1 + k2 = k, so that both powers of two are normal for every k in [-1077, 1026]. x is
/// clamped to [-746, 711] first; outside it, the result is 0 or +inf anyway (e^711 / 2
/// overflows too). NaN passes through the clamps (its comparisons are false) and gives NaN.
fn exp_scaled<T: MathOps>(x: T, s: f64) -> T {
    const LOG2E: f64 = std::f64::consts::LOG2_E;
    const LN2_HI: f64 = f64::from_bits(0x3fe6_2e42_fee0_0000); // 6.93147180369123816490e-01
    const LN2_LO: f64 = f64::from_bits(0x3dea_39ef_3579_3c76); // 1.90821492927058770002e-10

    let x = T::select(x.gt(711.0), x.lit(711.0), x);
    let x = T::select(x.lt(-746.0), x.lit(-746.0), x);

    let k = (x * LOG2E + ROUND) - ROUND;
    let r = x - k * LN2_HI;
    let r = r - k * LN2_LO;

    // 1/13!, ..., 1/1!, 1/0!
    let mut p = r.lit(1.0 / 6227020800.0);
    for c in [
        1.0 / 479001600.0,
        1.0 / 39916800.0,
        1.0 / 3628800.0,
        1.0 / 362880.0,
        1.0 / 40320.0,
        1.0 / 5040.0,
        1.0 / 720.0,
        1.0 / 120.0,
        1.0 / 24.0,
        1.0 / 6.0,
        0.5,
        1.0,
        1.0,
    ] {
        p = p * r + c;
    }

    let k1 = (k * 0.5 + ROUND) - ROUND;
    let k2 = k - k1;
    if s == 0.0 {
        p * k1.pow2i() * k2.pow2i()
    } else {
        p * k1.pow2i() * (k2 + s).pow2i()
    }
}

/// 1/n! (exact factorials up to 22!)
fn inv_factorial(n: u32) -> f64 {
    1.0 / (1..=n).map(|i| i as f64).product::<f64>()
}

/// sinh(x) for |x| < 1: x + x^3/3! + ... + x^19/19! (truncation error below 1e-17 relative).
/// Odd in x, so it keeps the sign of a zero.
fn sinh_series<T: MathOps>(x: T) -> T {
    let x2 = x * x;
    let mut p = x.lit(inv_factorial(19));
    for n in (1..=8).rev() {
        p = p * x2 + inv_factorial(2 * n + 1);
    }
    p = p * x2 + 1.0;
    x * p
}

/// cosh(x) for |x| < 1: 1 + x^2/2! + ... + x^20/20!.
fn cosh_series<T: MathOps>(x: T) -> T {
    let x2 = x * x;
    let mut p = x.lit(inv_factorial(20));
    for n in (1..=9).rev() {
        p = p * x2 + inv_factorial(2 * n);
    }
    p * x2 + 1.0
}

/// -1 for x < 0, else 1
fn sign<T: MathOps>(x: T) -> T {
    T::select(x.lt(0.0), x.lit(-1.0), x.lit(1.0))
}

// The hyperbolic functions. For |x| < 1, the Taylor series of sinh and cosh (no cancellation
// near 0, and the sign of a zero is kept: sinh(-0) = -0, coth(-0) = -inf). Otherwise:
// sinh and cosh from h = e^|x| / 2 (`exp_scaled`, which does not overflow until sinh and
// cosh do, near 710.48), and the others from e = e^-|x|, which does not overflow either and
// keeps their subnormal results (sech(720) is about 1.6e-313, where 1 / cosh(720) is 0):
//
//   sinh = sign (h - 1/(4h))          cosh = h + 1/(4h)
//   tanh = sign (1 - e^2) / (1 + e^2)  coth = sign (1 + e^2) / (1 - e^2)
//   sech = 2e / (1 + e^2)             csch = sign 2e / (1 - e^2)
//
// 1 - e^2 >= 1 - e^-2 for |x| >= 1, so there is no cancellation. Infinities give the limits
// (sinh(-inf) = -inf, tanh(inf) = 1, csch(-inf) = -0) and NaN gives NaN.

/// sinh(x)
pub fn sinh<T: MathOps>(x: T) -> T {
    let a = x.abs();
    let h = exp_scaled(a, -1.0);
    let big = (h - x.lit(0.25) / h) * sign(x);
    T::select(a.lt(1.0), sinh_series(x), big)
}

/// cosh(x)
pub fn cosh<T: MathOps>(x: T) -> T {
    let h = exp_scaled(x.abs(), -1.0);
    h + x.lit(0.25) / h
}

/// tanh(x)
pub fn tanh<T: MathOps>(x: T) -> T {
    let a = x.abs();
    let e = exp(a * -1.0);
    let e2 = e * e;
    let big = (e2 * -1.0 + 1.0) / (e2 + 1.0) * sign(x);
    T::select(a.lt(1.0), sinh_series(x) / cosh_series(x), big)
}

/// coth(x) = 1 / tanh(x)
pub fn coth<T: MathOps>(x: T) -> T {
    let a = x.abs();
    let e = exp(a * -1.0);
    let e2 = e * e;
    let big = (e2 + 1.0) / (e2 * -1.0 + 1.0) * sign(x);
    T::select(a.lt(1.0), cosh_series(x) / sinh_series(x), big)
}

/// sech(x) = 1 / cosh(x)
pub fn sech<T: MathOps>(x: T) -> T {
    let e = exp(x.abs() * -1.0);
    e * 2.0 / (e * e + 1.0)
}

/// csch(x) = 1 / sinh(x)
pub fn csch<T: MathOps>(x: T) -> T {
    let a = x.abs();
    let e = exp(a * -1.0);
    let big = e * 2.0 / (e * e * -1.0 + 1.0) * sign(x);
    T::select(a.lt(1.0), x.lit(1.0) / sinh_series(x), big)
}

// sin and cos.
//
// x = n pi/2 + r with an integer n and |r| <= pi/4: n = round(2x / pi), and pi/2 is split
// into four parts (fdlibm's pio2_1, pio2_2, pio2_3 and pio2_3t), the first three with 33
// significant bits, so that n times each is exact for n < 2^20, and x - n pio2_1 is exact
// (Sterbenz). Where x is close to a multiple of pi/2 and r is small, the subtractions cancel
// exactly; elsewhere their rounding errors are small relative to r. This holds for
// |x| < 2^20 pi/2 (`TRIG_LIMIT`); the kernels call the C functions for larger arguments, NaN and
// infinities (`domain_limit`), and the generic functions are inaccurate there.
//
// The reduced argument is kept in two parts, r + y, and sin(r + y) and cos(r + y) are musl's
// (fdlibm's) minimax polynomials on [-pi/4, pi/4] with their first-order corrections for y; cos
// keeps the rounding error of 1 - r^2/2 ((1 - w) - r^2/2, exact). Then, with q = n mod 4,
// sin(x) = sin r, cos r, -sin r, -cos r and cos(x) = cos r, -sin r, -cos r, sin r.

/// 2^20 pi/2 (rounded down): the reduction of sin and cos is accurate below it.
pub const TRIG_LIMIT: f64 = 1647099.0;

const INV_PIO2: f64 = std::f64::consts::FRAC_2_PI;
const PIO2: [f64; 4] = [
    f64::from_bits(0x3ff9_21fb_5440_0000), // 1.57079632673412561417e+00
    f64::from_bits(0x3dd0_b461_1a60_0000), // 6.07710050630396597660e-11
    f64::from_bits(0x3ba3_198a_2e00_0000), // 2.02226624871116645580e-21
    f64::from_bits(0x397b_839a_2520_49c1), // 8.47842766036889956997e-32
];

/// (n, r, y) with x = n pi/2 + r + y, |y| <= ulp(r) / 2: the reduced argument in two parts
fn reduce_pio2<T: MathOps>(x: T) -> (T, T, T) {
    let n = (x * INV_PIO2 + ROUND) - ROUND;
    let r0 = x - n * PIO2[0]; // exact
    let r1 = r0 - n * PIO2[1];
    let e1 = (r0 - r1) - n * PIO2[1]; // the rounding error of r1, exact
    let t = e1 - (n * PIO2[2] + n * PIO2[3]);
    // r1 + t = r + y (two-sum)
    let r = r1 + t;
    let b = r - r1;
    let y = (r1 - (r - b)) + (t - b);
    (n, r, y)
}

/// sin(r + y) for |r| <= pi/4, |y| <= ulp(r) / 2 (musl's __sin)
#[allow(clippy::excessive_precision)] // musl's coefficients, as published
fn sin_kernel<T: MathOps>(r: T, y: T) -> T {
    const S1: f64 = -1.66666666666666324348e-01;
    const S2: f64 = 8.33333333332248946124e-03;
    const S3: f64 = -1.98412698298579493134e-04;
    const S4: f64 = 2.75573137070700676789e-06;
    const S5: f64 = -2.50507602534068634195e-08;
    const S6: f64 = 1.58969099521155010221e-10;

    let z = r * r;
    let w = z * z;
    let p = (z * S4 + S3) * z + S2 + z * w * (z * S6 + S5);
    let v = z * r;
    r - ((z * (y * 0.5 - v * p) - y) - v * S1)
}

/// cos(r + y) for |r| <= pi/4, |y| <= ulp(r) / 2 (musl's __cos)
#[allow(clippy::excessive_precision)] // musl's coefficients, as published
fn cos_kernel<T: MathOps>(r: T, y: T) -> T {
    const C1: f64 = 4.16666666666666019037e-02;
    const C2: f64 = -1.38888888888741095749e-03;
    const C3: f64 = 2.48015872894767294178e-05;
    const C4: f64 = -2.75573143513906633035e-07;
    const C5: f64 = 2.08757232129817482790e-09;
    const C6: f64 = -1.13596475577881948265e-11;

    let z = r * r;
    let w = z * z;
    let p = z * ((z * C3 + C2) * z + C1) + w * w * ((z * C6 + C5) * z + C4);
    let hz = z * 0.5;
    let w = r.lit(1.0) - hz;
    w + (((r.lit(1.0) - w) - hz) + (z * p - r * y))
}

/// sin(n pi/2 + r) from s = sin(r) and c = cos(r): with q = n mod 4, s, c, -s or -c
fn by_quadrant<T: MathOps>(n: T, s: T, c: T) -> T {
    // floor(n / 4) and floor(q / 2), exact: n / 4 - 0.375 and q / 2 - 0.25 are not ties
    let m = ((n * 0.25 - 0.375) + ROUND) - ROUND;
    let q = n - m * 4.0;
    let h = ((q * 0.5 - 0.25) + ROUND) - ROUND;
    let odd = q - h * 2.0;
    let v = T::select(odd.gt(0.5), c, s);
    T::select(q.gt(1.5), v * -1.0, v)
}

/// sin(x), accurate for |x| < `TRIG_LIMIT`
pub fn sin<T: MathOps>(x: T) -> T {
    let (n, r, y) = reduce_pio2(x);
    let y = by_quadrant(n, sin_kernel(r, y), cos_kernel(r, y));
    // sin(x) = x for |x| < 2^-26 (and the sign of a zero is kept)
    T::select(x.abs().lt(1.4901161193847656e-8), x, y)
}

/// cos(x) = sin(x + pi/2), accurate for |x| < `TRIG_LIMIT`
pub fn cos<T: MathOps>(x: T) -> T {
    let (n, r, y) = reduce_pio2(x);
    by_quadrant(n + 1.0, sin_kernel(r, y), cos_kernel(r, y))
}

// The complex hyperbolic functions of z = x + iy, as (real part, imaginary part):
//
//   sinh z = sinh x cos y + i cosh x sin y      cosh z = cosh x cos y + i sinh x sin y
//
// and the others as conj(numerator) / |denominator|^2, whose denominators are sums of
// squares (no cancellation):
//
//   tanh z = (sinh x cosh x + i sin y cos y) / (sinh^2 x + cos^2 y)
//   coth z = (sinh x cosh x - i sin y cos y) / (sinh^2 x + sin^2 y)
//   sech z = (cosh x cos y - i sinh x sin y) / (sinh^2 x + cos^2 y)
//   csch z = (sinh x cos y - i cosh x sin y) / (sinh^2 x + sin^2 y)
//
// For |x| > 22, where sinh^2 x overflows from |x| = 355 on, their limits, to a relative
// error below e^-2|x|: tanh z = sign(x) + i sin y cos y / cosh^2 x (as `tanh_c` in code.rs),
// coth likewise with -i, sech z = (cos y - i sign(x) sin y) / cosh x and
// csch z = (sign(x) cos y - i sin y) / cosh x, which keep their subnormal results. The kernels
// use them for |x| < 710 and |y| < `TRIG_LIMIT` (`complex_domain`), and the complex C functions
// otherwise (also for NaN and infinities).

/// (sinh x, cosh x), from one exponential
fn sinh_cosh<T: MathOps>(x: T) -> (T, T) {
    let a = x.abs();
    let h = exp_scaled(a, -1.0);
    let q = x.lit(0.25) / h;
    let s = T::select(a.lt(1.0), sinh_series(x), (h - q) * sign(x));
    (s, h + q)
}

/// (sin y, cos y), from one reduction (accurate for |y| < `TRIG_LIMIT`)
fn sin_cos<T: MathOps>(y: T) -> (T, T) {
    let (n, r, t) = reduce_pio2(y);
    let (s, c) = (sin_kernel(r, t), cos_kernel(r, t));
    let sin = by_quadrant(n, s, c);
    let sin = T::select(y.abs().lt(1.4901161193847656e-8), y, sin);
    (sin, by_quadrant(n + 1.0, s, c))
}

/// |x| above which the complex tanh, coth, sech and csch use their limits
const COMPLEX_LARGE: f64 = 22.0;

/// sinh(x + iy)
pub fn csinh<T: MathOps>(x: T, y: T) -> (T, T) {
    let (sy, cy) = sin_cos(y);
    let (s, c) = sinh_cosh(x);
    (s * cy, c * sy)
}

/// cosh(x + iy)
pub fn ccosh<T: MathOps>(x: T, y: T) -> (T, T) {
    let (sy, cy) = sin_cos(y);
    let (s, c) = sinh_cosh(x);
    (c * cy, s * sy)
}

/// tanh(x + iy)
pub fn ctanh<T: MathOps>(x: T, y: T) -> (T, T) {
    let (sy, cy) = sin_cos(y);
    let (s, c) = sinh_cosh(x);
    let large = x.abs().gt(COMPLEX_LARGE);
    let d = s * s + cy * cy;
    let re = T::select(large, sign(x), s * c / d);
    let im = T::select(large, sy * cy / (c * c), sy * cy / d);
    (re, im)
}

/// coth(x + iy)
pub fn ccoth<T: MathOps>(x: T, y: T) -> (T, T) {
    let (sy, cy) = sin_cos(y);
    let (s, c) = sinh_cosh(x);
    let large = x.abs().gt(COMPLEX_LARGE);
    let d = s * s + sy * sy;
    let re = T::select(large, sign(x), s * c / d);
    let im = T::select(large, sy * cy / (c * c), sy * cy / d) * -1.0;
    (re, im)
}

/// sech(x + iy)
pub fn csech<T: MathOps>(x: T, y: T) -> (T, T) {
    let (sy, cy) = sin_cos(y);
    let (s, c) = sinh_cosh(x);
    let large = x.abs().gt(COMPLEX_LARGE);
    let d = s * s + cy * cy;
    let re = T::select(large, cy / c, c * cy / d);
    let im = T::select(large, sign(x) * sy / c, s * sy / d) * -1.0;
    (re, im)
}

/// csch(x + iy)
pub fn ccsch<T: MathOps>(x: T, y: T) -> (T, T) {
    let (sy, cy) = sin_cos(y);
    let (s, c) = sinh_cosh(x);
    let large = x.abs().gt(COMPLEX_LARGE);
    let d = s * s + sy * sy;
    let re = T::select(large, sign(x) * cy / c, s * cy / d);
    let im = T::select(large, sy / c, c * sy / d) * -1.0;
    (re, im)
}

/// exp(x + iy) = e^x cos y + i e^x sin y
pub fn cexp<T: MathOps>(x: T, y: T) -> (T, T) {
    let (sy, cy) = sin_cos(y);
    let e = exp(x);
    (e * cy, e * sy)
}

/// sin(x + iy) = sin x cosh y + i cos x sinh y
pub fn csin<T: MathOps>(x: T, y: T) -> (T, T) {
    let (sx, cx) = sin_cos(x);
    let (s, c) = sinh_cosh(y);
    (sx * c, cx * s)
}

/// cos(x + iy) = cos x cosh y - i sin x sinh y
pub fn ccos<T: MathOps>(x: T, y: T) -> (T, T) {
    let (sx, cx) = sin_cos(x);
    let (s, c) = sinh_cosh(y);
    (cx * c, sx * s * -1.0)
}

/// The complex functions of the library, with the real and imaginary parts of the argument
/// and of the result (`complexify`, `append_funclets`).
const COMPLEX_LIBRARY: &[(&str, TracedComplex)] = &[
    ("sinh", |x, y| csinh(x, y)),
    ("cosh", |x, y| ccosh(x, y)),
    ("tanh", |x, y| ctanh(x, y)),
    ("coth", |x, y| ccoth(x, y)),
    ("sech", |x, y| csech(x, y)),
    ("csch", |x, y| ccsch(x, y)),
    ("exp", |x, y| cexp(x, y)),
    ("sin", |x, y| csin(x, y)),
    ("cos", |x, y| ccos(x, y)),
];

/// A complex function of the library on traced values.
type TracedComplex = for<'t> fn(Val<'t>, Val<'t>) -> (Val<'t>, Val<'t>);

/// Whether the library has the complex function `name`.
pub(crate) fn has_complex_function(name: &str) -> bool {
    COMPLEX_LIBRARY.iter().any(|(n, _)| *n == name)
}

/// The bounds of |x| and |y| below which the library's complex `name` is used.
pub(crate) fn complex_domain(name: &str) -> (f64, f64) {
    match name {
        // e^x finite; sin and cos of y
        "exp" => (709.0, TRIG_LIMIT),
        // sin and cos of x; sinh and cosh of y
        "sin" | "cos" => (TRIG_LIMIT, 710.0),
        _ => (710.0, TRIG_LIMIT),
    }
}

/// Whether a complex kernel may call the real `name` only for a positive real argument (its
/// value is complex for a negative one): `ln`. (A complex kernel's `log` is the natural
/// logarithm, `VirtualTable::cplx_log`, but a real kernel's is log10, so complex kernels do not
/// call the library's `log`: `real_in_complex`.)
pub(crate) fn positive_domain(name: &str) -> bool {
    name == "ln"
}

/// Whether a complex kernel may call the library's real `name` for real arguments.
pub(crate) fn real_in_complex(name: &str) -> bool {
    has_function(name) && name != "log"
}

/// The arguments below which (in absolute value) the library's version of `name` is used;
/// the kernels call the C function for larger arguments, NaN and infinities.
pub(crate) fn domain_limit(name: &str) -> Option<f64> {
    match name {
        "sin" | "cos" => Some(TRIG_LIMIT),
        _ => None,
    }
}

// ln and log10 (fdlibm's e_log.c and e_log10.c): x = 2^k m with m in [sqrt(2)/2, sqrt(2))
// (`getexp`, `getmant`; subnormals scaled by 2^54 first), f = m - 1, s = f / (2 + f), and
// log(m) = f - f^2/2 + s (f^2/2 + R(s^2)) with fdlibm's minimax R. ln(x) = k ln 2 + log(m) with
// ln 2 in two parts; log10(x) = k log10(2) + log(m) / ln(10). Special values by selects:
// ln(0) = -inf, ln(x < 0) = NaN, ln(inf) = inf, ln(NaN) = NaN.

/// (k, f) with x = 2^k (1 + f), 1 + f in [sqrt(2)/2, sqrt(2)), for a positive finite x
fn log_reduce<T: MathOps>(x: T) -> (T, T) {
    let sub = x.lt(f64::MIN_POSITIVE);
    let xs = T::select(sub, x * 18014398509481984.0, x); // 2^54
    let k = xs.getexp() - T::select(sub, x.lit(54.0), x.lit(0.0));
    let m = xs.getmant();
    let big = m.gt(std::f64::consts::SQRT_2);
    let m = T::select(big, m * 0.5, m);
    let k = T::select(big, k + 1.0, k);
    (k, m - 1.0)
}

/// (f^2 / 2, s (f^2/2 + R(s^2))): log(1 + f) = f - f^2/2 + s (f^2/2 + R(s^2))
#[allow(clippy::excessive_precision)] // fdlibm's coefficients, as published
fn log_kernel<T: MathOps>(f: T) -> (T, T) {
    const LG1: f64 = 6.666666666666735130e-01;
    const LG2: f64 = 3.999999999940941908e-01;
    const LG3: f64 = 2.857142874366239149e-01;
    const LG4: f64 = 2.222219843214978396e-01;
    const LG5: f64 = 1.818357216161805012e-01;
    const LG6: f64 = 1.531383769920937332e-01;
    const LG7: f64 = 1.479819860511658591e-01;

    let hfsq = f * f * 0.5;
    let s = f / (f + 2.0);
    let z = s * s;
    let w = z * z;
    let t1 = w * ((w * LG6 + LG4) * w + LG2);
    let t2 = z * (((w * LG7 + LG5) * w + LG3) * w + LG1);
    (hfsq, s * (hfsq + (t2 + t1)))
}

/// The special values of a logarithm `r` of x: -inf for zeros, NaN for negative numbers
/// and NaN, inf for inf
#[allow(clippy::eq_op)] // x - x: NaN for NaN and the infinities, 0 otherwise
fn log_special<T: MathOps>(x: T, r: T) -> T {
    let r = r + (x - x); // NaN for NaN (and the infinities, fixed below)
    let r = T::select(x.gt(f64::MAX), x, r);
    let nonpositive = T::select(x.lt(0.0), x.lit(f64::NAN), x.lit(f64::NEG_INFINITY));
    T::select(x.lt(f64::from_bits(1)), nonpositive, r)
}

/// ln(x), the natural logarithm (symjit's `ln`)
pub fn ln<T: MathOps>(x: T) -> T {
    const LN2_HI: f64 = f64::from_bits(0x3fe6_2e42_fee0_0000);
    const LN2_LO: f64 = f64::from_bits(0x3dea_39ef_3579_3c76);
    let (k, f) = log_reduce(x);
    let (hfsq, t) = log_kernel(f);
    let r = k * LN2_HI - ((hfsq - (t + k * LN2_LO)) - f);
    log_special(x, r)
}

/// log10(x) (symjit's `log`)
#[allow(clippy::excessive_precision)] // fdlibm's constants, as published
pub fn log10<T: MathOps>(x: T) -> T {
    const IVLN10: f64 = std::f64::consts::LOG10_E; // 1 / ln(10), fdlibm's ivln10
    const LOG10_2HI: f64 = 3.01029995663611771306e-01;
    const LOG10_2LO: f64 = 3.69423907715893078616e-13;
    let (k, f) = log_reduce(x);
    let (hfsq, t) = log_kernel(f);
    let lnm = f - (hfsq - t);
    let r = k * LOG10_2HI + (k * LOG10_2LO + lnm * IVLN10);
    log_special(x, r)
}

/// The unary functions that `inline_calls` inlines: the name of the function in the models,
/// and the generic function on traced values.
const LIBRARY: &[(&str, Traced)] = &[
    ("exp", |x| exp(x)),
    ("sinh", |x| sinh(x)),
    ("cosh", |x| cosh(x)),
    ("tanh", |x| tanh(x)),
    ("coth", |x| coth(x)),
    ("sech", |x| sech(x)),
    ("csch", |x| csch(x)),
    ("sin", |x| sin(x)),
    ("cos", |x| cos(x)),
    ("ln", |x| ln(x)),
    ("log", |x| log10(x)),
];

/// A function of the library on traced values.
type Traced = for<'t> fn(Val<'t>) -> Val<'t>;

/***************************** tracing to MIR *****************************/

#[derive(Clone, Copy, Debug)]
enum Op {
    /// argument k: 0 in `Reg::Ret`, 1 in `Reg::Temp` (the imaginary part of a complex one)
    Arg(u8),
    Const(f64),
    Plus(usize, usize),
    Minus(usize, usize),
    Times(usize, usize),
    Divide(usize, usize),
    Lt(usize, usize),
    Abs(usize),
    Select(usize, usize, usize),
    // a * b + c, a * b - c, -a * b + c (`Trace::fuse`)
    MulAdd(usize, usize, usize),
    MulSub(usize, usize, usize),
    NegMulAdd(usize, usize, usize),
    Pow2i(usize),
    GetExp(usize),
    GetMant(usize),
}

impl Op {
    fn operands(&self) -> Vec<usize> {
        match *self {
            Op::Arg(_) | Op::Const(_) => vec![],
            Op::Plus(a, b)
            | Op::Minus(a, b)
            | Op::Times(a, b)
            | Op::Divide(a, b)
            | Op::Lt(a, b) => vec![a, b],
            Op::Select(m, a, b) => vec![m, a, b],
            Op::MulAdd(a, b, c) | Op::MulSub(a, b, c) | Op::NegMulAdd(a, b, c) => vec![a, b, c],
            Op::Pow2i(a) | Op::GetExp(a) | Op::GetMant(a) | Op::Abs(a) => vec![a],
        }
    }
}

/// The registers of the arguments and of the results of a funclet: a real value in `Ret`, a
/// complex one in `Ret` (real part) and `Temp` (imaginary part).
const OUT_REGS: [Reg; 2] = [Reg::Ret, Reg::Temp];

/// The operations of a function, in the order the generic function performs them.
#[derive(Default)]
struct Trace {
    ops: RefCell<Vec<Op>>,
}

impl Trace {
    fn push(&self, op: Op) -> usize {
        let mut ops = self.ops.borrow_mut();
        ops.push(op);
        ops.len() - 1
    }

    fn arg(&self) -> Val<'_> {
        self.arg_k(0)
    }

    fn arg_k(&self, k: u8) -> Val<'_> {
        Val {
            t: self,
            id: self.push(Op::Arg(k)),
        }
    }

    /// With fastmath, a product used once by a sum or a difference becomes a fused
    /// multiply-add. (The funclets are appended after the peephole optimizer, whose fusions
    /// assume that every value is used once.)
    fn fuse(&self, outs: &[usize]) {
        let mut ops = self.ops.borrow_mut();
        let mut uses = vec![0; ops.len()];
        for out in outs {
            uses[*out] += 1;
        }
        for op in ops.iter() {
            for a in op.operands() {
                uses[a] += 1;
            }
        }
        let product = |ops: &[Op], i: usize| match ops[i] {
            Op::Times(a, b) if uses[i] == 1 => Some((a, b)),
            _ => None,
        };
        for j in 0..ops.len() {
            let fused = match ops[j] {
                Op::Plus(t, c) => product(&ops, t)
                    .map(|(a, b)| Op::MulAdd(a, b, c))
                    .or_else(|| product(&ops, c).map(|(a, b)| Op::MulAdd(a, b, t))),
                Op::Minus(t, c) => product(&ops, t)
                    .map(|(a, b)| Op::MulSub(a, b, c))
                    .or_else(|| product(&ops, c).map(|(a, b)| Op::NegMulAdd(a, b, t))),
                _ => None,
            };
            if let Some(op) = fused {
                ops[j] = op;
            }
        }
    }

    /// Writes the trace as MIR computing `out` from the argument in `Reg::Ret`, with the
    /// result in `Reg::Ret`. Each value gets a register `Gen(0..count_regs)` until its last
    /// use (there are no stack slots: the compactor could give a slot of the caller, live
    /// across the call, to the funclet). Constants are appended to `consts`.
    #[cfg(test)]
    fn emit(&self, out: usize, mir: &mut Mir, consts: &mut Vec<f64>, count_regs: u8) -> Result<()> {
        self.emit_many(&[out], mir, consts, count_regs)
    }

    /// `emit` for functions of (`Reg::Ret`, `Reg::Temp`) (`Op::Arg`) with results in `Reg::Ret`
    /// and `Reg::Temp`: the real and imaginary parts of a complex function.
    fn emit_many(
        &self,
        outs: &[usize],
        mir: &mut Mir,
        consts: &mut Vec<f64>,
        count_regs: u8,
    ) -> Result<()> {
        if mir.config.fastmath() {
            self.fuse(outs);
        }
        let ops = self.ops.borrow();

        // the values `outs` depend on, and the last use of each
        let mut needed = vec![false; ops.len()];
        let mut last = vec![0; ops.len()];
        for out in outs {
            needed[*out] = true;
            last[*out] = ops.len();
        }
        for i in (0..ops.len()).rev() {
            if needed[i] {
                for a in ops[i].operands() {
                    needed[a] = true;
                    last[a] = last[a].max(i);
                }
            }
        }

        let mut free: Vec<u8> = (0..count_regs).rev().collect();
        let mut regs: Vec<Option<u8>> = vec![None; ops.len()];

        let alloc = |free: &mut Vec<u8>| {
            free.pop()
                .ok_or_else(|| anyhow!("libm: not enough registers to inline the function"))
        };

        for (i, op) in ops.iter().enumerate() {
            if !needed[i] {
                continue;
            }

            let src: Vec<Reg> = op
                .operands()
                .iter()
                .map(|a| Reg::Gen(regs[*a].unwrap()))
                .collect();

            // a select needs a scratch register besides its destination
            let scratch = if let Op::Select(..) = op {
                Some(alloc(&mut free)?)
            } else {
                None
            };

            // the operands used for the last time give back their registers first, so the
            // destination can be one of them
            for a in op.operands() {
                if last[a] == i {
                    if let Some(r) = regs[a].take() {
                        if !free.contains(&r) {
                            free.push(r);
                        }
                    }
                }
            }

            let d = alloc(&mut free)?;
            regs[i] = Some(d);
            let dst = Reg::Gen(d);

            match *op {
                Op::Arg(k) => mir.fmov(dst, OUT_REGS[k as usize]),
                Op::Const(v) => mir.load_const(dst, const_index(consts, v)),
                Op::Plus(..) => mir.plus(dst, src[0], src[1]),
                Op::Minus(..) => mir.minus(dst, src[0], src[1]),
                Op::Times(..) => mir.times(dst, src[0], src[1]),
                Op::Divide(..) => mir.divide(dst, src[0], src[1]),
                Op::Abs(..) => mir.abs(dst, src[0]),
                Op::Lt(..) => mir.lt(dst, src[0], src[1]),
                Op::Select(..) => {
                    // b ^ (m & (a ^ b)), with the symmetric `and` and `xor` (the ARM generators'
                    // `andnot` is s1 & !s2, the others' !s1 & s2)
                    let t = Reg::Gen(scratch.unwrap());
                    mir.xor(t, src[1], src[2]);
                    mir.and(t, src[0], t);
                    mir.xor(dst, src[2], t);
                }
                Op::MulAdd(..) => mir.fused_mul_add(dst, src[0], src[1], src[2]),
                Op::MulSub(..) => mir.fused_mul_sub(dst, src[0], src[1], src[2]),
                Op::NegMulAdd(..) => mir.fused_neg_mul_add(dst, src[0], src[1], src[2]),
                Op::Pow2i(..) => mir.pow2i(dst, src[0]),
                Op::GetExp(..) => mir.getexp(dst, src[0]),
                Op::GetMant(..) => mir.getmant(dst, src[0]),
            }

            if let Some(s) = scratch {
                free.push(s);
            }
        }

        for (out, r) in outs.iter().zip(OUT_REGS) {
            mir.fmov(r, Reg::Gen(regs[*out].unwrap()));
        }
        Ok(())
    }
}

/// The index of `v` in `consts`, appended if it is not there.
fn const_index(consts: &mut Vec<f64>, v: f64) -> u32 {
    match consts.iter().position(|c| c.to_bits() == v.to_bits()) {
        Some(idx) => idx as u32,
        None => {
            consts.push(v);
            (consts.len() - 1) as u32
        }
    }
}

/// A value of a traced function.
#[derive(Clone, Copy)]
pub struct Val<'t> {
    t: &'t Trace,
    id: usize,
}

impl<'t> Val<'t> {
    fn op(self, op: Op) -> Val<'t> {
        Val {
            t: self.t,
            id: self.t.push(op),
        }
    }
}

macro_rules! val_ops {
    ($($tr:ident $f:ident $op:ident),*) => {
        $(
            impl<'t> $tr for Val<'t> {
                type Output = Val<'t>;
                fn $f(self, rhs: Val<'t>) -> Val<'t> {
                    self.op(Op::$op(self.id, rhs.id))
                }
            }

            impl<'t> $tr<f64> for Val<'t> {
                type Output = Val<'t>;
                fn $f(self, rhs: f64) -> Val<'t> {
                    let c = self.lit(rhs);
                    self.op(Op::$op(self.id, c.id))
                }
            }
        )*
    };
}

val_ops!(Add add Plus, Sub sub Minus, Mul mul Times, Div div Divide);

impl<'t> MathOps for Val<'t> {
    type Mask = Val<'t>;

    fn lit(self, v: f64) -> Val<'t> {
        self.op(Op::Const(v))
    }

    fn gt(self, v: f64) -> Val<'t> {
        let c = self.lit(v);
        self.op(Op::Lt(c.id, self.id))
    }

    fn lt(self, v: f64) -> Val<'t> {
        let c = self.lit(v);
        self.op(Op::Lt(self.id, c.id))
    }

    fn select(m: Val<'t>, a: Val<'t>, b: Val<'t>) -> Val<'t> {
        m.op(Op::Select(m.id, a.id, b.id))
    }

    fn abs(self) -> Val<'t> {
        self.op(Op::Abs(self.id))
    }

    fn pow2i(self) -> Val<'t> {
        self.op(Op::Pow2i(self.id))
    }

    fn getexp(self) -> Val<'t> {
        self.op(Op::GetExp(self.id))
    }

    fn getmant(self) -> Val<'t> {
        self.op(Op::GetMant(self.id))
    }
}

/***************************** inlining *****************************/

pub(crate) fn funclet_label(name: &str) -> String {
    format!("{}{}", LIBM_PREFIX, name)
}

/// Whether the library has the (unary) function `name`.
pub(crate) fn has_function(name: &str) -> bool {
    LIBRARY.iter().any(|(n, _)| *n == name)
}

/// The funclets `@libm_<name>` of the functions `names` (those that fit in `count_regs`
/// registers), after a jump over them (`Branch(LIBM_END)` ... `Label(LIBM_END)`), and the
/// names of the functions they are for. Their constants are appended to `consts`.
fn funclets<'a>(
    config: &crate::config::Config,
    names: &[&'a str],
    consts: &mut Vec<f64>,
    count_regs: u8,
) -> (Mir, Vec<&'a str>) {
    let mut funclets = Mir::new(config.clone());
    let mut emitted = Vec::new();
    funclets.branch(LIBM_END);
    for name in names {
        // `cplx_<name>`: the complex function of (Ret, Temp) into (Ret, Temp)
        let trace = Trace::default();
        let outs = if let Some(base) = name.strip_prefix(LIBM_COMPLEX) {
            let Some((_, f)) = COMPLEX_LIBRARY.iter().find(|(n, _)| *n == base) else {
                continue;
            };
            let (re, im) = f(trace.arg_k(0), trace.arg_k(1));
            vec![re.id, im.id]
        } else {
            let Some((_, f)) = LIBRARY.iter().find(|(n, _)| n == name) else {
                continue;
            };
            vec![f(trace.arg()).id]
        };
        let mut body = Mir::new(config.clone());
        let mut body_consts = consts.clone();
        if trace
            .emit_many(&outs, &mut body, &mut body_consts, count_regs)
            .is_ok()
        {
            *consts = body_consts;
            funclets.set_label(&funclet_label(name));
            for ins in body.code.iter() {
                funclets.code.push(&ins);
            }
            funclets.branch(".ret");
            emitted.push(*name);
        }
    }
    funclets.set_label(LIBM_END);
    (funclets, emitted)
}

/// Appends the funclets of `names` to `mir`, with their constants in `mir.consts`
/// (`complexify`, whose MIR is compiled without `Builder::optimize_mir`). All of them must fit
/// in `count_regs` registers.
pub(crate) fn append_funclets(mir: &mut Mir, names: &[&str], count_regs: u8) -> Result<()> {
    let mut consts = std::mem::take(&mut mir.consts);
    let (funclets, emitted) = funclets(&mir.config, names, &mut consts, count_regs);
    mir.consts = consts;
    if emitted.len() != names.len() {
        return Err(anyhow!("libm: not enough registers to inline {:?}", names));
    }
    for ins in funclets.code.iter() {
        mir.code.push(&ins);
    }
    Ok(())
}

/// Replaces the calls of the library's functions (real kernels, `Config::inline_math`) by
/// calls of funclets appended to `mir`, whose constants are appended to `consts`. Runs after
/// the optimizations (`Builder::optimize_mir`): the funclets have their own registers and
/// fused multiply-adds (`Trace::emit`). The SIMD kernels run them; the other kernels call the
/// C functions (`Generator::inlines_math`). Complex kernels are handled by `complexify`.
pub(crate) fn inline_calls(mir: &mut Mir, consts: &mut Vec<f64>) -> Result<()> {
    if !mir.config.inline_math() || mir.config.is_complex() {
        return Ok(());
    }

    let mut used: Vec<&str> = Vec::new();
    for ins in mir.code.iter() {
        if let Instruction::Call { label, num_args: 1 } = &ins {
            if let Some((name, _)) = LIBRARY.iter().find(|(name, _)| name == label) {
                if !used.contains(name) {
                    used.push(name);
                }
            }
        }
    }

    if used.is_empty() {
        return Ok(());
    }

    // a function that needs too many registers is not inlined
    let count_regs = mir.config.count_scratch().min(12);
    let (funclets, inlined) = funclets(&mir.config, &used, consts, count_regs);

    let mut code = MirWriter::new();
    let mut count = 0;
    for ins in mir.code.iter() {
        match &ins {
            Instruction::Call { label, num_args: 1 } if inlined.contains(&label.as_str()) => {
                let call = Instruction::Call {
                    label: funclet_label(label),
                    num_args: 0,
                };
                if let Some(limit) = domain_limit(label) {
                    // |x| < limit (false for NaN and infinities): the library's function,
                    // else the C function. A SIMD row with both is computed by the scalar
                    // kernel (`branch_if`). The argument is in `Left`; the other registers
                    // are free at a call (calls clobber them).
                    let (inside, end) = (format!("@math_in_{count}"), format!("@math_end_{count}"));
                    count += 1;
                    for ins in [
                        Instruction::Uni {
                            op: UniOp::Abs,
                            dst: Reg::Temp,
                            s1: Reg::Left,
                        },
                        Instruction::LoadConst {
                            dst: Reg::Gen(0),
                            idx: const_index(consts, limit),
                        },
                        Instruction::Bi {
                            op: BinOp::LittleThan,
                            dst: Reg::Temp,
                            s1: Reg::Temp,
                            s2: Reg::Gen(0),
                        },
                        Instruction::BranchIf {
                            cond: Reg::Temp,
                            label: inside.clone(),
                            is_else: true,
                        },
                        ins.clone(),
                        Instruction::Branch { label: end.clone() },
                        Instruction::Label { label: inside },
                        call,
                        Instruction::Label { label: end },
                    ] {
                        code.push(&ins);
                    }
                } else {
                    code.push(&call);
                }
            }
            _ => code.push(&ins),
        }
    }
    for ins in funclets.code.iter() {
        code.push(&ins);
    }

    mir.code = code;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_f64_against_std() {
        // within 1 ulp of the C library's (correctly rounded in practice) exp
        let mut worst = 0u64;
        for i in 0..200001 {
            let x = -745.0 + 1455.0 * i as f64 / 200000.0;
            let (a, b) = (exp(x), x.exp());
            if b.is_finite() && b > 0.0 {
                let d = a.to_bits().abs_diff(b.to_bits());
                assert!(d <= 1, "exp({x:e}) = {a:e}, std: {b:e}");
                worst = worst.max(d);
            } else {
                assert_eq!(a, b, "exp({x:e})");
            }
        }
        for x in [
            0.0,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            1e300,
            -1e300,
            709.78,
            709.79,
            -745.13,
            -745.14,
        ] {
            assert_eq!(exp(x).to_bits(), x.exp().to_bits(), "exp({x:e})");
        }
        assert!(exp(f64::NAN).is_nan());
        assert!(worst <= 1);
    }

    fn inputs() -> Vec<f64> {
        let mut xs = vec![
            0.0,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            709.78,
            709.79,
            -745.13,
            -745.14,
            1e300,
            -1e300,
            1e-300,
        ];
        xs.extend((0..3000).map(|i| -745.0 + 1455.0 * (i as f64 * 0.618034).fract()));
        xs.extend((0..1000).map(|i| (i as f64 * 0.7548777).fract() * 2.0 - 1.0));
        xs
    }

    #[test]
    fn library_fits_in_registers() {
        // `inline_calls` and `append_funclets` give the funclets 12 registers
        for fastmath in [false, true] {
            for (name, f) in LIBRARY {
                let mut config = crate::config::Config::default();
                config.set_fastmath(fastmath);
                let fits = |count_regs: u8| {
                    let trace = Trace::default();
                    let out = f(trace.arg()).id;
                    let mut mir = Mir::new(config.clone());
                    trace
                        .emit(out, &mut mir, &mut Vec::new(), count_regs)
                        .is_ok()
                };
                let needed = (1..=16).find(|&n| fits(n)).unwrap();
                println!("{name} (fastmath={fastmath}): {needed} registers");
                assert!(needed <= 12, "{name} needs {needed} registers");
            }
        }
        for fastmath in [false, true] {
            for (name, f) in COMPLEX_LIBRARY {
                let mut config = crate::config::Config::default();
                config.set_fastmath(fastmath);
                let fits = |count_regs: u8| {
                    let trace = Trace::default();
                    let (re, im) = f(trace.arg_k(0), trace.arg_k(1));
                    let mut mir = Mir::new(config.clone());
                    trace
                        .emit_many(&[re.id, im.id], &mut mir, &mut Vec::new(), count_regs)
                        .is_ok()
                };
                let needed = (1..=16).find(|&n| fits(n)).unwrap();
                println!("complex {name} (fastmath={fastmath}): {needed} registers");
                assert!(needed <= 12, "complex {name} needs {needed} registers");
            }
        }
    }

    type Variant = (&'static str, fn(&mut Config));

    fn variants() -> Vec<Variant> {
        vec![
            ("default", |_| {}),
            ("opt0", |c| c.set_opt_level(0)),
            ("opt1", |c| c.set_opt_level(1)),
            ("opt3", |c| c.set_opt_level(3)),
            ("no-simd", |c| c.set_simd(false)),
            ("simd512", |c| c.enable_simd512(true)),
            ("cse-off", |c| c.set_cse(false)),
        ]
    }

    use crate::compiler::Translator;
    use crate::composer::Composer;
    use crate::config::Config;
    use crate::instruction::Slot;

    // f(x) and f(x - 3) + x (no multiply-add that fastmath could fuse in the argument)
    fn compile(config: Config, name: &str) -> crate::Application {
        let mut tr = Translator::new(config);
        tr.set_num_params(1);
        let three = tr
            .append_constant(num_complex::Complex::new(-3.0, 0.0))
            .unwrap();
        tr.append_fun(&Slot::Out(0), name, &[Slot::Param(0)], false)
            .unwrap();
        tr.append_add(&Slot::Temp(1), &[Slot::Param(0), Slot::Const(three)], 0)
            .unwrap();
        tr.append_fun(&Slot::Temp(2), name, &[Slot::Temp(1)], false)
            .unwrap();
        tr.append_add(&Slot::Out(1), &[Slot::Temp(2), Slot::Param(0)], 0)
            .unwrap();
        tr.compile().unwrap()
    }

    /// The C library's version of a function of the library, as the scalar kernels call it
    /// (`code.rs`).
    fn c_function(name: &str) -> fn(f64) -> f64 {
        match name {
            "exp" => f64::exp,
            "sinh" => f64::sinh,
            "cosh" => f64::cosh,
            "tanh" => f64::tanh,
            "coth" => |x| 1.0 / x.tanh(),
            "sech" => |x| 1.0 / x.cosh(),
            "csch" => |x| 1.0 / x.sinh(),
            "sin" => f64::sin,
            "cos" => f64::cos,
            "ln" => f64::ln,
            "log" => f64::log10,
            _ => unreachable!(),
        }
    }

    fn library_function(name: &str) -> fn(f64) -> f64 {
        match name {
            "exp" => exp,
            "sinh" => sinh,
            "cosh" => cosh,
            "tanh" => tanh,
            "coth" => coth,
            "sech" => sech,
            "csch" => csch,
            "sin" => sin,
            "cos" => cos,
            "ln" => ln,
            "log" => log10,
            _ => unreachable!(),
        }
    }

    #[test]
    fn hyperbolic_f64() {
        // against the C library's (whose 1 / f(x) are themselves an ulp or so off)
        for name in ["sinh", "cosh", "tanh", "coth", "sech", "csch"] {
            let (f, g) = (library_function(name), c_function(name));
            for i in 0..100001 {
                let x = -705.0 + 1410.0 * i as f64 / 100000.0;
                for x in [x, x * 1e-3] {
                    let (a, b) = (f(x), g(x));
                    if b.is_normal() {
                        let d = a.to_bits().abs_diff(b.to_bits());
                        assert!(d <= 6, "{name}({x:e}) = {a:e}, C: {b:e}");
                    }
                }
            }
        }
        let inf = f64::INFINITY;
        let cases: [(&str, f64, f64); 24] = [
            ("sinh", -0.0, -0.0),
            ("sinh", -inf, -inf),
            ("sinh", 710.4, 1.6663642832806494e308),
            ("sinh", 710.5, inf),
            ("cosh", -0.0, 1.0),
            ("cosh", -inf, inf),
            ("cosh", -710.4, 1.6663642832806494e308),
            ("cosh", 711.0, inf),
            ("tanh", -0.0, -0.0),
            ("tanh", -inf, -1.0),
            ("tanh", 1e-300, 1e-300),
            ("tanh", 800.0, 1.0),
            ("coth", -0.0, -inf),
            ("coth", 0.0, inf),
            ("coth", -inf, -1.0),
            ("coth", 1e-300, 1e300),
            ("sech", 0.0, 1.0),
            ("sech", inf, 0.0),
            ("sech", 720.0, 4.06446160484e-313),
            ("sech", -745.0, 9.881312916824931e-324),
            ("csch", -0.0, -inf),
            ("csch", -inf, -0.0),
            ("csch", -720.0, -4.06446160484e-313),
            ("csch", 1e-300, 1e300),
        ];
        for (name, x, want) in cases {
            let got = library_function(name)(x);
            let close = got.to_bits() == want.to_bits()
                || (want.is_normal() && ((got - want) / want).abs() < 4e-16)
                || (want.abs() < 1e-300 && (got - want).abs() <= 1e-323);
            assert!(close, "{name}({x:e}) = {got:e}, not {want:e}");
        }
        for name in ["sinh", "cosh", "tanh", "coth", "sech", "csch"] {
            assert!(library_function(name)(f64::NAN).is_nan(), "{name}(NaN)");
        }
    }

    /// The library's complex version of `name`, if it has one.
    fn complex_library_function(name: &str) -> Option<fn(f64, f64) -> (f64, f64)> {
        match name {
            "sinh" => Some(csinh),
            "cosh" => Some(ccosh),
            "tanh" => Some(ctanh),
            "coth" => Some(ccoth),
            "sech" => Some(csech),
            "csch" => Some(ccsch),
            "exp" => Some(cexp),
            "sin" => Some(csin),
            "cos" => Some(ccos),
            _ => None,
        }
    }

    #[test]
    fn complex_f64() {
        use crate::code::VirtualTable as V;
        use num_complex::Complex;
        type CFunc = extern "C" fn(f64, f64, &mut Complex<f64>);
        let fs: [(&str, CFunc); 9] = [
            ("exp", V::cplx_exp),
            ("sin", V::cplx_sin),
            ("cos", V::cplx_cos),
            ("sinh", V::cplx_sinh),
            ("cosh", V::cplx_cosh),
            ("tanh", V::cplx_tanh),
            ("coth", V::cplx_coth),
            ("sech", V::cplx_sech),
            ("csch", V::cplx_csch),
        ];
        // against the complex C functions (the C path's tanh, `tanh_c`, used the same formula
        // as the library's since 0020)
        for (name, g) in fs {
            let f = complex_library_function(name).unwrap();
            for i in 0..100000 {
                let t = (i as f64 * 0.6180339887).fract();
                let u = (i as f64 * 0.7548776662).fract();
                let (x, y) = match i % 3 {
                    0 => (t * 60.0 - 30.0, u * 20.0 - 10.0),
                    1 => (t * 1400.0 - 700.0, u * 2e3 - 1e3),
                    _ => (t * 4.0 - 2.0, u * 4.0 - 2.0),
                };
                let (re, im) = f(x, y);
                let mut z = Complex::new(0.0, 0.0);
                g(x, y, &mut z);
                let n = z.norm();
                if n.is_finite() && n > 1e-300 {
                    let e = (re - z.re).abs().max((im - z.im).abs()) / n;
                    assert!(e < 2e-15, "{name}({x:e}, {y:e}) = ({re:e}, {im:e}), C: {z}");
                }
            }
        }
        // real arguments: the real functions (and cosh x sin(+0) = +0)
        for x in [-3.5, -0.25, 0.0, 0.75, 20.0, 600.0] {
            let (re, im) = csinh(x, 0.0);
            assert_eq!(
                (re.to_bits(), im.to_bits()),
                (sinh(x).to_bits(), 0),
                "sinh({x})"
            );
            let (re, _) = ccosh(x, 0.0);
            assert_eq!(re.to_bits(), cosh(x).to_bits(), "cosh({x})");
        }
    }

    #[test]
    fn logarithm_f64() {
        for (name, f, g, ulps) in [
            ("ln", ln as fn(f64) -> f64, f64::ln as fn(f64) -> f64, 1),
            ("log10", log10, f64::log10, 2),
        ] {
            for i in 0..300000 {
                let t = (i as f64 * 0.6180339887).fract();
                for x in [
                    t * 4.0,
                    (t * 1400.0 - 700.0).exp(),
                    1.0 + (t - 0.5) * 1e-6,
                    f64::from_bits((t * 4.5e15) as u64 + 1), // subnormals
                ] {
                    let (a, b) = (f(x), g(x));
                    let d = a.to_bits().abs_diff(b.to_bits());
                    assert!(d <= ulps || a == b, "{name}({x:e}) = {a:e}, C: {b:e}");
                }
            }
            for x in [
                0.0,
                -0.0,
                -1.0,
                f64::INFINITY,
                f64::NEG_INFINITY,
                1.0,
                5e-324,
                f64::MAX,
            ] {
                let (a, b) = (f(x), g(x));
                assert!(
                    a == b || (a.is_nan() && b.is_nan()),
                    "{name}({x:e}) = {a:e}, C: {b:e}"
                );
            }
            assert!(f(f64::NAN).is_nan());
        }
    }

    #[test]
    fn trigonometric_f64() {
        use std::f64::consts::FRAC_PI_2;
        // against the C library's, also close to multiples of pi/2
        for (name, f, g) in [
            ("sin", sin as fn(f64) -> f64, f64::sin as fn(f64) -> f64),
            ("cos", cos, f64::cos),
        ] {
            for i in 0..400001 {
                let t = i as f64 / 400000.0;
                for x in [
                    t * 10.0 - 5.0,
                    (t - 0.5) * 2.0 * TRIG_LIMIT,
                    (t - 0.5) * 2e3,
                    FRAC_PI_2 * (i as f64 - 2e5) * 5.0,
                ] {
                    let (a, b) = (f(x), g(x));
                    let d = a.to_bits().abs_diff(b.to_bits());
                    assert!(
                        d <= 2 || (a - b).abs() < 1e-300,
                        "{name}({x:e}) = {a:e}, C: {b:e}"
                    );
                }
            }
        }
        assert_eq!(sin(-0.0f64).to_bits(), (-0.0f64).to_bits());
        assert_eq!(sin(1e-310f64), 1e-310);
        assert_eq!(cos(-0.0f64), 1.0);
        assert_eq!(cos(std::f64::consts::PI), -1.0);
    }

    // The SIMD kernels run the library's functions; the scalar kernel (one point at a time,
    // the rows left after the last full SIMD row, and kernels without SIMD) calls the C
    // library's.
    #[test]
    fn library_inlined_in_simd_kernels() {
        use crate::utils::Compiled;

        let xs = inputs();
        let n = xs.len();
        let same = |a: f64, b: f64| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());

        for (name, _) in LIBRARY {
            let (f, g) = (library_function(name), c_function(name));
            // outside a function's domain the kernels call the C function
            let inside = |x: f64| domain_limit(name).is_none_or(|l| x.abs() < l);
            let ours: Vec<[f64; 2]> = xs.iter().map(|&x| [f(x), f(x - 3.0) + x]).collect();
            let c_lib: Vec<[f64; 2]> = xs.iter().map(|&x| [g(x), g(x - 3.0) + x]).collect();
            // the library differs from the C library at some points: equal bits show which
            // one a kernel uses
            assert!(xs.iter().any(|x| f(*x) != g(*x)), "{name}");

            for fastmath in [false, true] {
                for (vname, variant) in variants() {
                    let mut config = Config::from_name("native", Config::default().opt).unwrap();
                    config.set_simd(true);
                    config.set_opt_level(2);
                    config.set_fastmath(fastmath);
                    config.set_option("inline_math", "true").unwrap();
                    variant(&mut config);
                    let app = compile(config, name);
                    let tag = format!("{name} {vname} fastmath={fastmath}");
                    let lanes = app.compiled_simd.as_ref().map_or(0, |c| c.count_lanes());

                    // fused multiply-adds change the rounding of some results by an ulp or two
                    let close = |a: f64, w: f64| {
                        same(a, w)
                            || (fastmath
                                && a.is_finite()
                                && (a.to_bits().abs_diff(w.to_bits()) <= 2
                                    || (a - w).abs() <= 1e-15 * w.abs().max(1.0)))
                    };

                    for (i, x) in xs.iter().enumerate() {
                        let mut outs = [0.0; 2];
                        app.evaluate(&[*x], &mut outs);
                        assert!(
                            same(outs[0], c_lib[i][0]) && same(outs[1], c_lib[i][1]),
                            "{tag} scalar x = {x:e}: {outs:?} != {:?}",
                            c_lib[i]
                        );
                    }

                    let mut outs = vec![0.0; 2 * n];
                    app.evaluate_matrix(&xs, &mut outs, n);
                    let simd_rows = if lanes > 1 { n / lanes * lanes } else { 0 };
                    for i in 0..n {
                        let got = [outs[2 * i], outs[2 * i + 1]];
                        // a SIMD row with an argument outside the domain: the scalar kernel
                        let row =
                            i / lanes.max(1) * lanes.max(1)..(i / lanes.max(1) + 1) * lanes.max(1);
                        let in_domain = row
                            .clone()
                            .all(|j| j >= n || (inside(xs[j]) && inside(xs[j] - 3.0)));
                        let ok = if i < simd_rows && in_domain {
                            close(got[0], ours[i][0]) && close(got[1], ours[i][1])
                        } else {
                            same(got[0], c_lib[i][0]) && same(got[1], c_lib[i][1])
                        };
                        assert!(
                            ok,
                            "{tag} matrix row {i} (SIMD rows: {simd_rows}) x = {:e}: {got:?}, \
                             library {:?}, C {:?}",
                            xs[i], ours[i], c_lib[i]
                        );
                    }
                }
            }
        }
    }

    // Complex kernels (`complexify`): the hyperbolic functions have complex funclets, used by
    // the SIMD rows whose arguments are all in their domain (real or complex); for exp, sin
    // and cos, SIMD rows whose arguments are all real get the real function. The other rows
    // (and single points) get the complex C function, as without `inline_math`.
    #[test]
    fn complex_kernels() {
        use crate::utils::Compiled;
        use num_complex::Complex;

        let n = 1024;
        let re: Vec<f64> = (0..n)
            .map(|i| 12.0 * (i as f64 * 0.618034).fract() - 6.0)
            .collect();
        // groups of 8 points: real, complex, mixed (lanes 0-3 real), complex with half of the
        // real parts outside the domain of the complex funclets, ...
        let zs: Vec<Complex<f64>> = (0..n)
            .map(|i| match (i / 8) % 4 {
                0 => Complex::new(re[i], 0.0),
                1 => Complex::new(re[i], 0.5 + (i % 7) as f64 * 0.25),
                2 if i % 8 < 4 => Complex::new(re[i], 0.0),
                2 => Complex::new(re[i], -1.25),
                _ if i % 8 < 4 => Complex::new(re[i] + 800.0, 0.5),
                _ => Complex::new(re[i], 0.5),
            })
            .collect();
        let in_domain = |name: &str, z: &Complex<f64>| {
            let (lx, ly) = complex_domain(name);
            z.re.abs() < lx && z.im.abs() < ly
        };

        for (name, _) in LIBRARY {
            let f = library_function(name);
            for fast_complex in [true, false] {
                for (vname, variant) in variants() {
                    let build = |inline: bool| {
                        let mut config =
                            Config::from_name("native", Config::default().opt).unwrap();
                        config.set_complex(true);
                        config.set_simd(true);
                        config.set_opt_level(2);
                        config.set_fastmath(false);
                        config
                            .set_option("fast_complex", &fast_complex.to_string())
                            .unwrap();
                        config
                            .set_option("inline_math", &inline.to_string())
                            .unwrap();
                        variant(&mut config);
                        let mut tr = Translator::new(config);
                        tr.set_num_params(1);
                        tr.append_fun(&Slot::Out(0), name, &[Slot::Param(0)], false)
                            .unwrap();
                        tr.compile().unwrap()
                    };
                    let (app, c_app) = (build(true), build(false));
                    let tag = format!("{name} {vname} fast_complex={fast_complex}");
                    let lanes = app.compiled_simd.as_ref().map_or(0, |c| c.count_lanes());

                    let mut got = vec![Complex::new(0.0, 0.0); n];
                    let mut c_lib = vec![Complex::new(0.0, 0.0); n];
                    app.evaluate_matrix(&zs, &mut got, n);
                    c_app.evaluate_matrix(&zs, &mut c_lib, n);
                    let same =
                        |a: f64, b: f64| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());

                    let complex_f = complex_library_function(name);
                    for i in 0..n {
                        let full_row = lanes > 1 && (i / lanes + 1) * lanes <= n;
                        // the real function: real arguments (positive for the logarithms)
                        let row_real = full_row
                            && (i / lanes * lanes..(i / lanes + 1) * lanes).all(|j| {
                                zs[j].im == 0.0 && (!positive_domain(name) || zs[j].re > 0.0)
                            })
                            && real_in_complex(name);
                        let row_in_domain = full_row
                            && (i / lanes * lanes..(i / lanes + 1) * lanes)
                                .all(|j| in_domain(name, &zs[j]));
                        let (w, ok) = if let (true, Some(cf)) = (row_in_domain, complex_f) {
                            // the complex funclet, also for real arguments
                            let (re, im) = cf(zs[i].re, zs[i].im);
                            (
                                Complex::new(re, im),
                                same(got[i].re, re) && same(got[i].im, im),
                            )
                        } else if row_real {
                            let w = Complex::new(f(zs[i].re), 0.0);
                            (w, same(got[i].re, w.re) && got[i].im.to_bits() == 0)
                        } else {
                            // a real argument gives +0 as the imaginary part (the complex C
                            // function gives a signed zero)
                            let w = c_lib[i];
                            (
                                w,
                                same(got[i].re, w.re)
                                    && (same(got[i].im, w.im) || got[i].im == w.im),
                            )
                        };
                        assert!(
                            ok,
                            "{tag} z = {}: {} != {} (real row: {row_real})",
                            zs[i], got[i], w
                        );
                    }

                    // one point at a time: the complex C function
                    for z in zs.iter().step_by(37) {
                        let (a, b) = (app.evaluate_single(&[*z]), c_app.evaluate_single(&[*z]));
                        assert!(
                            same(a.re, b.re) && (same(a.im, b.im) || a.im == b.im),
                            "{tag} single z = {z}: {a} != {b}"
                        );
                    }
                }
            }
        }
    }

    // `libm::exp` on the kernel builder's `Real` builds an exp kernel
    #[cfg(feature = "composer")]
    #[test]
    fn exp_with_the_kernel_builder() {
        use crate::kernel::{KernelBuilder, Real};

        for direct in [false, true] {
            let mut config = Config::from_name("native", Config::default().opt).unwrap();
            config.set_fastmath(false);
            config.set_direct(direct);
            let k = KernelBuilder::new(config);
            let x: Real = k.param();
            let app = k.compile(exp(x)).unwrap().seal().unwrap();
            for x in inputs() {
                let got = app.evaluate_single(&[x]);
                let w = exp(x);
                assert!(
                    got.to_bits() == w.to_bits() || (got.is_nan() && w.is_nan()),
                    "direct={direct} x = {x:e}: {got} != {w}"
                );
            }
        }
    }
}

// Kernels compiled both by the `#[kernel]` macro (this file is `include!`d by
// tests/source.rs) and by sjc's interpreter (which reads it as text). Every kernel here must
// compile with the macro.

use symjit::kernel::kernel;
use symjit::Complex;

const MIN_SIDE: f64 = 1.0 / 1024.0;
const N: usize = 4;
const STEPS: i32 = 3;

// arithmetic and functions (kernel_diff.rs)

#[kernel]
pub fn arith(x: f64, y: f64, z: f64) -> (f64, f64, f64, f64) {
    let a = x * y - z / (y * y + 1.0);
    let b = (x - y) * (x + y) - z.recip() * 0.5;
    let c = x.abs().sqrt() * y.min(z) - x.max(y) + z.powi(3);
    let d = (x * 3.0).floor() + (y * 2.5).ceil() - z.trunc() + (x * 7.0).round();
    (a, b, c, d)
}

#[kernel]
pub fn elementary(x: f64, y: f64) -> (f64, f64, f64, f64, f64) {
    let a = x.exp() * y.sin() + y.cos();
    let b = (x * x + 1.0).ln() - y.atan() + x.tanh();
    let c = (x.abs() + 0.5).powf(y) + x.cbrt();
    let d = x.sinh() - (y * 0.5).cosh() + (x * 0.25).tan();
    let e = (x * 0.3).asin() + (y * 0.3).acos() + x.asinh();
    (a, b, c, d, e)
}

#[kernel]
pub fn piecewise(x: f64, y: f64) -> (f64, f64) {
    let mut v = 1.0 - x;
    let mut w = y;
    if x < -1.0 {
        v = x * x;
        if y > 0.0 {
            w = y - x;
        }
    } else if x < 1.0 && y < 0.0 {
        v = x + y;
    } else if x >= 2.0 || y >= 2.0 {
        v = x * y;
        w = -w;
    } else {
        w = w * 0.5 + v;
    }
    (v, w)
}

#[kernel]
pub fn tree(x: f64, depth: f64) -> f64 {
    if depth <= 0.0 {
        x
    } else if x < 0.0 {
        tree(-x * 0.5, depth - 1.0) + 1.0
    } else {
        tree(x - 1.0, depth - 1.0) * 0.5 + tree(x * 0.25, depth - 2.0)
    }
}

#[kernel]
pub fn complex_ops(z: Complex<f64>, w: Complex<f64>) -> (Complex<f64>, Complex<f64>, Complex<f64>) {
    let a = z * w + z.conj() - w * 2.0;
    let b = z / (w * w + 4.0) + Complex::new(z.norm(), w.norm_sqr());
    let c = z.exp() * w.sin() + (w * w + 1.0).ln() - z.powi(3);
    (a, b, c)
}

// branches, helpers, recursion and loops (kernel_macro.rs)

#[kernel(inline)]
pub fn inside(z: Complex<f64>) -> f64 {
    if z.norm_sqr() <= 1.0 {
        1.0
    } else {
        0.0
    }
}

#[kernel(inline)]
pub fn mid(a: Complex<f64>, b: Complex<f64>) -> Complex<f64> {
    (a + b) * 0.5
}

#[kernel]
pub fn area(bl: Complex<f64>, tl: Complex<f64>, tr: Complex<f64>, br: Complex<f64>) -> f64 {
    let n = inside(bl) + inside(tl) + inside(tr) + inside(br);
    let (x0, y0, x1, y1) = (bl.re, bl.im, tr.re, tr.im);
    let (w, h) = (x1 - x0, y1 - y0);
    let crosses = x0 * x1 < 0.0 || y0 * y1 < 0.0;

    if n != 4.0 && (n != 0.0 || crosses) && w > MIN_SIDE && h > MIN_SIDE {
        let (left, top, right, bottom) = (mid(bl, tl), mid(tl, tr), mid(tr, br), mid(br, bl));
        let center = mid(bl, tr);
        area(bl, left, center, bottom)
            + area(left, tl, top, center)
            + area(center, top, tr, right)
            + area(bottom, center, right, br)
    } else {
        w * h * n * 0.25
    }
}

#[kernel]
pub fn fib(n: f64) -> f64 {
    if n < 2.0 {
        n
    } else {
        fib(n - 1.0) + fib(n - 2.0)
    }
}

#[kernel(inline)]
pub fn horner(x: f64, coefs: [f64; 4]) -> f64 {
    let mut acc: f64 = 0.0;
    for i in (0..4).rev() {
        acc = acc * x + coefs[i];
    }
    acc
}

#[allow(clippy::assign_op_pattern)]
#[kernel(inline)]
pub fn power(x: f64, n: i32) -> f64 {
    let mut p: f64 = 1.0;
    for i in 0..n {
        if i % 2 == 0 {
            p *= x;
        }
        if i % 2 == 1 {
            p = p * x;
        }
    }
    p
}

#[kernel]
pub fn several(x: f64, y: f64) -> (f64, f64, f64, f64) {
    let largest: f64 = if x > y { x } else { y };
    let sign = if x < 0.0 {
        -1.0
    } else if x > 0.0 {
        1.0
    } else {
        0.0
    };
    let half = 0.5;
    let p = horner(x, [1.0, -2.0, half, 0.75]) + horner(y, [x, 1.5, -0.25, y]);
    let scale = |v: f64| v * 0.25 + y;
    (largest, sign, p, scale(power(x, 5)))
}

#[kernel]
pub fn rotate(z: Complex<f64>, t: f64) -> Complex<f64> {
    let r = Complex::new(t.cos(), t.sin());
    let w = z * r;
    if w.re < 0.0 && t != 0.0 {
        w.conj()
    } else {
        w + Complex::new(0.0, 1.0)
    }
}

#[kernel]
pub fn count_inside(a: Complex<f64>, b: Complex<f64>, c: Complex<f64>) -> (f64, f64) {
    let mut n = 0.0;
    let mut last = -1.0;
    for (i, z) in [a, b, c].into_iter().enumerate() {
        if z.norm_sqr() <= 1.0 {
            n += 1.0;
            last = i as f64;
        }
    }
    (n, last)
}

#[kernel]
pub fn newton_sqrt(a: f64) -> (f64, f64) {
    let mut x = if a > 1.0 { a } else { 1.0 };
    let mut steps = 0.0;
    while (x * x - a).abs() > 1e-12 * a {
        x = 0.5 * (x + a / x);
        steps += 1.0;
    }
    (x, steps)
}

#[kernel]
pub fn mandelbrot(c: Complex<f64>) -> f64 {
    let mut z: Complex<f64> = Complex::new(0.0, 0.0);
    let mut n = 0.0;
    while n < 100.0 && z.norm_sqr() <= 4.0 {
        z = z * z + c;
        n += 1.0;
    }
    n
}

#[kernel]
pub fn mandelbrot_untyped(c: Complex<f64>) -> f64 {
    let mut z = Complex::new(0.0, 0.0);
    let mut n = 0.0;
    while n < 100.0 && z.norm_sqr() <= 4.0 {
        z = z * z + c;
        n += 1.0;
    }
    n
}

#[kernel]
pub fn fib_iter(n: f64) -> f64 {
    let (mut a, mut b): (f64, f64) = (0.0, 1.0);
    let mut i = 0.0;
    while i < n {
        (a, b) = (b, a + b);
        i += 1.0;
    }
    a
}

#[kernel]
pub fn collatz(n: f64) -> f64 {
    let mut x = n;
    let mut steps = 0.0;
    while x != 1.0 {
        let half = x * 0.5;
        if half.floor() == half {
            x = half;
        } else {
            x = 3.0 * x + 1.0;
        }
        steps += 1.0;
    }
    steps
}

// docs/KERNELS.md

#[kernel]
pub fn escape(c: Complex<f64>, limit: f64) -> f64 {
    let mut z = Complex::new(0.0, 0.0);
    let mut n = 0.0;
    while n < limit && z.norm_sqr() <= 4.0 {
        z = z * z + c;
        n += 1.0;
    }
    n
}

#[kernel(inline)]
pub fn midpoint(a: f64, h: f64, i: f64) -> f64 {
    a + (i + 0.5) * h
}

#[kernel]
pub fn midpoint_rule(a: f64, b: f64, n: f64) -> f64 {
    let h = (b - a) / n;
    let mut sum = 0.0;
    let mut i = 0.0;
    while i < n {
        let x = midpoint(a, h, i);
        sum += x.cos() / (1.0 + x * x);
        i += 1.0;
    }
    sum * h
}

// more constructs

/// Masks: outputs, `!`, `^`, `&&`/`||` of comparisons, a branch on a combination.
#[kernel]
pub fn masks(x: f64, y: f64) -> (bool, f64, bool, bool) {
    let a = x < y;
    let b = x * y >= 0.25;
    let s = if a && !b { x } else { y * 2.0 };
    (a && b, s, x == y || x > 1.5, a ^ b)
}

/// Build-time code: constants of the file, unrolled loops over arrays and ranges, integer
/// arithmetic, `f64` methods of numbers, closures, conditions on numbers.
#[kernel]
pub fn unrolled(x: f64, ys: [f64; 4]) -> (f64, f64, [f64; 4]) {
    let weights = [0.5, 0.25, 0.125, 0.0625];
    let mut dot = 0.0;
    for (w, y) in weights.iter().zip(ys.iter()) {
        dot += *w * *y;
    }
    let mut poly = 0.0;
    for k in 0..STEPS {
        let c = (k as f64 + 1.0).sqrt();
        if k % 2 == 0 {
            poly = poly * x + c;
        } else {
            poly = poly * x - c;
        }
    }
    let shift = |v: f64, by: i32| v + (by as f64) * std::f64::consts::PI;
    let mut out = [x; N];
    for i in 0..N {
        out[i] = shift(ys[i], i as i32) * f64::max(0.5, 0.25 * i as f64);
    }
    (dot, poly, out)
}

/// A kernel `if` whose sides give a real and a complex value, and assign variables.
#[kernel]
pub fn mixed(z: Complex<f64>, t: f64) -> (Complex<f64>, f64, Complex<f64>) {
    let mut acc = Complex::new(1.0, 0.0);
    let mut count = 0.0;
    let v = if t > 0.5 {
        acc = acc * z;
        count += 1.0;
        z * t
    } else if t > -0.5 {
        count -= 1.0;
        Complex::new(t, -t)
    } else {
        acc = z.conj();
        Complex::new(1.0, 2.0)
    };
    let r = if z.re > z.im { t * 2.0 } else { 0.5 };
    (v, count + r, acc)
}

/// A kernel loop with a kernel branch and an inline helper with an integer argument.
#[kernel]
pub fn bisect(a: f64, b: f64, target: f64) -> (f64, f64) {
    let (mut lo, mut hi) = (a, b);
    let mut iters = 0.0;
    while hi - lo > 1e-9 && iters < 200.0 {
        let m = 0.5 * (lo + hi);
        if power(m, 3) < target {
            lo = m;
        } else {
            hi = m;
        }
        iters += 1.0;
    }
    (0.5 * (lo + hi), iters)
}

/// Shadowing, blocks as values, tuple fields and nested tuples.
#[kernel]
pub fn shapes(x: f64, p: (f64, f64)) -> ((f64, f64), f64) {
    let x = x * 2.0;
    let q = {
        let s = p.0 + p.1;
        let d = p.0 - p.1;
        (s * x, d / (1.0 + x * x))
    };
    let r = if q.0 > q.1 { (q.1, q.0) } else { q };
    (r, r.0 * r.1)
}

/// Rounding and more functions.
#[kernel]
pub fn real_api(x: f64, y: f64) -> (f64, f64, f64, f64) {
    let a = x.round_ties_even() - x.fract() + (x * 3.0).trunc();
    let b = (x.abs() + 1.0).log10() + (x.abs() + 1.0).log2();
    let c = (x * 0.4).atanh() + (y.abs() + 1.0).acosh() + (x * 0.2).ln_1p() + x.exp_m1();
    let d = x.exp2() + signumless(y);
    (a, b, c, d)
}

#[kernel(inline)]
pub fn signumless(y: f64) -> f64 {
    y * 0.5
}

// examples/claude/pi_hypersphere.rs: recursion, unrolled loops over bits, `map` with closures

/// `bool` to 1.0 or 0.0, as `Mask::to_real` in the kernel.
pub trait ToReal {
    fn to_real(self) -> f64;
}

impl ToReal for bool {
    fn to_real(self) -> f64 {
        if self {
            1.0
        } else {
            0.0
        }
    }
}

const DIM: usize = 4;
const CORNERS: usize = 1 << DIM;

/// The volume of the part of the unit ball in the box `[lo, hi]` (corners
/// `(lo1.re, lo1.im, lo2.re, lo2.im)` and the same for `hi`), down to boxes of side
/// `min_side`.
#[kernel]
pub fn orthant(
    lo1: Complex<f64>,
    lo2: Complex<f64>,
    hi1: Complex<f64>,
    hi2: Complex<f64>,
    min_side: f64,
) -> f64 {
    let lo = [lo1.re, lo1.im, lo2.re, lo2.im];
    let hi = [hi1.re, hi1.im, hi2.re, hi2.im];

    // n = the number of corners inside the unit ball (the `for` loops are unrolled)
    let mut n = 0.0;
    for c in 0..CORNERS {
        let mut r2 = 0.0;
        for k in 0..DIM {
            let x = if c >> k & 1 == 1 { hi[k] } else { lo[k] };
            r2 += x * x;
        }
        n += (r2 <= 1.0).to_real();
    }

    // the boxes are cubes (halved from the unit cube), so one side is enough
    let side = hi[0] - lo[0];
    let volume = (side * side) * (side * side);

    if n != 0.0 && n != CORNERS as f64 && side > min_side {
        // the 16 sub-boxes: [lo, mid] or [mid, hi] on every axis
        let mid = [0, 1, 2, 3].map(|k| (lo[k] + hi[k]) * 0.5);
        let mut total = 0.0;
        for c in 0..CORNERS {
            let a = [0, 1, 2, 3].map(|k| if c >> k & 1 == 1 { mid[k] } else { lo[k] });
            let b = [0, 1, 2, 3].map(|k| if c >> k & 1 == 1 { hi[k] } else { mid[k] });
            total += orthant(
                Complex::new(a[0], a[1]),
                Complex::new(a[2], a[3]),
                Complex::new(b[0], b[1]),
                Complex::new(b[2], b[3]),
                min_side,
            );
        }
        total
    } else {
        volume * (n * (1.0 / CORNERS as f64))
    }
}

// calls of kernels and callbacks (kernel_diff.rs, kernel_calls.rs)

#[kernel]
pub fn step(x: f64, c: f64) -> f64 {
    x * x * 0.5 + c
}

#[kernel(callback)]
pub fn damp(x: f64) -> f64 {
    x / (1.0 + x.abs())
}

/// A kernel loop that calls a kernel and a callback, and accumulates.
#[kernel]
pub fn iterate(c: f64, n: f64) -> (f64, f64) {
    let mut x: f64 = 0.0;
    let mut sum = 0.0;
    let mut i = 0.0;
    while i < n && x.abs() < 1e6 {
        x = damp(step(x, c)) * 2.0;
        sum += x;
        i += 1.0;
    }
    (x, sum)
}

#[kernel]
pub fn complex_step(z: Complex<f64>, c: Complex<f64>) -> Complex<f64> {
    z * z * 0.5 + c
}

#[kernel]
pub fn orbit(c: Complex<f64>, limit: f64) -> (f64, Complex<f64>) {
    let mut z = Complex::new(0.0, 0.0);
    let mut n = 0.0;
    while n < limit && z.norm_sqr() <= 4.0 {
        z = complex_step(z, c);
        n += 1.0;
    }
    (n, z)
}

#[kernel]
pub fn integrand(x: f64) -> f64 {
    let s = x.sin();
    if s > 0.0 {
        s * s
    } else {
        x.exp()
    }
}

#[kernel]
pub fn midpoint_integral(a: f64, b: f64, n: f64) -> f64 {
    let h = (b - a) / n;
    let mut sum = 0.0;
    let mut i = 0.0;
    while i < n {
        sum += integrand(a + (i + 0.5) * h);
        i += 1.0;
    }
    sum * h
}

#[kernel]
pub fn pole(z: Complex<f64>, p: Complex<f64>) -> Complex<f64> {
    Complex::new(1.0, 0.0) / (z - p)
}

#[kernel]
pub fn around_circle(p: Complex<f64>, n: f64) -> Complex<f64> {
    let h = 2.0 * std::f64::consts::PI / n;
    let mut sum: Complex<f64> = Complex::new(0.0, 0.0);
    let mut k = 0.0;
    while k < n {
        let t = k * h;
        let z = Complex::new(t.cos(), t.sin());
        sum += pole(z, p) * z * Complex::new(0.0, h);
        k += 1.0;
    }
    sum
}

#[kernel]
pub fn ramp(x: f64) -> f64 {
    if x < 0.0 {
        -x * 0.5
    } else {
        x * x
    }
}

#[kernel]
pub fn ramped(x: f64, y: f64) -> f64 {
    ramp(x - y) + ramp(y) * 2.0
}

#[kernel]
pub fn square(x: f64) -> f64 {
    x * x
}

#[kernel]
pub fn norm2(x: f64, y: f64) -> f64 {
    square(x) + square(y)
}

/// An inline helper that calls a kernel: its caller registers the kernel.
#[kernel(inline)]
pub fn dist2(x0: f64, y0: f64, x1: f64, y1: f64) -> f64 {
    norm2(x1 - x0, y1 - y0)
}

/// Calls through an inline helper, of a kernel that calls a kernel, and of a recursive one.
#[kernel]
pub fn fib_path(n: f64, s: f64) -> f64 {
    let mut len = 0.0;
    let mut t = 0.0;
    while t < n {
        len += dist2(t * s, fib(t) * s, (t + 1.0) * s, fib(t + 1.0) * s).sqrt();
        t += 1.0;
    }
    len
}

/// A real kernel called from a complex one is compiled as a complex kernel.
#[kernel]
pub fn spiral(z: Complex<f64>, n: f64) -> Complex<f64> {
    let mut w = z;
    let mut i = 0.0;
    while i < n {
        w = w * Complex::new(0.0, 1.0) + norm2(w.re, w.im).sqrt();
        i += 1.0;
    }
    w
}

#[kernel(callback)]
pub fn stirling(x: f64) -> f64 {
    let x = x + 1.0;
    (2.0 * std::f64::consts::PI / x).sqrt() * (x / std::f64::consts::E).powf(x) / x
}

#[kernel(callback)]
pub fn length(x: f64, y: f64) -> f64 {
    x.hypot(y)
}

/// Callbacks of one and two arguments in a kernel loop.
#[kernel]
pub fn callbacks_in_loop(x: f64, n: f64) -> f64 {
    let mut sum = 0.0;
    let mut k = 0.0;
    while k < n {
        sum += length(stirling(k * x), k);
        k += 1.0;
    }
    sum
}

/// A callback in a complex kernel receives the real parts.
#[kernel]
pub fn callback_in_complex(z: Complex<f64>) -> Complex<f64> {
    z * length(z.re, z.im)
}

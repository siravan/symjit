//! The `#[kernel]` attribute macro (cargo feature `composer`): each kernel is compared with
//! the function it is made from, which the macro leaves as ordinary Rust.
#![cfg(feature = "composer")]
mod common;
use common::base_config;

use anyhow::Result;
use num_complex::Complex;
use symjit::kernel::kernel;
use symjit::{Applet, Application, Config};

const MIN_SIDE: f64 = 1.0 / 1024.0;

/// Without `fastmath`, which fuses multiplications and additions (e.g. on ARM64): the
/// kernels are compared with Rust bit for bit.
fn config(direct: bool) -> Config {
    let mut config = base_config();
    config.set_option("fastmath", "false").unwrap();
    config.set_direct(direct);
    config
}

fn compile(f: fn(Config) -> Result<Application>) -> Result<Vec<(bool, Applet)>> {
    [false, true]
        .into_iter()
        .map(|direct| Ok((direct, f(config(direct))?.seal()?)))
        .collect()
}

// pi as the area of the unit disk (the kernel of `pi_area.rs`)

/// 1.0 if `z` is in the closed unit disk, else 0.0.
#[kernel(inline)]
fn inside(z: Complex<f64>) -> f64 {
    if z.norm_sqr() <= 1.0 {
        1.0
    } else {
        0.0
    }
}

#[kernel(inline)]
fn mid(a: Complex<f64>, b: Complex<f64>) -> Complex<f64> {
    (a + b) * 0.5
}

/// The area of the unit disk inside the rectangle with the corners bottom-left, top-left,
/// top-right and bottom-right.
#[kernel]
fn area(bl: Complex<f64>, tl: Complex<f64>, tr: Complex<f64>, br: Complex<f64>) -> f64 {
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

fn corners(x0: f64, y0: f64, x1: f64, y1: f64) -> [Complex<f64>; 4] {
    [
        Complex::new(x0, y0),
        Complex::new(x0, y1),
        Complex::new(x1, y1),
        Complex::new(x1, y0),
    ]
}

#[test]
fn pi_from_unit_circle_area() -> Result<()> {
    let [bl, tl, tr, br] = corners(-1.0, -1.0, 1.0, 1.0);
    let pi = area(bl, tl, tr, br);
    assert!((pi - std::f64::consts::PI).abs() < 2e-4, "{pi} is not pi");

    for (direct, app) in compile(area::compile)? {
        for c in [
            corners(-1.0, -1.0, 1.0, 1.0),
            corners(-0.5, -0.5, 0.5, 0.25),
            corners(1.0, 1.0, 2.0, 3.0),
            corners(0.0, 0.0, 1.0, 1.0),
        ] {
            let expected = Complex::new(area(c[0], c[1], c[2], c[3]), 0.0);
            assert_eq!(
                app.evaluate_single(&c),
                expected,
                "{c:?}, direct = {direct}"
            );
        }
    }
    Ok(())
}

// recursion in a real kernel

#[kernel]
fn fib(n: f64) -> f64 {
    if n < 2.0 {
        n
    } else {
        fib(n - 1.0) + fib(n - 2.0)
    }
}

#[test]
fn recursive_real_kernel() -> Result<()> {
    for (direct, app) in compile(fib::compile)? {
        for n in [0.0, 1.0, 2.0, 10.0, 20.0] {
            assert_eq!(
                app.evaluate_single(&[n]),
                fib(n),
                "fib({n}), direct = {direct}"
            );
        }
    }
    Ok(())
}

// build-time Rust: loops, arrays, integer helpers' parameters, typed accumulators

/// Horner's rule; the loop runs while the kernel is built.
#[kernel(inline)]
fn horner(x: f64, coefs: [f64; 4]) -> f64 {
    let mut acc: f64 = 0.0;
    for i in (0..4).rev() {
        acc = acc * x + coefs[i];
    }
    acc
}

/// `x^n` by repeated multiplication, `n` known when the kernel is built (with `*=` and
/// `p = p * x`, both forms of assignment).
#[allow(clippy::assign_op_pattern)]
#[kernel(inline)]
fn power(x: f64, n: i32) -> f64 {
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
fn several(x: f64, y: f64) -> (f64, f64, f64, f64) {
    let largest: f64 = if x > y { x } else { y };
    let sign = if x < 0.0 {
        -1.0
    } else if x > 0.0 {
        1.0
    } else {
        0.0
    };
    // arrays mixing numbers with kernel values, and with numbers known while building
    let half = 0.5;
    let p = horner(x, [1.0, -2.0, half, 0.75]) + horner(y, [x, 1.5, -0.25, y]);
    let scale = |v: f64| v * 0.25 + y;
    (largest, sign, p, scale(power(x, 5)))
}

#[test]
fn several_outputs_and_build_time_code() -> Result<()> {
    for (direct, app) in compile(several::compile)? {
        for (x, y) in [(1.5, -0.25), (-2.0, 3.0), (0.0, 0.0), (0.3, 0.7)] {
            let mut outs = [0.0; 4];
            app.evaluate(&[x, y], &mut outs);
            let (a, b, c, d) = several(x, y);
            assert_eq!(outs, [a, b, c, d], "({x}, {y}), direct = {direct}");
        }
    }
    Ok(())
}

// complex values

#[kernel]
fn rotate(z: Complex<f64>, t: f64) -> Complex<f64> {
    let r = Complex::new(t.cos(), t.sin());
    let w = z * r;
    if w.re < 0.0 && t != 0.0 {
        w.conj()
    } else {
        w + Complex::new(0.0, 1.0)
    }
}

#[test]
fn complex_kernel() -> Result<()> {
    for (direct, app) in compile(rotate::compile)? {
        for (z, t) in [
            (Complex::new(1.0, 2.0), 0.5),
            (Complex::new(-1.0, 0.25), 2.0),
            (Complex::new(3.0, -1.0), 0.0),
        ] {
            let got = app.evaluate_single(&[z, Complex::new(t, 0.0)]);
            let expected = rotate(z, t);
            // the kernel's sin and cos are symjit's own: within a few ulps
            assert!(
                (got - expected).norm() <= 8.0 * f64::EPSILON * expected.norm(),
                "rotate({z}, {t}) = {got} != {expected}, direct = {direct}"
            );
        }
    }
    Ok(())
}

// assignments in kernel branches, and kernel loops (`while`)

/// Counts the points inside the unit disk, and the index of the last one: assignments in a
/// kernel branch, inside a loop that runs while the kernel is built.
#[kernel]
fn count_inside(a: Complex<f64>, b: Complex<f64>, c: Complex<f64>) -> (f64, f64) {
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

#[test]
fn assignments_in_kernel_branches() -> Result<()> {
    let points = [
        Complex::new(0.5, 0.5),
        Complex::new(2.0, 0.0),
        Complex::new(-0.25, 0.9),
        Complex::new(1.5, -1.5),
    ];
    for (direct, app) in compile(count_inside::compile)? {
        for a in points {
            for b in points {
                for c in points {
                    let mut outs = [Complex::default(); 2];
                    app.evaluate(&[a, b, c], &mut outs);
                    let (n, last) = count_inside(a, b, c);
                    let expected = [Complex::new(n, 0.0), Complex::new(last, 0.0)];
                    assert_eq!(outs, expected, "({a}, {b}, {c}), direct = {direct}");
                }
            }
        }
    }
    Ok(())
}

/// Newton's method for the square root, until it converges: a kernel loop.
#[kernel]
fn newton_sqrt(a: f64) -> (f64, f64) {
    let mut x = if a > 1.0 { a } else { 1.0 };
    let mut steps = 0.0;
    while (x * x - a).abs() > 1e-12 * a {
        x = 0.5 * (x + a / x);
        steps += 1.0;
    }
    (x, steps)
}

/// The escape time of the Mandelbrot iteration (at most 100 steps).
#[kernel]
fn mandelbrot(c: Complex<f64>) -> f64 {
    let mut z: Complex<f64> = Complex::new(0.0, 0.0);
    let mut n = 0.0;
    while n < 100.0 && z.norm_sqr() <= 4.0 {
        z = z * z + c;
        n += 1.0;
    }
    n
}

/// The escape time again, with an accumulator that needs no type: `Complex::new` of float
/// literals in a `let mut` makes a kernel value.
#[kernel]
fn mandelbrot_untyped(c: Complex<f64>) -> f64 {
    let mut z = Complex::new(0.0, 0.0);
    let mut n = 0.0;
    while n < 100.0 && z.norm_sqr() <= 4.0 {
        z = z * z + c;
        n += 1.0;
    }
    n
}

/// Fibonacci numbers by iteration: the loop state is updated by a tuple assignment.
#[kernel]
fn fib_iter(n: f64) -> f64 {
    let (mut a, mut b): (f64, f64) = (0.0, 1.0);
    let mut i = 0.0;
    while i < n {
        (a, b) = (b, a + b);
        i += 1.0;
    }
    a
}

/// The number of Collatz steps from `n` to 1: a kernel branch inside a kernel loop.
#[kernel]
fn collatz(n: f64) -> f64 {
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

/// The kernels with loops are compiled with the direct translator (`compile` sets it),
/// whatever the configuration says.
#[test]
fn kernel_loops() -> Result<()> {
    for (direct, app) in compile(newton_sqrt::compile)? {
        for a in [2.0, 0.25, 1e6, 7.0, 1.0] {
            let mut outs = [0.0; 2];
            app.evaluate(&[a], &mut outs);
            let (x, steps) = newton_sqrt(a);
            assert_eq!(outs, [x, steps], "newton_sqrt({a}), direct = {direct}");
        }
    }

    for (direct, app) in compile(mandelbrot::compile)? {
        for c in [
            Complex::new(0.0, 0.0),
            Complex::new(-1.0, 0.0),
            Complex::new(0.3, 0.5),
            Complex::new(-0.75, 0.1),
            Complex::new(1.0, 1.0),
            Complex::new(-2.0, 0.0),
        ] {
            let n = app.evaluate_single(&[c]);
            assert_eq!(
                n,
                Complex::new(mandelbrot(c), 0.0),
                "mandelbrot({c}), direct = {direct}"
            );
        }
    }

    for (direct, app) in compile(mandelbrot_untyped::compile)? {
        for c in [
            Complex::new(-1.0, 0.0),
            Complex::new(0.3, 0.5),
            Complex::new(1.0, 1.0),
        ] {
            assert_eq!(
                app.evaluate_single(&[c]),
                Complex::new(mandelbrot_untyped(c), 0.0),
                "mandelbrot_untyped({c}), direct = {direct}"
            );
        }
    }

    for (direct, app) in compile(fib_iter::compile)? {
        for n in [0.0, 1.0, 2.0, 30.0, 70.0] {
            assert_eq!(
                app.evaluate_single(&[n]),
                fib_iter(n),
                "fib_iter({n}), direct = {direct}"
            );
        }
    }

    for (direct, app) in compile(collatz::compile)? {
        for n in [1.0, 2.0, 6.0, 27.0, 97.0] {
            assert_eq!(
                app.evaluate_single(&[n]),
                collatz(n),
                "collatz({n}), direct = {direct}"
            );
        }
    }
    Ok(())
}

/// Rows whose loops run different numbers of iterations, evaluated together (with and
/// without SIMD).
#[test]
fn kernel_loops_in_matrices() -> Result<()> {
    let ns: Vec<f64> = (1..=37).map(|i| i as f64).collect();
    let expected: Vec<f64> = ns.iter().map(|&n| collatz(n)).collect();
    for simd in [false, true] {
        let mut config = base_config();
        config.set_simd(simd);
        let app = collatz::compile(config)?.seal()?;
        let mut outs = vec![0.0; ns.len()];
        app.evaluate_matrix(&ns, &mut outs, ns.len());
        assert_eq!(outs, expected, "simd = {simd}");
    }
    Ok(())
}

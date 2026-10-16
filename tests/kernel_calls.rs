//! Calls between `#[kernel]` kernels and of Rust callbacks (cargo feature `composer`): each
//! kernel is compared with the function it is made from, which the macro leaves as ordinary
//! Rust.
#![cfg(feature = "composer")]
mod common;
use common::base_config;

use anyhow::Result;
use num_complex::Complex;
use symjit::kernel::{kernel, KernelBuilder, Real};
use symjit::{Applet, Application, Compiler, Config, Defuns, Expr};
use symjit::prelude::*;

/// `a == b`, or both NaN.
fn same(a: f64, b: f64) -> bool {
    a == b || (a.is_nan() && b.is_nan())
}

/// No SIMD, SIMD of the native width, and AVX-512 where the processor has it.
const SIMD: [&str; 3] = ["off", "on", "512"];

fn config(direct: bool, simd: &str) -> Config {
    let mut config = base_config();
    config.set_direct(direct);
    config.set_simd(simd != "off");
    if simd == "512" {
        config.set_option("enable_simd512", "true").unwrap();
    }
    config
}

/// The kernel compiled with the direct and indirect translators, in every SIMD mode.
fn variants(f: fn(Config) -> Result<Application>) -> Result<Vec<(String, Applet)>> {
    let mut apps = Vec::new();
    for direct in [false, true] {
        for simd in SIMD {
            let app = f(config(direct, simd))?.seal()?;
            apps.push((format!("direct = {direct}, simd = {simd}"), app));
        }
    }
    Ok(apps)
}

// a kernel loop that calls another kernel and accumulates its values

#[kernel]
fn integrand(x: f64) -> f64 {
    let s = x.sin();
    if s > 0.0 {
        s * s
    } else {
        x.exp()
    }
}

/// The integral of `integrand` over [a, b] by the midpoint rule with `n` points.
#[kernel]
fn midpoint(a: f64, b: f64, n: f64) -> f64 {
    let h = (b - a) / n;
    let mut sum = 0.0;
    let mut i = 0.0;
    while i < n {
        sum += integrand(a + (i + 0.5) * h);
        i += 1.0;
    }
    sum * h
}

const RANGES: [[f64; 3]; 5] = [
    [0.0, 1.0, 10.0],
    [-2.0, 3.0, 100.0],
    [1.0, 1.0, 5.0],
    [0.5, 4.0, 1.0],
    // no points: 0 * (1 / 0)
    [0.0, 1.0, 0.0],
];

#[test]
fn kernel_loop_calls_a_kernel() -> Result<()> {
    for (opts, app) in variants(midpoint::compile)? {
        for r in RANGES {
            let (got, expected) = (app.evaluate_single(&r), midpoint(r[0], r[1], r[2]));
            assert!(
                same(got, expected),
                "midpoint{r:?} = {got} != {expected}, {opts}"
            );
        }
    }
    Ok(())
}

/// Rows with different numbers of iterations, evaluated together.
#[test]
fn kernel_loop_calls_a_kernel_in_matrices() -> Result<()> {
    let rows: Vec<[f64; 3]> = (0..37)
        .map(|i| [-1.0 + 0.1 * i as f64, 2.0, (i % 11) as f64 * 3.0])
        .collect();
    let args: Vec<f64> = rows.iter().flatten().copied().collect();
    let expected: Vec<f64> = rows.iter().map(|r| midpoint(r[0], r[1], r[2])).collect();
    for (opts, app) in variants(midpoint::compile)? {
        let mut outs = vec![0.0; rows.len()];
        app.evaluate_matrix(&args, &mut outs, rows.len());
        for (i, (&got, &expected)) in outs.iter().zip(expected.iter()).enumerate() {
            assert!(same(got, expected), "row {i}: {got} != {expected}, {opts}");
        }
    }
    Ok(())
}

// the same in a complex kernel: a contour integral

/// `1 / (z - p)`.
#[kernel]
fn pole(z: Complex<f64>, p: Complex<f64>) -> Complex<f64> {
    Complex::new(1.0, 0.0) / (z - p)
}

/// The integral of `1 / (z - p)` around the unit circle with `n` points: `2 pi i` if `p` is
/// inside it, 0 if it is outside.
#[kernel]
fn around_circle(p: Complex<f64>, n: f64) -> Complex<f64> {
    let h = 2.0 * std::f64::consts::PI / n;
    let mut sum: Complex<f64> = Complex::new(0.0, 0.0);
    let mut k = 0.0;
    while k < n {
        let t = k * h;
        let z = Complex::new(t.cos(), t.sin());
        // dz = i z dt
        sum += pole(z, p) * z * Complex::new(0.0, h);
        k += 1.0;
    }
    sum
}

#[test]
fn complex_kernel_loop_calls_a_kernel() -> Result<()> {
    let two_pi_i = Complex::new(0.0, 2.0 * std::f64::consts::PI);
    for (opts, app) in variants(around_circle::compile)? {
        for (p, inside) in [
            (Complex::new(0.0, 0.0), true),
            (Complex::new(0.3, -0.4), true),
            (Complex::new(2.0, 1.0), false),
            (Complex::new(-1.5, 0.0), false),
        ] {
            let got = app.evaluate_single(&[p, Complex::new(64.0, 0.0)]);
            let expected = around_circle(p, 64.0);
            // symjit's sin, cos and complex division are within a few ulps of Rust's
            assert!(
                (got - expected).norm() <= 1e-13,
                "around_circle({p}) = {got} != {expected}, {opts}"
            );
            // the error of the trapezoidal rule falls as (1 / |p|)^n for |p| > 1
            let exact = if inside {
                two_pi_i
            } else {
                Complex::new(0.0, 0.0)
            };
            assert!((got - exact).norm() < 1e-10, "{got} != {exact}, {opts}");
        }
    }
    Ok(())
}

// SIMD kernels whose lanes take different branches in a callee

#[kernel]
fn ramp(x: f64) -> f64 {
    if x < 0.0 {
        -x * 0.5
    } else {
        x * x
    }
}

/// No branch of its own: its SIMD kernel runs until `ramp` finds its lanes differ, and then
/// the rows are computed by the scalar kernel.
#[kernel]
fn ramped(x: f64, y: f64) -> f64 {
    ramp(x - y) + ramp(y) * 2.0
}

#[kernel]
fn countdown(x: f64) -> f64 {
    if x <= 0.0 {
        x
    } else {
        countdown(x - 1.0) + 1.0
    }
}

#[test]
fn diverging_lanes_in_callees() -> Result<()> {
    // rows in groups of 8 (any SIMD width): some whose lanes agree at every branch, some not
    let rows: Vec<[f64; 2]> = (0..64)
        .map(|i| {
            let x = if (i / 8) % 2 == 0 {
                3.0 + i as f64
            } else {
                (i % 5) as f64 - 2.0
            };
            [x, 0.25 * (i % 3) as f64]
        })
        .collect();
    let args: Vec<f64> = rows.iter().flatten().copied().collect();

    let expected: Vec<f64> = rows.iter().map(|r| ramped(r[0], r[1])).collect();
    for (opts, app) in variants(ramped::compile)? {
        let mut outs = vec![0.0; rows.len()];
        app.evaluate_matrix(&args, &mut outs, rows.len());
        assert_eq!(outs, expected, "ramped, {opts}");
    }

    let xs: Vec<f64> = rows.iter().map(|r| r[0]).collect();
    let expected: Vec<f64> = xs.iter().map(|&x| countdown(x)).collect();
    for (opts, app) in variants(countdown::compile)? {
        let mut outs = vec![0.0; xs.len()];
        app.evaluate_matrix(&xs, &mut outs, xs.len());
        assert_eq!(outs, expected, "countdown, {opts}");
    }
    Ok(())
}

// chains of calls, through inline helpers, and a recursive callee

#[kernel]
fn square(x: f64) -> f64 {
    x * x
}

#[kernel]
fn norm2(x: f64, y: f64) -> f64 {
    square(x) + square(y)
}

#[kernel(inline)]
fn dist2(x0: f64, y0: f64, x1: f64, y1: f64) -> f64 {
    norm2(x1 - x0, y1 - y0)
}

#[kernel]
fn fib(n: f64) -> f64 {
    if n < 2.0 {
        n
    } else {
        fib(n - 1.0) + fib(n - 2.0)
    }
}

/// The length of the polygon through (t, fib(t)) for t = 0..n, scaled by `s`.
#[kernel]
fn fib_path(n: f64, s: f64) -> f64 {
    let mut len = 0.0;
    let mut t = 0.0;
    while t < n {
        len += dist2(t * s, fib(t) * s, (t + 1.0) * s, fib(t + 1.0) * s).sqrt();
        t += 1.0;
    }
    len
}

#[test]
fn chains_of_calls() -> Result<()> {
    for (opts, app) in variants(fib_path::compile)? {
        for (n, s) in [(0.0, 1.0), (1.0, 0.5), (10.0, 0.25), (15.0, 3.0)] {
            assert_eq!(
                app.evaluate_single(&[n, s]),
                fib_path(n, s),
                "fib_path({n}, {s}), {opts}"
            );
        }
    }
    Ok(())
}

/// A real kernel called from a complex one is compiled as a complex kernel.
#[kernel]
fn spiral(z: Complex<f64>, n: f64) -> Complex<f64> {
    let mut w = z;
    let mut i = 0.0;
    while i < n {
        w = w * Complex::new(0.0, 1.0) + norm2(w.re, w.im).sqrt();
        i += 1.0;
    }
    w
}

#[test]
fn real_kernel_called_from_a_complex_kernel() -> Result<()> {
    for (opts, app) in variants(spiral::compile)? {
        for z in [Complex::new(0.5, 0.25), Complex::new(-1.0, 2.0)] {
            let got = app.evaluate_single(&[z, Complex::new(6.0, 0.0)]);
            let expected = spiral(z, 6.0);
            assert!(
                (got - expected).norm() <= 1e-14 * expected.norm(),
                "spiral({z}) = {got} != {expected}, {opts}"
            );
        }
    }
    Ok(())
}

// Rust callbacks

#[kernel(callback)]
fn gamma(x: f64) -> f64 {
    // not a function of symjit: Stirling's approximation with two correction terms
    let x = x + 1.0;
    (2.0 * std::f64::consts::PI / x).sqrt()
        * (x / std::f64::consts::E).powf(x)
        * (1.0 + 1.0 / (12.0 * x) + 1.0 / (288.0 * x * x))
        / x
}

#[kernel(callback)]
fn hypot(x: f64, y: f64) -> f64 {
    x.hypot(y)
}

/// `sum_{k < n} hypot(gamma(k x), k)`: callbacks in a kernel loop.
#[kernel]
fn callbacks_in_loop(x: f64, n: f64) -> f64 {
    let mut sum = 0.0;
    let mut k = 0.0;
    while k < n {
        sum += hypot(gamma(k * x), k);
        k += 1.0;
    }
    sum
}

#[test]
fn callbacks() -> Result<()> {
    for (opts, app) in variants(callbacks_in_loop::compile)? {
        for (x, n) in [(0.5, 0.0), (0.5, 1.0), (0.3, 7.0), (1.25, 12.0)] {
            assert_eq!(
                app.evaluate_single(&[x, n]),
                callbacks_in_loop(x, n),
                "callbacks_in_loop({x}, {n}), {opts}"
            );
        }
    }
    Ok(())
}

/// A callback in a complex kernel receives the real parts and returns a real value.
#[kernel]
fn callback_in_complex(z: Complex<f64>) -> Complex<f64> {
    z * hypot(z.re, z.im)
}

#[test]
fn callback_in_a_complex_kernel() -> Result<()> {
    let zs: Vec<Complex<f64>> = (0..16)
        .map(|i| Complex::new(i as f64 * 0.5 - 4.0, 2.0 - i as f64 * 0.25))
        .collect();
    let expected: Vec<Complex<f64>> = zs.iter().map(|&z| callback_in_complex(z)).collect();
    for (opts, app) in variants(callback_in_complex::compile)? {
        for (&z, &e) in zs.iter().zip(expected.iter()) {
            assert_eq!(
                app.evaluate_single(&[z]),
                e,
                "callback_in_complex({z}), {opts}"
            );
        }
        // the SIMD kernel: on ARM64, the call clobbered the imaginary part of the result
        let mut outs = vec![Complex::default(); zs.len()];
        app.evaluate_matrix(&zs, &mut outs, zs.len());
        assert_eq!(outs, expected, "callback_in_complex, matrix, {opts}");
    }
    Ok(())
}

// errors (these kernels are only compiled)

#[allow(dead_code)]
#[kernel]
fn complex_callee(z: Complex<f64>) -> Complex<f64> {
    z * z
}

#[allow(dead_code)]
#[kernel]
fn real_caller(x: f64) -> f64 {
    complex_callee(Complex::new(x, 0.0)).re
}

#[allow(dead_code)]
#[kernel]
fn two_outputs(x: f64) -> (f64, f64) {
    (x, -x)
}

#[allow(dead_code)]
#[kernel]
fn calls_two_outputs(x: f64) -> f64 {
    let (a, b) = two_outputs(x);
    a + b
}

#[allow(dead_code)]
#[kernel]
fn ping(x: f64) -> f64 {
    if x > 0.0 {
        pong(x - 1.0)
    } else {
        x
    }
}

#[allow(dead_code)]
#[kernel]
fn pong(x: f64) -> f64 {
    ping(x) + 1.0
}

fn error_of(r: Result<Application>) -> String {
    match r {
        Ok(_) => "no error".to_string(),
        Err(err) => format!("{err:#}"),
    }
}

#[test]
fn call_errors() {
    let err = error_of(real_caller::compile(base_config()));
    assert!(
        err.contains("complex kernel") && err.contains("complex_callee"),
        "{err}"
    );

    let err = error_of(calls_two_outputs::compile(base_config()));
    assert!(err.contains("returns one value, not 2"), "{err}");

    let err = error_of(ping::compile(base_config()));
    assert!(err.contains("cycle") && err.contains("ping -> "), "{err}");
}

// the builder: kernels and callbacks of a configuration's `Defuns`

extern "C" fn twice(x: f64) -> f64 {
    2.0 * x
}

#[test]
fn builder_calls() -> Result<()> {
    let callee = {
        let k = KernelBuilder::new(base_config());
        let (x, y): (Real, Real) = k.params();
        k.compile(x * y + 1.0)?.seal()?
    };
    // without its SIMD kernel, as when its SIMD compilation fails: the SIMD kernels calling
    // it are compiled without SIMD too
    let mut cubed = {
        let k = KernelBuilder::new(base_config());
        let x: Real = k.params();
        k.compile(x * x * x)?.seal()?
    };
    cubed.compiled_simd = None;
    // an applet made from an expression: its input is a state
    let t = Expr::var("t");
    let from_expr = Compiler::with_config(base_config())
        .compile(std::slice::from_ref(&t), &[&t * &t])?
        .seal()?;

    let mut df = Defuns::new();
    df.add_applet("f", callee);
    df.add_applet("cubed", cubed);
    df.add_applet("from_expr", from_expr);
    symjit::kernel::add_callback(&mut df, "twice", twice);
    let mut config = base_config();
    config.set_defuns(df);

    let k = KernelBuilder::new(config.clone());
    let (x, y): (Real, Real) = k.params();
    let fxy: Real = k.call("f", (x, y));
    let c: Real = k.call("cubed", (y,));
    let app = k.compile(k.callback("twice", fxy) + c)?.seal()?;
    assert_eq!(app.evaluate_single(&[3.0, 4.0]), 26.0 + 64.0);
    let args: Vec<f64> = (0..10).flat_map(|i| [i as f64, 2.0]).collect();
    let mut outs = vec![0.0; 10];
    app.evaluate_matrix(&args, &mut outs, 10);
    let expected: Vec<f64> = (0..10)
        .map(|i| 2.0 * (2.0 * i as f64 + 1.0) + 8.0)
        .collect();
    assert_eq!(outs, expected);

    // misuse
    let k = KernelBuilder::new(config.clone());
    let x: Real = k.params();
    let _: Real = k.call("g", (x,));
    assert!(error_of(k.compile(x)).contains("`g` is not a kernel"));

    let k = KernelBuilder::new(config.clone());
    let x: Real = k.params();
    let _: Real = k.call("f", (x,));
    assert!(error_of(k.compile(x)).contains("takes 2 arguments, not 1"));

    let k = KernelBuilder::new(config.clone());
    let x: Real = k.params();
    let _: Real = k.call("from_expr", (x,));
    assert!(error_of(k.compile(x)).contains("whose inputs are states"));

    let k = KernelBuilder::new(config);
    let x: Real = k.params();
    let _ = k.callback("twice", (x, x));
    assert!(error_of(k.compile(x)).contains("takes 1 arguments, not 2"));
    Ok(())
}

/// Kernels with calls compiled with a configuration that has functions of its own: the
/// functions are merged (`kernel::add_defuns`).
#[test]
fn calls_with_defuns_of_the_configuration() -> Result<()> {
    let with_defuns = || {
        let mut df = Defuns::new();
        symjit::kernel::add_callback(&mut df, "twice", twice);
        let mut config = base_config();
        config.set_defuns(df);
        config
    };

    let app = midpoint::compile(with_defuns())?.seal()?;
    assert_eq!(app.evaluate_single(&RANGES[1]), midpoint(-2.0, 3.0, 100.0));

    // the kernel `midpoint` added to them: both can be called
    let mut config = with_defuns();
    let mut df = Defuns::new();
    df.add_applet("midpoint", midpoint::compile(config.clone())?.seal()?);
    symjit::kernel::add_defuns(&mut config, df);

    let k = KernelBuilder::new(config);
    let (a, b): (Real, Real) = k.params();
    let m: Real = k.call("midpoint", (a, b, k.constant(10.0)));
    let app = k.compile(k.callback("twice", m))?.seal()?;
    assert_eq!(
        app.evaluate_single(&[0.0, 1.0]),
        2.0 * midpoint(0.0, 1.0, 10.0)
    );
    Ok(())
}

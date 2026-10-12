//! Bessel's equation solved by a Runge-Kutta kernel (cargo feature `composer`).
//!
//! ```text
//! cargo run --release --features composer --example rk4_bessel
//! ```
//!
//! Two `#[kernel]`s: `bessel` is the right-hand side of Bessel's equation of order nu,
//!
//! ```text
//! x^2 y'' + x y' + (x^2 - nu^2) y = 0,
//! ```
//!
//! and `rk4` integrates it with the classical fourth-order Runge-Kutta method, in a kernel
//! loop that calls `bessel` four times per step. Starting from J_nu(x0) and J_nu'(x0), the
//! solution is the Bessel function J_nu, which the example computes independently from
//! Bessel's integral
//!
//! ```text
//! J_n(x) = 1/pi * integral_0^pi cos(n t - x sin t) dt
//! ```
//!
//! (the trapezoidal rule converges exponentially for this periodic integrand). It checks
//! that
//!
//! 1. the kernel agrees with `rk4` run as plain Rust (the macro keeps the function),
//! 2. one point at a time (`evaluate_single`) and the SIMD kernel (`evaluate_matrix`) agree,
//! 3. the solution matches J_nu and J_nu' to the accuracy of RK4, and
//! 4. the error falls 16-fold when the number of steps doubles (fourth order).
//!
//! A called kernel returns one value, so the state (y, y') travels as one complex number,
//! y + i y'. Runge-Kutta is linear in the state, so complex arithmetic does exactly the
//! work on both components; `bessel` takes them apart with `.re` and `.im`. A complex
//! kernel holds its real values as complex numbers too, and divides with complex division,
//! which is within an ulp or two of the real quotient.

use std::f64::consts::PI;
use std::time::Instant;

use anyhow::{ensure, Result};
use num_complex::Complex;
use symjit::kernel::kernel;
use symjit::Config;

/// The right-hand side of Bessel's equation of order `nu` for the state `z = y + i y'`:
/// `(y', y'')` with `y'' = -y'/x - (1 - nu^2/x^2) y`.
#[kernel]
fn bessel(x: f64, z: Complex<f64>, nu: f64) -> Complex<f64> {
    let (y, dy) = (z.re, z.im);
    let ddy = -dy / x - (1.0 - nu * nu / (x * x)) * y;
    Complex::new(dy, ddy)
}

/// `steps` steps of the classical Runge-Kutta method for `bessel` from `(x0, z0)` to `x1`.
#[kernel]
fn rk4(x0: f64, z0: Complex<f64>, x1: f64, steps: f64, nu: f64) -> Complex<f64> {
    let h = (x1 - x0) / steps;
    let mut x = x0;
    let mut z = z0;
    let mut i = 0.0;
    while i < steps {
        let k1 = bessel(x, z, nu);
        let k2 = bessel(x + 0.5 * h, z + k1 * (0.5 * h), nu);
        let k3 = bessel(x + 0.5 * h, z + k2 * (0.5 * h), nu);
        let k4 = bessel(x + h, z + k3 * h, nu);
        z += (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (h / 6.0);
        i += 1.0;
        // x from the step count, not by adding h up
        x = x0 + i * h;
    }
    z
}

/// `J_n(x)` and `J_n'(x)` from Bessel's integral, by the trapezoidal rule with `m` intervals.
fn bessel_j(n: f64, x: f64) -> (f64, f64) {
    let m = 512;
    let (mut j, mut dj) = (0.0, 0.0);
    for k in 0..=m {
        let t = PI * k as f64 / m as f64;
        let w = if k == 0 || k == m { 0.5 } else { 1.0 };
        let phase = n * t - x * t.sin();
        j += w * phase.cos();
        dj += w * phase.sin() * t.sin();
    }
    (j / m as f64, dj / m as f64)
}

const X0: f64 = 1.0;
const ORDERS: [f64; 5] = [0.0, 1.0, 2.0, 3.0, 4.0];
const ENDS: [f64; 6] = [2.0, 5.0, 10.0, 15.0, 20.0, 30.0];

/// One row per order and end point: `(x0, z0, x1, steps, nu)`.
fn make_rows(steps: f64) -> Vec<[Complex<f64>; 5]> {
    let mut rows = Vec::new();
    for nu in ORDERS {
        let (j, dj) = bessel_j(nu, X0);
        for x1 in ENDS {
            rows.push([X0, 0.0, x1, steps, nu].map(|v| Complex::new(v, 0.0)));
            rows.last_mut().unwrap()[1] = Complex::new(j, dj);
        }
    }
    rows
}

/// The largest errors of y and y' against J_nu, over the rows.
fn max_errors(rows: &[[Complex<f64>; 5]], outs: &[Complex<f64>]) -> (f64, f64) {
    let (mut ey, mut edy) = (0.0f64, 0.0f64);
    for (r, z) in rows.iter().zip(outs) {
        let (j, dj) = bessel_j(r[4].re, r[2].re);
        ey = ey.max((z.re - j).abs());
        edy = edy.max((z.im - dj).abs());
    }
    (ey, edy)
}

fn main() -> Result<()> {
    // the reference against tabulated values
    for (n, x, j) in [
        (0.0, 1.0, 0.765_197_686_557_966_6),
        (1.0, 1.0, 0.440_050_585_744_933_5),
        (0.0, 10.0, -0.245_935_764_451_348_3),
        (2.0, 5.0, 0.046_565_116_277_752_2),
    ] {
        let got = bessel_j(n, x).0;
        ensure!((got - j).abs() < 1e-15, "J_{n}({x}) = {got}, tabulated {j}");
    }

    let mut config = Config::default();
    // bit-for-bit comparisons with the Rust function: no fused multiply-adds
    config.set_fastmath(false);
    let t = Instant::now();
    let app = rk4::compile(config)?.seal()?;
    println!("compiled rk4 and bessel in {:.1} ms", t.elapsed().as_secs_f64() * 1e3);

    let steps = 2000.0;
    let rows = make_rows(steps);
    let args: Vec<Complex<f64>> = rows.iter().flatten().copied().collect();

    let t = Instant::now();
    let mut outs = vec![Complex::default(); rows.len()];
    app.evaluate_matrix(&args, &mut outs, rows.len());
    let t_kernel = t.elapsed();

    let t = Instant::now();
    let twin: Vec<Complex<f64>> = rows
        .iter()
        .map(|r| rk4(r[0].re, r[1], r[2].re, r[3].re, r[4].re))
        .collect();
    let t_rust = t.elapsed();

    println!(
        "{} solutions of {steps} steps: kernel {:.2} ms, Rust {:.2} ms\n",
        rows.len(),
        t_kernel.as_secs_f64() * 1e3,
        t_rust.as_secs_f64() * 1e3
    );
    println!(
        "{:>3} {:>5} {:>22} {:>22} {:>9} {:>9}",
        "nu", "x", "y (rk4)", "J_nu(x)", "err y", "err y'"
    );

    let (mut kernel_vs_rust, mut single_vs_matrix) = (0.0f64, 0.0f64);
    for ((r, z), w) in rows.iter().zip(&outs).zip(&twin) {
        let (nu, x1) = (r[4].re, r[2].re);
        let (j, dj) = bessel_j(nu, x1);
        println!(
            "{nu:>3} {x1:>5} {:>22.16} {j:>22.16} {:>9.1e} {:>9.1e}",
            z.re,
            (z.re - j).abs(),
            (z.im - dj).abs()
        );
        kernel_vs_rust = kernel_vs_rust.max((z - w).norm() / w.norm());
        let single = app.evaluate_single(r);
        single_vs_matrix = single_vs_matrix.max((z - single).norm() / single.norm());
    }

    let (ey, edy) = max_errors(&rows, &outs);
    println!("\nlargest error: y {ey:.1e}, y' {edy:.1e}");
    println!("kernel against the Rust function: {kernel_vs_rust:.1e} (relative)");
    println!("evaluate_matrix against evaluate_single: {single_vs_matrix:.1e} (relative)");
    // RK4 with h <= 0.0145: about 1e-9 at x = 30
    ensure!(ey < 1e-8 && edy < 1e-8, "the kernel does not solve Bessel's equation");
    // complex division in the kernel against real division in Rust
    ensure!(kernel_vs_rust < 1e-12, "the kernel and the Rust function differ");
    // the scalar and SIMD kernels do complex arithmetic with different instructions, which
    // can round differently in the last bit
    ensure!(single_vs_matrix < 1e-12, "the scalar and SIMD kernels differ");

    // the order of the method: the error falls 2^4-fold per halving of the step
    println!("\n{:>6} {:>9} {:>6}", "steps", "err y", "ratio");
    let mut last: Option<f64> = None;
    for steps in [100.0, 200.0, 400.0, 800.0] {
        let rows = make_rows(steps);
        let args: Vec<Complex<f64>> = rows.iter().flatten().copied().collect();
        let mut outs = vec![Complex::default(); rows.len()];
        app.evaluate_matrix(&args, &mut outs, rows.len());
        let (ey, _) = max_errors(&rows, &outs);
        match last {
            Some(prev) => {
                let ratio = prev / ey;
                println!("{steps:>6} {ey:>9.2e} {ratio:>6.1}");
                ensure!((12.0..20.0).contains(&ratio), "not fourth order");
            }
            None => println!("{steps:>6} {ey:>9.2e}"),
        }
        last = Some(ey);
    }

    println!("\nok");
    Ok(())
}

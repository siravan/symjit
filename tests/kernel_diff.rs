//! Differential tests of `#[kernel]` (cargo feature `composer`): each kernel is evaluated at
//! random points and compared with the function it is made from, which the macro leaves as
//! ordinary Rust. Every kernel runs with the direct and indirect translators, with and
//! without SIMD, one point at a time and as a matrix; real kernels also run compiled as
//! complex kernels (`__build(config, true)`), where their results must be real. `fastmath`
//! is off.
//!
//! `SYMJIT_DIFF_POINTS` sets the number of points (default 200), `SYMJIT_DIFF_SEED` the
//! seed, and `SYMJIT_DIFF_VERBOSE=1` prints the largest error of each kernel and mode.
#![cfg(feature = "composer")]
mod common;
use common::base_config;

use anyhow::Result;
use num_complex::Complex;
use symjit::kernel::kernel;
use symjit::{Applet, Application, Config};

// the kernels

/// Arithmetic and rounding: exact, except where a kernel fuses a multiply and an add.
#[kernel]
fn arith(x: f64, y: f64, z: f64) -> (f64, f64, f64, f64) {
    let a = x * y - z / (y * y + 1.0);
    let b = (x - y) * (x + y) - z.recip() * 0.5;
    let c = x.abs().sqrt() * y.min(z) - x.max(y) + z.powi(3);
    let d = (x * 3.0).floor() + (y * 2.5).ceil() - z.trunc() + (x * 7.0).round();
    (a, b, c, d)
}

/// Elementary functions: symjit's own, within a few ulps of Rust's (libm).
#[kernel]
fn elementary(x: f64, y: f64) -> (f64, f64, f64, f64, f64) {
    let a = x.exp() * y.sin() + y.cos();
    let b = (x * x + 1.0).ln() - y.atan() + x.tanh();
    let c = (x.abs() + 0.5).powf(y) + x.cbrt();
    let d = x.sinh() - (y * 0.5).cosh() + (x * 0.25).tan();
    let e = (x * 0.3).asin() + (y * 0.3).acos() + x.asinh();
    (a, b, c, d, e)
}

/// Nested branches that assign variables, and comparisons combined with `&&`/`||`.
#[kernel]
fn piecewise(x: f64, y: f64) -> (f64, f64) {
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
fn step(x: f64, c: f64) -> f64 {
    x * x * 0.5 + c
}

#[kernel(callback)]
fn damp(x: f64) -> f64 {
    x / (1.0 + x.abs())
}

/// A kernel loop that calls a kernel and a callback, and accumulates.
#[kernel]
fn iterate(c: f64, n: f64) -> (f64, f64) {
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

/// Recursion with branches.
#[kernel]
fn tree(x: f64, depth: f64) -> f64 {
    if depth <= 0.0 {
        x
    } else if x < 0.0 {
        tree(-x * 0.5, depth - 1.0) + 1.0
    } else {
        tree(x - 1.0, depth - 1.0) * 0.5 + tree(x * 0.25, depth - 2.0)
    }
}

/// Complex arithmetic and functions.
#[kernel]
fn complex_ops(z: Complex<f64>, w: Complex<f64>) -> (Complex<f64>, Complex<f64>, Complex<f64>) {
    let a = z * w + z.conj() - w * 2.0;
    let b = z / (w * w + 4.0) + Complex::new(z.norm(), w.norm_sqr());
    let c = z.exp() * w.sin() + (w * w + 1.0).ln() - z.powi(3);
    (a, b, c)
}

#[kernel]
fn complex_step(z: Complex<f64>, c: Complex<f64>) -> Complex<f64> {
    z * z * 0.5 + c
}

/// A complex kernel loop that calls a kernel. The map contracts for |c| < 0.57, so that the
/// rounding differences of complex products (e.g. fused on ARM64) do not grow, as they would
/// in the Mandelbrot iteration.
#[kernel]
fn orbit(c: Complex<f64>, limit: f64) -> (f64, Complex<f64>) {
    let mut z = Complex::new(0.0, 0.0);
    let mut n = 0.0;
    while n < limit && z.norm_sqr() <= 4.0 {
        z = complex_step(z, c);
        n += 1.0;
    }
    (n, z)
}

// the driver

/// xorshift64*: reproducible points without a dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn uniform(&mut self, (lo, hi): (f64, f64)) -> f64 {
        lo + (hi - lo) * self.next()
    }
}

fn env_or(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn verbose() -> bool {
    std::env::var("SYMJIT_DIFF_VERBOSE").is_ok_and(|v| v == "1")
}

/// The error of `got` in units of `expected` (relative, absolute below `floor`); 0 if both
/// are NaN or both the same infinity.
fn error(got: f64, expected: f64, floor: f64) -> f64 {
    if got == expected || (got.is_nan() && expected.is_nan()) {
        0.0
    } else {
        (got - expected).abs() / expected.abs().max(floor)
    }
}

/// The modes a kernel is compiled in: (complex, direct, SIMD: off, on (the native width) or
/// 512 (AVX-512, where the processor has it)).
fn modes(complex_too: bool) -> Vec<(bool, bool, &'static str)> {
    let mut modes = Vec::new();
    for complex in [false, true] {
        if complex && !complex_too {
            continue;
        }
        for direct in [false, true] {
            for simd in ["off", "on", "512"] {
                modes.push((complex, direct, simd));
            }
        }
    }
    modes
}

fn config(direct: bool, simd: &str) -> Config {
    let mut config = base_config();
    // without fused multiplications and additions, which change the rounding of a few
    // results, and the kernel loops amplify the change
    config.set_option("fastmath", "false").unwrap();
    config.set_direct(direct);
    config.set_simd(simd != "off");
    if simd == "512" {
        config.set_option("enable_simd512", "true").unwrap();
    }
    config
}

/// A real kernel: `twin` is the Rust function, `ranges` those of its arguments, `tol` the
/// largest error allowed (relative, absolute below `floor`).
struct RealCase {
    name: &'static str,
    build: fn(Config, bool) -> Result<Application>,
    twin: fn(&[f64]) -> Vec<f64>,
    ranges: &'static [(f64, f64)],
    tol: f64,
    /// the tolerance when compiled as a complex kernel (complex division)
    tol_complex: f64,
    floor: f64,
}

fn points(ranges: &[(f64, f64)], seed: u64) -> Vec<Vec<f64>> {
    let n = env_or("SYMJIT_DIFF_POINTS", 200);
    let mut rng = Rng(seed | 1);
    (0..n)
        .map(|_| ranges.iter().map(|&r| rng.uniform(r)).collect())
        .collect()
}

fn check_real(case: &RealCase) -> Result<()> {
    let seed = env_or("SYMJIT_DIFF_SEED", 0x5eed) ^ case.name.len() as u64;
    let pts = points(case.ranges, seed);
    let expected: Vec<Vec<f64>> = pts.iter().map(|p| (case.twin)(p)).collect();
    let n_out = expected[0].len();

    for (complex, direct, simd) in modes(true) {
        let mode = format!(
            "{}: complex = {complex}, direct = {direct}, simd = {simd}",
            case.name
        );
        let app: Applet = (case.build)(config(direct, simd), complex)?.seal()?;
        let tol = if complex { case.tol_complex } else { case.tol };
        let mut worst = 0.0f64;
        let mut check = |i: usize, how: &str, got: &[f64], got_im: Option<&[f64]>| {
            for (j, (&g, &e)) in got.iter().zip(expected[i].iter()).enumerate() {
                let err = error(g, e, case.floor);
                worst = worst.max(err);
                assert!(
                    err <= tol,
                    "{mode}, {how}: output {j} at {:?} is {g}, expected {e} (error {err:.3e})",
                    pts[i]
                );
                if let Some(im) = got_im {
                    assert!(
                        im[j] == 0.0 || im[j].abs() <= tol * e.abs().max(case.floor),
                        "{mode}, {how}: output {j} at {:?} has the imaginary part {}",
                        pts[i],
                        im[j]
                    );
                }
            }
        };

        if complex {
            let args: Vec<Vec<Complex<f64>>> = pts
                .iter()
                .map(|p| p.iter().map(|&x| Complex::new(x, 0.0)).collect())
                .collect();
            for (i, a) in args.iter().enumerate() {
                let mut outs = vec![Complex::default(); n_out];
                app.evaluate(a, &mut outs);
                let (re, im): (Vec<f64>, Vec<f64>) = outs.iter().map(|z| (z.re, z.im)).unzip();
                check(i, "single", &re, Some(&im));
            }
            let flat: Vec<Complex<f64>> = args.concat();
            let mut outs = vec![Complex::default(); n_out * pts.len()];
            app.evaluate_matrix(&flat, &mut outs, pts.len());
            for i in 0..pts.len() {
                let row = &outs[i * n_out..(i + 1) * n_out];
                let (re, im): (Vec<f64>, Vec<f64>) = row.iter().map(|z| (z.re, z.im)).unzip();
                check(i, "matrix", &re, Some(&im));
            }
        } else {
            for (i, p) in pts.iter().enumerate() {
                let mut outs = vec![0.0; n_out];
                app.evaluate(p, &mut outs);
                check(i, "single", &outs, None);
            }
            let flat: Vec<f64> = pts.concat();
            let mut outs = vec![0.0; n_out * pts.len()];
            app.evaluate_matrix(&flat, &mut outs, pts.len());
            for i in 0..pts.len() {
                check(i, "matrix", &outs[i * n_out..(i + 1) * n_out], None);
            }
        }

        if verbose() {
            println!("{mode}: largest error {worst:.3e}");
        }
    }
    Ok(())
}

/// The range of a real number, or of a real or imaginary part.
type Range = (f64, f64);

/// A complex kernel; its real outputs are compared as complex numbers.
struct ComplexCase {
    name: &'static str,
    build: fn(Config, bool) -> Result<Application>,
    twin: fn(&[Complex<f64>]) -> Vec<Complex<f64>>,
    /// the ranges of the real and imaginary parts of the arguments
    ranges: &'static [(Range, Range)],
    tol: f64,
    floor: f64,
}

fn check_complex(case: &ComplexCase) -> Result<()> {
    let seed = env_or("SYMJIT_DIFF_SEED", 0x5eed) ^ case.name.len() as u64;
    let n = env_or("SYMJIT_DIFF_POINTS", 200);
    let mut rng = Rng(seed | 1);
    let pts: Vec<Vec<Complex<f64>>> = (0..n)
        .map(|_| {
            case.ranges
                .iter()
                .map(|&(re, im)| Complex::new(rng.uniform(re), rng.uniform(im)))
                .collect()
        })
        .collect();
    let expected: Vec<Vec<Complex<f64>>> = pts.iter().map(|p| (case.twin)(p)).collect();
    let n_out = expected[0].len();

    for (_, direct, simd) in modes(false) {
        let mode = format!("{}: direct = {direct}, simd = {simd}", case.name);
        let app: Applet = (case.build)(config(direct, simd), true)?.seal()?;
        let mut worst = 0.0f64;
        let mut check = |i: usize, how: &str, got: &[Complex<f64>]| {
            for (j, (&g, &e)) in got.iter().zip(expected[i].iter()).enumerate() {
                let err = if g == e {
                    0.0
                } else {
                    (g - e).norm() / e.norm().max(case.floor)
                };
                worst = worst.max(err);
                assert!(
                    err <= case.tol,
                    "{mode}, {how}: output {j} at {:?} is {g}, expected {e} (error {err:.3e})",
                    pts[i]
                );
            }
        };

        for (i, p) in pts.iter().enumerate() {
            let mut outs = vec![Complex::default(); n_out];
            app.evaluate(p, &mut outs);
            check(i, "single", &outs);
        }
        let flat: Vec<Complex<f64>> = pts.concat();
        let mut outs = vec![Complex::default(); n_out * pts.len()];
        app.evaluate_matrix(&flat, &mut outs, pts.len());
        for i in 0..pts.len() {
            check(i, "matrix", &outs[i * n_out..(i + 1) * n_out]);
        }

        if verbose() {
            println!("{mode}: largest error {worst:.3e}");
        }
    }
    Ok(())
}

/// The tolerances, from the errors measured on x86-64 and on ARM64 (qemu) with
/// `SYMJIT_DIFF_VERBOSE=1`, with some room. Real kernels give Rust's results exactly, apart
/// from the elementary functions, which are symjit's own (exact at these points on both).
/// Complex kernels compute real functions as complex ones (10 ulps in `elementary`, 2 in the
/// divisions of `arith`), and complex products round differently from `num_complex`'s
/// (fused on ARM64). Branches, loops, calls and recursion add no error of their own.
const EPS: f64 = f64::EPSILON;

#[test]
fn arithmetic() -> Result<()> {
    check_real(&RealCase {
        name: "arith",
        build: arith::__build,
        twin: |p| {
            let (a, b, c, d) = arith(p[0], p[1], p[2]);
            vec![a, b, c, d]
        },
        ranges: &[(-4.0, 4.0), (-4.0, 4.0), (0.25, 3.0)],
        tol: 0.0,
        tol_complex: 4.0 * EPS,
        floor: 1.0,
    })
}

#[test]
fn elementary_functions() -> Result<()> {
    check_real(&RealCase {
        name: "elementary",
        build: elementary::__build,
        twin: |p| {
            let (a, b, c, d, e) = elementary(p[0], p[1]);
            vec![a, b, c, d, e]
        },
        ranges: &[(-3.0, 3.0), (-3.0, 3.0)],
        tol: 4.0 * EPS,
        tol_complex: 32.0 * EPS,
        floor: 1.0,
    })
}

#[test]
fn branches() -> Result<()> {
    check_real(&RealCase {
        name: "piecewise",
        build: piecewise::__build,
        twin: |p| {
            let (v, w) = piecewise(p[0], p[1]);
            vec![v, w]
        },
        ranges: &[(-3.0, 3.0), (-3.0, 3.0)],
        tol: 0.0,
        tol_complex: 0.0,
        floor: 1.0,
    })
}

#[test]
fn loops_with_calls() -> Result<()> {
    check_real(&RealCase {
        name: "iterate",
        build: iterate::__build,
        twin: |p| {
            let (x, sum) = iterate(p[0], p[1]);
            vec![x, sum]
        },
        ranges: &[(-2.0, 1.0), (0.0, 40.0)],
        tol: 0.0,
        tol_complex: 0.0,
        floor: 1.0,
    })
}

#[test]
fn recursion() -> Result<()> {
    check_real(&RealCase {
        name: "tree",
        build: tree::__build,
        twin: |p| vec![tree(p[0], p[1])],
        ranges: &[(-8.0, 8.0), (0.0, 9.0)],
        tol: 0.0,
        tol_complex: 0.0,
        floor: 1.0,
    })
}

#[test]
fn complex_arithmetic_and_functions() -> Result<()> {
    check_complex(&ComplexCase {
        name: "complex_ops",
        build: complex_ops::__build,
        twin: |p| {
            let (a, b, c) = complex_ops(p[0], p[1]);
            vec![a, b, c]
        },
        ranges: &[((-2.0, 2.0), (-2.0, 2.0)), ((-2.0, 2.0), (-2.0, 2.0))],
        tol: 16.0 * EPS,
        floor: 1.0,
    })
}

#[test]
fn complex_loops_with_calls() -> Result<()> {
    check_complex(&ComplexCase {
        name: "orbit",
        build: orbit::__build,
        twin: |p| {
            let (n, z) = orbit(p[0], p[1].re);
            vec![Complex::new(n, 0.0), z]
        },
        ranges: &[((-0.4, 0.4), (-0.4, 0.4)), ((0.0, 60.0), (0.0, 0.0))],
        tol: 4.0 * EPS,
        floor: 1.0,
    })
}

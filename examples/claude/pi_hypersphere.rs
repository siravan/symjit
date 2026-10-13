//! Pi from the volume of the four-dimensional unit ball, with a recursive complex
//! `#[kernel]` (cargo feature `composer`). A port of `examples/claude/pi_hypersphere.py`,
//! which builds the same kernel with the Python `Composer`.
//!
//! ```text
//! cargo run --release --features composer --example pi_hypersphere            # depths 1..6
//! cargo run --release --features composer --example pi_hypersphere -- 8       # depths 1..8
//! ```
//!
//! The unit ball in four dimensions has volume pi^2 / 2; its part in the positive orthant
//! (all coordinates >= 0) has volume pi^2 / 32, so pi = sqrt(32 V).
//!
//! A point (x1, x2, x3, x4) is a pair of complex numbers (x1 + i x2, x3 + i x4). The kernel
//! takes an axis-aligned box in the positive orthant as two opposite corners, `lo` and `hi`
//! (four complex arguments), counts the corners of the box that lie in the unit ball
//! (n of 16), and returns
//!
//! - `volume * n / 16`, if n is 0 or 16, or the side is at most `min_side`: all corners
//!   inside means the box is inside (the ball is convex); none inside means the box is
//!   outside (in the positive orthant the ball is closed toward the origin, so if any point
//!   of the box were inside, so would be `lo`);
//! - otherwise, the sum of the 16 boxes obtained by halving every side (16 recursive calls).
//!
//! `min_side` is an argument, so one kernel serves every depth (the Python example compiles
//! one per depth). The kernel is checked against
//!
//! 1. the same function run as plain Rust (the macro keeps it), and
//! 2. the algorithm written breadth-first, level by level over the boxes still to be split
//!    (as the numpy reference of the Python example).
//!
//! All volumes are dyadic (sides 2^-k), so every sum is exact and the three must agree bit
//! for bit.
//!
//! The corners are counted with `(r2 <= 1.0).to_real()`: in the kernel the comparison is a
//! `Mask`, whose `to_real` gives 1.0 or 0.0 without a branch; for the plain Rust function
//! the trait `ToReal` gives `bool` the same method. Counting with `if r2 <= 1.0 { 1.0 }
//! else { 0.0 }` would also work, but makes 16 kernel branches per call, and the kernel
//! about 1.8 times slower.

use std::f64::consts::PI;
use std::time::Instant;

use anyhow::{ensure, Result};
use num_complex::Complex;
use symjit::kernel::kernel;
use symjit::Config;

/// `bool` to 1.0 or 0.0, as `Mask::to_real` in the kernel: counting without a branch.
trait ToReal {
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
/// The corners of a box: bit `k` of `c` picks `hi` (1) or `lo` (0) on axis `k`.
const CORNERS: usize = 1 << DIM;

/// The volume of the part of the unit ball in the box `[lo, hi]` (corners
/// `(lo1.re, lo1.im, lo2.re, lo2.im)` and the same for `hi`), down to boxes of side
/// `min_side`.
#[kernel]
fn orthant(
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

/// The same algorithm breadth-first: the lower corners of the boxes of each level.
fn reference(min_side: f64) -> f64 {
    let mut boxes = vec![[0.0; DIM]];
    let mut side = 1.0f64;
    let mut total = 0.0;
    while !boxes.is_empty() {
        let mut next = Vec::new();
        for lo in boxes {
            let corner = |c: usize, s: f64| -> [f64; DIM] {
                std::array::from_fn(|k| lo[k] + if c >> k & 1 == 1 { s } else { 0.0 })
            };
            let n = (0..CORNERS)
                .filter(|&c| corner(c, side).iter().map(|x| x * x).sum::<f64>() <= 1.0)
                .count();
            if n != 0 && n != CORNERS && side > min_side {
                next.extend((0..CORNERS).map(|c| corner(c, 0.5 * side)));
            } else {
                total += side.powi(DIM as i32) * n as f64 / CORNERS as f64;
            }
        }
        boxes = next;
        side *= 0.5;
    }
    total
}

fn main() -> Result<()> {
    let max_depth: i32 = match std::env::args().nth(1) {
        Some(d) => d.parse()?,
        None => 6,
    };

    let t = Instant::now();
    let app = orthant::compile(Config::default())?.seal()?;
    println!("compiled in {:.1} ms\n", t.elapsed().as_secs_f64() * 1e3);

    // the positive unit cube: lo = (0, 0, 0, 0), hi = (1, 1, 1, 1)
    let (zero, one) = (Complex::new(0.0, 0.0), Complex::new(1.0, 1.0));
    let row = |depth: i32| [zero, zero, one, one, Complex::new(2f64.powi(-depth), 0.0)];

    println!(
        "{:>5} {:>18} {:>10} {:>10} {:>10} {:>10}",
        "depth", "pi estimate", "error", "kernel", "Rust", "reference"
    );
    let mut volumes = Vec::new();
    for depth in 1..=max_depth {
        let r = row(depth);
        let min_side = r[4].re;

        let t0 = Instant::now();
        let v = app.evaluate_single(&r);
        let t1 = Instant::now();
        let twin = orthant(r[0], r[1], r[2], r[3], min_side);
        let t2 = Instant::now();
        let reference = reference(min_side);
        let t3 = Instant::now();

        ensure!(v.im == 0.0, "depth {depth}: complex volume {v}");
        ensure!(
            v.re == twin,
            "depth {depth}: kernel {} != Rust {twin}",
            v.re
        );
        ensure!(
            v.re == reference,
            "depth {depth}: kernel {} != reference {reference}",
            v.re
        );
        volumes.push(v);

        let pi = (32.0 * v.re).sqrt();
        let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
        println!(
            "{depth:>5} {pi:>18.12} {:>10.2e} {:>8.1}ms {:>8.1}ms {:>8.1}ms",
            pi - PI,
            ms(t0, t1),
            ms(t1, t2),
            ms(t2, t3)
        );
    }

    // all depths in one matrix: the rows recurse differently, so the SIMD kernel hands
    // them to the scalar kernel
    let args: Vec<Complex<f64>> = (1..=max_depth).flat_map(row).collect();
    let mut outs = vec![Complex::default(); volumes.len()];
    app.evaluate_matrix(&args, &mut outs, volumes.len());
    ensure!(
        outs == volumes,
        "evaluate_matrix differs from evaluate_single"
    );

    println!("\nok: the kernel matches the Rust function and the reference at every depth");
    Ok(())
}

//! The typed kernel builder (`symjit::kernel`, cargo feature `composer`): the kernel of
//! `pi_area.rs` written with it, and its operations checked against Rust's f64/Complex.
#![cfg(feature = "composer")]
// `!(x < y)` is the reference of `!x.lt(y)` (true for NaN)
#![allow(clippy::neg_cmp_op_on_partial_ord, clippy::type_complexity)]
mod common;
use common::base_config;

use anyhow::Result;
use num_complex::Complex;
use symjit::kernel::{Cplx, KernelBuilder, Real};
use symjit::{Applet, Config};
use symjit::prelude::*;

const MIN_SIDE: f64 = 1.0 / 1024.0;

fn config(complex: bool, direct: bool) -> Config {
    let mut config = base_config();
    config.set_complex(complex);
    config.set_direct(direct);
    config
}

/// The same algorithm as `pi_area.rs` on f64, as the reference.
fn area_reference(x0: f64, y0: f64, x1: f64, y1: f64) -> f64 {
    let inside = |x: f64, y: f64| (x * x + y * y <= 1.0) as u32 as f64;
    let n = inside(x0, y0) + inside(x0, y1) + inside(x1, y1) + inside(x1, y0);
    let (w, h) = (x1 - x0, y1 - y0);
    let crosses = x0 * x1 < 0.0 || y0 * y1 < 0.0;
    let recurse = n != 4.0 && (n != 0.0 || crosses) && w > MIN_SIDE && h > MIN_SIDE;

    if recurse {
        let (xm, ym) = (0.5 * (x0 + x1), 0.5 * (y0 + y1));
        area_reference(x0, y0, xm, ym)
            + area_reference(x0, ym, xm, y1)
            + area_reference(xm, ym, x1, y1)
            + area_reference(xm, y0, x1, ym)
    } else {
        w * h * n * 0.25
    }
}

/// The kernel of `pi_area.rs`: the area of the unit disk inside the rectangle with corners
/// bottom-left, top-left, top-right, bottom-right.
fn compile_area_kernel(direct: bool) -> Result<Applet> {
    let k = KernelBuilder::new(config(true, direct));
    let corners: [Cplx; 4] = k.params();
    let [bl, tl, tr, br] = corners;

    // helpers are functions, not closures: a closure `|z: Cplx| ...` does not tie the
    // lifetimes of its argument and result together
    fn inside<'k>(z: Cplx<'k>) -> Real<'k> {
        z.norm_sqr().le(1.0).to_real()
    }
    fn mid<'k>(a: Cplx<'k>, b: Cplx<'k>) -> Cplx<'k> {
        (a + b) * 0.5
    }

    let n = inside(bl) + inside(tl) + inside(tr) + inside(br);
    let (x0, y0, x1, y1) = (bl.re(), bl.im(), tr.re(), tr.im());
    let (w, h) = (x1 - x0, y1 - y0);
    let crosses = (x0 * x1).lt(0.0) | (y0 * y1).lt(0.0);
    let recurse = n.ne(4.0) & (n.ne(0.0) | crosses) & w.gt(MIN_SIDE) & h.gt(MIN_SIDE);

    let area = k.if_else(
        recurse,
        || {
            let (left, top, right, bottom) = (mid(bl, tl), mid(tl, tr), mid(tr, br), mid(br, bl));
            let center = mid(bl, tr);
            let a: Real = k.recurse([bl, left, center, bottom]);
            let b: Real = k.recurse([left, tl, top, center]);
            let c: Real = k.recurse([center, top, tr, right]);
            let d: Real = k.recurse([bottom, center, right, br]);
            a + b + c + d
        },
        || w * h * n * 0.25,
    );

    k.compile(area)?.seal()
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
    // all the areas are dyadic, so the kernel reproduces the reference exactly
    let reference = area_reference(-1.0, -1.0, 1.0, 1.0);
    assert!(
        (reference - std::f64::consts::PI).abs() < 2e-4,
        "{reference} is not pi"
    );

    for direct in [false, true] {
        let app = compile_area_kernel(direct)?;
        let pi = app.evaluate_single(&corners(-1.0, -1.0, 1.0, 1.0));
        assert_eq!(pi, Complex::new(reference, 0.0), "direct = {direct}");

        for (x0, y0, x1, y1) in [
            (-0.5, -0.5, 0.5, 0.25),
            (1.0, 1.0, 2.0, 3.0),
            (0.0, 0.0, 1.0, 1.0),
        ] {
            let v = app.evaluate_single(&corners(x0, y0, x1, y1));
            let expected = Complex::new(area_reference(x0, y0, x1, y1), 0.0);
            assert_eq!(v, expected, "({x0}, {y0}, {x1}, {y1}), direct = {direct}");
        }
    }
    Ok(())
}

const XS: [f64; 8] = [0.3, -1.7, 2.5, -0.5, 1e-3, -0.0, 7.25, 0.49999999999999994];

/// Compiles `f` over two real parameters for real and complex kernels and both translators,
/// and compares every output with `reference` bit for bit (but +0 == -0 in complex kernels), or, in complex kernels, within
/// `complex_ulps` if it is not 0 (and only for finite results).
fn check_real<const N: usize>(
    name: &str,
    f: impl for<'k> Fn(&'k KernelBuilder, Real<'k>, Real<'k>) -> [Real<'k>; N],
    reference: impl Fn(f64, f64) -> [f64; N],
    complex_ulps: f64,
) -> Result<()> {
    for complex in [false, true] {
        for direct in [false, true] {
            let k = KernelBuilder::new(config(complex, direct));
            let (x, y) = k.params();
            let app = k.compile(f(&k, x, y))?.seal()?;

            for &a in XS.iter() {
                for &b in XS.iter() {
                    let expected = reference(a, b);
                    let got: Vec<f64> = if complex {
                        let mut outs = [Complex::default(); N];
                        app.evaluate(&[Complex::new(a, 0.0), Complex::new(b, 0.0)], &mut outs);
                        outs.iter().map(|z| z.re).collect()
                    } else {
                        let mut outs = [0.0; N];
                        app.evaluate(&[a, b], &mut outs);
                        outs.to_vec()
                    };
                    for i in 0..N {
                        if complex && complex_ulps > 0.0 {
                            if expected[i].is_finite() {
                                let ulps = (got[i] - expected[i]).abs()
                                    / (f64::EPSILON * expected[i].abs().max(f64::MIN_POSITIVE));
                                assert!(
                                    ulps <= complex_ulps,
                                    "{name}[{i}]({a}, {b}) = {} != {}, direct = {direct}",
                                    got[i],
                                    expected[i]
                                );
                            }
                            continue;
                        }
                        // complex arithmetic can lose the sign of a zero
                        let same_zero = complex && got[i] == 0.0 && expected[i] == 0.0;
                        assert!(
                            same_zero
                                || got[i].to_bits() == expected[i].to_bits()
                                || (got[i].is_nan() && expected[i].is_nan()),
                            "{name}[{i}]({a}, {b}) = {} != {}, complex = {complex}, direct = {direct}",
                            got[i],
                            expected[i]
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn real_arithmetic() -> Result<()> {
    check_real(
        "arith",
        |_, x, y| {
            [
                x + y,
                x - y,
                x * y,
                -x,
                2.0 * x - y / 4.0,
                1.0 - x,
                x.modulo(0.75),
            ]
        },
        |x, y| {
            [
                x + y,
                x - y,
                x * y,
                -x,
                2.0 * x - y / 4.0,
                1.0 - x,
                x - (x / 0.75).floor() * 0.75,
            ]
        },
        0.0,
    )
}

/// Real division in complex kernels is complex division, `x conj(y) / |y|^2`: within an ulp
/// or two, and NaN for a zero divisor (instead of an infinity).
#[test]
fn real_division() -> Result<()> {
    check_real(
        "divide",
        |_, x, y| [x / y, y.recip(), 1.0 / x, x / 3.0],
        |x, y| [x / y, 1.0 / y, 1.0 / x, x / 3.0],
        2.0,
    )
}

#[test]
fn real_comparisons_and_masks() -> Result<()> {
    check_real(
        "compare",
        |_, x, y| {
            [
                x.lt(y).to_real(),
                x.le(y).to_real(),
                x.gt(y).to_real(),
                x.ge(y).to_real(),
                x.eq(y).to_real(),
                x.ne(y).to_real(),
                (x.lt(1.0) & y.gt(0.0)).to_real(),
                (x.lt(1.0) | y.gt(0.0)).to_real(),
                (x.lt(1.0) ^ y.gt(0.0)).to_real(),
                (!x.lt(y)).to_real(),
                x.lt(y).select(x, y),
                x.min(y),
                x.max(2.0),
            ]
        },
        |x, y| {
            let b = |c: bool| c as u32 as f64;
            [
                b(x < y),
                b(x <= y),
                b(x > y),
                b(x >= y),
                b(x == y),
                b(x != y),
                b(x < 1.0 && y > 0.0),
                b(x < 1.0 || y > 0.0),
                b((x < 1.0) ^ (y > 0.0)),
                b(!(x < y)),
                if x < y { x } else { y },
                if x < y { x } else { y },
                if x > 2.0 { x } else { 2.0 },
            ]
        },
        0.0,
    )
}

#[test]
fn real_functions() -> Result<()> {
    // the functions that are exact (correctly rounded or exact operations)
    check_real(
        "exact",
        |_, x, y| {
            [
                x.abs(),
                x.floor(),
                x.ceil(),
                x.trunc(),
                y.round(),
                x.sqrt(),
                x.powi(2),
                x.powi(3),
                y.round_ties_even(),
            ]
        },
        |x, y| {
            [
                x.abs(),
                x.floor(),
                x.ceil(),
                x.trunc(),
                y.round(),
                x.sqrt(),
                x * x,
                x * x * x,
                y.round_ties_even(),
            ]
        },
        0.0,
    )
}

#[test]
fn real_transcendental_functions() -> Result<()> {
    // symjit's own implementations: compared within a few ulps, and real for real arguments
    // in complex kernels
    type F = for<'k> fn(Real<'k>) -> Real<'k>;
    let cases: [(&str, F, fn(f64) -> f64); 7] = [
        ("exp", |x| x.exp(), f64::exp),
        ("sin", |x| x.sin(), f64::sin),
        ("cos", |x| x.cos(), f64::cos),
        ("tanh", |x| x.tanh(), f64::tanh),
        ("ln", |x| x.abs().ln(), |x| x.abs().ln()),
        ("atan", |x| x.atan(), f64::atan),
        ("cbrt", |x| x.cbrt(), f64::cbrt),
    ];

    for (name, f, reference) in cases {
        for complex in [false, true] {
            for direct in [false, true] {
                let k = KernelBuilder::new(config(complex, direct));
                let x: Real = k.param();
                let app = k.compile(f(x))?.seal()?;
                for &a in XS.iter().filter(|a| **a != 0.0) {
                    let (got, im) = if complex {
                        let z = app.evaluate_single(&[Complex::new(a, 0.0)]);
                        (z.re, z.im)
                    } else {
                        (app.evaluate_single(&[a]), 0.0)
                    };
                    let expected = reference(a);
                    let err = (got - expected).abs() / f64::EPSILON / expected.abs().max(1e-300);
                    assert!(
                        err <= 4.0,
                        "{name}({a}) = {got} != {expected}, complex = {complex}, direct = {direct}"
                    );
                    assert_eq!(
                        im, 0.0,
                        "{name}({a}), complex = {complex}, direct = {direct}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn complex_arithmetic() -> Result<()> {
    let zs = [
        Complex::new(0.5, -1.25),
        Complex::new(-2.0, 0.75),
        Complex::new(3.0, 0.0),
    ];

    for direct in [false, true] {
        let k = KernelBuilder::new(config(true, direct));
        let (z, w): (Cplx, Cplx) = k.params();
        let x: Real = k.param();
        let i = Complex::new(0.0, 1.0);
        let outs = [
            z + w,
            z - w,
            z * w,
            -z,
            z.conj(),
            z * x + i,
            x - z,
            k.complex(z.im(), z.re()),
            z.norm_sqr().to_complex(),
            k.if_else(x.gt(0.0), || z, || w),
        ];
        let app = k.compile(outs)?.seal()?;

        for &a in zs.iter() {
            for &b in zs.iter() {
                for xr in [2.0, -0.5] {
                    let mut got = [Complex::default(); 10];
                    // the imaginary part of a real parameter is dropped
                    app.evaluate(&[a, b, Complex::new(xr, 9.0)], &mut got);
                    let expected = [
                        a + b,
                        a - b,
                        a * b,
                        -a,
                        a.conj(),
                        a * xr + i,
                        xr - a,
                        Complex::new(a.im, a.re),
                        Complex::new(a.norm_sqr(), 0.0),
                        if xr > 0.0 { a } else { b },
                    ];
                    assert_eq!(got, expected, "({a}, {b}, {xr}), direct = {direct}");
                }
            }
        }
    }
    Ok(())
}

#[test]
fn branches_return_tuples() -> Result<()> {
    for direct in [false, true] {
        let k = KernelBuilder::new(config(false, direct));
        let (x, y): (Real, Real) = k.params();
        // nested branches, each returning a pair
        let (s, d) = k.if_else(
            x.gt(y),
            || (x + y, x - y),
            || k.if_else(x.eq(y), || (x * 2.0, k.constant(0.0)), || (y - x, x * y)),
        );
        let app = k.compile((s, d))?.seal()?;

        for (a, b) in [(3.0, 1.0), (2.0, 2.0), (1.0, 5.0)] {
            let mut outs = [0.0; 2];
            app.evaluate(&[a, b], &mut outs);
            let expected = if a > b {
                [a + b, a - b]
            } else if a == b {
                [a * 2.0, 0.0]
            } else {
                [b - a, a * b]
            };
            assert_eq!(outs, expected, "({a}, {b}), direct = {direct}");
        }
    }
    Ok(())
}

/// Recursion in a real kernel: the Fibonacci numbers.
#[test]
fn recursive_real_kernel() -> Result<()> {
    for direct in [false, true] {
        let k = KernelBuilder::new(config(false, direct));
        let n: Real = k.param();
        let fib = k.if_else(
            n.lt(2.0),
            || n,
            || {
                let a: Real = k.recurse(n - 1.0);
                let b: Real = k.recurse(n - 2.0);
                a + b
            },
        );
        let app = k.compile(fib)?.seal()?;
        assert_eq!(app.evaluate_single(&[20.0]), 6765.0, "direct = {direct}");
    }
    Ok(())
}

#[test]
fn errors_are_reported_by_compile() {
    // a complex value in a real kernel
    let k = KernelBuilder::new(config(false, true));
    let x: Real = k.param();
    let z = x.to_complex();
    let err = k.compile(z.re()).err().expect("an error");
    assert!(err.to_string().contains("complex kernel"), "{err}");

    // a value of another kernel
    let k1 = KernelBuilder::new(config(false, true));
    let k2 = KernelBuilder::new(config(false, true));
    let (x1, x2): (Real, Real) = (k1.param(), k2.param());
    let err = k1.compile(x1 + x2).err().expect("an error");
    assert!(err.to_string().contains("another kernel"), "{err}");
}

/// Kernel loops keep their state in fixed temporaries, which the indirect translator does
/// not support: `while_loop` reports it.
#[test]
fn kernel_loops_need_the_direct_translator() -> Result<()> {
    for direct in [false, true] {
        let k = KernelBuilder::new(config(false, direct));
        let a: Real = k.param();
        // Newton's method for sqrt(a)
        let x = k.while_loop(
            a.max(1.0),
            |x| (x * x - a).abs().gt(1e-12 * 1e6),
            |x| 0.5 * (x + a / x),
        );
        match k.compile(x) {
            Ok(app) => {
                assert!(direct);
                let v = app.seal()?.evaluate_single(&[2.0]);
                assert!((v - std::f64::consts::SQRT_2).abs() < 1e-6, "{v}");
            }
            Err(err) => {
                assert!(!direct);
                assert!(err.to_string().contains("direct translator"), "{err}");
            }
        }
    }
    Ok(())
}

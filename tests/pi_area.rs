//! Computes pi as the area of the unit disk with a recursive kernel (`Defuns::add_self`).
//!
//! The kernel takes the four corners of an axis-aligned rectangle as complex numbers, in the
//! order bottom-left, top-left, top-right, bottom-right, and counts the corners inside the
//! unit circle (|z|^2 <= 1):
//! - all four inside: the rectangle is inside (the disk is convex), return its area;
//! - none inside, and the rectangle does not cross an axis: return 0. Inside one quadrant the
//!   disk is closed toward the origin, so if any point of the rectangle were inside, the corner
//!   nearest the origin would be too. A rectangle that crosses an axis can hold part of the disk
//!   with every corner outside (the top-level square [-1,1]^2 is one), so it is split instead;
//! - otherwise, if the sides are shorter than `MIN_SIDE`, return the area times the fraction of
//!   corners inside; else split the rectangle into four at its center and sum the four calls.
//!
//! All the non-recursive cases are `area * n / 4` (n = corners inside), so the kernel computes
//! that and only branches on whether to recurse.
mod common;
use common::base_config;

use anyhow::Result;
use num_complex::Complex;
use symjit::{Application, Composer, Defuns, Slot, Translator};
use symjit::prelude::*;

const MIN_SIDE: f64 = 1.0 / 1024.0;

/// The same algorithm on f64, as the reference for the kernel.
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

/// Hands out the temporaries of the kernel.
struct Temps(usize);

impl Temps {
    fn next(&mut self) -> Slot {
        self.0 += 1;
        Slot::Temp(self.0 - 1)
    }
}

fn compile_area_kernel(direct: bool) -> Result<Application> {
    let mut config = base_config();
    config.set_complex(true);
    config.set_direct(direct);

    let mut df = Defuns::new();
    df.add_self("area");
    config.set_defuns(df);

    let mut ev = Translator::new(config);
    ev.set_num_params(4);
    let mut t = Temps(0);

    let zero = Slot::Const(ev.append_constant(Complex::new(0.0, 0.0))?);
    let one = Slot::Const(ev.append_constant(Complex::new(1.0, 0.0))?);
    let four = Slot::Const(ev.append_constant(Complex::new(4.0, 0.0))?);
    let half = Slot::Const(ev.append_constant(Complex::new(0.5, 0.0))?);
    let quarter = Slot::Const(ev.append_constant(Complex::new(0.25, 0.0))?);
    let min_side = Slot::Const(ev.append_constant(Complex::new(MIN_SIDE, 0.0))?);

    // comparisons return bit masks; `and` with 1.0 turns them into 1.0 or 0.0
    let mut test = |ev: &mut Translator, op: &str, a: &Slot, b: &Slot| -> Result<Slot> {
        let mask = t.next();
        ev.append_fun(&mask, op, &[a.clone(), b.clone()], true)?;
        let flag = t.next();
        ev.append_fun(&flag, "and", &[mask, one.clone()], true)?;
        Ok(flag)
    };

    // `is_real = true` tells `append_fun` that the arguments are real: it takes their real
    // parts first. So `abs2`, `real` and `imaginary` of the corners pass `false`, while the
    // comparisons and `and`/`or`, whose arguments are real, pass `true`.

    // n = number of corners inside the unit circle
    let corners = [Slot::Param(0), Slot::Param(1), Slot::Param(2), Slot::Param(3)];
    let mut inside = Vec::new();
    for z in &corners {
        let r2 = Slot::Temp(1000 + inside.len());
        ev.append_fun(&r2, "abs2", &[z.clone()], false)?;
        inside.push(test(&mut ev, "leq", &r2, &one)?);
    }
    let n = Slot::Temp(1010);
    ev.append_add(&n, &inside, 0)?;

    // the sides and the area, from the bottom-left and top-right corners
    let (x0, y0, x1, y1) = (
        Slot::Temp(1020),
        Slot::Temp(1021),
        Slot::Temp(1022),
        Slot::Temp(1023),
    );
    ev.append_fun(&x0, "real", &[Slot::Param(0)], false)?;
    ev.append_fun(&y0, "imaginary", &[Slot::Param(0)], false)?;
    ev.append_fun(&x1, "real", &[Slot::Param(2)], false)?;
    ev.append_fun(&y1, "imaginary", &[Slot::Param(2)], false)?;

    let (minus_x0, minus_y0, w, h) = (
        Slot::Temp(1024),
        Slot::Temp(1025),
        Slot::Temp(1026),
        Slot::Temp(1027),
    );
    ev.append_fun(&minus_x0, "neg", &[x0.clone()], true)?;
    ev.append_fun(&minus_y0, "neg", &[y0.clone()], true)?;
    ev.append_add(&w, &[x1.clone(), minus_x0], 0)?;
    ev.append_add(&h, &[y1.clone(), minus_y0], 0)?;

    // the non-recursive result: area * n / 4
    let partial = Slot::Temp(1030);
    ev.append_mul(&partial, &[w.clone(), h.clone(), n.clone(), quarter], 0)?;

    // recurse = (n != 4) and (n != 0 or crosses an axis) and (w > MIN_SIDE) and (h > MIN_SIDE)
    let (px, py) = (Slot::Temp(1031), Slot::Temp(1032));
    ev.append_mul(&px, &[x0, x1], 0)?;
    ev.append_mul(&py, &[y0, y1], 0)?;
    let crosses_x = test(&mut ev, "lt", &px, &zero)?;
    let crosses_y = test(&mut ev, "lt", &py, &zero)?;
    let some_inside = test(&mut ev, "neq", &n, &zero)?;
    let not_all = test(&mut ev, "neq", &n, &four)?;
    let wide = test(&mut ev, "gt", &w, &min_side)?;
    let tall = test(&mut ev, "gt", &h, &min_side)?;

    let (c1, c2, c3, recurse) = (
        Slot::Temp(1040),
        Slot::Temp(1041),
        Slot::Temp(1042),
        Slot::Temp(1043),
    );
    ev.append_fun(&c1, "or", &[some_inside, crosses_x], true)?;
    ev.append_fun(&c2, "or", &[c1, crosses_y], true)?;
    ev.append_fun(&c3, "and", &[c2, not_all], true)?;
    let c4 = Slot::Temp(1044);
    ev.append_fun(&c4, "and", &[c3, wide], true)?;
    ev.append_fun(&recurse, "and", &[c4, tall], true)?;

    const ELSE: usize = 1;
    const END: usize = 2;

    ev.append_if_else(&recurse, ELSE)?;

    // the midpoints of the sides and the center
    let mid = |ev: &mut Translator, a: &Slot, b: &Slot, dst: Slot| -> Result<Slot> {
        ev.append_mul(&dst, &[a.clone(), half.clone()], 0)?;
        let b_half = Slot::Temp(match &dst {
            Slot::Temp(i) => i + 100,
            _ => unreachable!(),
        });
        ev.append_mul(&b_half, &[b.clone(), half.clone()], 0)?;
        ev.append_add(&dst, &[dst.clone(), b_half], 0)?;
        Ok(dst)
    };
    let [bl, tl, tr, br] = corners;
    let left = mid(&mut ev, &bl, &tl, Slot::Temp(2000))?;
    let top = mid(&mut ev, &tl, &tr, Slot::Temp(2001))?;
    let right = mid(&mut ev, &tr, &br, Slot::Temp(2002))?;
    let bottom = mid(&mut ev, &br, &bl, Slot::Temp(2003))?;
    let center = mid(&mut ev, &bl, &tr, Slot::Temp(2004))?;

    let quadrants = [
        [bl, left.clone(), center.clone(), bottom.clone()],
        [left, tl, top.clone(), center.clone()],
        [center.clone(), top, tr, right.clone()],
        [bottom, center, right, br],
    ];
    let mut parts = Vec::new();
    for (i, q) in quadrants.iter().enumerate() {
        let part = Slot::Temp(3000 + i);
        ev.append_fun(&part, "area", q, false)?;
        parts.push(part);
    }
    let sum = Slot::Temp(3010);
    ev.append_add(&sum, &parts, 0)?;
    ev.append_goto(END)?;

    ev.append_label(ELSE)?;
    let leaf = Slot::Temp(3011);
    ev.append_assign(&leaf, &partial)?;

    ev.append_label(END)?;
    ev.append_join(&Slot::Out(0), &recurse, &sum, &leaf)?;

    ev.compile()
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
    assert!((reference - std::f64::consts::PI).abs() < 2e-4, "{reference} is not pi");

    for direct in [false, true] {
        let app = compile_area_kernel(direct)?.seal()?;
        let pi = app.evaluate_single(&corners(-1.0, -1.0, 1.0, 1.0));
        assert_eq!(pi, Complex::new(reference, 0.0), "direct = {direct}");
    }
    Ok(())
}

#[test]
fn area_kernel_base_cases() -> Result<()> {
    for direct in [false, true] {
        let app = compile_area_kernel(direct)?.seal()?;

        // all corners inside: the area
        let v = app.evaluate_single(&corners(-0.5, -0.5, 0.5, 0.25));
        assert_eq!(v, Complex::new(0.75, 0.0), "direct = {direct}");
        // all corners outside, in one quadrant: 0
        let v = app.evaluate_single(&corners(1.0, 1.0, 2.0, 3.0));
        assert_eq!(v, Complex::new(0.0, 0.0), "direct = {direct}");
        // a quadrant: a quarter of the disk
        let v = app.evaluate_single(&corners(0.0, 0.0, 1.0, 1.0));
        assert_eq!(v, Complex::new(area_reference(0.0, 0.0, 1.0, 1.0), 0.0), "direct = {direct}");
    }
    Ok(())
}

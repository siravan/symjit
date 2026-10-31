//! Common-subexpression elimination with a conditional. `e = (p1 p2 + p3) p1` is computed in
//! the taken-if-nonzero branch of an `IfElse` on `p0` and twice after its `Join`. The first pass
//! of `Block::elimination_pass` finds `e` shared by the two statements after the join (it only
//! looks at depth 0); the second pass rewrote every occurrence, the one inside the branch too,
//! and assigned the shared temporary at its first occurrence, inside the branch. At `p0 = 0` the
//! branch does not run, and the two later statements read an unset temporary (it reads `0` here,
//! at every opt level, so `out1 = p2` and `out2 = p3`).
mod common;
use common::base_config;

use anyhow::Result;
use num_complex::Complex;
use symjit::{Application, Composer, Config, Slot, Translator};

/// out0 = if p0 != 0 { e } else { p2 }, out1 = e + p2, out2 = e + p3, with e = (p1 p2 + p3) p1.
fn compile(config: Config) -> Result<Application> {
    let mut t = Translator::new(config);
    t.set_num_params(4);
    let (p0, p1, p2, p3) = (
        Slot::Param(0),
        Slot::Param(1),
        Slot::Param(2),
        Slot::Param(3),
    );
    t.append_if_else(&p0, 0)?;
    t.append_mul(&Slot::Temp(0), &[p1, p2], 0)?;
    t.append_add(&Slot::Temp(1), &[Slot::Temp(0), p3], 0)?;
    t.append_mul(&Slot::Temp(2), &[Slot::Temp(1), p1], 0)?;
    t.append_goto(1)?;
    t.append_label(0)?;
    t.append_assign(&Slot::Temp(3), &p2)?;
    t.append_label(1)?;
    t.append_join(&Slot::Out(0), &p0, &Slot::Temp(2), &Slot::Temp(3))?;
    t.append_mul(&Slot::Temp(4), &[p1, p2], 0)?;
    t.append_add(&Slot::Temp(5), &[Slot::Temp(4), p3], 0)?;
    t.append_mul(&Slot::Temp(6), &[Slot::Temp(5), p1], 0)?;
    t.append_add(&Slot::Out(1), &[Slot::Temp(6), p2], 0)?;
    t.append_mul(&Slot::Temp(7), &[p1, p2], 0)?;
    t.append_add(&Slot::Temp(8), &[Slot::Temp(7), p3], 0)?;
    t.append_mul(&Slot::Temp(9), &[Slot::Temp(8), p1], 0)?;
    t.append_add(&Slot::Out(2), &[Slot::Temp(9), p3], 0)?;
    let mut app = t.compile()?;
    app.prepare_simd();
    Ok(app)
}

#[test]
fn shared_subexpression_after_an_untaken_branch() -> Result<()> {
    for opt_level in 0..=3 {
        for complex in [false, true] {
            let mut config = base_config();
            config.set_opt_level(opt_level);
            config.set_complex(complex);
            let app = compile(config)?.seal()?;
            // p0 = 0 first (nothing computed e yet), then p0 = 1, then p0 = 0 with other values
            // (a value of e left from the p0 = 1 row is wrong there)
            for [p0, a, b, d] in [
                [0.0, 1.5, -2.0, 0.25],
                [1.0, 2.0, 3.0, 5.0],
                [0.0, -1.0, 4.0, 2.0],
            ] {
                let e = (a * b + d) * a;
                let want = [if p0 != 0.0 { e } else { b }, e + b, e + d];
                let what =
                    format!("opt level {opt_level}, complex = {complex}, p = {p0} {a} {b} {d}");
                let got: Vec<f64> = if complex {
                    let x = [p0, a, b, d].map(|v| Complex::new(v, 0.0));
                    let mut outs = [Complex::new(f64::NAN, f64::NAN); 3];
                    app.evaluate_matrix(&x, &mut outs, 1);
                    for z in outs {
                        assert_eq!(z.im, 0.0, "{what}: {outs:?}");
                    }
                    outs.iter().map(|z| z.re).collect()
                } else {
                    let mut outs = [f64::NAN; 3];
                    app.evaluate_matrix(&[p0, a, b, d], &mut outs, 1);
                    outs.to_vec()
                };
                assert_eq!(got, want, "{what}");
            }
        }
    }
    Ok(())
}

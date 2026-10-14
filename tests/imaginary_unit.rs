//! `x * i` and `x * -i` in real kernels. The builder rewrote them as `times_i(x)` /
//! `times_neg_i(x)` in real kernels too, where the native generators emit nothing: the result
//! was whatever the register held (often `x`). A real kernel holds the real part, 0.
mod common;
use common::base_config;

use anyhow::Result;
use num_complex::Complex;
use symjit::{Applet, Composer, Config, Slot, Translator};

/// out0 = p0 * i + p1, out1 = -i * p0 + p1, out2 = p2 * i * p1 + 1
fn kernel(config: Config) -> Result<Applet> {
    let mut ev = Translator::new(config);
    ev.set_num_params(3);
    let one = ev.append_constant(Complex::new(1.0, 0.0))?;
    let i = ev.append_constant(Complex::new(0.0, 1.0))?;
    let neg_i = ev.append_constant(Complex::new(0.0, -1.0))?;

    ev.append_mul(&Slot::Temp(0), &[Slot::Param(0), Slot::Const(i)], 0)?;
    ev.append_add(&Slot::Out(0), &[Slot::Temp(0), Slot::Param(1)], 0)?;
    ev.append_mul(&Slot::Temp(1), &[Slot::Const(neg_i), Slot::Param(0)], 0)?;
    ev.append_add(&Slot::Out(1), &[Slot::Temp(1), Slot::Param(1)], 0)?;
    ev.append_mul(
        &Slot::Temp(2),
        &[Slot::Param(2), Slot::Const(i), Slot::Param(1)],
        0,
    )?;
    ev.append_add(&Slot::Out(2), &[Slot::Temp(2), Slot::Const(one)], 0)?;
    ev.compile()?.seal()
}

fn points() -> Vec<[f64; 3]> {
    (0..13)
        .map(|k| {
            let k = k as f64;
            [0.5 + 0.25 * k, 2.0 - 0.125 * k, -1.0 + 0.5 * k]
        })
        .collect()
}

#[test]
fn real_kernels_take_the_real_part() -> Result<()> {
    let pts = points();
    for direct in [false, true] {
        for opt_level in 0..=3 {
            for compress in [false, true] {
                for simd in [false, true] {
                    let opts = format!(
                        "direct = {direct}, opt_level = {opt_level}, compress = {compress}, simd = {simd}"
                    );
                    let mut config = base_config();
                    config.set_direct(direct);
                    config.set_opt_level(opt_level);
                    config.set_compress(compress);
                    config.set_simd(simd);
                    let app = kernel(config)?;

                    let want: Vec<f64> = pts.iter().flat_map(|p| [p[1], p[1], 1.0]).collect();

                    let mut got = Vec::new();
                    for p in pts.iter() {
                        let mut outs = [0.0; 3];
                        app.evaluate(p, &mut outs);
                        got.extend(outs);
                    }
                    assert_eq!(got, want, "evaluate, {opts}");

                    let args: Vec<f64> = pts.iter().flatten().copied().collect();
                    let mut outs = vec![0.0; 3 * pts.len()];
                    app.evaluate_matrix(&args, &mut outs, pts.len());
                    assert_eq!(outs, want, "evaluate_matrix, {opts}");
                }
            }
        }
    }
    Ok(())
}

#[test]
fn complex_kernels_multiply_by_i() -> Result<()> {
    for direct in [false, true] {
        let mut config = base_config();
        config.set_complex(true);
        config.set_direct(direct);
        let app = kernel(config)?;
        for p in points() {
            let args = p.map(|v| Complex::new(v, 0.0));
            let mut outs = [Complex::default(); 3];
            app.evaluate(&args, &mut outs);
            let want = [
                Complex::new(p[1], p[0]),
                Complex::new(p[1], -p[0]),
                Complex::new(1.0, p[2] * p[1]),
            ];
            assert_eq!(outs, want, "{p:?}, direct = {direct}");
        }
    }
    Ok(())
}

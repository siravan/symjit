//! With `simd_branch` the jump at the end of a then part was dropped, so the else part ran on
//! every evaluation, in the scalar and the SIMD kernels, even when no point needed it (a
//! Symbolica program whose else parts held most of the work ran 7-20x slower). The jump is now
//! taken when no lane has the condition zero. Here the else part calls a function that counts
//! its calls: rows that all take the then part must not call it.
mod common;
use common::base_config;

use anyhow::Result;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use symjit::{Applet, Composer, Defuns, Slot, Translator};

/// out0 = if p0 != 0 { p1 p2 } else { f(p1) + p2 }, with f(y) = 3 + y^2 + y counting its calls.
fn compile(direct: bool, simd_branch: bool, opt_level: u8) -> Result<(Applet, Arc<AtomicUsize>)> {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let f: Box<dyn Fn(&[f64]) -> f64 + Send + Sync> = Box::new(move |args| {
        counter.fetch_add(1, Ordering::Relaxed);
        3.0 + args[0] * args[0] + args[0]
    });
    let mut defuns = Defuns::new();
    defuns.add_sliced_func("f", f)?;
    let mut config = base_config();
    config.set_opt_level(opt_level);
    config.set_direct(direct);
    config.set_simd_branch(simd_branch);
    config.set_defuns(defuns);
    let mut t = Translator::new(config);
    t.set_num_params(3);
    let (p0, p1, p2) = (Slot::Param(0), Slot::Param(1), Slot::Param(2));
    t.append_if_else(&p0, 0)?;
    t.append_mul(&Slot::Temp(0), &[p1, p2], 0)?;
    t.append_goto(1)?;
    t.append_label(0)?;
    t.append_fun(&Slot::Temp(1), "f", &[p1], true)?;
    t.append_add(&Slot::Temp(2), &[Slot::Temp(1), p2], 0)?;
    t.append_label(1)?;
    t.append_join(&Slot::Out(0), &p0, &Slot::Temp(0), &Slot::Temp(2))?;
    let mut app = t.compile()?;
    app.prepare_simd();
    Ok((app.seal()?, calls))
}

#[test]
fn else_part_skipped_when_no_row_needs_it() -> Result<()> {
    const ROWS: usize = 19; // SIMD blocks and a remainder
    for direct in [false, true] {
        for simd_branch in [false, true] {
            for opt_level in 0..=3 {
                let what = format!(
                    "direct = {direct}, simd_branch = {simd_branch}, opt level {opt_level}"
                );
                let cases: [(&str, fn(usize) -> bool); 3] = [
                    ("all then", |_| true),
                    ("all else", |_| false),
                    ("mixed", |i| i % 3 != 1),
                ];
                for (case, take_then) in cases {
                    let (app, calls) = compile(direct, simd_branch, opt_level)?;
                    let x: Vec<f64> = (0..ROWS)
                        .flat_map(|i| {
                            let y = i as f64 * 0.25 - 2.0;
                            [if take_then(i) { 1.0 } else { 0.0 }, y, 1.5 - y]
                        })
                        .collect();
                    let mut outs = vec![f64::NAN; ROWS];
                    app.evaluate_matrix(&x, &mut outs, ROWS);
                    for i in 0..ROWS {
                        let (y, z) = (x[3 * i + 1], x[3 * i + 2]);
                        let want = if take_then(i) {
                            y * z
                        } else {
                            3.0 + y * y + y + z
                        };
                        assert_eq!(outs[i], want, "{what}, {case}: row {i}");
                    }
                    let calls = calls.load(Ordering::Relaxed);
                    match case {
                        "all then" => assert_eq!(calls, 0, "{what}, {case}: f called"),
                        _ => assert!(calls > 0, "{what}, {case}: f not called"),
                    }
                }
            }
        }
    }
    Ok(())
}

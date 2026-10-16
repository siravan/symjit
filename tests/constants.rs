//! `append_constant` returns the index that `Slot::Const` reads back, for both translators
//! and in both real and complex mode. In complex mode, `IndirectTranslator` returned
//! 0, 0, 1, 1, 2, 2, ... (so a second constant aliased the first); in real mode,
//! `DirectTranslator` returned 1, 3, 5, ... (and indexed past its table).
mod common;
use common::base_config;

use anyhow::Result;
use num_complex::Complex;
use symjit::{Composer, Slot, Translator};
use symjit::prelude::*;

/// p0 + 1 * 10 + 2 * 100 + 3 * 1000, with all six numbers as constants
fn weighted_sum(complex: bool, direct: bool) -> Result<f64> {
    let mut config = base_config();
    config.set_complex(complex);
    config.set_direct(direct);

    let mut ev = Translator::new(config);
    ev.set_num_params(1);

    let mut consts = Vec::new();
    for v in [1.0, 2.0, 3.0, 10.0, 100.0, 1000.0] {
        consts.push(ev.append_constant(Complex::new(v, 0.0))?);
    }
    assert_eq!(consts, [0, 1, 2, 3, 4, 5]);

    for i in 0..3 {
        ev.append_mul(
            &Slot::Temp(i),
            &[Slot::Const(consts[i]), Slot::Const(consts[i + 3])],
            0,
        )?;
    }
    ev.append_add(
        &Slot::Out(0),
        &[Slot::Param(0), Slot::Temp(0), Slot::Temp(1), Slot::Temp(2)],
        0,
    )?;
    let app = ev.compile()?.seal()?;

    Ok(if complex {
        let v = app.evaluate_single(&[Complex::new(0.5, 0.0)]);
        assert_eq!(v.im, 0.0);
        v.re
    } else {
        app.evaluate_single(&[0.5])
    })
}

#[test]
fn constant_indices() -> Result<()> {
    for complex in [false, true] {
        for direct in [false, true] {
            assert_eq!(
                weighted_sum(complex, direct)?,
                3210.5,
                "complex = {complex}, direct = {direct}"
            );
        }
    }
    Ok(())
}

//! Complex division by a constant (x86 fast complex: the divisor was clobbered).
mod common;
use common::base_config;

use anyhow::Result;
use num_complex::Complex;
use symjit::{Composer, Slot, Translator};
use symjit::prelude::*;

/// x86 fast complex: a constant divisor is loaded into `Reg::Temp`, which `divide` used as
/// scratch, so `z / c` divided by a garbage value.
#[test]
fn complex_division_by_constant() -> Result<()> {
    for direct in [false, true] {
        for fast_complex in [true, false] {
            let mut config = base_config();
            config.set_complex(true);
            config.set_direct(direct);
            config.set_fast_complex(fast_complex);
            let mut t = Translator::new(config);
            t.set_num_params(1);
            let c = t.append_constant(Complex::new(4.0, 0.0))?;
            let d = t.append_constant(Complex::new(4.0, 1.0))?;
            // the indirect translator rewrites `z * recip(c)` into `z / c`
            t.append_fun(&Slot::Temp(0), "recip", &[Slot::Const(c)], false)?;
            t.append_mul(&Slot::Out(0), &[Slot::Param(0), Slot::Temp(0)], 0)?;
            t.append_fun(&Slot::Temp(1), "recip", &[Slot::Const(d)], false)?;
            t.append_mul(&Slot::Out(1), &[Slot::Param(0), Slot::Temp(1)], 0)?;
            let app = t.compile()?.seal()?;

            let z = Complex::new(0.5, -1.5);
            let mut outs = [Complex::default(); 2];
            app.evaluate(&[z], &mut outs);
            let expected = [z / 4.0, z / Complex::new(4.0, 1.0)];
            for (got, expected) in outs.iter().zip(expected.iter()) {
                assert!(
                    (got - expected).norm() <= 4.0 * f64::EPSILON * expected.norm(),
                    "{got} != {expected}, direct = {direct}, fast_complex = {fast_complex}"
                );
            }
        }
    }
    Ok(())
}

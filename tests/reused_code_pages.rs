// Machine code is freed when its last owner drops, so the allocator hands its pages out again for
// the next kernel. On a core whose instruction cache is not coherent with its data cache (aarch64),
// the new code must be flushed to it, or the core may run the previous kernel's instructions (wrong
// values or SIGILL). Each round compiles a kernel of another shape with another constant, evaluates
// it and drops it, so consecutive kernels differ at the same addresses.

mod common;
use common::base_config;

use anyhow::Result;
use num_complex::Complex;
use symjit::{Composer, Slot, Translator};

fn kernel(shape: usize, c: f64) -> Result<symjit::Application> {
    let mut config = base_config();
    config.set_complex(false);
    config.set_direct(false);

    let mut t = Translator::new(config);
    t.set_num_params(2);
    let k = Slot::Const(t.append_constant(Complex::new(c, 0.0))?);
    let (x, y) = (Slot::Param(0), Slot::Param(1));
    match shape {
        0 => t.append_mul(&Slot::Out(0), &[x, k], 0)?,
        1 => t.append_add(&Slot::Out(0), &[x, k], 0)?,
        2 => {
            t.append_mul(&Slot::Temp(0), &[x, y], 0)?;
            t.append_add(&Slot::Out(0), &[Slot::Temp(0), k], 0)?;
        }
        _ => {
            t.append_add(&Slot::Temp(0), &[x, k], 0)?;
            t.append_mul(&Slot::Out(0), &[Slot::Temp(0), y], 0)?;
        }
    }
    t.compile()
}

#[test]
fn kernels_on_reused_pages_run_their_own_code() {
    let (x, y) = (1.25, -3.5);
    for i in 0..400 {
        let shape = i % 4;
        let c = i as f64 + 0.5;
        let expected = match shape {
            0 => x * c,
            1 => x + c,
            2 => x * y + c,
            _ => (x + c) * y,
        };
        let got = kernel(shape, c).unwrap().evaluate_single(&[x, y]);
        assert!(
            (got - expected).abs() <= 1e-12 * expected.abs().max(1.0),
            "round {i}, shape {shape}: {got} != {expected}"
        );
    }
}

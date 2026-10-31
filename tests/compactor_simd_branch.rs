//! The compactor returns a stack slot to the pool at its last use, except in a loop. It used to
//! find loops by counting labels against jumps: a label not jumped to before was a loop start.
//! With `simd_branch` an if/else has no jump to its end label (`translate_goto`), so every
//! if/else was taken for the start of a loop that never ended, and no slot was reused after the
//! first one. Loops are now found from the jumps back to their labels, and twenty if/else
//! blocks in a row need the same stack with and without `simd_branch`.
mod common;
use common::base_config;

use anyhow::Result;
use symjit::{Composer, Config, Slot, Translator};

/// out0 = the sum over b < 20 of: if p[b % 4] != 0 { (p p + p) p } else { p p + p }, with
/// other parameters in each block.
fn stack_size(config: Config) -> Result<usize> {
    let mut t = Translator::new(config);
    t.set_num_params(4);
    let p = [0, 1, 2, 3].map(Slot::Param);
    let mut temps = 0;
    let mut temp = || {
        temps += 1;
        Slot::Temp(temps - 1)
    };
    let mut total: Option<Slot> = None;
    for b in 0..20 {
        let cond = p[b % 4];
        t.append_if_else(&cond, 2 * b)?;
        let (x, y, z) = (temp(), temp(), temp());
        t.append_mul(&x, &[p[(b + 1) % 4], p[(b + 2) % 4]], 0)?;
        t.append_add(&y, &[x, p[(b + 3) % 4]], 0)?;
        t.append_mul(&z, &[y, p[b % 4]], 0)?;
        t.append_goto(2 * b + 1)?;
        t.append_label(2 * b)?;
        let (u, v) = (temp(), temp());
        t.append_mul(&u, &[p[(b + 3) % 4], p[(b + 1) % 4]], 0)?;
        t.append_add(&v, &[u, p[(b + 2) % 4]], 0)?;
        t.append_label(2 * b + 1)?;
        let j = temp();
        t.append_join(&j, &cond, &z, &v)?;
        total = Some(match total {
            None => j,
            Some(s) => {
                let r = temp();
                t.append_add(&r, &[s, j], 0)?;
                r
            }
        });
    }
    t.append_assign(&Slot::Out(0), &total.unwrap())?;
    Ok(t.compile()?.measure("stack-size"))
}

#[test]
fn simd_branch_needs_no_more_stack() -> Result<()> {
    for opt_level in 0..=3 {
        let mut sizes = [0; 2];
        for (k, simd_branch) in [false, true].into_iter().enumerate() {
            let mut config = base_config();
            config.set_opt_level(opt_level);
            config.set_simd_branch(simd_branch);
            sizes[k] = stack_size(config)?;
        }
        assert_eq!(sizes[0], sizes[1], "opt level {opt_level}");
    }
    Ok(())
}

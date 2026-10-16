//! The MIR of object files: the calls of symjit's real functions that the C math library does
//! not have (`csc`, `sec`, `cot`, `csch`, `sech`, `coth`, `sinc`, `arg`) are composed of C
//! functions, as `code.rs` computes them, so that the kernel calls only the C library. At a
//! call the argument is in `Left` (`Ret`) and the result goes to `Ret`; the other registers
//! are free (calls clobber them), and a value kept across the call goes to a new stack slot.

use anyhow::Result;

use crate::mir::{Instruction, Mir};
use crate::model::Program;
use crate::utils::Reg;

/// The index of `v` in the program's constants (added if missing).
fn constant(prog: &mut Program, v: f64) -> u32 {
    let consts = &mut prog.builder.consts;
    match consts.iter().position(|c| c.to_bits() == v.to_bits()) {
        Some(idx) => idx as u32,
        None => {
            consts.push(v);
            (consts.len() - 1) as u32
        }
    }
}

/// `mir` with the calls of the real functions C does not have composed of C functions. New
/// constants and a stack slot (for `sinc`) are added to `prog`.
pub fn lower_for_object(mir: &Mir, prog: &mut Program) -> Result<Mir> {
    let mut out = Mir::new(mir.config.clone());
    let mut slot: Option<u32> = None;

    for ins in mir.code.iter() {
        let Instruction::Call { label, num_args: 1 } = &ins else {
            out.code.push(&ins);
            continue;
        };

        match label.as_str() {
            // 1.0 / sin(x), ...: `recip(Ret, Ret)` divides 1.0 (in `Temp`) by `Ret`
            "csc" | "sec" | "cot" | "csch" | "sech" | "coth" => {
                let base = match label.as_str() {
                    "csc" => "sin",
                    "sec" => "cos",
                    "cot" => "tan",
                    "csch" => "sinh",
                    "sech" => "cosh",
                    _ => "tanh",
                };
                out.call(base, 1)?;
                out.recip(Reg::Ret, Reg::Ret);
            }
            // `code.rs` returns 0.0
            "arg" => out.xor(Reg::Ret, Reg::Ret, Reg::Ret),
            // x == 0 ? 1.0 : sin(x) / x, the select as r ^ ((r ^ 1) & m)
            "sinc" => {
                let s = *slot.get_or_insert_with(|| {
                    let s = prog.builder.stack_size() as u32;
                    prog.builder.count_stack = Some(s as usize + 1);
                    s
                });
                let one = constant(prog, 1.0);
                let (x, m, u) = (Reg::Gen(0), Reg::Gen(1), Reg::Gen(2));
                out.save_stack(Reg::Ret, s);
                out.call("sin", 1)?;
                out.load_stack(x, s);
                out.divide(Reg::Ret, Reg::Ret, x);
                out.xor(m, m, m);
                out.eq(m, x, m);
                out.load_const(u, one);
                out.xor(u, Reg::Ret, u);
                out.and(u, u, m);
                out.xor(Reg::Ret, Reg::Ret, u);
            }
            _ => out.code.push(&ins),
        }
    }

    out.consts = prog.builder.consts.clone();
    out.populate_labels();
    Ok(out)
}

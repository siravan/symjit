use anyhow::{anyhow, Result};
use num_complex::Complex;
use std::collections::HashSet;

use super::code::VirtualTable;
use super::config::Config;
use super::instruction::{BuiltinSymbol, Slot};
use super::mir::Mir;
use super::model::{CellModel, Program};
use super::operation::Operation;
use super::runnable::Application;
use super::symbol::Loc;
use super::utils::*;

pub trait Composer {
    fn append_constant(&mut self, z: Complex<f64>) -> Result<usize>;
    fn append_add(&mut self, lhs: &Slot, args: &[Slot], num_reals: usize) -> Result<()>;
    fn append_mul(&mut self, lhs: &Slot, args: &[Slot], num_reals: usize) -> Result<()>;
    /// `lhs = numerator / denominator`, divided directly (not as a product with the reciprocal).
    fn append_div(
        &mut self,
        lhs: &Slot,
        numerator: &Slot,
        denominator: &Slot,
        numerator_is_real: bool,
        denominator_is_real: bool,
    ) -> Result<()>;
    /// Raise to an integer power with magnitude at most `i32::MAX` (larger ones are an error).
    fn append_pow(&mut self, lhs: &Slot, arg: &Slot, p: i64, is_real: bool) -> Result<()>;
    fn append_powf(&mut self, lhs: &Slot, arg: &Slot, p: &Slot, is_real: bool) -> Result<()>;
    fn append_assign(&mut self, lhs: &Slot, rhs: &Slot) -> Result<()>;
    fn append_label(&mut self, id: usize) -> Result<()>;
    fn append_if_else(&mut self, cond: &Slot, id: usize) -> Result<()>;
    fn append_goto(&mut self, id: usize) -> Result<()>;
    fn append_external_fun(&mut self, lhs: &Slot, op: &str, args: &[Slot]) -> Result<()>;
    fn append_fun(&mut self, lhs: &Slot, fun: &str, args: &[Slot], is_real: bool) -> Result<()>;
    #[allow(unused)]
    fn append_fun_v1(
        &mut self,
        lhs: &Slot,
        fun: &BuiltinSymbol,
        arg: &Slot,
        is_real: bool,
    ) -> Result<()>;
    fn append_join(
        &mut self,
        lhs: &Slot,
        cond: &Slot,
        true_val: &Slot,
        false_val: &Slot,
    ) -> Result<()>;
    fn set_num_params(&mut self, num_params: usize);
    fn compile(&mut self) -> Result<Application>;
}

#[derive(Debug)]
pub struct DirectTranslator {
    pub mir: Mir,
    pub consts: Vec<f64>,
    pub reals: HashSet<Loc>,
    pub num_params: usize,
    pub count_params: usize,
    pub count_outs: usize,
    pub ft: HashSet<String>,
    pub prog: Program,
    open_ifs: Vec<(usize, Slot)>, // the if/else blocks entered: else label, condition
    last_jump: Option<Jump>,      // the last jump, while only labels follow it
}

// A jump written to the MIR, with the MIR's length before it (its condition included) and
// after it, and the labels written since.
#[derive(Debug)]
struct Jump {
    target: usize,
    start: (usize, usize), // (bytes, instructions) before the jump
    end: usize,            // instructions after the jump
    labels: Vec<String>,
}

impl DirectTranslator {
    pub fn new(mut config: Config) -> DirectTranslator {
        // opt_level should be 2 for proper register allocation.
        config.set_opt_level(2);

        let ml = CellModel::new();
        let mir = Mir::new(config.clone());
        let prog = Program::new(&ml, config).unwrap();

        DirectTranslator {
            mir,
            consts: Vec::new(),
            reals: HashSet::new(),
            num_params: 0,
            count_params: 0,
            count_outs: 0,
            ft: HashSet::new(),
            prog,
            open_ifs: Vec::new(),
            last_jump: None,
        }
    }

    fn begin_jump(&self) -> (usize, usize) {
        (self.mir.code.buf.len(), self.mir.code.ip)
    }

    fn end_jump(&mut self, target: usize, start: (usize, usize)) {
        self.last_jump = Some(Jump {
            target,
            start,
            end: self.mir.code.ip,
            labels: Vec::new(),
        });
    }

    fn find_sym(&mut self, name: &str) -> Option<Loc> {
        self.prog
            .builder
            .symbol_table()
            .find_sym(name)
            .map(|s| s.borrow().loc)
    }

    fn load(&mut self, dst: Reg, slot: &Slot) -> Result<()> {
        match slot {
            Slot::Const(idx) => {
                let n = 2 * *idx as u32;
                if n as usize > self.consts.len() {
                    return Err(anyhow!(
                        "constant not found. Make sure constants are defined first."
                    ));
                }
                if self.consts[(n + 1) as usize] == 0.0 {
                    self.mir.load_const(dst, n);
                } else {
                    // Important! We cannot use reg(0) and reg(1) here to
                    // prevent conflict.
                    self.mir.load_const(Reg::Ret, n);
                    self.mir.load_const(dst, n + 1);
                    self.mir.complex(dst, Reg::Ret, dst);
                }
            }
            Slot::Param(idx) => {
                for i in self.count_params..=*idx {
                    let name = format!("Param{}", i);
                    self.prog.builder.symbol_table().add_param(&name);
                }
                self.count_params = self.count_params.max(*idx + 1);
                let name = format!("Param{}", idx);
                if let Some(Loc::Param(i)) = self.find_sym(&name) {
                    self.mir.load_param(dst, i);
                    if self.reals.contains(&Loc::Param(i)) {
                        self.mir.real(dst, dst);
                    }
                } else {
                    return Err(anyhow!("error adding {:?}.", name));
                }
            }
            Slot::Out(idx) => {
                let name = format!("Out{}", idx);
                if let Some(Loc::Stack(i)) = self.find_sym(&name) {
                    self.mir.load_stack(dst, i);
                } else {
                    return Err(anyhow!("{:?} not found.", name));
                }
            }
            Slot::Temp(idx) => {
                let name = format!("Temp{}", idx);
                if let Some(Loc::Stack(i)) = self.find_sym(&name) {
                    self.mir.load_stack(dst, i);
                } else {
                    return Err(anyhow!("{:?} not found.", name));
                }
            }
            _ => return Err(anyhow!("{:?} is not defined.", slot)),
        }

        Ok(())
    }

    fn as_loc(&mut self, slot: &Slot) -> Option<Loc> {
        match slot {
            Slot::Param(idx) => self.find_sym(&format!("Param{}", idx)),
            Slot::Arg(idx) => self.find_sym(&format!("__Arg{}", idx)),
            Slot::Out(idx) => self.find_sym(&format!("Out{}", idx)),
            Slot::Temp(idx) => self.find_sym(&format!("Temp{}", idx)),
            _ => None,
        }
    }

    fn is_minus_one(&self, slot: &Slot) -> bool {
        if let Slot::Const(idx) = slot {
            let n = 2 * *idx;
            self.consts[n] == -1.0 && self.consts[n + 1] == 0.0
        } else {
            false
        }
    }

    fn add_stack(&mut self, name: &str) -> Result<u32> {
        if let Some(Loc::Stack(i)) = self.find_sym(name) {
            Ok(i)
        } else {
            self.prog.builder.symbol_table().add_stack(name);
            if let Some(Loc::Stack(i)) = self.find_sym(name) {
                Ok(i)
            } else {
                Err(anyhow!("error adding {:?}", name))
            }
        }
    }

    fn save(&mut self, src: Reg, slot: &Slot) -> Result<()> {
        match slot {
            Slot::Out(idx) => {
                self.count_outs = self.count_outs.max(*idx + 1);
                let name = format!("Out{}", idx);
                let i = self.add_stack(&name)?;
                self.mir.save_stack(src, i);
            }
            Slot::Temp(idx) => {
                let name = format!("Temp{}", idx);
                let i = self.add_stack(&name)?;
                self.mir.save_stack(src, i);
            }
            Slot::Arg(_) => {
                if let Some(Loc::Stack(i)) = self.as_loc(slot) {
                    self.mir.save_stack(src, i);
                } else {
                    return Err(anyhow!("argument slot {:?} is not reserved", slot));
                }
            }
            _ => unreachable!(),
        };

        Ok(())
    }

    fn mark_real(&mut self, slot: &Slot, is_real: bool) {
        if let Slot::Param(idx) = slot {
            if is_real {
                self.reals
                    .insert(Loc::Param((self.prog.config().sizeof() * idx) as u32));
            }
        }
    }

    fn compile_unary(&mut self, op: &str, dst: Reg, r: Reg) -> Result<()> {
        match op {
            "neg" => self.mir.neg(dst, r),
            "not" => self.mir.not(dst, r),
            "abs" => self.mir.abs(dst, r),
            "abs2" => self.mir.abs2(dst, r),
            "root" => self.mir.root(dst, r),
            "real_root" => self.mir.real_root(dst, r),
            "square" => self.mir.square(dst, r),
            "cube" => self.mir.cube(dst, r),
            "recip" => self.mir.recip(dst, r),
            "round" => self.mir.round(dst, r),
            "floor" => self.mir.floor(dst, r),
            "ceiling" => self.mir.ceiling(dst, r),
            "trunc" => self.mir.trunc(dst, r),
            "frac" => self.mir.frac(dst, r),
            "real" => self.mir.real(dst, r),
            "imaginary" => self.mir.imaginary(dst, r),
            "conjugate" => self.mir.conjugate(dst, r),
            "iszero" => self.mir.iszero(dst, r),
            _ => return Err(anyhow!("unary operator {:?} is not recognized", op)),
        };

        Ok(())
    }

    fn compile_binary(&mut self, op: &str, dst: Reg, l: Reg, r: Reg) -> Result<()> {
        match op {
            "plus" => self.mir.plus(dst, l, r),
            "minus" => self.mir.minus(dst, l, r),
            "times" => self.mir.times(dst, l, r),
            "divide" => self.mir.divide(dst, l, r),
            "rem" => self.mir.fmod(dst, l, r),
            "gt" => self.mir.gt(dst, l, r),
            "geq" => self.mir.geq(dst, l, r),
            "lt" => self.mir.lt(dst, l, r),
            "leq" => self.mir.leq(dst, l, r),
            "eq" => self.mir.eq(dst, l, r),
            "neq" => self.mir.neq(dst, l, r),
            "and" => self.mir.and(dst, l, r),
            "or" => self.mir.or(dst, l, r),
            "xor" => self.mir.xor(dst, l, r),
            "complex" => self.mir.complex(dst, l, r),
            _ => return Err(anyhow!("binary operator {:?} is not recognized", op)),
        }

        Ok(())
    }

    fn append_fun_generic(
        &mut self,
        lhs: &Slot,
        op: &str,
        args: &[Slot],
        is_real: bool,
    ) -> Result<()> {
        let n = args.len();
        self.mark_real(lhs, is_real);

        // kernel functions (recursive calls, applets) take their arguments in the
        // `__Arg` slots, not in registers, even when named `composer_*`
        let by_register = !self.mir.config.is_kernel_func(op)
            && (VirtualTable::from_str(op).is_ok() || op.starts_with("composer_"));

        if by_register {
            if n == 1 {
                self.load(reg(0), &args[0])?;
                self.mir.setup_call_unary(reg(0));
                self.mir.call(op, 1)?;
                self.save(Reg::Ret, lhs)?;
                self.ft.insert(op.to_string());
            } else if n == 2 {
                self.load(reg(0), &args[0])?;
                self.load(reg(1), &args[1])?;
                self.mir.setup_call_binary(reg(0), reg(1));
                self.mir.call(op, 2)?;
                self.save(Reg::Ret, lhs)?;
                self.ft.insert(op.to_string());
            } else {
                return Err(anyhow!("wrong number of arguments to {:?}", op));
            }
        } else if self.mir.config.is_intrinsic_unary(&Operation::new(op)) && n == 1 {
            self.load(reg(0), &args[0])?;
            self.compile_unary(op, reg(1), reg(0))?;
            self.save(reg(1), lhs)?;
        } else if self.mir.config.is_intrinsic_binary(&Operation::new(op)) && n == 2 {
            self.load(reg(0), &args[0])?;
            self.load(reg(1), &args[1])?;
            self.compile_binary(op, reg(2), reg(0), reg(1))?;
            self.save(reg(2), lhs)?;
        } else {
            if n > self.mir.config.num_args() as usize {
                return Err(anyhow!(
                    "function {:?} requires {} arguments, but only {} argument slots are reserved",
                    op,
                    n,
                    self.mir.config.num_args()
                ));
            }
            for (i, arg) in args.iter().enumerate() {
                self.load(reg(0), arg)?;
                self.save(reg(0), &Slot::Arg(i))?;
            }

            self.mir.call(op, n)?;
            self.save(Reg::Ret, lhs)?;
            self.ft.insert(op.to_string());
        }

        Ok(())
    }
}

impl Composer for DirectTranslator {
    fn append_constant(&mut self, z: Complex<f64>) -> Result<usize> {
        // `consts` holds (re, im) pairs; `load` reads constant `idx` at `2 * idx`
        self.consts.push(z.re);
        self.consts.push(z.im);
        Ok(self.consts.len() / 2 - 1)
    }

    fn append_add(&mut self, lhs: &Slot, args: &[Slot], num_reals: usize) -> Result<()> {
        self.mark_real(&args[0], 0 < num_reals);
        self.load(reg(0), &args[0])?;

        for (i, arg) in args.iter().enumerate().skip(1) {
            self.mark_real(arg, i < num_reals);
            self.load(reg(1), arg)?;
            self.mir.plus(reg(0), reg(0), reg(1));
        }
        self.save(reg(0), lhs)?;

        Ok(())
    }

    fn append_mul(&mut self, lhs: &Slot, args: &[Slot], num_reals: usize) -> Result<()> {
        self.mark_real(&args[0], 0 < num_reals);
        self.load(reg(0), &args[0])?;

        let mut negate = false;

        for (i, arg) in args.iter().enumerate().skip(1) {
            if self.is_minus_one(arg) {
                negate = !negate;
            } else {
                self.mark_real(arg, i < num_reals);
                self.load(reg(1), arg)?;
                self.mir.times(reg(0), reg(0), reg(1));
            }
        }

        if negate {
            self.mir.neg(reg(0), reg(0));
        }

        self.save(reg(0), lhs)?;

        Ok(())
    }

    fn append_div(
        &mut self,
        lhs: &Slot,
        numerator: &Slot,
        denominator: &Slot,
        numerator_is_real: bool,
        denominator_is_real: bool,
    ) -> Result<()> {
        self.mark_real(numerator, numerator_is_real);
        self.load(reg(0), numerator)?;
        self.mark_real(denominator, denominator_is_real);
        self.load(reg(1), denominator)?;
        self.mir.divide(reg(0), reg(0), reg(1));
        self.save(reg(0), lhs)?;
        Ok(())
    }

    fn append_pow(&mut self, lhs: &Slot, arg: &Slot, p: i64, is_real: bool) -> Result<()> {
        if p.unsigned_abs() > i32::MAX as u64 {
            return Err(anyhow!("integer power magnitude exceeds i32::MAX"));
        }
        self.mark_real(arg, is_real);
        self.load(reg(0), arg)?;

        match p {
            2 => self.mir.square(reg(1), reg(0)),
            3 => self.mir.cube(reg(1), reg(0)),
            -1 => self.mir.recip(reg(1), reg(0)),
            p => self.mir.powi(reg(1), reg(0), p as i32),
        }
        self.save(reg(1), lhs)?;

        Ok(())
    }

    fn append_powf(&mut self, lhs: &Slot, arg: &Slot, p: &Slot, is_real: bool) -> Result<()> {
        self.load(reg(0), arg)?;
        self.mark_real(arg, is_real);

        self.load(reg(1), p)?;
        self.mir.setup_call_binary(reg(0), reg(1));
        self.mir.call("power", 2)?;
        self.save(Reg::Ret, lhs)?;
        self.ft.insert("power".to_string());
        Ok(())
    }

    fn append_assign(&mut self, lhs: &Slot, rhs: &Slot) -> Result<()> {
        self.load(reg(0), rhs)?;
        self.save(reg(0), lhs)?;
        Ok(())
    }

    fn append_label(&mut self, id: usize) -> Result<()> {
        let label = format!(".S{}", id);
        // A jump to this label with only labels after it jumps to where it would go anyway:
        // remove it and its condition from the MIR, and write the labels again (see
        // `Block::add_label`)
        if let Some(jump) = self.last_jump.take() {
            if self.mir.code.ip == jump.end + jump.labels.len() {
                if jump.target == id {
                    self.mir.code.buf.truncate(jump.start.0);
                    self.mir.code.ip = jump.start.1;
                    for l in &jump.labels {
                        self.mir.set_label(l);
                    }
                } else {
                    let mut jump = jump;
                    jump.labels.push(label.clone());
                    self.last_jump = Some(jump);
                }
            }
        }
        self.mir.set_label(&label);
        // the else part of an open if/else starts here (see `IndirectTranslator::translate_label`)
        if let Some(k) = self.open_ifs.iter().rposition(|(l, _)| *l == id) {
            self.open_ifs.truncate(k);
        }
        Ok(())
    }

    fn append_if_else(&mut self, cond: &Slot, id: usize) -> Result<()> {
        let label = format!(".S{}", id);
        let start = self.begin_jump();
        self.load(reg(0), cond)?;
        self.mir.iszero(reg(1), reg(0));
        self.mir.branch_if(reg(1), &label, true);
        self.end_jump(id, start);
        self.open_ifs.push((id, *cond));
        Ok(())
    }

    // With `simd_branch`, the jump over the else part is taken when no lane has the condition
    // zero (see `IndirectTranslator::translate_goto`). The condition is read again, as the join
    // after the if/else reads it.
    fn append_goto(&mut self, id: usize) -> Result<()> {
        let label = format!(".S{}", id);
        let start = self.begin_jump();
        if !self.mir.config.simd_branch() || !self.mir.config.symbolica() {
            self.mir.branch(&label);
        } else if let Some((_, cond)) = self.open_ifs.last() {
            let cond = *cond;
            self.load(reg(0), &cond)?;
            self.mir.iszero(reg(1), reg(0));
            self.mir.branch_if(reg(1), &label, false);
        } else {
            return Ok(());
        }
        self.end_jump(id, start);
        Ok(())
    }

    fn append_join(
        &mut self,
        lhs: &Slot,
        cond: &Slot,
        true_val: &Slot,
        false_val: &Slot,
    ) -> Result<()> {
        self.load(reg(0), cond)?;
        self.mir.xor(Reg::Ret, reg(0), reg(0));
        self.mir.eq(reg(1), reg(0), Reg::Ret);
        self.save(reg(1), &Slot::Arg(0))?;
        self.load(reg(0), true_val)?;
        self.load(reg(1), false_val)?;
        let loc = self.as_loc(&Slot::Arg(0)).unwrap();
        self.mir.ifelse(reg(2), reg(1), reg(0), loc);
        self.save(reg(2), lhs)?;

        Ok(())
    }

    fn append_external_fun(&mut self, lhs: &Slot, op: &str, args: &[Slot]) -> Result<()> {
        self.append_fun_generic(lhs, op, args, false)
    }

    fn append_fun_v1(
        &mut self,
        lhs: &Slot,
        fun: &BuiltinSymbol,
        arg: &Slot,
        is_real: bool,
    ) -> Result<()> {
        let op = match fun.0 {
            2 => "exp",
            3 => "ln",
            4 => "sin",
            5 => "cos",
            6 => {
                if is_real {
                    "real_root"
                } else {
                    "root"
                }
            }
            7 => "conjugate",
            8 => "abs",
            _ => return Err(anyhow!("Builtin function {} is not defined.", fun.0)),
        };

        self.append_fun_generic(lhs, op, &[*arg], is_real)
    }

    fn append_fun(&mut self, lhs: &Slot, fun: &str, args: &[Slot], is_real: bool) -> Result<()> {
        self.append_fun_generic(
            lhs,
            &self.mir.config.symbolica_fun(fun, is_real),
            args,
            is_real,
        )
    }

    fn set_num_params(&mut self, num_params: usize) {
        self.num_params = num_params
    }

    fn compile(&mut self) -> Result<Application> {
        let k = if self.mir.config.is_complex() { 2 } else { 1 };

        for i in 0..self.count_outs {
            self.load(reg(0), &Slot::Out(i))?;
            self.mir.save_mem(reg(0), (k * i) as u32);
        }

        let mut prog: Program = self.prog.clone();

        prog.count_params = k * self.count_params.max(self.num_params);
        prog.count_obs = k * self.count_outs;
        prog.builder.consts = self.consts.clone();
        prog.builder.ft = self.ft.clone();

        let mut mir: Mir = std::mem::take(&mut self.mir);
        prog.builder.optimize_mir(&mut mir)?;
        let mut app = Application::with_mir(prog, self.reals.clone(), mir)?;
        app.prepare_simd();

        Ok(app)
    }
}

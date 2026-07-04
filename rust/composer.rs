use anyhow::{anyhow, Result};
use num_complex::Complex;
use std::collections::HashSet;

use crate::code::VirtualTable;
use crate::config::{Config, SLICE_CAP};
use crate::instruction::{BuiltinSymbol, Slot};
use crate::mir::Mir;
use crate::model::{CellModel, Program};
use crate::runnable::Application;
use crate::symbol::Loc;
use crate::utils::*;

pub trait Composer {
    fn append_constant(&mut self, z: Complex<f64>) -> Result<usize>;
    fn append_add(&mut self, lhs: &Slot, args: &[Slot], num_reals: usize) -> Result<()>;
    fn append_mul(&mut self, lhs: &Slot, args: &[Slot], num_reals: usize) -> Result<()>;
    fn append_pow(&mut self, lhs: &Slot, arg: &Slot, p: i64, is_real: bool) -> Result<()>;
    fn append_powf(&mut self, lhs: &Slot, arg: &Slot, p: &Slot, is_real: bool) -> Result<()>;
    fn append_assign(&mut self, lhs: &Slot, rhs: &Slot) -> Result<()>;
    fn append_label(&mut self, id: usize) -> Result<()>;
    fn append_if_else(&mut self, cond: &Slot, id: usize) -> Result<()>;
    fn append_goto(&mut self, id: usize) -> Result<()>;
    fn append_external_fun(&mut self, lhs: &Slot, op: &str, args: &[Slot]) -> Result<()>;
    fn append_fun(&mut self, lhs: &Slot, fun: &str, args: &[Slot], is_real: bool) -> Result<()>;
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
    pub conditions: Vec<Slot>,
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
            conditions: Vec::new(),
        }
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
                } else {
                    return Err(anyhow!("error adding {:?}.", &name));
                }
            }
            Slot::Out(idx) => {
                let name = format!("Out{}", idx);
                if let Some(Loc::Stack(i)) = self.find_sym(&name) {
                    self.mir.load_stack(dst, i);
                } else {
                    return Err(anyhow!("{:?} not found.", &name));
                }
            }
            Slot::Temp(idx) => {
                let name = format!("Temp{}", idx);
                if let Some(Loc::Stack(i)) = self.find_sym(&name) {
                    self.mir.load_stack(dst, i);
                } else {
                    return Err(anyhow!("{:?} not found.", &name));
                }
            }
            _ => return Err(anyhow!("{:?} is not defined.", &slot)),
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
                }
            }
            _ => unreachable!(),
        };

        Ok(())
    }

    fn mark_real(&mut self, slot: &Slot, is_real: bool) {
        if let Slot::Param(idx) = slot {
            if is_real {
                self.reals.insert(Loc::Param(*idx as u32));
            }
        }
    }

    fn compile_unary(&mut self, op: &str, dst: Reg, r: Reg) -> Result<()> {
        match op {
            "neg" => self.mir.neg(dst, r),
            "not" => self.mir.not(dst, r),
            "abs" => self.mir.abs(dst, r),
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
        assert!(n <= SLICE_CAP);

        if let Slot::Param(idx) = lhs {
            if is_real {
                self.reals.insert(Loc::Param(*idx as u32));
            }
        }

        if VirtualTable::from_str(op).is_ok() || op.starts_with("composer_") {
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
        } else if self.mir.config.is_intrinsic_unary(op) && n == 1 {
            self.load(reg(0), &args[0])?;
            self.compile_unary(op, reg(1), reg(0))?;
            self.save(reg(1), lhs)?;
        } else if self.mir.config.is_intrinsic_binary(op) && n == 2 {
            self.load(reg(0), &args[0])?;
            self.load(reg(1), &args[1])?;
            self.compile_binary(op, reg(2), reg(0), reg(1))?;
            self.save(reg(2), lhs)?;
        } else {
            for (i, arg) in args.iter().enumerate() {
                self.load(reg(0), arg)?;
                self.save(reg(0), &Slot::Arg(i))?;
            }

            let op = format!("${}", op);
            self.mir.call(&op, n)?;
            self.save(Reg::Ret, lhs)?;
            self.ft.insert(op.to_string());
        }

        Ok(())
    }
}

impl Composer for DirectTranslator {
    fn append_constant(&mut self, z: Complex<f64>) -> Result<usize> {
        self.consts.push(z.re);
        self.consts.push(z.im);

        if self.mir.config.is_complex() {
            Ok((self.consts.len() - 1) / 2)
        } else {
            Ok(self.consts.len() - 1)
        }
    }

    fn append_add(&mut self, lhs: &Slot, args: &[Slot], num_reals: usize) -> Result<()> {
        self.load(reg(0), &args[0])?;
        self.mark_real(&args[0], 0 < num_reals);

        for (i, arg) in args.iter().enumerate().skip(1) {
            self.load(reg(1), arg)?;
            self.mark_real(arg, i < num_reals);
            self.mir.plus(reg(0), reg(0), reg(1));
        }
        self.save(reg(0), lhs)?;

        Ok(())
    }

    fn append_mul(&mut self, lhs: &Slot, args: &[Slot], num_reals: usize) -> Result<()> {
        self.load(reg(0), &args[0])?;
        self.mark_real(&args[0], 0 < num_reals);

        let mut negate = false;

        for (i, arg) in args.iter().enumerate().skip(1) {
            if self.is_minus_one(arg) {
                negate = !negate;
            } else {
                self.load(reg(1), arg)?;
                self.mark_real(arg, i < num_reals);
                self.mir.times(reg(0), reg(0), reg(1));
            }
        }

        if negate {
            self.mir.neg(reg(0), reg(0));
        }

        self.save(reg(0), lhs)?;

        Ok(())
    }

    fn append_pow(&mut self, lhs: &Slot, arg: &Slot, p: i64, is_real: bool) -> Result<()> {
        self.load(reg(0), arg)?;
        self.mark_real(arg, is_real);

        match p {
            2 => self.mir.square(reg(1), reg(0)),
            3 => self.mir.cube(reg(1), reg(0)),
            -1 => self.mir.recip(reg(1), reg(0)),
            -2 => {
                self.mir.recip(reg(1), reg(0));
                self.mir.square(reg(1), reg(0))
            }
            -3 => {
                self.mir.recip(reg(1), reg(0));
                self.mir.cube(reg(1), reg(0))
            }
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
        Ok(())
    }

    fn append_assign(&mut self, lhs: &Slot, rhs: &Slot) -> Result<()> {
        self.load(reg(0), rhs)?;
        self.save(reg(0), lhs)?;
        Ok(())
    }

    fn append_label(&mut self, id: usize) -> Result<()> {
        let label = format!(".S{}", id);
        self.mir.set_label(&label);
        Ok(())
    }

    fn append_if_else(&mut self, cond: &Slot, id: usize) -> Result<()> {
        let label = format!(".S{}", id);
        self.load(reg(0), cond)?;
        self.mir.iszero(reg(1), reg(0));
        self.mir.branch_if(reg(1), &label, true);
        self.conditions.push(*cond);
        Ok(())
    }

    fn append_goto(&mut self, id: usize) -> Result<()> {
        let cond = self.conditions.pop().unwrap();
        let label = format!(".S{}", id);

        if !self.mir.config.symbolica() || !self.mir.config.use_simd() {
            self.mir.branch(&label);
        } else {
            self.load(reg(0), &cond)?;
            self.mir.iszero(reg(1), reg(0));
            self.mir.branch_if(reg(1), &label, false);
        }

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
        prog.builder.optimize_mir(&mut mir)?;
        let mut app = Application::with_mir(prog, self.reals.clone(), mir)?;
        app.prepare_simd();

        Ok(app)
    }
}

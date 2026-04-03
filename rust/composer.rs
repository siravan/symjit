use anyhow::{anyhow, Result};
use num_complex::Complex;
use std::collections::HashSet;

use crate::config::{Config, SLICE_CAP, SPILL_AREA};
use crate::defuns::Defuns;
use crate::expr::Expr;
use crate::instruction::{BuiltinSymbol, Instruction, Slot, SymbolicaModel};
use crate::mir::Mir;
use crate::model::{CellModel, Equation, Program, Variable};
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
    fn append_fun(
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
pub struct Transliterator {
    pub mir: Mir,
    pub consts: Vec<f64>,
    pub reals: HashSet<Loc>,
    pub num_params: usize,
    pub count_params: usize,
    pub count_temps: usize,
    pub count_outs: usize,
}

impl Transliterator {
    pub fn new(mut config: Config, df: Defuns) -> Transliterator {
        config.set_defuns(df);

        Transliterator {
            mir: Mir::new(config),
            consts: Vec::new(),
            reals: HashSet::new(),
            num_params: 0,
            count_params: 0,
            count_temps: 0,
            count_outs: 0,
        }
    }

    fn is_const(slot: &Slot) -> bool {
        matches!(slot, Slot::Const(..))
    }

    fn as_loc(&self, slot: &Slot) -> Option<Loc> {
        match slot {
            Slot::Out(idx) => Some(Loc::Mem(*idx as u32)),
            Slot::Param(idx) => Some(Loc::Param(*idx as u32)),
            Slot::Temp(idx) => Some(Loc::Stack((*idx + SPILL_AREA + SLICE_CAP) as u32)),
            _ => None,
        }
    }

    fn as_const(&self, slot: &Slot) -> Option<u32> {
        if let Slot::Const(idx) = slot {
            Some(*idx as u32)
        } else {
            None
        }
    }

    fn load(&mut self, dst: Reg, slot: &Slot) -> Result<()> {
        let k = if self.mir.config.is_complex() { 2 } else { 1 };

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
                    self.mir.load_const(reg(0), n);
                    self.mir.load_const(reg(1), n + 1);
                    self.mir.complex(dst, reg(0), reg(1));
                }
            }
            Slot::Param(idx) => {
                self.count_params = self.count_params.max(*idx);
                self.mir.load_param(dst, (*idx * k) as u32);
            }
            Slot::Out(idx) => {
                self.count_outs = self.count_outs.max(*idx);
                self.mir.load_mem(dst, (*idx * k) as u32);
            }
            Slot::Temp(idx) => {
                self.count_temps = self.count_temps.max(*idx);
                self.mir.load_stack(dst, (*idx * k) as u32);
            }
            _ => return Err(anyhow!("slot not defined")),
        }

        Ok(())
    }

    fn save(&mut self, src: Reg, slot: &Slot) {
        match slot {
            Slot::Out(idx) => self.mir.save_mem(src, *idx as u32),
            Slot::Temp(idx) => self.mir.save_stack(src, *idx as u32),
            _ => unreachable!(),
        }
    }

    fn mark_real(&mut self, slot: &Slot, is_real: bool) {
        if let Slot::Param(idx) = slot {
            if is_real {
                self.reals.insert(Loc::Param(*idx as u32));
            }
        }
    }
}

impl Composer for Transliterator {
    fn append_constant(&mut self, z: Complex<f64>) -> Result<usize> {
        self.consts.push(z.re);
        self.consts.push(z.im);
        Ok(self.consts.len() - 1)
    }

    fn append_add(&mut self, lhs: &Slot, args: &[Slot], num_reals: usize) -> Result<()> {
        self.load(reg(0), &args[0])?;
        self.mark_real(&args[0], 0 < num_reals);

        for i in 1..args.len() {
            self.load(reg(1), &args[i])?;
            self.mark_real(&args[i], i < num_reals);
            self.mir.plus(reg(0), reg(0), reg(1));
        }
        self.save(reg(0), lhs);

        Ok(())
    }

    fn append_mul(&mut self, lhs: &Slot, args: &[Slot], num_reals: usize) -> Result<()> {
        self.load(reg(0), &args[0])?;
        self.mark_real(&args[0], 0 < num_reals);

        for i in 1..args.len() {
            self.load(reg(1), &args[i])?;
            self.mark_real(&args[i], i < num_reals);
            self.mir.times(reg(0), reg(0), reg(1));
        }
        self.save(reg(0), lhs);

        Ok(())
    }

    fn append_pow(&mut self, lhs: &Slot, arg: &Slot, p: i64, is_real: bool) -> Result<()> {
        self.load(reg(0), arg)?;
        self.mark_real(&arg, is_real);

        match p {
            2 => self.mir.square(reg(0), reg(0)),
            3 => self.mir.cube(reg(0), reg(0)),
            -1 => self.mir.recip(reg(0), reg(0)),
            -2 => {
                self.mir.recip(reg(0), reg(0));
                self.mir.square(reg(0), reg(0))
            }
            -3 => {
                self.mir.recip(reg(0), reg(0));
                self.mir.cube(reg(0), reg(0))
            }
            _ => self.mir.powi(reg(0), reg(0), p as i32),
        }
        self.save(reg(0), lhs);

        Ok(())
    }

    fn append_powf(&mut self, lhs: &Slot, arg: &Slot, p: &Slot, is_real: bool) -> Result<()> {
        self.load(reg(0), arg)?;
        self.mark_real(&arg, is_real);

        self.load(reg(1), p)?;
        self.mir.setup_call_binary(reg(0), reg(1));
        self.mir.call("power", 2)?;
        self.save(Reg::Ret, lhs);
        Ok(())
    }

    fn append_assign(&mut self, lhs: &Slot, rhs: &Slot) -> Result<()> {
        self.load(reg(0), rhs)?;
        self.save(reg(0), lhs);
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
        self.mir.xor(reg(1), reg(1), reg(1));
        self.mir.eq(reg(2), reg(0), reg(1));
        self.mir.branch_if(reg(2), &label, false);
        Ok(())
    }

    fn append_goto(&mut self, id: usize) -> Result<()> {
        let label = format!(".S{}", id);
        self.mir.branch(&label);
        Ok(())
    }

    fn append_external_fun(&mut self, lhs: &Slot, op: &str, args: &[Slot]) -> Result<()> {
        unimplemented!()
    }

    fn append_fun(
        &mut self,
        lhs: &Slot,
        fun: &BuiltinSymbol,
        arg: &Slot,
        is_real: bool,
    ) -> Result<()> {
        self.load(reg(0), arg)?;
        self.mark_real(&arg, is_real);

        self.mir.setup_call_unary(reg(0));

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
            _ => return Err(anyhow!("function is not defined.")),
        };

        self.mir.call(op, 1)?;
        self.save(Reg::Ret, lhs);

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
        self.mir.xor(reg(1), reg(1), reg(1));
        self.mir.eq(reg(2), reg(0), reg(1));
        self.mir.save_stack(reg(2), 4);
        self.load(reg(0), true_val)?;
        self.load(reg(1), false_val)?;
        self.mir.ifelse(reg(2), reg(0), reg(1), Loc::Stack(4));
        self.save(reg(2), lhs);

        Ok(())
    }

    fn set_num_params(&mut self, num_params: usize) {
        self.num_params = num_params
    }

    fn compile(&mut self) -> Result<Application> {
        let params: Vec<Variable> = (0..=self.count_params.max(self.num_params.max(1) - 1))
            .map(|idx| Variable {
                name: format!("Param{}", idx),
            })
            .collect();

        let outs: Vec<Expr> = (0..=self.count_outs)
            .map(|idx| Expr::var(&format!("Out{}", idx)))
            .collect();

        let obs: Vec<Equation> = outs
            .iter()
            .map(|v| Equation {
                lhs: v.clone(),
                rhs: v.clone(),
            })
            .collect();

        let ml = CellModel {
            iv: Expr::var("$_").to_variable().unwrap(),
            params,
            states: Vec::new(),
            algs: Vec::new(),
            odes: Vec::new(),
            obs,
        };

        let mut prog = Program::new(&ml, self.mir.config.clone())?;

        for i in 0..self.count_temps {
            prog.builder
                .block()
                .create_tmp_named(&format!("__Temp{}", i));
        }

        prog.builder.consts = self.consts.clone();

        let mir: Mir = std::mem::take(&mut self.mir);

        println!("{:?}", &mir);

        let mut app = Application::with_mir(prog, self.reals.clone(), mir)?;
        app.prepare_simd();

        app.dump("test.bin", "simd");

        Ok(app)
    }
}

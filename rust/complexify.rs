use anyhow::{anyhow, Result};
use std::collections::HashSet;

use crate::code::Func;
use crate::config::Config;
use crate::defuns::Defuns;
use crate::generator::Generator;
use crate::mir::Mir;
use crate::symbol::Loc;
use crate::utils::Reg;

fn re(reg: Reg) -> Reg {
    match reg {
        Reg::Ret => Reg::Ret,
        Reg::Temp => Reg::Gen(0),
        Reg::Left => Reg::Left,
        Reg::Right => Reg::Gen(0),
        Reg::Gen(r) => Reg::Gen(4 + 2 * r),
        Reg::Static(_) => unreachable!(),
    }
}

fn im(reg: Reg) -> Reg {
    match reg {
        Reg::Ret => Reg::Temp,
        Reg::Temp => Reg::Gen(1),
        Reg::Left => Reg::Temp,
        Reg::Right => Reg::Gen(1),
        Reg::Gen(r) => Reg::Gen(4 + 2 * r + 1),
        Reg::Static(_) => unreachable!(),
    }
}

fn ϕ(r: Reg) -> usize {
    match r {
        Reg::Ret => 0,
        Reg::Temp => 1,
        Reg::Left => 0,
        Reg::Right => 1,
        Reg::Gen(dst) => dst as usize + 2,
        Reg::Static(..) => panic!("passing static registers to codegen"),
    }
}

pub struct Complexifier {
    mir: Mir,
    real_locs: HashSet<Loc>,
    real_regs: [bool; 32],
}

impl Complexifier {
    pub fn new(reals: &HashSet<Loc>, config: Config, df: &Defuns) -> Complexifier {
        let mut real_locs: HashSet<Loc> = HashSet::new();
        for loc in reals {
            if let Loc::Param(idx) = loc {
                real_locs.insert(Loc::Param(*idx * 2));
            }
        }

        Complexifier {
            mir: Mir::new(config, df),
            real_locs,
            real_regs: [false; 32],
        }
    }

    pub fn complexify(&mut self, mir: &Mir) -> Result<Mir> {
        self.mir.consts = mir.consts.clone();
        self.mir.labels = mir.labels.clone();

        mir.rerun(self)?;

        Ok(std::mem::take(&mut self.mir))
    }

    // temporary registers
    const T0: Reg = Reg::Gen(2);
    const T1: Reg = Reg::Gen(3);

    fn is_real_loc(&self, loc: Loc) -> bool {
        self.real_locs.contains(&loc)
    }

    fn is_real_reg(&self, s1: Reg) -> bool {
        self.real_regs[ϕ(s1)]
    }

    fn set_loc_real(&mut self, loc: Loc) {
        self.real_locs.insert(loc);
    }

    fn set_loc_complex(&mut self, loc: Loc) {
        self.real_locs.remove(&loc);
    }

    fn set_reg_real(&mut self, dst: Reg) {
        self.real_regs[ϕ(dst)] = true;
    }

    fn set_reg_complex(&mut self, dst: Reg) {
        self.real_regs[ϕ(dst)] = false;
    }

    fn copy_real(&mut self, dst: Reg, s1: Reg) {
        self.real_regs[ϕ(dst)] = self.is_real_reg(s1);
    }

    fn promote_real(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.real_regs[ϕ(dst)] = self.is_real_reg(s1) && self.is_real_reg(s2);
    }

    fn ensure_complex(&mut self, dst: Reg) {
        if self.is_real_reg(dst) {
            self.mir.xor(im(dst), im(dst), im(dst));
        }
        self.set_reg_complex(dst);
    }
}

impl Generator for Complexifier {
    fn count_shadows(&self) -> u8 {
        0
    }

    fn three_address(&self) -> bool {
        true
    }

    fn bytes(&mut self) -> Vec<u8> {
        Vec::new()
    }

    fn seal(&mut self) {}
    fn align(&mut self) {}

    fn set_label(&mut self, label: &str) {
        self.mir.set_label(label);
    }

    fn branch(&mut self, label: &str) {
        self.mir.branch(label);
    }

    fn branch_if(&mut self, cond: Reg, label: &str, is_else: bool) {
        self.mir.branch_if(re(cond), label, is_else);
    }

    /***********************************/
    fn fmov(&mut self, dst: Reg, s1: Reg) {
        self.mir.fmov(re(dst), re(s1));

        if !self.is_real_reg(s1) {
            self.mir.fmov(im(dst), im(s1));
        }

        self.copy_real(dst, s1);
    }

    fn fxchg(&mut self, s1: Reg, s2: Reg) {
        self.mir.fxchg(re(s1), re(s2));

        if !self.is_real_reg(s1) || !self.is_real_reg(s2) {
            self.ensure_complex(s1);
            self.ensure_complex(s2);
            self.mir.fxchg(im(s1), im(s2));
        }
    }

    fn load_const(&mut self, dst: Reg, idx: u32) {
        self.mir.load_const(re(dst), idx);
        self.set_reg_real(dst);
    }

    fn load_mem(&mut self, dst: Reg, idx: u32) {
        self.mir.load_mem(re(dst), idx);
        self.mir.load_mem(im(dst), idx + 1);
        self.set_reg_complex(dst);
    }

    fn save_mem(&mut self, s1: Reg, idx: u32) {
        self.ensure_complex(s1);
        self.mir.save_mem(re(s1), idx);
        self.mir.save_mem(im(s1), idx + 1);
    }

    fn load_param(&mut self, dst: Reg, idx: u32) {
        self.mir.load_param(re(dst), idx);

        if self.is_real_loc(Loc::Param(idx)) {
            self.set_reg_real(dst);
        } else {
            self.mir.load_param(im(dst), idx + 1);
            self.set_reg_complex(dst);
        }
    }

    fn load_stack(&mut self, dst: Reg, idx: u32) {
        self.mir.load_stack(re(dst), idx);

        if self.is_real_loc(Loc::Stack(idx)) {
            self.set_reg_real(dst);
        } else {
            self.mir.load_stack(im(dst), idx + 1);
            self.set_reg_complex(dst);
        }
    }

    fn save_stack(&mut self, s1: Reg, idx: u32) {
        self.mir.save_stack(re(s1), idx);

        if self.is_real_reg(s1) {
            self.set_loc_real(Loc::Stack(idx));
        } else {
            self.mir.save_stack(im(s1), idx + 1);
            self.set_loc_complex(Loc::Stack(idx));
        }
    }

    fn save_mem_result(&mut self, idx: u32) {
        self.save_mem(Reg::Ret, idx);
    }

    fn save_stack_result(&mut self, idx: u32) {
        self.save_stack(Reg::Ret, idx);
    }

    fn neg(&mut self, dst: Reg, s1: Reg) {
        self.mir.neg(re(dst), re(s1));

        if !self.is_real_reg(s1) {
            self.mir.neg(im(dst), im(s1));
        }

        self.copy_real(dst, s1);
    }

    fn abs(&mut self, dst: Reg, s1: Reg) {
        if self.is_real_reg(s1) {
            self.mir.abs(dst, s1);
            self.set_reg_real(dst);
        } else {
            self.mir.times(Self::T0, re(s1), re(s1));
            self.mir.times(Self::T1, im(s1), im(s1));
            self.mir.plus(re(dst), Self::T0, Self::T1);
            self.mir.root(re(dst), re(dst));
            self.mir.xor(im(dst), im(dst), im(dst));
            self.set_reg_complex(dst);
        }
    }

    fn root(&mut self, dst: Reg, s1: Reg) {
        self.ensure_complex(s1);
        self.mir.fmov(re(Reg::Ret), re(s1));
        self.mir.fmov(im(Reg::Ret), im(s1));
        self.mir.call("root", 1).unwrap();
        self.mir.fmov(re(dst), re(Reg::Ret));
        self.mir.fmov(im(dst), im(Reg::Ret));
        self.set_reg_complex(dst);
    }

    fn real_root(&mut self, dst: Reg, s1: Reg) {
        if self.is_real_reg(s1) {
            self.mir.root(re(dst), re(s1));
            self.set_reg_real(dst);
        } else {
            self.root(dst, s1);
        }
    }

    fn recip(&mut self, dst: Reg, s1: Reg) {
        if self.is_real_reg(s1) {
            self.mir.recip(dst, s1);
            self.set_reg_real(dst);
        } else {
            self.mir.times(Self::T0, re(s1), re(s1));
            self.mir.times(Self::T1, im(s1), im(s1));
            self.mir.plus(Self::T0, Self::T0, Self::T1);
            self.mir.divide(re(dst), re(s1), Self::T0);
            self.mir.divide(im(dst), im(s1), Self::T0);
            self.mir.neg(im(dst), im(dst));
            self.set_reg_complex(dst);
        }
    }

    fn round(&mut self, dst: Reg, s1: Reg) {
        self.mir.round(re(dst), re(s1));

        if !self.is_real_reg(s1) {
            self.mir.round(im(dst), im(s1));
        }

        self.copy_real(dst, s1);
    }

    fn floor(&mut self, dst: Reg, s1: Reg) {
        self.mir.floor(re(dst), re(s1));

        if !self.is_real_reg(s1) {
            self.mir.floor(im(dst), im(s1));
        }

        self.copy_real(dst, s1);
    }

    fn ceiling(&mut self, dst: Reg, s1: Reg) {
        self.mir.ceiling(re(dst), re(s1));

        if !self.is_real_reg(s1) {
            self.mir.ceiling(im(dst), im(s1));
        }

        self.copy_real(dst, s1);
    }

    fn trunc(&mut self, dst: Reg, s1: Reg) {
        self.mir.trunc(re(dst), re(s1));

        if !self.is_real_reg(s1) {
            self.mir.trunc(im(dst), im(s1));
        }

        self.copy_real(dst, s1);
    }

    fn frac(&mut self, dst: Reg, s1: Reg) {
        self.mir.frac(re(dst), re(s1));

        if !self.is_real_reg(s1) {
            self.mir.frac(im(dst), im(s1));
        }

        self.copy_real(dst, s1);
    }

    fn plus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.plus(re(dst), re(s1), re(s2));

        if self.is_real_reg(s1) && !self.is_real_reg(s2) {
            self.mir.fmov(im(dst), im(s2));
        } else if !self.is_real_reg(s1) && self.is_real_reg(s2) {
            self.mir.fmov(im(dst), im(s1));
        } else if !self.is_real_reg(s1) && !self.is_real_reg(s2) {
            self.mir.plus(im(dst), im(s1), im(s2));
        }

        self.promote_real(dst, s1, s2);
    }

    fn minus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.minus(re(dst), re(s1), re(s2));

        if self.is_real_reg(s1) && !self.is_real_reg(s2) {
            self.mir.neg(im(dst), im(s2));
        } else if !self.is_real_reg(s1) && self.is_real_reg(s2) {
            self.mir.fmov(im(dst), im(s1));
        } else if !self.is_real_reg(s1) && !self.is_real_reg(s2) {
            self.mir.minus(im(dst), im(s1), im(s2));
        }

        self.promote_real(dst, s1, s2);
    }

    fn times(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        if self.is_real_reg(s1) && self.is_real_reg(s2) {
            self.mir.times(re(dst), re(s1), re(s2));
        } else if self.is_real_reg(s1) && !self.is_real_reg(s2) {
            self.mir.times(im(dst), re(s1), im(s2));
            self.mir.times(re(dst), re(s1), re(s2));
        } else if !self.is_real_reg(s1) && self.is_real_reg(s2) {
            self.mir.times(im(dst), im(s1), re(s2));
            self.mir.times(re(dst), re(s1), re(s2));
        } else {
            self.mir.times(Self::T0, re(s1), re(s2));
            self.mir.times(Self::T1, im(s1), im(s2));
            self.mir.minus(Self::T0, Self::T0, Self::T1);

            self.mir.times(Self::T1, re(s1), im(s2));
            self.mir.times(im(dst), im(s1), re(s2));
            self.mir.plus(im(dst), im(dst), Self::T1);

            self.mir.fmov(re(dst), Self::T0);
        }

        self.promote_real(dst, s1, s2);
    }

    fn divide(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        // note: we can use Reg::Temp here because the
        // operators using Reg::Temp do not use division,
        // except for powi_mod which needs to be modified (TODO)
        let t = re(Reg::Temp);

        if self.is_real_reg(s1) && self.is_real_reg(s2) {
            self.mir.divide(re(dst), re(s1), re(s2));
        } else if !self.is_real_reg(s1) && self.is_real_reg(s2) {
            self.mir.divide(im(dst), im(s1), re(s2));
            self.mir.divide(re(dst), re(s1), re(s2));
        } else if self.is_real_reg(s1) && !self.is_real_reg(s2) {
            self.mir.times(Self::T0, re(s2), re(s2));
            self.mir.times(Self::T1, im(s2), im(s2));
            self.mir.plus(t, Self::T0, Self::T1);

            self.mir.times(Self::T0, re(s1), re(s2));
            self.mir.times(Self::T1, re(s1), im(s2));
            self.mir.neg(im(dst), Self::T1);

            self.mir.divide(im(dst), im(dst), t);
            self.mir.divide(re(dst), Self::T0, t);
        } else {
            self.mir.times(Self::T0, re(s2), re(s2));
            self.mir.times(Self::T1, im(s2), im(s2));
            self.mir.plus(t, Self::T0, Self::T1);

            self.mir.times(Self::T0, re(s1), re(s2));
            self.mir.times(Self::T1, im(s1), im(s2));
            self.mir.plus(Self::T0, Self::T0, Self::T1);

            self.mir.times(Self::T1, re(s1), im(s2));
            self.mir.times(im(dst), im(s1), re(s2));
            self.mir.minus(im(dst), im(dst), Self::T1);

            self.mir.divide(im(dst), im(dst), t);
            self.mir.divide(re(dst), Self::T0, t);
        }

        self.promote_real(dst, s1, s2);
    }

    fn real(&mut self, dst: Reg, s1: Reg) {
        self.mir.fmov(re(dst), re(s1));
        self.set_reg_real(dst);
    }

    fn imaginary(&mut self, dst: Reg, s1: Reg) {
        self.mir.fmov(re(dst), im(s1));
        self.set_reg_real(dst);
    }

    fn conjugate(&mut self, dst: Reg, s1: Reg) {
        self.mir.fmov(re(dst), re(s1));

        if !self.is_real_reg(s1) {
            self.mir.fmov(im(dst), im(s1));
            self.mir.neg(im(dst), im(dst));
        }

        self.copy_real(dst, s1);
    }

    fn complex(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        // Important! The order of these two statements matters.
        // The imaginary part needs to be set first to prevent
        // conflict if dst == s2.
        self.mir.fmov(im(dst), re(s2));
        self.mir.fmov(re(dst), re(s1));
        self.set_reg_complex(dst);
    }

    fn gt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.gt(re(dst), re(s1), re(s2));
        self.mir.fmov(im(dst), re(dst));
        self.set_reg_complex(dst);
    }

    fn geq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.geq(re(dst), re(s1), re(s2));
        self.mir.fmov(im(dst), re(dst));
        self.set_reg_complex(dst);
    }

    fn lt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.lt(re(dst), re(s1), re(s2));
        self.mir.fmov(im(dst), re(dst));
        self.set_reg_complex(dst);
    }

    fn leq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.leq(re(dst), re(s1), re(s2));
        self.mir.fmov(im(dst), re(dst));
        self.set_reg_complex(dst);
    }

    fn eq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.eq(re(dst), re(s1), re(s2));
        self.mir.fmov(im(dst), re(dst));
        self.set_reg_complex(dst);
    }

    fn neq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.neq(re(dst), re(s1), re(s2));
        self.mir.fmov(im(dst), re(dst));
        self.set_reg_complex(dst);
    }

    fn and(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.and(re(dst), re(s1), re(s2));

        if self.is_real_reg(s1) != self.is_real_reg(s2) {
            self.mir.xor(im(dst), im(dst), im(dst));
        } else if !self.is_real_reg(s1) && !self.is_real_reg(s2) {
            self.mir.and(im(dst), im(s1), im(s2));
        }

        self.promote_real(dst, s1, s2);
    }

    fn andnot(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.andnot(re(dst), re(s1), re(s2));

        if self.is_real_reg(s1) && !self.is_real_reg(s2) {
            self.mir.fmov(im(dst), im(s2));
        } else if !self.is_real_reg(s1) && self.is_real_reg(s2) {
            self.mir.xor(im(dst), im(dst), im(dst));
        } else if !self.is_real_reg(s1) && !self.is_real_reg(s2) {
            self.mir.andnot(im(dst), im(s1), im(s2));
        }

        self.promote_real(dst, s1, s2);
    }

    fn or(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.or(re(dst), re(s1), re(s2));

        if self.is_real_reg(s1) != self.is_real_reg(s2) {
            self.mir.fmov(im(dst), im(s2));
        } else if !self.is_real_reg(s1) && self.is_real_reg(s2) {
            self.mir.fmov(im(dst), im(s1));
        } else if !self.is_real_reg(s1) && !self.is_real_reg(s2) {
            self.mir.or(im(dst), im(s1), im(s2));
        }

        self.promote_real(dst, s1, s2);
    }

    fn xor(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.xor(re(dst), re(s1), re(s2));

        if self.is_real_reg(s1) && !self.is_real_reg(s2) {
            self.mir.fmov(im(dst), im(s2));
        } else if !self.is_real_reg(s1) && self.is_real_reg(s2) {
            self.mir.fmov(im(dst), im(s1));
        } else if !self.is_real_reg(s1) && !self.is_real_reg(s2) {
            self.mir.xor(im(dst), im(s1), im(s2));
        }

        self.promote_real(dst, s1, s2);
    }

    fn not(&mut self, dst: Reg, s1: Reg) {
        self.mir.not(re(dst), re(s1));

        if !self.is_real_reg(s1) {
            self.mir.not(im(dst), im(s1));
        }

        self.copy_real(dst, s1);
    }

    fn fused_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        self.times(Reg::Ret, s1, s2);
        self.plus(dst, Reg::Ret, s3);
    }

    fn fused_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        self.times(Reg::Ret, s1, s2);
        self.minus(dst, Reg::Ret, s3);
    }

    fn fused_neg_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        self.times(Reg::Ret, s1, s2);
        self.minus(dst, s3, Reg::Ret);
    }

    fn fused_neg_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        self.times(Reg::Ret, s1, s2);
        self.plus(dst, Reg::Ret, s3);
        self.neg(dst, dst);
    }

    fn add_consts(&mut self, consts: &[f64]) {
        self.mir.add_consts(consts);
    }

    fn add_func(&mut self, _f: &str, _p: Func) {}

    fn call(&mut self, op: &str, num_args: usize) -> Result<()> {
        self.mir.call(op, num_args)
    }

    fn call_complex(&mut self, op: &str, num_args: usize) -> Result<()> {
        self.ensure_complex(Reg::Ret);

        match num_args {
            1 => {}
            2 => self.ensure_complex(Reg::Temp),
            _ => return Err(anyhow!("complex functions expect 1 or 2 arguments.")),
        }

        self.mir.call(op, num_args)
    }

    fn prologue_fast(&mut self, _cap: usize, _count_states: usize, _count_obs: usize) {}
    fn epilogue_fast(
        &mut self,
        _cap: usize,
        _count_states: usize,
        _count_obs: usize,
        _idx_ret: i32,
    ) {
    }

    fn prologue_indirect(
        &mut self,
        _cap: usize,
        _count_states: usize,
        _count_obs: usize,
        _count_params: usize,
    ) {
    }
    fn epilogue_indirect(
        &mut self,
        _cap: usize,
        _count_states: usize,
        _count_obs: usize,
        _count_params: usize,
    ) {
    }

    fn save_used_registers(&mut self, _used: &[u8]) {}
    fn load_used_registers(&mut self, _used: &[u8]) {}

    fn ifelse(&mut self, dst: Reg, true_val: Reg, false_val: Reg, idx: u32) {
        let loc = Loc::Stack(idx);
        self.mir.ifelse(re(dst), re(true_val), re(false_val), loc);
        self.mir.ifelse(im(dst), im(true_val), im(false_val), loc);
        self.promote_real(dst, true_val, false_val);
    }
}

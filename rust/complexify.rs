use anyhow::{anyhow, Result};
use std::collections::HashSet;

use crate::code::Func;
use crate::config::Config;
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

enum Types {
    RR,
    CR,
    RC,
    CC,
}

pub struct Complexifier {
    mir: Mir,
    real_locs: HashSet<Loc>,
    real_regs: [bool; 32],
}

impl Complexifier {
    pub fn new(reals: &HashSet<Loc>, config: Config) -> Complexifier {
        let mut real_locs: HashSet<Loc> = HashSet::new();
        for loc in reals {
            if let Loc::Param(idx) = loc {
                real_locs.insert(Loc::Param(*idx * 2));
            }
        }

        Complexifier {
            mir: Mir::new(config),
            real_locs,
            real_regs: [false; 32],
        }
    }

    pub fn complexify(&mut self, mir: &Mir) -> Result<Mir> {
        self.mir.consts = mir.consts.clone();
        // self.mir.labels = mir.labels.clone();
        mir.rerun(self)?;
        self.mir.populate_labels();
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

    fn types(&self, s1: Reg, s2: Reg) -> Types {
        if self.is_real_reg(s1) && self.is_real_reg(s2) {
            Types::RR
        } else if self.is_real_reg(s1) && !self.is_real_reg(s2) {
            Types::RC
        } else if !self.is_real_reg(s1) && self.is_real_reg(s2) {
            Types::CR
        } else {
            Types::CC
        }
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
        // self.mir.load_mem(re(dst), idx);
        // self.mir.load_mem(im(dst), idx + 1);
        self.mir.load_mem_complex(re(dst), im(dst), idx);
        self.set_reg_complex(dst);
    }

    fn save_mem(&mut self, s1: Reg, idx: u32) {
        self.ensure_complex(s1);
        // self.mir.save_mem(re(s1), idx);
        // self.mir.save_mem(im(s1), idx + 1);
        self.mir.save_mem_complex(re(s1), im(s1), idx);
    }

    fn load_param(&mut self, dst: Reg, idx: u32) {
        if self.is_real_loc(Loc::Param(idx)) {
            self.mir.load_param(re(dst), idx);
            self.set_reg_real(dst);
        } else {
            // self.mir.load_param(re(dst), idx);
            // self.mir.load_param(im(dst), idx + 1);
            self.mir.load_param_complex(re(dst), im(dst), idx);
            self.set_reg_complex(dst);
        }
    }

    fn load_stack(&mut self, dst: Reg, idx: u32) {
        if self.is_real_loc(Loc::Stack(idx)) {
            self.mir.load_stack(re(dst), idx);
            self.set_reg_real(dst);
        } else {
            // self.mir.load_stack(re(dst), idx);
            // self.mir.load_stack(im(dst), idx + 1);
            self.mir.load_stack_complex(re(dst), im(dst), idx);
            self.set_reg_complex(dst);
        }
    }

    fn save_stack(&mut self, s1: Reg, idx: u32) {
        if self.is_real_reg(s1) {
            self.mir.save_stack(re(s1), idx);
            self.set_loc_real(Loc::Stack(idx));
        } else {
            // self.mir.save_stack(re(s1), idx);
            // self.mir.save_stack(im(s1), idx + 1);
            self.mir.save_stack_complex(re(s1), im(s1), idx);
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
            self.mir.recip(re(dst), re(s1));
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
        match self.types(s1, s2) {
            Types::RC => {
                self.mir.plus(re(dst), re(s1), re(s2));
                self.mir.fmov(im(dst), im(s2));
            }
            Types::CR => {
                self.mir.plus(re(dst), re(s1), re(s2));
                self.mir.fmov(im(dst), im(s1));
            }
            Types::CC => {
                //self.mir.plus(im(dst), im(s1), im(s2)),
                self.mir
                    .plus_complex(re(dst), im(dst), re(s1), im(s1), re(s2), im(s2));
            }
            Types::RR => {
                self.mir.plus(re(dst), re(s1), re(s2));
            }
        }

        self.promote_real(dst, s1, s2);
    }

    fn minus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        match self.types(s1, s2) {
            Types::RC => {
                self.mir.minus(re(dst), re(s1), re(s2));
                self.mir.neg(im(dst), im(s2));
            }
            Types::CR => {
                self.mir.minus(re(dst), re(s1), re(s2));
                self.mir.fmov(im(dst), im(s1));
            }
            Types::CC => {
                //self.mir.minus(im(dst), im(s1), im(s2)),
                self.mir
                    .minus_complex(re(dst), im(dst), re(s1), im(s1), re(s2), im(s2));
            }
            Types::RR => {
                self.mir.minus(re(dst), re(s1), re(s2));
            }
        }

        self.promote_real(dst, s1, s2);
    }

    fn times(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        match self.types(s1, s2) {
            Types::RR => self.mir.times(re(dst), re(s1), re(s2)),
            Types::RC => {
                self.mir.times(im(dst), re(s1), im(s2));
                self.mir.times(re(dst), re(s1), re(s2));
            }
            Types::CR => {
                self.mir.times(im(dst), im(s1), re(s2));
                self.mir.times(re(dst), re(s1), re(s2));
            }
            Types::CC => {
                /*
                self.mir.times(Self::T0, re(s1), re(s2));
                self.mir.times(Self::T1, im(s1), im(s2));
                self.mir.minus(Self::T0, Self::T0, Self::T1);

                self.mir.times(Self::T1, re(s1), im(s2));
                self.mir.times(im(dst), im(s1), re(s2));
                self.mir.plus(im(dst), im(dst), Self::T1);

                self.mir.fmov(re(dst), Self::T0);
                */
                self.mir
                    .times_complex(re(dst), im(dst), re(s1), im(s1), re(s2), im(s2));
            }
        }

        self.promote_real(dst, s1, s2);
    }

    fn divide(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        // note: we can use Reg::Temp here because the
        // operators using Reg::Temp do not use division,
        // except for powi_mod which needs to be modified (TODO)
        let t = re(Reg::Temp);

        match self.types(s1, s2) {
            Types::RR => self.mir.divide(re(dst), re(s1), re(s2)),
            Types::CR => {
                self.mir.divide(im(dst), im(s1), re(s2));
                self.mir.divide(re(dst), re(s1), re(s2));
            }
            Types::RC => {
                self.mir.times(Self::T0, re(s2), re(s2));
                self.mir.times(Self::T1, im(s2), im(s2));
                self.mir.plus(t, Self::T0, Self::T1);

                self.mir.times(Self::T0, re(s1), re(s2));
                self.mir.times(Self::T1, re(s1), im(s2));
                self.mir.neg(im(dst), Self::T1);

                self.mir.divide(im(dst), im(dst), t);
                self.mir.divide(re(dst), Self::T0, t);
            }
            Types::CC => {
                /*
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
                */
                self.mir
                    .divide_complex(re(dst), im(dst), re(s1), im(s1), re(s2), im(s2));
            }
        }

        self.promote_real(dst, s1, s2);
    }

    fn times_complex(&mut self, xd: Reg, yd: Reg, x1: Reg, y1: Reg, x2: Reg, y2: Reg) -> bool {
        unreachable!()
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
        self.set_reg_real(dst);
    }

    fn geq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.geq(re(dst), re(s1), re(s2));
        self.set_reg_real(dst);
    }

    fn lt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.lt(re(dst), re(s1), re(s2));
        self.set_reg_real(dst);
    }

    fn leq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.leq(re(dst), re(s1), re(s2));
        self.set_reg_real(dst);
    }

    fn eq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.eq(re(dst), re(s1), re(s2));
        self.set_reg_real(dst);
    }

    fn neq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.neq(re(dst), re(s1), re(s2));
        self.set_reg_real(dst);
    }

    fn and(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        match self.types(s1, s2) {
            Types::RR => {}
            Types::RC => self.mir.and(im(dst), re(s1), im(s2)),
            Types::CR => self.mir.and(im(dst), im(s1), re(s2)),
            Types::CC => self.mir.and(im(dst), im(s1), im(s2)),
        }

        self.mir.and(re(dst), re(s1), re(s2));
        self.promote_real(dst, s1, s2);
    }

    fn andnot(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        match self.types(s1, s2) {
            Types::RR => {}
            Types::RC => self.mir.andnot(im(dst), re(s1), im(s2)),
            Types::CR => self.mir.andnot(im(dst), im(s1), re(s2)),
            Types::CC => self.mir.andnot(im(dst), im(s1), im(s2)),
        }

        self.mir.andnot(re(dst), re(s1), re(s2));
        self.promote_real(dst, s1, s2);
    }

    fn or(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        match self.types(s1, s2) {
            Types::RR => {}
            Types::RC => self.mir.or(im(dst), re(s1), im(s2)),
            Types::CR => self.mir.or(im(dst), im(s1), re(s2)),
            Types::CC => self.mir.or(im(dst), im(s1), im(s2)),
        }

        self.mir.or(re(dst), re(s1), re(s2));
        self.promote_real(dst, s1, s2);
    }

    fn xor(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        match self.types(s1, s2) {
            Types::RR => {}
            Types::RC => self.mir.xor(im(dst), re(s1), im(s2)),
            Types::CR => self.mir.xor(im(dst), im(s1), re(s2)),
            Types::CC => self.mir.xor(im(dst), im(s1), im(s2)),
        }

        self.mir.xor(re(dst), re(s1), re(s2));
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

    fn fuse_load_math(&mut self) {}

    fn ifelse(&mut self, dst: Reg, true_val: Reg, false_val: Reg, idx: u32) {
        let loc = Loc::Stack(idx);

        self.ensure_complex(true_val);
        self.ensure_complex(false_val);

        self.mir.ifelse(re(dst), re(true_val), re(false_val), loc);
        self.mir.ifelse(im(dst), im(true_val), im(false_val), loc);

        self.set_reg_complex(dst);
        // self.promote_real(dst, true_val, false_val);
    }
}

/*  Complex Genetic Functions */

impl Complexifier {
    pub fn generic_complex_plus(
        ir: &mut dyn Generator,
        xd: Reg,
        yd: Reg,
        x1: Reg,
        y1: Reg,
        x2: Reg,
        y2: Reg,
    ) {
        ir.plus(xd, x1, x2);
        ir.plus(yd, y1, y2);
    }

    pub fn generic_complex_minus(
        ir: &mut dyn Generator,
        xd: Reg,
        yd: Reg,
        x1: Reg,
        y1: Reg,
        x2: Reg,
        y2: Reg,
    ) {
        ir.minus(xd, x1, x2);
        ir.minus(yd, y1, y2);
    }

    pub fn generic_complex_times(
        ir: &mut dyn Generator,
        xd: Reg,
        yd: Reg,
        x1: Reg,
        y1: Reg,
        x2: Reg,
        y2: Reg,
    ) {
        ir.times(Self::T0, x1, x2);
        ir.times(Self::T1, y1, y2);
        ir.minus(Self::T0, Self::T0, Self::T1);

        ir.times(Self::T1, x1, y2);
        ir.times(yd, y1, x2);
        ir.plus(yd, yd, Self::T1);

        ir.fmov(xd, Self::T0);
    }

    pub fn generic_complex_divide(
        ir: &mut dyn Generator,
        xd: Reg,
        yd: Reg,
        x1: Reg,
        y1: Reg,
        x2: Reg,
        y2: Reg,
    ) {
        let t = re(Reg::Temp);

        ir.times(Self::T0, x2, x2);
        ir.times(Self::T1, y2, y2);
        ir.plus(t, Self::T0, Self::T1);

        ir.times(Self::T0, x1, x2);
        ir.times(Self::T1, y1, y2);
        ir.plus(Self::T0, Self::T0, Self::T1);

        ir.times(Self::T1, x1, y2);
        ir.times(yd, y1, x2);
        ir.minus(yd, yd, Self::T1);

        ir.divide(yd, yd, t);
        ir.divide(xd, Self::T0, t);
    }
}

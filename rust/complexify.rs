use anyhow::{anyhow, Result};

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

pub struct Complexifier {
    mir: Mir,
}

impl Complexifier {
    pub fn new(config: Config) -> Complexifier {
        Complexifier {
            mir: Mir::new(config, &Defuns::new()),
        }
    }

    pub fn complexify(&mut self, mir: &Mir) -> Result<Mir> {
        if mir.df.len() != 0 {
            return Err(anyhow!(
                "Complex functions do not support user-defined functions"
            ));
        }

        self.mir.consts = mir.consts.clone();
        self.mir.labels = mir.labels.clone();

        mir.rerun(self)?;

        Ok(self.mir.clone())
    }

    // temporary registers
    const T0: Reg = Reg::Gen(2);
    const T1: Reg = Reg::Gen(3);
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

    fn branch_if(&mut self, cond: Reg, label: &str) {
        self.mir.branch_if(re(cond), label);
    }

    /***********************************/
    fn fmov(&mut self, dst: Reg, s1: Reg) {
        self.mir.fmov(re(dst), re(s1));
        self.mir.fmov(im(dst), im(s1));
    }

    fn fxchg(&mut self, dst: Reg, s1: Reg) {
        self.mir.fxchg(re(dst), re(s1));
        self.mir.fxchg(im(dst), im(s1));
    }

    fn load_const(&mut self, dst: Reg, idx: u32) {
        // TODO: loading complex constants
        self.mir.load_const(re(dst), idx);
        self.mir.xor(im(dst), im(dst), im(dst));
    }

    fn load_mem(&mut self, dst: Reg, idx: u32) {
        self.mir.load_mem(re(dst), idx);
        self.mir.load_mem(im(dst), idx + 1);
    }

    fn save_mem(&mut self, dst: Reg, idx: u32) {
        self.mir.save_mem(re(dst), idx);
        self.mir.save_mem(im(dst), idx + 1);
    }

    fn load_param(&mut self, dst: Reg, idx: u32) {
        self.mir.load_param(re(dst), idx);
        self.mir.load_param(im(dst), idx + 1);
    }

    fn load_stack(&mut self, dst: Reg, idx: u32) {
        self.mir.load_stack(re(dst), idx);
        self.mir.load_stack(im(dst), idx + 1);
    }

    fn save_stack(&mut self, dst: Reg, idx: u32) {
        self.mir.save_stack(re(dst), idx);
        self.mir.save_stack(im(dst), idx + 1);
    }

    fn save_mem_result(&mut self, idx: u32) {
        self.save_mem(Reg::Ret, idx);
    }

    fn save_stack_result(&mut self, idx: u32) {
        self.save_stack(Reg::Ret, idx);
    }

    fn neg(&mut self, dst: Reg, s1: Reg) {
        self.mir.neg(re(dst), re(s1));
        self.mir.neg(im(dst), im(s1));
    }

    fn abs(&mut self, dst: Reg, s1: Reg) {
        self.mir.times(Self::T0, re(s1), re(s1));
        self.mir.times(Self::T1, im(s1), im(s1));
        self.mir.plus(re(dst), Self::T0, Self::T1);
        self.mir.root(re(dst), re(dst));
        self.mir.xor(im(dst), im(dst), im(dst));
    }

    fn root(&mut self, _dst: Reg, _s1: Reg) {
        // complex root compiles into a function call
    }

    fn recip(&mut self, dst: Reg, s1: Reg) {
        self.mir.times(Self::T0, re(s1), re(s1));
        self.mir.times(Self::T1, im(s1), im(s1));
        self.mir.plus(Self::T0, Self::T0, Self::T1);
        self.mir.divide(re(dst), re(s1), Self::T0);
        self.mir.divide(im(dst), im(s1), Self::T0);
        self.mir.neg(im(dst), im(dst));
    }

    fn round(&mut self, dst: Reg, s1: Reg) {
        self.mir.round(re(dst), re(s1));
        self.mir.round(im(dst), im(s1));
    }

    fn floor(&mut self, dst: Reg, s1: Reg) {
        self.mir.floor(re(dst), re(s1));
        self.mir.floor(im(dst), im(s1));
    }

    fn ceiling(&mut self, dst: Reg, s1: Reg) {
        self.mir.ceiling(re(dst), re(s1));
        self.mir.ceiling(im(dst), im(s1));
    }

    fn trunc(&mut self, dst: Reg, s1: Reg) {
        self.mir.trunc(re(dst), re(s1));
        self.mir.trunc(im(dst), im(s1));
    }

    fn frac(&mut self, dst: Reg, s1: Reg) {
        self.mir.frac(re(dst), re(s1));
        self.mir.frac(im(dst), im(s1));
    }

    fn plus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.plus(re(dst), re(s1), re(s2));
        self.mir.plus(im(dst), im(s1), im(s2));
    }

    fn minus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.minus(re(dst), re(s1), re(s2));
        self.mir.minus(im(dst), im(s1), im(s2));
    }

    fn times(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.times(Self::T0, re(s1), re(s2));
        self.mir.times(Self::T1, im(s1), im(s2));
        self.mir.minus(Self::T0, Self::T0, Self::T1);

        self.mir.times(Self::T1, re(s1), im(s2));
        self.mir.times(im(dst), im(s1), re(s2));
        self.mir.plus(im(dst), im(dst), Self::T1);

        self.mir.fmov(re(dst), Self::T0);
    }

    fn divide(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        // note: we can use Reg::Temp here because the
        // operators using Reg::Temp do not use division,
        // except for powi_mod which needs to be modified (TODO)
        let t = re(Reg::Temp);

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

    fn real(&mut self, dst: Reg, s1: Reg) {
        self.mir.fmov(re(dst), re(s1));
        self.mir.xor(im(dst), im(dst), im(dst));
    }

    fn imaginary(&mut self, dst: Reg, s1: Reg) {
        self.mir.fmov(re(dst), im(s1));
        self.mir.xor(im(dst), im(dst), im(dst));
    }

    fn conjugate(&mut self, dst: Reg, s1: Reg) {
        self.mir.fmov(re(dst), re(s1));
        self.mir.fmov(im(dst), im(s1));
        self.mir.neg(im(dst), im(dst));
    }

    fn gt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.gt(re(dst), re(s1), re(s2));
        self.mir.fmov(im(dst), re(dst));
    }

    fn geq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.geq(re(dst), re(s1), re(s2));
        self.mir.fmov(im(dst), re(dst));
    }

    fn lt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.lt(re(dst), re(s1), re(s2));
        self.mir.fmov(im(dst), re(dst));
    }

    fn leq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.leq(re(dst), re(s1), re(s2));
        self.mir.fmov(im(dst), re(dst));
    }

    fn eq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.eq(re(dst), re(s1), re(s2));
        self.mir.fmov(im(dst), re(dst));
    }

    fn neq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.neq(re(dst), re(s1), re(s2));
        self.mir.fmov(im(dst), re(dst));
    }

    fn and(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.and(re(dst), re(s1), re(s2));
        self.mir.and(im(dst), im(s1), im(s2));
    }

    fn andnot(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.andnot(re(dst), re(s1), re(s2));
        self.mir.andnot(im(dst), im(s1), im(s2));
    }

    fn or(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.or(re(dst), re(s1), re(s2));
        self.mir.or(im(dst), im(s1), im(s2));
    }

    fn xor(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.xor(re(dst), re(s1), re(s2));
        self.mir.xor(im(dst), im(s1), im(s2));
    }

    fn not(&mut self, dst: Reg, s1: Reg) {
        self.mir.not(re(dst), re(s1));
        self.mir.not(im(dst), im(s1));
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

    fn prologue_indirect(&mut self, _cap: usize, _count_states: usize, _count_obs: usize) {}
    fn epilogue_indirect(&mut self, _cap: usize, _count_states: usize, _count_obs: usize) {}

    fn save_used_registers(&mut self, _used: &[u8]) {}
    fn load_used_registers(&mut self, _used: &[u8]) {}

    fn ifelse(&mut self, dst: Reg, true_val: Reg, false_val: Reg, idx: u32) {
        let loc = Loc::Stack(idx);
        self.mir.ifelse(re(dst), re(true_val), re(false_val), loc);
        self.mir.ifelse(im(dst), im(true_val), im(false_val), loc);
    }
}

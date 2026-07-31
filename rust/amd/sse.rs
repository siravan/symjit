use super::super::code::Func;
use super::super::config::{Config, KernelType, ABI_AREA};
use super::super::generator::{FuncletType, Generator, StackRegions};
use super::super::symbol::Loc;
use super::super::utils::align_stack;
use super::super::utils::{is_external_func, DataType, Reg};
use anyhow::Result;

use super::asm::{Amd, RoundingMode};
use super::*;

const RET: u8 = 0;
const REG_SIZE: u32 = 8;

macro_rules! binop {
    ($self:ident, $sse:ident, $dst:expr, $s1: expr, $s2: expr, $com:ident) => {{
        let (x, y) = $self.shrink($dst, $s1, $s2, $com);
        $self.amd.$sse(ϕ(x), ϕ(y));
    }};
}

macro_rules! uniop {
    ($self:ident, $sse:ident, $dst:expr, $s1: expr) => {
        $self.amd.$sse(ϕ($dst), ϕ($s1));
    };
}

macro_rules! roundop {
    ($self:ident, $dst:expr, $s1: expr, $mode: expr) => {
        $self.amd.roundsd(ϕ($dst), ϕ($s1), $mode);
    };
}

pub struct AmdSSEGenerator {
    amd: Amd,
    config: Config,
    last_load: usize,
}

impl AmdSSEGenerator {
    pub fn new(config: Config) -> AmdSSEGenerator {
        AmdSSEGenerator {
            amd: Amd::new(DataType::F64),
            config,
            last_load: 0,
        }
    }

    fn append_quad(&mut self, u: u64) {
        self.amd.a.append_quad(u);
    }

    fn apply_jumps(&mut self) {
        self.amd.a.apply_jumps();
    }

    /*
        shrink is a helper function used to generate
        SSE codes from 3-address inputs.

        IMPORTANT! this function can overwrite the values of
        a and/or b. Therefore, cannot assume a and b are intact
        after calling this function.
    */
    fn shrink(&mut self, dst: Reg, s1: Reg, s2: Reg, commutative: bool) -> (Reg, Reg) {
        if dst == s1 {
            (dst, s2)
        } else if dst == s2 {
            // difficult case: dst == b, dst != a
            if !commutative {
                self.fxchg(s1, s2);
            };
            (dst, s1)
        } else {
            // dst != a, dst != b, a ?= b
            self.fmov(dst, s1);
            (dst, s2)
        }
    }

    fn load_const_by_name(&mut self, dst: Reg, label: &str) {
        self.amd.movsd_xmm_label(ϕ(dst), label);
    }

    fn call_external(&mut self, op: &str, num_args: usize) -> Result<()> {
        let cap = ABI_AREA as u32;

        self.amd.mov_reg_label(ARGS[0], &format!("_env_{}_", op));
        self.amd.lea_mem(ARGS[1], STACK, (cap * REG_SIZE) as i32);
        self.amd.mov_imm(ARGS[2], num_args as u32);
        self.amd.lea_mem(ARGS[3], SP, 4 * REG_SIZE as i32);

        self.amd.call_indirect(&format!("_func_{}_", op));
        self.load_stack(Reg::Ret, 4);

        if self.config.is_complex() {
            self.load_stack(Reg::Temp, 5);
        }

        Ok(())
    }

    fn predefined_consts(&mut self) {
        self.align();
        predefined_consts(&mut self.amd);
    }
}

impl Generator for AmdSSEGenerator {
    fn bytes(&mut self) -> Vec<u8> {
        self.amd.a.bytes()
    }

    fn count_shadows(&self) -> u8 {
        if cfg!(target_family = "windows") {
            4 // xmm2-xmm5
        } else {
            14 // xmm2-xmm15
        }
    }

    fn three_address(&self) -> bool {
        false
    }

    fn support_funclet(&self) -> FuncletType {
        FuncletType::Complex
    }

    fn seal(&mut self) {
        self.predefined_consts();
        self.apply_jumps();
    }

    fn align(&mut self) {
        let mut n = self.amd.a.ip();

        while (n & 7) != 0 {
            self.amd.nop();
            n += 1
        }
    }

    fn set_label(&mut self, label: &str) {
        self.amd.a.set_label(label);
    }

    fn branch(&mut self, label: &str) {
        self.amd.xor(Amd::RAX, Amd::RAX);
        self.amd.jz(label);
    }

    /// jump to label if cond == is_else
    /// note that `is_else` is not the correct name anymore and should be
    /// changed to `expectation`
    fn branch_if(&mut self, cond: Reg, label: &str, is_else: bool) {
        self.amd.vucomisd(ϕ(cond), ϕ(cond));
        /*
         * if is_else (expectation) is true, jump if cond is true (all-1, NaN).
         * In this situation, vucomisd returns an unordered result, setting
         * PF = 1 (jpe)
         */
        if is_else {
            self.amd.jpe(label);
        } else {
            self.amd.jpo(label);
        }
    }

    fn fuse_load_math(&mut self) {
        fuse_load_math(&mut self.amd, self.last_load);
    }

    //***********************************

    fn fmov(&mut self, dst: Reg, s1: Reg) {
        if dst != s1 {
            self.amd.movapd(ϕ(dst), ϕ(s1));
        }
    }

    fn fxchg(&mut self, s1: Reg, s2: Reg) {
        self.amd.xorpd(ϕ(s1), ϕ(s2));
        self.amd.xorpd(ϕ(s2), ϕ(s1));
        self.amd.xorpd(ϕ(s1), ϕ(s2));
    }

    fn load_const(&mut self, dst: Reg, idx: u32) {
        self.last_load = self.amd.a.ip();
        let label = format!("_const_{}_", idx);
        self.amd.movsd_xmm_label(ϕ(dst), label.as_str());
    }

    fn load_mem(&mut self, dst: Reg, idx: u32) {
        self.last_load = self.amd.a.ip();
        self.amd.movsd_xmm_mem(ϕ(dst), MEM, (idx * REG_SIZE) as i32);
    }

    fn save_mem(&mut self, dst: Reg, idx: u32) {
        self.amd.movsd_mem_xmm(MEM, (idx * REG_SIZE) as i32, ϕ(dst));
    }

    fn save_mem_result(&mut self, idx: u32) {
        self.save_mem(Reg::Ret, idx);
    }

    fn load_param(&mut self, dst: Reg, idx: u32) {
        self.last_load = self.amd.a.ip();

        if matches!(self.config.kernel_type(), KernelType::RowFirst) {
            self.amd
                .movsd_xmm_mem(ϕ(dst), PARAMS, (idx * REG_SIZE) as i32);
        } else {
            self.amd.movsd_xmm_mem(ϕ(dst), PARAMS, 8 * idx as i32);
        }
    }

    fn load_stack(&mut self, dst: Reg, idx: u32) {
        self.last_load = self.amd.a.ip();

        if idx < ABI_AREA as u32 {
            self.amd.movsd_xmm_mem(ϕ(dst), SP, (idx * REG_SIZE) as i32);
        } else {
            self.amd
                .movsd_xmm_mem(ϕ(dst), STACK, (idx * REG_SIZE) as i32);
        }
    }

    fn save_stack(&mut self, dst: Reg, idx: u32) {
        if idx < ABI_AREA as u32 {
            self.amd.movsd_mem_xmm(SP, (idx * REG_SIZE) as i32, ϕ(dst));
        } else {
            self.amd
                .movsd_mem_xmm(STACK, (idx * REG_SIZE) as i32, ϕ(dst));
        }
    }

    fn load_mem_complex(&mut self, xd: Reg, yd: Reg, idx: u32) {
        self.load_mem(xd, idx);
        self.load_mem(yd, idx + 1);
    }

    fn save_mem_complex(&mut self, xs: Reg, ys: Reg, idx: u32) {
        self.save_mem(xs, idx);
        self.save_mem(ys, idx + 1);
    }

    fn load_param_complex(&mut self, xd: Reg, yd: Reg, idx: u32) {
        self.load_param(xd, idx);
        self.load_param(yd, idx + 1);
    }

    fn load_stack_complex(&mut self, xd: Reg, yd: Reg, idx: u32) {
        self.load_stack(xd, idx);
        self.load_stack(yd, idx + 1);
    }

    fn save_stack_complex(&mut self, xs: Reg, ys: Reg, idx: u32) {
        self.save_stack(xs, idx);
        self.save_stack(ys, idx + 1);
    }

    fn save_stack_result(&mut self, idx: u32) {
        self.save_stack(Reg::Ret, idx);
    }

    fn neg(&mut self, dst: Reg, s1: Reg) {
        self.load_const_by_name(Reg::Temp, "_minus_zero_");
        self.xor(dst, s1, Reg::Temp);
    }

    fn abs(&mut self, dst: Reg, s1: Reg) {
        self.load_const_by_name(Reg::Temp, "_minus_zero_");
        self.andnot(dst, Reg::Temp, s1);
    }

    fn root(&mut self, dst: Reg, s1: Reg) {
        uniop!(self, sqrtsd, dst, s1);
    }

    fn real_root(&mut self, dst: Reg, s1: Reg) {
        self.root(dst, s1);
    }

    fn recip(&mut self, dst: Reg, s1: Reg) {
        self.load_const_by_name(Reg::Temp, "_one_");
        self.divide(dst, Reg::Temp, s1);
    }

    fn half(&mut self, dst: Reg, s1: Reg) {
        self.load_const_by_name(Reg::Temp, "_half_");
        self.times(dst, s1, Reg::Temp);
    }

    fn round(&mut self, dst: Reg, s1: Reg) {
        roundop!(self, dst, s1, RoundingMode::Round);
    }

    fn floor(&mut self, dst: Reg, s1: Reg) {
        roundop!(self, dst, s1, RoundingMode::Floor);
    }

    fn ceiling(&mut self, dst: Reg, s1: Reg) {
        roundop!(self, dst, s1, RoundingMode::Ceiling);
    }

    fn trunc(&mut self, dst: Reg, s1: Reg) {
        roundop!(self, dst, s1, RoundingMode::Trunc);
    }

    fn frac(&mut self, dst: Reg, s1: Reg) {
        self.floor(Reg::Temp, s1);
        self.minus(dst, s1, Reg::Temp);
    }

    fn plus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, addsd, dst, s1, s2, true);
    }

    fn minus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, subsd, dst, s1, s2, false);
    }

    fn times(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, mulsd, dst, s1, s2, true);
    }

    fn divide(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, divsd, dst, s1, s2, false);
    }

    fn times_complex(
        &mut self,
        _xd: Reg,
        _yd: Reg,
        _x1: Reg,
        _y1: Reg,
        _x2: Reg,
        _y2: Reg,
    ) -> bool {
        false
    }

    fn divide_complex(
        &mut self,
        _xd: Reg,
        _yd: Reg,
        _x1: Reg,
        _y1: Reg,
        _x2: Reg,
        _y2: Reg,
    ) -> bool {
        false
    }

    fn support_times2(&self) -> bool {
        false
    }

    fn times2_loc(&mut self, _d1: Reg, _s1: Reg, _l1: Loc, _d2: Reg, _s2: Reg, _l2: Loc) {
        unreachable!()
    }

    fn real(&mut self, dst: Reg, s1: Reg) {
        self.fmov(dst, s1);
    }

    fn imaginary(&mut self, dst: Reg, _s1: Reg) {
        self.xor(dst, dst, dst);
    }

    fn conjugate(&mut self, dst: Reg, s1: Reg) {
        self.fmov(dst, s1);
    }

    fn complex(&mut self, dst: Reg, s1: Reg, _s2: Reg) {
        self.fmov(dst, s1);
    }

    fn gt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, cmpnlesd, dst, s1, s2, false);
    }

    fn geq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, cmpnltsd, dst, s1, s2, false);
    }

    fn lt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, cmpltsd, dst, s1, s2, false);
    }

    fn leq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, cmplesd, dst, s1, s2, false);
    }

    fn eq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, cmpeqsd, dst, s1, s2, true);
    }

    fn neq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, cmpneqsd, dst, s1, s2, true);
    }

    fn and(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, andpd, dst, s1, s2, true);
    }

    fn andnot(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, andnpd, dst, s1, s2, false);
    }

    fn or(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, orpd, dst, s1, s2, true);
    }

    fn xor(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, xorpd, dst, s1, s2, true);
    }

    fn not(&mut self, dst: Reg, s1: Reg) {
        self.load_const_by_name(Reg::Temp, "_all_ones_");
        self.xor(dst, s1, Reg::Temp);
    }

    fn fused_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        self.times(s1, s1, s2);
        self.plus(dst, s1, s3);
    }

    fn fused_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        self.times(s1, s1, s2);
        self.minus(dst, s1, s3);
    }

    fn fused_neg_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        self.times(s1, s1, s2);
        self.minus(dst, s3, s1);
    }

    fn fused_neg_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        self.times(s1, s1, s2);
        self.plus(dst, s1, s3);
        self.neg(dst, dst);
    }

    fn add_consts(&mut self, consts: &[f64]) {
        for (idx, val) in consts.iter().enumerate() {
            let label = format!("_const_{}_", idx);
            self.set_label(label.as_str());
            self.append_quad((*val).to_bits());
        }
    }

    fn add_func(&mut self, op: &str, f: Func) {
        add_func(&mut self.amd, op, f);
    }

    fn call(&mut self, op: &str, num_args: usize) -> Result<()> {
        if is_external_func(op) {
            return self.call_external(op, num_args);
        }

        let label = format!("_func_{}_", op);
        self.amd.call_indirect(&label);

        Ok(())
    }

    fn call_complex(&mut self, op: &str, num_args: usize) -> Result<()> {
        let label = format!("_func_{}_", op);

        if num_args == 2 {
            self.save_stack(Reg::Gen(0), 4);
            self.save_stack(Reg::Gen(1), 5);
        }

        if cfg!(target_family = "windows") {
            self.amd.lea_mem(Amd::R8, SP, 32);
        } else {
            self.amd.lea_mem(Amd::RDI, SP, 32);
        }

        self.amd.call_indirect(&label);

        self.load_stack(Reg::Ret, 4);
        self.load_stack(Reg::Temp, 5);

        Ok(())
    }

    fn call_funclet(&mut self, label: &str) {
        self.amd.call_relative(label);
    }

    fn ret(&mut self) {
        self.amd.ret();
    }

    fn ifelse(&mut self, dst: Reg, true_val: Reg, false_val: Reg, idx: u32) {
        if true_val == false_val {
            self.fmov(dst, true_val);
        } else if dst != false_val {
            self.load_stack(Reg::Temp, idx);
            self.and(dst, Reg::Temp, true_val);
            self.andnot(Reg::Temp, Reg::Temp, false_val);
            self.or(dst, dst, Reg::Temp);
        } else {
            // dst == false_val && dst != true_val
            self.load_stack(Reg::Temp, idx);
            self.andnot(dst, Reg::Temp, false_val);
            self.and(Reg::Temp, Reg::Temp, true_val);
            self.or(dst, dst, Reg::Temp);
        }
    }

    /****************** Prologues/Epilogues ********************/

    #[cfg(target_family = "unix")]
    fn prologue_fast(&mut self, cap: usize, count_states: usize, count_obs: usize) {
        self.amd.push(Amd::RBP);
        self.amd.push(STACK);
        self.amd.push(MEM);
        self.amd.mov(Amd::RBP, SP);

        let frame_size = align_stack((count_states + count_obs) as u32 * REG_SIZE);
        sub_rsp(&mut self.amd, frame_size);
        self.amd.mov(MEM, SP);

        sub_rsp(&mut self.amd, align_stack(cap as u32 * REG_SIZE));
        self.amd.mov(STACK, SP);

        for i in 0..count_states {
            self.amd.movsd_mem_xmm(MEM, (i * 8) as i32, i as u8);
        }
    }

    #[cfg(target_family = "windows")]
    fn prologue_fast(&mut self, cap: usize, count_states: usize, count_obs: usize) {
        self.amd.push(Amd::RBP);
        self.amd.push(STACK);
        self.amd.push(MEM);
        self.amd.mov(Amd::RBP, SP);

        let frame_size = align_stack((count_states + count_obs) as u32 * REG_SIZE);
        sub_rsp(&mut self.amd, frame_size);
        self.amd.mov(MEM, SP);

        sub_rsp(&mut self.amd, align_stack(cap as u32 * REG_SIZE));
        self.amd.mov(STACK, SP);

        for i in 0..count_states.min(4) {
            self.amd
                .movsd_mem_xmm(MEM, (i as u32 * REG_SIZE) as i32, i as u8);
        }

        for i in 4..count_states {
            let i = i as u32;
            // the offset of the fifth or eight arguments:
            // +4 for the 32-byte home
            // +1 for the return address in the stack
            // +1 for RBP in the stack
            // -4 for the first four arguments passed in XMM0-XMM3
            self.amd
                .movsd_xmm_mem(0, MEM, (frame_size + (i + 2) * REG_SIZE) as i32);
            self.amd.movsd_mem_xmm(MEM, (i * REG_SIZE) as i32, 0);
        }
    }

    fn epilogue_fast(
        &mut self,
        _cap: usize,
        _count_states: usize,
        _count_obs: usize,
        idx_ret: i32,
    ) {
        self.amd.movsd_xmm_mem(0, MEM, idx_ret * REG_SIZE as i32);
        self.amd.mov(SP, Amd::RBP);
        self.amd.pop(MEM);
        self.amd.pop(STACK);
        self.amd.pop(Amd::RBP);
        self.amd.ret();
    }

    fn prologue_indirect(
        &mut self,
        cap: usize,
        count_states: usize,
        count_obs: usize,
        count_params: usize,
    ) {
        let regions = StackRegions::new(cap, count_states, count_obs, count_params);

        if matches!(self.config.kernel_type(), KernelType::RowFirst) {
            self.prologue_symbolica(&regions)
        } else {
            self.prologue_sympy(&regions)
        }
    }

    fn epilogue_indirect(
        &mut self,
        cap: usize,
        count_states: usize,
        count_obs: usize,
        count_params: usize,
    ) {
        let regions = StackRegions::new(cap, count_states, count_obs, count_params);

        if matches!(self.config.kernel_type(), KernelType::RowFirst) {
            self.epilogue_symbolica(&regions)
        } else {
            self.epilogue_sympy(&regions)
        }
    }

    fn save_used_registers(&mut self, used: &[Reg]) {
        if cfg!(target_family = "windows") {
            for r in used {
                let phys_reg = ϕ(*r);
                if (6..=15).contains(&phys_reg) {
                    self.save_stack(*r, phys_reg as u32);
                }
            }
        }
    }

    fn load_used_registers(&mut self, used: &[Reg]) {
        if cfg!(target_family = "windows") {
            for r in used {
                let phys_reg = ϕ(*r);
                if (6..=15).contains(&phys_reg) {
                    self.load_stack(*r, phys_reg as u32);
                }
            }
        }
    }
}

impl AmdSSEGenerator {
    fn prologue_sympy(&mut self, regions: &StackRegions) {
        save_nonvolatile_regs(&mut self.amd);

        self.amd.or(STATES, STATES);
        self.amd.jz("@main");

        let frame_size = align_stack((regions.count_states + regions.count_obs) * REG_SIZE);
        sub_rsp(&mut self.amd, frame_size);
        self.amd.mov(MEM, SP); // in indirect mode, MEM is allocated on the stack

        for i in 0..regions.count_states {
            self.amd.mov_reg_mem(Amd::RAX, STATES, 2 * 8 * i as i32);
            let k = i * REG_SIZE;
            self.amd.movsd_xmm_indexed(RET, Amd::RAX, IDX, 8);
            self.amd.movsd_mem_xmm(MEM, k as i32, RET);
        }

        self.set_label("@main");
        allocate_stack(&mut self.amd, regions.cap * REG_SIZE, false);
    }

    fn epilogue_sympy(&mut self, regions: &StackRegions) {
        self.amd.xor(Amd::RAX, Amd::RAX);
        self.set_label("@epilogue");

        self.amd.or(STATES, STATES);
        self.amd.jz("@done");

        for i in 0..regions.count_obs {
            self.amd
                .mov_reg_mem(Amd::RCX, STATES, 2 * 8 * (regions.count_states + i) as i32);
            let k = (regions.count_states + i) * REG_SIZE;
            self.amd.movsd_xmm_mem(RET, MEM, k as i32);
            self.amd.movsd_indexed_xmm(Amd::RCX, IDX, 8, RET);
        }

        self.set_label("@done");

        load_nonvolatile_regs(&mut self.amd);
        self.amd.ret();
    }

    fn prologue_symbolica(&mut self, regions: &StackRegions) {
        save_nonvolatile_regs(&mut self.amd);
        allocate_stack(&mut self.amd, regions.cap * REG_SIZE, true);
    }

    fn epilogue_symbolica(&mut self, _regions: &StackRegions) {
        self.amd.xor(Amd::RAX, Amd::RAX);
        self.set_label("@epilogue");
        load_nonvolatile_regs(&mut self.amd);
        self.amd.ret();
    }
}

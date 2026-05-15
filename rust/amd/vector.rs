use crate::code::Func;
use crate::config::{Config, SPILL_AREA};
use crate::generator::{FuncletType, Generator};
use crate::symbol::Loc;
use crate::utils::align_stack;
use crate::utils::{is_external_func, reg, DataType, Reg};
use anyhow::{anyhow, Result};

use super::asm::{Amd, RoundingMode};
use super::*;

const REG_SIZE: u32 = 32;

macro_rules! binop {
    ($self:ident, $simd:ident, $dst:expr, $s1: expr, $s2: expr) => {
        $self.amd.$simd(ϕ($dst), ϕ($s1), ϕ($s2));
    };
}

macro_rules! uniop {
    ($self:ident, $simd:ident, $dst:expr, $s1: expr) => {
        $self.amd.$simd(ϕ($dst), ϕ($s1));
    };
}

macro_rules! roundop {
    ($self:ident, $dst:expr, $s1: expr, $mode: expr) => {
        $self.amd.vroundpd(ϕ($dst), ϕ($s1), $mode);
    };
}

macro_rules! fuseop {
    ($self:ident, $f132:ident, $f213:ident, $f231:ident, $dst: expr, $a: expr, $b: expr, $c:ident) => {{
        if $dst == $a {
            $self.amd.$f132(ϕ($a), ϕ($c), ϕ($b));
        } else if $dst == $b {
            $self.amd.$f213(ϕ($b), ϕ($a), ϕ($c));
        } else if $dst == $c {
            $self.amd.$f231(ϕ($c), ϕ($a), ϕ($b));
        } else {
            $self.fmov($dst, $a);
            $self.amd.$f132(ϕ($dst), ϕ($c), ϕ($b));
        }
    }};
}

pub struct AmdVectorGenerator {
    amd: Amd,
    config: Config,
    last_load: usize,
}

impl AmdVectorGenerator {
    pub fn new(config: Config) -> AmdVectorGenerator {
        AmdVectorGenerator {
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

    fn load_const_by_name(&mut self, dst: Reg, label: &str) {
        self.amd.vbroadcastsd_label(ϕ(dst), label);
    }

    fn vzeroupper(&mut self) {
        self.amd.vzeroupper();
    }

    fn call_vector_unary(&mut self, label: &str) {
        // reserves 64 bytes in the stack
        // 32 bytes for shadow store (mandatory in Windows)
        // 32 bytes to save ymm0
        self.amd.vmovpd_mem_ymm(STACK, 32, 0);

        self.vzeroupper();

        for i in 0..4 {
            if i > 0 {
                self.amd.vmovsd_xmm_mem(0, STACK, 32 + i * 8);
            }
            self.amd.call_indirect(label);
            self.amd.vmovsd_mem_xmm(STACK, 32 + i * 8, 0);
        }

        self.amd.vmovpd_ymm_mem(0, STACK, 32);
    }

    fn call_vector_binary(&mut self, label: &str) {
        // reserves 96 bytes in the stack
        // 32 bytes for shadow store (mandatory in Windows)
        // 32 bytes to save ymm0
        // 32 bytes to save ymm1
        self.amd.vmovpd_mem_ymm(STACK, 32, 0);
        self.amd.vmovpd_mem_ymm(STACK, 64, 1);

        self.vzeroupper();

        for i in 0..4 {
            if i > 0 {
                self.amd.vmovsd_xmm_mem(0, STACK, 32 + i * 8);
                self.amd.vmovsd_xmm_mem(1, STACK, 64 + i * 8);
            }
            self.amd.call_indirect(label);
            self.amd.vmovsd_mem_xmm(STACK, 32 + i * 8, 0);
        }

        self.amd.vmovpd_ymm_mem(0, STACK, 32);
    }

    fn call_complex_vector_unary(&mut self, label: &str) {
        self.amd.vmovpd_mem_ymm(STACK, 64, 0);
        self.amd.vmovpd_mem_ymm(STACK, 96, 1);

        self.vzeroupper();

        for i in 0..4 {
            if i > 0 {
                self.amd.vmovsd_xmm_mem(0, STACK, 64 + i * 8);
                self.amd.vmovsd_xmm_mem(1, STACK, 96 + i * 8);
            }

            if cfg!(target_family = "windows") {
                self.amd.lea_mem(Amd::R8, STACK, 32);
            } else {
                self.amd.lea_mem(Amd::RDI, STACK, 32);
            }

            self.amd.call_indirect(label);

            self.amd.vmovsd_xmm_mem(0, STACK, 32);
            self.amd.vmovsd_xmm_mem(1, STACK, 40);
            self.amd.vmovsd_mem_xmm(STACK, 64 + i * 8, 0);
            self.amd.vmovsd_mem_xmm(STACK, 96 + i * 8, 1);
        }

        self.amd.vmovpd_ymm_mem(0, STACK, 64);
        self.amd.vmovpd_ymm_mem(1, STACK, 96);
    }

    fn call_complex_vector_binary(&mut self, label: &str) {
        self.amd.vmovpd_mem_ymm(STACK, 64, 0);
        self.amd.vmovpd_mem_ymm(STACK, 96, 1);
        self.amd.vmovpd_mem_ymm(STACK, 128, 2);
        self.amd.vmovpd_mem_ymm(STACK, 160, 3);

        self.vzeroupper();

        for i in 0..4 {
            if i > 0 {
                self.amd.vmovsd_xmm_mem(0, STACK, 64 + i * 8);
                self.amd.vmovsd_xmm_mem(1, STACK, 96 + i * 8);
                self.amd.vmovsd_xmm_mem(2, STACK, 128 + i * 8);
                self.amd.vmovsd_xmm_mem(3, STACK, 160 + i * 8);
            }

            self.amd.vmovsd_mem_xmm(STACK, 32, 2);
            self.amd.vmovsd_mem_xmm(STACK, 40, 3);

            if cfg!(target_family = "windows") {
                self.amd.lea_mem(Amd::R8, STACK, 32);
            } else {
                self.amd.lea_mem(Amd::RDI, STACK, 32);
            }

            self.amd.call_indirect(label);

            self.amd.vmovsd_xmm_mem(0, STACK, 32);
            self.amd.vmovsd_xmm_mem(1, STACK, 40);
            self.amd.vmovsd_mem_xmm(STACK, 64 + i * 8, 0);
            self.amd.vmovsd_mem_xmm(STACK, 96 + i * 8, 1);
        }

        self.amd.vmovpd_ymm_mem(0, STACK, 64);
        self.amd.vmovpd_ymm_mem(1, STACK, 96);
    }

    fn call_external(&mut self, op: &str, num_args: usize) -> Result<()> {
        let cap = SPILL_AREA as u32;

        self.amd.mov_reg_label(ARGS[0], &format!("_env_{}_", op));
        self.amd.lea_mem(ARGS[1], STACK, (cap * REG_SIZE) as i32);
        self.amd.mov_imm(ARGS[2], num_args as u32);
        self.amd.lea_mem(ARGS[3], STACK, 4 * REG_SIZE as i32);
        self.vzeroupper();

        self.amd.call_indirect(&format!("_simd_{}_", op));

        if self.config.is_complex() {
            let l1 = format!(".P{}", self.amd.a.ip());
            let l2 = format!(".Q{}", self.amd.a.ip());

            self.amd.or(Amd::RAX, Amd::RAX);
            self.amd.jz(&l1);

            self.amd.vmovpd_ymm_mem(2, STACK, 4 * REG_SIZE as i32);
            self.amd.vmovpd_ymm_mem(3, STACK, 5 * REG_SIZE as i32);
            self.amd.vshufpd(0, 2, 3, 0);
            self.amd.vshufpd(1, 2, 3, 0x0f);

            self.amd.jmp(&l2);
            self.set_label(&l1);

            self.amd.vmovpd_ymm_mem(0, STACK, 4 * REG_SIZE as i32);
            self.amd.vmovpd_ymm_mem(1, STACK, 5 * REG_SIZE as i32);

            self.set_label(&l2);
        } else {
            self.amd.vmovpd_ymm_mem(0, STACK, 4 * REG_SIZE as i32);
        }

        Ok(())
    }

    fn predefined_consts(&mut self) {
        self.align();
        predefined_consts(&mut self.amd);
    }
}

impl Generator for AmdVectorGenerator {
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
        true
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

    fn branch_if(&mut self, cond: Reg, label: &str, is_else: bool) {
        self.amd.vmovmskpd(Amd::RAX, ϕ(cond));
        self.amd.and_imm(Amd::RAX, 15);

        if !is_else {
            self.amd.cmp_imm(Amd::RAX, 15);
        }

        self.amd.jz(label);

        if !self.config.simd_branch() {
            self.amd.or(Amd::RAX, Amd::RAX);
            self.amd.jnz("@epilogue");
        }
    }

    fn fuse_load_math(&mut self) {
        fuse_load_math(&mut self.amd, self.last_load);
    }

    //***********************************

    fn fmov(&mut self, dst: Reg, s1: Reg) {
        if dst != s1 {
            self.amd.vmovapd(ϕ(dst), ϕ(s1));
        }
    }

    fn fxchg(&mut self, s1: Reg, s2: Reg) {
        self.amd.vxorpd(ϕ(s1), ϕ(s1), ϕ(s2));
        self.amd.vxorpd(ϕ(s2), ϕ(s1), ϕ(s2));
        self.amd.vxorpd(ϕ(s1), ϕ(s1), ϕ(s2));
    }

    fn load_const(&mut self, dst: Reg, idx: u32) {
        self.last_load = self.amd.a.ip();
        let label = format!("_const_{}_", idx);
        self.amd.vbroadcastsd_label(ϕ(dst), label.as_str());
    }

    fn load_mem(&mut self, dst: Reg, idx: u32) {
        self.last_load = self.amd.a.ip();
        self.amd
            .vmovpd_ymm_mem(ϕ(dst), MEM, (idx * REG_SIZE) as i32);
    }

    fn save_mem(&mut self, dst: Reg, idx: u32) {
        self.amd
            .vmovpd_mem_ymm(MEM, (idx * REG_SIZE) as i32, ϕ(dst));
    }

    fn save_mem_result(&mut self, idx: u32) {
        self.save_mem(Reg::Ret, idx);
    }

    fn load_param(&mut self, dst: Reg, idx: u32) {
        self.last_load = self.amd.a.ip();

        if self.config.symbolica() {
            self.amd
                .vmovpd_ymm_mem(ϕ(dst), PARAMS, (idx * REG_SIZE) as i32);
        } else {
            self.amd.vbroadcastsd(ϕ(dst), PARAMS, 8 * idx as i32);
        }
    }

    fn load_stack(&mut self, dst: Reg, idx: u32) {
        self.last_load = self.amd.a.ip();
        self.amd
            .vmovpd_ymm_mem(ϕ(dst), STACK, (idx * REG_SIZE) as i32);
    }

    fn save_stack(&mut self, dst: Reg, idx: u32) {
        self.amd
            .vmovpd_mem_ymm(STACK, (idx * REG_SIZE) as i32, ϕ(dst));
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
        uniop!(self, vsqrtpd, dst, s1);
    }

    fn real_root(&mut self, dst: Reg, s1: Reg) {
        self.root(dst, s1);
    }

    fn recip(&mut self, dst: Reg, s1: Reg) {
        self.load_const_by_name(Reg::Temp, "_one_");
        self.divide(dst, Reg::Temp, s1);
    }

    fn half(&mut self, dst: Reg, s1: Reg) {
        self.load_const_by_name(Reg::Temp, "_two_");
        self.divide(dst, s1, Reg::Temp);
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
        binop!(self, vaddpd, dst, s1, s2);
    }

    fn minus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, vsubpd, dst, s1, s2);
    }

    fn times(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, vmulpd, dst, s1, s2);
    }

    fn divide(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, vdivpd, dst, s1, s2);
    }

    fn times_complex(&mut self, xd: Reg, yd: Reg, x1: Reg, y1: Reg, x2: Reg, y2: Reg) -> bool {
        let xt = Reg::Gen(2);
        let yt = Reg::Gen(3);

        self.times(xt, y1, y2);
        self.times(yt, x1, y2);

        if xd != x2 && xd != y1 {
            self.fused_mul_sub(xd, x1, x2, xt);
            self.fused_mul_add(yd, x2, y1, yt);
        } else {
            self.fused_mul_sub(xt, x1, x2, xt);
            self.fused_mul_add(yd, x2, y1, yt);
            self.fmov(xd, xt);
        }

        true
    }

    fn divide_complex(&mut self, xd: Reg, yd: Reg, x1: Reg, y1: Reg, x2: Reg, y2: Reg) -> bool {
        let xt = Reg::Gen(2);
        let yt = Reg::Gen(3);
        let t = Reg::Temp;

        self.times(xt, y1, y2);
        self.fused_mul_add(xt, x1, x2, xt);
        self.times(yt, x1, y2);
        self.fused_mul_sub(yt, x2, y1, yt);
        self.times(t, x2, x2);
        self.fused_mul_add(t, y2, y2, t);
        self.divide(xd, xt, t);
        self.divide(yd, yt, t);

        true
    }

    fn support_times2(&self) -> bool {
        false
    }

    fn times2_loc(&mut self, d1: Reg, s1: Reg, l1: Loc, d2: Reg, s2: Reg, l2: Loc) {
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
        binop!(self, vcmpnlepd, dst, s1, s2);
    }

    fn geq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, vcmpnltpd, dst, s1, s2);
    }

    fn lt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, vcmpltpd, dst, s1, s2);
    }

    fn leq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, vcmplepd, dst, s1, s2);
    }

    fn eq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, vcmpeqpd, dst, s1, s2);
    }

    fn neq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, vcmpneqpd, dst, s1, s2);
    }

    fn and(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, vandpd, dst, s1, s2);
    }

    fn andnot(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, vandnpd, dst, s1, s2);
    }

    fn or(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, vorpd, dst, s1, s2);
    }

    fn xor(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, vxorpd, dst, s1, s2);
    }

    fn not(&mut self, dst: Reg, s1: Reg) {
        self.load_const_by_name(Reg::Temp, "_all_ones_");
        self.xor(dst, s1, Reg::Temp);
    }

    fn fused_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        fuseop!(self, vfmadd132pd, vfmadd213pd, vfmadd231pd, dst, s1, s2, s3);
    }

    fn fused_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        fuseop!(self, vfmsub132pd, vfmsub213pd, vfmsub231pd, dst, s1, s2, s3);
    }

    fn fused_neg_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        fuseop!(
            self,
            vfnmadd132pd,
            vfnmadd213pd,
            vfnmadd231pd,
            dst,
            s1,
            s2,
            s3
        );
    }

    fn fused_neg_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        fuseop!(
            self,
            vfnmsub132pd,
            vfnmsub213pd,
            vfnmsub231pd,
            dst,
            s1,
            s2,
            s3
        );
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

        match num_args {
            1 => self.call_vector_unary(&label),
            2 => self.call_vector_binary(&label),
            _ => return Err(anyhow!("invalid number of arguments")),
        }

        Ok(())
    }

    fn call_complex(&mut self, op: &str, num_args: usize) -> Result<()> {
        let label = format!("_func_{}_", op);

        match num_args {
            1 => self.call_complex_vector_unary(&label),
            2 => self.call_complex_vector_binary(&label),
            _ => return Err(anyhow!("invalid number of arguments")),
        }

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

        let frame_size = align_stack((count_states + count_obs) as u32 * REG_SIZE);
        sub_rsp(&mut self.amd, frame_size);
        self.amd.mov(MEM, STACK);
        sub_rsp(&mut self.amd, align_stack(cap as u32 * REG_SIZE));

        for i in 0..count_states {
            self.amd.vmovsd_mem_xmm(MEM, (i * 8) as i32, i as u8);
        }
    }

    #[cfg(target_family = "windows")]
    fn prologue_fast(&mut self, cap: usize, count_states: usize, count_obs: usize) {
        self.amd.push(Amd::RBP);

        let frame_size = align_stack((count_states + count_obs) as u32 * REG_SIZE);
        sub_rsp(&mut self.amd, frame_size);
        self.amd.mov(MEM, STACK);
        sub_rsp(&mut self.amd, align_stack(cap as u32 * REG_SIZE));

        for i in 0..count_states.min(4) {
            self.amd
                .vmovsd_mem_xmm(MEM, (i as u32 * REG_SIZE) as i32, i as u8);
        }

        for i in 4..count_states {
            let i = i as u32;
            // the offset of the fifth or eight arguments:
            // +4 for the 32-byte home
            // +1 for the return address in the stack
            // +1 for RBP in the stack
            // -4 for the first four arguments passed in XMM0-XMM3
            self.amd
                .vmovsd_xmm_mem(0, MEM, (frame_size + (i + 2) * REG_SIZE) as i32);
            self.amd.vmovsd_mem_xmm(MEM, (i * REG_SIZE) as i32, 0);
        }
    }

    fn epilogue_fast(&mut self, cap: usize, count_states: usize, count_obs: usize, idx_ret: i32) {
        self.vzeroupper();
        self.amd.movsd_xmm_mem(0, MEM, idx_ret * REG_SIZE as i32);

        let total_size = align_stack(cap as u32 * REG_SIZE)
            + align_stack((count_states + count_obs) as u32 * REG_SIZE);
        add_rsp(&mut self.amd, total_size);

        self.amd.pop(Amd::RBP);
        self.amd.ret();
    }

    /*
     * prologue_indirect generates the stack frame. It works in two modes:
     *  * Direct mode: MEM (state variables + obs) is passed directly as the first
     *      argument. The second argument is null.
     *  * Indirect mode: the second argument is a pointer to an array of pointers
     *      to states and obs. The third argument is the index into these arrays.
     *      MEM is allocated on the stack and filled based on the second and thirds args.
     *
     * Noth that the second argument determines whether it is the direct (args[1] == null) or indirect mode.
     * In both modes, the fourth argument points to an array of params.
     *
     * # Stack Frame Layout:
     *
     * x86-64 stack frame is composed of four segments. The length of each segment should
     *      be a multiple of 16.
     *  1. The return addess + old RBP (16 bytes).
     *  2. General registers area (32 bytes in Linux, uses the home area in Windows).
     *      call to `save_nonvolatile_regs`.
     *  3. Optional mem area to store state variables and observables in vectorized calls.
     *      Of length `frame_size`, which is aligned to 16 by calling `align_stack`.
     *  4. The temporary variables area of size `align_stack(cap * REG_SIZE)`.
     *      It has two sub-segments:
     *  4a. The actual temporary variables area.
     *  4b. A default spill area of `16 * REG_SIZE` bytes. It is generated by
     *      `SymbolTable::new` adding 16 dummy temp variables. The top 10 slots are used
     *      to store callee-saved xmm/ymm registers (`save_used_registers`). The bottom
     *      6 slots are the work area for various call routines, Specifically, the bottom
     *      32 bytes is reserved as the home area for Windows call ABI.
     */
    fn prologue_indirect(
        &mut self,
        cap: usize,
        count_states: usize,
        count_obs: usize,
        count_params: usize,
    ) {
        if self.config.symbolica() {
            return self.prologue_symbolica(cap, count_params, count_obs);
        }

        self.amd.push(Amd::RBP);
        save_nonvolatile_regs(&mut self.amd);

        self.amd.mov(MEM, ARGS[0]); // first arg = mem if direct mode, otherwise null
        self.amd.mov(STATES, ARGS[1]); // second arg = states+obs if indirect mode, otherwise null
        self.amd.mov(IDX, ARGS[2]); // third arg = index if indirect mode
        self.amd.mov(PARAMS, ARGS[3]); // fourth arg = params

        self.amd.or(STATES, STATES);
        self.amd.jz("@main");

        let frame_size = align_stack((count_states + count_obs) as u32 * REG_SIZE);
        sub_rsp(&mut self.amd, frame_size);
        self.amd.mov(MEM, STACK); // in indirect mode, MEM is allocated on the stack

        // multiply IDX by 4 to convert from f64x4 index to f64 index
        self.amd.add(IDX, IDX);
        self.amd.add(IDX, IDX);

        for i in 0..count_states {
            self.amd.mov_reg_mem(Amd::RAX, STATES, 2 * 8 * i as i32);
            let k = i as u32 * REG_SIZE;
            self.amd.vmovpd_ymm_indexed(RET, Amd::RAX, IDX, 8);
            self.amd.vmovpd_mem_ymm(MEM, k as i32, RET);
        }

        // may save idx (RDX) as double in RBP + 8/32 * count_states

        self.set_label("@main");
        sub_rsp(&mut self.amd, align_stack(cap as u32 * REG_SIZE));
    }

    fn epilogue_indirect(
        &mut self,
        cap: usize,
        count_states: usize,
        count_obs: usize,
        count_params: usize,
    ) {
        self.amd.xor(Amd::RAX, Amd::RAX);
        self.set_label("@epilogue");

        if self.config.symbolica() {
            return self.epilogue_symbolica(cap, count_params, count_obs);
        }

        add_rsp(&mut self.amd, align_stack(cap as u32 * REG_SIZE));

        self.amd.or(STATES, STATES);
        self.amd.jz("@done");

        for i in 0..count_obs {
            self.amd
                .mov_reg_mem(Amd::RCX, STATES, 2 * 8 * (count_states + i) as i32);
            let k = (count_states + i) as u32 * REG_SIZE;
            self.amd.vmovpd_ymm_mem(RET, MEM, k as i32);
            self.amd.vmovpd_indexed_ymm(Amd::RCX, IDX, 8, RET);
        }

        let frame_size = align_stack((count_states + count_obs) as u32 * REG_SIZE);
        add_rsp(&mut self.amd, frame_size);
        self.set_label("@done");

        self.vzeroupper();

        load_nonvolatile_regs(&mut self.amd);
        self.amd.pop(Amd::RBP);
        self.amd.ret();
    }

    fn save_used_registers(&mut self, used: &[u8]) {
        let count_shadows = self.count_shadows();

        for r in used {
            if *r >= count_shadows {
                self.save_stack(reg(*r), *r as u32 + 2);
            }
        }
    }

    fn load_used_registers(&mut self, used: &[u8]) {
        let count_shadows = self.count_shadows();

        for r in used {
            if *r >= count_shadows {
                self.load_stack(reg(*r), *r as u32 + 2);
            }
        }
    }
}

impl AmdVectorGenerator {
    fn prologue_symbolica(&mut self, cap: usize, count_params: usize, count_obs: usize) {
        self.amd.push(Amd::RBP);
        save_nonvolatile_regs(&mut self.amd);

        self.amd.mov(MEM, ARGS[0]); // first arg = mem if direct mode, otherwise null
        self.amd.mov(STATES, ARGS[1]); // second arg = states+obs if indirect mode, otherwise null
        self.amd.mov(IDX, ARGS[2]); // third arg = index if indirect mode
        self.amd.mov(PARAMS, ARGS[3]); // fourth arg = params

        if REG_SIZE == 32 {
            self.amd.or(IDX, IDX);
            self.amd.jz("@main");

            sub_rsp(&mut self.amd, align_stack(count_params as u32 * 32));
            self.amd.mov(Amd::RAX, PARAMS);
            self.amd.mov(PARAMS, STACK);

            for j in 0..4 {
                for i in 0..count_params {
                    self.amd
                        .vmovsd_xmm_mem(RET, Amd::RAX, 8 * (i + j * count_params) as i32);
                    self.amd.vmovsd_mem_xmm(PARAMS, 8 * (i * 4 + j) as i32, RET);
                }
            }

            sub_rsp(&mut self.amd, align_stack(count_obs as u32 * 32));
            self.amd.mov(STATES, MEM);
            self.amd.mov(MEM, STACK);

            self.set_label("@main");
        }
        sub_rsp(&mut self.amd, align_stack(cap as u32 * REG_SIZE));
    }

    fn epilogue_symbolica(&mut self, cap: usize, count_params: usize, count_obs: usize) {
        add_rsp(&mut self.amd, align_stack(cap as u32 * REG_SIZE));

        if REG_SIZE == 32 {
            self.amd.or(IDX, IDX);
            self.amd.jz("@done");

            for j in 0..4 {
                for i in 0..count_obs {
                    self.amd.vmovsd_xmm_mem(RET, MEM, 8 * (i * 4 + j) as i32);
                    self.amd
                        .vmovsd_mem_xmm(STATES, 8 * (i + j * count_obs) as i32, 0);
                }
            }

            let frame_size =
                align_stack(count_params as u32 * 32) + align_stack(count_obs as u32 * 32);
            add_rsp(&mut self.amd, frame_size);
            self.set_label("@done");
        }

        self.vzeroupper();

        load_nonvolatile_regs(&mut self.amd);
        self.amd.pop(Amd::RBP);
        self.amd.ret();
    }
}

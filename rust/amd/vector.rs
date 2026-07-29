use super::super::code::Func;
use super::super::config::{Config, ABI_AREA};
use super::super::generator::{FuncletType, Generator, StackRegions};
use super::super::symbol::Loc;
use super::super::utils::align_stack;
use super::super::utils::{is_external_func, DataType, Reg};
use anyhow::{anyhow, Result};

use super::asm::{Amd, RoundingMode};
use super::*;

const NUM_LANES: u32 = 4;
const REG_SIZE: i32 = 8 * NUM_LANES as i32;
const REG_USIZE: u32 = 8 * NUM_LANES;

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

pub struct AmdVectorF64x4Generator {
    amd: Amd,
    config: Config,
    last_load: usize,
}

impl AmdVectorF64x4Generator {
    pub fn new(config: Config) -> AmdVectorF64x4Generator {
        AmdVectorF64x4Generator {
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
        self.amd.vmovpd_mem_ymm(STACK, REG_SIZE, 0);

        self.vzeroupper();

        for i in 0..NUM_LANES as i32 {
            if i > 0 {
                self.amd.vmovsd_xmm_mem(0, SP, REG_SIZE + i * 8);
            }
            self.amd.call_indirect(label);
            self.amd.vmovsd_mem_xmm(SP, REG_SIZE + i * 8, 0);
        }

        self.amd.vmovpd_ymm_mem(0, SP, REG_SIZE);
    }

    fn call_vector_binary(&mut self, label: &str) {
        // reserves 96 bytes in the stack
        // 32 bytes for shadow store (mandatory in Windows)
        // 32 bytes to save ymm0
        // 32 bytes to save ymm1
        self.amd.vmovpd_mem_ymm(SP, REG_SIZE, 0);
        self.amd.vmovpd_mem_ymm(SP, REG_SIZE * 2, 1);

        self.vzeroupper();

        for i in 0..4 {
            if i > 0 {
                self.amd.vmovsd_xmm_mem(0, SP, REG_SIZE + i * 8);
                self.amd.vmovsd_xmm_mem(1, SP, REG_SIZE * 2 + i * 8);
            }
            self.amd.call_indirect(label);
            self.amd.vmovsd_mem_xmm(SP, REG_SIZE + i * 8, 0);
        }

        self.amd.vmovpd_ymm_mem(0, SP, REG_SIZE);
    }

    fn call_complex_vector_unary(&mut self, label: &str) {
        self.amd.vmovpd_mem_ymm(SP, REG_SIZE * 2, 0);
        self.amd.vmovpd_mem_ymm(SP, REG_SIZE * 3, 1);

        self.vzeroupper();

        for i in 0..NUM_LANES as i32 {
            if i > 0 {
                self.amd.vmovsd_xmm_mem(0, SP, REG_SIZE * 2 + i * 8);
                self.amd.vmovsd_xmm_mem(1, SP, REG_SIZE * 3 + i * 8);
            }

            if cfg!(target_family = "windows") {
                self.amd.lea_mem(Amd::R8, SP, REG_SIZE);
            } else {
                self.amd.lea_mem(Amd::RDI, SP, REG_SIZE);
            }

            self.amd.call_indirect(label);

            self.amd.vmovsd_xmm_mem(0, SP, REG_SIZE);
            self.amd.vmovsd_xmm_mem(1, SP, REG_SIZE + 8);
            self.amd.vmovsd_mem_xmm(SP, REG_SIZE * 2 + i * 8, 0);
            self.amd.vmovsd_mem_xmm(SP, REG_SIZE * 3 + i * 8, 1);
        }

        self.amd.vmovpd_ymm_mem(0, SP, REG_SIZE * 2);
        self.amd.vmovpd_ymm_mem(1, SP, REG_SIZE * 3);
    }

    fn call_complex_vector_binary(&mut self, label: &str) {
        self.amd.vmovpd_mem_ymm(SP, REG_SIZE * 2, 0);
        self.amd.vmovpd_mem_ymm(SP, REG_SIZE * 3, 1);
        self.amd.vmovpd_mem_ymm(SP, REG_SIZE * 4, 2);
        self.amd.vmovpd_mem_ymm(SP, REG_SIZE * 5, 3);

        self.vzeroupper();

        for i in 0..NUM_LANES as i32 {
            if i > 0 {
                self.amd.vmovsd_xmm_mem(0, SP, REG_SIZE * 2 + i * 8);
                self.amd.vmovsd_xmm_mem(1, SP, REG_SIZE * 3 + i * 8);
                self.amd.vmovsd_xmm_mem(2, SP, REG_SIZE * 4 + i * 8);
                self.amd.vmovsd_xmm_mem(3, SP, REG_SIZE * 5 + i * 8);
            }

            self.amd.vmovsd_mem_xmm(SP, REG_SIZE, 2);
            self.amd.vmovsd_mem_xmm(SP, REG_SIZE + 8, 3);

            if cfg!(target_family = "windows") {
                self.amd.lea_mem(Amd::R8, SP, REG_SIZE);
            } else {
                self.amd.lea_mem(Amd::RDI, SP, REG_SIZE);
            }

            self.amd.call_indirect(label);

            self.amd.vmovsd_xmm_mem(0, SP, REG_SIZE);
            self.amd.vmovsd_xmm_mem(1, SP, REG_SIZE + 8);
            self.amd.vmovsd_mem_xmm(SP, REG_SIZE * 2 + i * 8, 0);
            self.amd.vmovsd_mem_xmm(SP, REG_SIZE * 3 + i * 8, 1);
        }

        self.amd.vmovpd_ymm_mem(0, SP, REG_SIZE * 2);
        self.amd.vmovpd_ymm_mem(1, SP, REG_SIZE * 3);
    }

    fn call_external(&mut self, op: &str, num_args: usize) -> Result<()> {
        let cap = ABI_AREA as i32;

        self.amd.mov_reg_label(ARGS[0], &format!("_env_{}_", op));
        self.amd.lea_mem(ARGS[1], STACK, cap * REG_SIZE);
        self.amd.mov_imm(ARGS[2], num_args as u32);
        self.amd.lea_mem(ARGS[3], SP, 4 * REG_SIZE);
        self.vzeroupper();

        self.amd.call_indirect(&format!("_simd_{}_", op));

        if self.config.is_complex() {
            let l1 = format!(".P{}", self.amd.a.ip());
            let l2 = format!(".Q{}", self.amd.a.ip());

            self.amd.or(Amd::RAX, Amd::RAX);
            self.amd.jz(&l1);

            self.amd.vmovpd_ymm_mem(2, SP, 4 * REG_SIZE);
            self.amd.vmovpd_ymm_mem(3, SP, 5 * REG_SIZE);
            self.amd.vshufpd(0, 2, 3, 0);
            self.amd.vshufpd(1, 2, 3, 0x0f);

            self.amd.jmp(&l2);
            self.set_label(&l1);

            self.amd.vmovpd_ymm_mem(0, SP, 4 * REG_SIZE);
            self.amd.vmovpd_ymm_mem(1, SP, 5 * REG_SIZE);

            self.set_label(&l2);
        } else {
            self.amd.vmovpd_ymm_mem(0, SP, 4 * REG_SIZE);
        }

        Ok(())
    }

    fn predefined_consts(&mut self) {
        self.align();
        predefined_consts(&mut self.amd);
    }
}

impl Generator for AmdVectorF64x4Generator {
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

    /// jump to label if all bits of cond == is_else
    fn branch_if(&mut self, cond: Reg, label: &str, is_else: bool) {
        self.amd.vmovmskpd(Amd::RAX, ϕ(cond));
        self.amd.and_imm(Amd::RAX, (1 << NUM_LANES) - 1);

        if is_else {
            self.amd.cmp_imm(Amd::RAX, (1 << NUM_LANES) - 1);
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
        self.amd.vmovpd_ymm_mem(ϕ(dst), MEM, idx as i32 * REG_SIZE);
    }

    fn save_mem(&mut self, dst: Reg, idx: u32) {
        self.amd.vmovpd_mem_ymm(MEM, idx as i32 * REG_SIZE, ϕ(dst));
    }

    fn save_mem_result(&mut self, idx: u32) {
        self.save_mem(Reg::Ret, idx);
    }

    fn load_param(&mut self, dst: Reg, idx: u32) {
        self.last_load = self.amd.a.ip();

        if self.config.symbolica() {
            self.amd
                .vmovpd_ymm_mem(ϕ(dst), PARAMS, idx as i32 * REG_SIZE);
        } else {
            self.amd.vbroadcastsd(ϕ(dst), PARAMS, 8 * idx as i32);
        }
    }

    fn load_stack(&mut self, dst: Reg, idx: u32) {
        self.last_load = self.amd.a.ip();
        if idx < 16 {
            self.amd.vmovpd_ymm_mem(ϕ(dst), SP, idx as i32 * REG_SIZE);
        } else {
            self.amd
                .vmovpd_ymm_mem(ϕ(dst), STACK, idx as i32 * REG_SIZE);
        }
    }

    fn save_stack(&mut self, dst: Reg, idx: u32) {
        if idx < 16 {
            self.amd.vmovpd_mem_ymm(SP, idx as i32 * REG_SIZE, ϕ(dst));
        } else {
            self.amd
                .vmovpd_mem_ymm(STACK, idx as i32 * REG_SIZE, ϕ(dst));
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

        if xd != x1 && xd != x2 {
            self.times(xd, y1, y2);
            self.fused_mul_sub(xd, x1, x2, xd);
            self.times(yd, x1, y2);
            self.fused_mul_add(yd, x2, y1, yd);
        } else if xd == x1 && xd != x2 {
            self.times(xt, y1, y2);
            self.fused_mul_sub(xt, x1, x2, xt);
            self.times(yd, x2, y1);
            self.fused_mul_add(yd, x1, y2, yd);
            self.fmov(xd, xt);
        } else if xd != x1 && xd == x2 {
            self.times(xt, y1, y2);
            self.fused_mul_sub(xt, x1, x2, xt);
            self.times(yd, x1, y2);
            self.fused_mul_add(yd, x2, y1, yd);
            self.fmov(xd, xt);
        } else {
            self.times(xt, y1, y2);
            self.fused_mul_sub(xt, x1, x2, xt);
            self.times(yt, x1, y2);
            self.fused_mul_add(yt, x2, y1, yt);
            self.fmov(xd, xt);
            self.fmov(yd, yt);
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
    fn prologue_fast(&mut self, _cap: usize, _count_states: usize, _count_obs: usize) {
        unreachable!()
    }

    #[cfg(target_family = "windows")]
    fn prologue_fast(&mut self, cap: usize, count_states: usize, count_obs: usize) {
        unreachable!()
    }

    fn epilogue_fast(
        &mut self,
        _cap: usize,
        _count_states: usize,
        _count_obs: usize,
        _idx_ret: i32,
    ) {
        unreachable!()
    }

    fn prologue_indirect(
        &mut self,
        cap: usize,
        count_states: usize,
        count_obs: usize,
        count_params: usize,
    ) {
        let regions = StackRegions::new(cap, count_states, count_obs, count_params);

        if self.config.symbolica() {
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

        if self.config.symbolica() {
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

impl AmdVectorF64x4Generator {
    fn prologue_sympy(&mut self, regions: &StackRegions) {
        save_nonvolatile_regs(&mut self.amd);

        self.amd.or(STATES, STATES);
        self.amd.jz("@main");

        let frame_size = align_stack((regions.count_states + regions.count_obs) * REG_USIZE);
        sub_rsp(&mut self.amd, frame_size);
        self.amd.mov(MEM, SP); // in indirect mode, MEM is allocated on the stack

        // multiply IDX by 4 to convert from f64x4 index to f64 index
        self.amd.add(IDX, IDX);
        self.amd.add(IDX, IDX);

        for i in 0..regions.count_states {
            self.amd.mov_reg_mem(Amd::RAX, STATES, 2 * 8 * i as i32);
            self.amd.vmovpd_ymm_indexed(RET, Amd::RAX, IDX, 8);
            self.amd.vmovpd_mem_ymm(MEM, i as i32 * REG_SIZE, RET);
        }

        self.set_label("@main");
        allocate_stack(&mut self.amd, regions.cap * REG_USIZE, false);
    }

    fn epilogue_sympy(&mut self, regions: &StackRegions) {
        self.amd.xor(Amd::RAX, Amd::RAX);
        self.set_label("@epilogue");

        self.amd.or(STATES, STATES);
        self.amd.jz("@done");

        for i in 0..regions.count_obs {
            self.amd
                .mov_reg_mem(Amd::RCX, STATES, 2 * 8 * (regions.count_states + i) as i32);
            self.amd
                .vmovpd_ymm_mem(RET, MEM, (regions.count_states + i) as i32 * REG_SIZE);
            self.amd.vmovpd_indexed_ymm(Amd::RCX, IDX, 8, RET);
        }

        self.set_label("@done");
        load_nonvolatile_regs(&mut self.amd);
        self.amd.ret();
    }

    fn prologue_symbolica(&mut self, regions: &StackRegions) {
        save_nonvolatile_regs(&mut self.amd);

        self.amd.or(IDX, IDX);
        self.amd.jz("@main");

        sub_rsp(&mut self.amd, align_stack(regions.count_params * REG_USIZE));
        self.amd.mov(Amd::RAX, PARAMS);
        self.amd.mov(PARAMS, SP);

        self.amd.mov_imm(Amd::RCX, regions.count_params);
        self.set_label(".load");

        for j in 0..NUM_LANES {
            self.amd
                .vmovsd_xmm_mem(RET, Amd::RAX, (8 * j * regions.count_params) as i32);
            self.amd.vmovsd_mem_xmm(PARAMS, 8 * j as i32, RET);
        }
        self.amd.add_imm(Amd::RAX, 8);
        self.amd.add_imm(PARAMS, 8 * NUM_LANES);
        self.amd.dec(Amd::RCX);
        self.amd.jnz(".load");

        self.amd
            .sub_imm(PARAMS, 8 * regions.count_params * NUM_LANES);

        sub_rsp(&mut self.amd, align_stack(regions.count_obs * REG_USIZE));
        self.amd.mov(STATES, MEM);
        self.amd.mov(MEM, SP);

        self.set_label("@main");
        allocate_stack(&mut self.amd, regions.cap * REG_USIZE, true);
    }

    fn epilogue_symbolica(&mut self, regions: &StackRegions) {
        self.amd.xor(Amd::RAX, Amd::RAX);
        self.set_label("@epilogue");

        for j in 0..NUM_LANES {
            for i in 0..regions.count_obs {
                self.amd
                    .vmovsd_xmm_mem(RET, MEM, 8 * (i * NUM_LANES + j) as i32);
                self.amd
                    .vmovsd_mem_xmm(STATES, 8 * (i + j * regions.count_obs) as i32, 0);
            }
        }

        self.set_label("@done");
        load_nonvolatile_regs(&mut self.amd);
        self.amd.ret();
    }
}

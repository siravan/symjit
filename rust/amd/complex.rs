use super::super::code::Func;
use super::super::config::{Config, KernelType, ABI_AREA};
use super::super::generator::{FuncletType, Generator, StackRegions};
use super::super::symbol::Loc;
use super::super::utils::align_stack;
use super::super::utils::{is_external_func, DataType, Reg};
use anyhow::Result;

use super::asm::{Amd, RoundingMode};
use super::*;

const REG_SIZE: u32 = 8;
const T0: u8 = 1; // Reg::Temp
const T1: u8 = 2;
const T2: u8 = 3;

macro_rules! binop {
    ($self:ident, $avx:ident, $dst:expr, $s1: expr, $s2: expr) => {
        $self.amd.$avx(ϕ($dst), ϕ($s1), ϕ($s2));
    };
}

macro_rules! roundop {
    ($self:ident, $dst:expr, $s1: expr, $mode: expr) => {
        $self.amd.vrounddd(ϕ($dst), ϕ($s1), $mode);
    };
}

/*
 *  ϕ translates a logical register number (in Reg) to a physical
 *  register number, according to the ABI.
 */
fn ϕ(r: Reg) -> u8 {
    match r {
        Reg::Ret => 0,
        Reg::Temp => 1,
        Reg::Left => 0,
        Reg::Right => 1,
        Reg::Gen(dst) => dst + 4,
        Reg::Static(..) => panic!("passing static registers to codegen"),
    }
}

pub struct AmdComplexGenerator {
    amd: Amd,
    config: Config,
    last_load: usize,
}

impl AmdComplexGenerator {
    pub fn new(config: Config) -> AmdComplexGenerator {
        AmdComplexGenerator {
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
        // self.amd.vbroadcastsd_label(ϕ(dst), label);
        self.amd.vmovsd_xmm_label(ϕ(dst), label);
    }

    fn vzeroupper(&mut self) {
        self.amd.vzeroupper();
    }

    fn call_external(&mut self, op: &str, num_args: usize) -> Result<()> {
        let cap = ABI_AREA as u32;

        self.amd.mov_reg_label(ARGS[0], &format!("_env_{}_", op));
        self.amd.lea_mem(ARGS[1], STACK, (cap * REG_SIZE) as i32);
        self.amd.mov_imm(ARGS[2], num_args as u32);
        self.amd.lea_mem(ARGS[3], SP, 4 * REG_SIZE as i32);
        self.vzeroupper();

        self.amd.call_indirect(&format!("_func_{}_", op));
        self.load_stack(Reg::Ret, 4);

        Ok(())
    }

    fn predefined_consts(&mut self) {
        self.align();
        predefined_consts(&mut self.amd);
    }
}

impl Generator for AmdComplexGenerator {
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
        FuncletType::Real
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

    //***********************************/
    fn fmov(&mut self, dst: Reg, s1: Reg) {
        if dst != s1 {
            self.amd.vmovapd(ϕ(dst), ϕ(s1));
        }
    }

    fn fxchg(&mut self, s1: Reg, s2: Reg) {
        self.amd.vxordd(ϕ(s1), ϕ(s1), ϕ(s2));
        self.amd.vxordd(ϕ(s2), ϕ(s1), ϕ(s2));
        self.amd.vxordd(ϕ(s1), ϕ(s1), ϕ(s2));
    }

    fn load_const(&mut self, dst: Reg, idx: u32) {
        self.last_load = self.amd.a.ip();
        let label = format!("_const_{}_", idx);
        // self.amd.vbroadcastsd_label(ϕ(dst), label.as_str());
        self.amd.vmovsd_xmm_label(ϕ(dst), label.as_str());
    }

    fn load_mem(&mut self, dst: Reg, idx: u32) {
        self.last_load = self.amd.a.ip();
        self.amd
            .vmovdd_xmm_mem(ϕ(dst), MEM, (idx * REG_SIZE) as i32);
    }

    fn save_mem(&mut self, dst: Reg, idx: u32) {
        self.amd
            .vmovdd_mem_xmm(MEM, (idx * REG_SIZE) as i32, ϕ(dst));
    }

    fn save_mem_result(&mut self, idx: u32) {
        self.save_mem(Reg::Ret, idx);
    }

    fn load_param(&mut self, dst: Reg, idx: u32) {
        self.last_load = self.amd.a.ip();
        self.amd
            .vmovdd_xmm_mem(ϕ(dst), PARAMS, (idx * REG_SIZE) as i32);
    }

    fn load_stack(&mut self, dst: Reg, idx: u32) {
        self.last_load = self.amd.a.ip();

        if idx < ABI_AREA as u32 {
            self.amd.vmovdd_xmm_mem(ϕ(dst), SP, (idx * REG_SIZE) as i32);
        } else {
            self.amd
                .vmovdd_xmm_mem(ϕ(dst), STACK, (idx * REG_SIZE) as i32);
        }
    }

    fn save_stack(&mut self, dst: Reg, idx: u32) {
        if idx < ABI_AREA as u32 {
            self.amd.vmovdd_mem_xmm(SP, (idx * REG_SIZE) as i32, ϕ(dst));
        } else {
            self.amd
                .vmovdd_mem_xmm(STACK, (idx * REG_SIZE) as i32, ϕ(dst));
        }
    }

    fn load_mem_complex(&mut self, _xd: Reg, _yd: Reg, _idx: u32) {}

    fn save_mem_complex(&mut self, _xs: Reg, _ys: Reg, _idx: u32) {}

    fn load_param_complex(&mut self, _xd: Reg, _yd: Reg, _idx: u32) {}

    fn load_stack_complex(&mut self, _xd: Reg, _yd: Reg, _idx: u32) {}

    fn save_stack_complex(&mut self, _xs: Reg, _ys: Reg, _idx: u32) {}

    fn save_stack_result(&mut self, idx: u32) {
        self.save_stack(Reg::Ret, idx);
    }

    fn load_arg(&mut self, arg: u8, loc: Loc) {
        if arg < 16 {
            load_f64x2_from_loc(&mut self.amd, arg, loc);
        } else {
            load_f64x2_from_loc(&mut self.amd, 0, loc);
            save_f64x2_to_loc(&mut self.amd, 0, self.config.location(arg));
        }
    }

    fn save_arg(&mut self, arg: u8, _loc: Loc) {
        if arg < 16 {
            save_f64x2_to_loc(&mut self.amd, arg, self.config.location(arg));
        }
    }

    fn load_arg_complex(&mut self, _arg: u8, _loc: Loc) {}
    fn save_arg_complex(&mut self, _arg: u8, _loc: Loc) {}

    fn neg(&mut self, dst: Reg, s1: Reg) {
        self.load_const_by_name(Reg::Temp, "_minus_zero_");
        self.amd.vunpckldd(ϕ(Reg::Temp), ϕ(Reg::Temp), ϕ(Reg::Temp));
        self.xor(dst, s1, Reg::Temp);
    }

    fn abs(&mut self, dst: Reg, s1: Reg) {
        self.amd.vmuldd(T1, ϕ(s1), ϕ(s1));
        self.amd.vhadddd(T1, T1, T1);
        self.amd.vsqrtsd(T2, T1);
        self.amd.vxorpd(T1, T1, T1);
        self.amd.vunpckldd(ϕ(dst), T2, T1);
    }

    fn root(&mut self, dst: Reg, s1: Reg) {
        self.amd.vmovq_reg_xmm(Amd::RAX, ϕ(s1));

        self.amd.vmuldd(T1, ϕ(s1), ϕ(s1));
        self.amd.vhadddd(T1, T1, T1);

        self.amd.vsqrtsd(T1, T1);
        self.amd.vmovsd_xmm_label(T0, "_minus_zero_");
        self.amd.vandnpd(T2, T0, ϕ(s1));
        self.amd.vaddsd(T1, T1, T2);
        self.amd.vmovsd_xmm_label(T0, "_half_");
        self.amd.vmulsd(T1, T1, T0);
        self.amd.vsqrtsd(T1, T1);

        self.amd.vunpckhdd(T2, ϕ(s1), ϕ(s1));
        self.amd.vdivsd(T2, T2, T1);
        self.amd.vmulsd(T2, T2, T0);

        self.amd.vcmpeqsd(T0, T2, T2);
        self.amd.vandpd(T2, T2, T0);

        self.amd.vunpckldd(ϕ(dst), T2, T1);

        let label = format!(".Y{}", self.amd.a.ip());
        // self.amd.mov_reg_mem(Amd::RAX, SP, 0);
        self.amd.or(Amd::RAX, Amd::RAX);
        self.amd.js(&label);
        self.amd.vshufdd(ϕ(dst), ϕ(dst), ϕ(dst), 1);
        self.set_label(&label);
    }

    fn real_root(&mut self, dst: Reg, s1: Reg) {
        self.amd.vxorpd(T1, T1, T1);
        self.amd.vsqrtsd(ϕ(dst), ϕ(s1));
        self.amd.vunpckldd(ϕ(dst), ϕ(dst), T1);
    }

    fn recip(&mut self, dst: Reg, s1: Reg) {
        self.amd.vshufdd(T1, ϕ(s1), ϕ(s1), 1);
        self.amd.vxorpd(T2, T2, T2);
        self.amd.vaddsubdd(T1, T2, T1);
        self.amd.vshufdd(T2, T1, T1, 1);

        self.amd.vmuldd(T1, ϕ(s1), ϕ(s1));
        self.amd.vhadddd(T1, T1, T1);
        self.amd.vdivdd(ϕ(dst), T2, T1);
    }

    fn half(&mut self, dst: Reg, s1: Reg) {
        self.load_const_by_name(Reg::Temp, "_half_");
        self.amd.vunpckldd(ϕ(Reg::Temp), ϕ(Reg::Temp), ϕ(Reg::Temp));
        self.amd.vmuldd(ϕ(dst), ϕ(s1), ϕ(Reg::Temp));
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
        binop!(self, vadddd, dst, s1, s2);
    }

    fn minus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, vsubdd, dst, s1, s2);
    }

    fn times(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.amd.vunpckldd(T1, ϕ(s1), ϕ(s1)); // duplicate real
        self.amd.vunpckhdd(T2, ϕ(s1), ϕ(s1)); // duplicate imag
        self.amd.vmuldd(T1, T1, ϕ(s2));
        self.amd.vmuldd(T2, T2, ϕ(s2));
        self.amd.vshufdd(T2, T2, T2, 1); // exchange real/imag
        self.amd.vaddsubdd(ϕ(dst), T1, T2);
    }

    fn divide(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.amd.vmuldd(T0, ϕ(s2), ϕ(s2));
        self.amd.vhadddd(T0, T0, T0);

        self.amd.vunpckldd(T1, ϕ(s1), ϕ(s1)); // duplicate real
        self.amd.vunpckhdd(T2, ϕ(s1), ϕ(s1)); // duplicate imag
        self.amd.vmuldd(T1, T1, ϕ(s2));
        self.amd.vmuldd(T2, T2, ϕ(s2));
        self.amd.vshufdd(T1, T1, T1, 1); // exchange real/imag
        self.amd.vaddsubdd(ϕ(dst), T2, T1);
        self.amd.vshufdd(ϕ(dst), ϕ(dst), ϕ(dst), 1);
        self.amd.vdivdd(ϕ(dst), ϕ(dst), T0);
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
        unreachable!()
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
        unreachable!()
    }

    fn support_times2(&self) -> bool {
        self.config.parallel_mul()
    }

    fn times2_loc(&mut self, d1: Reg, s1: Reg, l1: Loc, d2: Reg, s2: Reg, l2: Loc) {
        if d1 == s2 {
            match l1 {
                Loc::Mem(idx) => self.load_mem(Reg::Temp, idx),
                Loc::Param(idx) => self.load_param(Reg::Temp, idx),
                Loc::Stack(idx) => self.load_stack(Reg::Temp, idx),
            }
            self.times(d1, s1, Reg::Temp);

            match l2 {
                Loc::Mem(idx) => self.load_mem(Reg::Temp, idx),
                Loc::Param(idx) => self.load_param(Reg::Temp, idx),
                Loc::Stack(idx) => self.load_stack(Reg::Temp, idx),
            }
            self.times(d2, s2, Reg::Temp);
        } else {
            match l1 {
                Loc::Mem(idx) => self.amd.vmovdd_xmm_mem(T0, MEM, (idx * REG_SIZE) as i32),
                Loc::Param(idx) => self.amd.vmovdd_xmm_mem(T0, PARAMS, (idx * REG_SIZE) as i32),
                Loc::Stack(idx) => self.amd.vmovdd_xmm_mem(T0, SP, (idx * REG_SIZE) as i32),
            }

            match l2 {
                Loc::Mem(idx) => self
                    .amd
                    .vinsertf128_mem(T0, T0, MEM, (idx * REG_SIZE) as i32, 1),
                Loc::Param(idx) => {
                    self.amd
                        .vinsertf128_mem(T0, T0, PARAMS, (idx * REG_SIZE) as i32, 1)
                }
                Loc::Stack(idx) => self
                    .amd
                    .vinsertf128_mem(T0, T0, SP, (idx * REG_SIZE) as i32, 1),
            }

            self.amd.vinsertf128(ϕ(s1), ϕ(s1), ϕ(s2), 1);

            self.amd.vunpcklpd(T1, ϕ(s1), ϕ(s1)); // duplicate real
            self.amd.vunpckhpd(T2, ϕ(s1), ϕ(s1)); // duplicate imag
            self.amd.vmulpd(T1, T1, T0);
            self.amd.vmulpd(T2, T2, T0);
            self.amd.vshufpd(T2, T2, T2, 5); // exchange real/imag
            self.amd.vaddsubpd(ϕ(d1), T1, T2);

            self.amd.vextractf128(ϕ(d2), ϕ(d1), 1);
        }
    }

    fn real(&mut self, dst: Reg, s1: Reg) {
        self.amd.vxorpd(T1, T1, T1);
        self.amd.vunpckldd(ϕ(dst), ϕ(s1), T1);
    }

    fn imaginary(&mut self, dst: Reg, s1: Reg) {
        self.amd.vxorpd(T1, T1, T1);
        self.amd.vunpckhdd(ϕ(dst), ϕ(s1), T1);
    }

    fn conjugate(&mut self, dst: Reg, s1: Reg) {
        self.amd.vxorpd(T1, T1, T1);
        self.amd.vshufdd(ϕ(dst), ϕ(s1), ϕ(s1), 1);
        self.amd.vaddsubdd(ϕ(dst), T1, ϕ(dst));
        self.amd.vshufdd(ϕ(dst), ϕ(dst), ϕ(dst), 1);
    }

    fn complex(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.amd.vunpckldd(ϕ(dst), ϕ(s1), ϕ(s2));
    }

    fn gt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, vcmpnlesd, dst, s1, s2);
        binop!(self, vunpckldd, dst, dst, dst);
    }

    fn geq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, vcmpnltsd, dst, s1, s2);
        binop!(self, vunpckldd, dst, dst, dst);
    }

    fn lt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, vcmpltsd, dst, s1, s2);
        binop!(self, vunpckldd, dst, dst, dst);
    }

    fn leq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, vcmplesd, dst, s1, s2);
        binop!(self, vunpckldd, dst, dst, dst);
    }

    fn eq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, vcmpeqsd, dst, s1, s2);
        binop!(self, vunpckldd, dst, dst, dst);
    }

    fn neq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, vcmpneqsd, dst, s1, s2);
        binop!(self, vunpckldd, dst, dst, dst);
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
        self.amd.vunpckldd(ϕ(Reg::Temp), ϕ(Reg::Temp), ϕ(Reg::Temp));
        self.xor(dst, s1, Reg::Temp);
    }

    fn fused_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        self.amd.vunpckldd(T1, ϕ(s1), ϕ(s1)); // duplicate real
        self.amd.vunpckhdd(T2, ϕ(s1), ϕ(s1)); // duplicate imag
        self.amd.vfmadd132dd(T1, ϕ(s3), ϕ(s2));
        self.amd.vmuldd(T2, T2, ϕ(s2));
        self.amd.vshufdd(T2, T2, T2, 1); // exchange real/imag
        self.amd.vaddsubdd(ϕ(dst), T1, T2);
    }

    fn fused_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        self.amd.vunpckldd(T1, ϕ(s1), ϕ(s1)); // duplicate real
        self.amd.vunpckhdd(T2, ϕ(s1), ϕ(s1)); // duplicate imag
        self.amd.vfmsub132dd(T1, ϕ(s3), ϕ(s2));
        self.amd.vmuldd(T2, T2, ϕ(s2));
        self.amd.vshufdd(T2, T2, T2, 1); // exchange real/imag
        self.amd.vaddsubdd(ϕ(dst), T1, T2);
    }

    fn fused_neg_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        self.amd.vunpckldd(T1, ϕ(s1), ϕ(s1)); // duplicate real
        self.amd.vunpckhdd(T2, ϕ(s1), ϕ(s1)); // duplicate imag
        self.amd.vfnmadd132dd(T1, ϕ(s3), ϕ(s2));
        self.amd.vmuldd(T2, T2, ϕ(s2));
        self.amd.vshufdd(T1, T1, T1, 1);
        self.amd.vaddsubdd(T1, T1, T2);
        self.amd.vshufdd(ϕ(dst), T1, T1, 1); // exchange real/imag
    }

    fn fused_neg_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        self.amd.vunpckldd(T1, ϕ(s1), ϕ(s1)); // duplicate real
        self.amd.vunpckhdd(T2, ϕ(s1), ϕ(s1)); // duplicate imag
        self.amd.vfnmsub132dd(T1, ϕ(s3), ϕ(s2));
        self.amd.vmuldd(T2, T2, ϕ(s2));
        self.amd.vshufdd(T1, T1, T1, 1);
        self.amd.vaddsubdd(T1, T1, T2);
        self.amd.vshufdd(ϕ(dst), T1, T1, 1); // exchange real/imag
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
        self.vzeroupper();
        self.amd.call_indirect(&label);

        Ok(())
    }

    fn call_complex(&mut self, op: &str, num_args: usize) -> Result<()> {
        let label = format!("_func_{}_", op);

        if num_args == 2 {
            self.save_stack(Reg::Right, 4);
        }

        // loading the imaginary part of the argument into xmm1
        self.amd.vunpckhdd(1, 0, 0);

        self.vzeroupper();

        if cfg!(target_family = "windows") {
            self.amd.lea_mem(Amd::R8, SP, 32);
        } else {
            self.amd.lea_mem(Amd::RDI, SP, 32);
        }

        self.amd.call_indirect(&label);

        self.load_stack(Reg::Ret, 4);

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

        for i in (0..count_states).step_by(2) {
            self.amd
                .vmovdd_mem_xmm(MEM, (i as u32 * REG_SIZE) as i32, i as u8);
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

        for i in (0..count_states.min(8)).step_by(2) {
            self.amd
                .vmovdd_mem_xmm(MEM, (i as u32 * REG_SIZE) as i32, i as u8);
        }

        for i in (8..count_states).step_by(2) {
            let i = i as u32;
            // the offset of the fifth or eight arguments:
            // +4 for the 32-byte home
            // +1 for the return address in the stack
            // +1 for RBP in the stack
            // -4 for the first four arguments passed in XMM0-XMM3
            self.amd
                .vmovdd_xmm_mem(0, MEM, (frame_size + (i + 2) * REG_SIZE) as i32);
            self.amd.vmovdd_mem_xmm(MEM, (i * REG_SIZE) as i32, 0);
        }
    }

    fn epilogue_fast(
        &mut self,
        _cap: usize,
        _count_states: usize,
        _count_obs: usize,
        idx_ret: i32,
    ) {
        self.vzeroupper();
        self.amd.vmovdd_xmm_mem(0, MEM, idx_ret * REG_SIZE as i32);
        self.vzeroupper();
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

impl AmdComplexGenerator {
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
            self.amd.vmovsd_xmm_indexed(RET, Amd::RAX, IDX, 8);
            self.amd.vmovsd_mem_xmm(MEM, k as i32, RET);
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
            self.amd.vmovsd_xmm_mem(RET, MEM, k as i32);
            self.amd.vmovsd_indexed_xmm(Amd::RCX, IDX, 8, RET);
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

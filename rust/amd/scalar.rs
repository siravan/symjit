use crate::code::Func;
use crate::config::{Config, SPILL_AREA};
use crate::generator::{FuncletType, Generator};
use crate::symbol::Loc;
use crate::utils::align_stack;
use crate::utils::{is_external_func, reg, DataType, Reg};
use anyhow::Result;

use super::asm::*;
use super::*;

const REG_SIZE: u32 = 8;

/*
macro_rules! binop {
    ($self:ident, $avx:ident, $dst:expr, $s1: expr, $s2: expr) => {
        $self.amd.$avx(ϕ($dst), ϕ($s1), ϕ($s2));
    };
}

macro_rules! uniop {
    ($self:ident, $avx:ident, $dst:expr, $s1: expr) => {
        $self.amd.$avx(ϕ($dst), ϕ($s1));
    };
}

macro_rules! roundop {
    ($self:ident, $dst:expr, $s1: expr, $mode: expr) => {
        $self.amd.vroundsd(ϕ($dst), ϕ($s1), $mode);
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
*/

pub struct AmdScalarGenerator {
    amd: Amd,
    config: Config,
    last_load: usize,
}

impl AmdScalarGenerator {
    pub fn new(config: Config) -> AmdScalarGenerator {
        AmdScalarGenerator {
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
        self.amd.vmovsd_xmm_label(ϕ(dst), label);
    }

    fn call_external(&mut self, op: &str, num_args: usize) -> Result<()> {
        let cap = SPILL_AREA as u32;

        // self.amd.mov_reg_label(ARGS[0], &format!("_env_{}_", op));
        //self.amd.lea_mem(ARGS[1], STACK, (cap * REG_SIZE) as i32);
        //self.amd.mov_imm(ARGS[2], num_args as u32);
        //self.amd.lea_mem(ARGS[3], STACK, 4 * REG_SIZE as i32);

        amd! {mov r(ARGS[0]), [&format!("_env_{}_", op)]; self.amd};
        amd! {lea r(ARGS[1]), [r(STACK) + cap * REG_SIZE]; self.amd};
        amd! {mov r(ARGS[2]), num_args; self.amd};
        amd! {lea r(ARGS[3]), [r(STACK) + 4 * REG_SIZE]; self.amd};

        amd! {vzeroupper; self.amd};

        //self.amd.call_indirect(&format!("_func_{}_", op));
        amd! {call [rip + &format!("_func_{}_", op)];self.amd};
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

impl Generator for AmdScalarGenerator {
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
            // self.amd.nop();
            amd! {nop ; self.amd};
            n += 1
        }
    }

    fn set_label(&mut self, label: &str) {
        self.amd.a.set_label(label);
    }

    fn branch(&mut self, label: &str) {
        // self.amd.xor(Amd::RAX, Amd::RAX);
        // self.amd.jz(label);
        amd! {xor r(Amd::RAX), r(Amd::RAX); self.amd};
        amd! {jz label; self.amd};
    }

    /// jump to label if cond == is_else
    /// note that `is_else` is not the correct name anymore and should be
    /// changed to `expectation`
    fn branch_if(&mut self, cond: Reg, label: &str, is_else: bool) {
        // self.amd.vucomisd(ϕ(cond), ϕ(cond));
        amd! {vucomisd xmm(ϕ(cond)), xmm(ϕ(cond)); self.amd};
        /*
         * if is_else (expectation) is true, jump if cond is true (all-1, NaN).
         * In this situation, vucomisd returns an unordered result, setting
         * PF = 1 (jpe)
         */
        if is_else {
            // self.amd.jpe(label);
            amd! {jpe label; self.amd};
        } else {
            // self.amd.jpo(label);
            amd! {jpo label; self.amd};
        }
    }

    fn fuse_load_math(&mut self) {
        fuse_load_math(&mut self.amd, self.last_load);
    }

    //***********************************

    fn fmov(&mut self, dst: Reg, s1: Reg) {
        if dst != s1 {
            // self.amd.vmovapd(ϕ(dst), ϕ(s1));
            amd! {vmovapd xmm(ϕ(dst)), xmm(ϕ(s1)); self.amd};
        }
    }

    fn fxchg(&mut self, s1: Reg, s2: Reg) {
        // self.amd.vxorpd(ϕ(s1), ϕ(s1), ϕ(s2));
        // self.amd.vxorpd(ϕ(s2), ϕ(s1), ϕ(s2));
        // self.amd.vxorpd(ϕ(s1), ϕ(s1), ϕ(s2));
        amd! {vxorpd xmm(ϕ(s1)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
        amd! {vxorpd xmm(ϕ(s2)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
        amd! {vxorpd xmm(ϕ(s1)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
    }

    fn load_const(&mut self, dst: Reg, idx: u32) {
        self.last_load = self.amd.a.ip();
        let label = format!("_const_{}_", idx);
        // self.amd.vmovsd_xmm_label(ϕ(dst), label.as_str());
        amd! {vmovsd xmm(ϕ(dst)), label.as_str(); self.amd};
    }

    fn load_mem(&mut self, dst: Reg, idx: u32) {
        self.last_load = self.amd.a.ip();
        // self.amd.vmovsd_xmm_mem(ϕ(dst), MEM, (idx * REG_SIZE) as i32);
        amd! {vmovsd xmm(ϕ(dst)), [r(MEM) + idx * REG_SIZE]; self.amd};
    }

    fn save_mem(&mut self, dst: Reg, idx: u32) {
        // self.amd.vmovsd_mem_xmm(MEM, (idx * REG_SIZE) as i32, ϕ(dst));
        amd! {vmovsd [r(MEM) + idx * REG_SIZE], xmm(ϕ(dst)); self.amd};
    }

    fn save_mem_result(&mut self, idx: u32) {
        self.save_mem(Reg::Ret, idx);
    }

    fn load_param(&mut self, dst: Reg, idx: u32) {
        self.last_load = self.amd.a.ip();
        // self.amd.vmovsd_xmm_mem(ϕ(dst), PARAMS, (idx * REG_SIZE) as i32);
        amd! {vmovsd xmm(ϕ(dst)), [r(PARAMS) + (idx * REG_SIZE)]; self.amd};
    }

    fn load_stack(&mut self, dst: Reg, idx: u32) {
        self.last_load = self.amd.a.ip();
        // self.amd.vmovsd_xmm_mem(ϕ(dst), STACK, (idx * REG_SIZE) as i32);
        amd! {vmovsd xmm(ϕ(dst)), [r(STACK) + idx * REG_SIZE]; self.amd};
    }

    fn save_stack(&mut self, dst: Reg, idx: u32) {
        // self.amd.vmovsd_mem_xmm(STACK, (idx * REG_SIZE) as i32, ϕ(dst));
        amd! {vmovsd [r(STACK) + idx * REG_SIZE], xmm(ϕ(dst)); self.amd};
    }

    fn load_mem_complex(&mut self, xd: Reg, yd: Reg, idx: u32) {
        // self.amd.vmovdd_xmm_mem(ϕ(xd), MEM, (idx * REG_SIZE) as i32);
        // self.amd.vshufdd(ϕ(yd), ϕ(xd), ϕ(xd), 1);

        amd! {vmovupd xmm(ϕ(xd)), [r(MEM) + idx * REG_SIZE]; self.amd};
        amd! {vshufpd xmm(ϕ(yd)), xmm(ϕ(xd)), xmm(ϕ(xd)), 1; self.amd};
    }

    fn save_mem_complex(&mut self, xs: Reg, ys: Reg, idx: u32) {
        // self.amd.vunpckldd(ϕ(xs), ϕ(xs), ϕ(ys));
        // self.amd.vmovdd_mem_xmm(MEM, (idx * REG_SIZE) as i32, ϕ(xs));

        amd! {vunpcklpd xmm(ϕ(xs)), xmm(ϕ(xs)), xmm(ϕ(ys)); self.amd};
        amd! {vmovupd [r(MEM) + idx * REG_SIZE], xmm(ϕ(xs)); self.amd};
    }

    fn load_param_complex(&mut self, xd: Reg, yd: Reg, idx: u32) {
        // self.amd.vmovdd_xmm_mem(ϕ(xd), PARAMS, (idx * REG_SIZE) as i32);
        // self.amd.vshufdd(ϕ(yd), ϕ(xd), ϕ(xd), 1);

        amd! {vmovupd xmm(ϕ(xd)), [r(PARAMS) + idx * REG_SIZE]; self.amd};
        amd! {vshufpd xmm(ϕ(yd)), xmm(ϕ(xd)), xmm(ϕ(xd)), 1; self.amd};
    }

    fn load_stack_complex(&mut self, xd: Reg, yd: Reg, idx: u32) {
        // self.amd.vmovdd_xmm_mem(ϕ(xd), STACK, (idx * REG_SIZE) as i32);
        // self.amd.vshufdd(ϕ(yd), ϕ(xd), ϕ(xd), 1);

        amd! {vmovupd xmm(ϕ(xd)), [r(STACK) + idx * REG_SIZE]; self.amd};
        amd! {vshufpd xmm(ϕ(yd)), xmm(ϕ(xd)), xmm(ϕ(xd)), 1; self.amd};
    }

    fn save_stack_complex(&mut self, xs: Reg, ys: Reg, idx: u32) {
        // self.amd.vunpckldd(ϕ(xs), ϕ(xs), ϕ(ys));
        // self.amd.vmovdd_mem_xmm(STACK, (idx * REG_SIZE) as i32, ϕ(xs));

        amd! {vunpcklpd xmm(ϕ(xs)), xmm(ϕ(xs)), xmm(ϕ(ys)); self.amd};
        amd! {vmovupd [r(STACK) + idx * REG_SIZE], xmm(ϕ(xs)); self.amd};
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
        // uniop!(self, vsqrtsd, dst, s1);
        amd! {vsqrtsd xmm(ϕ(dst)), xmm(ϕ(s1)); self.amd};
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
        // roundop!(self, dst, s1, RoundingMode::Round);
        amd! {vroundsd xmm(ϕ(dst)), xmm(ϕ(s1)), RoundingMode::Round; self.amd};
    }

    fn floor(&mut self, dst: Reg, s1: Reg) {
        // roundop!(self, dst, s1, RoundingMode::Floor);
        amd! {vroundsd xmm(ϕ(dst)), xmm(ϕ(s1)), RoundingMode::Floor; self.amd};
    }

    fn ceiling(&mut self, dst: Reg, s1: Reg) {
        // roundop!(self, dst, s1, RoundingMode::Ceiling);
        amd! {vroundsd xmm(ϕ(dst)), xmm(ϕ(s1)), RoundingMode::Ceiling; self.amd};
    }

    fn trunc(&mut self, dst: Reg, s1: Reg) {
        // roundop!(self, dst, s1, RoundingMode::Trunc);
        amd! {vroundsd xmm(ϕ(dst)), xmm(ϕ(s1)), RoundingMode::Trunc; self.amd};
    }

    fn frac(&mut self, dst: Reg, s1: Reg) {
        self.floor(Reg::Temp, s1);
        self.minus(dst, s1, Reg::Temp);
    }

    fn plus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vaddsd xmm(ϕ(dst)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
    }

    fn minus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vsubsd xmm(ϕ(dst)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
    }

    fn times(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vmulsd xmm(ϕ(dst)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
    }

    fn divide(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vdivsd xmm(ϕ(dst)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
    }

    fn times_complex(&mut self, xd: Reg, yd: Reg, x1: Reg, y1: Reg, x2: Reg, y2: Reg) -> bool {
        let xt = Reg::Gen(2);
        let yt = Reg::Gen(3);

        /*
            self.amd.vunpckldd(ϕ(x1), ϕ(x1), ϕ(x1));
            self.amd.vunpckldd(ϕ(y1), ϕ(y1), ϕ(y1));
            self.amd.vunpckldd(ϕ(yt), ϕ(x2), ϕ(y2));
            self.amd.vmuldd(ϕ(xt), ϕ(x1), ϕ(yt));
            self.amd.vmuldd(ϕ(yt), ϕ(y1), ϕ(yt));
            self.amd.vshufdd(ϕ(xd), ϕ(yt), ϕ(yt), 1);
            self.amd.vaddsubdd(ϕ(xd), ϕ(xt), ϕ(xd));
            self.amd.vshufdd(ϕ(yd), ϕ(xd), ϕ(xd), 1);
        */

        self.times(xt, y1, y2);
        self.fused_mul_sub(xt, x1, x2, xt);
        self.times(yt, x1, y2);
        self.fused_mul_add(yd, x2, y1, yt);
        self.fmov(xd, xt);

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
        // binop!(self, vcmpnlesd, dst, s1, s2);
        amd! {vcmpnlesd xmm(ϕ(dst)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
    }

    fn geq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        // binop!(self, vcmpnltsd, dst, s1, s2);
        amd! {vcmpnltsd xmm(ϕ(dst)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
    }

    fn lt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        // binop!(self, vcmpltsd, dst, s1, s2);
        amd! {vcmpltsd xmm(ϕ(dst)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
    }

    fn leq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        // binop!(self, vcmplesd, dst, s1, s2);
        amd! {vcmplesd xmm(ϕ(dst)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
    }

    fn eq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        // binop!(self, vcmpeqsd, dst, s1, s2);
        amd! {vcmpeqsd xmm(ϕ(dst)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
    }

    fn neq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        // binop!(self, vcmpneqsd, dst, s1, s2);
        amd! {vcmpneqsd xmm(ϕ(dst)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
    }

    fn and(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        // binop!(self, vandpd, dst, s1, s2);
        amd! {vandpd xmm(ϕ(dst)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
    }

    fn andnot(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        // binop!(self, vandnpd, dst, s1, s2);
        amd! {vandnpd xmm(ϕ(dst)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
    }

    fn or(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        // binop!(self, vorpd, dst, s1, s2);
        amd! {vorpd xmm(ϕ(dst)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
    }

    fn xor(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        // binop!(self, vxorpd, dst, s1, s2);
        amd! {vxorpd xmm(ϕ(dst)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
    }

    fn not(&mut self, dst: Reg, s1: Reg) {
        self.load_const_by_name(Reg::Temp, "_all_ones_");
        self.xor(dst, s1, Reg::Temp);
    }

    fn fused_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        if dst == s1 {
            amd! {vfmadd132sd xmm(ϕ(s1)), xmm(ϕ(s2)), xmm(ϕ(s3)); self.amd};
        } else if dst == s2 {
            amd! {vfmadd213sd xmm(ϕ(s1)), xmm(ϕ(s2)), xmm(ϕ(s3)); self.amd};
        } else if dst == s3 {
            amd! {vfmadd231sd xmm(ϕ(s1)), xmm(ϕ(s2)), xmm(ϕ(s3)); self.amd};
        } else {
            amd! {vmovapd xmm(ϕ(dst)), xmm(ϕ(s1)); self.amd};
            amd! {vfmadd132sd xmm(ϕ(s1)), xmm(ϕ(s2)), xmm(ϕ(s3)); self.amd};
        }
        // fuseop!(self, vfmadd132sd, vfmadd213sd, vfmadd231sd, dst, s1, s2, s3);
    }

    fn fused_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        if dst == s1 {
            amd! {vfmsub132sd xmm(ϕ(s1)), xmm(ϕ(s2)), xmm(ϕ(s3)); self.amd};
        } else if dst == s2 {
            amd! {vfmsub213sd xmm(ϕ(s1)), xmm(ϕ(s2)), xmm(ϕ(s3)); self.amd};
        } else if dst == s3 {
            amd! {vfmsub231sd xmm(ϕ(s1)), xmm(ϕ(s2)), xmm(ϕ(s3)); self.amd};
        } else {
            amd! {vmovapd xmm(ϕ(dst)), xmm(ϕ(s1)); self.amd};
            amd! {vfmsub132sd xmm(ϕ(s1)), xmm(ϕ(s2)), xmm(ϕ(s3)); self.amd};
        }
        // fuseop!(self, vfmsub132sd, vfmsub213sd, vfmsub231sd, dst, s1, s2, s3);
    }

    fn fused_neg_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        if dst == s1 {
            amd! {vfnmadd132sd xmm(ϕ(s1)), xmm(ϕ(s2)), xmm(ϕ(s3)); self.amd};
        } else if dst == s2 {
            amd! {vfnmadd213sd xmm(ϕ(s1)), xmm(ϕ(s2)), xmm(ϕ(s3)); self.amd};
        } else if dst == s3 {
            amd! {vfnmadd231sd xmm(ϕ(s1)), xmm(ϕ(s2)), xmm(ϕ(s3)); self.amd};
        } else {
            amd! {vmovapd xmm(ϕ(dst)), xmm(ϕ(s1)); self.amd};
            amd! {vfnmadd132sd xmm(ϕ(s1)), xmm(ϕ(s2)), xmm(ϕ(s3)); self.amd};
        }
        /*
        fuseop!(
            self,
            vfnmadd132sd,
            vfnmadd213sd,
            vfnmadd231sd,
            dst,
            s1,
            s2,
            s3
        );
        */
    }

    fn fused_neg_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        if dst == s1 {
            amd! {vfnmsub132sd xmm(ϕ(s1)), xmm(ϕ(s2)), xmm(ϕ(s3)); self.amd};
        } else if dst == s2 {
            amd! {vfnmsub213sd xmm(ϕ(s1)), xmm(ϕ(s2)), xmm(ϕ(s3)); self.amd};
        } else if dst == s3 {
            amd! {vfnmsub231sd xmm(ϕ(s1)), xmm(ϕ(s2)), xmm(ϕ(s3)); self.amd};
        } else {
            amd! {vmovapd xmm(ϕ(dst)), xmm(ϕ(s1)); self.amd};
            amd! {vfnmsub132sd xmm(ϕ(s1)), xmm(ϕ(s2)), xmm(ϕ(s3)); self.amd};
        }
        /*
        fuseop!(
            self,
            vfnmsub132sd,
            vfnmsub213sd,
            vfnmsub231sd,
            dst,
            s1,
            s2,
            s3
        );
        */
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
        amd! {vzeroupper; self.amd};

        // self.amd.call_indirect(&label);
        amd! {call &label; self.amd};

        Ok(())
    }

    fn call_complex(&mut self, op: &str, num_args: usize) -> Result<()> {
        let label = format!("_func_{}_", op);

        if num_args == 2 {
            self.save_stack(Reg::Gen(0), 4);
            self.save_stack(Reg::Gen(1), 5);
        }

        amd! {vzeroupper; self.amd};

        if cfg!(target_family = "windows") {
            // self.amd.lea_mem(Amd::R8, STACK, 32);
            amd! {lea r(Amd::R8), [r(STACK) + 32]; self.amd};
        } else {
            // self.amd.lea_mem(Amd::RDI, STACK, 32);
            amd! {lea r(Amd::RDI), [r(STACK) + 32]; self.amd};
        }

        // self.amd.call_indirect(&label);
        amd! {call &label; self.amd};

        self.load_stack(Reg::Ret, 4);
        self.load_stack(Reg::Temp, 5);

        Ok(())
    }

    fn call_funclet(&mut self, label: &str) {
        // self.amd.call_relative(label);
        amd! {call [rip + &label]; self.amd};
    }

    fn ret(&mut self) {
        // self.amd.ret();
        amd! {ret; self.amd};
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
        // self.amd.push(Amd::RBP);
        amd! {push r(Amd::RBP); self.amd};

        let frame_size = align_stack((count_states + count_obs) as u32 * REG_SIZE);

        // sub_rsp(&mut self.amd, frame_size);
        // self.amd.mov(MEM, STACK);
        // sub_rsp(&mut self.amd, align_stack(cap as u32 * REG_SIZE));

        amd! {sub rsp, frame_size; self.amd};
        amd! {mov r(MEM), r(STACK); self.amd};
        amd! {sub rsp, align_stack(cap as u32 * REG_SIZE); self.amd};

        for i in 0..count_states {
            // self.amd.vmovsd_mem_xmm(MEM, (i * 8) as i32, i as u8);
            amd! {vmovsd [r(MEM) + i * 8], xmm(i as u8); self.amd};
        }
    }

    #[cfg(target_family = "windows")]
    fn prologue_fast(&mut self, cap: usize, count_states: usize, count_obs: usize) {
        self.amd.push(Amd::RBP);
        amd! {push r(Amd::RBP); self.amd};

        let frame_size = align_stack((count_states + count_obs) as u32 * REG_SIZE);

        // sub_rsp(&mut self.amd, frame_size);
        // self.amd.mov(MEM, STACK);
        // sub_rsp(&mut self.amd, align_stack(cap as u32 * REG_SIZE));

        amd! {sub rsp, frame_size; self.amd};
        amd! {mov r(MEM), r(STACK); self.amd};
        amd! {sub rsp, align_stack(cap as u32 * REG_SIZE); self.amd};

        for i in 0..count_states.min(4) {
            // self.amd.vmovsd_mem_xmm(MEM, (i as u32 * REG_SIZE) as i32, i as u8);
            amd! {vmovsd [r(MEM) + i * REG_SIZE], xmm(i as u8); self.amd};
        }

        for i in 4..count_states {
            let i = i as u32;
            // the offset of the fifth or eight arguments:
            // +4 for the 32-byte home
            // +1 for the return address in the stack
            // +1 for RBP in the stack
            // -4 for the first four arguments passed in XMM0-XMM3

            // self.amd.vmovsd_xmm_mem(0, MEM, (frame_size + (i + 2) * REG_SIZE) as i32);
            // self.amd.vmovsd_mem_xmm(MEM, (i * REG_SIZE) as i32, 0);
            amd! {vmovsd xmm(0), [r(MEM) + frame_size + (i + 2) * REG_SIZE]; self.amd};
            amd! {vmodsd [r(MEM), i * REG_SIZE], xmm(0); self.amd};
        }
    }

    fn epilogue_fast(&mut self, cap: usize, count_states: usize, count_obs: usize, idx_ret: i32) {
        amd! {vzeroupper; self.amd};
        // self.amd.vmovsd_xmm_mem(0, MEM, idx_ret * REG_SIZE as i32);
        amd! {vmovsd xmm(0), [r(MEM) + idx_ret * REG_SIZE as i32]; self.amd};

        let total_size = align_stack(cap as u32 * REG_SIZE)
            + align_stack((count_states + count_obs) as u32 * REG_SIZE);
        // add_rsp(&mut self.amd, total_size);
        // self.amd.pop(Amd::RBP);
        // self.amd.ret();

        amd! {add rsp, total_size; self.amd};
        amd! {pop r(Amd::RBP); self.amd};
        amd! {ret; self.amd};
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

        // self.amd.push(Amd::RBP);
        amd! {push r(Amd::RBP); self.amd};
        save_nonvolatile_regs(&mut self.amd);

        // self.amd.mov(MEM, ARGS[0]); // first arg = mem if direct mode, otherwise null
        // self.amd.mov(STATES, ARGS[1]); // second arg = states+obs if indirect mode, otherwise null
        // self.amd.mov(IDX, ARGS[2]); // third arg = index if indirect mode
        // self.amd.mov(PARAMS, ARGS[3]); // fourth arg = params

        amd! {mov r(MEM), r(ARGS[0]); self.amd};
        amd! {mov r(STATES), r(ARGS[1]) ; self.amd};
        amd! {mov r(IDX), r(ARGS[2]); self.amd};
        amd! {mov r(PARAMS), r(ARGS[3]) ; self.amd};

        // self.amd.or(STATES, STATES);
        // self.amd.jz("@main");

        amd! {or r(STATES), r(STATES); self.amd};
        amd! {jz "@main"; self.amd};

        let frame_size = align_stack((count_states + count_obs) as u32 * REG_SIZE);

        // sub_rsp(&mut self.amd, frame_size);
        // self.amd.mov(MEM, STACK); // in indirect mode, MEM is allocated on the stack

        amd! {sub rsp, frame_size; self.amd};
        amd! {mov r(MEM), r(STACK); self.amd};

        for i in 0..count_states {
            let k = i as u32 * REG_SIZE;

            // self.amd.mov_reg_mem(Amd::RAX, STATES, 2 * 8 * i as i32);
            // self.amd.vmovsd_xmm_indexed(RET, Amd::RAX, IDX, 8);
            // self.amd.vmovsd_mem_xmm(MEM, k as i32, RET);

            amd! {mov r(Amd::RAX), [r(STATES) + 2 * 8 * i as i32]; self.amd};
            amd! {vmovsd xmm(RET), [r(Amd::RAX) + r(IDX) * 8] ; self.amd};
            amd! {vmovsd [r(MEM) + k], xmm(RET); self.amd};
        }

        // may save idx (RDX) as double in RBP + 8/32 * count_states

        self.set_label("@main");
        // sub_rsp(&mut self.amd, align_stack(cap as u32 * REG_SIZE));
        amd! {sub rsp, align_stack(cap as u32 * REG_SIZE); self.amd};
    }

    fn epilogue_indirect(
        &mut self,
        cap: usize,
        count_states: usize,
        count_obs: usize,
        count_params: usize,
    ) {
        // self.amd.xor(Amd::RAX, Amd::RAX);
        amd! {xor r(Amd::RAX), r(Amd::RAX); self.amd};
        self.set_label("@epilogue");

        if self.config.symbolica() {
            return self.epilogue_symbolica(cap, count_params, count_obs);
        }

        // add_rsp(&mut self.amd, align_stack(cap as u32 * REG_SIZE));
        // self.amd.or(STATES, STATES);
        // self.amd.jz("@done");

        amd! {add rsp, align_stack(cap as u32 * REG_SIZE) ; self.amd};
        amd! {or r(STATES), r(STATES); self.amd};
        amd! {jz "@done"; self.amd};

        for i in 0..count_obs {
            let k = (count_states + i) as u32 * REG_SIZE;
            // self.amd.mov_reg_mem(Amd::RCX, STATES, 2 * 8 * (count_states + i) as i32);
            // self.amd.vmovsd_xmm_mem(RET, MEM, k as i32);
            // self.amd.vmovsd_indexed_xmm(Amd::RCX, IDX, 8, RET);

            amd! {mov r(Amd::RCX), [r(STATES) + 2 * 8 * (count_states + i) as i32]; self.amd};
            amd! {vmovsd xmm(RET), [r(MEM) + k]; self.amd};
            amd! {vmovsd [r(Amd::RCX) + r(IDX) * 8], xmm(RET); self.amd};
        }

        let frame_size = align_stack((count_states + count_obs) as u32 * REG_SIZE);
        // add_rsp(&mut self.amd, frame_size);
        amd! {add rsp, frame_size; self.amd};
        self.set_label("@done");

        amd! {vzeroupper; self.amd};

        load_nonvolatile_regs(&mut self.amd);
        // self.amd.pop(Amd::RBP);
        // self.amd.ret();

        amd! {pop r(Amd::RBP) ; self.amd};
        amd! {ret ; self.amd};
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

impl AmdScalarGenerator {
    fn prologue_symbolica(&mut self, cap: usize, _count_params: usize, _count_obs: usize) {
        // self.amd.push(Amd::RBP);
        amd! {push r(Amd::RBP); self.amd};
        save_nonvolatile_regs(&mut self.amd);

        // self.amd.mov(MEM, ARGS[0]); // first arg = mem if direct mode, otherwise null
        // self.amd.mov(STATES, ARGS[1]); // second arg = states+obs if indirect mode, otherwise null
        // self.amd.mov(IDX, ARGS[2]); // third arg = index if indirect mode
        // self.amd.mov(PARAMS, ARGS[3]); // fourth arg = params

        amd! {mov r(MEM), r(ARGS[0]); self.amd};
        amd! {mov r(STATES), r(ARGS[1]) ; self.amd};
        amd! {mov r(IDX), r(ARGS[2]); self.amd};
        amd! {mov r(PARAMS), r(ARGS[3]); self.amd};

        // sub_rsp(&mut self.amd, align_stack(cap as u32 * REG_SIZE));
        amd! {sub rsp, align_stack(cap as u32 * REG_SIZE); self.amd};
    }

    fn epilogue_symbolica(&mut self, cap: usize, _count_params: usize, _count_obs: usize) {
        // add_rsp(&mut self.amd, align_stack(cap as u32 * REG_SIZE));
        amd! {add rsp, align_stack(cap as u32 * REG_SIZE); self.amd};

        amd! {vzeroupper; self.amd};

        load_nonvolatile_regs(&mut self.amd);

        // self.amd.pop(Amd::RBP);
        // self.amd.ret();

        amd! {pop r(Amd::RBP); self.amd};
        amd! {ret; self.amd};
    }
}

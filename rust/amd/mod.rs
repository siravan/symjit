use crate::generator::Generator;
use crate::utils::align_stack;
use crate::utils::{reg, DataType, Reg};

mod asm;
mod fused;

use asm::{Amd, RoundingMode};

const RET: u8 = 0;

macro_rules! binop {
    ($self:ident, $sse:ident, $avx:ident, $simd:ident, $dst:expr, $s1: expr, $s2: expr, $com:ident) => {
        match $self.family {
            AmdFamily::AvxScalar => $self.amd.$avx(ϕ($dst), ϕ($s1), ϕ($s2)),
            AmdFamily::AvxVector => $self.amd.$simd(ϕ($dst), ϕ($s1), ϕ($s2)),
            AmdFamily::SSEScalar => {
                let (x, y) = $self.shrink($dst, $s1, $s2, $com);
                $self.amd.$sse(ϕ(x), ϕ(y));
            }
        }
    };
}

macro_rules! select {
    ($self:ident, $sse:ident, $avx:ident, $simd:ident, $x:expr, $y: expr, $z: expr, $w: expr) => {
        match $self.family {
            AmdFamily::AvxScalar => $self.amd.$avx($x, $y, $z, $w),
            AmdFamily::AvxVector => $self.amd.$simd($x, $y, $z, $w),
            AmdFamily::SSEScalar => $self.amd.$sse($x, $y, $z, $w),
        }
    };
    ($self:ident, $sse:ident, $avx:ident, $simd:ident, $x:expr, $y: expr, $z: expr) => {
        match $self.family {
            AmdFamily::AvxScalar => $self.amd.$avx($x, $y, $z),
            AmdFamily::AvxVector => $self.amd.$simd($x, $y, $z),
            AmdFamily::SSEScalar => $self.amd.$sse($x, $y, $z),
        }
    };
    ($self:ident, $sse:ident, $avx:ident, $simd:ident, $x:expr, $y: expr) => {
        match $self.family {
            AmdFamily::AvxScalar => $self.amd.$avx($x, $y),
            AmdFamily::AvxVector => $self.amd.$simd($x, $y),
            AmdFamily::SSEScalar => $self.amd.$sse($x, $y),
        }
    };
}

macro_rules! uniop {
    ($self:ident, $sse:ident, $avx:ident, $simd:ident, $dst:expr, $s1: expr) => {
        select!($self, $sse, $avx, $simd, ϕ($dst), ϕ($s1));
    };
}

macro_rules! roundop {
    ($self:ident, $dst:expr, $s1: expr, $mode: expr) => {
        select!($self, roundsd, vroundsd, vroundpd, ϕ($dst), ϕ($s1), $mode);
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

pub enum AmdFamily {
    AvxScalar,
    AvxVector,
    SSEScalar,
}

pub struct AmdGenerator {
    amd: Amd,
    family: AmdFamily,
}

const MEM: u8 = Amd::RBP;
const STATES: u8 = Amd::R13;
const IDX: u8 = Amd::R12;
const PARAMS: u8 = Amd::RBX;

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
        Reg::Gen(dst) => dst + 2,
        Reg::Static(..) => panic!("passing static registers to codegen"),
    }
}

impl AmdGenerator {
    pub fn new(family: AmdFamily) -> AmdGenerator {
        AmdGenerator {
            amd: Amd::new(DataType::F64),
            family,
        }
    }

    fn reg_size(&self) -> u32 {
        match self.family {
            AmdFamily::AvxScalar | AmdFamily::SSEScalar => 8,
            AmdFamily::AvxVector => 32,
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
        select!(
            self,
            movsd_xmm_label,
            vmovsd_xmm_label,
            vbroadcastsd_label,
            ϕ(dst),
            label
        );
    }

    fn vzeroupper(&mut self) {
        match self.family {
            AmdFamily::AvxScalar | AmdFamily::AvxVector => self.amd.vzeroupper(),
            AmdFamily::SSEScalar => {}
        }
    }

    fn call_vector_unary(&mut self, label: &str) {
        // reserves 64 bytes in the stack
        // 32 bytes for shadow store (mandatory in Windows)
        // 32 bytes to save ymm0
        self.amd.sub_rsp(32 * 2);
        self.amd.vmovpd_mem_ymm(Amd::RSP, 32, 0);

        self.vzeroupper();

        for i in 0..4 {
            self.amd.movsd_xmm_mem(0, Amd::RSP, 32 + i * 8);
            //self.amd.call(Amd::R12);
            self.amd.call_indirect(label);
            self.amd.movsd_mem_xmm(Amd::RSP, 32 + i * 8, 0);
        }

        self.amd.vmovpd_ymm_mem(0, Amd::RSP, 32);
        self.amd.add_rsp(32 * 2);
    }

    fn call_vector_binary(&mut self, label: &str) {
        // reserves 96 bytes in the stack
        // 32 bytes for shadow store (mandatory in Windows)
        // 32 bytes to save ymm0
        // 32 bytes to save ymm1
        self.amd.sub_rsp(32 * 3);
        self.amd.vmovpd_mem_ymm(Amd::RSP, 32, 0);
        self.amd.vmovpd_mem_ymm(Amd::RSP, 64, 1);

        self.vzeroupper();

        for i in 0..4 {
            self.amd.movsd_xmm_mem(0, Amd::RSP, 32 + i * 8);
            self.amd.movsd_xmm_mem(1, Amd::RSP, 64 + i * 8);
            //self.amd.call(Amd::R12);
            self.amd.call_indirect(label);
            self.amd.movsd_mem_xmm(Amd::RSP, 32 + i * 8, 0);
        }

        self.amd.vmovpd_ymm_mem(0, Amd::RSP, 32);
        self.amd.add_rsp(32 * 3);
    }

    fn predefined_consts(&mut self) {
        self.align();

        self.set_label("_minus_zero_");
        self.append_quad((-0.0f64).to_bits());

        self.set_label("_one_");
        self.append_quad(1.0f64.to_bits());

        self.set_label("_all_ones_");
        self.append_quad(0xffffffffffffffff);
    }

    fn frame_size(&self, cap: u32) -> u32 {
        align_stack(self.reg_size() * cap + 8) - 8
    }

    fn save_nonvolatile_regs(&mut self) {
        if cfg!(target_family = "windows") {
            self.amd.mov_mem_reg(Amd::RSP, 0x08, MEM);
            self.amd.mov_mem_reg(Amd::RSP, 0x10, PARAMS);
            self.amd.mov_mem_reg(Amd::RSP, 0x18, IDX);
            self.amd.mov_mem_reg(Amd::RSP, 0x20, STATES);
        } else {
            self.amd.sub_rsp(32);
            self.amd.mov_mem_reg(Amd::RSP, 0x00, MEM);
            self.amd.mov_mem_reg(Amd::RSP, 0x08, PARAMS);
            self.amd.mov_mem_reg(Amd::RSP, 0x10, IDX);
            self.amd.mov_mem_reg(Amd::RSP, 0x18, STATES);
        }
    }

    fn load_nonvolatile_regs(&mut self) {
        if cfg!(target_family = "windows") {
            self.amd.mov_reg_mem(STATES, Amd::RSP, 0x20);
            self.amd.mov_reg_mem(IDX, Amd::RSP, 0x18);
            self.amd.mov_reg_mem(PARAMS, Amd::RSP, 0x10);
            self.amd.mov_reg_mem(MEM, Amd::RSP, 0x08);
        } else {
            self.amd.mov_reg_mem(STATES, Amd::RSP, 0x18);
            self.amd.mov_reg_mem(IDX, Amd::RSP, 0x10);
            self.amd.mov_reg_mem(PARAMS, Amd::RSP, 0x08);
            self.amd.mov_reg_mem(MEM, Amd::RSP, 0x00);
            self.amd.add_rsp(32);
        }
    }

    #[cfg(target_family = "unix")]
    fn chkstk(&mut self, size: u32) {
        self.amd.sub_rsp(size);
    }

    #[cfg(target_family = "windows")]
    fn chkstk(&mut self, mut size: u32) {
        const PAGE_SIZE: u32 = 4096;

        while size > PAGE_SIZE {
            self.amd.sub_rsp(PAGE_SIZE);
            self.amd.mov_reg_mem(Amd::RAX, Amd::RSP, 0);
            size -= PAGE_SIZE;
        }

        self.amd.sub_rsp(size);
    }
}

impl Generator for AmdGenerator {
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
        !matches!(self.family, AmdFamily::SSEScalar)
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

    fn branch_if(&mut self, cond: Reg, label: &str) {
        self.amd.vucomisd(ϕ(cond), ϕ(cond));
        self.amd.jpe(label);
    }

    //***********************************

    fn fmov(&mut self, dst: Reg, s1: Reg) {
        if dst != s1 {
            select!(self, movapd, vmovapd, vmovapd, ϕ(dst), ϕ(s1));
        }
    }

    fn fxchg(&mut self, s1: Reg, s2: Reg) {
        match self.family {
            AmdFamily::AvxScalar | AmdFamily::AvxVector => {
                self.amd.vxorpd(ϕ(s1), ϕ(s1), ϕ(s2));
                self.amd.vxorpd(ϕ(s2), ϕ(s1), ϕ(s2));
                self.amd.vxorpd(ϕ(s1), ϕ(s1), ϕ(s2));
            }
            AmdFamily::SSEScalar => {
                self.amd.xorpd(ϕ(s1), ϕ(s2));
                self.amd.xorpd(ϕ(s2), ϕ(s1));
                self.amd.xorpd(ϕ(s1), ϕ(s2));
            }
        }
    }

    fn load_const(&mut self, dst: Reg, idx: u32) {
        let label = format!("_const_{}_", idx);

        select!(
            self,
            movsd_xmm_label,
            vmovsd_xmm_label,
            vbroadcastsd_label,
            ϕ(dst),
            label.as_str()
        );
    }

    fn load_mem(&mut self, dst: Reg, idx: u32) {
        select!(
            self,
            movsd_xmm_mem,
            vmovsd_xmm_mem,
            vmovpd_ymm_mem,
            ϕ(dst),
            MEM,
            (idx * self.reg_size()) as i32
        );
    }

    fn save_mem(&mut self, dst: Reg, idx: u32) {
        select!(
            self,
            movsd_mem_xmm,
            vmovsd_mem_xmm,
            vmovpd_mem_ymm,
            MEM,
            (idx * self.reg_size()) as i32,
            ϕ(dst)
        );
    }

    fn save_mem_result(&mut self, idx: u32) {
        self.save_mem(Reg::Ret, idx);
    }

    fn load_param(&mut self, dst: Reg, idx: u32) {
        select!(
            self,
            movsd_xmm_mem,
            vmovsd_xmm_mem,
            vbroadcastsd,
            ϕ(dst),
            PARAMS,
            8 * idx as i32
        );
    }

    fn load_stack(&mut self, dst: Reg, idx: u32) {
        select!(
            self,
            movsd_xmm_mem,
            vmovsd_xmm_mem,
            vmovpd_ymm_mem,
            ϕ(dst),
            Amd::RSP,
            (idx * self.reg_size()) as i32
        );
    }

    fn save_stack(&mut self, dst: Reg, idx: u32) {
        select!(
            self,
            movsd_mem_xmm,
            vmovsd_mem_xmm,
            vmovpd_mem_ymm,
            Amd::RSP,
            (idx * self.reg_size()) as i32,
            ϕ(dst)
        );
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
        uniop!(self, sqrtsd, vsqrtsd, vsqrtpd, dst, s1);
    }

    fn recip(&mut self, dst: Reg, s1: Reg) {
        self.load_const_by_name(Reg::Temp, "_one_");
        self.divide(dst, Reg::Temp, s1);
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
        binop!(self, addsd, vaddsd, vaddpd, dst, s1, s2, true);
    }

    fn minus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, subsd, vsubsd, vsubpd, dst, s1, s2, false);
    }

    fn times(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, mulsd, vmulsd, vmulpd, dst, s1, s2, true);
    }

    fn divide(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, divsd, vdivsd, vdivpd, dst, s1, s2, false);
    }

    fn gt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, cmpnlesd, vcmpnlesd, vcmpnlepd, dst, s1, s2, false);
    }

    fn geq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, cmpnltsd, vcmpnltsd, vcmpnltpd, dst, s1, s2, false);
    }

    fn lt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, cmpltsd, vcmpltsd, vcmpltpd, dst, s1, s2, false);
    }

    fn leq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, cmplesd, vcmplesd, vcmplepd, dst, s1, s2, false);
    }

    fn eq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, cmpeqsd, vcmpeqsd, vcmpeqpd, dst, s1, s2, true);
    }

    fn neq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, cmpneqsd, vcmpneqsd, vcmpneqpd, dst, s1, s2, true);
    }

    fn and(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, andpd, vandpd, vandpd, dst, s1, s2, true);
    }

    fn andnot(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, andnpd, vandnpd, vandnpd, dst, s1, s2, false);
    }

    fn or(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, orpd, vorpd, vorpd, dst, s1, s2, true);
    }

    fn xor(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        binop!(self, xorpd, vxorpd, vxorpd, dst, s1, s2, true);
    }

    fn not(&mut self, dst: Reg, s1: Reg) {
        self.load_const_by_name(Reg::Temp, "_all_ones_");
        self.xor(dst, s1, Reg::Temp);
    }

    fn fused_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        match self.family {
            AmdFamily::AvxScalar => {
                fuseop!(self, vfmadd132sd, vfmadd213sd, vfmadd231sd, dst, s1, s2, s3)
            }
            AmdFamily::AvxVector => {
                fuseop!(self, vfmadd132pd, vfmadd213pd, vfmadd231pd, dst, s1, s2, s3)
            }
            _ => {
                self.times(s1, s1, s2);
                self.plus(dst, s1, s3);
            }
        }
    }

    fn fused_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        match self.family {
            AmdFamily::AvxScalar => {
                fuseop!(self, vfmsub132sd, vfmsub213sd, vfmsub231sd, dst, s1, s2, s3)
            }
            AmdFamily::AvxVector => {
                fuseop!(self, vfmsub132pd, vfmsub213pd, vfmsub231pd, dst, s1, s2, s3)
            }
            _ => {
                self.times(s1, s1, s2);
                self.minus(dst, s1, s3);
            }
        }
    }

    fn fused_neg_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        match self.family {
            AmdFamily::AvxScalar => {
                fuseop!(
                    self,
                    vfnmadd132sd,
                    vfnmadd213sd,
                    vfnmadd231sd,
                    dst,
                    s1,
                    s2,
                    s3
                )
            }
            AmdFamily::AvxVector => {
                fuseop!(
                    self,
                    vfnmadd132pd,
                    vfnmadd213pd,
                    vfnmadd231pd,
                    dst,
                    s1,
                    s2,
                    s3
                )
            }
            _ => {
                self.times(s1, s1, s2);
                self.minus(dst, s3, s1);
            }
        }
    }

    fn fused_neg_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        match self.family {
            AmdFamily::AvxScalar => {
                fuseop!(
                    self,
                    vfnmsub132sd,
                    vfnmsub213sd,
                    vfnmsub231sd,
                    dst,
                    s1,
                    s2,
                    s3
                )
            }
            AmdFamily::AvxVector => {
                fuseop!(
                    self,
                    vfnmsub132pd,
                    vfnmsub213pd,
                    vfnmsub231pd,
                    dst,
                    s1,
                    s2,
                    s3
                )
            }
            _ => {
                self.times(s1, s1, s2);
                self.plus(dst, s1, s3);
                self.neg(dst, dst);
            }
        }
    }

    fn add_consts(&mut self, consts: &[f64]) {
        for (idx, val) in consts.iter().enumerate() {
            let label = format!("_const_{}_", idx);
            self.set_label(label.as_str());
            self.append_quad((*val).to_bits());
        }
    }

    fn add_func(&mut self, f: &str, p: crate::code::Func) {
        let label = format!("_func_{}_", f);
        self.set_label(label.as_str());
        self.append_quad(p.func_ptr());
    }

    fn call(&mut self, op: &str, num_args: usize) {
        let label = format!("_func_{}_", op);

        match self.family {
            AmdFamily::AvxScalar | AmdFamily::SSEScalar => {
                self.vzeroupper();
                #[cfg(target_family = "windows")]
                self.amd.sub_rsp(32);

                //self.amd.call(Amd::R12);
                self.amd.call_indirect(&label);

                #[cfg(target_family = "windows")]
                self.amd.add_rsp(32);
            }
            AmdFamily::AvxVector => match num_args {
                1 => self.call_vector_unary(&label),
                2 => self.call_vector_binary(&label),
                _ => {
                    panic!("invalid number of arguments")
                }
            },
        }
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
    fn prologue_fast(&mut self, cap: u32, num_args: u32) {
        self.amd.push(MEM);
        self.amd.push(PARAMS);
        self.amd.sub_rsp(self.frame_size(cap));
        self.amd.mov(MEM, Amd::RSP);

        for i in 0..num_args {
            self.amd.movsd_mem_xmm(MEM, (i * 8) as i32, i as u8);
        }
    }

    #[cfg(target_family = "unix")]
    fn epilogue_fast(&mut self, cap: u32, idx_ret: i32) {
        self.vzeroupper();
        self.amd.movsd_xmm_mem(0, Amd::RSP, 8 * idx_ret);

        self.amd.add_rsp(self.frame_size(cap));
        self.amd.pop(PARAMS);
        self.amd.pop(MEM);
        self.amd.ret();
    }

    /*
     * prologue_indirect generates the stack frame. It works in two modes:
     *  Direct mode: MEM (state variables + obs) is passed directly as the first argument. The second argument is null.
     *  Indirect mode: the second argument is a pointer to an array of pointers to states and obs. The third argument
     *      is the index into these arrays. MEM is allocated on the stack and filled based on the second and thirds args.
     *
     * Noth that the second argument determines whether it is the direct (args[1] == null) or indirect mode.
     * In both modes, the fourth argument points to an array of params.
     */
    fn prologue_indirect(&mut self, cap: u32, count_states: usize, count_obs: usize) {
        let win = cfg!(target_family = "windows");
        self.save_nonvolatile_regs();

        self.amd.mov(MEM, if win { Amd::RCX } else { Amd::RDI }); // first arg = mem if direct mode, otherwise null
        self.amd.mov(STATES, if win { Amd::RDX } else { Amd::RSI }); // second arg = states+obs if indirect mode, otherwise null
        self.amd.mov(IDX, if win { Amd::R8 } else { Amd::RDX }); // third arg = index if indirect mode
        self.amd.mov(PARAMS, if win { Amd::R9 } else { Amd::RCX }); // fourth arg = params

        self.amd.or(STATES, STATES);
        self.amd.jz("@main");

        let size = (count_states + count_obs + 1) as u32 * self.reg_size();
        //self.amd.sub_rsp(size);
        self.chkstk(size);
        self.amd.mov(MEM, Amd::RSP); // in indirect mode, MEM is allocated on the stack

        for i in 0..count_states {
            self.amd.mov_reg_mem(Amd::RAX, STATES, 8 * i as i32);
            let k = i as u32 * self.reg_size();
            select!(
                self,
                movsd_xmm_indexed,
                vmovsd_xmm_indexed,
                vmovpd_ymm_indexed,
                RET,
                Amd::RAX,
                IDX,
                8
            );
            select!(
                self,
                movsd_mem_xmm,
                vmovsd_mem_xmm,
                vmovpd_mem_ymm,
                MEM,
                k as i32,
                RET
            );
        }

        // may save idx (RDX) as double in RBP + 8/32 * count_states

        self.set_label("@main");
        // self.amd.sub_rsp(self.frame_size(cap));
        self.chkstk(self.frame_size(cap));
    }

    fn epilogue_indirect(&mut self, cap: u32, count_states: usize, count_obs: usize) {
        self.amd.add_rsp(self.frame_size(cap));

        self.amd.or(STATES, STATES);
        self.amd.jz("@done");

        let size = (count_states + count_obs + 1) as u32 * self.reg_size();

        for i in 0..count_obs {
            self.amd
                .mov_reg_mem(Amd::RAX, STATES, 8 * (count_states + i) as i32);
            let k = (count_states + i + 1) as u32 * self.reg_size();
            select!(
                self,
                movsd_xmm_mem,
                vmovsd_xmm_mem,
                vmovpd_ymm_mem,
                RET,
                MEM,
                k as i32
            );
            select!(
                self,
                movsd_indexed_xmm,
                vmovsd_indexed_xmm,
                vmovpd_indexed_ymm,
                Amd::RAX,
                IDX,
                8,
                RET
            );
        }

        self.amd.add_rsp(size);
        self.set_label("@done");

        self.vzeroupper();

        self.load_nonvolatile_regs();
        self.amd.ret();
    }

    #[cfg(target_family = "windows")]
    fn prologue_fast(&mut self, cap: u32, num_args: u32) {
        self.amd.mov_mem_reg(Amd::RSP, 0x08, MEM);
        self.amd.mov_mem_reg(Amd::RSP, 0x10, PARAMS);

        let frame_size = self.frame_size(cap);
        self.amd.sub_rsp(frame_size);
        self.amd.mov(MEM, Amd::RSP);

        for i in 0..num_args.min(4) {
            self.amd.movsd_mem_xmm(MEM, (i * 8) as i32, i as u8);
        }

        for i in 4..num_args {
            // the offset of the fifth or eight arguments:
            // +4 for the 32-byte home
            // +1 for the return address in the stack
            // -4 for the first four arguments passed in XMM0-XMM3
            self.amd.movsd_xmm_mem(
                0,
                MEM,
                (frame_size + self.reg_size() * (4 + 1 + i - 4)) as i32,
            );
            self.amd.movsd_mem_xmm(MEM, (i * 8) as i32, 0);
        }
    }

    #[cfg(target_family = "windows")]
    fn epilogue_fast(&mut self, cap: u32, idx_ret: i32) {
        self.vzeroupper();
        self.amd.movsd_xmm_mem(0, MEM, 8 * idx_ret);

        self.amd.add_rsp(self.frame_size(cap));
        self.amd.mov_reg_mem(PARAMS, Amd::RSP, 0x10);
        self.amd.mov_reg_mem(MEM, Amd::RSP, 0x08);
        self.amd.ret();
        self.predefined_consts();
    }

    fn save_used_registers(&mut self, used: &[u8]) {
        let count_shadows = self.count_shadows();

        for r in used {
            if *r >= count_shadows {
                self.save_stack(reg(*r), *r as u32);
            }
        }
    }

    fn load_used_registers(&mut self, used: &[u8]) {
        let count_shadows = self.count_shadows();

        for r in used {
            if *r >= count_shadows {
                self.load_stack(reg(*r), *r as u32);
            }
        }
    }
}

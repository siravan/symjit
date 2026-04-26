use crate::code::Func;
use crate::utils::Reg;

mod asm;
mod fused;

use asm::Amd;

mod scalar;
mod sse;
mod vector;
mod complex;

pub use scalar::AmdScalarGenerator;
pub use sse::AmdSSEGenerator;
pub use vector::AmdVectorGenerator;
pub use complex::AmdComplexGenerator;

#[cfg(target_family = "windows")]
const ARGS: [u8; 4] = [Amd::RCX, Amd::RDX, Amd::R8, Amd::R9];

#[cfg(target_family = "unix")]
const ARGS: [u8; 4] = [Amd::RDI, Amd::RSI, Amd::RDX, Amd::RCX];

const RET: u8 = 0;

const MEM: u8 = Amd::RBP;
const STATES: u8 = Amd::R13;
const IDX: u8 = Amd::R12;
const PARAMS: u8 = Amd::RBX;
const STACK: u8 = Amd::RSP;

fn save_nonvolatile_regs(amd: &mut Amd) {
    if cfg!(target_family = "windows") {
        amd.mov_mem_reg(STACK, 0x10, PARAMS);
        amd.mov_mem_reg(STACK, 0x18, IDX);
        amd.mov_mem_reg(STACK, 0x20, STATES);
    } else {
        amd.sub_rsp(32);
        amd.mov_mem_reg(STACK, 0x08, PARAMS);
        amd.mov_mem_reg(STACK, 0x10, IDX);
        amd.mov_mem_reg(STACK, 0x18, STATES);
    }
}

fn load_nonvolatile_regs(amd: &mut Amd) {
    if cfg!(target_family = "windows") {
        amd.mov_reg_mem(PARAMS, STACK, 0x10);
        amd.mov_reg_mem(IDX, STACK, 0x18);
        amd.mov_reg_mem(STATES, STACK, 0x20);
    } else {
        amd.mov_reg_mem(PARAMS, STACK, 0x08);
        amd.mov_reg_mem(IDX, STACK, 0x10);
        amd.mov_reg_mem(STATES, STACK, 0x18);
        amd.add_rsp(32);
    }
}

#[cfg(target_family = "unix")]
fn sub_rsp(amd: &mut Amd, size: u32) {
    if size != 0 {
        amd.sub_rsp(size);
    }
}

#[cfg(target_family = "windows")]
fn sub_rsp(amd: &mut Amd, mut size: u32) {
    // chkstk function
    const PAGE_SIZE: u32 = 4096;

    while size > PAGE_SIZE {
        amd.sub_rsp(PAGE_SIZE);
        amd.mov_reg_mem(Amd::RAX, STACK, 0);
        size -= PAGE_SIZE;
    }

    amd.sub_rsp(size);
}

fn add_rsp(amd: &mut Amd, size: u32) {
    if size != 0 {
        amd.add_rsp(size);
    }
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
        Reg::Gen(dst) => dst + 2,
        Reg::Static(..) => panic!("passing static registers to codegen"),
    }
}

fn predefined_consts(amd: &mut Amd) {
    amd.a.set_label("_minus_zero_");
    amd.a.append_quad((-0.0f64).to_bits());

    amd.a.set_label("_one_");
    amd.a.append_quad(1.0f64.to_bits());

    amd.a.set_label("_two_");
    amd.a.append_quad(2.0f64.to_bits());

    amd.a.set_label("_all_ones_");
    amd.a.append_quad(0xffffffffffffffff);
}

/*
 * fuse_load_math tries to fuse the last two instructions if
 * the last one is a math-op and the one before is a load
 * instruction. For example,
 *
 * vmovsd xmm0, [rbp + 0x1234]
 * vaddsd xmm2, xmm3, xmm0
 *
 * fuses into
 *
 * vaddsd xmm2, xmm3, [rbp + 0x1234]
 *
 */
fn fuse_load_math(amd: &mut Amd, last_load: usize) {
    let ip0 = last_load; // the address of the last load instruction
    let ip1 = amd.a.ip() - 4; // the address of the last math op

    if ip1 - ip0 > 10 {
        return;
    }

    let b: &mut [u8] = &mut amd.a.buf;

    // Conditions:
    //
    // the first bytes are 0xc5, i.e., VEX prefix
    // 0x10 means a load instruction (vmovsd or vmovpd)
    // `b[ip0 + 3] & 0x38 == 0` means the destination of the load istruction
    // is xmm0.
    // `b[ip1 + 3] & 0x07 == 0` means the second source of the math op
    // is xmm0.
    //
    // Note that `Node.load_math` specifically uses Reg::Ret (i.e., xmm0)
    // to signal this function it is safe to fuse the operations.
    if b[ip1] == 0xc5 && b[ip0] == 0xc5 && b[ip0 + 2] == 0x10 {
        if b[ip0 + 3] & 0x38 == 0 && b[ip1 + 3] & 0x07 == 0 {
            // if (b[ip0 + 3] & 0x38) >> 3 == b[ip1 + 3] & 0x07 {
            b[ip0 + 1] = b[ip1 + 1]; // copy VEX prefix
            b[ip0 + 2] = b[ip1 + 2]; // copy OpCode

            // Fusing ModR/M byte. Destination comes from the math op and
            // source comes the load instruction.
            b[ip0 + 3] |= b[ip1 + 3] & 0x38;

            for _ in 0..4 {
                amd.a.buf.pop().unwrap();
            }
        }
    }
}

<<<<<<< HEAD
fn add_func(amd: &mut Amd, op: &str, f: Func) {
    if let Func::Slice {
        f_scalar,
        f_simd,
        env,
        ..
    } = f
    {
=======
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

    fn branch(&mut self, label: &str) {
        self.amd.xor(Amd::RAX, Amd::RAX);
        self.amd.jz(label);
    }

    fn branch_if(&mut self, cond: Reg, label: &str, is_else: bool) {
        match self.family {
            AmdFamily::AvxScalar | AmdFamily::SSEScalar => {
                self.amd.vucomisd(ϕ(cond), ϕ(cond));

                if is_else {
                    self.amd.jpo(label);
                } else {
                    self.amd.jpe(label);
                }
            }
            AmdFamily::AvxVector => {
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
        }
    }

    /*
     * fuse_load_math tries to fuse the last two instructions if
     * the last one is a math-op and the one before is a load
     * instruction. For example,
     *
     * vmovsd xmm0, [rbp + 0x1234]
     * vaddsd xmm2, xmm3, xmm0
     *
     * fuses into
     *
     * vaddsd xmm2, xmm3, [rbp + 0x1234]
     *
     */
    fn fuse_load_math(&mut self) {
        let ip0 = self.last_load; // the address of the last load instruction
        let ip1 = self.amd.a.ip() - 4; // the address of the last math op

        if ip1 - ip0 > 10 {
            return;
        }

        let b: &mut [u8] = &mut self.amd.a.buf;

        // Conditions:
        //
        // the first bytes are 0xc5, i.e., VEX prefix
        // 0x10 means a load instruction (vmovsd or vmovpd)
        // `b[ip0 + 3] & 0x38 == 0` means the destination of the load istruction
        // is xmm0.
        // `b[ip1 + 3] & 0x07 == 0` means the second source of the math op
        // is xmm0.
        //
        // Note that `Node.load_math` specifically uses Reg::Ret (i.e., xmm0)
        // to signal this function it is safe to fuse the operations.
        if b[ip1] == 0xc5 && b[ip0] == 0xc5 && b[ip0 + 2] == 0x10 {
            if b[ip0 + 3] & 0x38 == 0 && b[ip1 + 3] & 0x07 == 0 {
                // if (b[ip0 + 3] & 0x38) >> 3 == b[ip1 + 3] & 0x07 {
                b[ip0 + 1] = b[ip1 + 1]; // copy VEX prefix
                b[ip0 + 2] = b[ip1 + 2]; // copy OpCode

                // Fusing ModR/M byte. Destination comes from the math op and
                // source comes the load instruction.
                b[ip0 + 3] |= b[ip1 + 3] & 0x38;

                for _ in 0..4 {
                    self.amd.a.buf.pop().unwrap();
                }
            }
        }
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
        self.last_load = self.amd.a.ip();

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
        self.last_load = self.amd.a.ip();

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
        self.last_load = self.amd.a.ip();

        if self.config.symbolica() {
            select!(
                self,
                movsd_xmm_mem,
                vmovsd_xmm_mem,
                vmovpd_ymm_mem,
                ϕ(dst),
                PARAMS,
                (idx * self.reg_size()) as i32
            );
        } else {
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
    }

    fn load_stack(&mut self, dst: Reg, idx: u32) {
        self.last_load = self.amd.a.ip();

        select!(
            self,
            movsd_xmm_mem,
            vmovsd_xmm_mem,
            vmovpd_ymm_mem,
            ϕ(dst),
            STACK,
            (idx * self.reg_size()) as i32
        );
    }

    fn save_stack(&mut self, dst: Reg, idx: u32) {
        select!(
            self,
            movsd_mem_xmm,
            vmovsd_mem_xmm,
            vmovpd_mem_ymm,
            STACK,
            (idx * self.reg_size()) as i32,
            ϕ(dst)
        );
    }

    fn load_mem_complex(&mut self, xd: Reg, yd: Reg, idx: u32) {
        if self.config.permissive() && matches!(self.family, AmdFamily::AvxScalar) {
            self.amd
                .vmovdd_xmm_mem(ϕ(xd), MEM, (idx * self.reg_size()) as i32);
            self.amd.vshufdd(ϕ(yd), ϕ(xd), ϕ(xd), 1);
        } else {
            self.load_mem(xd, idx);
            self.load_mem(yd, idx + 1);
        }
    }

    fn save_mem_complex(&mut self, xs: Reg, ys: Reg, idx: u32) {
        if self.config.permissive() && matches!(self.family, AmdFamily::AvxScalar) {
            self.amd.vunpckldd(ϕ(xs), ϕ(xs), ϕ(ys));
            self.amd
                .vmovdd_mem_xmm(MEM, (idx * self.reg_size()) as i32, ϕ(xs));
        } else {
            self.save_mem(xs, idx);
            self.save_mem(ys, idx + 1);
        }
    }

    fn load_param_complex(&mut self, xd: Reg, yd: Reg, idx: u32) {
        if self.config.permissive() && matches!(self.family, AmdFamily::AvxScalar) {
            self.amd
                .vmovdd_xmm_mem(ϕ(xd), PARAMS, (idx * self.reg_size()) as i32);
            self.amd.vshufdd(ϕ(yd), ϕ(xd), ϕ(xd), 1);
        } else {
            self.load_param(xd, idx);
            self.load_param(yd, idx + 1);
        }
    }

    fn load_stack_complex(&mut self, xd: Reg, yd: Reg, idx: u32) {
        if self.config.permissive() && matches!(self.family, AmdFamily::AvxScalar) {
            self.amd
                .vmovdd_xmm_mem(ϕ(xd), STACK, (idx * self.reg_size()) as i32);
            self.amd.vshufdd(ϕ(yd), ϕ(xd), ϕ(xd), 1);
        } else {
            self.load_stack(xd, idx);
            self.load_stack(yd, idx + 1);
        }
    }

    fn save_stack_complex(&mut self, xs: Reg, ys: Reg, idx: u32) {
        if self.config.permissive() && matches!(self.family, AmdFamily::AvxScalar) {
            self.amd.vunpckldd(ϕ(xs), ϕ(xs), ϕ(ys));
            self.amd
                .vmovdd_mem_xmm(STACK, (idx * self.reg_size()) as i32, ϕ(xs));
        } else {
            self.save_stack(xs, idx);
            self.save_stack(ys, idx + 1);
        }
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

    fn times_complex(&mut self, xd: Reg, yd: Reg, x1: Reg, y1: Reg, x2: Reg, y2: Reg) -> bool {
        if !matches!(self.family, AmdFamily::SSEScalar) && self.config.permissive() {
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
        } else {
            false
        }
    }

    fn divide_complex(&mut self, xd: Reg, yd: Reg, x1: Reg, y1: Reg, x2: Reg, y2: Reg) -> bool {
        if !matches!(self.family, AmdFamily::SSEScalar) && self.config.permissive() {
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
        } else {
            false
        }
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

    fn add_func(&mut self, op: &str, f: Func) {
        if let Func::Slice {
            f_scalar,
            f_simd,
            env,
            ..
        } = f
        {
            let label = format!("_func_{}_", op);
            self.set_label(label.as_str());
            // let f_scalar = trampoline_homogenous::<f64> as *const c_void;
            self.append_quad(f_scalar as u64);

            let label = format!("_simd_{}_", op);
            self.set_label(label.as_str());
            // let f_simd = trampoline_heterogenous::<f64x4, f64> as *const c_void;
            self.append_quad(f_simd as u64);

            let label = format!("_env_{}_", op);
            self.set_label(label.as_str());
            self.append_quad(env as u64);
        } else {
            let label = format!("_func_{}_", op);
            self.set_label(label.as_str());
            self.append_quad(f.func_ptr());
        }
    }

    fn call(&mut self, op: &str, num_args: usize) -> Result<()> {
        if is_external_func(op) {
            return self.call_external(op, num_args);
        }

>>>>>>> v216
        let label = format!("_func_{}_", op);
        amd.a.set_label(label.as_str());
        // let f_scalar = trampoline_homogenous::<f64> as *const c_void;
        amd.a.append_quad(f_scalar as u64);

        let label = format!("_simd_{}_", op);
        amd.a.set_label(label.as_str());
        // let f_simd = trampoline_heterogenous::<f64x4, f64> as *const c_void;
        amd.a.append_quad(f_simd as u64);

        let label = format!("_env_{}_", op);
        amd.a.set_label(label.as_str());
        amd.a.append_quad(env as u64);
    } else {
        let label = format!("_func_{}_", op);
        amd.a.set_label(label.as_str());
        amd.a.append_quad(f.func_ptr());
    }
}

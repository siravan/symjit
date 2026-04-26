use crate::code::Func;
use crate::config::{Config, SPILL_AREA};
use crate::generator::Generator;
use crate::utils::align_stack;
use crate::utils::{is_external_func, reg, DataType, Reg};
use anyhow::{anyhow, Result};

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
    config: Config,
    last_load: usize,
}

#[cfg(target_family = "windows")]
const ARGS: [u8; 4] = [Amd::RCX, Amd::RDX, Amd::R8, Amd::R9];

#[cfg(target_family = "unix")]
const ARGS: [u8; 4] = [Amd::RDI, Amd::RSI, Amd::RDX, Amd::RCX];

const MEM: u8 = Amd::RBP;
const STATES: u8 = Amd::R13;
const IDX: u8 = Amd::R12;
const PARAMS: u8 = Amd::RBX;
const STACK: u8 = Amd::RSP;

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
    pub fn new(family: AmdFamily, config: Config) -> AmdGenerator {
        AmdGenerator {
            amd: Amd::new(DataType::F64),
            family,
            config,
            last_load: 0,
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
        self.amd.vmovpd_mem_ymm(STACK, 32, 0);

        self.vzeroupper();

        for i in 0..4 {
            if i > 0 {
                self.amd.movsd_xmm_mem(0, STACK, 32 + i * 8);
            }
            self.amd.call_indirect(label);
            self.amd.movsd_mem_xmm(STACK, 32 + i * 8, 0);
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
                self.amd.movsd_xmm_mem(0, STACK, 32 + i * 8);
                self.amd.movsd_xmm_mem(1, STACK, 64 + i * 8);
            }
            self.amd.call_indirect(label);
            self.amd.movsd_mem_xmm(STACK, 32 + i * 8, 0);
        }

        self.amd.vmovpd_ymm_mem(0, STACK, 32);
    }

    fn call_complex_vector_unary(&mut self, label: &str) {
        self.amd.vmovpd_mem_ymm(STACK, 64, 0);
        self.amd.vmovpd_mem_ymm(STACK, 96, 1);

        self.vzeroupper();

        for i in 0..4 {
            if i > 0 {
                self.amd.movsd_xmm_mem(0, STACK, 64 + i * 8);
                self.amd.movsd_xmm_mem(1, STACK, 96 + i * 8);
            }

            if cfg!(target_family = "windows") {
                self.amd.lea_mem(Amd::R8, STACK, 32);
            } else {
                self.amd.lea_mem(Amd::RDI, STACK, 32);
            }

            self.amd.call_indirect(label);

            self.amd.movsd_xmm_mem(0, STACK, 32);
            self.amd.movsd_xmm_mem(1, STACK, 40);
            self.amd.movsd_mem_xmm(STACK, 64 + i * 8, 0);
            self.amd.movsd_mem_xmm(STACK, 96 + i * 8, 1);
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
                self.amd.movsd_xmm_mem(0, STACK, 64 + i * 8);
                self.amd.movsd_xmm_mem(1, STACK, 96 + i * 8);
                self.amd.movsd_xmm_mem(2, STACK, 128 + i * 8);
                self.amd.movsd_xmm_mem(3, STACK, 160 + i * 8);
            }

            self.amd.movsd_mem_xmm(STACK, 32, 2);
            self.amd.movsd_mem_xmm(STACK, 40, 3);

            if cfg!(target_family = "windows") {
                self.amd.lea_mem(Amd::R8, STACK, 32);
            } else {
                self.amd.lea_mem(Amd::RDI, STACK, 32);
            }

            self.amd.call_indirect(label);

            self.amd.movsd_xmm_mem(0, STACK, 32);
            self.amd.movsd_xmm_mem(1, STACK, 40);
            self.amd.movsd_mem_xmm(STACK, 64 + i * 8, 0);
            self.amd.movsd_mem_xmm(STACK, 96 + i * 8, 1);
        }

        self.amd.vmovpd_ymm_mem(0, STACK, 64);
        self.amd.vmovpd_ymm_mem(1, STACK, 96);
    }

    fn call_external(&mut self, op: &str, num_args: usize) -> Result<()> {
        let cap = SPILL_AREA as u32;

        self.amd.mov_reg_label(ARGS[0], &format!("_env_{}_", op));
        self.amd
            .lea_mem(ARGS[1], STACK, (cap * self.reg_size()) as i32);
        self.amd.mov_imm(ARGS[2], num_args as u32);
        self.amd.lea_mem(ARGS[3], STACK, 4 * self.reg_size() as i32);
        self.vzeroupper();

        match self.family {
            AmdFamily::AvxScalar | AmdFamily::SSEScalar => {
                self.amd.call_indirect(&format!("_func_{}_", op));
                self.load_stack(Reg::Ret, 4);
                if self.config.is_complex() {
                    self.load_stack(Reg::Temp, 5);
                }
            }
            AmdFamily::AvxVector => {
                self.amd.call_indirect(&format!("_simd_{}_", op));

                if self.config.is_complex() {
                    let l1 = format!(".P{}", self.amd.a.ip());
                    let l2 = format!(".Q{}", self.amd.a.ip());

                    self.amd.or(Amd::RAX, Amd::RAX);
                    self.amd.jz(&l1);

                    self.amd
                        .vmovpd_ymm_mem(2, STACK, 4 * self.reg_size() as i32);
                    self.amd
                        .vmovpd_ymm_mem(3, STACK, 5 * self.reg_size() as i32);
                    self.amd.vshufpd(0, 2, 3, 0);
                    self.amd.vshufpd(1, 2, 3, 0x0f);

                    self.amd.jmp(&l2);
                    self.set_label(&l1);

                    self.amd
                        .vmovpd_ymm_mem(0, STACK, 4 * self.reg_size() as i32);

                    self.amd
                        .vmovpd_ymm_mem(1, STACK, 5 * self.reg_size() as i32);

                    self.set_label(&l2);
                } else {
                    self.amd
                        .vmovpd_ymm_mem(0, STACK, 4 * self.reg_size() as i32);
                }
            }
        }

        Ok(())
    }

    fn predefined_consts(&mut self) {
        self.align();

        self.set_label("_minus_zero_");
        self.append_quad((-0.0f64).to_bits());

        self.set_label("_one_");
        self.append_quad(1.0f64.to_bits());

        self.set_label("_two_");
        self.append_quad(2.0f64.to_bits());

        self.set_label("_all_ones_");
        self.append_quad(0xffffffffffffffff);
    }

    fn save_nonvolatile_regs(&mut self) {
        if cfg!(target_family = "windows") {
            self.amd.mov_mem_reg(STACK, 0x10, PARAMS);
            self.amd.mov_mem_reg(STACK, 0x18, IDX);
            self.amd.mov_mem_reg(STACK, 0x20, STATES);
        } else {
            self.amd.sub_rsp(32);
            self.amd.mov_mem_reg(STACK, 0x08, PARAMS);
            self.amd.mov_mem_reg(STACK, 0x10, IDX);
            self.amd.mov_mem_reg(STACK, 0x18, STATES);
        }
    }

    fn load_nonvolatile_regs(&mut self) {
        if cfg!(target_family = "windows") {
            self.amd.mov_reg_mem(PARAMS, STACK, 0x10);
            self.amd.mov_reg_mem(IDX, STACK, 0x18);
            self.amd.mov_reg_mem(STATES, STACK, 0x20);
        } else {
            self.amd.mov_reg_mem(PARAMS, STACK, 0x08);
            self.amd.mov_reg_mem(IDX, STACK, 0x10);
            self.amd.mov_reg_mem(STATES, STACK, 0x18);
            self.amd.add_rsp(32);
        }
    }

    #[cfg(target_family = "unix")]
    fn sub_rsp(&mut self, size: u32) {
        if size != 0 {
            self.amd.sub_rsp(size);
        }
    }

    #[cfg(target_family = "windows")]
    fn sub_rsp(&mut self, mut size: u32) {
        // chkstk function
        const PAGE_SIZE: u32 = 4096;

        while size > PAGE_SIZE {
            self.amd.sub_rsp(PAGE_SIZE);
            self.amd.mov_reg_mem(Amd::RAX, STACK, 0);
            size -= PAGE_SIZE;
        }

        self.amd.sub_rsp(size);
    }

    fn add_rsp(&mut self, size: u32) {
        if size != 0 {
            self.amd.add_rsp(size);
        }
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

        let label = format!("_func_{}_", op);

        match self.family {
            AmdFamily::AvxScalar | AmdFamily::SSEScalar => {
                self.vzeroupper();
                self.amd.call_indirect(&label);
            }
            AmdFamily::AvxVector => match num_args {
                1 => self.call_vector_unary(&label),
                2 => self.call_vector_binary(&label),
                _ => return Err(anyhow!("invalid number of arguments")),
            },
        }

        Ok(())
    }

    fn call_complex(&mut self, op: &str, num_args: usize) -> Result<()> {
        let label = format!("_func_{}_", op);

        match self.family {
            AmdFamily::AvxScalar | AmdFamily::SSEScalar => {
                if num_args == 2 {
                    self.save_stack(Reg::Gen(0), 4);
                    self.save_stack(Reg::Gen(1), 5);
                }

                self.vzeroupper();

                if cfg!(target_family = "windows") {
                    self.amd.lea_mem(Amd::R8, STACK, 32);
                } else {
                    self.amd.lea_mem(Amd::RDI, STACK, 32);
                }

                self.amd.call_indirect(&label);

                self.load_stack(Reg::Ret, 4);
                self.load_stack(Reg::Temp, 5);
            }
            AmdFamily::AvxVector => match num_args {
                1 => self.call_complex_vector_unary(&label),
                2 => self.call_complex_vector_binary(&label),
                _ => return Err(anyhow!("invalid number of arguments")),
            },
        }

        Ok(())
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

        let frame_size = align_stack((count_states + count_obs) as u32 * self.reg_size());
        self.sub_rsp(frame_size);
        self.amd.mov(MEM, STACK);
        self.sub_rsp(align_stack(cap as u32 * self.reg_size()));

        for i in 0..count_states {
            self.amd.movsd_mem_xmm(MEM, (i * 8) as i32, i as u8);
        }
    }

    #[cfg(target_family = "windows")]
    fn prologue_fast(&mut self, cap: usize, count_states: usize, count_obs: usize) {
        self.amd.push(Amd::RBP);

        let frame_size = align_stack((count_states + count_obs) as u32 * self.reg_size());
        self.sub_rsp(frame_size);
        self.amd.mov(MEM, STACK);
        self.sub_rsp(align_stack(cap as u32 * self.reg_size()));

        for i in 0..count_states.min(4) {
            self.amd
                .movsd_mem_xmm(MEM, (i as u32 * self.reg_size()) as i32, i as u8);
        }

        for i in 4..count_states {
            let i = i as u32;
            // the offset of the fifth or eight arguments:
            // +4 for the 32-byte home
            // +1 for the return address in the stack
            // +1 for RBP in the stack
            // -4 for the first four arguments passed in XMM0-XMM3
            self.amd
                .movsd_xmm_mem(0, MEM, (frame_size + (i + 2) * self.reg_size()) as i32);
            self.amd.movsd_mem_xmm(MEM, (i * self.reg_size()) as i32, 0);
        }
    }

    fn epilogue_fast(&mut self, cap: usize, count_states: usize, count_obs: usize, idx_ret: i32) {
        self.vzeroupper();
        self.amd
            .movsd_xmm_mem(0, MEM, idx_ret * self.reg_size() as i32);

        let total_size = align_stack(cap as u32 * self.reg_size())
            + align_stack((count_states + count_obs) as u32 * self.reg_size());
        self.amd.add_rsp(total_size);

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
     *  4. The temporary variables area of size `align_stack(cap * self.reg_size())`.
     *      It has two sub-segments:
     *  4a. The actual temporary variables area.
     *  4b. A default spill area of `16 * self.reg_size()` bytes. It is generated by
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
        self.save_nonvolatile_regs();

        self.amd.mov(MEM, ARGS[0]); // first arg = mem if direct mode, otherwise null
        self.amd.mov(STATES, ARGS[1]); // second arg = states+obs if indirect mode, otherwise null
        self.amd.mov(IDX, ARGS[2]); // third arg = index if indirect mode
        self.amd.mov(PARAMS, ARGS[3]); // fourth arg = params

        self.amd.or(STATES, STATES);
        self.amd.jz("@main");

        let frame_size = align_stack((count_states + count_obs) as u32 * self.reg_size());
        self.sub_rsp(frame_size);
        self.amd.mov(MEM, STACK); // in indirect mode, MEM is allocated on the stack

        // multiply IDX by 4 to convert from f64x4 index to f64 index
        if self.reg_size() == 32 {
            self.amd.add(IDX, IDX);
            self.amd.add(IDX, IDX);
        }

        for i in 0..count_states {
            self.amd.mov_reg_mem(Amd::RAX, STATES, 2 * 8 * i as i32);
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
        self.sub_rsp(align_stack(cap as u32 * self.reg_size()));
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

        self.add_rsp(align_stack(cap as u32 * self.reg_size()));

        self.amd.or(STATES, STATES);
        self.amd.jz("@done");

        for i in 0..count_obs {
            self.amd
                .mov_reg_mem(Amd::RCX, STATES, 2 * 8 * (count_states + i) as i32);
            let k = (count_states + i) as u32 * self.reg_size();
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
                Amd::RCX,
                IDX,
                8,
                RET
            );
        }

        let frame_size = align_stack((count_states + count_obs) as u32 * self.reg_size());
        self.amd.add_rsp(frame_size);
        self.set_label("@done");

        self.vzeroupper();

        self.load_nonvolatile_regs();
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

impl AmdGenerator {
    fn prologue_symbolica(&mut self, cap: usize, count_params: usize, count_obs: usize) {
        self.amd.push(Amd::RBP);
        self.save_nonvolatile_regs();

        self.amd.mov(MEM, ARGS[0]); // first arg = mem if direct mode, otherwise null
        self.amd.mov(STATES, ARGS[1]); // second arg = states+obs if indirect mode, otherwise null
        self.amd.mov(IDX, ARGS[2]); // third arg = index if indirect mode
        self.amd.mov(PARAMS, ARGS[3]); // fourth arg = params

        if self.reg_size() == 32 {
            self.amd.or(IDX, IDX);
            self.amd.jz("@main");

            self.sub_rsp(align_stack(count_params as u32 * 32));
            self.amd.mov(Amd::RAX, PARAMS);
            self.amd.mov(PARAMS, STACK);

            for j in 0..4 {
                for i in 0..count_params {
                    self.amd
                        .vmovsd_xmm_mem(RET, Amd::RAX, 8 * (i + j * count_params) as i32);
                    self.amd.vmovsd_mem_xmm(PARAMS, 8 * (i * 4 + j) as i32, RET);
                }
            }

            self.sub_rsp(align_stack(count_obs as u32 * 32));
            self.amd.mov(STATES, MEM);
            self.amd.mov(MEM, STACK);

            self.set_label("@main");
        }
        self.sub_rsp(align_stack(cap as u32 * self.reg_size()));
    }

    fn epilogue_symbolica(&mut self, cap: usize, count_params: usize, count_obs: usize) {
        self.add_rsp(align_stack(cap as u32 * self.reg_size()));

        if self.reg_size() == 32 {
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
            self.amd.add_rsp(frame_size);
            self.set_label("@done");
        }

        self.vzeroupper();

        self.load_nonvolatile_regs();
        self.amd.pop(Amd::RBP);
        self.amd.ret();
    }
}

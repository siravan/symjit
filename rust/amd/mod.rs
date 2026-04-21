use crate::code::Func;
use crate::config::{Config, SPILL_AREA};
use crate::generator::Generator;
use crate::utils::align_stack;
use crate::utils::{is_external_func, reg, DataType, Reg};
use anyhow::{anyhow, Result};

mod asm;
mod fused;

use asm::{Amd, RoundingMode};

mod scalar;
mod sse;
mod vector;

pub use scalar::AmdScalarGenerator;
pub use sse::AmdSSEGenerator;
pub use vector::AmdVectorGenerator;

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

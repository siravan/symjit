use super::code::Func;
use super::utils::{align_stack, Reg};

mod asm;
mod f64x8;
mod fused;

use asm::Amd;

mod complex;
mod scalar;
mod sse;
mod vector;
mod vector_f64x8;

pub use f64x8::Prefix;

pub use complex::AmdComplexGenerator;
pub use scalar::AmdScalarGenerator;
pub use sse::AmdSSEGenerator;
pub use vector::AmdVectorF64x4Generator;
pub use vector_f64x8::AmdVectorF64x8Generator;

#[cfg(target_family = "windows")]
const ARGS: [u8; 4] = [Amd::RCX, Amd::RDX, Amd::R8, Amd::R9];

#[cfg(target_family = "unix")]
const ARGS: [u8; 4] = [Amd::RDI, Amd::RSI, Amd::RDX, Amd::RCX];

const RET: u8 = 0;

const MEM: u8 = Amd::R13;
const STATES: u8 = Amd::R14;
const IDX: u8 = Amd::R12; // modrm_mem does not handle R12 correctly, therefore IDX, which is not a pointer, is assigned to R12
const PARAMS: u8 = Amd::R15;
const STACK: u8 = Amd::RBX;
const SP: u8 = Amd::RSP;

fn save_nonvolatile_regs(amd: &mut Amd) {
    amd.push(Amd::RBP);
    amd.sub_rsp(48);
    amd.mov_mem_reg(SP, 0x00, MEM);
    amd.mov_mem_reg(SP, 0x08, STATES);
    amd.mov_mem_reg(SP, 0x10, IDX);
    amd.mov_mem_reg(SP, 0x18, PARAMS);
    amd.mov_mem_reg(SP, 0x20, STACK);
    // note that [SP + 0x28] is reserved for allocate_stack
    amd.mov(Amd::RBP, SP);

    amd.mov(MEM, ARGS[0]); // first arg = mem if direct mode, otherwise null
    amd.mov(STATES, ARGS[1]); // second arg = states+obs if indirect mode, otherwise null or arena
    amd.mov(IDX, ARGS[2]); // third arg = index if indirect mode
    amd.mov(PARAMS, ARGS[3]); // fourth arg = params
}

fn load_nonvolatile_regs(amd: &mut Amd) {
    amd.vzeroupper();
    amd.mov(SP, Amd::RBP);
    amd.mov_reg_mem(MEM, SP, 0x00);
    amd.mov_reg_mem(STATES, SP, 0x08);
    amd.mov_reg_mem(IDX, SP, 0x10);
    amd.mov_reg_mem(PARAMS, SP, 0x18);
    amd.mov_reg_mem(STACK, SP, 0x20);
    amd.add_rsp(48);
    amd.pop(Amd::RBP);
}

fn allocate_stack(amd: &mut Amd, size: u32, _with_arena: bool) {
    amd.sub_rsp(align_stack(size));
    amd.and_imm(SP, 0xffffffc0);
    amd.mov(STACK, SP);
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

/*
fn add_rsp(amd: &mut Amd, size: u32) {
    if size != 0 {
        amd.add_rsp(size);
    }
}
*/

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

    amd.a.set_label("_half_");
    amd.a.append_quad(0.5f64.to_bits());

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
#[allow(clippy::collapsible_if)]
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

fn add_func(amd: &mut Amd, op: &str, f: Func) {
    if let Func::Slice {
        f_scalar,
        f_simd,
        env,
        ..
    } = f
    {
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

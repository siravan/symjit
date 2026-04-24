#[macro_use]
mod macros;

use anyhow::{anyhow, Result};

use crate::assembler::{Assembler, Jumper};
use crate::code::Func;
use crate::config::{Config, SPILL_AREA};
use crate::generator::Generator;
use crate::utils::{align_stack, is_external_func, reg, Reg};

const SP: u8 = 31;

const MEM: u8 = 19; // first arg = mem if direct mode, otherwise null
const PARAMS: u8 = 20; // fourth arg = params
const STATES: u8 = 21; // second arg = states+obs if indirect mode, otherwise null
const IDX: u8 = 22; // third arg = index if indirect mode
const CALL: u8 = 23; // call pointer

const SCRATCH1: u8 = 9;
const SCRATCH2: u8 = 10;
const TEMP: u8 = 1;

mod scalar;
mod vector;

pub use scalar::ArmGenerator;
pub use vector::ArmSimdGenerator;

fn ϕ(r: Reg) -> u8 {
    match r {
        Reg::Ret => 0,  // d0
        Reg::Temp => 1, // d1
        Reg::Left => 0,
        Reg::Right => 1,
        Reg::Gen(dst) => {
            if dst < 6 {
                dst + 2 // d2-d7
            } else if dst < 22 {
                dst + 10 // d16-d31
            } else {
                dst - 14 // d8-d15 (non-volatile)
            }
        }
        Reg::Static(..) => panic!("passing static registers to codegen"),
    }
}

fn emit(a: &mut Assembler, w: u32) {
    a.append_word(w);
}

fn load_d_from_mem(a: &mut Assembler, d: u8, base: u8, idx: u32) {
    if idx < 4096 {
        emit(a, arm! {ldr d(d), [x(base), #8*idx]});
    } else if idx < 65536 {
        emit(a, arm! {movz x(SCRATCH1), #idx});
        emit(a, arm! {ldr d(d), [x(base), x(SCRATCH1), lsl #3]});
    } else {
        emit(a, arm! {movz x(SCRATCH1), #idx & 0xffff});
        emit(a, arm! {movk_lsl16 x(SCRATCH1), #idx >> 16});
        emit(a, arm! {ldr d(d), [x(base), x(SCRATCH1), lsl #3]});
    }
}

fn save_d_to_mem(a: &mut Assembler, d: u8, base: u8, idx: u32) {
    if idx < 4096 {
        emit(a, arm! {str d(d), [x(base), #8*idx]});
    } else if idx < 65536 {
        emit(a, arm! {movz x(SCRATCH1), #idx});
        emit(a, arm! {str d(d), [x(base), x(SCRATCH1), lsl #3]});
    } else {
        emit(a, arm! {movz x(SCRATCH1), #idx & 0xffff});
        emit(a, arm! {movk_lsl16 x(SCRATCH1), #idx >> 16});
        emit(a, arm! {str d(d), [x(base), x(SCRATCH1), lsl #3]});
    }
}

fn load_q_from_mem(a: &mut Assembler, d: u8, base: u8, mut idx: u32) {
    if idx < 4096 {
        emit(a, arm! {ldr q(d), [x(base), #16*idx]});
    } else if idx < 65536 {
        emit(a, arm! {movz x(SCRATCH1), #idx});
        emit(a, arm! {ldr q(d), [x(base), x(SCRATCH1), lsl #4]});
    } else {
        emit(a, arm! {movz x(SCRATCH1), #idx & 0xffff});
        emit(a, arm! {movk_lsl16 x(SCRATCH1), #idx >> 16});
        emit(a, arm! {ldr q(d), [x(base), x(SCRATCH1), lsl #4]});
    }
}

fn save_q_to_mem(a: &mut Assembler, d: u8, base: u8, mut idx: u32) {
    if idx < 4096 {
        emit(a, arm! {str q(d), [x(base), #16*idx]});
    } else if idx < 65536 {
        emit(a, arm! {movz x(SCRATCH1), #idx});
        emit(a, arm! {str q(d), [x(base), x(SCRATCH1), lsl #4]});
    } else {
        emit(a, arm! {movz x(SCRATCH1), #idx & 0xffff});
        emit(a, arm! {movk_lsl16 x(SCRATCH1), #idx >> 16});
        emit(a, arm! {str q(d), [x(base), x(SCRATCH1), lsl #4]});
    }
}

fn load_x_from_mem(a: &mut Assembler, r: u8, base: u8, idx: u32) {
    assert!(r != 9);

    if idx < 4096 {
        emit(a, arm! {ldr x(r), [x(base), #8*idx]});
    } else if idx < 65536 {
        emit(a, arm! {movz x(SCRATCH1), #idx});
        emit(a, arm! {ldr x(r), [x(base), x(SCRATCH1), lsl #3]});
    } else {
        emit(a, arm! {movz x(SCRATCH1), #idx & 0xffff});
        emit(a, arm! {movk_lsl16 x(SCRATCH1), #idx >> 16});
        emit(a, arm! {ldr x(r), [x(base), x(SCRATCH1), lsl #3]});
    }
}

fn load_x_from_label(a: &mut Assembler, dst: u8, label: &str) {
    a.jump_abs(label, (self.ip() & 0xfffff000) as u32, |offset, pg| {
        arm! {adrp x(9), label((offset - pg as i32) as u32)}
    });

    a.jump_abs(
        label,
        dst as u32,
        |offset, dst| arm! {ldr x(dst), [x(9), #offset & 0x0fff]},
    );
}

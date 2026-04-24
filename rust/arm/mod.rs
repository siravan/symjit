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

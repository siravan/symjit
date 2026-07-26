#[macro_use]
mod macros;

use crate::assembler::Assembler;
use crate::code::Func;
use crate::utils::Reg;

const SP: u8 = 31;

const MEM: u8 = 19; // first arg = mem if direct mode, otherwise null
const PARAMS: u8 = 20; // fourth arg = params
const STATES: u8 = 21; // second arg = states+obs if indirect mode, otherwise null
const IDX: u8 = 22; // third arg = index if indirect mode
const CALL: u8 = 23; // call pointer
const STACK: u8 = 24;

const SCRATCH1: u8 = 9;
const SCRATCH2: u8 = 10;
const SCRATCH3: u8 = 11;
const COUNTER: u8 = 12;
const TEMP: u8 = ϕ(Reg::Temp);

/*
 * registers v8 to v16 are ABI-preserved
 * registers v29-v31 can be temporary
 */
const FMAP: [u8; 30] = [
    2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 8, 9, 10, 11, 12, 13, 14,
    15, 29, 30, 31,
];

const T0: u8 = 29;
const T1: u8 = 30;
const T2: u8 = 31;

mod complex;
mod scalar;
mod vector;

pub use complex::ArmComplexGenerator;
pub use scalar::ArmGenerator;
pub use vector::ArmSimdGenerator;

const fn ϕ(r: Reg) -> u8 {
    match r {
        Reg::Ret => 0,  // d0
        Reg::Temp => 1, // d1
        Reg::Left => 0,
        Reg::Right => 1,
        Reg::Gen(dst) => FMAP[dst as usize],
        Reg::Static(..) => panic!("passing static registers to codegen"),
    }
}

fn emit(a: &mut Assembler, w: u32) {
    a.append_word(w);
}

fn load_long(a: &mut Assembler, reg: u8, label: &str) {
    let data = (a.ip() & 0xfffff000) as u32 | reg as u32;

    a.jump_abs(label, data, |offset, data| {
        let pg = (data & 0xfffff000) as i32;
        let reg = (data & 0xff) as u8;
        arm! {adrp x(reg), label((offset - pg) as u32)}
    });

    a.jump_abs(
        label,
        reg as u32,
        |offset, reg| arm! {ldr x(reg), [x(reg), #offset & 0x0fff]},
    );
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

fn load_paired_d_from_mem(a: &mut Assembler, d1: u8, d2: u8, base: u8, idx: u32) {
    assert!(idx & 1 == 0);

    match idx {
        0..64 => emit(a, arm! {ldp d(d1), d(d2), [x(base), #8*idx]}),
        64..256 => {
            emit(a, arm! {add x(SCRATCH1), x(base), #8*idx});
            emit(a, arm! {ldp d(d1), d(d2), [x(SCRATCH1), #0]});
        }
        256..4096 => {
            emit(a, arm! {ldr d(d1), [x(base), #8*idx]});
            emit(a, arm! {ldr d(d2), [x(base), #8*(idx+1)]});
        }
        4096..65536 => {
            emit(a, arm! {movz x(SCRATCH1), #idx});
            emit(a, arm! {add x(SCRATCH1), x(base), x(SCRATCH1), lsl #3});
            emit(a, arm! {ldp d(d1), d(d2), [x(SCRATCH1), #0]});
        }
        65536..131072 => {
            emit(a, arm! {movz x(SCRATCH1), #idx >> 1});
            emit(a, arm! {add x(SCRATCH1), x(base), x(SCRATCH1), lsl #4});
            emit(a, arm! {ldp d(d1), d(d2), [x(SCRATCH1), #0]});
        }
        idx => {
            emit(a, arm! {movz x(SCRATCH1), #idx & 0xffff});
            emit(a, arm! {movk_lsl16 x(SCRATCH1), #idx >> 16});
            emit(a, arm! {add x(SCRATCH1), x(base), x(SCRATCH1), lsl #3});
            emit(a, arm! {ldp d(d1), d(d2), [x(SCRATCH1), #0]});
        }
    }
}

fn save_paired_d_to_mem(a: &mut Assembler, d1: u8, d2: u8, base: u8, idx: u32) {
    assert!(idx & 1 == 0);

    match idx {
        0..64 => emit(a, arm! {stp d(d1), d(d2), [x(base), #8*idx]}),
        64..256 => {
            emit(a, arm! {add x(SCRATCH1), x(base), #8*idx});
            emit(a, arm! {stp d(d1), d(d2), [x(SCRATCH1), #0]});
        }
        256..4096 => {
            emit(a, arm! {str d(d1), [x(base), #8*idx]});
            emit(a, arm! {str d(d2), [x(base), #8*(idx+1)]});
        }
        4096..65536 => {
            emit(a, arm! {movz x(SCRATCH1), #idx});
            emit(a, arm! {add x(SCRATCH1), x(base), x(SCRATCH1), lsl #3});
            emit(a, arm! {stp d(d1), d(d2), [x(SCRATCH1), #0]});
        }
        65536..131072 => {
            emit(a, arm! {movz x(SCRATCH1), #idx >> 1});
            emit(a, arm! {add x(SCRATCH1), x(base), x(SCRATCH1), lsl #4});
            emit(a, arm! {stp d(d1), d(d2), [x(SCRATCH1), #0]});
        }
        idx => {
            emit(a, arm! {movz x(SCRATCH1), #idx & 0xffff});
            emit(a, arm! {movk_lsl16 x(SCRATCH1), #idx >> 16});
            emit(a, arm! {add x(SCRATCH1), x(base), x(SCRATCH1), lsl #3});
            emit(a, arm! {stp d(d1), d(d2), [x(SCRATCH1), #0]});
        }
    }
}

fn load_q_from_mem(a: &mut Assembler, d: u8, base: u8, idx: u32) {
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

fn save_q_to_mem(a: &mut Assembler, d: u8, base: u8, idx: u32) {
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

fn load_paired_q_from_mem(a: &mut Assembler, d1: u8, d2: u8, base: u8, idx: u32) {
    assert!(idx & 1 == 0);

    match idx {
        0..64 => emit(a, arm! {ldp q(d1), q(d2), [x(base), #16*idx]}),
        64..256 => {
            emit(a, arm! {add x(SCRATCH1), x(base), #16*idx});
            emit(a, arm! {ldp q(d1), q(d2), [x(SCRATCH1), #0]});
        }
        256..4096 => {
            emit(a, arm! {ldr q(d1), [x(base), #16*idx]});
            emit(a, arm! {ldr q(d2), [x(base), #16*(idx+1)]});
        }
        4096..65536 => {
            emit(a, arm! {movz x(SCRATCH1), #idx});
            emit(a, arm! {add x(SCRATCH1), x(base), x(SCRATCH1), lsl #4});
            emit(a, arm! {ldp q(d1), q(d2), [x(SCRATCH1), #0]});
        }
        65536..131072 => {
            emit(a, arm! {movz x(SCRATCH1), #idx >> 1});
            emit(a, arm! {add x(SCRATCH1), x(base), x(SCRATCH1), lsl #5});
            emit(a, arm! {ldp q(d1), q(d2), [x(SCRATCH1), #0]});
        }
        idx => {
            emit(a, arm! {movz x(SCRATCH1), #idx & 0xffff});
            emit(a, arm! {movk_lsl16 x(SCRATCH1), #idx >> 16});
            emit(a, arm! {add x(SCRATCH1), x(base), x(SCRATCH1), lsl #4});
            emit(a, arm! {ldp q(d1), q(d2), [x(SCRATCH1), #0]});
        }
    }
}

fn save_paired_q_to_mem(a: &mut Assembler, d1: u8, d2: u8, base: u8, idx: u32) {
    assert!(idx & 1 == 0);

    match idx {
        0..64 => emit(a, arm! {stp q(d1), q(d2), [x(base), #16*idx]}),
        64..256 => {
            emit(a, arm! {add x(SCRATCH1), x(base), #16*idx});
            emit(a, arm! {stp q(d1), q(d2), [x(SCRATCH1), #0]});
        }
        256..4096 => {
            emit(a, arm! {str q(d1), [x(base), #16*idx]});
            emit(a, arm! {str q(d2), [x(base), #16*(idx+1)]});
        }
        4096..65536 => {
            emit(a, arm! {movz x(SCRATCH1), #idx});
            emit(a, arm! {add x(SCRATCH1), x(base), x(SCRATCH1), lsl #4});
            emit(a, arm! {stp q(d1), q(d2), [x(SCRATCH1), #0]});
        }
        65536..131072 => {
            emit(a, arm! {movz x(SCRATCH1), #idx >> 1});
            emit(a, arm! {add x(SCRATCH1), x(base), x(SCRATCH1), lsl #5});
            emit(a, arm! {stp q(d1), q(d2), [x(SCRATCH1), #0]});
        }
        idx => {
            emit(a, arm! {movz x(SCRATCH1), #idx & 0xffff});
            emit(a, arm! {movk_lsl16 x(SCRATCH1), #idx >> 16});
            emit(a, arm! {add x(SCRATCH1), x(base), x(SCRATCH1), lsl #4});
            emit(a, arm! {stp q(d1), q(d2), [x(SCRATCH1), #0]});
        }
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
    load_long(a, dst, label);
}

fn add_consts(a: &mut Assembler, consts: &[f64]) {
    for (idx, val) in consts.iter().enumerate() {
        let label = format!("_const_{}_", idx);
        a.set_label(label.as_str());
        a.append_quad((*val).to_bits());
    }
}

fn add_func(a: &mut Assembler, op: &str, f: Func) {
    if let Func::Slice {
        f_scalar,
        f_simd,
        env,
        ..
    } = f
    {
        let label = format!("_func_{}_", op);
        a.set_label(label.as_str());
        a.append_quad(f_scalar as u64);

        let label = format!("_simd_{}_", op);
        a.set_label(label.as_str());
        a.append_quad(f_simd as u64);

        let label = format!("_env_{}_", op);
        a.set_label(label.as_str());
        a.append_quad(env as u64);
    } else {
        let label = format!("_func_{}_", op);
        a.set_label(label.as_str());
        a.append_quad(f.func_ptr());
    }
}

fn sub_stack(a: &mut Assembler, size: u32) {
    emit(a, arm! {sub sp, sp, #size & 0x0fff});
    if size >> 12 != 0 {
        emit(a, arm! {sub sp, sp, #size >> 12, lsl #12});
    }
}

fn add_stack(a: &mut Assembler, size: u32) {
    if size >> 12 != 0 {
        emit(a, arm! {add sp, sp, #size >> 12, lsl #12});
    }
    emit(a, arm! {add sp, sp, #size & 0x0fff});
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::generator::{FuncletType, Generator};

    fn first_funclet_call<G: Generator>(mut generator: G) -> u32 {
        generator.call_funclet("target");
        generator.branch("done");
        generator.set_label("target");
        generator.ret();
        generator.set_label("done");
        generator.seal();

        let bytes = generator.bytes();
        u32::from_le_bytes(bytes[..4].try_into().unwrap())
    }

    #[test]
    fn generators_emit_relative_funclet_calls() {
        let config = Config::default();

        let scalar = ArmGenerator::new(config.clone());
        assert!(matches!(scalar.support_funclet(), FuncletType::Complex));
        assert_eq!(first_funclet_call(scalar), 0x9400_0002);

        let vector = ArmSimdGenerator::new(config.clone());
        assert!(matches!(vector.support_funclet(), FuncletType::Complex));
        assert_eq!(first_funclet_call(vector), 0x9400_0002);

        let complex = ArmComplexGenerator::new(config);
        assert!(matches!(complex.support_funclet(), FuncletType::Real));
        assert_eq!(first_funclet_call(complex), 0x9400_0002);
    }
}

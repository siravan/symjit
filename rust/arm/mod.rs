#[macro_use]
mod macros;

use crate::assembler::Assembler;
use crate::generator::Generator;
use crate::utils::{align_stack, Reg};

const SP: u8 = 31;
const MEM: u8 = 19; // first arg = mem if direct mode, otherwise null
const STATES: u8 = 21; // second arg = states+obs if indirect mode, otherwise null
const IDX: u8 = 22; // third arg = index if indirect mode
const PARAMS: u8 = 20; // fourth arg = params
const SCRATCH1: u8 = 9;
const SCRATCH2: u8 = 10;
const TEMP: u8 = 1;

pub struct ArmGenerator {
    a: Assembler,
    mask: u32,
}

fn ϕ(r: Reg) -> u8 {
    match r {
        Reg::Ret => 0,
        Reg::Temp => 1,
        Reg::Left => 0,
        Reg::Right => 1,
        Reg::Gen(dst) => dst + 2,
    }
}

impl ArmGenerator {
    pub fn new() -> ArmGenerator {
        ArmGenerator {
            a: Assembler::new(0, 3),
            mask: 0x00ff,
        }
    }

    fn count_shadows(&self) -> u8 {
        6
    }

    fn reg_size(&self) -> u32 {
        8
    }

    fn append_quad(&mut self, u: u64) {
        self.a.append_quad(u);
    }

    fn set_label(&mut self, label: &str) {
        self.a.set_label(label);
    }

    fn jump(&mut self, label: &str, code: u32) {
        self.a.jump(label, code)
    }

    fn apply_jumps(&mut self) {
        self.a.apply_jumps();
    }

    fn emit(&mut self, w: u32) {
        self.a.append_word(w);
    }

    fn flush(&mut self, dst: Reg) {
        let reg = ϕ(dst);
        let m = 1 << reg;

        if self.mask & m == 0 {
            // self.emit(arm! {str d(reg), [sp, #8*idx]});
            self.save_d_to_mem(reg, SP, reg as u32);
        }

        self.mask |= m;
    }

    fn restore_regs(&mut self) {
        let last = ϕ(Reg::Gen(self.count_shadows()));

        for reg in last..16 {
            let m = 1 << reg;

            if self.mask & m != 0 {
                // self.emit(arm! {ldr d(reg), [sp, #8*idx]});
                self.load_d_from_mem(reg, SP, reg as u32);
            }
        }
    }

    fn load_d_from_mem(&mut self, d: u8, base: u8, idx: u32) {
        if idx < 4096 {
            self.emit(arm! {ldr d(d), [x(base), #8*idx]});
        } else {
            self.emit(arm! {movz x(SCRATCH1), #idx});
            self.emit(arm! {ldr d(d), [x(base), x(SCRATCH1), lsl #3]});
        }
    }

    fn save_d_to_mem(&mut self, d: u8, base: u8, idx: u32) {
        if idx < 4096 {
            self.emit(arm! {str d(d), [x(base), #8*idx]});
        } else {
            self.emit(arm! {movz x(SCRATCH1), #idx});
            self.emit(arm! {str d(d), [x(base), x(SCRATCH1), lsl #3]});
        }
    }

    fn load_x_from_mem(&mut self, r: u8, base: u8, idx: u32) {
        assert!(r != 9);

        if idx < 4096 {
            self.emit(arm! {ldr x(r), [x(base), #8*idx]});
        } else {
            self.emit(arm! {movz x(SCRATCH1), #idx});
            self.emit(arm! {ldr x(r), [x(base), x(SCRATCH1), lsl #3]});
        }
    }

    fn sub_stack(&mut self, size: u32) {
        self.emit(arm! {sub sp, sp, #size & 0x0fff});
        if size >> 12 != 0 {
            self.emit(arm! {sub sp, sp, #size >> 12, lsl #12});
        }
    }

    fn add_stack(&mut self, size: u32) {
        if size >> 12 != 0 {
            self.emit(arm! {add sp, sp, #size >> 12, lsl #12});
        }
        self.emit(arm! {add sp, sp, #size & 0x0fff});
    }
}

impl Generator for ArmGenerator {
    fn bytes(&mut self) -> Vec<u8> {
        self.a.bytes()
    }

    fn three_address(&self) -> bool {
        true
    }

    fn seal(&mut self) {
        self.apply_jumps();
    }

    //***********************************

    fn fmov(&mut self, dst: Reg, s1: Reg) {
        if dst == s1 {
            return;
        }

        self.flush(dst);
        self.emit(arm! {fmov d(ϕ(dst)), d(ϕ(s1))});
    }

    fn fxchg(&mut self, s1: Reg, s2: Reg) {
        self.flush(s1);
        self.flush(s2);

        self.emit(arm! {eor v(ϕ(s1)).8b, v(ϕ(s1)).8b, v(ϕ(s2)).8b});
        self.emit(arm! {eor v(ϕ(s2)).8b, v(ϕ(s1)).8b, v(ϕ(s2)).8b});
        self.emit(arm! {eor v(ϕ(s1)).8b, v(ϕ(s1)).8b, v(ϕ(s2)).8b});
    }

    fn load_const(&mut self, dst: Reg, idx: u32) {
        self.flush(dst);
        let label = format!("_const_{}_", idx);
        self.jump(label.as_str(), arm! {ldr d(ϕ(dst)), label});
    }

    fn load_mem(&mut self, dst: Reg, idx: u32) {
        self.flush(dst);
        self.load_d_from_mem(ϕ(dst), MEM, idx);
    }

    fn save_mem(&mut self, dst: Reg, idx: u32) {
        self.save_d_to_mem(ϕ(dst), MEM, idx);
    }

    fn save_mem_result(&mut self, idx: u32) {
        self.save_mem(Reg::Ret, idx);
    }

    fn load_param(&mut self, dst: Reg, idx: u32) {
        self.flush(dst);
        self.load_d_from_mem(ϕ(dst), PARAMS, idx);
    }

    fn load_stack(&mut self, dst: Reg, idx: u32) {
        self.flush(dst);
        self.load_d_from_mem(ϕ(dst), SP, idx);
    }

    fn save_stack(&mut self, dst: Reg, idx: u32) {
        self.save_d_to_mem(ϕ(dst), SP, idx);
    }

    fn save_stack_result(&mut self, idx: u32) {
        self.save_stack(Reg::Ret, idx);
    }

    fn neg(&mut self, dst: Reg, s1: Reg) {
        self.flush(dst);
        self.emit(arm! {fneg d(ϕ(dst)), d(ϕ(s1))});
    }

    fn abs(&mut self, dst: Reg, s1: Reg) {
        self.flush(dst);
        self.emit(arm! {fabs d(ϕ(dst)), d(ϕ(s1))});
    }

    fn root(&mut self, dst: Reg, s1: Reg) {
        self.flush(dst);
        self.emit(arm! {fsqrt d(ϕ(dst)), d(ϕ(s1))});
    }

    fn recip(&mut self, dst: Reg, s1: Reg) {
        self.flush(dst);
        self.emit(arm! {fmov d(TEMP), #1.0});
        self.emit(arm! {fdiv d(ϕ(dst)), d(TEMP), d(ϕ(s1))});
    }

    fn round(&mut self, dst: Reg, s1: Reg) {
        self.flush(dst);
        self.emit(arm! {frinti d(ϕ(dst)), d(ϕ(s1))});
    }

    fn floor(&mut self, dst: Reg, s1: Reg) {
        self.flush(dst);
        self.emit(arm! {frintm d(ϕ(dst)), d(ϕ(s1))});
    }

    fn ceiling(&mut self, dst: Reg, s1: Reg) {
        self.flush(dst);
        self.emit(arm! {frintp d(ϕ(dst)), d(ϕ(s1))});
    }

    fn trunc(&mut self, dst: Reg, s1: Reg) {
        self.flush(dst);
        self.emit(arm! {frintz d(ϕ(dst)), d(ϕ(s1))});
    }

    fn frac(&mut self, dst: Reg, s1: Reg) {
        self.floor(Reg::Temp, s1);
        self.minus(dst, s1, Reg::Temp);
    }

    fn plus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.flush(dst);
        self.emit(arm! {fadd d(ϕ(dst)), d(ϕ(s1)), d(ϕ(s2))});
    }

    fn minus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.flush(dst);
        self.emit(arm! {fsub d(ϕ(dst)), d(ϕ(s1)), d(ϕ(s2))});
    }

    fn times(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.flush(dst);
        self.emit(arm! {fmul d(ϕ(dst)), d(ϕ(s1)), d(ϕ(s2))});
    }

    fn divide(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.flush(dst);
        self.emit(arm! {fdiv d(ϕ(dst)), d(ϕ(s1)), d(ϕ(s2))});
    }

    fn gt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.flush(dst);
        self.emit(arm! {fcmgt d(ϕ(dst)), d(ϕ(s1)), d(ϕ(s2))});
    }

    fn geq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.flush(dst);
        self.emit(arm! {fcmge d(ϕ(dst)), d(ϕ(s1)), d(ϕ(s2))});
    }

    fn lt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.flush(dst);
        self.emit(arm! {fcmlt d(ϕ(dst)), d(ϕ(s1)), d(ϕ(s2))});
    }

    fn leq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.flush(dst);
        self.emit(arm! {fcmle d(ϕ(dst)), d(ϕ(s1)), d(ϕ(s2))});
    }

    fn eq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.flush(dst);
        self.emit(arm! {fcmeq d(ϕ(dst)), d(ϕ(s1)), d(ϕ(s2))});
    }

    fn neq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.flush(dst);
        self.emit(arm! {fcmeq d(ϕ(dst)), d(ϕ(s1)), d(ϕ(s2))});
        self.emit(arm! {not v(ϕ(dst)).8b, v(ϕ(dst)).8b});
    }

    fn and(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.flush(dst);
        self.emit(arm! {and v(ϕ(dst)).8b, v(ϕ(s1)).8b, v(ϕ(s2)).8b});
    }

    fn andnot(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.flush(dst);
        self.emit(arm! {not v(ϕ(s1)).8b, v(ϕ(s1)).8b});
        self.emit(arm! {and v(ϕ(dst)).8b, v(ϕ(s1)).8b, v(ϕ(s2)).8b});
    }

    fn or(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.flush(dst);
        self.emit(arm! {orr v(ϕ(dst)).8b, v(ϕ(s1)).8b, v(ϕ(s2)).8b});
    }

    fn xor(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.flush(dst);
        self.emit(arm! {eor v(ϕ(dst)).8b, v(ϕ(s1)).8b, v(ϕ(s2)).8b});
    }

    fn not(&mut self, dst: Reg, s1: Reg) {
        self.flush(dst);
        self.emit(arm! {not v(ϕ(dst)).8b, v(ϕ(s1)).8b});
    }

    fn fused_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        // self.times(Reg::Temp, s1, s2);
        // self.plus(dst, Reg::Temp, s3);
        self.emit(arm! {fmadd d(ϕ(dst)), d(ϕ(s1)), d(ϕ(s2)), d(ϕ(s3))});
    }

    // fused_mul_sub is s1 * s2 - s3, corresponding to fnmsub in aarch64
    // and vmsub... in amd64
    fn fused_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        // self.times(Reg::Temp, s1, s2);
        // self.minus(dst, Reg::Temp, s3);
        self.emit(arm! {fnmsub d(ϕ(dst)), d(ϕ(s1)), d(ϕ(s2)), d(ϕ(s3))});
    }

    // fused_neg_mul_add is s3 - s1 * s2, corresponding to fmsub in aarch64
    // and vnmadd... in amd64
    fn fused_neg_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        // self.times(Reg::Temp, s1, s2);
        // self.minus(dst, s3, Reg::Temp);
        self.emit(arm! {fmsub d(ϕ(dst)), d(ϕ(s1)), d(ϕ(s2)), d(ϕ(s3))});
    }

    // fused_neg_mul_sub is -s3 - s1 * s2, corresponding to fnmadd in aarch64
    // and vnmsub... in amd64
    fn fused_neg_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        // self.times(Reg::Temp, s1, s2);
        // self.plus(dst, Reg::Temp, s3);
        // self.neg(dst, dst);
        self.emit(arm! {fnmadd d(ϕ(dst)), d(ϕ(s1)), d(ϕ(s2)), d(ϕ(s3))});
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

    fn call(&mut self, op: &str, _num_args: usize) {
        let label = format!("_func_{}_", op);
        self.jump(label.as_str(), arm! {ldr x(0), label});
        self.emit(arm! {blr x(0)});
    }

    fn prologue_fast(&mut self, cap: u32, num_args: u32) {
        self.emit(arm! {sub sp, sp, #16});
        self.emit(arm! {str lr, [sp, #0]});
        self.emit(arm! {str x(MEM), [sp, #8]});

        let stack_size = align_stack(self.reg_size() * cap);
        self.sub_stack(stack_size);

        self.emit(arm! {mov x(MEM), sp});

        let num_args = num_args as i32;

        for i in 0..num_args {
            self.emit(arm! {str d(i), [sp, #8*i]});
            self.mask |= 1 << i;
        }
    }

    fn epilogue_fast(&mut self, cap: u32, idx_ret: i32) {
        self.restore_regs();

        self.emit(arm! {ldr d(0), [sp, #8*idx_ret]});

        let stack_size = align_stack(self.reg_size() * cap);
        self.add_stack(stack_size);

        self.emit(arm! {ldr x(MEM), [sp, #8]});
        self.emit(arm! {ldr lr, [sp, #0]});
        self.emit(arm! {add sp, sp, #16});
        self.emit(arm! {ret});
    }

    /*
     * MEM => first arg = mem if direct mode, otherwise null
     * STATES => second arg = states+obs if indirect mode, otherwise null
     * IDX => third arg = index if indirect mode
     * PARAMS => fourth arg = params
     */
    fn prologue_indirect(&mut self, cap: u32, count_states: usize, count_obs: usize) {
        self.emit(arm! {sub sp, sp, #48});
        self.emit(arm! {str lr, [sp, #0]});
        self.emit(arm! {str x(MEM), [sp, #8]});
        self.emit(arm! {str x(PARAMS), [sp, #16]});
        self.emit(arm! {str x(STATES), [sp, #24]});
        self.emit(arm! {str x(IDX), [sp, #32]});

        self.emit(arm! {mov x(MEM), x(0)});
        self.emit(arm! {mov x(STATES), x(1)});
        self.emit(arm! {mov x(IDX), x(2)});
        self.emit(arm! {mov x(PARAMS), x(3)});

        self.emit(arm! {tst x(STATES), x(STATES)});
        self.jump("@main", arm! {b.eq label});

        let size = align_stack((count_states + count_obs + 1) as u32 * self.reg_size());
        self.sub_stack(size);
        self.emit(arm! {mov x(MEM), sp});

        for i in 0..count_states {
            // self.emit(arm! {ldr x(10), [x(states), #8*i]});
            self.load_x_from_mem(SCRATCH2, STATES, i as u32);
            self.emit(arm! {ldr d(0), [x(SCRATCH2), x(IDX), lsl #3]});
            // self.emit(arm! {str d(0), [x(mem), #8*i]});
            self.save_d_to_mem(0, MEM, i as u32);
        }

        // TODO: may save idx (RDX) as double in RBP + 8/32 * count_states

        self.set_label("@main");

        let stack_size = align_stack(self.reg_size() * cap);
        self.sub_stack(stack_size);
    }

    fn epilogue_indirect(&mut self, cap: u32, count_states: usize, count_obs: usize) {
        let stack_size = align_stack(self.reg_size() * cap);
        self.add_stack(stack_size);

        self.emit(arm! {tst x(STATES), x(STATES)});
        self.jump("@done", arm! {b.eq label});

        for i in 0..count_obs {
            // self.emit(arm! {ldr x(10), [x(states), #8*(count_states+i)]});
            self.load_x_from_mem(SCRATCH2, STATES, (count_states + i) as u32);
            let k = (count_states + i + 1) as u32;
            //self.emit(arm! {ldr d(0), [x(mem), #8*k]});
            self.load_d_from_mem(0, MEM, k);
            self.emit(arm! {str d(0), [x(SCRATCH2), x(IDX), lsl #3]});
        }

        let size = align_stack((count_states + count_obs + 1) as u32 * self.reg_size());
        self.add_stack(size);

        self.set_label("@done");

        self.restore_regs();

        self.emit(arm! {ldr x(IDX), [sp, #32]});
        self.emit(arm! {ldr x(STATES), [sp, #24]});
        self.emit(arm! {ldr x(PARAMS), [sp, #16]});
        self.emit(arm! {ldr x(MEM), [sp, #8]});
        self.emit(arm! {ldr lr, [sp, #0]});
        self.emit(arm! {add sp, sp, #48});
        self.emit(arm! {ret});
    }
}

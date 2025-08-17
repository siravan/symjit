#[macro_use]
mod macros;

use crate::assembler::Assembler;
use crate::generator::{fmod, powi, powi_mod, setup_call_binary, setup_call_unary, Generator};
use crate::utils::{align_stack, Reg};

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

const TEMP: u8 = 1;

impl ArmGenerator {
    pub fn new() -> ArmGenerator {
        ArmGenerator {
            a: Assembler::new(0, 3),
            mask: 0x00ff,
        }
    }

    fn emit(&mut self, w: u32) {
        self.a.append_word(w);
    }

    fn flush(&mut self, dst: Reg) {
        let reg = ϕ(dst);

        let m = 1 << reg;
        let idx = reg as i32;

        if self.mask & m == 0 {
            self.emit(arm! {str d(reg), [sp, #8*idx]});
        }

        self.mask |= m;
    }

    fn restore_regs(&mut self) {
        let last = ϕ(Reg::Gen(self.count_shadows()));

        for reg in last..16 {
            let m = 1 << reg;
            let idx = reg as i32;

            if self.mask & m != 0 {
                self.emit(arm! {ldr d(reg), [sp, #8*idx]});
            }
        }
    }
}

impl Generator for ArmGenerator {
    fn count_shadows(&self) -> u8 {
        6
    }

    fn reg_size(&self) -> u32 {
        8
    }

    fn a(&mut self) -> &mut Assembler {
        &mut self.a
    }

    fn three_address(&self) -> bool {
        true
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

    fn load_const(&mut self, dst: Reg, label: &str) {
        self.flush(dst);
        self.jump(label, arm! {ldr d(ϕ(dst)), label});
    }

    fn load_mem(&mut self, dst: Reg, idx: u32) {
        self.flush(dst);
        self.emit(arm! {ldr d(ϕ(dst)), [x(19), #8*idx]});
    }

    fn save_mem(&mut self, dst: Reg, idx: u32) {
        self.emit(arm! {str d(ϕ(dst)), [x(19), #8*idx]});
    }

    fn save_mem_result(&mut self, idx: u32) {
        self.save_mem(Reg::Ret, idx);
    }

    fn load_param(&mut self, dst: Reg, idx: u32) {
        self.flush(dst);
        self.emit(arm! {ldr d(ϕ(dst)), [x(20), #8*idx]});
    }

    fn load_stack(&mut self, dst: Reg, idx: u32) {
        self.emit(arm! {ldr d(ϕ(dst)), [sp, #8*idx]});
    }

    fn save_stack(&mut self, dst: Reg, idx: u32) {
        self.emit(arm! {str d(ϕ(dst)), [sp, #8*idx]});
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

    fn square(&mut self, dst: Reg, s1: Reg) {
        self.flush(dst);
        self.times(dst, s1, s1);
    }

    fn cube(&mut self, dst: Reg, s1: Reg) {
        self.flush(dst);
        self.times(Reg::Temp, s1, s1);
        self.times(dst, s1, Reg::Temp);
    }

    fn recip(&mut self, dst: Reg, s1: Reg) {
        self.flush(dst);
        self.emit(arm! {fmov d(TEMP), #1.0});
        self.emit(arm! {fdiv d(ϕ(dst)), d(TEMP), d(ϕ(s1))});
    }

    fn powi(&mut self, dst: Reg, s1: Reg, power: i32) {
        self.flush(dst);

        if power == 0 {
            self.emit(arm! {fmov d(ϕ(dst)), #1.0});
        } else {
            powi(self, dst, s1, power);
        }
    }

    fn powi_mod(&mut self, dst: Reg, s1: Reg, power: i32, modulus: Reg) {
        self.flush(dst);

        if power == 0 {
            self.emit(arm! {fmov d(ϕ(dst)), #1.0});
        } else {
            powi_mod(self, dst, s1, power, modulus);
        }
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

    fn fmod(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        fmod(self, dst, s1, s2);
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

    fn setup_call_unary(&mut self, s1: Reg) {
        setup_call_unary(self, s1);
    }

    fn setup_call_binary(&mut self, s1: Reg, s2: Reg) {
        setup_call_binary(self, s1, s2);
    }

    fn call(&mut self, label: &str, _num_args: usize) {
        self.jump(label, arm! {ldr x(0), label});
        self.emit(arm! {blr x(0)});
    }

    fn select_if(&mut self, dst: Reg, cond: Reg, s1: Reg) {
        self.flush(dst);
        self.and(dst, cond, s1);
    }

    fn select_else(&mut self, dst: Reg, cond: Reg, s1: Reg) {
        self.flush(dst);
        self.andnot(dst, cond, s1);
    }

    fn prologue(&mut self, cap: u32) {
        self.emit(arm! {sub sp, sp, #32});
        self.emit(arm! {str lr, [sp, #0]});
        self.emit(arm! {str x(19), [sp, #8]}); // mem
        self.emit(arm! {str x(20), [sp, #16]}); // param

        let stack_size = align_stack(self.reg_size() * cap);
        self.emit(arm! {sub sp, sp, #stack_size & 0x0fff});
        if stack_size >> 12 != 0 {
            self.emit(arm! {sub sp, sp, #stack_size >> 12, lsl #12});
        }

        self.emit(arm! {mov x(19), x(0)});
        self.emit(arm! {mov x(20), x(1)});
    }

    fn epilogue(&mut self, cap: u32) {
        self.restore_regs();

        let stack_size = align_stack(self.reg_size() * cap);
        if stack_size >> 12 != 0 {
            self.emit(arm! {add sp, sp, #stack_size >> 12, lsl #12});
        }
        self.emit(arm! {add sp, sp, #stack_size & 0x0fff});

        self.emit(arm! {ldr x(20), [sp, #16]});
        self.emit(arm! {ldr x(19), [sp, #8]});
        self.emit(arm! {ldr lr, [sp, #0]});
        self.emit(arm! {add sp, sp, #32});
        self.emit(arm! {ret});
    }

    fn prologue_fast(&mut self, cap: u32, num_args: u32) {
        self.emit(arm! {sub sp, sp, #16});
        self.emit(arm! {str lr, [sp, #0]});
        self.emit(arm! {str x(19), [sp, #8]});

        let stack_size = align_stack(self.reg_size() * cap);
        self.emit(arm! {sub sp, sp, #stack_size & 0x0fff});
        if stack_size >> 12 != 0 {
            self.emit(arm! {sub sp, sp, #stack_size >> 12, lsl #12});
        }

        self.emit(arm! {mov x(19), sp});

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
        if stack_size >> 12 != 0 {
            self.emit(arm! {add sp, sp, #stack_size >> 12, lsl #12});
        }
        self.emit(arm! {add sp, sp, #stack_size & 0x0fff});

        self.emit(arm! {ldr x(19), [sp, #8]});
        self.emit(arm! {ldr lr, [sp, #0]});
        self.emit(arm! {add sp, sp, #16});
        self.emit(arm! {ret});
    }

    fn prologue_indirect(&mut self, cap: u32, count_states: usize, count_obs: usize) {
        self.emit(arm! {sub sp, sp, #48});
        self.emit(arm! {str lr, [sp, #0]});
        self.emit(arm! {str x(19), [sp, #8]}); // mem
        self.emit(arm! {str x(20), [sp, #16]}); // param
        self.emit(arm! {str x(21), [sp, #24]}); // states
        self.emit(arm! {str x(22), [sp, #32]}); // idx

        let mem: u8 = 19; // first arg = mem if direct mode, otherwise null
        let states: u8 = 21; // second arg = states+obs if indirect mode, otherwise null
        let idx: u8 = 22; // third arg = index if indirect mode
        let params: u8 = 20; // fourth arg = params

        self.emit(arm! {mov x(mem), x(0)});
        self.emit(arm! {mov x(states), x(1)});
        self.emit(arm! {mov x(idx), x(2)});
        self.emit(arm! {mov x(params), x(3)});

        self.emit(arm! {tst x(states), x(states)});
        self.jump("@main", arm! {b.eq label});

        let size = align_stack((count_states + count_obs + 1) as u32 * self.reg_size());
        self.emit(arm! {sub sp, sp, #size});
        self.emit(arm! {mov x(mem), sp});

        for i in 0..count_states {
            self.emit(arm! {ldr x(9), [x(states), #8*i]});
            self.emit(arm! {ldr d(0), [x(9), x(idx), lsl #3]});
            self.emit(arm! {str d(0), [x(mem), #8*i]});
        }

        // may save idx (RDX) as double in RBP + 8/32 * count_states

        self.set_label("@main");

        let stack_size = align_stack(self.reg_size() * cap);
        self.emit(arm! {sub sp, sp, #stack_size & 0x0fff});
        if stack_size >> 12 != 0 {
            self.emit(arm! {sub sp, sp, #stack_size >> 12, lsl #12});
        }
    }

    fn epilogue_indirect(&mut self, cap: u32, count_states: usize, count_obs: usize) {
        let mem: u8 = 19; // first arg = mem if direct mode, otherwise null
        let states: u8 = 21; // second arg = states+obs if indirect mode, otherwise null
        let idx: u8 = 22; // third arg = index if indirect mode

        let stack_size = align_stack(self.reg_size() * cap);
        if stack_size >> 12 != 0 {
            self.emit(arm! {add sp, sp, #stack_size >> 12, lsl #12});
        }
        self.emit(arm! {add sp, sp, #stack_size & 0x0fff});

        self.emit(arm! {tst x(states), x(states)});
        self.jump("@done", arm! {b.eq label});

        for i in 0..count_obs {
            self.emit(arm! {ldr x(9), [x(states), #8*(count_states+i)]});
            let k = (count_states + i + 1) as u32;
            self.emit(arm! {ldr d(0), [x(mem), #8*k]});
            self.emit(arm! {str d(0), [x(9), x(idx), lsl #3]});
        }

        let size = align_stack((count_states + count_obs + 1) as u32 * self.reg_size());
        self.emit(arm! {add sp, sp, #size});

        self.set_label("@done");

        self.restore_regs();

        self.emit(arm! {ldr x(22), [sp, #32]});
        self.emit(arm! {ldr x(21), [sp, #24]});
        self.emit(arm! {ldr x(20), [sp, #16]});
        self.emit(arm! {ldr x(19), [sp, #8]});
        self.emit(arm! {ldr lr, [sp, #0]});
        self.emit(arm! {add sp, sp, #48});
        self.emit(arm! {ret});
    }
}

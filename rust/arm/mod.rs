#[macro_use]
mod macros;

use crate::assembler::Assembler;
use crate::generator::{fmod, powi, powi_mod, Generator};
use crate::utils::align_stack;

pub struct ArmGenerator {
    a: Assembler,
    r0: Option<u32>,
    mask: u32,
}

impl ArmGenerator {
    pub fn new() -> ArmGenerator {
        ArmGenerator {
            a: Assembler::new(0, 3),
            r0: None,
            mask: 0x00ff,
        }
    }

    fn emit(&mut self, w: u32) {
        self.a.append_word(w);
    }

    fn flush(&mut self, dst: u8) {
        if dst == 0 {
            if let Some(idx) = self.r0 {
                self.emit(arm! {str d(dst), [sp, #8*idx]});
            };

            self.r0 = None;
        } else {
            let m = 1 << dst;
            let idx = dst as i32;

            if self.mask & m == 0 {
                self.emit(arm! {str d(dst), [sp, #8*idx]});
            }

            self.mask |= m;
        }
    }

    fn restore_regs(&mut self) {
        let last = self.first_shadow() + self.count_shadows();

        for dst in last..16 {
            let m = 1 << dst;
            let idx = dst as i32;

            if self.mask & m != 0 {
                self.emit(arm! {ldr d(dst), [sp, #8*idx]});
            }
        }
    }
}

impl Generator for ArmGenerator {
    fn first_shadow(&self) -> u8 {
        2
    }

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
    fn fmov(&mut self, dst: u8, r: u8) {
        if dst == r {
            return;
        }

        self.flush(dst);
        self.emit(arm! {fmov d(dst), d(r)});
    }

    fn fxchg(&mut self, a: u8, b: u8) {
        self.flush(a);
        self.flush(b);

        self.emit(arm! {eor v(a).8b, v(a).8b, v(b).8b});
        self.emit(arm! {eor v(b).8b, v(a).8b, v(b).8b});
        self.emit(arm! {eor v(a).8b, v(a).8b, v(b).8b});
    }

    fn load_const(&mut self, dst: u8, label: &str) {
        self.flush(dst);
        self.jump(label, arm! {ldr d(dst), label});
    }

    fn load_mem(&mut self, dst: u8, idx: u32) {
        self.flush(dst);
        self.emit(arm! {ldr d(dst), [x(19), #8*idx]});
    }

    fn save_mem(&mut self, src: u8, idx: u32) {
        self.emit(arm! {str d(src), [x(19), #8*idx]});
    }

    fn load_stack(&mut self, dst: u8, idx: u32) {
        if let Some(k) = self.r0 {
            if k == idx {
                self.emit(arm! {fmov d(dst), d(0)});
                self.r0 = None;
                return;
            }
        };
        self.emit(arm! {ldr d(dst), [sp, #8*idx]});
    }

    fn save_stack(&mut self, src: u8, idx: u32) {
        if src == 0 {
            self.r0 = Some(idx);
            return;
        };
        self.emit(arm! {str d(src), [sp, #8*idx]});
    }

    fn neg(&mut self, dst: u8, r: u8) {
        self.flush(dst);
        self.emit(arm! {fneg d(dst), d(r)});
    }

    fn abs(&mut self, dst: u8, r: u8) {
        self.flush(dst);
        self.emit(arm! {fabs d(dst), d(r)});
    }

    fn root(&mut self, dst: u8, r: u8) {
        self.flush(dst);
        self.emit(arm! {fsqrt d(dst), d(r)});
    }

    fn square(&mut self, dst: u8, r: u8) {
        self.flush(dst);
        self.times(dst, r, r);
    }

    fn cube(&mut self, dst: u8, r: u8) {
        self.flush(dst);
        self.times(1, r, r);
        self.times(dst, r, 1);
    }

    fn recip(&mut self, dst: u8, r: u8) {
        self.flush(dst);
        self.emit(arm! {fmov d(1), #1.0});
        self.emit(arm! {fdiv d(dst), d(1), d(r)});
    }

    fn powi(&mut self, dst: u8, r: u8, power: i32) {
        self.flush(dst);

        if power == 0 {
            self.emit(arm! {fmov d(dst), #1.0});
        } else {
            powi(self, dst, r, power);
        }
    }

    fn powi_mod(&mut self, dst: u8, r: u8, power: i32, modulus: u8) {
        self.flush(dst);

        if power == 0 {
            self.emit(arm! {fmov d(dst), #1.0});
        } else {
            powi_mod(self, dst, r, power, modulus);
        }
    }

    fn round(&mut self, dst: u8, r: u8) {
        self.flush(dst);
        self.emit(arm! {frinti d(dst), d(r)});
    }

    fn floor(&mut self, dst: u8, r: u8) {
        self.flush(dst);
        self.emit(arm! {frintm d(dst), d(r)});
    }

    fn ceiling(&mut self, dst: u8, r: u8) {
        self.flush(dst);
        self.emit(arm! {frintp d(dst), d(r)});
    }

    fn trunc(&mut self, dst: u8, r: u8) {
        self.flush(dst);
        self.emit(arm! {frintz d(dst), d(r)});
    }

    fn fmod(&mut self, dst: u8, a: u8, b: u8) {
        fmod(self, dst, a, b);
    }

    fn plus(&mut self, dst: u8, a: u8, b: u8) {
        self.flush(dst);
        self.emit(arm! {fadd d(dst), d(a), d(b)});
    }

    fn minus(&mut self, dst: u8, a: u8, b: u8) {
        self.flush(dst);
        self.emit(arm! {fsub d(dst), d(a), d(b)});
    }

    fn times(&mut self, dst: u8, a: u8, b: u8) {
        self.flush(dst);
        self.emit(arm! {fmul d(dst), d(a), d(b)});
    }

    fn divide(&mut self, dst: u8, a: u8, b: u8) {
        self.flush(dst);
        self.emit(arm! {fdiv d(dst), d(a), d(b)});
    }

    fn gt(&mut self, dst: u8, a: u8, b: u8) {
        self.flush(dst);
        self.emit(arm! {fcmgt d(dst), d(a), d(b)});
    }

    fn geq(&mut self, dst: u8, a: u8, b: u8) {
        self.flush(dst);
        self.emit(arm! {fcmge d(dst), d(a), d(b)});
    }

    fn lt(&mut self, dst: u8, a: u8, b: u8) {
        self.flush(dst);
        self.emit(arm! {fcmlt d(dst), d(a), d(b)});
    }

    fn leq(&mut self, dst: u8, a: u8, b: u8) {
        self.flush(dst);
        self.emit(arm! {fcmle d(dst), d(a), d(b)});
    }

    fn eq(&mut self, dst: u8, a: u8, b: u8) {
        self.flush(dst);
        self.emit(arm! {fcmeq d(dst), d(a), d(b)});
    }

    fn neq(&mut self, dst: u8, a: u8, b: u8) {
        self.flush(dst);
        self.emit(arm! {fcmeq d(dst), d(a), d(b)});
        self.emit(arm! {not v(dst).8b, v(dst).8b});
    }

    fn and(&mut self, dst: u8, a: u8, b: u8) {
        self.flush(dst);
        self.emit(arm! {and v(dst).8b, v(a).8b, v(b).8b});
    }

    fn andnot(&mut self, dst: u8, a: u8, b: u8) {
        self.flush(dst);
        self.emit(arm! {not v(a).8b, v(a).8b});
        self.emit(arm! {and v(dst).8b, v(a).8b, v(b).8b});
    }

    fn or(&mut self, dst: u8, a: u8, b: u8) {
        self.flush(dst);
        self.emit(arm! {orr v(dst).8b, v(a).8b, v(b).8b});
    }

    fn xor(&mut self, dst: u8, a: u8, b: u8) {
        self.flush(dst);
        self.emit(arm! {eor v(dst).8b, v(a).8b, v(b).8b});
    }

    fn not(&mut self, dst: u8, r: u8) {
        self.flush(dst);
        self.emit(arm! {not v(dst).8b, v(r).8b});
    }

    fn call(&mut self, label: &str, num_args: usize) {
        self.jump(label, arm! {ldr x(0), label});
        self.emit(arm! {blr x(0)});
    }

    fn select_if(&mut self, dst: u8, cond: u8, a: u8) {
        self.flush(dst);
        self.and(dst, cond, a);
    }

    fn select_else(&mut self, dst: u8, cond: u8, a: u8) {
        self.flush(dst);
        self.andnot(dst, cond, a);
    }

    fn prologue(&mut self, cap: u32) {
        self.emit(arm! {sub sp, sp, #16});
        self.emit(arm! {str lr, [sp, #0]});
        self.emit(arm! {str x(19), [sp, #8]});

        let stack_size = align_stack(self.reg_size() * cap);
        self.emit(arm! {sub sp, sp, #stack_size & 0x0fff});
        if stack_size >> 12 != 0 {
            self.emit(arm! {sub sp, sp, #stack_size >> 12, lsl #12});
        }

        self.emit(arm! {mov x(19), x(0)});
    }

    fn epilogue(&mut self, cap: u32) {
        self.restore_regs();

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
}

#[macro_use]
mod macros;

use crate::assembler::Assembler;
use crate::code::BinaryFunc;
use crate::generator::Generator;
use crate::utils::align_stack;

pub struct ArmGenerator {
    a: Assembler,
    r0: Option<u32>,
}

impl ArmGenerator {
    pub fn new() -> ArmGenerator {
        ArmGenerator {
            a: Assembler::new(0, 3),
            r0: None,
        }
    }

    fn emit(&mut self, w: u32) {
        self.a.append_word(w);
    }

    fn flush(&mut self, dst: u8) {
        if dst != 0 {
            return;
        }

        if let Some(idx) = self.r0 {
            self.emit(arm! {str d(dst), [sp, #8*idx]});
        };

        self.r0 = None;
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
        self.emit(arm! {fmul d(dst), d(r), d(r)});
    }

    fn cube(&mut self, dst: u8, r: u8) {
        self.flush(dst);
        self.emit(arm! {fmul d(1), d(r), d(r)});
        self.emit(arm! {fmul d(dst), d(r), d(1)});
    }

    fn recip(&mut self, dst: u8, r: u8) {
        self.flush(dst);
        self.emit(arm! {fmov d(1), #1.0});
        self.emit(arm! {fdiv d(dst), d(1), d(r)});
    }
    
    fn powi(&mut self, dst: u8, r: u8, n: i32) {
        if n == 0 {
            self.emit(arm! {fmov d(dst), #1.0});
        } else if n > 0 {
            let t = n.trailing_zeros();
            
            if n.count_ones() == 1 {
                self.fmov(dst, r);        
            } else {
                self.fmov(1, r);        
                let mut n = n >> (t+1);            
            
                while n > 0 {
                    self.times(1, 1, 1);
                    if n & 1 != 0 {
                        self.times(dst, dst, 1);
                    };
                    n = n >> 1;
                }
            }
            
            for _ in 0..t {
                self.times(dst, dst, dst);
            }
        } else {
            self.powi(dst, r, -n);
            self.recip(dst, dst);
        }
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
        self.emit(arm! {not v(dst).8b, v(a).8b});
        self.emit(arm! {and v(dst).8b, v(dst).8b, v(b).8b});
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
        let stack_size = align_stack(self.reg_size() * cap);

        self.emit(arm! {sub sp, sp, #16});
        self.emit(arm! {str lr, [sp, #0]});
        self.emit(arm! {str x(19), [sp, #8]});
        self.emit(arm! {sub sp, sp, #stack_size});
        self.emit(arm! {mov x(19), x(0)});
    }

    fn epilogue(&mut self, cap: u32) {
        let stack_size = align_stack(self.reg_size() * cap);

        self.emit(arm! {add sp, sp, #stack_size});
        self.emit(arm! {ldr x(19), [sp, #8]});
        self.emit(arm! {ldr lr, [sp, #0]});
        self.emit(arm! {add sp, sp, #16});
        self.emit(arm! {ret});
    }
}

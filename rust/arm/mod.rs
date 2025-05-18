#[macro_use]
mod macros;

use crate::assembler::Assembler;
use crate::code::BinaryFunc;
use crate::generator::Generator;

pub struct ArmGenerator {
    a: Assembler,
}

impl ArmGenerator {
    pub fn new() -> ArmGenerator {
        ArmGenerator {
            a: Assembler::new(0, 3)
        }
    }
    
    fn emit(&mut self, w: u32) {
        self.a.append_word(w);
    }    
}

impl Generator for ArmGenerator {
    fn first_shadow(&self) -> u8 {
        return 2;
    }

    fn count_shadows(&self) -> u8 {
        return 6;
    }

    fn reg_size(&self) -> u32 {
        return 8;
    }

    fn a(&mut self) -> &mut Assembler {
        &mut self.a
    }

    //***********************************
    fn fmov(&mut self, dst: u8, r: u8) {
        self.emit(arm! {fmov d(dst), d(r)});        
    }

    fn fxchg(&mut self, a: u8, b: u8) {
        self.emit(arm! {eor v(a).8b, v(a).8b, v(b).8b});
        self.emit(arm! {eor v(b).8b, v(a).8b, v(b).8b});
        self.emit(arm! {eor v(a).8b, v(a).8b, v(b).8b});
    }

    fn load_const(&mut self, dst: u8, label: &str) {
        self.jump(label, arm! {ldr d(dst), label});
    }

    fn load_mem(&mut self, dst: u8, idx: u32) {
        self.emit(arm! {ldr d(dst), [x(19), #8*idx]});
    }

    fn save_mem(&mut self, src: u8, idx: u32) {
        self.emit(arm! {str d(src), [x(19), #8*idx]});
    }

    fn load_stack(&mut self, dst: u8, idx: u32) {
        self.emit(arm! {ldr d(dst), [sp, #8*idx]});
    }

    fn save_stack(&mut self, src: u8, idx: u32) {
        self.emit(arm! {str d(src), [sp, #8*idx]});
    }

    fn neg(&mut self, dst: u8) {
        self.emit(arm! {fneg d(dst), d(dst)});
    }

    fn abs(&mut self, dst: u8) {
        self.emit(arm! {fabs d(dst), d(dst)});
    }

    fn root(&mut self, dst: u8) {
        self.emit(arm! {fsqrt d(dst), d(dst)});
    }

    fn square(&mut self, dst: u8) {
        self.emit(arm! {fmul d(dst), d(dst), d(dst)});
    }

    fn cube(&mut self, dst: u8) {
        self.emit(arm! {fmul d(1), d(dst), d(dst)});
        self.emit(arm! {fmul d(dst), d(dst), d(1)});
    }

    fn recip(&mut self, dst: u8) {
        self.emit(arm! {fmov d(1), #1.0});
        self.emit(arm! {fdiv d(dst), d(1), d(dst)});
    }

    fn plus(&mut self, dst: u8, a: u8, b: u8) {
        self.emit(arm! {fadd d(dst), d(a), d(b)});
    }

    fn minus(&mut self, dst: u8, a: u8, b: u8) {
        self.emit(arm! {fsub d(dst), d(a), d(b)});
    }

    fn times(&mut self, dst: u8, a: u8, b: u8) {
        self.emit(arm! {fmul d(dst), d(a), d(b)});
    }

    fn divide(&mut self, dst: u8, a: u8, b: u8) {
        self.emit(arm! {fdiv d(dst), d(a), d(b)});
    }

    fn gt(&mut self, dst: u8, a: u8, b: u8) {
        self.emit(arm! {fcmgt d(dst), d(a), d(b)});
    }

    fn geq(&mut self, dst: u8, a: u8, b: u8) {
        self.emit(arm! {fcmge d(dst), d(a), d(b)});
    }

    fn lt(&mut self, dst: u8, a: u8, b: u8) {
        self.emit(arm! {fcmlt d(dst), d(a), d(b)});
    }

    fn leq(&mut self, dst: u8, a: u8, b: u8) {
        self.emit(arm! {fcmle d(dst), d(a), d(b)});
    }

    fn eq(&mut self, dst: u8, a: u8, b: u8) {
        self.emit(arm! {fcmeq d(dst), d(a), d(b)});
    }

    fn neq(&mut self, dst: u8, a: u8, b: u8) {
        self.emit(arm! {fcmeq d(dst), d(a), d(b)});
        self.emit(arm! {not v(dst).8b, v(dst).8b});
    }

    fn and(&mut self, dst: u8, a: u8, b: u8) {
        self.emit(arm! {and v(dst).8b, v(a).8b, v(b).8b});
    }

    fn andnot(&mut self, dst: u8, a: u8, b: u8) {
        self.emit(arm! {not v(dst).8b, v(a).8b});
        self.emit(arm! {and v(dst).8b, v(dst).8b, v(b).8b});
    }

    fn or(&mut self, dst: u8, a: u8, b: u8) {
        self.emit(arm! {orr v(dst).8b, v(a).8b, v(b).8b});
    }

    fn xor(&mut self, dst: u8, a: u8, b: u8) {
        self.emit(arm! {eor v(dst).8b, v(a).8b, v(b).8b});
    }

    fn not(&mut self, dst: u8) {
        self.emit(arm! {not v(dst).8b, v(dst).8b});
    }

    fn call(&mut self, label: &str, num_args: usize) {
        self.jump(label, arm! {ldr x(0), label});
        self.emit(arm! {blr x(0)});   
    }

    fn select_if(&mut self, dst: u8, cond: u8, a: u8) {
        self.and(dst, cond, a);
    }

    fn select_else(&mut self, dst: u8, cond: u8, a: u8) {
        self.andnot(dst, cond, a);
    }
    
    fn prologue(&mut self, n: u32) {
        self.emit(arm! {sub sp, sp, #16});
        self.emit(arm! {str lr, [sp, #0]});
        self.emit(arm! {str x(19), [sp, #8]});
        self.emit(arm! {sub sp, sp, #8*n});
        self.emit(arm! {mov x(19), x(0)});        
    }

    fn epilogue(&mut self, n: u32) {
        self.emit(arm! {add sp, sp, #8*n});
        self.emit(arm! {ldr x(19), [sp, #8]});
        self.emit(arm! {ldr lr, [sp, #0]});
        self.emit(arm! {add sp, sp, #16});
        self.emit(arm! {ret});
    }
}

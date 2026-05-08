use anyhow::Result;

use crate::assembler::{Assembler, Jumper};
use crate::code::Func;
use crate::config::{Config, SPILL_AREA};
use crate::generator::{FuncletType, Generator};
use crate::utils::{align_stack, is_external_func, reg, Reg};

use super::*;

const REG_SIZE: u32 = 8;
const T0: u8 = 24; // Reg::Temp
const T1: u8 = 25;
const T2: u8 = 26;

pub struct ArmComplexGenerator {
    a: Assembler,
    config: Config,
}

impl ArmComplexGenerator {
    pub fn new(config: Config) -> ArmComplexGenerator {
        ArmComplexGenerator {
            a: Assembler::new(),
            config,
        }
    }

    pub fn jump(&mut self, label: &str, code: u32, f: Jumper) {
        self.a.jump(label, code, f)
    }

    pub fn jump_abs(&mut self, label: &str, code: u32, f: Jumper) {
        self.a.jump_abs(label, code, f);
    }

    pub fn ip(&self) -> usize {
        self.a.ip()
    }

    fn apply_jumps(&mut self) {
        self.a.apply_jumps();
    }

    fn emit(&mut self, w: u32) {
        self.a.append_word(w);
    }

    fn load_d_from_mem(&mut self, d: u8, base: u8, idx: u32) {
        load_d_from_mem(&mut self.a, d, base, idx);
    }

    fn save_d_to_mem(&mut self, d: u8, base: u8, idx: u32) {
        save_d_to_mem(&mut self.a, d, base, idx);
    }

    fn load_q_from_mem(&mut self, d: u8, base: u8, idx: u32) {
        load_q_from_mem(&mut self.a, d, base, idx / 2);
    }

    fn save_q_to_mem(&mut self, d: u8, base: u8, idx: u32) {
        save_q_to_mem(&mut self.a, d, base, idx / 2);
    }

    fn load_x_from_mem(&mut self, r: u8, base: u8, idx: u32) {
        load_x_from_mem(&mut self.a, r, base, idx);
    }

    fn load_x_from_label(&mut self, dst: u8, label: &str) {
        load_x_from_label(&mut self.a, dst, label);
    }

    fn sub_stack(&mut self, size: u32) {
        sub_stack(&mut self.a, size);
    }

    fn add_stack(&mut self, size: u32) {
        add_stack(&mut self.a, size);
    }

    fn call_external(&mut self, op: &str, num_args: usize) -> Result<()> {
        self.load_x_from_label(0, &format!("_env_{}_", op));
        let ofs = SPILL_AREA as u32 * REG_SIZE;
        self.emit(arm! {add x(1), x(31), #ofs});
        self.emit(arm! {movz x(2), #num_args});
        self.emit(arm! {add x(3), x(SP), #0});

        let label = format!("_func_{}_", op);
        load_long(&mut self.a, 9, &label);
        self.emit(arm! {blr x(9)});

        self.load_stack(Reg::Ret, 0);

        Ok(())
    }
}

impl Generator for ArmComplexGenerator {
    fn bytes(&mut self) -> Vec<u8> {
        self.a.bytes()
    }

    fn three_address(&self) -> bool {
        true
    }

    fn count_shadows(&self) -> u8 {
        14
    }

    fn support_funclet(&self) -> FuncletType {
        FuncletType::None
    }

    fn seal(&mut self) {
        self.apply_jumps();
    }

    fn align(&mut self) {
        if self.a.ip() & 7 != 0 {
            self.emit(arm! {nop});
        }
    }

    fn set_label(&mut self, label: &str) {
        self.a.set_label(label);
    }

    fn branch(&mut self, label: &str) {
        self.jump(label, 0, |offset, _| arm! {b label(offset)});
    }

    fn branch_if(&mut self, cond: Reg, label: &str, is_else: bool) {
        self.emit(arm! {fcmp d(ϕ(cond)), #0.0});

        let l = self.a.create_label();

        if is_else {
            self.jump(&l, 0, |offset, _| arm! {b.ne label(offset)});
        } else {
            self.jump(&l, 0, |offset, _| arm! {b.eq label(offset)});
        }

        self.branch(label);
        self.set_label(&l);
    }

    fn fuse_load_math(&mut self) {}

    //***********************************

    fn fmov(&mut self, dst: Reg, s1: Reg) {
        if dst == s1 {
            return;
        }

        self.emit(arm! {fmov q(ϕ(dst)), q(ϕ(s1))});
    }

    fn fxchg(&mut self, s1: Reg, s2: Reg) {
        self.emit(arm! {eor v(ϕ(s1)).16b, v(ϕ(s1)).16b, v(ϕ(s2)).16b});
        self.emit(arm! {eor v(ϕ(s2)).16b, v(ϕ(s1)).16b, v(ϕ(s2)).16b});
        self.emit(arm! {eor v(ϕ(s1)).16b, v(ϕ(s1)).16b, v(ϕ(s2)).16b});
    }

    fn load_const(&mut self, dst: Reg, idx: u32) {
        self.xor(dst, dst, dst);
        let label = format!("_const_{}_", idx);
        self.jump_abs(&label, (self.ip() & 0xfffff000) as u32, |offset, pg| {
            arm! {adrp x(0), label((offset - pg as i32) as u32)}
        });

        self.jump_abs(
            &label,
            ϕ(dst) as u32,
            |offset, dst| arm! {ldr d(dst), [x(0), #offset & 0x0fff]},
        );
    }

    fn load_mem(&mut self, dst: Reg, idx: u32) {
        self.load_q_from_mem(ϕ(dst), MEM, idx);
    }

    fn save_mem(&mut self, dst: Reg, idx: u32) {
        self.save_q_to_mem(ϕ(dst), MEM, idx);
    }

    fn save_mem_result(&mut self, idx: u32) {
        self.save_mem(Reg::Ret, idx);
    }

    fn load_param(&mut self, dst: Reg, idx: u32) {
        self.load_q_from_mem(ϕ(dst), PARAMS, idx);
    }

    fn load_stack(&mut self, dst: Reg, idx: u32) {
        self.load_q_from_mem(ϕ(dst), SP, idx);
    }

    fn save_stack(&mut self, dst: Reg, idx: u32) {
        self.save_q_to_mem(ϕ(dst), SP, idx);
    }

    fn load_mem_complex(&mut self, xd: Reg, yd: Reg, idx: u32) {}

    fn save_mem_complex(&mut self, xs: Reg, ys: Reg, idx: u32) {}

    fn load_param_complex(&mut self, xd: Reg, yd: Reg, idx: u32) {}

    fn load_stack_complex(&mut self, xd: Reg, yd: Reg, idx: u32) {}

    fn save_stack_complex(&mut self, xs: Reg, ys: Reg, idx: u32) {}

    fn save_stack_result(&mut self, idx: u32) {
        self.save_stack(Reg::Ret, idx);
    }

    fn neg(&mut self, dst: Reg, s1: Reg) {
        self.emit(arm! {fneg q(ϕ(dst)), q(ϕ(s1))});
    }

    fn abs(&mut self, dst: Reg, s1: Reg) {
        self.emit(arm! {fmul q(T2), q(ϕ(s1)), q(ϕ(s1))});
        self.emit(arm! {eor v(ϕ(dst)).16b, v(ϕ(dst)).16b, v(ϕ(dst)).16b});
        self.emit(arm! {faddp d(ϕ(dst)), q(T2)});
        self.emit(arm! {fsqrt d(ϕ(dst)), d(ϕ(dst))});
    }

    fn root(&mut self, dst: Reg, s1: Reg) {
        self.emit(arm! {fmov x(0), d(ϕ(s1))});

        self.emit(arm! {fmul q(T1), q(ϕ(s1)), q(ϕ(s1))});
        self.emit(arm! {faddp d(T1), q(T1)});
        self.emit(arm! {fsqrt d(T1), d(T1)});
        self.emit(arm! {fabs d(T2), d(ϕ(s1))});
        self.emit(arm! {fadd d(T1), d(T1), d(T2)});
        self.emit(arm! {fmov d(T0), #2.0});
        self.emit(arm! {fdiv d(T1), d(T1), d(T0)});
        self.emit(arm! {fsqrt d(T1), d(T1)});

        self.emit(arm! {zip2 q(T2), q(ϕ(s1)), q(ϕ(s1))});
        self.emit(arm! {fdiv d(T2), d(T2), d(T1)});
        self.emit(arm! {fdiv d(T2), d(T2), d(T0)});

        self.emit(arm! {fcmeq d(T0), d(T2), d(T2)});
        self.emit(arm! {and v(T2).8b, v(T2).8b, v(T0).8b});

        self.emit(arm! {zip1 q(ϕ(dst)), q(T2), q(T1)});
        let label = self.a.create_label();
        self.emit(arm! {tst x(0), x(0)});
        self.jump(&label, 0, |offset, _| arm! {b.mi label(offset)});
        self.emit(arm! {zip1 q(ϕ(dst)), q(T1), q(T2)});
        self.set_label(&label);
    }

    fn real_root(&mut self, dst: Reg, s1: Reg) {
        self.emit(arm! {fsqrt q(ϕ(dst)), q(ϕ(s1))});
    }

    fn recip(&mut self, dst: Reg, s1: Reg) {
        self.emit(arm! {zip2 q(T1), q(ϕ(s1)), q(ϕ(s1))});
        self.emit(arm! {fneg q(T1), q(T1)});
        self.emit(arm! {zip1 q(T1), q(ϕ(s1)), q(T1)});

        self.emit(arm! {fmul q(T2), q(ϕ(s1)), q(ϕ(s1))});
        self.emit(arm! {faddp d(T2), q(T2)});
        self.emit(arm! {dup q(T2), q(T2)[0]});
        self.emit(arm! {fdiv q(ϕ(dst)), q(T1), q(T2)});
    }

    fn half(&mut self, dst: Reg, s1: Reg) {
        self.emit(arm! {fmov d(TEMP), #2.0});
        self.emit(arm! {dup q(TEMP), q(TEMP)[0]});
        self.emit(arm! {fdiv q(ϕ(dst)), q(ϕ(s1)), q(TEMP)});
    }

    fn round(&mut self, dst: Reg, s1: Reg) {
        self.emit(arm! {frinti q(ϕ(dst)), q(ϕ(s1))});
    }

    fn floor(&mut self, dst: Reg, s1: Reg) {
        self.emit(arm! {frintm q(ϕ(dst)), q(ϕ(s1))});
    }

    fn ceiling(&mut self, dst: Reg, s1: Reg) {
        self.emit(arm! {frintp q(ϕ(dst)), q(ϕ(s1))});
    }

    fn trunc(&mut self, dst: Reg, s1: Reg) {
        self.emit(arm! {frintz q(ϕ(dst)), q(ϕ(s1))});
    }

    fn frac(&mut self, dst: Reg, s1: Reg) {
        self.floor(Reg::Temp, s1);
        self.minus(dst, s1, Reg::Temp);
    }

    fn plus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(arm! {fadd q(ϕ(dst)), q(ϕ(s1)), q(ϕ(s2))});
    }

    fn minus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(arm! {fsub q(ϕ(dst)), q(ϕ(s1)), q(ϕ(s2))});
    }

    fn times(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        /*
        self.emit(arm! {eor v(T1).16b, v(T1).16b, v(T1).16b});
        self.emit(arm! {fcmla q(T1), q(ϕ(s1)), q(ϕ(s2)), #0});
        self.emit(arm! {fcmla q(T1), q(ϕ(s1)), q(ϕ(s2)), #90});
        self.emit(arm! {fmov q(ϕ(dst)), q(T1)});
        */

        self.emit(arm! { fmul d(T0), d(ϕ(s1)), d(ϕ(s2)) }); // T0 = x1*x2
        self.emit(arm! { dup q(T1), q(ϕ(s1))[1] }); // T1 = y1
        self.emit(arm! { dup q(T2), q(ϕ(s2))[1] }); // T2 = y2
        self.emit(arm! { fmsub d(T0), d(T1), d(T2), d(T0) }); // T0 = x1*x2 - y1*y2
        self.emit(arm! { fmul d(T2), d(ϕ(s1)), d(T2) }); // T2 = x1*y2
        self.emit(arm! { fmadd d(T2), d(ϕ(s2)), d(T1), d(T2) }); // T2 = x1*y2 + x2*y1

        self.emit(arm! { zip1 q(ϕ(dst)), q(T0), q(T2) }); // dst = (x1*x2 - y1*y2) + (x1*y2 + x2*y1)*im
    }

    fn divide(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        /*
        self.emit(arm! {eor v(T1).16b, v(T1).16b, v(T1).16b});
        self.emit(arm! {fcmla q(T1), q(ϕ(s2)), q(ϕ(s1)), #0});
        self.emit(arm! {fcmla q(T1), q(ϕ(s2)), q(ϕ(s1)), #270});

        self.emit(arm! {fmul q(T2), q(ϕ(s2)), q(ϕ(s2))});
        self.emit(arm! {faddp d(T2), q(T2)});
        self.emit(arm! {dup q(T2), q(T2)[0]});
        self.emit(arm! {fdiv q(ϕ(dst)), q(T1), q(T2)});
        */

        self.emit(arm! { fmul d(T0), d(ϕ(s1)), d(ϕ(s2)) }); // T0 = x1*x2
        self.emit(arm! { dup q(T1), q(ϕ(s1))[1] }); // T1 = y1
        self.emit(arm! { dup q(T2), q(ϕ(s2))[1] }); // T2 = y2
        self.emit(arm! { fmadd d(T0), d(T1), d(T2), d(T0) }); // T0 = x1*x2 + y1*y2
        self.emit(arm! { fmul d(T1), d(ϕ(s2)), d(T1) }); // T1 = x2*y1
        self.emit(arm! { fmsub d(T1), d(ϕ(s1)), d(T2), d(T1) }); // T1 = x2*y1 - x1*y2

        self.emit(arm! { zip1 q(T0), q(T0), q(T1) }); // T0 = (x1*x2 + y1*y2) + (x2*y1 - x1*y2)*im

        self.emit(arm! { fmul d(T1), d(ϕ(s2)), d(ϕ(s2)) }); // T1 = x2*x2
        self.emit(arm! { fmadd d(T1), d(T2), d(T2), d(T1) }); // T1 = x2*x2 + y2*y2
        self.emit(arm! { dup q(T1), q(T1)[0] });
        self.emit(arm! { fdiv q(ϕ(dst)), q(T0), q(T1) }); // T0 = (x1*x2 + y1*y2)/T1 + (x2*y1 - x1*y2)/T1*im
    }

    fn times_complex(&mut self, xd: Reg, yd: Reg, x1: Reg, y1: Reg, x2: Reg, y2: Reg) -> bool {
        false
    }

    fn divide_complex(&mut self, xd: Reg, yd: Reg, x1: Reg, y1: Reg, x2: Reg, y2: Reg) -> bool {
        false
    }

    fn real(&mut self, dst: Reg, s1: Reg) {
        self.emit(arm! {eor v(T1).16b, v(T1).16b, v(T1).16b});
        self.emit(arm! {zip1 q(ϕ(dst)), q(ϕ(s1)), q(T1)});
    }

    fn imaginary(&mut self, dst: Reg, s1: Reg) {
        self.emit(arm! {eor v(T1).16b, v(T1).16b, v(T1).16b});
        self.emit(arm! {zip2 q(ϕ(dst)), q(ϕ(s1)), q(T1)});
    }

    fn conjugate(&mut self, dst: Reg, s1: Reg) {
        self.emit(arm! {zip2 q(T1), q(ϕ(s1)), q(ϕ(s1))});
        self.emit(arm! {fneg q(T1), q(T1)});
        self.emit(arm! {zip1 q(ϕ(dst)), q(ϕ(s1)), q(T1)});
    }

    fn complex(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(arm! {zip1 q(ϕ(dst)), q(ϕ(s1)), q(ϕ(s2))});
    }

    fn gt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(arm! {fcmgt d(ϕ(dst)), d(ϕ(s1)), d(ϕ(s2))});
        self.emit(arm! {dup q(ϕ(dst)), q(ϕ(dst))[0]});
    }

    fn geq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(arm! {fcmge d(ϕ(dst)), d(ϕ(s1)), d(ϕ(s2))});
        self.emit(arm! {dup q(ϕ(dst)), q(ϕ(dst))[0]});
    }

    fn lt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(arm! {fcmlt d(ϕ(dst)), d(ϕ(s1)), d(ϕ(s2))});
        self.emit(arm! {dup q(ϕ(dst)), q(ϕ(dst))[0]});
    }

    fn leq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(arm! {fcmle d(ϕ(dst)), d(ϕ(s1)), d(ϕ(s2))});
        self.emit(arm! {dup q(ϕ(dst)), q(ϕ(dst))[0]});
    }

    fn eq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(arm! {fcmeq d(ϕ(dst)), d(ϕ(s1)), d(ϕ(s2))});
        self.emit(arm! {dup q(ϕ(dst)), q(ϕ(dst))[0]});
    }

    fn neq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(arm! {fcmeq d(ϕ(dst)), d(ϕ(s1)), d(ϕ(s2))});
        self.emit(arm! {not v(ϕ(dst)).8b, v(ϕ(dst)).8b});
        self.emit(arm! {dup q(ϕ(dst)), q(ϕ(dst))[0]});
    }

    fn and(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(arm! {and v(ϕ(dst)).16b, v(ϕ(s1)).16b, v(ϕ(s2)).16b});
    }

    fn andnot(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(arm! {bic v(ϕ(dst)).16b, v(ϕ(s1)).16b, v(ϕ(s2)).16b});
    }

    fn or(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(arm! {orr v(ϕ(dst)).16b, v(ϕ(s1)).16b, v(ϕ(s2)).16b});
    }

    fn xor(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(arm! {eor v(ϕ(dst)).16b, v(ϕ(s1)).16b, v(ϕ(s2)).16b});
    }

    fn not(&mut self, dst: Reg, s1: Reg) {
        self.emit(arm! {not v(ϕ(dst)).16b, v(ϕ(s1)).16b});
    }

    fn fused_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        self.fmov(Reg::Temp, s3);
        self.emit(arm! {fmla q(ϕ(Reg::Temp)), q(ϕ(s1)), q(ϕ(s2))});
        self.fmov(dst, Reg::Temp);
    }

    // fused_mul_sub is s1 * s2 - s3, corresponding to fnmsub in aarch64
    // and vmsub... in amd64
    fn fused_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        self.fmov(Reg::Temp, s3);
        self.emit(arm! {fmls q(ϕ(Reg::Temp)), q(ϕ(s1)), q(ϕ(s2))});
        self.neg(dst, Reg::Temp);
    }

    // fused_neg_mul_add is s3 - s1 * s2, corresponding to fmsub in aarch64
    // and vnmadd... in amd64
    fn fused_neg_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        self.fmov(Reg::Temp, s3);
        self.emit(arm! {fmls q(ϕ(Reg::Temp)), q(ϕ(s1)), q(ϕ(s2))});
        self.fmov(dst, Reg::Temp);
    }

    // fused_neg_mul_sub is -s3 - s1 * s2, corresponding to fnmadd in aarch64
    // and vnmsub... in amd64
    fn fused_neg_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        self.fmov(Reg::Temp, s3);
        self.emit(arm! {fmla q(ϕ(Reg::Temp)), q(ϕ(s1)), q(ϕ(s2))});
        self.neg(dst, Reg::Temp);
    }

    fn add_consts(&mut self, consts: &[f64]) {
        self.align();
        add_consts(&mut self.a, consts);
    }

    fn add_func(&mut self, op: &str, f: Func) {
        add_func(&mut self.a, op, f);
    }

    fn call(&mut self, op: &str, num_args: usize) -> Result<()> {
        if is_external_func(op) {
            return self.call_external(op, num_args);
        }

        let label = format!("_func_{}_", op);
        load_long(&mut self.a, 9, &label);
        self.emit(arm! {blr x(9)});

        Ok(())
    }

    fn call_complex(&mut self, op: &str, num_args: usize) -> Result<()> {
        self.emit(arm! {add x(0), x(SP), #0});

        if num_args == 2 {
            self.save_stack(Reg::Right, 0);
        }

        self.emit(arm! {dup q(1), q(ϕ(Reg::Left))[1]});

        self.call(op, num_args)?;

        self.load_stack(Reg::Ret, 0);
        Ok(())
    }

    fn call_funclet(&mut self, _label: &str) {
        todo!();
    }

    fn ret(&mut self) {
        self.emit(arm! {ret});
    }

    fn ifelse(&mut self, dst: Reg, true_val: Reg, false_val: Reg, idx: u32) {
        if true_val == false_val {
            self.fmov(dst, true_val);
        } else if dst != true_val && dst != false_val {
            self.load_stack(dst, idx);
            self.emit(arm! {bsl v(ϕ(dst)).8b, v(ϕ(true_val)).8b, v(ϕ(false_val)).8b});
        } else {
            self.load_stack(Reg::Temp, idx);
            self.emit(arm! {bsl v(ϕ(Reg::Temp)).8b, v(ϕ(true_val)).8b, v(ϕ(false_val)).8b});
            self.fmov(dst, Reg::Temp);
        }
    }

    /**************************************************/

    fn prologue_fast(&mut self, cap: usize, count_states: usize, count_obs: usize) {
        self.emit(arm! {sub sp, sp, #16});
        self.emit(arm! {str lr, [sp, #0]});
        self.emit(arm! {str x(MEM), [sp, #8]});

        let frame_size = align_stack((count_states + count_obs) as u32 * REG_SIZE);
        self.sub_stack(frame_size);
        self.emit(arm! {mov x(MEM), sp});
        let stack_size = align_stack(cap as u32 * REG_SIZE);
        self.sub_stack(stack_size);

        for i in 0..count_states {
            self.emit(arm! {str d(i), [x(MEM), #8*i]});
        }
    }

    fn epilogue_fast(&mut self, cap: usize, count_states: usize, count_obs: usize, idx_ret: i32) {
        self.emit(arm! {ldr d(0), [x(MEM), #8*idx_ret]});

        let total_size = align_stack(cap as u32 * REG_SIZE)
            + align_stack((count_states + count_obs) as u32 * REG_SIZE);
        self.add_stack(total_size);

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
    fn prologue_indirect(
        &mut self,
        cap: usize,
        count_states: usize,
        count_obs: usize,
        _count_params: usize,
    ) {
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
        self.jump("@main", 0, |offset, _| arm! {b.eq label(offset)});

        let frame_size = align_stack((count_states + count_obs) as u32 * REG_SIZE);
        self.sub_stack(frame_size);
        self.emit(arm! {mov x(MEM), sp});

        for i in 0..count_states {
            self.load_x_from_mem(SCRATCH2, STATES, 2 * i as u32);
            self.emit(arm! {ldr d(0), [x(SCRATCH2), x(IDX), lsl #3]});
            self.save_d_to_mem(0, MEM, i as u32);
        }

        // TODO: may save idx (RDX) as double in RBP + 8/32 * count_states

        self.set_label("@main");

        let stack_size = align_stack(cap as u32 * REG_SIZE);
        self.sub_stack(stack_size);
    }

    fn epilogue_indirect(
        &mut self,
        cap: usize,
        count_states: usize,
        count_obs: usize,
        _count_params: usize,
    ) {
        let stack_size = align_stack(cap as u32 * REG_SIZE);
        self.add_stack(stack_size);

        self.emit(arm! {tst x(STATES), x(STATES)});
        self.jump("@done", 0, |offset, _| arm! {b.eq label(offset)});

        for i in 0..count_obs {
            self.load_x_from_mem(SCRATCH2, STATES, 2 * (count_states + i) as u32);
            let k = (count_states + i) as u32;
            self.load_d_from_mem(0, MEM, k);
            self.emit(arm! {str d(0), [x(SCRATCH2), x(IDX), lsl #3]});
        }

        let frame_size = align_stack((count_states + count_obs) as u32 * REG_SIZE);
        self.add_stack(frame_size);

        self.set_label("@done");

        self.emit(arm! {ldr x(IDX), [sp, #32]});
        self.emit(arm! {ldr x(STATES), [sp, #24]});
        self.emit(arm! {ldr x(PARAMS), [sp, #16]});
        self.emit(arm! {ldr x(MEM), [sp, #8]});
        self.emit(arm! {ldr lr, [sp, #0]});
        self.emit(arm! {add sp, sp, #48});
        self.emit(arm! {ret});
    }

    fn save_used_registers(&mut self, used: &[u8]) {
        for r in used {
            if *r >= 22 {
                self.save_stack(reg(*r), *r as u32 - 14);
            }
        }
    }

    fn load_used_registers(&mut self, used: &[u8]) {
        for r in used {
            if *r >= 22 {
                self.load_stack(reg(*r), *r as u32 - 14);
            }
        }
    }
}

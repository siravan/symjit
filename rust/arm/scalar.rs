use anyhow::Result;

use super::super::assembler::{Assembler, Jumper};
use super::super::code::Func;
use super::super::config::{Config, ABI_AREA};
use super::super::generator::Generator;
use super::super::symbol::Loc;
use super::super::utils::{align_stack, Reg};

use super::*;

const REG_SIZE: u32 = 8;

pub struct ArmGenerator {
    a: Assembler,
    config: Config,
}

impl ArmGenerator {
    pub fn new(config: Config) -> ArmGenerator {
        ArmGenerator {
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

    fn sub_stack(&mut self, size: u32) {
        sub_stack(&mut self.a, size);
    }

    /*
    fn add_stack(&mut self, size: u32) {
        add_stack(&mut self.a, size);
    }
    */

    fn call_external(&mut self, op: &str, num_args: usize) -> Result<()> {
        let ofs = ABI_AREA as u32 * REG_SIZE;

        if self.config.is_kernel_func(op) {
            self.emit(arm! {add x(0), x(SP), #0});
            self.emit(arm! {eor x(1), x(1), x(1)});
            self.emit(arm! {eor x(2), x(2), x(2)});
            self.emit(arm! {add x(3), x(STACK), #ofs});
        } else {
            load_x_from_label(&mut self.a, 0, &format!("_env_{}_", op));
            self.emit(arm! {add x(1), x(STACK), #ofs});
            self.emit(arm! {movz x(2), #num_args});
            self.emit(arm! {add x(3), x(SP), #0});
        }

        let label = format!("_func_{}_", op);
        load_long(&mut self.a, 9, &label);
        self.emit(arm! {blr x(9)});

        self.load_stack(Reg::Ret, 0);
        if self.config.is_complex() {
            self.load_stack(Reg::Temp, 1);
        }

        Ok(())
    }
}

impl Generator for ArmGenerator {
    fn bytes(&mut self) -> Vec<u8> {
        self.a.bytes()
    }

    fn three_address(&self) -> bool {
        true
    }

    fn count_shadows(&self) -> u8 {
        14
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
        self.emit(arm! {umov x(0), v(ϕ(cond)).d[0]});

        let l = self.a.create_label();

        if is_else {
            self.jump(&l, 0, |offset, _| arm! {tbz x(0), #0, label(offset)});
        } else {
            self.jump(&l, 0, |offset, _| arm! {tbnz x(0), #0, label(offset)});
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

        self.emit(arm! {fmov d(ϕ(dst)), d(ϕ(s1))});
    }

    fn fxchg(&mut self, s1: Reg, s2: Reg) {
        self.emit(arm! {eor v(ϕ(s1)).8b, v(ϕ(s1)).8b, v(ϕ(s2)).8b});
        self.emit(arm! {eor v(ϕ(s2)).8b, v(ϕ(s1)).8b, v(ϕ(s2)).8b});
        self.emit(arm! {eor v(ϕ(s1)).8b, v(ϕ(s1)).8b, v(ϕ(s2)).8b});
    }

    fn load_const(&mut self, dst: Reg, idx: u32) {
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
        load_d_from_mem(&mut self.a, ϕ(dst), MEM, idx);
    }

    fn save_mem(&mut self, dst: Reg, idx: u32) {
        save_d_to_mem(&mut self.a, ϕ(dst), MEM, idx);
    }

    fn save_mem_result(&mut self, idx: u32) {
        self.save_mem(Reg::Ret, idx);
    }

    fn load_param(&mut self, dst: Reg, idx: u32) {
        load_d_from_mem(&mut self.a, ϕ(dst), PARAMS, idx);
    }

    fn load_stack(&mut self, dst: Reg, idx: u32) {
        if idx < 16 {
            load_d_from_mem(&mut self.a, ϕ(dst), SP, idx);
        } else {
            load_d_from_mem(&mut self.a, ϕ(dst), STACK, idx);
        }
    }

    fn save_stack(&mut self, dst: Reg, idx: u32) {
        if idx < 16 {
            save_d_to_mem(&mut self.a, ϕ(dst), SP, idx);
        } else {
            save_d_to_mem(&mut self.a, ϕ(dst), STACK, idx);
        }
    }

    fn load_mem_complex(&mut self, xd: Reg, yd: Reg, idx: u32) {
        load_paired_d_from_mem(&mut self.a, ϕ(xd), ϕ(yd), MEM, idx);
    }

    fn save_mem_complex(&mut self, xs: Reg, ys: Reg, idx: u32) {
        save_paired_d_to_mem(&mut self.a, ϕ(xs), ϕ(ys), MEM, idx);
    }

    fn load_param_complex(&mut self, xd: Reg, yd: Reg, idx: u32) {
        load_paired_d_from_mem(&mut self.a, ϕ(xd), ϕ(yd), PARAMS, idx);
    }

    fn load_stack_complex(&mut self, xd: Reg, yd: Reg, idx: u32) {
        if idx < 16 {
            load_paired_d_from_mem(&mut self.a, ϕ(xd), ϕ(yd), SP, idx);
        } else {
            load_paired_d_from_mem(&mut self.a, ϕ(xd), ϕ(yd), STACK, idx);
        }
    }

    fn save_stack_complex(&mut self, xs: Reg, ys: Reg, idx: u32) {
        if idx < 16 {
            save_paired_d_to_mem(&mut self.a, ϕ(xs), ϕ(ys), SP, idx);
        } else {
            save_paired_d_to_mem(&mut self.a, ϕ(xs), ϕ(ys), STACK, idx);
        }
    }

    fn save_stack_result(&mut self, idx: u32) {
        self.save_stack(Reg::Ret, idx);
    }

    fn load_args(&mut self, locs: Vec<Loc>, ultra: bool) {
        load_args_helper(
            &mut self.a,
            &self.config,
            &locs[..],
            ultra,
            32,
            |a, src, dst| {
                load_d_from_loc(a, 0, src);
                save_d_to_loc(a, 0, dst);
            },
            |a, arg, src| {
                load_d_from_loc(a, arg, src);
            },
        );
    }

    fn save_args(&mut self, num_args: u8, ultra: bool) {
        save_args_helper(
            &mut self.a,
            &self.config,
            num_args,
            ultra,
            32,
            |a, arg| {
                emit(a, arm! {ldr d(arg), [x(STACK), x(8), lsl #3]});
            },
            |a, arg, dst| {
                save_d_to_loc(a, arg, dst);
            },
        );
    }

    fn load_args_complex(&mut self, locs: Vec<Loc>, ultra: bool) {
        load_args_helper(
            &mut self.a,
            &self.config,
            &locs[..],
            ultra,
            32,
            |a, src, dst| {
                load_c_from_loc(a, 0, src);
                save_c_to_loc(a, 0, dst);
            },
            |a, arg, src| {
                load_c_from_loc(a, arg, src);
            },
        );
    }

    fn save_args_complex(&mut self, num_args: u8, ultra: bool) {
        save_args_helper(
            &mut self.a,
            &self.config,
            num_args,
            ultra,
            32,
            |a, arg| {
                emit(a, arm! {lsr x(8), x(8), #1});
                emit(a, arm! {ldr q(arg), [x(STACK), x(8), lsl #4]});
            },
            |a, arg, dst| {
                save_c_to_loc(a, arg, dst);
            },
        );
    }

    fn neg(&mut self, dst: Reg, s1: Reg) {
        self.emit(arm! {fneg d(ϕ(dst)), d(ϕ(s1))});
    }

    fn abs(&mut self, dst: Reg, s1: Reg) {
        self.emit(arm! {fabs d(ϕ(dst)), d(ϕ(s1))});
    }

    fn root(&mut self, dst: Reg, s1: Reg) {
        self.emit(arm! {fsqrt d(ϕ(dst)), d(ϕ(s1))});
    }

    fn real_root(&mut self, dst: Reg, s1: Reg) {
        self.root(dst, s1);
    }

    fn recip(&mut self, dst: Reg, s1: Reg) {
        self.emit(arm! {fmov d(TEMP), #1.0});
        self.emit(arm! {fdiv d(ϕ(dst)), d(TEMP), d(ϕ(s1))});
    }

    fn half(&mut self, dst: Reg, s1: Reg) {
        self.emit(arm! {fmov d(TEMP), #0.5});
        self.emit(arm! {fmul d(ϕ(dst)), d(ϕ(s1)), d(TEMP)});
    }

    fn round(&mut self, dst: Reg, s1: Reg) {
        self.emit(arm! {frinti d(ϕ(dst)), d(ϕ(s1))});
    }

    fn floor(&mut self, dst: Reg, s1: Reg) {
        self.emit(arm! {frintm d(ϕ(dst)), d(ϕ(s1))});
    }

    fn ceiling(&mut self, dst: Reg, s1: Reg) {
        self.emit(arm! {frintp d(ϕ(dst)), d(ϕ(s1))});
    }

    fn trunc(&mut self, dst: Reg, s1: Reg) {
        self.emit(arm! {frintz d(ϕ(dst)), d(ϕ(s1))});
    }

    fn frac(&mut self, dst: Reg, s1: Reg) {
        self.floor(Reg::Temp, s1);
        self.minus(dst, s1, Reg::Temp);
    }

    fn plus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(arm! {fadd d(ϕ(dst)), d(ϕ(s1)), d(ϕ(s2))});
    }

    fn minus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(arm! {fsub d(ϕ(dst)), d(ϕ(s1)), d(ϕ(s2))});
    }

    fn times(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(arm! {fmul d(ϕ(dst)), d(ϕ(s1)), d(ϕ(s2))});
    }

    fn divide(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(arm! {fdiv d(ϕ(dst)), d(ϕ(s1)), d(ϕ(s2))});
    }

    fn times_complex(&mut self, xd: Reg, yd: Reg, x1: Reg, y1: Reg, x2: Reg, y2: Reg) -> bool {
        let xt = Reg::Gen(2);
        let yt = Reg::Gen(3);

        self.times(xt, y1, y2);
        self.times(yt, x1, y2);
        self.fused_mul_add(yd, x2, y1, yt);
        self.fused_mul_sub(xd, x1, x2, xt);

        true
    }

    fn divide_complex(&mut self, xd: Reg, yd: Reg, x1: Reg, y1: Reg, x2: Reg, y2: Reg) -> bool {
        let xt = Reg::Gen(2);
        let yt = Reg::Gen(3);
        let t = Reg::Temp;

        self.times(xt, y1, y2);
        self.fused_mul_add(xt, x1, x2, xt);
        self.times(yt, x1, y2);
        self.fused_mul_sub(yt, x2, y1, yt);
        self.times(t, x2, x2);
        self.fused_mul_add(t, y2, y2, t);
        self.divide(xd, xt, t);
        self.divide(yd, yt, t);
        true
    }

    fn support_times2(&self) -> bool {
        false
    }

    fn times2_loc(&mut self, _d1: Reg, _s1: Reg, _l1: Loc, _d2: Reg, _s2: Reg, _l2: Loc) {
        unreachable!()
    }

    fn real(&mut self, dst: Reg, s1: Reg) {
        self.fmov(dst, s1);
    }

    fn imaginary(&mut self, dst: Reg, _s1: Reg) {
        self.xor(dst, dst, dst);
    }

    fn conjugate(&mut self, dst: Reg, s1: Reg) {
        self.fmov(dst, s1);
    }

    fn complex(&mut self, dst: Reg, s1: Reg, _s2: Reg) {
        self.fmov(dst, s1);
    }

    fn gt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(arm! {fcmgt d(ϕ(dst)), d(ϕ(s1)), d(ϕ(s2))});
    }

    fn geq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(arm! {fcmge d(ϕ(dst)), d(ϕ(s1)), d(ϕ(s2))});
    }

    fn lt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(arm! {fcmlt d(ϕ(dst)), d(ϕ(s1)), d(ϕ(s2))});
    }

    fn leq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(arm! {fcmle d(ϕ(dst)), d(ϕ(s1)), d(ϕ(s2))});
    }

    fn eq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(arm! {fcmeq d(ϕ(dst)), d(ϕ(s1)), d(ϕ(s2))});
    }

    fn neq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(arm! {fcmeq d(ϕ(dst)), d(ϕ(s1)), d(ϕ(s2))});
        self.emit(arm! {not v(ϕ(dst)).8b, v(ϕ(dst)).8b});
    }

    fn and(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(arm! {and v(ϕ(dst)).8b, v(ϕ(s1)).8b, v(ϕ(s2)).8b});
    }

    fn andnot(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(arm! {bic v(ϕ(dst)).8b, v(ϕ(s1)).8b, v(ϕ(s2)).8b});
    }

    fn or(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(arm! {orr v(ϕ(dst)).8b, v(ϕ(s1)).8b, v(ϕ(s2)).8b});
    }

    fn xor(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        if s1 == s2 {
            self.emit(arm! {movi d(ϕ(dst)), #0});
        } else {
            self.emit(arm! {eor v(ϕ(dst)).8b, v(ϕ(s1)).8b, v(ϕ(s2)).8b});
        }
    }

    fn not(&mut self, dst: Reg, s1: Reg) {
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
        self.align();
        add_consts(&mut self.a, consts);
    }

    fn add_func(&mut self, op: &str, f: Func) {
        add_func(&mut self.a, op, f);
    }

    fn call(&mut self, op: &str, num_args: usize) -> Result<()> {
        if self.config.is_external_func(op) {
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
            self.save_stack(Reg::Gen(0), 0);
            self.save_stack(Reg::Gen(1), 1);
        }

        self.call(op, num_args)?;

        self.load_stack(Reg::Ret, 0);
        self.load_stack(Reg::Temp, 1);
        Ok(())
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
        self.emit(arm! {sub sp, sp, #32});
        self.emit(arm! {stp lr, x(FP), [sp, #0]});
        self.emit(arm! {stp x(MEM), x(STACK), [sp, #16]});
        self.emit(arm! {mov x(FP), sp});

        let frame_size = align_stack((count_states + count_obs) as u32 * REG_SIZE);
        self.sub_stack(frame_size);
        self.emit(arm! {mov x(MEM), sp});

        let stack_size = align_stack(cap as u32 * REG_SIZE);
        self.sub_stack(stack_size);
        self.emit(arm! {mov x(STACK), sp});

        for i in 0..count_states {
            self.emit(arm! {str d(i), [x(MEM), #8*i]});
        }
    }

    fn epilogue_fast(
        &mut self,
        _cap: usize,
        _count_states: usize,
        _count_obs: usize,
        idx_ret: i32,
    ) {
        self.emit(arm! {ldr d(0), [x(MEM), #8*idx_ret]});

        self.emit(arm! {mov sp, x(FP)});
        self.emit(arm! {ldp lr, x(FP), [sp, #0]});
        self.emit(arm! {ldp x(MEM), x(STACK), [sp, #16]});
        self.emit(arm! {add sp, sp, #32});
        self.emit(arm! {eor x(0), x(0), x(0)});
        self.emit(arm! {ret});
    }

    fn prologue_indirect(
        &mut self,
        cap: usize,
        count_states: usize,
        count_obs: usize,
        _count_params: usize,
    ) {
        save_nonvolatile_regs(&mut self.a);

        self.emit(arm! {tst x(STATES), x(STATES)});
        self.jump("@main", 0, |offset, _| arm! {b.eq label(offset)});

        let frame_size = align_stack((count_states + count_obs) as u32 * REG_SIZE);
        self.sub_stack(frame_size);
        self.emit(arm! {mov x(MEM), sp});

        for i in 0..count_states {
            load_x_from_mem(&mut self.a, SCRATCH2, STATES, 2 * i as u32);
            self.emit(arm! {ldr d(0), [x(SCRATCH2), x(IDX), lsl #3]});
            save_d_to_mem(&mut self.a, 0, MEM, i as u32);
        }

        self.set_label("@main");

        let stack_size = align_stack(cap as u32 * REG_SIZE);
        allocate_stack(&mut self.a, stack_size, self.config.symbolica());
    }

    fn epilogue_indirect(
        &mut self,
        _cap: usize,
        count_states: usize,
        count_obs: usize,
        _count_params: usize,
    ) {
        self.emit(arm! {eor x(0), x(0), x(0)});
        self.set_label("@epilogue");

        self.emit(arm! {tst x(STATES), x(STATES)});
        self.jump("@done", 0, |offset, _| arm! {b.eq label(offset)});

        for i in 0..count_obs {
            load_x_from_mem(&mut self.a, SCRATCH2, STATES, 2 * (count_states + i) as u32);
            let k = (count_states + i) as u32;
            load_d_from_mem(&mut self.a, 0, MEM, k);
            self.emit(arm! {str d(0), [x(SCRATCH2), x(IDX), lsl #3]});
        }

        self.set_label("@done");

        load_nonvolatile_regs(&mut self.a);
        self.emit(arm! {eor x(0), x(0), x(0)});
        self.emit(arm! {ret});
    }

    fn save_used_registers(&mut self, used: &[Reg]) {
        for r in used {
            let phys_reg = ϕ(*r);
            if (8..=15).contains(&phys_reg) {
                self.save_stack(*r, phys_reg as u32);
            }
        }
    }

    fn load_used_registers(&mut self, used: &[Reg]) {
        for r in used {
            let phys_reg = ϕ(*r);
            if (8..=15).contains(&phys_reg) {
                self.load_stack(*r, phys_reg as u32);
            }
        }
    }
}

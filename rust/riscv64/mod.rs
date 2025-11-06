#[macro_use]
mod macros;

use crate::assembler::Assembler;
use crate::generator::Generator;
use crate::utils::{align_stack, reg, Reg};

const RA: u8 = 1;
const SP: u8 = 2;

const MEM: u8 = 18; // first arg = mem if direct mode, otherwise null
const STATES: u8 = 19; // second arg = states+obs if indirect mode, otherwise null
const IDX: u8 = 20; // third arg = index if indirect mode
const PARAMS: u8 = 21; // fourth arg = params
const TEXT: u8 = 22;
const SAVED: u8 = 8;
const RET: u8 = 10;
const TEMP: u8 = 11;

pub struct RiscVGenerator {
    a: Assembler,
}

const REG_MAP: [u8; 16] = [10, 11, 12, 13, 14, 15, 16, 17, 5, 6, 7, 8, 28, 29, 30, 31];

fn ϕ(r: Reg) -> u8 {
    match r {
        Reg::Ret => 10,
        Reg::Temp => 11,
        Reg::Left => 10,
        Reg::Right => 11,
        Reg::Gen(dst) => REG_MAP[dst as usize],
        Reg::Static(..) => panic!("passing static registers to codegen"),
    }
}

impl RiscVGenerator {
    pub fn new() -> RiscVGenerator {
        RiscVGenerator {
            a: Assembler::new(),
        }
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

    //fn jump(&mut self, label: &str, code: u32) {
    //    self.a.jump(label, code)
    //}

    fn apply_jumps(&mut self) {
        self.a.apply_jumps();
    }

    fn emit(&mut self, w: u32) {
        self.a.append_word(w);
    }

    fn load_d_from_mem(&mut self, d: u8, base: u8, idx: u32) {
        self.emit(rvv! {fld f(d), x(base), 8*idx});
    }

    fn save_d_to_mem(&mut self, d: u8, base: u8, idx: u32) {
        self.emit(rvv! {fsd f(d), x(base), 8*idx});
    }

    fn load_x_from_mem(&mut self, r: u8, base: u8, idx: u32) {
        self.emit(rvv! {ld x(r), x(base), 8*idx});
    }

    fn sub_stack(&mut self, size: u32) {
        self.emit(rvv! {addi x(SP), x(SP), -(size as i32)});
    }

    fn add_stack(&mut self, size: u32) {
        self.emit(rvv! {addi x(SP), x(SP), (size as i32)});
    }
}

impl Generator for RiscVGenerator {
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

    fn align(&mut self) {}

    //***********************************/

    fn fmov(&mut self, dst: Reg, s1: Reg) {
        if dst == s1 {
            return;
        }
        self.emit(rvv! {fmv.d f(ϕ(dst)), f(ϕ(s1))});
    }

    fn fxchg(&mut self, s1: Reg, s2: Reg) {
        todo!();
    }

    fn load_const(&mut self, dst: Reg, idx: u32) {
        let label = format!("_const_{}_", idx);
        let f = |offset| itype!(0, 0, offset, 0);
        self.a
            .jump(label.as_str(), rvv! {fld f(ϕ(dst)), x(TEXT), 0}, f);
    }

    fn load_mem(&mut self, dst: Reg, idx: u32) {
        self.load_d_from_mem(ϕ(dst), MEM, idx);
    }

    fn save_mem(&mut self, dst: Reg, idx: u32) {
        self.save_d_to_mem(ϕ(dst), MEM, idx);
    }

    fn save_mem_result(&mut self, idx: u32) {
        self.save_mem(Reg::Ret, idx);
    }

    fn load_param(&mut self, dst: Reg, idx: u32) {
        self.load_d_from_mem(ϕ(dst), PARAMS, idx);
    }

    fn load_stack(&mut self, dst: Reg, idx: u32) {
        self.load_d_from_mem(ϕ(dst), SP, idx);
    }
    fn save_stack(&mut self, dst: Reg, idx: u32) {
        self.save_d_to_mem(ϕ(dst), SP, idx);
    }

    fn save_stack_result(&mut self, idx: u32) {
        self.save_stack(Reg::Ret, idx);
    }

    fn neg(&mut self, dst: Reg, s1: Reg) {
        self.emit(rvv! {fneg.d f(ϕ(dst)), f(ϕ(s1))});
    }

    fn abs(&mut self, dst: Reg, s1: Reg) {
        self.emit(rvv! {fabs.d f(ϕ(dst)), f(ϕ(s1))});
    }

    fn root(&mut self, dst: Reg, s1: Reg) {
        self.emit(rvv! {fsqrt.d f(ϕ(dst)), f(ϕ(s1))});
    }

    fn recip(&mut self, dst: Reg, s1: Reg) {
        // self.emit(arm! {fmov d(TEMP), #1.0});
        self.emit(rvv! {fdiv.d f(ϕ(dst)), f(TEMP),f(ϕ(s1))});
    }

    fn round(&mut self, dst: Reg, s1: Reg) {
        self.emit(rvv! {fround.d f(ϕ(dst)), f(ϕ(s1)), 0});
    }

    fn floor(&mut self, dst: Reg, s1: Reg) {
        self.emit(rvv! {fround.d f(ϕ(dst)), f(ϕ(s1)), 2});
    }

    fn ceiling(&mut self, dst: Reg, s1: Reg) {
        self.emit(rvv! {fround.d f(ϕ(dst)), f(ϕ(s1)), 3});
    }

    fn trunc(&mut self, dst: Reg, s1: Reg) {
        self.emit(rvv! {fround.d f(ϕ(dst)), f(ϕ(s1)), 1});
    }

    fn frac(&mut self, dst: Reg, s1: Reg) {
        self.floor(Reg::Temp, s1);
        self.minus(dst, s1, Reg::Temp);
    }

    fn plus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(rvv! {fadd.d f(ϕ(dst)), f(ϕ(s1)), f(ϕ(s2))});
    }

    fn minus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(rvv! {fsub.d f(ϕ(dst)), f(ϕ(s1)), f(ϕ(s2))});
    }

    fn times(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(rvv! {fmul.d f(ϕ(dst)), f(ϕ(s1)), f(ϕ(s2))});
    }

    fn divide(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(rvv! {fdiv.d f(ϕ(dst)), f(ϕ(s1)), f(ϕ(s2))});
    }

    fn gt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(rvv! {fgt.d x(ϕ(dst)), f(ϕ(s1)), f(ϕ(s2))});
    }

    fn geq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(rvv! {fge.d x(ϕ(dst)), f(ϕ(s1)), f(ϕ(s2))});
    }

    fn lt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(rvv! {flt.d x(ϕ(dst)), f(ϕ(s1)), f(ϕ(s2))});
    }

    fn leq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(rvv! {fle.d x(ϕ(dst)), f(ϕ(s1)), f(ϕ(s2))});
    }

    fn eq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(rvv! {feq.d x(ϕ(dst)), f(ϕ(s1)), f(ϕ(s2))});
    }

    fn neq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(rvv! {feq.d x(ϕ(dst)), f(ϕ(s1)), f(ϕ(s2))});
        self.emit(rvv! {not x(ϕ(dst)), x(ϕ(dst))});
    }

    fn and(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(rvv! {and x(ϕ(dst)), x(ϕ(s1)), x(ϕ(s2))});
    }

    fn andnot(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(rvv! {and x(ϕ(dst)), x(ϕ(s1)), x(ϕ(s2))});
        self.emit(rvv! {not x(ϕ(dst)), x(ϕ(dst))});
    }

    fn or(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(rvv! {or x(ϕ(dst)), x(ϕ(s1)), x(ϕ(s2))});
    }

    fn xor(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(rvv! {xor x(ϕ(dst)), x(ϕ(s1)), x(ϕ(s2))});
    }

    fn not(&mut self, dst: Reg, s1: Reg) {
        self.emit(rvv! {not x(ϕ(dst)), x(ϕ(dst))});
    }

    fn fused_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        todo!();
    }

    // fused_mul_sub is s1 * s2 - s3, corresponding to fnmsub in aarch64
    // and vmsub... in amd64
    fn fused_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        todo!();
    }

    // fused_neg_mul_add is s3 - s1 * s2, corresponding to fmsub in aarch64
    // and vnmadd... in amd64
    fn fused_neg_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        todo!();
    }

    // fused_neg_mul_sub is -s3 - s1 * s2, corresponding to fnmadd in aarch64
    // and vnmsub... in amd64
    fn fused_neg_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        todo!();
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
        let f = |offset| itype!(0, 0, offset, 0);
        self.a.jump(label.as_str(), rvv! {ld x(RET), x(TEXT), 0}, f);
        self.emit(rvv! {jalr x(RA), x(RET), 0});
    }

    fn prologue_fast(&mut self, cap: u32, num_args: u32) {
        /*
        self.emit(arm! {sub sp, sp, #16});
        self.emit(arm! {str lr, [sp, #0]});
        self.emit(arm! {str x(MEM), [sp, #8]});

        let stack_size = align_stack(self.reg_size() * cap);
        self.sub_stack(stack_size);

        self.emit(arm! {mov x(MEM), sp});

        let num_args = num_args as i32;

        for i in 0..num_args {
            self.emit(arm! {str d(i), [sp, #8*i]});
        }
        */
    }

    fn epilogue_fast(&mut self, cap: u32, idx_ret: i32) {
        /*
        self.emit(arm! {ldr d(0), [sp, #8*idx_ret]});

        let stack_size = align_stack(self.reg_size() * cap);
        self.add_stack(stack_size);

        self.emit(arm! {ldr x(MEM), [sp, #8]});
        self.emit(arm! {ldr lr, [sp, #0]});
        self.emit(arm! {add sp, sp, #16});
        self.emit(arm! {ret});
        */
    }

    /*
     * MEM => first arg = mem if direct mode, otherwise null
     * STATES => second arg = states+obs if indirect mode, otherwise null
     * IDX => third arg = index if indirect mode
     * PARAMS => fourth arg = params
     */
    fn prologue_indirect(&mut self, cap: u32, count_states: usize, count_obs: usize) {
	self.set_label("@text");
        self.emit(rvv! {auipc x(5), 0});
        self.emit(rvv! {addi x(SP), x(SP), -64});

        self.emit(rvv! {sd x(RA), x(SP), 0});
        self.emit(rvv! {sd x(MEM), x(SP), 8});
        self.emit(rvv! {sd x(PARAMS), x(SP), 16});
        self.emit(rvv! {sd x(STATES), x(SP), 24});
        self.emit(rvv! {sd x(IDX), x(SP), 32});
        self.emit(rvv! {sd x(TEXT), x(SP), 40});
        self.emit(rvv! {sd x(SAVED), x(SP), 48});

        self.emit(rvv! {mv x(MEM), x(10)});
        self.emit(rvv! {mv x(STATES), x(11)});
        self.emit(rvv! {mv x(IDX), x(12)});
        self.emit(rvv! {mv x(PARAMS), x(13)});
        self.emit(rvv! {mv x(TEXT), x(5)});

        /*
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
        */

        let stack_size = align_stack(self.reg_size() * cap);
        self.sub_stack(stack_size);
    }

    fn epilogue_indirect(&mut self, cap: u32, count_states: usize, count_obs: usize) {
        let stack_size = align_stack(self.reg_size() * cap);
        self.add_stack(stack_size);

        /*
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
        */

        self.emit(rvv! {ld x(RA), x(SP), 0});
        self.emit(rvv! {ld x(MEM), x(SP), 8});
        self.emit(rvv! {ld x(PARAMS), x(SP), 16});
        self.emit(rvv! {ld x(STATES), x(SP), 24});
        self.emit(rvv! {ld x(IDX), x(SP), 32});
        self.emit(rvv! {ld x(TEXT), x(SP), 40});
        self.emit(rvv! {ld x(SAVED), x(SP), 48});

        self.emit(rvv! {addi x(SP), x(SP), 64});
        self.emit(rvv! {ret});
    }

    fn save_used_registers(&mut self, used: &[u8]) {
        let count_shadows = self.count_shadows();

        for r in used {
            if *r >= count_shadows {
                self.save_stack(reg(*r), *r as u32);
            }
        }
    }

    fn load_used_registers(&mut self, used: &[u8]) {
        let count_shadows = self.count_shadows();

        for r in used {
            if *r >= count_shadows {
                self.load_stack(reg(*r), *r as u32);
            }
        }
    }
}

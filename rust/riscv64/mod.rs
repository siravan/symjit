#[macro_use]
mod macros;

use crate::assembler::Assembler;
use crate::generator::Generator;
use crate::utils::{align_stack, Reg};

pub struct RiscV {
    a: Assembler,
}

impl RiscV {
    const zero: u8 = 0;
    const ra: u8 = 1;
    const sp: u8 = 2;
    const gp: u8 = 3;
    const tp: u8 = 4;
    const t0: u8 = 5;
    const t1: u8 = 6;
    const t2: u8 = 7;
    const s0: u8 = 8;
    const fp: u8 = 8;
    const s1: u8 = 9;
    const a0: u8 = 10;
    const a1: u8 = 11;
    const a2: u8 = 12;
    const a3: u8 = 13;
    const a4: u8 = 14;
    const a5: u8 = 15;
    const a6: u8 = 16;
    const a7: u8 = 17;
    const s2: u8 = 18;
    const s3: u8 = 19;
    const s4: u8 = 20;
    const s5: u8 = 21;
    const s6: u8 = 22;
    const s7: u8 = 23;
    const s8: u8 = 24;
    const s9: u8 = 25;
    const s10: u8 = 26;
    const s11: u8 = 27;
    const t3: u8 = 28;
    const t4: u8 = 29;
    const t5: u8 = 30;
    const t6: u8 = 31;

    const ft0: u8 = 0;
    const ft1: u8 = 1;
    const ft2: u8 = 2;
    const ft3: u8 = 3;
    const ft4: u8 = 4;
    const ft5: u8 = 5;
    const ft6: u8 = 6;
    const ft7: u8 = 7;
    const fs0: u8 = 8;
    const fs1: u8 = 9;
    const fa0: u8 = 10;
    const fa1: u8 = 11;
    const fa2: u8 = 12;
    const fa3: u8 = 13;
    const fa4: u8 = 14;
    const fa5: u8 = 15;
    const fa6: u8 = 16;
    const fa7: u8 = 17;
    const fs2: u8 = 18;
    const fs3: u8 = 19;
    const fs4: u8 = 20;
    const fs5: u8 = 21;
    const fs6: u8 = 22;
    const fs7: u8 = 23;
    const fs8: u8 = 24;
    const fs9: u8 = 25;
    const fs10: u8 = 26;
    const fs11: u8 = 27;
    const ft8: u8 = 28;
    const ft9: u8 = 29;
    const ft10: u8 = 30;
    const ft11: u8 = 31;
}

const FMAP: [u8; 16] = [
    RiscV::fa0,
    RiscV::fa1,
    RiscV::fa2,
    RiscV::fa3,
    RiscV::fa4,
    RiscV::fa5,
    RiscV::fa6,
    RiscV::fa7,
    RiscV::ft4,
    RiscV::ft5,
    RiscV::ft6,
    RiscV::ft7,
    RiscV::ft8,
    RiscV::ft9,
    RiscV::ft10,
    RiscV::ft11,
];

fn ϕ(r: Reg) -> u8 {
    match r {
        Reg::Ret | Reg::Left => RiscV::fa0,
        Reg::Temp | Reg::Right => RiscV::fa1,
        Reg::Gen(dst) => FMAP[dst as usize],
        Reg::Static(..) => panic!("passing static registers to codegen"),
    }
}

const XMAP: [u8; 16] = [
    RiscV::a0,
    99,
    RiscV::a2,
    RiscV::a3,
    RiscV::a4,
    RiscV::a5,
    RiscV::a6,
    RiscV::a7,
    RiscV::a2,
    RiscV::t0,
    RiscV::t1,
    RiscV::t2,
    RiscV::t3,
    RiscV::t4,
    RiscV::t5,
    RiscV::t6,
];

fn λ(r: Reg) -> u8 {
    match r {
        Reg::Ret | Reg::Left => RiscV::a0,
        Reg::Temp | Reg::Right => panic!("reg Temp/Right not defined for x-registers"),
        Reg::Gen(dst) => XMAP[dst as usize],
        Reg::Static(..) => panic!("passing static registers to codegen"),
    }
}

const MEM: u8 = RiscV::fs2; // first arg = mem if direct mode, otherwise null
const STATES: u8 = RiscV::fs3; // second arg = states+obs if indirect mode, otherwise null
const IDX: u8 = RiscV::fs4; // third arg = index if indirect mode
const PARAMS: u8 = RiscV::fs5; // fourth arg = params

const RET: u8 = RiscV::fa0;
const TEMP: u8 = RiscV::fa1;

impl RiscV {
    pub fn new() -> RiscV {
        RiscV {
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

    fn sub_stack(&mut self, size: u32) {
        self.emit(rvv! {addi x(Self::sp), x(Self::sp), -(size as i32)});
    }

    fn add_stack(&mut self, size: u32) {
        self.emit(rvv! {addi x(Self::sp), x(Self::sp), (size as i32)});
    }
}

impl Generator for RiscV {
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
        println!("{:#?}", &self.a);
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

        let f = |offset| utype!(0, offset, 0);
        self.a.jump(label.as_str(), rvv! {auipc x(Self::a0), 0}, f);

        let f = |offset| itype!(0, 0, offset + 4, 0);
        self.a
            .jump(label.as_str(), rvv! {fld f(ϕ(dst)), x(Self::a0), 0}, f);
    }

    fn load_mem(&mut self, dst: Reg, idx: u32) {
        self.emit(rvv! {fld f(ϕ(dst)), x(MEM), 8*idx});
    }

    fn save_mem(&mut self, dst: Reg, idx: u32) {
        self.emit(rvv! {fsd f(ϕ(dst)), x(MEM), 8*idx});
    }

    fn save_mem_result(&mut self, idx: u32) {
        self.emit(rvv! {fsd f(Self::fa0), x(MEM), 8*idx});
    }

    fn load_param(&mut self, dst: Reg, idx: u32) {
        self.emit(rvv! {fld f(ϕ(dst)), x(PARAMS), 8*idx});
    }

    fn load_stack(&mut self, dst: Reg, idx: u32) {
        self.emit(rvv! {fld f(ϕ(dst)), x(Self::sp), 8*idx});
    }

    fn save_stack(&mut self, dst: Reg, idx: u32) {
        self.emit(rvv! {fsd f(ϕ(dst)), x(Self::sp), 8*idx});
    }

    fn save_stack_result(&mut self, idx: u32) {
        self.emit(rvv! {fsd f(Self::fa0), x(Self::sp), 8*idx});
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
        self.emit(rvv! {addi x(Self::t0), x(Self::zero), 1});
        self.emit(rvv! {fcvt.d.w f(Self::fa0), x(Self::t0)});
        self.emit(rvv! {fdiv.d f(ϕ(dst)), f(Self::fa0), f(ϕ(s1))});
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
        self.emit(rvv! {fgt.d x(Self::t0), f(ϕ(s1)), f(ϕ(s2))});
        self.emit(rvv! {fcvt.d.w f(ϕ(dst)), x(Self::t0)});
    }

    fn geq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(rvv! {fge.d x(Self::t0), f(ϕ(s1)), f(ϕ(s2))});
        self.emit(rvv! {fcvt.d.w f(ϕ(dst)), x(Self::t0)});
    }

    fn lt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(rvv! {flt.d x(Self::t0), f(ϕ(s1)), f(ϕ(s2))});
        self.emit(rvv! {fcvt.d.w f(ϕ(dst)), x(Self::t0)});
    }

    fn leq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(rvv! {fle.d x(Self::t0), f(ϕ(s1)), f(ϕ(s2))});
        self.emit(rvv! {fcvt.d.w f(ϕ(dst)), x(Self::t0)});
    }

    fn eq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(rvv! {feq.d x(Self::t0), f(ϕ(s1)), f(ϕ(s2))});
        self.emit(rvv! {fcvt.d.w f(ϕ(dst)), x(Self::t0)});
    }

    fn neq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(rvv! {feq.d x(Self::t0), f(ϕ(s1)), f(ϕ(s2))});
        self.emit(rvv! {not x(Self::t0), x(Self::t0)});
        self.emit(rvv! {fcvt.d.w f(ϕ(dst)), x(Self::t0)});
    }

    fn and(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(rvv! {fmin.d f(ϕ(dst)), f(ϕ(s1)), f(ϕ(s2))});
    }

    fn andnot(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(rvv! {fcvt.w.d x(Self::t0), f(ϕ(s1))});
        self.emit(rvv! {fcvt.w.d x(Self::t1), f(ϕ(s2))});
        self.emit(rvv! {not x(Self::t0), x(Self::t0)});
        self.emit(rvv! {and x(Self::t0), x(Self::t0), x(Self::t1)});
        self.emit(rvv! {fcvt.d.w f(ϕ(dst)), x(Self::t0)});
    }

    fn or(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(rvv! {fmax.d f(ϕ(dst)), f(ϕ(s1)), f(ϕ(s2))});
    }

    fn xor(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.neq(dst, s1, s2);
    }

    fn not(&mut self, dst: Reg, s1: Reg) {
        self.emit(rvv! {fcvt.w.d x(Self::t0), f(ϕ(s1))});
        self.emit(rvv! {not x(Self::t0), x(Self::t0)});
        self.emit(rvv! {fcvt.d.w f(ϕ(dst)), x(Self::t0)});
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

        let f = |offset| utype!(0, offset, 0);
        self.a.jump(label.as_str(), rvv! {auipc x(Self::a0), 0}, f);

        let f = |offset| itype!(0, 0, offset + 4, 0);
        self.a
            .jump(label.as_str(), rvv! {ld x(Self::a0), x(Self::a0), 0}, f);

        self.emit(rvv! {jalr x(Self::ra), x(Self::a0), 0});
    }

    fn ifelse(&mut self, dst: Reg, true_val: Reg, false_val: Reg, idx: u32) {
        self.emit(rvv! {ld x(Self::a0), x(MEM), idx});
        self.emit(rvv! {beq x(Self::a0), x(Self::zero), 12});
        self.emit(rvv! {fmv.d f(ϕ(dst)), f(ϕ(true_val))});
        self.emit(rvv! {beq x(Self::zero), x(Self::zero), 8});
        self.emit(rvv! {fmv.d f(ϕ(dst)), f(ϕ(false_val))});
    }

    /********************************************************/

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
        self.emit(rvv! {addi x(Self::sp), x(Self::sp), -64});

        self.emit(rvv! {sd x(Self::ra), x(Self::sp), 0});
        self.emit(rvv! {sd x(MEM), x(Self::sp), 8});
        self.emit(rvv! {sd x(STATES), x(Self::sp), 16});
        self.emit(rvv! {sd x(IDX), x(Self::sp), 24});
        self.emit(rvv! {sd x(PARAMS), x(Self::sp), 32});

        self.emit(rvv! {mv x(MEM), x(Self::a0)});
        self.emit(rvv! {mv x(STATES), x(Self::a1)});
        self.emit(rvv! {mv x(IDX), x(Self::a2)});
        self.emit(rvv! {mv x(PARAMS), x(Self::a3)});

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

        self.emit(rvv! {ld x(Self::ra), x(Self::sp), 0});
        self.emit(rvv! {ld x(MEM), x(Self::sp), 8});
        self.emit(rvv! {ld x(STATES), x(Self::sp), 16});
        self.emit(rvv! {ld x(IDX), x(Self::sp), 24});
        self.emit(rvv! {ld x(PARAMS), x(Self::sp), 32});

        self.emit(rvv! {addi x(Self::sp), x(Self::sp), 64});
        self.emit(rvv! {ret});
    }

    fn save_used_registers(&mut self, _used: &[u8]) {}

    fn load_used_registers(&mut self, _used: &[u8]) {}
}

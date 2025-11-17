#[macro_use]
mod macros;

use crate::assembler::{Assembler, Jumper};
use crate::generator::Generator;
use crate::utils::{align_stack, Reg};

fn hi(x: u32) -> u32 {
    if x & 0x0800 != 0 {
        (x >> 12) + 1
    } else {
        x >> 12
    }
}

fn lo(x: u32) -> u32 {
    x & 0x0fff
}

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

const FMAP: [u8; 14] = [
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

const MEM: u8 = RiscV::fs0; // first arg = mem if direct mode, otherwise null
const STATES: u8 = RiscV::fs1; // second arg = states+obs if indirect mode, otherwise null
const IDX: u8 = RiscV::fs2; // third arg = index if indirect mode
const PARAMS: u8 = RiscV::fs3; // fourth arg = params

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

    fn apply_jumps(&mut self) {
        self.a.apply_jumps();
    }

    fn jump(&mut self, label: &str, code: u32, f: Jumper) {
        self.a.jump(label, code, f)
    }

    fn emit(&mut self, w: u32) {
        self.a.append_word(w);
    }

    fn load_float(&mut self, d: u8, base: u8, offset: u32) {
        if offset < 2048 {
            self.emit(rvv! {fld f(d), x(base), offset});
        } else {
            self.emit(rvv! {lui x(Self::t0), hi(offset)});
            self.emit(rvv! {addi x(Self::t0), x(base), lo(offset)});
            self.emit(rvv! {fld f(d), x(Self::t0), 0});
        }
    }

    fn save_float(&mut self, d: u8, base: u8, offset: u32) {
        if offset < 2048 {
            self.emit(rvv! {fsd f(d), x(base), offset});
        } else {
            self.emit(rvv! {lui x(Self::t0), hi(offset)});
            self.emit(rvv! {addi x(Self::t0), x(base), lo(offset)});
            self.emit(rvv! {fsd f(d), x(Self::t0), 0});
        }
    }

    fn load_int(&mut self, d: u8, base: u8, offset: u32) {
        if offset < 2048 {
            self.emit(rvv! {ld x(d), x(base), offset});
        } else {
            self.emit(rvv! {lui x(Self::t0), hi(offset)});
            self.emit(rvv! {addi x(Self::t0), x(base), lo(offset)});
            self.emit(rvv! {ld x(d), x(Self::t0), 0});
        }
    }

    fn li(&mut self, dst: u8, val: i32) {
        if !(-2048..2048).contains(&val) {
            self.emit(rvv! {lui x(dst), hi(val as u32)});
            self.emit(rvv! {addi x(dst), x(dst), lo(val as u32)});
        } else {
            self.emit(rvv! {addi x(dst), x(Self::zero), lo(val as u32)});
        }
    }

    fn sub_stack(&mut self, size: u32) {
        if size < 2048 {
            self.emit(rvv! {addi x(Self::sp), x(Self::sp), -(size as i32)});
        } else {
            self.li(Self::t0, -(size as i32));
            self.emit(rvv! {add x(Self::sp), x(Self::sp), x(Self::t0)});
        }
    }

    fn add_stack(&mut self, size: u32) {
        if size < 2048 {
            self.emit(rvv! {addi x(Self::sp), x(Self::sp), size});
        } else {
            self.li(Self::t0, size as i32);
            self.emit(rvv! {add x(Self::sp), x(Self::sp), x(Self::t0)});
        }
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
    }

    fn align(&mut self) {}

    fn set_label(&mut self, label: &str) {
        self.a.set_label(label);
    }

    fn branch_if(&mut self, cond: Reg, label: &str) {
        self.emit(rvv! {fmv.x.d x(Self::t0), f(ϕ(cond))});
        self.emit(rvv! {beq x(Self::t0), x(Self::zero), 8});
        self.jump(label, 0, |offset, _| rvv! {j offset});
    }

    //***********************************/
    fn fmov(&mut self, dst: Reg, s1: Reg) {
        if dst == s1 {
            return;
        }
        self.emit(rvv! {fmv.d f(ϕ(dst)), f(ϕ(s1))});
    }

    fn fxchg(&mut self, _s1: Reg, _s2: Reg) {
        todo!();
    }

    fn load_const(&mut self, dst: Reg, idx: u32) {
        let label = format!("_const_{}_", idx);

        self.jump(
            label.as_str(),
            0,
            |offset, _| rvv! {auipc x(Self::a0), hi(offset as u32)},
        );

        self.jump(
            label.as_str(),
            ϕ(dst) as u32,
            |offset, r| rvv! {fld f(r), x(Self::a0), lo((offset + 4) as u32)},
        );
    }

    fn load_mem(&mut self, dst: Reg, idx: u32) {
        self.load_float(ϕ(dst), MEM, 8 * idx);
    }

    fn save_mem(&mut self, dst: Reg, idx: u32) {
        self.save_float(ϕ(dst), MEM, 8 * idx);
    }

    fn save_mem_result(&mut self, idx: u32) {
        self.save_float(Self::fa0, MEM, 8 * idx);
    }

    fn load_param(&mut self, dst: Reg, idx: u32) {
        self.load_float(ϕ(dst), PARAMS, 8 * idx);
    }

    fn load_stack(&mut self, dst: Reg, idx: u32) {
        self.load_float(ϕ(dst), Self::sp, 8 * idx);
    }

    fn save_stack(&mut self, dst: Reg, idx: u32) {
        self.save_float(ϕ(dst), Self::sp, 8 * idx);
    }

    fn save_stack_result(&mut self, idx: u32) {
        self.save_float(Self::fa0, Self::sp, 8 * idx);
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
        self.emit(rvv! {fcvt.l.d x(Self::t0), f(ϕ(s1)), 0});
        self.emit(rvv! {fcvt.d.l f(ϕ(dst)), x(Self::t0)});
    }

    fn trunc(&mut self, dst: Reg, s1: Reg) {
        self.emit(rvv! {fcvt.l.d x(Self::t0), f(ϕ(s1)), 1});
        self.emit(rvv! {fcvt.d.l f(ϕ(dst)), x(Self::t0)});
    }

    fn floor(&mut self, dst: Reg, s1: Reg) {
        self.emit(rvv! {fcvt.l.d x(Self::t0), f(ϕ(s1)), 2});
        self.emit(rvv! {fcvt.d.l f(ϕ(dst)), x(Self::t0)});
    }

    fn ceiling(&mut self, dst: Reg, s1: Reg) {
        self.emit(rvv! {fcvt.l.d x(Self::t0), f(ϕ(s1)), 3});
        self.emit(rvv! {fcvt.d.l f(ϕ(dst)), x(Self::t0)});
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
        self.emit(rvv! {neg x(Self::t0), x(Self::t0)});
        self.emit(rvv! {fmv.d.x f(ϕ(dst)), x(Self::t0)});
    }

    fn geq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(rvv! {fge.d x(Self::t0), f(ϕ(s1)), f(ϕ(s2))});
        self.emit(rvv! {neg x(Self::t0), x(Self::t0)});
        self.emit(rvv! {fmv.d.x f(ϕ(dst)), x(Self::t0)});
    }

    fn lt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(rvv! {flt.d x(Self::t0), f(ϕ(s1)), f(ϕ(s2))});
        self.emit(rvv! {neg x(Self::t0), x(Self::t0)});
        self.emit(rvv! {fmv.d.x f(ϕ(dst)), x(Self::t0)});
    }

    fn leq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(rvv! {fle.d x(Self::t0), f(ϕ(s1)), f(ϕ(s2))});
        self.emit(rvv! {neg x(Self::t0), x(Self::t0)});
        self.emit(rvv! {fmv.d.x f(ϕ(dst)), x(Self::t0)});
    }

    fn eq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(rvv! {feq.d x(Self::t0), f(ϕ(s1)), f(ϕ(s2))});
        self.emit(rvv! {neg x(Self::t0), x(Self::t0)});
        self.emit(rvv! {fmv.d.x f(ϕ(dst)), x(Self::t0)});
    }

    fn neq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(rvv! {feq.d x(Self::t0), f(ϕ(s1)), f(ϕ(s2))});
        self.emit(rvv! {neg x(Self::t0), x(Self::t0)});
        self.emit(rvv! {not x(Self::t0), x(Self::t0)});
        self.emit(rvv! {fmv.d.x f(ϕ(dst)), x(Self::t0)});
    }

    fn and(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(rvv! {fmv.x.d x(Self::t0), f(ϕ(s1))});
        self.emit(rvv! {fmv.x.d x(Self::t1), f(ϕ(s2))});
        self.emit(rvv! {and x(Self::t0), x(Self::t0), x(Self::t1)});
        self.emit(rvv! {fmv.d.x f(ϕ(dst)), x(Self::t0)});
    }

    fn andnot(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(rvv! {fmv.x.d x(Self::t0), f(ϕ(s1))});
        self.emit(rvv! {not x(Self::t0), x(Self::t0)});
        self.emit(rvv! {fmv.x.d x(Self::t1), f(ϕ(s2))});
        self.emit(rvv! {and x(Self::t0), x(Self::t0), x(Self::t1)});
        self.emit(rvv! {fmv.d.x f(ϕ(dst)), x(Self::t0)});
    }

    fn or(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(rvv! {fmv.x.d x(Self::t0), f(ϕ(s1))});
        self.emit(rvv! {fmv.x.d x(Self::t1), f(ϕ(s2))});
        self.emit(rvv! {or x(Self::t0), x(Self::t0), x(Self::t1)});
        self.emit(rvv! {fmv.d.x f(ϕ(dst)), x(Self::t0)});
    }

    fn xor(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(rvv! {fmv.x.d x(Self::t0), f(ϕ(s1))});
        self.emit(rvv! {fmv.x.d x(Self::t1), f(ϕ(s2))});
        self.emit(rvv! {xor x(Self::t0), x(Self::t0), x(Self::t1)});
        self.emit(rvv! {fmv.d.x f(ϕ(dst)), x(Self::t0)});
    }

    fn not(&mut self, dst: Reg, s1: Reg) {
        self.emit(rvv! {fmv.x.d x(Self::t0), f(ϕ(s1))});
        self.emit(rvv! {not x(Self::t0), x(Self::t0)});
        self.emit(rvv! {fmv.d.x f(ϕ(dst)), x(Self::t0)});
    }

    fn fused_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        self.emit(rvv! {fmadd.d f(ϕ(dst)), f(ϕ(s1)), f(ϕ(s2)), f(ϕ(s3))});
    }

    // fused_mul_sub is s1 * s2 - s3, corresponding to fnmsub in aarch64,
    // vmsub... in amd64 and fmsub in risc v
    fn fused_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        self.emit(rvv! {fmsub.d f(ϕ(dst)), f(ϕ(s1)), f(ϕ(s2)), f(ϕ(s3))});
    }

    // fused_neg_mul_add is s3 - s1 * s2, corresponding to fmsub in aarch64,
    // vnmadd... in amd64 and fnmsub in risc v
    fn fused_neg_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        self.emit(rvv! {fnmsub.d f(ϕ(dst)), f(ϕ(s1)), f(ϕ(s2)), f(ϕ(s3))});
    }

    // fused_neg_mul_sub is -s3 - s1 * s2, corresponding to fnmadd in aarch64,
    // vnmsub... in amd64 and fnmadd in risc v
    fn fused_neg_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        self.emit(rvv! {fnmadd.d f(ϕ(dst)), f(ϕ(s1)), f(ϕ(s2)), f(ϕ(s3))});
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

        self.jump(
            label.as_str(),
            0,
            |offset, _| rvv! {auipc x(Self::a0), hi(offset as u32)},
        );

        self.jump(
            label.as_str(),
            0,
            |offset, _| rvv! {ld x(Self::a0), x(Self::a0), lo((offset + 4) as u32)},
        );

        self.emit(rvv! {jalr x(Self::ra), x(Self::a0), 0});
    }

    fn ifelse(&mut self, dst: Reg, true_val: Reg, false_val: Reg, idx: u32) {
        self.load_int(Self::t0, Self::sp, 8 * idx);
        self.emit(rvv! {fmv.x.d x(Self::t1), f(ϕ(true_val))});
        self.emit(rvv! {fmv.x.d x(Self::t2), f(ϕ(false_val))});
        self.emit(rvv! {and x(Self::t1), x(Self::t1), x(Self::t0)});
        self.emit(rvv! {not x(Self::t0), x(Self::t0)});
        self.emit(rvv! {and x(Self::t2), x(Self::t2), x(Self::t0)});
        self.emit(rvv! {or x(Self::t0), x(Self::t1), x(Self::t2)});
        self.emit(rvv! {fmv.d.x f(ϕ(dst)), x(Self::t0)});
    }

    /********************************************************/

    fn prologue_fast(&mut self, cap: u32, num_args: u32) {
        self.sub_stack(16);

        self.emit(rvv! {sd x(Self::ra), x(Self::sp), 0});
        self.emit(rvv! {sd x(MEM), x(Self::sp), 8});

        let stack_size = align_stack(self.reg_size() * cap);
        self.sub_stack(stack_size);

        self.emit(rvv! {mv x(MEM), x(Self::sp)});

        let num_args = num_args as i32;

        for i in 0..num_args {
            self.emit(rvv! {fsd f(Self::fa0 + i as u8), x(MEM), 8 * i});
        }
    }

    fn epilogue_fast(&mut self, cap: u32, idx_ret: i32) {
        self.emit(rvv! {fld f(Self::fa0), x(MEM), 8*idx_ret});

        let stack_size = align_stack(self.reg_size() * cap);
        self.add_stack(stack_size);

        self.emit(rvv! {ld x(Self::ra), x(Self::sp), 0});
        self.emit(rvv! {ld x(MEM), x(Self::sp), 8});

        self.add_stack(16);
        self.emit(rvv! {ret});
    }

    /*
     * MEM => first arg = mem if direct mode, otherwise null
     * STATES => second arg = states+obs if indirect mode, otherwise null
     * IDX => third arg = index if indirect mode
     * PARAMS => fourth arg = params
     */
    fn prologue_indirect(&mut self, cap: u32, count_states: usize, count_obs: usize) {
        self.sub_stack(64);

        self.emit(rvv! {sd x(Self::ra), x(Self::sp), 0});
        self.emit(rvv! {sd x(MEM), x(Self::sp), 8});
        self.emit(rvv! {sd x(STATES), x(Self::sp), 16});
        self.emit(rvv! {sd x(IDX), x(Self::sp), 24});
        self.emit(rvv! {sd x(PARAMS), x(Self::sp), 32});

        self.emit(rvv! {mv x(MEM), x(Self::a0)});
        self.emit(rvv! {mv x(STATES), x(Self::a1)});
        self.emit(rvv! {mv x(IDX), x(Self::a2)});
        self.emit(rvv! {mv x(PARAMS), x(Self::a3)});

        self.jump(
            "@main",
            0,
            |offset, _| rvv! {beq x(STATES), x(Self::zero), offset},
        );

        let size = align_stack((count_states + count_obs + 1) as u32 * self.reg_size());
        self.sub_stack(size);
        self.emit(rvv! {mv x(MEM), x(Self::sp)});
        self.emit(rvv! {slli x(IDX), x(IDX), 3});

        if count_states > 16 {
            self.emit(rvv! {mv x(Self::t1), x(MEM)});
            self.li(Self::t2, count_states as i32);

            self.set_label("@pro");
            self.prologue_inner();
            self.emit(rvv! {addi x(Self::t2), x(Self::t2), -1});

            self.jump("@pro", 0, |offset, _| {
                rvv! {bne x(Self::t2), x(Self::zero), offset}
            });
        } else if count_states > 0 {
            self.emit(rvv! {mv x(Self::t1), x(MEM)});

            for _ in 0..count_states {
                self.prologue_inner();
            }
        }

        // TODO: may save idx (RDX) as double in RBP + 8/32 * count_states

        self.set_label("@main");

        let stack_size = align_stack(self.reg_size() * cap);
        self.sub_stack(stack_size);
    }

    fn epilogue_indirect(&mut self, cap: u32, count_states: usize, count_obs: usize) {
        let stack_size = align_stack(self.reg_size() * cap);
        self.add_stack(stack_size);

        self.jump("@done", 0, |offset, _| {
            rvv! {beq x(STATES), x(Self::zero), offset}
        });

        if count_obs > 16 {
            self.li(Self::t1, 8 * (count_states + 1) as i32);
            self.emit(rvv! {add x(Self::t1), x(Self::t1), x(MEM)});
            self.li(Self::t2, count_obs as i32);

            self.set_label("@epi");
            self.epilogue_inner();
            self.emit(rvv! {addi x(Self::t2), x(Self::t2), -1});

            self.jump("@epi", 0, |offset, _| {
                rvv! {bne x(Self::t2), x(Self::zero), offset}
            });
        } else if count_obs > 0 {
            self.li(Self::t1, 8 * (count_states + 1) as i32);
            self.emit(rvv! {add x(Self::t1), x(Self::t1), x(MEM)});

            for _ in 0..count_obs {
                self.epilogue_inner();
            }
        }

        let size = align_stack((count_states + count_obs + 1) as u32 * self.reg_size());
        self.add_stack(size);

        self.set_label("@done");

        self.emit(rvv! {ld x(Self::ra), x(Self::sp), 0});
        self.emit(rvv! {ld x(MEM), x(Self::sp), 8});
        self.emit(rvv! {ld x(STATES), x(Self::sp), 16});
        self.emit(rvv! {ld x(IDX), x(Self::sp), 24});
        self.emit(rvv! {ld x(PARAMS), x(Self::sp), 32});

        self.add_stack(64);
        self.emit(rvv! {ret});
    }

    fn save_used_registers(&mut self, _used: &[u8]) {}

    fn load_used_registers(&mut self, _used: &[u8]) {}
}

impl RiscV {
    fn prologue_inner(&mut self) {
        self.emit(rvv! {ld x(Self::t0), x(STATES), 0});
        self.emit(rvv! {add x(Self::t0), x(Self::t0), x(IDX)});

        self.emit(rvv! {fld f(Self::fa0), x(Self::t0), 0});
        self.emit(rvv! {fsd f(Self::fa0), x(Self::t1), 0});

        self.emit(rvv! {addi x(STATES), x(STATES), 8});
        self.emit(rvv! {addi x(Self::t1), x(Self::t1), 8});
    }

    fn epilogue_inner(&mut self) {
        self.emit(rvv! {ld x(Self::t0), x(STATES), 0});
        self.emit(rvv! {add x(Self::t0), x(Self::t0), x(IDX)});

        self.emit(rvv! {fld f(Self::fa0), x(Self::t1), 0});
        self.emit(rvv! {fsd f(Self::fa0), x(Self::t0), 0});

        self.emit(rvv! {addi x(STATES), x(STATES), 8});
        self.emit(rvv! {addi x(Self::t1), x(Self::t1), 8});
    }
}

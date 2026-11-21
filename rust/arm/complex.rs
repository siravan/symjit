use anyhow::Result;

use crate::assembler::{Assembler, Jumper};
use crate::code::Func;
use crate::config::{Config, ABI_AREA};
use crate::generator::{Generator, GeneratorType};
use crate::symbol::Loc;
use crate::utils::{align_stack, Reg};

use super::*;

const REG_SIZE: u32 = 8;

// the constants of `Config::stable_math` (see `Complexifier::stable_exponent`)
const STABLE_INF: u64 = 0x7ff0_0000_0000_0000;
const STABLE_FLOOR: u64 = 0x0010_0000_0000_0000; // 2^-1022
const STABLE_MASK_FINITE: u64 = 0x7fe0_0000_0000_0000; // 2^1023
const STABLE_FLOOR_FINITE: u64 = 0x0020_0000_0000_0000; // 2^-1021

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
            self.emit(arm! {add x(0), x(STACK), #0});
            self.emit(arm! {eor x(1), x(1), x(1)});
            self.emit(arm! {eor x(2), x(2), x(2)});
            self.emit(arm! {add x(3), x(STACK), #ofs});
        } else {
            load_x_from_label(&mut self.a, 0, &format!("_env_{}_", op));
            self.emit(arm! {add x(1), x(STACK), #ofs});
            self.emit(arm! {movz x(2), #num_args});
            self.emit(arm! {add x(3), x(STACK), #0});
        }

        if op == "@self" {
            self.call_funclet("@self");
        } else {
            let label = format!("_func_{}_", op);
            load_long(&mut self.a, 9, &label);
            self.emit(arm! {blr x(9)});
        }

        self.load_stack(Reg::Ret, 0);

        Ok(())
    }

    /*********************** stable_math ***********************/

    /// `d(reg) := f64::from_bits(bits)` for `bits` with only the top 16 bits set.
    fn stable_bits(&mut self, reg: u8, bits: u64) {
        debug_assert!(bits & 0x0000_ffff_ffff_ffff == 0);
        self.emit(arm! {movz x(0), #0});
        self.emit(arm! {movk_lsl48 x(0), #(bits >> 48) as u32});
        self.emit(arm! {fmov d(reg), x(0)});
    }

    /// `d(T1) := p = (|re s|/2 + |im s|/2 + floor) & mask`, a power of two (see
    /// `Complexifier::stable_exponent`). `s` is a physical register; T2 is clobbered.
    fn stable_exponent(&mut self, s: u8, mask: u64, floor: u64) {
        self.emit(arm! {fabs q(T1), q(s)});
        self.emit(arm! {fmov q(T2), #0.5});
        self.emit(arm! {fmul q(T1), q(T1), q(T2)});
        self.emit(arm! {faddp d(T1), q(T1)});
        self.stable_bits(T2, floor);
        self.emit(arm! {fadd d(T1), d(T1), d(T2)});
        self.stable_bits(T2, mask);
        self.emit(arm! {and v(T1).8b, v(T1).8b, v(T2).8b});
    }

    /// `q(T2) := (k, k)` with `k = 0.5 / p`, which scales `max(|re s|, |im s|)` into [0.5, 2)
    /// (`p ^ inf` is `2 / p` for a power of two, see `Complexifier::stable_divisor_scale`).
    /// T0 is clobbered.
    fn stable_divisor_scale(&mut self, s: u8) {
        self.stable_exponent(s, STABLE_INF, STABLE_FLOOR);
        self.stable_bits(T2, STABLE_INF);
        self.emit(arm! {eor v(T2).8b, v(T1).8b, v(T2).8b});
        self.emit(arm! {fmov q(T0), #0.5});
        self.emit(arm! {fmul d(T2), d(T2), d(T0)});
        self.emit(arm! {fmul d(T2), d(T2), d(T0)});
        self.emit(arm! {dup q(T2), q(T2)[0]});
    }

    /// `divide` (the formula without FCMA) on both operands scaled by `stable_divisor_scale(s2)`;
    /// the scaling is exact. The fused products `y1 y2 + x1 x2` and `x2 y1 - x1 y2` are
    /// computed together, as `(x1 x2, x2 y1) + (y1, -x1) * y2`.
    fn stable_divide(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        let dst = ϕ(dst);

        self.stable_divisor_scale(ϕ(s2));
        self.emit(arm! {fmul q(T1), q(ϕ(s2)), q(T2)}); // (x2, y2) scaled
        self.emit(arm! {fmul q(T2), q(ϕ(s1)), q(T2)}); // (x1, y1) scaled

        self.emit(arm! {ext q(dst), q(T2), q(T2), #8}); // (y1, x1)
        self.stable_bits(T0, 0x8000_0000_0000_0000);
        self.emit(arm! {ext q(T0), q(T0), q(T0), #8}); // (0, -0)
        self.emit(arm! {eor v(dst).16b, v(dst).16b, v(T0).16b}); // (y1, -x1)
        self.emit(arm! {fmul q(T0), q(T2), d(T1)[0]}); // (x1 x2, y1 x2)
        self.emit(arm! {fmla q(T0), q(dst), d(T1)[1]}); // + (y1 y2, -x1 y2)

        self.emit(arm! {dup q(T2), q(T1)[1]});
        self.emit(arm! {fmul d(dst), d(T1), d(T1)});
        self.emit(arm! {fmadd d(dst), d(T2), d(T2), d(dst)}); // x2^2 + y2^2
        self.emit(arm! {dup q(dst), q(dst)[0]});
        self.emit(arm! {fdiv q(dst), q(T0), q(dst)});
    }

    /// `recip` of `s1 * k`: `(x k^2, -y k^2) / |s1 k|^2` (the products with `k` are exact
    /// unless they underflow, when the result is subnormal anyway).
    fn stable_recip(&mut self, dst: Reg, s1: Reg) {
        self.stable_divisor_scale(ϕ(s1));
        self.emit(arm! {fmul q(T1), q(ϕ(s1)), q(T2)}); // z' = z * k

        self.emit(arm! {fmul q(T0), q(T1), q(T1)});
        self.emit(arm! {faddp d(T0), q(T0)}); // |z'|^2
        self.emit(arm! {fmul q(T1), q(T1), q(T2)}); // z' * k

        self.emit(arm! {zip2 q(T2), q(T1), q(T1)});
        self.emit(arm! {fneg q(T2), q(T2)});
        self.emit(arm! {zip1 q(T2), q(T1), q(T2)});
        self.emit(arm! {dup q(T0), q(T0)[0]});
        self.emit(arm! {fdiv q(ϕ(dst)), q(T2), q(T0)});
    }

    /// `abs` of `s1 / p`, times `p`.
    fn stable_abs(&mut self, dst: Reg, s1: Reg) {
        self.stable_exponent(ϕ(s1), STABLE_MASK_FINITE, STABLE_FLOOR_FINITE);
        self.stable_bits(T2, STABLE_INF);
        self.emit(arm! {eor v(T2).8b, v(T1).8b, v(T2).8b});
        self.emit(arm! {fmov q(T0), #0.5});
        self.emit(arm! {fmul d(T2), d(T2), d(T0)}); // k = 1 / p
        self.emit(arm! {dup q(T2), q(T2)[0]});
        self.emit(arm! {fmul q(T2), q(ϕ(s1)), q(T2)});

        self.emit(arm! {fmul q(T2), q(T2), q(T2)});
        self.emit(arm! {eor v(ϕ(dst)).16b, v(ϕ(dst)).16b, v(ϕ(dst)).16b});
        self.emit(arm! {faddp d(ϕ(dst)), q(T2)});
        self.emit(arm! {fsqrt d(ϕ(dst)), d(ϕ(dst))});
        self.emit(arm! {fmul d(ϕ(dst)), d(ϕ(dst)), d(T1)});
    }

    /// The part of `@complex_root` (argument in `Temp`) that computes `d(T1) := t` and
    /// `d(T2) := im(z) / t / 2` (with `d(T0) = 0.5`), on `z / r^2` for a power of two
    /// `r ~ sqrt(max(|re z|, |im z|))`, as the default code does on `z`. `Ret` is a scratch
    /// register (the funclet writes it last).
    fn stable_root_body(&mut self) {
        let s1 = ϕ(Reg::Temp);
        let u = ϕ(Reg::Ret);

        self.stable_exponent(s1, STABLE_MASK_FINITE, STABLE_FLOOR_FINITE);
        self.emit(arm! {fsqrt d(T1), d(T1)});
        self.stable_bits(T2, STABLE_MASK_FINITE);
        self.emit(arm! {and v(T1).8b, v(T1).8b, v(T2).8b}); // r
        self.emit(arm! {fmul d(T2), d(T1), d(T1)});
        self.stable_bits(T0, STABLE_INF);
        self.emit(arm! {eor v(T2).8b, v(T2).8b, v(T0).8b});
        self.emit(arm! {fmov q(T0), #0.5});
        self.emit(arm! {fmul d(T2), d(T2), d(T0)}); // k = 1 / r^2
        self.emit(arm! {dup q(T2), q(T2)[0]});
        self.emit(arm! {fmul q(u), q(s1), q(T2)}); // z' = z * k

        self.emit(arm! {fmul q(T2), q(u), q(u)});
        self.emit(arm! {faddp d(T2), q(T2)});
        self.emit(arm! {fsqrt d(T2), d(T2)});
        self.emit(arm! {fabs d(T0), d(u)});
        self.emit(arm! {fadd d(T2), d(T2), d(T0)});
        self.emit(arm! {fmov d(T0), #0.5});
        self.emit(arm! {fmul d(T2), d(T2), d(T0)});
        self.emit(arm! {fsqrt d(T2), d(T2)});
        self.emit(arm! {fmul d(T1), d(T2), d(T1)}); // t = t' * r

        self.emit(arm! {zip2 q(T2), q(s1), q(s1)});
        self.emit(arm! {fdiv d(T2), d(T2), d(T1)});
        self.emit(arm! {fmul d(T2), d(T2), d(T0)});
    }

    #[cfg(target_arch = "aarch64")]
    fn supports_fcma(&self) -> bool {
        std::arch::is_aarch64_feature_detected!("fcma")
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn supports_fcma(&self) -> bool {
        false
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

    fn what(&self) -> GeneratorType {
        GeneratorType::ArmComplex
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

    fn mark_data(&mut self) {
        self.a.set_label("@data");
    }

    fn take_labels(&mut self) -> Vec<(usize, String)> {
        self.a.take_labels()
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
        load_q_from_mem(&mut self.a, ϕ(dst), MEM, idx / 2);
    }

    fn save_mem(&mut self, dst: Reg, idx: u32) {
        save_q_to_mem(&mut self.a, ϕ(dst), MEM, idx / 2);
    }

    fn save_mem_result(&mut self, idx: u32) {
        self.save_mem(Reg::Ret, idx);
    }

    fn load_param(&mut self, dst: Reg, idx: u32) {
        load_q_from_mem(&mut self.a, ϕ(dst), PARAMS, idx / 2);
    }

    fn load_stack(&mut self, dst: Reg, idx: u32) {
        load_q_from_mem(&mut self.a, ϕ(dst), STACK, idx / 2);
    }

    fn save_stack(&mut self, dst: Reg, idx: u32) {
        save_q_to_mem(&mut self.a, ϕ(dst), STACK, idx / 2);
    }

    fn load_mem_complex(&mut self, _xd: Reg, _yd: Reg, _idx: u32) {}

    fn save_mem_complex(&mut self, _xs: Reg, _ys: Reg, _idx: u32) {}

    fn load_param_complex(&mut self, _xd: Reg, _yd: Reg, _idx: u32) {}

    fn load_stack_complex(&mut self, _xd: Reg, _yd: Reg, _idx: u32) {}

    fn save_stack_complex(&mut self, _xs: Reg, _ys: Reg, _idx: u32) {}

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
                load_c_from_loc(a, 0, src);
                save_c_to_loc(a, 0, dst);
            },
            |a, arg, src| {
                load_c_from_loc(a, arg, src);
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
                emit(a, arm! {lsr x(8), x(8), #1});
                emit(a, arm! {ldr q(arg), [x(STACK), x(8), lsl #4]});
            },
            |a, arg, dst| {
                save_c_to_loc(a, arg, dst);
            },
        );
    }

    fn load_args_complex(&mut self, _locs: Vec<Loc>, _ultra: bool) {
        unreachable!()
    }

    fn save_args_complex(&mut self, _num_args: u8, _ultra: bool) {
        unreachable!()
    }

    fn neg(&mut self, dst: Reg, s1: Reg) {
        self.emit(arm! {fneg q(ϕ(dst)), q(ϕ(s1))});
    }

    fn sign(&mut self, dst: Reg, s1: Reg) {
        self.emit(arm! {fmov q(T2), #0.0});
        self.emit(arm! {fneg q(T2), q(T2)});
        self.emit(arm! {and v(ϕ(dst)).16b, v(ϕ(s1)).16b, v(T2).16b});
    }

    fn abs(&mut self, dst: Reg, s1: Reg) {
        if self.config.stable_math() {
            return self.stable_abs(dst, s1);
        }
        self.emit(arm! {fmul q(T2), q(ϕ(s1)), q(ϕ(s1))});
        self.emit(arm! {eor v(ϕ(dst)).16b, v(ϕ(dst)).16b, v(ϕ(dst)).16b});
        self.emit(arm! {faddp d(ϕ(dst)), q(T2)});
        self.emit(arm! {fsqrt d(ϕ(dst)), d(ϕ(dst))});
    }

    fn abs2(&mut self, dst: Reg, s1: Reg) {
        self.emit(arm! {fmul q(T2), q(ϕ(s1)), q(ϕ(s1))});
        self.emit(arm! {eor v(ϕ(dst)).16b, v(ϕ(dst)).16b, v(ϕ(dst)).16b});
        self.emit(arm! {faddp d(ϕ(dst)), q(T2)});
    }

    fn times_i(&mut self, dst: Reg, s1: Reg) {
        self.emit(arm! {fneg q(T1), q(ϕ(s1))});
        self.emit(arm! {ext q(ϕ(dst)), q(T1), q(ϕ(s1)), #8});
    }

    fn times_neg_i(&mut self, dst: Reg, s1: Reg) {
        self.emit(arm! {fneg d(T1), d(ϕ(s1))});
        self.emit(arm! {ext q(ϕ(dst)), q(ϕ(s1)), q(T1), #8});
    }

    fn root(&mut self, dst: Reg, s1: Reg) {
        self.fmov(Reg::Temp, s1);
        self.call_funclet("@complex_root");
        self.fmov(dst, Reg::Ret);

        if !self.a.has_label("@complex_root") {
            self.branch("@jump_over_complex_root");

            self.set_label("@complex_root");

            let s1 = ϕ(Reg::Temp);
            let dst = ϕ(Reg::Ret);

            if self.config.stable_math() {
                self.stable_root_body();
            } else {
                self.emit(arm! {fmul q(T1), q(s1), q(s1)});
                self.emit(arm! {faddp d(T1), q(T1)});
                self.emit(arm! {fsqrt d(T1), d(T1)});
                self.emit(arm! {fabs d(T2), d(s1)});
                self.emit(arm! {fadd d(T1), d(T1), d(T2)});
                self.emit(arm! {fmov d(T0), #0.5});
                self.emit(arm! {fmul d(T1), d(T1), d(T0)});
                self.emit(arm! {fsqrt d(T1), d(T1)});

                self.emit(arm! {zip2 q(T2), q(s1), q(s1)});
                self.emit(arm! {fdiv d(T2), d(T2), d(T1)});
                self.emit(arm! {fmul d(T2), d(T2), d(T0)});
            }

            self.emit(arm! {fcmeq d(T0), d(T2), d(T2)});
            self.emit(arm! {and v(T2).8b, v(T2).8b, v(T0).8b});

            self.emit(arm! {fcmp d(s1), #0.0});
            self.emit(arm! {fcsel d(T0), d(T1), d(T2), ge});
            self.emit(arm! {fcsel d(T2), d(T1), d(T2), lt});
            self.emit(arm! {fabs d(T1), d(T0)});
            self.emit(arm! {eor v(T0).8b, v(T0).8b, v(T1).8b});
            self.emit(arm! {eor v(T2).8b, v(T0).8b, v(T2).8b});

            self.emit(arm! {zip1 q(dst), q(T1), q(T2)});
            self.ret();

            self.set_label("@jump_over_complex_root");
        }
    }

    fn real_root(&mut self, dst: Reg, s1: Reg) {
        self.emit(arm! {eor v(T1).16b, v(T1).16b, v(T1).16b});
        self.emit(arm! {fsqrt q(ϕ(dst)), q(ϕ(s1))});
        self.emit(arm! {zip1 q(ϕ(dst)), q(ϕ(dst)), q(T1)});
    }

    fn recip(&mut self, dst: Reg, s1: Reg) {
        if self.config.stable_math() {
            return self.stable_recip(dst, s1);
        }
        self.emit(arm! {zip2 q(T1), q(ϕ(s1)), q(ϕ(s1))});
        self.emit(arm! {fneg q(T1), q(T1)});
        self.emit(arm! {zip1 q(T1), q(ϕ(s1)), q(T1)});

        self.emit(arm! {fmul q(T2), q(ϕ(s1)), q(ϕ(s1))});
        self.emit(arm! {faddp d(T2), q(T2)});
        self.emit(arm! {dup q(T2), q(T2)[0]});
        self.emit(arm! {fdiv q(ϕ(dst)), q(T1), q(T2)});
    }

    fn half(&mut self, dst: Reg, s1: Reg) {
        self.emit(arm! {fmov q(T2), #0.5});
        self.emit(arm! {fmul q(ϕ(dst)), q(ϕ(s1)), q(T2)});
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
        self.emit(arm! {frintm q(T2), q(ϕ(s1))});
        self.emit(arm! {fsub q(ϕ(dst)), q(ϕ(s1)), q(T2)});
    }

    fn plus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(arm! {fadd q(ϕ(dst)), q(ϕ(s1)), q(ϕ(s2))});
    }

    fn minus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(arm! {fsub q(ϕ(dst)), q(ϕ(s1)), q(ϕ(s2))});
    }

    fn times(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        if self.supports_fcma() {
            // physical registers: `Reg::Left` and `Reg::Ret` are both q0
            if ϕ(dst) != ϕ(s1) && ϕ(dst) != ϕ(s2) {
                self.emit(arm! {fmov q(ϕ(dst)), #0.0});
                self.emit(arm! {fcmla q(ϕ(dst)), q(ϕ(s1)), q(ϕ(s2)), #0});
                self.emit(arm! {fcmla q(ϕ(dst)), q(ϕ(s1)), q(ϕ(s2)), #90});
            } else {
                self.emit(arm! {fmov q(T1), #0.0});
                self.emit(arm! {fcmla q(T1), q(ϕ(s1)), q(ϕ(s2)), #0});
                self.emit(arm! {fcmla q(T1), q(ϕ(s1)), q(ϕ(s2)), #90});
                self.emit(arm! {fmov q(ϕ(dst)), q(T1)});
            }
        } else {
            self.emit(arm! { fmul d(T0), d(ϕ(s1)), d(ϕ(s2)) }); // T0 = x1*x2
            self.emit(arm! { dup q(T1), q(ϕ(s1))[1] }); // T1 = y1
            self.emit(arm! { dup q(T2), q(ϕ(s2))[1] }); // T2 = y2
            self.emit(arm! { fmsub d(T0), d(T1), d(T2), d(T0) }); // T0 = x1*x2 - y1*y2
            self.emit(arm! { fmul d(T2), d(ϕ(s1)), d(T2) }); // T2 = x1*y2
            self.emit(arm! { fmadd d(T2), d(ϕ(s2)), d(T1), d(T2) }); // T2 = x1*y2 + x2*y1

            self.emit(arm! { zip1 q(ϕ(dst)), q(T0), q(T2) }); // dst = (x1*x2 - y1*y2) + (x1*y2 + x2*y1)*im
        }
    }

    fn divide(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        if self.config.stable_math() {
            return self.stable_divide(dst, s1, s2);
        }
        if self.config.compress() && self.supports_fcma() {
            self.emit(arm! {eor v(T1).16b, v(T1).16b, v(T1).16b});
            self.emit(arm! {fcmla q(T1), q(ϕ(s2)), q(ϕ(s1)), #0});
            self.emit(arm! {fcmla q(T1), q(ϕ(s2)), q(ϕ(s1)), #270});

            self.emit(arm! {fmul q(T2), q(ϕ(s2)), q(ϕ(s2))});
            self.emit(arm! {faddp d(T2), q(T2)});
            self.emit(arm! {dup q(T2), q(T2)[0]});
            self.emit(arm! {fdiv q(ϕ(dst)), q(T1), q(T2)});
        } else {
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
    }

    fn times_complex(
        &mut self,
        _xd: Reg,
        _yd: Reg,
        _x1: Reg,
        _y1: Reg,
        _x2: Reg,
        _y2: Reg,
    ) -> bool {
        false
    }

    fn divide_complex(
        &mut self,
        _xd: Reg,
        _yd: Reg,
        _x1: Reg,
        _y1: Reg,
        _x2: Reg,
        _y2: Reg,
    ) -> bool {
        false
    }

    fn support_times2(&self) -> bool {
        false
    }

    fn times2_loc(&mut self, _d1: Reg, _s1: Reg, _l1: Loc, _d2: Reg, _s2: Reg, _l2: Loc) {
        unreachable!()
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
        self.emit(arm! {fcmeq q(ϕ(dst)), q(ϕ(s1)), q(ϕ(s2))});
        self.emit(arm! {ext q(T1), q(ϕ(dst)), q(ϕ(dst)), #8}); // flipping lanes
        self.emit(arm! {and v(ϕ(dst)).16b, v(ϕ(dst)).16b, v(T1).16b});
    }

    fn neq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.emit(arm! {fcmeq q(ϕ(dst)), q(ϕ(s1)), q(ϕ(s2))});
        self.emit(arm! {not v(ϕ(dst)).16b, v(ϕ(dst)).16b});
        self.emit(arm! {ext q(T1), q(ϕ(dst)), q(ϕ(dst)), #8});
        self.emit(arm! {orr v(ϕ(dst)).16b, v(ϕ(dst)).16b, v(T1).16b});
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
        if s1 == s2 {
            self.emit(arm! {movi v(ϕ(dst)).16b, #0});
        } else {
            self.emit(arm! {eor v(ϕ(dst)).16b, v(ϕ(s1)).16b, v(ϕ(s2)).16b});
        }
    }

    fn not(&mut self, dst: Reg, s1: Reg) {
        self.emit(arm! {not v(ϕ(dst)).16b, v(ϕ(s1)).16b});
    }

    fn fused_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        if self.supports_fcma() {
            if dst == s3 && dst != s1 && dst != s2 {
                self.emit(arm! {fcmla q(ϕ(dst)), q(ϕ(s1)), q(ϕ(s2)), #0});
                self.emit(arm! {fcmla q(ϕ(dst)), q(ϕ(s1)), q(ϕ(s2)), #90});
            } else {
                self.emit(arm! {fmov q(T1), q(ϕ(s3))});
                self.emit(arm! {fcmla q(T1), q(ϕ(s1)), q(ϕ(s2)), #0});
                self.emit(arm! {fcmla q(T1), q(ϕ(s1)), q(ϕ(s2)), #90});
                self.emit(arm! {fmov q(ϕ(dst)), q(T1)});
            }
        } else if s3 != Reg::Temp {
            self.times(Reg::Temp, s1, s2);
            self.plus(dst, Reg::Temp, s3);
        } else {
            self.times(Reg::Ret, s1, s2);
            self.plus(dst, Reg::Ret, s3);
        }
    }

    // fused_mul_sub is s1 * s2 - s3, corresponding to fnmsub in aarch64
    // and vmsub... in amd64
    fn fused_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        if self.supports_fcma() {
            if dst == s3 && dst != s1 && dst != s2 {
                self.emit(arm! {fneg q(ϕ(dst)), q(ϕ(s3))});
                self.emit(arm! {fcmla q(ϕ(dst)), q(ϕ(s1)), q(ϕ(s2)), #0});
                self.emit(arm! {fcmla q(ϕ(dst)), q(ϕ(s1)), q(ϕ(s2)), #90});
            } else {
                self.emit(arm! {fneg q(T1), q(ϕ(s3))});
                self.emit(arm! {fcmla q(T1), q(ϕ(s1)), q(ϕ(s2)), #0});
                self.emit(arm! {fcmla q(T1), q(ϕ(s1)), q(ϕ(s2)), #90});
                self.emit(arm! {fmov q(ϕ(dst)), q(T1)});
            }
        } else if s3 != Reg::Temp {
            self.times(Reg::Temp, s1, s2);
            self.minus(dst, Reg::Temp, s3);
        } else {
            self.times(Reg::Ret, s1, s2);
            self.minus(dst, Reg::Ret, s3);
        }
    }

    // fused_neg_mul_add is s3 - s1 * s2, corresponding to fmsub in aarch64
    // and vnmadd... in amd64
    fn fused_neg_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        if self.supports_fcma() {
            self.emit(arm! {fneg q(T1), q(ϕ(s3))});
            self.emit(arm! {fcmla q(T1), q(ϕ(s1)), q(ϕ(s2)), #0});
            self.emit(arm! {fcmla q(T1), q(ϕ(s1)), q(ϕ(s2)), #90});
            self.emit(arm! {fneg q(ϕ(dst)), q(T1)});
        } else if s3 != Reg::Temp {
            self.times(Reg::Temp, s1, s2);
            self.minus(dst, s3, Reg::Temp);
        } else {
            self.times(Reg::Ret, s1, s2);
            self.minus(dst, s3, Reg::Ret);
        }
    }

    // fused_neg_mul_sub is -s3 - s1 * s2, corresponding to fnmadd in aarch64
    // and vnmsub... in amd64
    fn fused_neg_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        if self.supports_fcma() {
            self.emit(arm! {fmov q(T1), q(ϕ(s3))});
            self.emit(arm! {fcmla q(T1), q(ϕ(s1)), q(ϕ(s2)), #0});
            self.emit(arm! {fcmla q(T1), q(ϕ(s1)), q(ϕ(s2)), #90});
            self.emit(arm! {fneg q(ϕ(dst)), q(T1)});
        } else if s3 != Reg::Temp {
            self.times(Reg::Temp, s1, s2);
            self.plus(dst, Reg::Temp, s3);
            self.neg(dst, dst);
        } else {
            self.times(Reg::Ret, s1, s2);
            self.plus(dst, Reg::Ret, s3);
            self.neg(dst, dst);
        }
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
        self.emit(arm! {add x(0), x(STACK), #0});

        if num_args == 2 {
            self.save_stack(Reg::Right, 0);
        }

        self.emit(arm! {dup q(1), q(ϕ(Reg::Left))[1]});

        self.call(op, num_args)?;

        self.load_stack(Reg::Ret, 0);
        Ok(())
    }

    fn call_funclet(&mut self, label: &str) {
        call_funclet(&mut self.a, label);
    }

    fn ret(&mut self) {
        self.emit(arm! {ret});
    }

    fn ifelse(&mut self, dst: Reg, true_val: Reg, false_val: Reg, idx: u32) {
        if true_val == false_val {
            self.fmov(dst, true_val);
        } else if dst != true_val && dst != false_val {
            self.load_stack(dst, idx);
            self.emit(arm! {bsl v(ϕ(dst)).16b, v(ϕ(true_val)).16b, v(ϕ(false_val)).16b});
        } else {
            self.load_stack(Reg::Temp, idx);
            self.emit(arm! {bsl v(ϕ(Reg::Temp)).16b, v(ϕ(true_val)).16b, v(ϕ(false_val)).16b});
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

        for i in (0..count_states).step_by(2) {
            self.emit(arm! {str q(i), [x(MEM), #8*i]});
        }
    }

    fn epilogue_fast(
        &mut self,
        _cap: usize,
        _count_states: usize,
        _count_obs: usize,
        idx_ret: i32,
    ) {
        self.emit(arm! {ldr q(0), [x(MEM), #8*idx_ret]});

        self.emit(arm! {mov sp, x(FP)});
        self.emit(arm! {ldp lr, x(FP), [sp, #0]});
        self.emit(arm! {ldp x(MEM), x(STACK), [sp, #16]});
        self.emit(arm! {add sp, sp, #32});
        self.emit(arm! {eor x(0), x(0), x(0)});
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
        self.emit(arm! {ret});
    }

    fn save_used_registers(&mut self, used: &[Reg]) {
        for r in used {
            let phys_reg = ϕ(*r);
            if (8..=15).contains(&phys_reg) {
                // self.save_stack(*r, phys_reg as u32);
                save_d_to_mem(&mut self.a, phys_reg, STACK, phys_reg as u32);
            }
        }
    }

    fn load_used_registers(&mut self, used: &[Reg]) {
        self.set_label("@success");
        self.emit(arm! {eor x(0), x(0), x(0)});
        self.set_label("@epilogue");

        for r in used {
            let phys_reg = ϕ(*r);
            if (8..=15).contains(&phys_reg) {
                // self.load_stack(*r, phys_reg as u32);
                load_d_from_mem(&mut self.a, phys_reg, STACK, phys_reg as u32);
            }
        }
    }
}

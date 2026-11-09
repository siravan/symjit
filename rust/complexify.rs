use anyhow::{anyhow, Result};
use std::collections::HashSet;

use super::code::Func;
use super::config::Config;
use super::generator::{Generator, GeneratorType};
use super::mir::Mir;
use super::symbol::Loc;
use super::utils::Reg;

fn re(reg: Reg) -> Reg {
    match reg {
        Reg::Ret => Reg::Ret,
        Reg::Temp => Reg::Gen(0),
        Reg::Left => Reg::Left,
        Reg::Right => Reg::Gen(0),
        Reg::Gen(r) => Reg::Gen(4 + 2 * r),
        Reg::Static(_) => unreachable!(),
    }
}

fn im(reg: Reg) -> Reg {
    match reg {
        Reg::Ret => Reg::Temp,
        Reg::Temp => Reg::Gen(1),
        Reg::Left => Reg::Temp,
        Reg::Right => Reg::Gen(1),
        Reg::Gen(r) => Reg::Gen(4 + 2 * r + 1),
        Reg::Static(_) => unreachable!(),
    }
}

fn ϕ(r: Reg) -> usize {
    match r {
        Reg::Ret => 0,
        Reg::Temp => 1,
        Reg::Left => 0,
        Reg::Right => 1,
        Reg::Gen(dst) => dst as usize + 2,
        Reg::Static(..) => panic!("passing static registers to codegen"),
    }
}

enum Types {
    RR,
    CR,
    RC,
    CC,
}

pub struct Complexifier {
    mir: Mir,
    real_locs: HashSet<Loc>,
    real_regs: [bool; 32],
    root_defined: bool,
}

impl Complexifier {
    pub fn new(reals: &HashSet<Loc>, config: Config) -> Complexifier {
        Complexifier {
            mir: Mir::new(config),
            // Real declarations have the same scalar offsets as MIR loads.
            real_locs: reals.clone(),
            real_regs: [false; 32],
            root_defined: false,
        }
    }

    pub fn complexify(&mut self, mir: &Mir) -> Result<Mir> {
        if mir.code.is_complexified {
            return Ok(mir.clone());
        }

        self.mir.consts = mir.consts.clone();
        // self.mir.labels = mir.labels.clone();
        mir.rerun(self)?;
        self.mir.populate_labels();
        // Not `std::mem::take`: its `Mir::default()` calls `Config::default()`, which
        // reads `$SYMJIT_TOML` or `./symjit.toml`, on every load of a saved application.
        let config = self.mir.config.clone();
        Ok(std::mem::replace(&mut self.mir, Mir::new(config)))
    }

    // temporary registers
    const T0: Reg = Reg::Gen(2);
    const T1: Reg = Reg::Gen(3);

    fn is_real_loc(&self, loc: Loc) -> bool {
        self.real_locs.contains(&loc)
    }

    fn is_real_reg(&self, s1: Reg) -> bool {
        self.real_regs[ϕ(s1)]
    }

    #[allow(unused)]
    fn set_loc_real(&mut self, loc: Loc) {
        self.real_locs.insert(loc);
    }

    fn set_loc_complex(&mut self, loc: Loc) {
        self.real_locs.remove(&loc);
    }

    fn set_reg_real(&mut self, dst: Reg) {
        self.real_regs[ϕ(dst)] = true;
    }

    fn set_reg_complex(&mut self, dst: Reg) {
        self.real_regs[ϕ(dst)] = false;
    }

    fn copy_real(&mut self, dst: Reg, s1: Reg) {
        self.real_regs[ϕ(dst)] = self.is_real_reg(s1);
    }

    fn promote_real(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.real_regs[ϕ(dst)] = self.is_real_reg(s1) && self.is_real_reg(s2);
    }

    fn ensure_complex(&mut self, dst: Reg) {
        if self.is_real_reg(dst) {
            self.mir.xor(im(dst), im(dst), im(dst));
        }
        self.set_reg_complex(dst);
    }

    /****************** stable_math ******************/

    /// The index of constant `val`, appended to the constants if it is not there yet.
    /// `Application::with_mir` copies the appended constants to the program.
    fn stable_const(&mut self, val: f64) -> u32 {
        let bits = val.to_bits();
        if let Some(idx) = self.mir.consts.iter().position(|c| c.to_bits() == bits) {
            return idx as u32;
        }
        self.mir.consts.push(val);
        (self.mir.consts.len() - 1) as u32
    }

    /// `T0 := s & mask` with `s = |x|/2 + |y|/2 + floor`: a power of two in (s/2, s] for
    /// `mask = inf` and in (s/4, s] for `mask = 2^1023` (which also maps inf and NaN to a
    /// finite value). `x` and `y` are real registers; T1 is clobbered.
    fn stable_exponent(&mut self, x: Reg, y: Reg, mask: f64, floor: f64) {
        self.mir.abs(Self::T0, x);
        self.mir.half(Self::T0, Self::T0);
        self.mir.abs(Self::T1, y);
        self.mir.half(Self::T1, Self::T1);
        self.mir.plus(Self::T0, Self::T0, Self::T1);
        let idx = self.stable_const(floor);
        self.mir.load_const(Self::T1, idx);
        self.mir.plus(Self::T0, Self::T0, Self::T1);
        let idx = self.stable_const(mask);
        self.mir.load_const(Self::T1, idx);
        self.mir.and(Self::T0, Self::T0, Self::T1);
    }

    /// `T0 := 0.5 / p` for the power of two `p` of `stable_exponent(x, y, inf, 2^-1022)`:
    /// the exact factor that scales `max(|x|, |y|)` into [0.5, 2). For a power of two
    /// `p = 2^e` (exponent field `e + 1023`), `p ^ inf` flips the exponent field to
    /// `1024 - e`, which is `2 / p`: no division.
    fn stable_divisor_scale(&mut self, x: Reg, y: Reg) {
        self.stable_exponent(x, y, f64::INFINITY, f64::MIN_POSITIVE);
        self.mir.xor(Self::T0, Self::T0, Self::T1); // 2 / p (T1 is still the mask, inf)
        let idx = self.stable_const(0.25);
        self.mir.load_const(Self::T1, idx);
        self.mir.times(Self::T0, Self::T0, Self::T1);
    }

    /// `dst := s1 / s2` for a complex `s2`, with both operands scaled by `stable_divisor_scale`
    /// (exact, so in the normal range the result has the bits of the default code). `s1` may
    /// be real.
    ///
    /// The scaled operands a', b' (numerator) and c', d' (divisor) go to `Gen(0)`, `Gen(1)`
    /// and to `dst`. A `dst` that is not a `Gen` pair is `Left`, whose parts are the machine
    /// registers `Ret` and `Temp`: a quotient passed straight to a call by the indirect
    /// translator at opt level >= 1. Then the numerator goes to `dst` and the divisor to
    /// `Gen(0)`, `Gen(1)`, because the generators' `divide_complex` (non-fast complex) uses
    /// `Reg::Temp` as scratch after it has read the first operand but before it reads the second
    /// (tests/stable_divide_call_argument.rs).
    fn stable_divide(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        let (a, b, c, d) = (re(s1), im(s1), re(s2), im(s2));
        let (t0, t1) = (Self::T0, Self::T1);
        let (a_, b_, c_, d_) = if !matches!(dst, Reg::Gen(_)) {
            (re(dst), im(dst), Reg::Gen(0), Reg::Gen(1))
        } else {
            (Reg::Gen(0), Reg::Gen(1), re(dst), im(dst))
        };

        assert!(s1 != Reg::Ret && s1 != Reg::Temp);
        assert!(s2 != Reg::Ret && s2 != Reg::Temp);

        self.stable_divisor_scale(c, d); // T0 = k

        if self.is_real_reg(s1) {
            self.mir.times(a_, a, t0); // a'
            self.mir.times(c_, c, t0); // c'
            self.mir.times(d_, d, t0); // d'

            self.mir.times(t1, c_, c_);
            self.mir.times(b_, a_, d_); // a'd'
            self.mir.times(a_, a_, c_); // a'c'
            self.mir.fused_mul_add(t1, d_, d_, t1); // c'^2 + d'^2
            self.mir.neg(b_, b_);
            self.mir.divide(im(dst), b_, t1);
            self.mir.divide(re(dst), a_, t1);
        } else {
            self.mir.times(a_, a, t0); // a'
            self.mir.times(b_, b, t0); // b'
            self.mir.times(c_, c, t0); // c'
            self.mir.times(d_, d, t0); // d'

            self.mir.divide_complex(re(dst), im(dst), a_, b_, c_, d_);
        }

        self.set_reg_complex(dst);
    }

    /// `dst := 1 / s1` for a complex `s1`: `conj(s1') / |s1'|^2 * k` with `s1' = s1 * k`.
    fn stable_recip(&mut self, dst: Reg, s1: Reg) {
        let (c, d) = (re(s1), im(s1));
        let (t0, t1) = (Self::T0, Self::T1);
        let (c_, d_) = (Reg::Gen(0), Reg::Gen(1));

        self.stable_divisor_scale(c, d); // T0 = k
        self.mir.times(c_, c, t0); // c'
        self.mir.times(d_, d, t0); // d'
        self.mir.times(t1, c_, c_);

        self.mir.fused_mul_add(t1, d_, d_, t1); // c'^2 + d'^2

        self.mir.divide(c_, c_, t1);
        self.mir.divide(d_, d_, t1);
        self.mir.neg(d_, d_);
        self.mir.times(re(dst), c_, t0);
        self.mir.times(im(dst), d_, t0);

        self.set_reg_complex(dst);
    }

    /// `dst := |s1|` for a complex `s1`, scaled by a power of two `p` near `max(|re|, |im|)`.
    fn stable_abs(&mut self, dst: Reg, s1: Reg) {
        let (a, b) = (re(s1), im(s1));
        let (t0, b_, a_, k) = (Self::T0, Self::T1, Reg::Ret, Self::T1);

        assert!(s1 != Reg::Ret);

        self.stable_exponent(a, b, Self::EXP_MASK_FINITE, Self::FLOOR_FINITE); // T0 = p
        let idx = self.stable_const(f64::INFINITY);
        self.mir.load_const(k, idx);
        self.mir.xor(k, t0, k);
        self.mir.half(k, k); // k = 1 / p (see `stable_divisor_scale`)
        self.mir.times(a_, a, k); // x'
        self.mir.times(b_, b, k); // y'; note k and b_ use the same register
        self.mir.times(a_, a_, a_);
        self.mir.times(b_, b_, b_);
        self.mir.plus(a_, a_, b_); // as the default code (no FMA): the same bits
        self.mir.root(a_, a_);
        self.mir.times(re(dst), a_, t0);
        self.mir.xor(im(dst), im(dst), im(dst));

        self.set_reg_real(dst);
    }

    /// The part of the `@complex_root` funclet that computes `T0 := sqrt((|z| + |re z|) / 2)`
    /// for `z` in `Temp`, on `z / r^2` with a power of two `r ~ sqrt(max(|re|, |im|))`
    /// (exact, so the bits are those of the default code in the normal range). `re(Temp)`
    /// is free afterwards: the funclet only needs `im(Temp)` and the sign mask on the stack.
    fn stable_root_magnitude(&mut self) {
        let (x, y) = (re(Reg::Temp), im(Reg::Temp));
        let (t0, t1, u) = (Self::T0, Self::T1, Reg::Ret);

        self.stable_exponent(x, y, Self::EXP_MASK_FINITE, Self::FLOOR_FINITE); // T0 = p
        self.mir.root(t0, t0);
        self.mir.and(t0, t0, t1); // r (T1 still holds the mask)
        self.mir.times(t1, t0, t0);
        let idx = self.stable_const(f64::INFINITY);
        self.mir.load_const(u, idx);
        self.mir.xor(t1, t1, u);
        self.mir.half(t1, t1); // k = 1 / r^2 (see `stable_divisor_scale`)
        self.mir.times(u, x, t1); // x'
        self.mir.times(t1, y, t1); // y'
        self.mir.times(x, u, u);
        self.mir.fused_mul_add(x, t1, t1, x);
        self.mir.root(x, x);
        self.mir.abs(t1, u);
        self.mir.plus(x, x, t1);
        self.mir.half(x, x);
        self.mir.root(x, x);
        self.mir.times(t0, x, t0); // t = t' * r
    }

    // 2^1023: masks the exponent field without its lowest bit, so inf and NaN give 2^1023
    const EXP_MASK_FINITE: f64 = f64::from_bits(0x7fe0_0000_0000_0000);
    // 2^-1021: with `EXP_MASK_FINITE`, the smallest exponent is not zero
    const FLOOR_FINITE: f64 = f64::from_bits(0x0020_0000_0000_0000);

    fn types(&self, s1: Reg, s2: Reg) -> Types {
        if self.is_real_reg(s1) && self.is_real_reg(s2) {
            Types::RR
        } else if self.is_real_reg(s1) && !self.is_real_reg(s2) {
            Types::RC
        } else if !self.is_real_reg(s1) && self.is_real_reg(s2) {
            Types::CR
        } else {
            Types::CC
        }
    }
}

impl Generator for Complexifier {
    fn count_shadows(&self) -> u8 {
        0
    }

    fn three_address(&self) -> bool {
        true
    }

    fn bytes(&mut self) -> Vec<u8> {
        Vec::new()
    }

    fn what(&self) -> GeneratorType {
        GeneratorType::Complexifier
    }

    fn seal(&mut self) {}
    fn align(&mut self) {}

    fn set_label(&mut self, label: &str) {
        self.mir.set_label(label);
    }

    fn branch(&mut self, label: &str) {
        self.mir.branch(label);
    }

    fn branch_if(&mut self, cond: Reg, label: &str, is_else: bool) {
        self.mir.branch_if(re(cond), label, is_else);
    }

    /***********************************/
    fn fmov(&mut self, dst: Reg, s1: Reg) {
        self.mir.fmov(re(dst), re(s1));

        if !self.is_real_reg(s1) {
            self.mir.fmov(im(dst), im(s1));
        }

        self.copy_real(dst, s1);
    }

    fn fxchg(&mut self, s1: Reg, s2: Reg) {
        self.mir.fxchg(re(s1), re(s2));

        if !self.is_real_reg(s1) || !self.is_real_reg(s2) {
            self.ensure_complex(s1);
            self.ensure_complex(s2);
            self.mir.fxchg(im(s1), im(s2));
        }
    }

    fn load_const(&mut self, dst: Reg, idx: u32) {
        self.mir.load_const(re(dst), idx);
        self.set_reg_real(dst);
    }

    fn load_mem(&mut self, dst: Reg, idx: u32) {
        // self.mir.load_mem(re(dst), idx);
        // self.mir.load_mem(im(dst), idx + 1);
        self.mir.load_mem_complex(re(dst), im(dst), idx);
        self.set_reg_complex(dst);
    }

    fn save_mem(&mut self, s1: Reg, idx: u32) {
        self.ensure_complex(s1);
        // self.mir.save_mem(re(s1), idx);
        // self.mir.save_mem(im(s1), idx + 1);
        self.mir.save_mem_complex(re(s1), im(s1), idx);
    }

    fn load_param(&mut self, dst: Reg, idx: u32) {
        if self.is_real_loc(Loc::Param(idx)) {
            self.mir.load_param(re(dst), idx);
            self.set_reg_real(dst);
        } else {
            // self.mir.load_param(re(dst), idx);
            // self.mir.load_param(im(dst), idx + 1);
            self.mir.load_param_complex(re(dst), im(dst), idx);
            self.set_reg_complex(dst);
        }
    }

    fn load_stack(&mut self, dst: Reg, idx: u32) {
        if self.is_real_loc(Loc::Stack(idx)) {
            self.mir.load_stack(re(dst), idx);
            self.set_reg_real(dst);
        } else {
            // self.mir.load_stack(re(dst), idx);
            // self.mir.load_stack(im(dst), idx + 1);
            self.mir.load_stack_complex(re(dst), im(dst), idx);
            self.set_reg_complex(dst);
        }
    }

    fn save_stack(&mut self, s1: Reg, idx: u32) {
        /*
         * the reason to use `ensure_complex` here is to correctly
         * handle slots which are updated using `assign`.
         * The problem was discovered on testing
         * `composer/mandelbrot.py`.
         */
        self.ensure_complex(s1);
        self.mir.save_stack_complex(re(s1), im(s1), idx);
        self.set_loc_complex(Loc::Stack(idx));
        /*
        if self.is_real_reg(s1) {
            self.mir.save_stack(re(s1), idx);
            self.set_loc_real(Loc::Stack(idx));
        } else {
            // self.mir.save_stack(re(s1), idx);
            // self.mir.save_stack(im(s1), idx + 1);
            self.mir.save_stack_complex(re(s1), im(s1), idx);
            self.set_loc_complex(Loc::Stack(idx));
        }
        */
    }

    fn load_mem_complex(&mut self, _xd: Reg, _yd: Reg, _idx: u32) {
        unreachable!()
    }

    fn save_mem_complex(&mut self, _xs: Reg, _ys: Reg, _idx: u32) {
        unreachable!()
    }

    fn load_param_complex(&mut self, _xd: Reg, _yd: Reg, _idx: u32) {
        unreachable!()
    }

    fn load_stack_complex(&mut self, _xd: Reg, _yd: Reg, _idx: u32) {
        unreachable!()
    }

    fn save_stack_complex(&mut self, _xs: Reg, _ys: Reg, _idx: u32) {
        unreachable!()
    }

    fn save_mem_result(&mut self, idx: u32) {
        self.save_mem(Reg::Ret, idx);
    }

    fn save_stack_result(&mut self, idx: u32) {
        self.save_stack(Reg::Ret, idx);
    }

    fn load_args(&mut self, locs: Vec<Loc>, ultra: bool) {
        self.mir.load_args_complex(locs, ultra)
    }

    fn save_args(&mut self, num_args: u8, ultra: bool) {
        self.mir.save_args_complex(num_args, ultra)
    }

    fn load_args_complex(&mut self, _loc: Vec<Loc>, _ultra: bool) {}

    fn save_args_complex(&mut self, _num_args: u8, _ultra: bool) {}

    fn neg(&mut self, dst: Reg, s1: Reg) {
        self.mir.neg(re(dst), re(s1));

        if !self.is_real_reg(s1) {
            self.mir.neg(im(dst), im(s1));
        }

        self.copy_real(dst, s1);
    }

    fn sign(&mut self, dst: Reg, s1: Reg) {
        self.mir.sign(re(dst), re(s1));

        if !self.is_real_reg(s1) {
            self.mir.sign(im(dst), im(s1));
        }

        self.copy_real(dst, s1);
    }

    fn abs(&mut self, dst: Reg, s1: Reg) {
        if self.is_real_reg(s1) {
            self.mir.abs(re(dst), re(s1));
            self.set_reg_real(dst);
        } else if self.mir.config.stable_math() {
            self.stable_abs(dst, s1);
            self.set_reg_complex(dst);
        } else {
            self.mir.times(Self::T0, re(s1), re(s1));
            self.mir.times(Self::T1, im(s1), im(s1));
            self.mir.plus(re(dst), Self::T0, Self::T1);
            self.mir.root(re(dst), re(dst));
            self.mir.xor(im(dst), im(dst), im(dst));
            self.set_reg_complex(dst);
        }
    }

    fn abs2(&mut self, dst: Reg, s1: Reg) {
        if self.is_real_reg(s1) {
            self.mir.times(re(dst), re(s1), re(s1));
            self.set_reg_real(dst);
        } else {
            self.mir.times(Self::T0, re(s1), re(s1));
            self.mir.times(Self::T1, im(s1), im(s1));
            self.mir.plus(re(dst), Self::T0, Self::T1);
            self.mir.xor(im(dst), im(dst), im(dst));
            self.set_reg_complex(dst);
        }
    }

    fn times_i(&mut self, dst: Reg, s1: Reg) {
        if self.is_real_reg(s1) {
            self.mir.fmov(im(dst), re(s1));
            self.mir.xor(re(dst), re(dst), re(dst));
        } else {
            self.mir.neg(Self::T1, im(s1));
            self.mir.fmov(im(dst), re(s1));
            self.mir.fmov(re(dst), Self::T1);
        }

        self.set_reg_complex(dst);
    }

    fn times_neg_i(&mut self, dst: Reg, s1: Reg) {
        if self.is_real_reg(s1) {
            self.mir.neg(im(dst), re(s1));
            self.mir.xor(re(dst), re(dst), re(dst));
        } else {
            self.mir.fmov(Self::T1, im(s1));
            self.mir.neg(im(dst), re(s1));
            self.mir.fmov(re(dst), Self::T1);
        }

        self.set_reg_complex(dst);
    }

    fn root(&mut self, dst: Reg, s1: Reg) {
        self.ensure_complex(s1);
        self.fmov(Reg::Temp, s1);
        self.call_funclet("@complex_root");
        self.set_reg_complex(Reg::Ret);
        self.fmov(dst, Reg::Ret);

        if !self.root_defined {
            self.root_defined = true;

            self.branch("@jump_over_complex_root");
            self.set_label("@complex_root");

            let s1 = Reg::Temp;
            let x = Self::T0;
            let y = Self::T1;

            self.mir.xor(x, x, x);
            self.mir.lt(x, x, re(s1));
            self.mir.save_stack(x, 2);

            if self.mir.config.stable_math() {
                self.stable_root_magnitude();
            } else {
                self.mir.times(x, re(s1), re(s1));
                self.mir.fused_mul_add(x, im(s1), im(s1), x);

                self.mir.root(x, x);
                self.mir.abs(y, re(s1));
                self.mir.plus(x, x, y);
                self.mir.half(x, x);
                self.mir.root(x, x);
            }
            self.mir.divide(y, im(s1), x);
            self.mir.half(y, y);

            self.mir.eq(re(s1), y, y);
            self.mir.and(y, y, re(s1));

            // note that we need to use `temp` and not `dst` here because
            // `ifelse` in x64 uses Reg::Temp, which is im(dst).
            self.mir.ifelse(re(s1), x, y, Loc::Stack(2));
            self.mir.ifelse(im(s1), y, x, Loc::Stack(2));

            self.mir.sign(x, re(s1));
            self.mir.xor(re(Reg::Ret), re(s1), x);
            self.mir.xor(im(Reg::Ret), im(s1), x);

            self.ret();

            self.set_label("@jump_over_complex_root");
        }
    }

    fn real_root(&mut self, dst: Reg, s1: Reg) {
        if self.is_real_reg(s1) {
            self.mir.root(re(dst), re(s1));
            self.set_reg_real(dst);
        } else {
            self.root(dst, s1);
        }
    }

    fn recip(&mut self, dst: Reg, s1: Reg) {
        if self.is_real_reg(s1) {
            self.mir.recip(re(dst), re(s1));
            self.set_reg_real(dst);
        } else if self.mir.config.stable_math() {
            self.stable_recip(dst, s1);
            self.set_reg_complex(dst);
        } else {
            self.mir.times(Self::T0, re(s1), re(s1));
            self.mir.fused_mul_add(Self::T0, im(s1), im(s1), Self::T0);
            self.mir.divide(re(dst), re(s1), Self::T0);
            self.mir.divide(im(dst), im(s1), Self::T0);
            self.mir.neg(im(dst), im(dst));
            self.set_reg_complex(dst);
        }
    }

    fn half(&mut self, dst: Reg, s1: Reg) {
        self.mir.half(im(dst), im(s1));
        self.mir.half(re(dst), re(s1));
    }

    fn round(&mut self, dst: Reg, s1: Reg) {
        self.mir.round(re(dst), re(s1));

        if !self.is_real_reg(s1) {
            self.mir.round(im(dst), im(s1));
        }

        self.copy_real(dst, s1);
    }

    fn floor(&mut self, dst: Reg, s1: Reg) {
        self.mir.floor(re(dst), re(s1));

        if !self.is_real_reg(s1) {
            self.mir.floor(im(dst), im(s1));
        }

        self.copy_real(dst, s1);
    }

    fn ceiling(&mut self, dst: Reg, s1: Reg) {
        self.mir.ceiling(re(dst), re(s1));

        if !self.is_real_reg(s1) {
            self.mir.ceiling(im(dst), im(s1));
        }

        self.copy_real(dst, s1);
    }

    fn trunc(&mut self, dst: Reg, s1: Reg) {
        self.mir.trunc(re(dst), re(s1));

        if !self.is_real_reg(s1) {
            self.mir.trunc(im(dst), im(s1));
        }

        self.copy_real(dst, s1);
    }

    fn frac(&mut self, dst: Reg, s1: Reg) {
        self.mir.frac(re(dst), re(s1));

        if !self.is_real_reg(s1) {
            self.mir.frac(im(dst), im(s1));
        }

        self.copy_real(dst, s1);
    }

    fn plus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        match self.types(s1, s2) {
            Types::RC => {
                self.mir.plus(re(dst), re(s1), re(s2));
                self.mir.fmov(im(dst), im(s2));
            }
            Types::CR => {
                self.mir.plus(re(dst), re(s1), re(s2));
                self.mir.fmov(im(dst), im(s1));
            }
            Types::CC => {
                //self.mir.plus(im(dst), im(s1), im(s2)),
                self.mir
                    .plus_complex(re(dst), im(dst), re(s1), im(s1), re(s2), im(s2));
            }
            Types::RR => {
                self.mir.plus(re(dst), re(s1), re(s2));
            }
        }

        self.promote_real(dst, s1, s2);
    }

    fn minus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        match self.types(s1, s2) {
            Types::RC => {
                self.mir.minus(re(dst), re(s1), re(s2));
                self.mir.neg(im(dst), im(s2));
            }
            Types::CR => {
                self.mir.minus(re(dst), re(s1), re(s2));
                self.mir.fmov(im(dst), im(s1));
            }
            Types::CC => {
                //self.mir.minus(im(dst), im(s1), im(s2)),
                self.mir
                    .minus_complex(re(dst), im(dst), re(s1), im(s1), re(s2), im(s2));
            }
            Types::RR => {
                self.mir.minus(re(dst), re(s1), re(s2));
            }
        }

        self.promote_real(dst, s1, s2);
    }

    fn times(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        match self.types(s1, s2) {
            Types::RR => self.mir.times(re(dst), re(s1), re(s2)),
            Types::RC => {
                self.mir.times(im(dst), re(s1), im(s2));
                self.mir.times(re(dst), re(s1), re(s2));
            }
            Types::CR => {
                self.mir.times(im(dst), im(s1), re(s2));
                self.mir.times(re(dst), re(s1), re(s2));
            }
            Types::CC => {
                /*
                self.mir.times(Self::T0, re(s1), re(s2));
                self.mir.times(Self::T1, im(s1), im(s2));
                self.mir.minus(Self::T0, Self::T0, Self::T1);

                self.mir.times(Self::T1, re(s1), im(s2));
                self.mir.times(im(dst), im(s1), re(s2));
                self.mir.plus(im(dst), im(dst), Self::T1);

                self.mir.fmov(re(dst), Self::T0);
                */
                self.mir
                    .times_complex(re(dst), im(dst), re(s1), im(s1), re(s2), im(s2));
            }
        }

        self.promote_real(dst, s1, s2);
    }

    fn divide(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        // note: we can use Reg::Temp here because the
        // operators using Reg::Temp do not use division,
        // except for powi_mod which needs to be modified (TODO)
        let t = re(Reg::Temp);

        match self.types(s1, s2) {
            Types::RR => self.mir.divide(re(dst), re(s1), re(s2)),
            Types::CR => {
                self.mir.divide(im(dst), im(s1), re(s2));
                self.mir.divide(re(dst), re(s1), re(s2));
            }
            Types::RC | Types::CC if self.mir.config.stable_math() => {
                self.stable_divide(dst, s1, s2);
            }
            Types::RC => {
                self.mir.times(Self::T0, re(s2), re(s2));
                self.mir.fused_mul_add(t, im(s2), im(s2), Self::T0);

                self.mir.times(Self::T0, re(s1), re(s2));
                self.mir.times(Self::T1, re(s1), im(s2));
                self.mir.neg(im(dst), Self::T1);

                self.mir.divide(im(dst), im(dst), t);
                self.mir.divide(re(dst), Self::T0, t);
            }
            Types::CC => {
                /*
                self.mir.times(Self::T0, re(s2), re(s2));
                self.mir.times(Self::T1, im(s2), im(s2));
                self.mir.plus(t, Self::T0, Self::T1);

                self.mir.times(Self::T0, re(s1), re(s2));
                self.mir.times(Self::T1, im(s1), im(s2));
                self.mir.plus(Self::T0, Self::T0, Self::T1);

                self.mir.times(Self::T1, re(s1), im(s2));
                self.mir.times(im(dst), im(s1), re(s2));
                self.mir.minus(im(dst), im(dst), Self::T1);

                self.mir.divide(im(dst), im(dst), t);
                self.mir.divide(re(dst), Self::T0, t);
                */
                self.mir
                    .divide_complex(re(dst), im(dst), re(s1), im(s1), re(s2), im(s2));
            }
        }

        self.promote_real(dst, s1, s2);
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
        unreachable!()
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
        unreachable!()
    }

    fn support_times2(&self) -> bool {
        false
    }

    fn times2_loc(&mut self, _d1: Reg, _s1: Reg, _l1: Loc, _d2: Reg, _s2: Reg, _l2: Loc) {
        unreachable!()
    }

    fn real(&mut self, dst: Reg, s1: Reg) {
        self.mir.fmov(re(dst), re(s1));
        self.mir.xor(im(dst), im(dst), im(dst));
        self.set_reg_real(dst);
    }

    fn imaginary(&mut self, dst: Reg, s1: Reg) {
        if !self.is_real_reg(s1) {
            self.mir.fmov(re(dst), im(s1));
        } else {
            self.mir.xor(re(dst), re(dst), re(dst));
        }
        self.mir.xor(im(dst), im(dst), im(dst));
        self.set_reg_real(dst);
    }

    fn conjugate(&mut self, dst: Reg, s1: Reg) {
        self.mir.fmov(re(dst), re(s1));

        if !self.is_real_reg(s1) {
            self.mir.fmov(im(dst), im(s1));
            self.mir.neg(im(dst), im(dst));
        }

        self.copy_real(dst, s1);
    }

    fn complex(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        // Important! The order of these two statements matters.
        // The imaginary part needs to be set first to prevent
        // conflict if dst == s2.
        self.mir.fmov(im(dst), re(s2));
        self.mir.fmov(re(dst), re(s1));
        self.set_reg_complex(dst);
    }

    fn gt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.gt(re(dst), re(s1), re(s2));
        self.set_reg_real(dst);
    }

    fn geq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.geq(re(dst), re(s1), re(s2));
        self.set_reg_real(dst);
    }

    fn lt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.lt(re(dst), re(s1), re(s2));
        self.set_reg_real(dst);
    }

    fn leq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.mir.leq(re(dst), re(s1), re(s2));
        self.set_reg_real(dst);
    }

    fn eq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.ensure_complex(s1);
        self.ensure_complex(s2);
        self.mir.eq(im(dst), im(s1), im(s2));
        self.mir.eq(re(dst), re(s1), re(s2));
        self.mir.and(re(dst), re(dst), im(dst));
        self.set_reg_real(dst);
    }

    fn neq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.ensure_complex(s1);
        self.ensure_complex(s2);
        self.mir.neq(im(dst), im(s1), im(s2));
        self.mir.neq(re(dst), re(s1), re(s2));
        self.mir.or(re(dst), re(dst), im(dst)); // de Morgan's law
        self.set_reg_real(dst);
    }

    fn and(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        match self.types(s1, s2) {
            Types::RR => {}
            Types::RC => self.mir.and(im(dst), re(s1), im(s2)),
            Types::CR => self.mir.and(im(dst), im(s1), re(s2)),
            Types::CC => self.mir.and(im(dst), im(s1), im(s2)),
        }

        self.mir.and(re(dst), re(s1), re(s2));
        self.promote_real(dst, s1, s2);
    }

    fn andnot(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        match self.types(s1, s2) {
            Types::RR => {}
            Types::RC => self.mir.andnot(im(dst), re(s1), im(s2)),
            Types::CR => self.mir.andnot(im(dst), im(s1), re(s2)),
            Types::CC => self.mir.andnot(im(dst), im(s1), im(s2)),
        }

        self.mir.andnot(re(dst), re(s1), re(s2));
        self.promote_real(dst, s1, s2);
    }

    fn or(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        match self.types(s1, s2) {
            Types::RR => {}
            Types::RC => self.mir.or(im(dst), re(s1), im(s2)),
            Types::CR => self.mir.or(im(dst), im(s1), re(s2)),
            Types::CC => self.mir.or(im(dst), im(s1), im(s2)),
        }

        self.mir.or(re(dst), re(s1), re(s2));
        self.promote_real(dst, s1, s2);
    }

    fn xor(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        match self.types(s1, s2) {
            Types::RR => {}
            Types::RC => self.mir.xor(im(dst), re(s1), im(s2)),
            Types::CR => self.mir.xor(im(dst), im(s1), re(s2)),
            Types::CC => self.mir.xor(im(dst), im(s1), im(s2)),
        }

        self.mir.xor(re(dst), re(s1), re(s2));
        self.promote_real(dst, s1, s2);
    }

    // `not` negates a mask, which is real (comparisons give real masks): the imaginary part of
    // a mask kept in a complex register or stack slot is zero and stays zero. Negating it too
    // made every negated mask non-zero for `iszero` (the test of `if` and `while` in complex
    // kernels).
    fn not(&mut self, dst: Reg, s1: Reg) {
        self.mir.not(re(dst), re(s1));
        self.set_reg_real(dst);
    }

    fn fused_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        assert!(s3 != Reg::Ret);
        self.times(Reg::Ret, s1, s2);
        self.plus(dst, Reg::Ret, s3);
    }

    fn fused_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        assert!(s3 != Reg::Ret);
        self.times(Reg::Ret, s1, s2);
        self.minus(dst, Reg::Ret, s3);
    }

    fn fused_neg_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        assert!(s3 != Reg::Ret);
        self.times(Reg::Ret, s1, s2);
        self.minus(dst, s3, Reg::Ret);
    }

    fn fused_neg_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        assert!(s3 != Reg::Ret);
        self.times(Reg::Ret, s1, s2);
        self.plus(dst, Reg::Ret, s3);
        self.neg(dst, dst);
    }

    fn add_consts(&mut self, consts: &[f64]) {
        self.mir.add_consts(consts);
    }

    fn add_func(&mut self, _f: &str, _p: Func) {}

    fn call(&mut self, op: &str, num_args: usize) -> Result<()> {
        // Important! We need to mark Reg::Ret as complex because
        // of a bug discovered while testing complex applet calls.
        self.ensure_complex(Reg::Ret);

        // Kernels and external functions take their arguments in memory and return complex
        // values; the other functions are real, with their arguments in registers.
        let real = !self.mir.config.is_external_func(op);

        if real && num_args == 2 {
            // the second argument is the real part of `Right`, but a real call reads
            // `Right` itself, which is the imaginary part of `Ret` (zeroed above)
            self.mir.fmov(Reg::Right, re(Reg::Right));
        }

        self.mir.call(op, num_args)?;

        // A real function (e.g. a callback) returns a real value, and the call can clobber
        // the imaginary part of `Ret` (on ARM64, the C function's scalar result zeroes only
        // the upper lane of the vector register): it is zeroed when it is used.
        if real {
            self.set_reg_real(Reg::Ret);
        }
        Ok(())
    }

    fn call_complex(&mut self, op: &str, num_args: usize) -> Result<()> {
        self.ensure_complex(Reg::Ret);

        match num_args {
            1 => {}
            2 => self.ensure_complex(Reg::Temp),
            _ => return Err(anyhow!("complex functions expect 1 or 2 arguments.")),
        }

        self.mir.call(op, num_args)
    }

    fn call_funclet(&mut self, label: &str) {
        self.mir.call(label, 0).unwrap();
        // A helper returns a complete complex value, independently of the
        // caller's prior temporary. Change metadata, not the returned data.
        self.set_reg_complex(Reg::Ret);
    }

    fn ret(&mut self) {
        // A real helper result still owes the caller a zero imaginary part.
        self.ensure_complex(Reg::Ret);
        self.branch(".ret");
    }

    fn prologue_fast(&mut self, _cap: usize, _count_states: usize, _count_obs: usize) {}
    fn epilogue_fast(
        &mut self,
        _cap: usize,
        _count_states: usize,
        _count_obs: usize,
        _idx_ret: i32,
    ) {
    }

    fn prologue_indirect(
        &mut self,
        _cap: usize,
        _count_states: usize,
        _count_obs: usize,
        _count_params: usize,
    ) {
    }
    fn epilogue_indirect(
        &mut self,
        _cap: usize,
        _count_states: usize,
        _count_obs: usize,
        _count_params: usize,
    ) {
    }

    fn save_used_registers(&mut self, _used: &[Reg]) {}
    fn load_used_registers(&mut self, _used: &[Reg]) {}

    fn fuse_load_math(&mut self) {}

    fn ifelse(&mut self, dst: Reg, true_val: Reg, false_val: Reg, idx: u32) {
        let loc = Loc::Stack(idx);

        self.ensure_complex(true_val);
        self.ensure_complex(false_val);

        self.mir.ifelse(re(dst), re(true_val), re(false_val), loc);
        self.mir.ifelse(im(dst), im(true_val), im(false_val), loc);

        self.set_reg_complex(dst);
        // self.promote_real(dst, true_val, false_val);
    }
}

/*  Complex Genetic Functions */

impl Complexifier {
    pub fn generic_complex_plus(
        ir: &mut dyn Generator,
        xd: Reg,
        yd: Reg,
        x1: Reg,
        y1: Reg,
        x2: Reg,
        y2: Reg,
    ) {
        ir.plus(xd, x1, x2);
        ir.plus(yd, y1, y2);
    }

    pub fn generic_complex_minus(
        ir: &mut dyn Generator,
        xd: Reg,
        yd: Reg,
        x1: Reg,
        y1: Reg,
        x2: Reg,
        y2: Reg,
    ) {
        ir.minus(xd, x1, x2);
        ir.minus(yd, y1, y2);
    }

    pub fn generic_complex_times(
        ir: &mut dyn Generator,
        xd: Reg,
        yd: Reg,
        x1: Reg,
        y1: Reg,
        x2: Reg,
        y2: Reg,
    ) {
        ir.times(Self::T0, x1, x2);
        ir.times(Self::T1, y1, y2);
        ir.minus(Self::T0, Self::T0, Self::T1);

        ir.times(Self::T1, x1, y2);
        ir.times(yd, y1, x2);
        ir.plus(yd, yd, Self::T1);

        ir.fmov(xd, Self::T0);
    }

    pub fn generic_complex_divide(
        ir: &mut dyn Generator,
        xd: Reg,
        yd: Reg,
        x1: Reg,
        y1: Reg,
        x2: Reg,
        y2: Reg,
    ) {
        let t = re(Reg::Temp);

        ir.times(Self::T0, x2, x2);
        ir.times(Self::T1, y2, y2);
        ir.plus(t, Self::T0, Self::T1);

        ir.times(Self::T0, x1, x2);
        ir.times(Self::T1, y1, y2);
        ir.plus(Self::T0, Self::T0, Self::T1);

        ir.times(Self::T1, x1, y2);
        ir.times(yd, y1, x2);
        ir.minus(yd, yd, Self::T1);

        ir.divide(yd, yd, t);
        ir.divide(xd, Self::T0, t);
    }
}

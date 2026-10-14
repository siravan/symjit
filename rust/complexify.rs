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
    /// the functions of the math library called (`call_real_or_complex`; feature `math`)
    #[cfg(feature = "math")]
    libm_used: Vec<String>,
    #[cfg(feature = "math")]
    count_libm_calls: usize,
}

impl Complexifier {
    pub fn new(reals: &HashSet<Loc>, config: Config) -> Complexifier {
        Complexifier {
            mir: Mir::new(config),
            // Real declarations have the same scalar offsets as MIR loads.
            real_locs: reals.clone(),
            real_regs: [false; 32],
            root_defined: false,
            #[cfg(feature = "math")]
            libm_used: Vec::new(),
            #[cfg(feature = "math")]
            count_libm_calls: 0,
        }
    }

    pub fn complexify(&mut self, mir: &Mir) -> Result<Mir> {
        self.mir.consts = mir.consts.clone();
        // self.mir.labels = mir.labels.clone();
        mir.rerun(self)?;

        #[cfg(feature = "math")]
        if !self.libm_used.is_empty() {
            // the registers of the real kernel (not the complex ones of `count_scratch`)
            let names: Vec<&str> = self.libm_used.iter().map(|s| s.as_str()).collect();
            crate::libm::append_funclets(&mut self.mir, &names, 12)?;
        }

        self.mir.populate_labels();
        Ok(std::mem::take(&mut self.mir))
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

#[cfg(feature = "math")]
impl Complexifier {
    fn use_libm(&mut self, name: &str) {
        if !self.libm_used.iter().any(|n| n == name) {
            self.libm_used.push(name.to_string());
        }
    }

    fn const_index(&mut self, v: f64) -> u32 {
        match self
            .mir
            .consts
            .iter()
            .position(|c| c.to_bits() == v.to_bits())
        {
            Some(idx) => idx as u32,
            None => {
                self.mir.consts.push(v);
                (self.mir.consts.len() - 1) as u32
            }
        }
    }

    /// A call of a complex function `op` that the library has in complex form (the
    /// hyperbolic functions): a real argument known at compile time gets the real function;
    /// otherwise the complex funclet `@libm_cplx_<op>` when |re| and |im| are within
    /// `libm::complex_domain` (false for NaN and infinities), else the complex C function. In
    /// a SIMD kernel, a row whose lanes do not agree is computed again by the scalar kernel
    /// (`branch_if`).
    fn call_complex_inline(&mut self, op: &str) -> Result<()> {
        if self.is_real_reg(Reg::Ret) {
            self.use_libm(op);
            self.mir.call(&crate::libm::funclet_label(op), 0)?;
            self.set_reg_real(Reg::Ret);
            return Ok(());
        }

        let name = format!("{}{}", crate::mir::LIBM_COMPLEX, op);
        self.use_libm(&name);
        self.ensure_complex(Reg::Ret);

        let n = self.count_libm_calls;
        self.count_libm_calls += 1;
        let inside = format!("@cplx_inside_{}", n);
        let end = format!("@cplx_call_end_{}", n);

        // |re| < x_limit and |im| < y_limit; the registers of `Temp` are free in a unary call
        let (x_limit, y_limit) = crate::libm::complex_domain(op);
        let (t, u, c) = (Self::T0, Self::T1, im(Reg::Temp));
        let (ix, iy) = (self.const_index(x_limit), self.const_index(y_limit));
        self.mir.abs(t, re(Reg::Ret));
        self.mir.load_const(c, ix);
        self.mir.lt(t, t, c);
        self.mir.abs(u, im(Reg::Ret));
        self.mir.load_const(c, iy);
        self.mir.lt(u, u, c);
        self.mir.and(t, t, u);
        self.mir.branch_if(t, &inside, true);
        self.mir.call(op, 1)?;
        self.mir.branch(&end);
        self.mir.set_label(&inside);
        self.mir.call(&crate::libm::funclet_label(&name), 0)?;
        self.mir.set_label(&end);

        self.set_reg_complex(Reg::Ret);
        Ok(())
    }

    /// A call of a complex function `op` that the math library (`libm.rs`) has in real form
    /// only (the logarithms): when the argument is real at run time (its imaginary part is
    /// zero), the real function is called (the inlined `@libm_<op>` in SIMD kernels,
    /// `Generator::inlines_math`, the C function otherwise), and the complex C function when
    /// it is not. In a SIMD kernel, a row whose lanes do not agree is computed again by the
    /// scalar kernel (`branch_if`). A zero imaginary part of the argument gives +0 as the
    /// imaginary part of the result (its sign is not kept). For a function with a domain
    /// (`libm::domain_limit`), the real part must also be below the limit in absolute value;
    /// for the logarithms (`libm::positive_domain`), it must be positive (and the test is not
    /// skipped for an argument known to be real).
    fn call_real_or_complex(&mut self, op: &str) -> Result<()> {
        if !self.libm_used.iter().any(|n| n == op) {
            self.libm_used.push(op.to_string());
        }
        let real = crate::libm::funclet_label(op);
        let limit = crate::libm::domain_limit(op);
        let positive = crate::libm::positive_domain(op);

        if self.is_real_reg(Reg::Ret) && limit.is_none() && !positive {
            self.mir.call(&real, 0)?;
            self.set_reg_real(Reg::Ret);
            return Ok(());
        }

        self.ensure_complex(Reg::Ret);
        let n = self.count_libm_calls;
        self.count_libm_calls += 1;
        let real_arg = format!("@cplx_real_arg_{}", n);
        let end = format!("@cplx_call_end_{}", n);

        // all lanes real: the real function; none: the complex one (`branch_if` with
        // is_else = true jumps when the condition is true)
        let t = Self::T0;
        self.mir.xor(t, t, t);
        self.mir.eq(t, im(Reg::Ret), t);
        if let Some(limit) = limit {
            // and |re| < limit (false for NaN and infinities); the registers of `Temp` are
            // free in a unary call
            let (u, c) = (Self::T1, im(Reg::Temp));
            let idx = match self
                .mir
                .consts
                .iter()
                .position(|v| v.to_bits() == limit.to_bits())
            {
                Some(idx) => idx,
                None => {
                    self.mir.consts.push(limit);
                    self.mir.consts.len() - 1
                }
            };
            self.mir.abs(u, re(Reg::Ret));
            self.mir.load_const(c, idx as u32);
            self.mir.lt(u, u, c);
            self.mir.and(t, t, u);
        }
        if positive {
            // and re > 0 (false for NaN): a negative real has a complex logarithm
            let u = Self::T1;
            self.mir.xor(u, u, u);
            self.mir.lt(u, u, re(Reg::Ret));
            self.mir.and(t, t, u);
        }
        self.mir.branch_if(t, &real_arg, true);
        self.mir.call(op, 1)?;
        self.mir.branch(&end);
        self.mir.set_label(&real_arg);
        self.mir.call(&real, 0)?;
        self.mir.xor(im(Reg::Ret), im(Reg::Ret), im(Reg::Ret));
        self.mir.set_label(&end);

        self.set_reg_complex(Reg::Ret);
        Ok(())
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
            self.mir.lt(x, x, re(s1)); // lt intead of ge for SSE to work correctly
            self.mir.save_stack(x, 2);

            self.mir.times(x, re(s1), re(s1));
            self.mir.fused_mul_add(x, im(s1), im(s1), x);

            self.mir.root(x, x);
            self.mir.abs(y, re(s1));
            self.mir.plus(x, x, y);
            self.mir.half(x, x);
            self.mir.root(x, x);
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
        } else if self.mir.config.is_sse() {
            self.mir.times(Self::T0, im(s1), im(s1));
            self.mir.times(Self::T1, re(s1), re(s1));
            self.mir.plus(Self::T0, Self::T0, Self::T1);
            self.mir.divide(re(dst), re(s1), Self::T0);
            self.mir.divide(im(dst), im(s1), Self::T0);
            self.mir.neg(im(dst), im(dst));
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

    // pow2i of the real part; the result is real
    fn pow2i(&mut self, dst: Reg, s1: Reg) {
        self.mir.pow2i(re(dst), re(s1));
        self.mir.xor(im(dst), im(dst), im(dst));
        self.set_reg_real(dst);
    }

    // getexp of the real part; the result is real
    fn getexp(&mut self, dst: Reg, s1: Reg) {
        self.mir.getexp(re(dst), re(s1));
        self.mir.xor(im(dst), im(dst), im(dst));
        self.set_reg_real(dst);
    }

    // getmant of the real part; the result is real
    fn getmant(&mut self, dst: Reg, s1: Reg) {
        self.mir.getmant(re(dst), re(s1));
        self.mir.xor(im(dst), im(dst), im(dst));
        self.set_reg_real(dst);
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
            Types::RC => {
                if self.mir.config.is_sse() {
                    self.mir.times(Self::T0, re(s2), re(s2));
                    self.mir.times(Self::T1, im(s2), im(s2));
                    self.mir.plus(t, Self::T0, Self::T1);

                    self.mir.times(Self::T0, re(s1), re(s2));
                    self.mir.times(Self::T1, re(s1), im(s2));
                    self.mir.neg(im(dst), Self::T1);

                    self.mir.divide(im(dst), im(dst), t);
                    self.mir.divide(re(dst), Self::T0, t);
                } else {
                    self.mir.times(Self::T0, re(s2), re(s2));
                    self.mir.fused_mul_add(t, im(s2), im(s2), Self::T0);

                    self.mir.times(Self::T0, re(s1), re(s2));
                    self.mir.times(Self::T1, re(s1), im(s2));
                    self.mir.neg(im(dst), Self::T1);

                    self.mir.divide(im(dst), im(dst), t);
                    self.mir.divide(re(dst), Self::T0, t);
                }
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

    fn not(&mut self, dst: Reg, s1: Reg) {
        self.mir.not(re(dst), re(s1));

        if !self.is_real_reg(s1) {
            self.mir.not(im(dst), im(s1));
        }

        self.copy_real(dst, s1);
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
        #[cfg(feature = "math")]
        if num_args == 1 && self.mir.config.inline_math() {
            if crate::libm::has_complex_function(op) {
                return self.call_complex_inline(op);
            }
            if crate::libm::real_in_complex(op) {
                return self.call_real_or_complex(op);
            }
        }

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

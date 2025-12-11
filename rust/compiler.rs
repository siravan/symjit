#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{__m256d, _mm256_setzero_pd};

use anyhow::Result;

use crate::defuns::Defuns;
use crate::expr::Expr;
use crate::model::{CellModel, Equation, Program, Variable};
use crate::{CompilerType, Runnable};

// #[derive(Debug)]
pub struct Compiler {
    opt: u32,
    ty: CompilerType,
    params: Vec<Expr>,
}

/// The central hub of the Rust interface. It compiles a list of
/// variables and expressions into a callable object (of type `Runnable`).
///
/// # Workflow
///
/// 1. Create variables and expressions using `Expr` methods.
/// 2. Create a new `Compiler` object  using one of the constructors: `new` and `with_compile_type`,
/// which accepts a `ty` argument of type `CompilerType`.
/// 3. Optionally, fine-tune the optimization passes using functions `opt_level`, `simd`, `fastmath`,
///     and `cse`.
/// 4. Generate the JIT code (say, `comp`) by calling `compile`.
/// 5. Optionally, change parameters by writing directly to `comp.params`.
/// 6. Call the compiled code using one of the `call` functions:
///     * `call`: for scalar function without rewriting parames.
///     * `call_params`: for scalar function with rewriting parames.
///     * `call_simd`: for simd (__m256d, i.e., packed f64 x 4) function without rewriting parames.
///     * `call_simd_params`: for simd (__m256d, i.e., packed f64 x 4) function with rewriting parames.
///
///
/// Note that the simd functions are marked as unsafe.
///
/// # Examples
///
/// ```rust
/// use anyhow::Result;
/// use symjit::{Compiler, Expr};
///
/// pub fn main() -> Result<()> {
///     let x = Expr::var("x");
///     let y = Expr::var("y");
///     let p = &x + &y;
///     let q = &x * &y;
///
///     let mut comp = Compiler::new();
///     comp.opt_level(2);  # optional (opt_level 0 to 2; default 1)
///     let mut func = comp.compile(&[x, y], &[p, q])?;
///     let v = func.call(&[3.0, 5.0]);
///     println!("{:?}", &v);
///
///     Ok(())
/// }
/// ```
impl Compiler {
    const USE_SIMD: u32 = 0x01;
    const USE_THREADS: u32 = 0x02;
    const CSE: u32 = 0x04;
    const FASTMATH: u32 = 0x08;
    const SANITIZE: u32 = 0x10;

    const OPT_LEVEL_0: u32 = 0x0000;
    const OPT_LEVEL_1: u32 = 0x0100;
    const OPT_LEVEL_2: u32 = 0x0200;
    const OPT_LEVEL_MASK: u32 = 0x0f00;
    const OPT_LEVEL_SHIFT: usize = 8;

    pub const DEFAULT: u32 = Self::CSE | Self::SANITIZE | Self::OPT_LEVEL_1 | Self::USE_SIMD;

    /// Creates a new `Compiler` object with default settings.
    pub fn new() -> Compiler {
        Compiler {
            opt: Self::DEFAULT,
            ty: CompilerType::Native,
            params: Vec::new(),
        }
    }

    /// Creates a new `Compiler` object based on `ty`:
    ///
    /// * `CompilerType::Native`: generates code for the detected CPU (default)
    /// * `CompilerType::Amd`: generates x86-64 (AMD64) code.
    /// * `CompilerType::AmdAVX`: generates AVX code for x86-64 architecture.
    /// * `CompilerType::AmdSSE`: generates SSE2 code for x86-64 architecture.
    /// * `CompilerType::Arm`: generates aarch64 (ARM64) code.
    /// * `CompilerType::RiscV`: generates riscv64 (RISC V) code.
    /// * `CompilerType::ByteCode`: generates bytecode (interpreter).
    /// * `CompilerType::Debug`: debug mode, generates both bytecode and native codes
    ///     and compares the outputs.
    ///
    pub fn with_compiler_type(ty: CompilerType) -> Compiler {
        Compiler {
            opt: Self::DEFAULT,
            ty,
            params: Vec::new(),
        }
    }

    /// Sets of optimization level. The valid values are 0, 1, 2, which roughly correspond to gcc O0, O1, and O2 levels.
    pub fn opt_level(&mut self, opt_level: u8) {
        self.opt =
            (self.opt & !Self::OPT_LEVEL_MASK) | ((opt_level as u32) << Self::OPT_LEVEL_SHIFT);
    }

    /// Enables Common-Subexpression-Elimination.
    pub fn cse(&mut self, enabled: bool) {
        self.opt = (self.opt & !Self::CSE) | if enabled { Self::CSE } else { 0 };
    }

    /// Enables fastmath mode. The main effect is to generate fused-multiply-addition
    /// instructions if possible.
    pub fn fastmath(&mut self, enabled: bool) {
        self.opt = (self.opt & !Self::FASTMATH) | if enabled { Self::FASTMATH } else { 0 };
    }

    /// Enables SIMD mode.
    pub fn simd(&mut self, enabled: bool) {
        self.opt = (self.opt & !Self::USE_SIMD) | if enabled { Self::USE_SIMD } else { 0 };
    }

    /// Sets params. The argument is a list of variables, created by `Expr::var`.
    pub fn def_params(&mut self, params: Vec<Expr>) {
        self.params = params;
    }

    /// Compiles the model.
    ///
    /// `states` is a list of variables, created by `Expr::var`.
    /// `obs` is a list of expressions.
    pub fn compile(&mut self, states: &[Expr], obs: &[Expr]) -> Result<Runnable> {
        let mut vars: Vec<Variable> = Vec::new();

        for state in states.iter() {
            let v = state.to_variable()?;
            vars.push(v);
        }

        let mut params: Vec<Variable> = Vec::new();

        for p in self.params.iter() {
            let v = p.to_variable()?;
            params.push(v);
        }

        let mut eqs: Vec<Equation> = Vec::new();

        for (i, expr) in obs.iter().enumerate() {
            let name = format!("${}", i);
            let lhs = Expr::var(&name);
            eqs.push(Expr::equation(&lhs, &expr));
        }

        let ml = CellModel {
            iv: Expr::var("$_").to_variable()?,
            params,
            states: vars,
            algs: Vec::new(),
            odes: Vec::new(),
            obs: eqs,
        };

        let prog = Program::new(&ml, false)?;
        let df = Defuns::new();
        Runnable::new(prog, self.ty, self.opt, &df)
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_slice(a: &[f64]) -> &[__m256d] {
    assert!(a.len() & 3 == 0);
    let p: *const f64 = a.as_ptr();
    let v = unsafe { std::slice::from_raw_parts(p as *const __m256d, a.len() >> 2) };
    v
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_slice_mut(a: &mut [f64]) -> &mut [__m256d] {
    assert!(a.len() & 3 == 0);
    let p: *mut f64 = a.as_mut_ptr();
    let v: &mut [__m256d] =
        unsafe { std::slice::from_raw_parts_mut(p as *mut __m256d, a.len() >> 2) };
    v
}

pub enum FastFunc<'a> {
    F1(fn(f64) -> f64, &'a Runnable),
    F2(fn(f64, f64) -> f64, &'a Runnable),
    F3(fn(f64, f64, f64) -> f64, &'a Runnable),
    F4(fn(f64, f64, f64, f64) -> f64, &'a Runnable),
    F5(fn(f64, f64, f64, f64, f64) -> f64, &'a Runnable),
    F6(fn(f64, f64, f64, f64, f64, f64) -> f64, &'a Runnable),
    F7(fn(f64, f64, f64, f64, f64, f64, f64) -> f64, &'a Runnable),
    F8(
        fn(f64, f64, f64, f64, f64, f64, f64, f64) -> f64,
        &'a Runnable,
    ),
}

impl Runnable {
    /// Calls the compiled function.
    ///
    /// `args` is a slice of f64 values, corresponding to the states.
    ///
    /// The output is a `Vec<f64>`, corresponding to the observables (the expressions passed
    /// to `compile`).
    pub fn call(&mut self, args: &[f64]) -> Vec<f64> {
        {
            let mem = self.compiled.mem_mut();
            mem[self.idx_iv] = 0.0;
            let states = &mut mem[self.first_state..self.first_state + self.count_states];
            states.copy_from_slice(args);
        }

        self.compiled.exec(&self.params[..]);

        let obs = {
            let mem = self.compiled.mem();
            &mem[self.first_obs..self.first_obs + self.count_obs]
        };

        obs.to_vec()
    }

    /// Sets the params and calls the compiled function.
    ///
    /// `args` is a slice of f64 values, corresponding to the states.
    /// `params` is a slice of f64 values, corresponding to the params.
    ///
    /// The output is a `Vec<f64>`, corresponding to the observables (the expressions passed
    /// to `compile`).
    pub fn call_params(&mut self, args: &[f64], params: &[f64]) -> Vec<f64> {
        {
            let mem = self.compiled.mem_mut();
            mem[self.idx_iv] = 0.0;
            let states = &mut mem[self.first_state..self.first_state + self.count_states];
            states.copy_from_slice(args);
        }

        self.compiled.exec(params);

        let obs = {
            let mem = self.compiled.mem();
            &mem[self.first_obs..self.first_obs + self.count_obs]
        };

        obs.to_vec()
    }

    /// Calls the compiled SIMD function.
    ///
    /// `args` is a slice of __m256d values, corresponding to the states.
    ///
    /// The output is an `Option` wrapped `Vec<__m256d>`, corresponding to the observables (the expressions passed
    /// to `compile`).
    ///
    /// Note: currently, this function only works on X86-64 CPUs with the AVX extension. Intel
    /// introduced the AVX instruction set in 2011; therefore, most intel and AMD processors
    /// support it. If SIMD is not supported, this function returns `None`.
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn call_simd(&mut self, args: &[__m256d]) -> Option<Vec<__m256d>> {
        if let Some(f) = &mut self.compiled_simd {
            {
                let mem = f.mem_mut();
                let states = simd_slice_mut(
                    &mut mem[self.first_state * 4..(self.first_state + self.count_states) * 4],
                );
                states.copy_from_slice(args);
            }

            f.exec(&self.params);

            {
                let mem = f.mem();
                let obs =
                    simd_slice(&mem[self.first_obs * 4..(self.first_obs + self.count_obs) * 4]);
                let mut res = vec![_mm256_setzero_pd(); self.count_obs];
                res.copy_from_slice(obs);
                Some(res)
            }
        } else {
            self.prepare_simd();
            if self.compiled_simd.is_some() {
                self.call_simd(args)
            } else {
                None
            }
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub unsafe fn call_simd(&mut self, args: &[__m256d]) -> Option<Vec<__m256d>> {
        None
    }

    /// Sets the params and calls the compiled SIMD function.
    ///
    /// `args` is a slice of __m256d values, corresponding to the states.
    /// `params` is a slice of f64 values.
    ///
    /// The output is an `Option` wrapped `Vec<__m256d>`, corresponding to the observables (the expressions passed
    /// to `compile`).
    ///
    /// Note: currently, this function only works on X86-64 CPUs with the AVX extension. Intel
    /// introduced the AVX instruction set in 2011; therefore, most intel and AMD processors
    /// support it. If SIMD is not supported, this function returns `None`.
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn call_simd_params(
        &mut self,
        args: &[__m256d],
        params: &[f64],
    ) -> Option<Vec<__m256d>> {
        if let Some(f) = &mut self.compiled_simd {
            {
                let mem = f.mem_mut();
                let states = simd_slice_mut(
                    &mut mem[self.first_state * 4..(self.first_state + self.count_states) * 4],
                );
                states.copy_from_slice(args);
            }

            f.exec(params);

            {
                let mem = f.mem();
                let obs =
                    simd_slice(&mem[self.first_obs * 4..(self.first_obs + self.count_obs) * 4]);
                let mut res = vec![_mm256_setzero_pd(); self.count_obs];
                res.copy_from_slice(obs);
                Some(res)
            }
        } else {
            self.prepare_simd();
            if self.compiled_simd.is_some() {
                self.call_simd(args)
            } else {
                None
            }
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub unsafe fn call_simd_params(
        &mut self,
        args: &[__m256d],
        params: &[f64],
    ) -> Option<Vec<__m256d>> {
        None
    }

    pub fn fast_func<'a>(&'a mut self) -> Option<FastFunc<'a>> {
        let f = self.get_fast();

        if let Some(f) = f {
            match self.count_states {
                1 => {
                    let g: fn(f64) -> f64 = unsafe { std::mem::transmute(f) };
                    Some(FastFunc::F1(g, self))
                }
                2 => {
                    let g: fn(f64, f64) -> f64 = unsafe { std::mem::transmute(f) };
                    Some(FastFunc::F2(g, self))
                }
                3 => {
                    let g: fn(f64, f64, f64) -> f64 = unsafe { std::mem::transmute(f) };
                    Some(FastFunc::F3(g, self))
                }
                4 => {
                    let g: fn(f64, f64, f64, f64) -> f64 = unsafe { std::mem::transmute(f) };
                    Some(FastFunc::F4(g, self))
                }
                5 => {
                    let g: fn(f64, f64, f64, f64, f64) -> f64 = unsafe { std::mem::transmute(f) };
                    Some(FastFunc::F5(g, self))
                }
                6 => {
                    let g: fn(f64, f64, f64, f64, f64, f64) -> f64 =
                        unsafe { std::mem::transmute(f) };
                    Some(FastFunc::F6(g, self))
                }
                7 => {
                    let g: fn(f64, f64, f64, f64, f64, f64, f64) -> f64 =
                        unsafe { std::mem::transmute(f) };
                    Some(FastFunc::F7(g, self))
                }
                8 => {
                    let g: fn(f64, f64, f64, f64, f64, f64, f64, f64) -> f64 =
                        unsafe { std::mem::transmute(f) };
                    Some(FastFunc::F8(g, self))
                }
                _ => None,
            }
        } else {
            None
        }
    }
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{__m256d, _mm256_setzero_pd};

use anyhow::{anyhow, Result};

use crate::defuns::Defuns;
use crate::expr::Expr;
use crate::model::{CellModel, Equation, Program, Variable};
use crate::{Application, CompilerType};

// #[derive(Debug)]
pub struct Compiler {
    opt: u32,
    ty: CompilerType,
    df: Defuns,
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(non_camel_case_types)]
type __m256d = [f64; 4];

/// The central hub of the Rust interface. It compiles a list of
/// variables and expressions into a callable object (of type `Application`).
///
/// # Workflow
///
/// 1. Create terminals (variables and constants) and compose expressions using `Expr` methods:
///    * Constructors: `var`, `from`, `unary`, `binary`, ...
///    * Standard algebraic operations: `add`, `mul`, ...
///    * Standard operators `+`, `-`, `*`, `/`, `%`, `&`, `|`, `^`, `!`.
///    * Unary functions such as `sin`, `exp`, and other standard mathematical functions.
///    * Binary functions such as `pow`, `min`, ...
///    * IfElse operation `ifelse(cond, true_val, false_val)`.
///    * Heavide function: `heaviside(x)`, which returns 1 if `x >= 0`; otherwise 0.
///    * Comparison methods `eq`, `ne`, `lt`, `le`, `gt`, and `ge`.
///    * Looping constructs `sum` and `prod`.
/// 2. Create a new `Compiler` object (say, `comp`) using one of its constructors: `new()`
///    or `with_compile_type(ty: CompilerType)`.
/// 3. Fine-tune the optimization passes using `opt_level`, `simd`, `fastmath`,
///    and `cse` methods (optional).
/// 4. Define user-defined functions by called `comp.def_unary` and `comp.def_binary`
///    (optional).
/// 5. Compile by calling `comp.compile` or `comp.compile_params`. The result is of
///    type `Application` (say, `app`).
/// 6. Execute the compiled code using one of the `app`'s `call` functions:
///    * `call(&[f64])`: scalar call.
///    * `call_params(&[f64], &[f64])`: scalar call with parameters.
///    * `call_simd(&[__m256d])`: simd call.
///    * `call_simd_params(&[__m256d], &[f64])`: simd call with parameters.
/// 7. Optionally, generate a standalone fast function to execute.
///
/// Currently, SIMD is only supported on x86-64 CPUs with AVX instruction sets.
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
///     let u = &x + &y;
///     let v = &x * &y;
///
///     let mut comp = Compiler::new();
///     comp.opt_level(2);  // optional (opt_level 0 to 2; default 1)
///     let mut app = comp.compile(&[x, y], &[u, v])?;
///     let res = app.call(&[3.0, 5.0]);
///     println!("{:?}", &res);
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
            df: Defuns::new(),
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
    ///    and compares the outputs.
    ///
    pub fn with_compiler_type(ty: CompilerType) -> Compiler {
        Compiler {
            opt: Self::DEFAULT,
            ty,
            df: Defuns::new(),
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

    /// Compiles a model.
    ///
    /// `states` is a list of variables, created by `Expr::var`.
    /// `obs` is a list of expressions.
    pub fn compile(&mut self, states: &[Expr], obs: &[Expr]) -> Result<Application> {
        self.compile_params(states, obs, &[])
    }

    /// Compiles a model with parameters.
    ///
    /// `states` is a list of variables, created by `Expr::var`.
    /// `obs` is a list of expressions.
    /// `params` is a list of parameters, created by `Expr::var`.
    ///
    /// Note: for scalar functions, the difference between states and params
    ///     is mostly by convenion. However, they are different in SIMD cases,
    ///     as params are always f64.
    pub fn compile_params(
        &mut self,
        states: &[Expr],
        obs: &[Expr],
        params: &[Expr],
    ) -> Result<Application> {
        let mut vars: Vec<Variable> = Vec::new();

        for state in states.iter() {
            let v = state.to_variable()?;
            vars.push(v);
        }

        let mut ps: Vec<Variable> = Vec::new();

        for p in params.iter() {
            let v = p.to_variable()?;
            ps.push(v);
        }

        let mut eqs: Vec<Equation> = Vec::new();

        for (i, expr) in obs.iter().enumerate() {
            let name = format!("${}", i);
            let lhs = Expr::var(&name);
            eqs.push(Expr::equation(&lhs, expr));
        }

        let ml = CellModel {
            iv: Expr::var("$_").to_variable()?,
            params: ps,
            states: vars,
            algs: Vec::new(),
            odes: Vec::new(),
            obs: eqs,
        };

        let prog = Program::new(&ml, false)?;
        // let df = Defuns::new();
        let mut app = Application::new(prog, self.ty, self.opt, &self.df);

        #[cfg(target_arch = "aarch64")]
        if let Ok(app) = &mut app {
            // this is a hack to give enough delay to prevent a bus error
            app.dump("dump.bin", "scalar");
            std::fs::remove_file("dump.bin")?;
        };

        app
    }

    /// Registers a user-defined unary function.
    pub fn def_unary(&mut self, op: &str, f: extern "C" fn(f64) -> f64) {
        self.df.add_unary(op, f)
    }

    /// Registers a user-defined binary function.
    pub fn def_binary(&mut self, op: &str, f: extern "C" fn(f64, f64) -> f64) {
        self.df.add_binary(op, f)
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
    F1(extern "C" fn(f64) -> f64, &'a Application),
    F2(extern "C" fn(f64, f64) -> f64, &'a Application),
    F3(extern "C" fn(f64, f64, f64) -> f64, &'a Application),
    F4(extern "C" fn(f64, f64, f64, f64) -> f64, &'a Application),
    F5(
        extern "C" fn(f64, f64, f64, f64, f64) -> f64,
        &'a Application,
    ),
    F6(
        extern "C" fn(f64, f64, f64, f64, f64, f64) -> f64,
        &'a Application,
    ),
    F7(
        extern "C" fn(f64, f64, f64, f64, f64, f64, f64) -> f64,
        &'a Application,
    ),
    F8(
        extern "C" fn(f64, f64, f64, f64, f64, f64, f64, f64) -> f64,
        &'a Application,
    ),
}

impl Application {
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
    /// The output is an `Result` wrapping `Vec<__m256d>`, corresponding to the observables
    /// (the expressions passed to `compile`).
    ///
    /// Note: currently, this function only works on X86-64 CPUs with the AVX extension. Intel
    /// introduced the AVX instruction set in 2011; therefore, most intel and AMD processors
    /// support it. If SIMD is not supported, this function returns `None`.
    ///
    #[cfg(target_arch = "x86_64")]
    pub fn call_simd(&mut self, args: &[__m256d]) -> Result<Vec<__m256d>> {
        if let Some(f) = &mut self.compiled_simd {
            {
                let mem = f.mem_mut();
                let states = unsafe {
                    simd_slice_mut(
                        &mut mem[self.first_state * 4..(self.first_state + self.count_states) * 4],
                    )
                };
                states.copy_from_slice(args);
            }

            f.exec(&self.params);

            {
                let mem = f.mem();
                let obs = unsafe {
                    simd_slice(&mem[self.first_obs * 4..(self.first_obs + self.count_obs) * 4])
                };
                let mut res = unsafe { vec![_mm256_setzero_pd(); self.count_obs] };
                res.copy_from_slice(obs);
                Ok(res)
            }
        } else {
            self.prepare_simd();
            if self.compiled_simd.is_some() {
                self.call_simd(args)
            } else {
                Err(anyhow!("cannot compile SIMD"))
            }
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub unsafe fn call_simd(&mut self, args: &[__m256d]) -> Result<Vec<__m256d>> {
        Err(anyhow!("cannot compile SIMD"))
    }

    /// Sets the params and calls the compiled SIMD function.
    ///
    /// `args` is a slice of __m256d values, corresponding to the states.
    ///
    /// `params` is a slice of f64 values.
    ///
    /// The output is a `Result` wrapping a `Vec<__m256d>`, corresponding to the observables
    /// (the expressions passed to `compile`).
    ///
    /// Note: currently, this function only works on X86-64 CPUs with the AVX extension. Intel
    /// introduced the AVX instruction set in 2011; therefore, most intel and AMD processors
    /// support it. If SIMD is not supported, this function returns `None`.
    ///
    #[cfg(target_arch = "x86_64")]
    pub fn call_simd_params(&mut self, args: &[__m256d], params: &[f64]) -> Result<Vec<__m256d>> {
        if let Some(f) = &mut self.compiled_simd {
            {
                let mem = f.mem_mut();
                let states = unsafe {
                    simd_slice_mut(
                        &mut mem[self.first_state * 4..(self.first_state + self.count_states) * 4],
                    )
                };
                states.copy_from_slice(args);
            }

            f.exec(params);

            {
                let mem = f.mem();
                let obs = unsafe {
                    simd_slice(&mem[self.first_obs * 4..(self.first_obs + self.count_obs) * 4])
                };
                let mut res = unsafe { vec![_mm256_setzero_pd(); self.count_obs] };
                res.copy_from_slice(obs);
                Ok(res)
            }
        } else {
            self.prepare_simd();
            if self.compiled_simd.is_some() {
                self.call_simd_params(args, params)
            } else {
                Err(anyhow!("cannot compile SIMD"))
            }
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub unsafe fn call_simd_params(
        &mut self,
        args: &[__m256d],
        params: &[f64],
    ) -> Result<Vec<__m256d>> {
        Err(anyhow!("cannot compile SIMD"))
    }

    /// Returns a fast function.
    ///
    /// `Application` call functions need to copy the input argument slice into
    /// the function memory area and then copy the output to a `Vec`. This process
    /// is acceptable for large and complex functions but incurs a penalty for
    /// small functions. Therefore, for a certain subset of applications, Symjit
    /// can compile a fast funcction and return a function pointer. Examples:
    ///
    /// ```rust
    /// fn test_fast() -> Result<()> {
    ///     let x = Expr::var("x");
    ///     let y = Expr::var("y");
    ///     let z = Expr::var("z");
    ///     let u = &x * &(&y - &z).pow(&Expr::from(2));
    ///
    ///     let mut comp = Compiler::new();
    ///     let mut app = comp.compile(&[x, y, z], &[u])?;
    ///     let f = app.fast_func()?;
    ///
    ///     if let FastFunc::F3(f, _) = f {
    ///         let res = f(3.0, 5.0, 9.0);
    ///         println!("fast\t{:?}", &res);
    ///     }
    ///
    ///     Ok(())
    /// }
    /// ```
    ///
    /// The conditions for a fast function are:
    ///
    /// * A fast function can have 1 to 8 arguments.
    /// * No SIMD and no parameters.
    /// * It returns only a single value.
    ///
    /// If these conditions are met, you can generate a fast functin by calling
    /// `app.fast_func()`, with a return type of `Result<FastFunc>`. `FastFunc` is an
    /// enum with eight variants `F1, `F2`, ..., `F8`, corresponding to
    /// functions with 1 to 8 arguments.
    ///
    pub fn fast_func(&mut self) -> Result<FastFunc<'_>> {
        let f = self.get_fast();

        if let Some(f) = f {
            match self.count_states {
                1 => {
                    let g: extern "C" fn(f64) -> f64 = unsafe { std::mem::transmute(f) };
                    Ok(FastFunc::F1(g, self))
                }
                2 => {
                    let g: extern "C" fn(f64, f64) -> f64 = unsafe { std::mem::transmute(f) };
                    Ok(FastFunc::F2(g, self))
                }
                3 => {
                    let g: extern "C" fn(f64, f64, f64) -> f64 = unsafe { std::mem::transmute(f) };
                    Ok(FastFunc::F3(g, self))
                }
                4 => {
                    let g: extern "C" fn(f64, f64, f64, f64) -> f64 =
                        unsafe { std::mem::transmute(f) };
                    Ok(FastFunc::F4(g, self))
                }
                5 => {
                    let g: extern "C" fn(f64, f64, f64, f64, f64) -> f64 =
                        unsafe { std::mem::transmute(f) };
                    Ok(FastFunc::F5(g, self))
                }
                6 => {
                    let g: extern "C" fn(f64, f64, f64, f64, f64, f64) -> f64 =
                        unsafe { std::mem::transmute(f) };
                    Ok(FastFunc::F6(g, self))
                }
                7 => {
                    let g: extern "C" fn(f64, f64, f64, f64, f64, f64, f64) -> f64 =
                        unsafe { std::mem::transmute(f) };
                    Ok(FastFunc::F7(g, self))
                }
                8 => {
                    let g: extern "C" fn(f64, f64, f64, f64, f64, f64, f64, f64) -> f64 =
                        unsafe { std::mem::transmute(f) };
                    Ok(FastFunc::F8(g, self))
                }
                _ => Err(anyhow!("not a fast function")),
            }
        } else {
            Err(anyhow!("not a fast function"))
        }
    }
}

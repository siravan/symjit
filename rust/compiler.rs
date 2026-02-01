use serde::Deserialize;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{__m256d, _mm256_setzero_pd};

use std::collections::HashMap;

use anyhow::{anyhow, Result};
use num_complex::Complex;
use rayon::prelude::*;

use crate::config::Config;
use crate::defuns::Defuns;
use crate::expr::Expr;
use crate::model::{CellModel, Equation, Program, Variable};
use crate::utils::CompiledFunc;
use crate::Application;

// #[derive(Debug)]
pub struct Compiler {
    config: Config,
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
///     let mut config = Config::default();
///     config.set_opt_level(2);
///     let mut comp = Compiler::with_config(config);
///     let mut app = comp.compile(&[x, y], &[u, v])?;
///     let res = app.call(&[3.0, 5.0]);
///     println!("{:?}", &res);
///
///     Ok(())
/// }
/// ```
impl Compiler {
    /// Creates a new `Compiler` object with default settings.
    pub fn new() -> Compiler {
        Compiler {
            config: Config::default(),
            df: Defuns::new(),
        }
    }

    pub fn with_config(config: Config) -> Compiler {
        Compiler {
            config,
            df: Defuns::new(),
        }
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

        let prog = Program::new(&ml, self.config)?;
        // let df = Defuns::new();
        let mut app = Application::new(prog, &self.df);

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

    /// Generic evaluate function for compiled Symbolica expressions
    #[inline(always)]
    pub fn evaluate<T: Sized + Copy>(&mut self, args: &[T], outs: &mut [T]) {
        if self.prog.config().is_bytecode() {
            let mut stack: Vec<f64> = vec![0.0; self.prog.builder.block().sym_table.num_stack];
            let mut regs = vec![0.0; 16];

            let outs: &mut [f64] = unsafe { std::mem::transmute(outs) };
            let args: &[f64] = unsafe { std::mem::transmute(args) };

            self.mir.exec_instruction(outs, &mut stack, &mut regs, args);

            return;
        }

        let f = self.compiled.func();

        f(
            outs.as_ptr() as *mut f64,
            std::ptr::null(),
            0,
            args.as_ptr() as *const f64,
        );
    }

    /// Generic evaluate_single function for compiled Symbolica expressions
    #[inline(always)]
    pub fn evaluate_single<T: Sized + Copy + Default>(&mut self, args: &[T]) -> T {
        let mut outs = [T::default(); 1];
        self.evaluate(args, &mut outs);
        outs[0]
    }

    /// Generic SIMD evaluate function for compiled Symbolica expressions
    #[inline(always)]
    pub fn evaluate_simd<T: Sized + Copy>(&mut self, args: &[T], outs: &mut [T]) {
        if let Some(g) = &mut self.compiled_simd {
            let f = g.func();

            f(
                outs.as_ptr() as *mut f64,
                std::ptr::null(),
                0,
                args.as_ptr() as *const f64,
            );
        }
    }

    /// Generic SIMD evaluate_single function for compiled Symbolica expressions
    #[inline(always)]
    pub fn evaluate_simd_single<T: Sized + Copy + Default>(&mut self, args: &[T]) -> T {
        let mut outs = [T::default(); 1];
        self.evaluate_simd(args, &mut outs);
        outs[0]
    }

    fn evaluate_row(
        args: &[f64],
        args_idx: usize,
        outs: &[f64],
        outs_idx: usize,
        f: CompiledFunc<f64>,
    ) {
        unsafe {
            f(
                (outs.as_ptr() as *const f64).add(outs_idx),
                std::ptr::null(),
                0,
                (args.as_ptr() as *const f64).add(args_idx),
            );
        }
    }

    fn evaluate_row_simd(
        args: &[f64],
        args_idx: usize,
        outs: &[f64],
        outs_idx: usize,
        f: CompiledFunc<f64>,
    ) {
        unsafe {
            f(
                outs.as_ptr().add(outs_idx),
                std::ptr::null(),
                1,
                args.as_ptr().add(args_idx),
            );
        }
    }

    /// Generic evaluate function for compiled Symbolica expressions
    fn evaluate_matrix_with_threads(&mut self, args: &[f64], outs: &mut [f64], n: usize) {
        let count_params = self.count_params;
        let count_obs = self.count_obs;
        let f = self.compiled.func();

        (0..n)
            .into_par_iter()
            .for_each(|t| Self::evaluate_row(args, t * count_params, outs, t * count_obs, f));
    }

    /// Generic evaluate function for compiled Symbolica expressions
    fn evaluate_matrix_without_threads(&mut self, args: &[f64], outs: &mut [f64], n: usize) {
        let count_params = self.count_params;
        let count_obs = self.count_obs;
        let f = self.compiled.func();

        for t in 0..n {
            Self::evaluate_row(args, t * count_params, outs, t * count_obs, f);
        }
    }

    fn evaluate_matrix_with_threads_simd(&mut self, args: &[f64], outs: &mut [f64], n: usize) {
        let count_params = self.count_params;
        let count_obs = self.count_obs;

        if let Some(compiled) = &self.compiled_simd {
            let g = compiled.func();
            let l = compiled.count_lanes();

            (0..n / l).into_par_iter().for_each(|t| {
                Self::evaluate_row_simd(args, t * count_params * l, outs, t * count_obs * l, g)
            });

            let f = self.compiled.func();
            for t in l * (n / l)..n {
                Self::evaluate_row(args, t * count_params, outs, t * count_obs, f);
            }
        }
    }

    fn evaluate_matrix_without_threads_simd(&mut self, args: &[f64], outs: &mut [f64], n: usize) {
        let count_params = self.count_params;
        let count_obs = self.count_obs;

        if let Some(compiled) = &self.compiled_simd {
            let g = compiled.func();
            let l = compiled.count_lanes();

            for t in 0..n / l {
                Self::evaluate_row_simd(args, t * count_params * l, outs, t * count_obs * l, g);
            }

            let f = self.compiled.func();
            for t in l * (n / l)..n {
                Self::evaluate_row(args, t * count_params, outs, t * count_obs, f);
            }
        }
    }

    /// Generic evaluate function for compiled Symbolica expressions
    pub fn evaluate_matrix(&mut self, args: &[f64], outs: &mut [f64], n: usize) {
        if self.use_threads {
            if self.compiled_simd.is_some() {
                self.evaluate_matrix_with_threads_simd(args, outs, n);
            } else {
                self.evaluate_matrix_with_threads(args, outs, n);
            }
        } else {
            if self.compiled_simd.is_some() {
                self.evaluate_matrix_without_threads_simd(args, outs, n);
            } else {
                self.evaluate_matrix_without_threads(args, outs, n);
            }
        }
    }

    /// Generic evaluate function for compiled Symbolica expressions
    pub fn evaluate_simd_matrix<T: Sized + Copy>(&mut self, args: &[T], outs: &mut [T], n: usize) {
        let args_size = args.len() / n;
        let outs_size = outs.len() / n;

        for (p, q) in args.chunks(args_size).zip(outs.chunks_mut(outs_size)) {
            self.evaluate_simd(p, q);
        }
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

/************************* Symbolica *****************************/

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BuiltinSymbol(u32);

impl<'de> serde::Deserialize<'de> for BuiltinSymbol {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let id: u32 = u32::deserialize(deserializer)?;
        Ok(BuiltinSymbol(id))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
enum Slot {
    /// An entry in the list of parameters.
    Param(usize),
    /// An entry in the list of constants.
    Const(usize),
    /// An entry in the list of temporary storage.
    Temp(usize),
    /// An entry in the list of results.
    Out(usize),
    /// Static-Single-Assingment Form
    Static(usize),
}

#[derive(Debug, Clone, Deserialize)]
enum Instruction {
    /// `Add(o, [i0,...,i_n])` means `o = i0 + ... + i_n`.
    Add(Slot, Vec<Slot>),
    /// `Mul(o, [i0,...,i_n])` means `o = i0 * ... * i_n`.
    Mul(Slot, Vec<Slot>),
    /// `Pow(o, b, e)` means `o = b^e`.
    Pow(Slot, Slot, i64),
    /// `Powf(o, b, e)` means `o = b^e`.
    Powf(Slot, Slot, Slot),
    /// `Fun(o, s, a)` means `o = s(a)`, where `s` is assumed to
    /// be a built-in function such as `sin`.
    Fun(Slot, BuiltinSymbol, Slot),
    /// `ExternalFun(o, s, a,...)` means `o = s(a, ...)`, where `s` is an external function.
    ExternalFun(Slot, String, Vec<Slot>),
    /// `Assign(o, v)` means `o = v`.
    Assign(Slot, Slot),
    /// `IfElse(cond, label)` means jump to `label` if `cond` is zero.
    IfElse(Slot, usize),
    /// Unconditional jump to `label`.
    Goto(usize),
    /// A position in the instruction list to jump to.
    Label(usize),
    /// `Join(o, cond, t, f)` means `o = cond ? t : f`.
    Join(Slot, Slot, Slot, Slot),
}

#[derive(Debug, Clone, Deserialize)]
enum Value {
    Single(f64),
}

impl Value {
    fn value(&self) -> f64 {
        let Value::Single(x) = self;
        *x
    }
}

#[derive(Debug, Clone, Deserialize)]
struct Rational {
    numerator: Value,
    denominator: Value,
}

impl Rational {
    fn value(&self) -> f64 {
        self.numerator.value() / self.denominator.value()
    }
}

#[derive(Debug, Clone, Deserialize)]
struct ComplexRational {
    re: Rational,
    im: Rational,
}

impl ComplexRational {
    fn value(&self) -> Complex<f64> {
        Complex::new(self.re.value(), self.im.value())
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum ConstType {
    Complex(ComplexRational),
    Single(f64),
}

impl ConstType {
    fn value(&self) -> Complex<f64> {
        match self {
            ConstType::Single(x) => Complex::new(*x, 0.0),
            ConstType::Complex(x) => x.value(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct SymbolicaModel(Vec<Instruction>, usize, Vec<ConstType>);

/// Translates Symbolica IR (generated by export_instructions) into a Symjit Model
#[derive(Debug, Clone)]
struct Translator {
    consts: Vec<Complex<f64>>, // constants
    count_params: usize,
    count_statics: usize,
    eqs: Vec<Equation>,            // Symjit Equations (output)
    temps: HashMap<usize, Slot>,   // Temp idx => Static idx
    counts: HashMap<usize, usize>, // Static idx => number of usage on the RHS
    cache: HashMap<usize, Expr>,   // cache of Static variables (Static idx => Expr)
    outs: HashMap<usize, Expr>,    // cache of Outs (Out idx => Expr)
}

impl Translator {
    fn new() -> Translator {
        Translator {
            consts: Vec::new(),
            count_params: 0,
            count_statics: 0,
            eqs: Vec::new(),
            temps: HashMap::new(),
            counts: HashMap::new(),
            cache: HashMap::new(),
            outs: HashMap::new(),
        }
    }

    /// The first pass by converting Symbolica IR into
    /// Static-Single-Assingment (SSA) Form
    fn convert(&mut self, model: &SymbolicaModel) -> Result<Vec<Instruction>> {
        let mut ssa: Vec<Instruction> = Vec::new();

        for line in model.0.iter() {
            match line {
                Instruction::Add(lhs, args) => {
                    let args = self.consume_list(args)?;
                    let lhs = self.produce(lhs)?;
                    ssa.push(Instruction::Add(lhs, args));
                }
                Instruction::Mul(lhs, args) => {
                    let args = self.consume_list(args)?;
                    let lhs = self.produce(lhs)?;
                    ssa.push(Instruction::Mul(lhs, args));
                }
                Instruction::Pow(lhs, arg, p) => {
                    let arg = self.consume(arg)?;
                    let lhs = self.produce(lhs)?;
                    ssa.push(Instruction::Pow(lhs, arg, *p));
                }
                Instruction::Powf(lhs, arg, p) => {
                    let arg = self.consume(arg)?;
                    let p = self.consume(p)?;
                    let lhs = self.produce(lhs)?;
                    ssa.push(Instruction::Powf(lhs, arg, p));
                }
                Instruction::Assign(lhs, rhs) => {
                    let rhs = self.consume(rhs)?;
                    let lhs = self.produce(lhs)?;
                    ssa.push(Instruction::Assign(lhs, rhs));
                }
                Instruction::Fun(lhs, fun, arg) => {
                    let arg = self.consume(arg)?;
                    let lhs = self.produce(lhs)?;
                    ssa.push(Instruction::Fun(lhs, *fun, arg));
                }
                Instruction::Join(lhs, cond, true_val, false_val) => {
                    let cond = self.consume(cond)?;
                    let true_val = self.consume(true_val)?;
                    let false_val = self.consume(false_val)?;
                    let lhs = self.produce(lhs)?;
                    ssa.push(Instruction::Join(lhs, cond, true_val, false_val));
                }
                Instruction::Label(id) => {
                    ssa.push(Instruction::Label(*id));
                }
                Instruction::IfElse(cond, id) => {
                    let cond = self.consume(cond)?;
                    ssa.push(Instruction::IfElse(cond, *id));
                }
                Instruction::Goto(id) => {
                    ssa.push(Instruction::Goto(*id));
                }
                Instruction::ExternalFun(lhs, op, args) => {
                    let args = self.consume_list(args)?;
                    let lhs = self.produce(lhs)?;
                    ssa.push(Instruction::ExternalFun(lhs, op.clone(), args));
                }
            }
        }

        Ok(ssa)
    }

    /// Produces a new Static variable if needed.
    /// slot should be an LHS.
    fn produce(&mut self, slot: &Slot) -> Result<Slot> {
        match slot {
            Slot::Temp(idx) => {
                let s = Slot::Static(self.count_statics);
                self.counts.insert(self.count_statics, 0);
                self.count_statics += 1;
                self.temps.insert(*idx, s);
                Ok(s)
            }
            Slot::Out(idx) => Ok(Slot::Out(*idx)),
            _ => Err(anyhow!("unacceptable lhs.")),
        }
    }

    /// Consumes a slot.
    /// slot should be an RHS.
    fn consume(&mut self, slot: &Slot) -> Result<Slot> {
        match slot {
            Slot::Temp(idx) => {
                if let Some(Slot::Static(s)) = self.temps.get(idx) {
                    *self.counts.get_mut(s).unwrap() += 1;
                    Ok(Slot::Static(*s))
                } else {
                    Err(anyhow!("Not a static reg."))
                }
            }
            Slot::Out(idx) => Ok(Slot::Out(*idx)),
            Slot::Param(idx) => Ok(Slot::Param(*idx)),
            Slot::Const(idx) => Ok(Slot::Const(*idx)),
            Slot::Static(_) => Err(anyhow!("Undefined Static.")),
        }
    }

    fn consume_list(&mut self, slots: &[Slot]) -> Result<Vec<Slot>> {
        slots.iter().map(|s| self.consume(s)).collect()
    }

    /// The second pass. It translates the SSA-form into a Symjit model.
    fn translate(&mut self, model: &SymbolicaModel) -> Result<CellModel> {
        // self.consts = model.2.iter().map(|x| x.value().re).collect::<Vec<f64>>();
        self.consts = model
            .2
            .iter()
            .map(|x| Complex::new(x.value().re, x.value().im))
            .collect::<Vec<Complex<f64>>>();

        let ssa = self.convert(model)?;

        for line in ssa.iter() {
            match line {
                Instruction::Add(lhs, args) => self.translate_nary("plus", lhs, args)?,
                Instruction::Mul(lhs, args) => self.translate_nary("times", lhs, args)?,
                Instruction::Pow(lhs, arg, p) => {
                    let p = Expr::from(*p as f64);
                    self.translate_pow(lhs, arg, &p)?
                }
                Instruction::Powf(lhs, arg, p) => {
                    let p = self.expr(p);
                    self.translate_pow(lhs, arg, &p)?
                }
                Instruction::Assign(lhs, rhs) => self.translate_assign(lhs, rhs)?,
                Instruction::Fun(lhs, fun, arg) => self.translate_fun(lhs, fun, arg)?,
                Instruction::Join(lhs, cond, true_val, false_val) => {
                    self.translate_join(lhs, cond, true_val, false_val)?
                }
                Instruction::Label(id) => {
                    self.eqs
                        .push(Expr::equation(&Expr::Special, &Expr::Label { id: *id }));
                }
                Instruction::IfElse(cond, id) => self.translate_ifelse(cond, *id)?,
                Instruction::Goto(id) => self.translate_goto(*id)?,
                Instruction::ExternalFun(lhs, op, args) => {
                    self.translate_external_fun(lhs, op, args)?
                }
            }
        }

        // Important! Outs are cached and should be written to final outputs.
        for (k, eq) in self.outs.iter() {
            let out = Expr::var(&format!("Out{}", k));
            self.eqs.push(Expr::equation(&out, eq));
        }

        let params: Vec<Variable> = (0..=self.count_params)
            .map(|idx| self.expr(&Slot::Param(idx)).to_variable().unwrap())
            .collect();

        Ok(CellModel {
            iv: Expr::var("$_").to_variable().unwrap(),
            params,
            states: Vec::new(),
            algs: Vec::new(),
            odes: Vec::new(),
            obs: self.eqs.clone(),
        })
    }

    // The counterpark of consume for the second-pass
    fn expr(&mut self, slot: &Slot) -> Expr {
        match slot {
            Slot::Param(idx) => {
                self.count_params = self.count_params.max(*idx);
                Expr::var(&format!("Param{}", idx))
            }
            Slot::Out(idx) => {
                if let Some(e) = self.outs.get(idx) {
                    e.clone()
                } else {
                    Expr::var(&format!("Out{}", idx))
                }
            }
            Slot::Temp(idx) => Expr::var(&format!("__Temp{}", idx)),
            Slot::Const(idx) => {
                let val = self.consts[*idx];
                if val.im != 0.0 {
                    Expr::binary("complex", &Expr::from(val.re), &Expr::from(val.im))
                } else {
                    Expr::from(self.consts[*idx].re)
                }
            }
            Slot::Static(idx) => self
                .cache
                .remove(idx)
                .unwrap_or(Expr::var(&format!("__Static{}", idx))),
        }
    }

    // The counterpart of produce for the second-pass
    fn assign(&mut self, lhs: &Slot, rhs: Expr) -> Result<()> {
        if let Slot::Static(idx) = lhs {
            // Important! If a static variable is used only once, it
            // is pushed into the cache to be incorporated into the
            // destination expression tree.
            if self.counts.get(idx).is_some_and(|c| *c == 1) {
                self.cache.insert(*idx, rhs);
                return Ok(());
            }
        }

        if let Slot::Out(idx) = lhs {
            self.outs.insert(*idx, rhs.clone());
            return Ok(());
        }

        let lhs = self.expr(lhs);
        self.eqs.push(Expr::equation(&lhs, &rhs));
        Ok(())
    }

    fn translate_nary(&mut self, op: &str, lhs: &Slot, args: &[Slot]) -> Result<()> {
        let args: Vec<Expr> = args.iter().map(|x| self.expr(x)).collect();
        let p: Vec<&Expr> = args.iter().collect();
        self.assign(lhs, Expr::nary(op, &p))
    }

    fn translate_pow(&mut self, lhs: &Slot, arg: &Slot, power: &Expr) -> Result<()> {
        let arg = self.expr(arg);
        self.assign(lhs, Expr::binary("power", &arg, power))
    }

    fn translate_assign(&mut self, lhs: &Slot, rhs: &Slot) -> Result<()> {
        let rhs = self.expr(rhs);
        self.assign(lhs, rhs)
    }

    fn translate_fun(&mut self, lhs: &Slot, fun: &BuiltinSymbol, arg: &Slot) -> Result<()> {
        let arg = self.expr(arg);

        let op = match fun.0 {
            2 => "exp",
            3 => "ln",
            4 => "sin",
            5 => "cos",
            6 => "root",
            7 => "conjugate",
            _ => return Err(anyhow!("function is not defined.")),
        };

        self.assign(lhs, Expr::unary(op, &arg))
    }

    fn translate_external_fun(&mut self, lhs: &Slot, op: &str, args: &[Slot]) -> Result<()> {
        match args.len() {
            1 => {
                let arg = self.expr(&args[0]);
                self.assign(lhs, Expr::unary(op, &arg))?;
            }
            2 => {
                let l = self.expr(&args[0]);
                let r = self.expr(&args[1]);
                self.assign(lhs, Expr::binary(op, &l, &r))?;
            }
            _ => {
                return Err(anyhow!(
                    "only unary and binary external functions are supported"
                ))
            }
        }

        Ok(())
    }

    fn translate_join(
        &mut self,
        lhs: &Slot,
        cond: &Slot,
        true_val: &Slot,
        false_val: &Slot,
    ) -> Result<()> {
        let cond = Expr::binary(
            "gt",
            &Expr::unary("abs", &self.expr(cond)),
            &Expr::from(f64::EPSILON),
        );
        let t = self.expr(true_val);
        let f = self.expr(false_val);
        self.assign(lhs, cond.ifelse(&t, &f))
    }

    fn translate_ifelse(&mut self, cond: &Slot, id: usize) -> Result<()> {
        let cond = Expr::binary(
            "lt",
            &Expr::unary("abs", &self.expr(cond)),
            &Expr::from(f64::EPSILON),
        );

        self.eqs.push(Expr::equation(
            &Expr::Special,
            &Expr::IfElse {
                cond: Box::new(cond),
                id,
            },
        ));
        Ok(())
    }

    fn translate_goto(&mut self, id: usize) -> Result<()> {
        let cond = Expr::from(f64::from_bits(!0)); // cond = true
        self.eqs.push(Expr::equation(
            &Expr::Special,
            &Expr::IfElse {
                cond: Box::new(cond),
                id,
            },
        ));

        Ok(())
    }
}

impl Compiler {
    /// Compiles a Symbolica model.
    ///
    /// `json` is the JSON-encoded output of Symbolica `export_instructions`.
    ///
    /// Example:
    ///
    /// ```rust
    /// let params = vec![parse!("x"), parse!("y")];
    /// let eval = parse!("x + y^2")
    ///     .evaluator(&FunctionMap::new(), &params, OptimizationSettings::default())?
    ///
    /// let json = serde_json::to_string(&eval.export_instructions())?;
    /// let mut comp = Compiler::new();
    /// let mut app = comp.translate(&json)?;
    /// assert!(app.evaluate_single(&[2.0, 3.0]) == 11.0);
    /// ```
    pub fn translate(&mut self, json: &str) -> Result<Application> {
        self.config.set_symbolica(true);

        let model: SymbolicaModel = serde_json::from_str(json)?;
        let mut translator = Translator::new();
        let ml = translator.translate(&model)?;

        let prog = Program::new(&ml, self.config)?;
        let df = Defuns::new();
        let mut app = Application::new(prog, &df)?;

        app.prepare_simd();

        // #[cfg(target_arch = "aarch64")]
        // if let Ok(app) = &mut app {
        //     // this is a hack to give enough delay to prevent a bus error
        //     app.dump("dump.bin", "scalar");
        //     std::fs::remove_file("dump.bin")?;
        // };

        Ok(app)
    }
}

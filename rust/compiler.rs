use crate::CompileExt;

use anyhow::{anyhow, Result};

use super::applet::{recast_as_f64, recast_as_f64_mut};
use super::compilation::Compilation;
use super::composer::Composer;
use super::config::Config;
use super::expr::Expr;
use super::instruction::SymbolicaModel;
use super::machine::Compiled;
use super::parser::Parser;
use super::runnable::Application;
use super::translator::Translator;
use super::types::Element;

// #[derive(Debug)]
pub struct Compiler {
    config: Config,
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
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
impl Compiler {
    /// Creates a new `Compiler` object with default settings.
    pub fn new() -> Compiler {
        Compiler {
            config: Config::default(),
        }
    }

    pub fn with_config(config: Config) -> Compiler {
        Compiler { config }
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
        let c = Compilation::from_exprs(states, obs, params, self.config.clone())?;
        Application::from_compilation(c)
    }
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
        if let Some(f) = &mut self.compiled {
            {
                let mem = f.mem_mut();
                let states = &mut mem[self.first_state..self.first_state + self.count_states];
                states.copy_from_slice(args);
            }

            f.exec(&self.params[..]);

            let obs = {
                let mem = f.mem();
                &mem[self.first_obs..self.first_obs + self.count_obs]
            };

            obs.to_vec()
        } else {
            Vec::new()
        }
    }

    /// Sets the params and calls the compiled function.
    ///
    /// `args` is a slice of f64 values, corresponding to the states.
    /// `params` is a slice of f64 values, corresponding to the params.
    ///
    /// The output is a `Vec<f64>`, corresponding to the observables (the expressions passed
    /// to `compile`).
    pub fn call_params(&mut self, args: &[f64], params: &[f64]) -> Vec<f64> {
        if let Some(f) = &mut self.compiled {
            {
                let mem = f.mem_mut();
                let states = &mut mem[self.first_state..self.first_state + self.count_states];
                states.copy_from_slice(args);
            }

            f.exec(params);

            let obs = {
                let mem = f.mem();
                &mem[self.first_obs..self.first_obs + self.count_obs]
            };

            obs.to_vec()
        } else {
            Vec::new()
        }
    }

    pub fn interpret<T>(&mut self, args: &[T], outs: &mut [T])
    where
        T: Element,
    {
        let args = recast_as_f64(args);
        let outs = recast_as_f64_mut(outs);

        let mut regs = [0.0; 32];
        self.bytecode
            .mir
            .exec_instruction(outs, &mut self.bytecode.stack, &mut regs, args);
    }

    pub fn interpret_matrix(&mut self, args: &[f64], outs: &mut [f64], n: usize) {
        let count_params = self.count_params;
        let count_obs = self.count_obs;

        for i in 0..n {
            self.interpret(
                &args[i * count_params..(i + 1) * count_params],
                &mut outs[i * count_obs..(i + 1) * count_obs],
            );
        }
    }

    /// Generic evaluate function for compiled Symbolica expressions
    pub fn evaluate<T>(&self, args: &[T], outs: &mut [T])
    where
        T: Element,
    {
        self.as_applet().evaluate(args, outs);
    }

    /// Generic evaluate_single function for compiled Symbolica expressions
    #[inline(always)]
    pub fn evaluate_single<T>(&self, args: &[T]) -> T
    where
        T: Element + Copy,
    {
        self.as_applet().evaluate_single(args)
    }

    /// Generic evaluate function for compiled Symbolica expressions
    /// The main entry point to compute matrices.
    /// The actual dispatched method depends on the configuration and the
    /// type of the arguments.
    pub fn evaluate_matrix<T>(&self, args: &[T], outs: &mut [T], n: usize)
    where
        T: Element,
    {
        self.as_applet().evaluate_matrix(args, outs, n);
    }

    /// Returns a fast function.
    ///
    /// `Application` call functions need to copy the input argument slice into
    /// the function memory area and then copy the output to a `Vec`. This process
    /// is acceptable for large and complex functions but incurs a penalty for
    /// small functions. Therefore, for a certain subset of applications, Symjit
    /// can compile a fast funcction and return a function pointer. Examples:
    ///
    ///
    /// The conditions for a fast function are:
    ///
    /// * A fast function can have 1 to 8 arguments.
    /// * No SIMD and no parameters.
    /// * It returns only a single value.
    ///
    /// If these conditions are met, you can generate a fast functin by calling
    /// app.fast_func(), with a return type of Result<FastFunc>. FastFunc is an
    /// enum with eight variants F1, `F2`, ..., `F8, corresponding to
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

impl Compiler {
    pub fn translate(&mut self, json: String, num_params: usize) -> Result<Application> {
        let mut translator = Translator::new(self.config.clone());

        let model: SymbolicaModel = if json.starts_with("[[{") {
            serde_json::from_str(json.as_str())?
        } else {
            Parser::new(json).parse()?
        };

        translator.parse_model(&model)?;
        translator.set_num_params(num_params);
        //let (ml, reals) = translator.translate()?;

        //let prog = Program::new(&ml, translator.config)?;
        //let mut app = Application::new(prog, reals)?;
        let app = translator.compile()?;

        Ok(app)
    }
}

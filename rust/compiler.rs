use std::collections::{HashMap, HashSet};
use std::io::Write;

use anyhow::{anyhow, Result};
use num_complex::Complex;

use super::applet::{recast_as_f64, recast_as_f64_mut};
use super::builder::Builder;
use super::code::VirtualTable;
use super::composer::{Composer, DirectTranslator};
use super::config::{Config, SLICE_CAP};
use super::expr::Expr;
use super::instruction::{rationalize_complex, BuiltinSymbol, Instruction, Slot, SymbolicaModel};
use super::model::{CellModel, Equation, Program, Variable};
use super::node::Node;
use super::operation::Operation;
use super::parser::Parser;
use super::symbol::Loc;
use super::types::Element;
use super::utils::{advance_id, Compiled};
use super::Application;

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

        let prog = Program::new(&ml, self.config.clone())?;
        // let df = Defuns::new();
        let app = Application::new(prog, HashSet::new())?;
        // app.prepare_simd();

        // #[cfg(target_arch = "aarch64")]
        // if let Ok(app) = &app {
        //     // this is a hack to give enough delay to prevent a bus error
        //     app.dump("dump.bin", "scalar");
        //     std::fs::remove_file("dump.bin")?;
        // };

        Ok(app)
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

pub struct Translator {
    composer: Box<dyn Composer>,
    config: Config,
    salt: String,
    instructions: Vec<String>,
    count_temp: usize,
    constants: Vec<String>,
    spy: bool,
}

impl Translator {
    pub fn new(config: Config) -> Translator {
        // let salt = Alphanumeric.sample_string(&mut rand::rng(), 8);
        //let salt = format!("{:05}", advance_id());
        let salt = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
            .to_string();

        let composer: Box<dyn Composer> = if config.direct() {
            Box::new(DirectTranslator::new(config.clone()))
        } else {
            Box::new(IndirectTranslator::new(config.clone(), &salt))
        };

        let spy = config.debug_instructions();

        Translator {
            composer,
            config,
            salt,
            instructions: Vec::new(),
            count_temp: 0,
            constants: Vec::new(),
            spy,
        }
    }

    pub fn parse_model(&mut self, model: &SymbolicaModel) -> Result<()> {
        for c in model.2.iter() {
            let val = Complex::new(c.value().re, c.value().im);
            //self.consts.push(val);
            self.append_constant(val)?;
        }

        self.convert(model)?;
        Ok(())
    }

    /// The first pass by converting Symbolica IR into
    /// Static-Single-Assingment (SSA) Form
    fn convert(&mut self, model: &SymbolicaModel) -> Result<()> {
        for line in model.0.iter() {
            match line {
                Instruction::Add(lhs, args, num_reals) => self.append_add(lhs, args, *num_reals)?,
                Instruction::Mul(lhs, args, num_reals) => self.append_mul(lhs, args, *num_reals)?,
                Instruction::Pow(lhs, arg, p, is_real) => {
                    self.append_pow(lhs, arg, *p, *is_real)?
                }
                Instruction::Powf(lhs, arg, p, is_real) => {
                    self.append_powf(lhs, arg, p, *is_real)?
                }
                Instruction::Assign(lhs, rhs) => self.append_assign(lhs, rhs)?,
                Instruction::Fun(lhs, fun, args, is_real) => {
                    self.append_fun(lhs, fun, args, *is_real)?
                }
                Instruction::Join(lhs, cond, true_val, false_val) => {
                    // self.depth -= 1;
                    self.append_join(lhs, cond, true_val, false_val)?
                }
                Instruction::Label(id) => self.append_label(*id)?,
                Instruction::IfElse(cond, id) => {
                    self.append_if_else(cond, *id)?;
                    // self.depth += 1;
                }
                Instruction::Goto(id) => self.append_goto(*id)?,
                Instruction::ExternalFun(lhs, op, args) => {
                    self.append_external_fun(lhs, op, args)?
                }
            }
        }

        Ok(())
    }

    fn dump(&self) {
        let mut fs = std::fs::File::create(format!("instructions_{}.txt", self.salt)).unwrap();
        let _ = writeln!(
            fs,
            "([{}], {}, [{}])",
            self.instructions.join(",\n"),
            self.count_temp,
            self.constants.join(", "),
        );
    }
}

impl Composer for Translator {
    fn append_constant(&mut self, z: Complex<f64>) -> Result<usize> {
        if self.spy {
            self.constants.push(rationalize_complex(z));
        }
        self.composer.append_constant(z)
    }

    fn append_add(&mut self, lhs: &Slot, args: &[Slot], num_reals: usize) -> Result<()> {
        if self.spy {
            self.instructions
                .push(Instruction::Add(*lhs, args.to_vec(), num_reals).to_string());
        }
        self.composer.append_add(lhs, args, num_reals)
    }

    fn append_mul(&mut self, lhs: &Slot, args: &[Slot], num_reals: usize) -> Result<()> {
        if self.spy {
            self.instructions
                .push(Instruction::Mul(*lhs, args.to_vec(), num_reals).to_string());
        }
        self.composer.append_mul(lhs, args, num_reals)
    }

    fn append_pow(&mut self, lhs: &Slot, arg: &Slot, p: i64, is_real: bool) -> Result<()> {
        if self.spy {
            self.instructions
                .push(Instruction::Pow(*lhs, *arg, p, is_real).to_string());
        }
        self.composer.append_pow(lhs, arg, p, is_real)
    }

    fn append_powf(&mut self, lhs: &Slot, arg: &Slot, p: &Slot, is_real: bool) -> Result<()> {
        if self.spy {
            self.instructions
                .push(Instruction::Powf(*lhs, *arg, *p, is_real).to_string());
        }
        self.composer.append_powf(lhs, arg, p, is_real)
    }

    fn append_assign(&mut self, lhs: &Slot, rhs: &Slot) -> Result<()> {
        if self.spy {
            self.instructions
                .push(Instruction::Assign(*lhs, *rhs).to_string());
        }
        self.composer.append_assign(lhs, rhs)
    }

    fn append_label(&mut self, id: usize) -> Result<()> {
        if self.spy {
            self.instructions.push(Instruction::Label(id).to_string());
        }
        self.composer.append_label(id)
    }

    fn append_if_else(&mut self, cond: &Slot, id: usize) -> Result<()> {
        if self.spy {
            self.instructions
                .push(Instruction::IfElse(*cond, id).to_string());
        }
        self.composer.append_if_else(cond, id)
    }

    fn append_goto(&mut self, id: usize) -> Result<()> {
        if self.spy {
            self.instructions.push(Instruction::Goto(id).to_string());
        }
        self.composer.append_goto(id)
    }

    fn append_external_fun(&mut self, lhs: &Slot, op: &str, args: &[Slot]) -> Result<()> {
        self.composer.append_external_fun(lhs, op, args)
    }

    fn append_fun_v1(
        &mut self,
        lhs: &Slot,
        fun: &BuiltinSymbol,
        arg: &Slot,
        is_real: bool,
    ) -> Result<()> {
        self.composer.append_fun_v1(lhs, fun, arg, is_real)
    }

    fn append_fun(&mut self, lhs: &Slot, fun: &str, args: &[Slot], is_real: bool) -> Result<()> {
        if self.spy {
            self.instructions
                .push(Instruction::Fun(*lhs, fun.into(), args.to_vec(), is_real).to_string());
        }
        self.composer.append_fun(lhs, fun, args, is_real)
    }

    fn append_join(
        &mut self,
        lhs: &Slot,
        cond: &Slot,
        true_val: &Slot,
        false_val: &Slot,
    ) -> Result<()> {
        if self.spy {
            self.instructions
                .push(Instruction::Join(*lhs, *cond, *true_val, *false_val).to_string());
        }
        self.composer.append_join(lhs, cond, true_val, false_val)
    }

    fn set_num_params(&mut self, num_params: usize) {
        self.composer.set_num_params(num_params);
    }

    fn compile(&mut self) -> Result<Application> {
        if self.spy {
            self.dump();
        }

        let mut app = self.composer.compile()?;

        if self.config.debug_stats() {
            app.dump(&format!("symjit_{}_stats.txt", self.salt), "stats");
        };

        if self.config.debug_bytecode() {
            app.dump(&format!("symjit_{}_bytecode.txt", self.salt), "bytecode");
        };

        if self.config.debug_scalar() {
            app.dump(&format!("symjit_{}_scalar.bin", self.salt), "scalar");
        };

        if self.config.debug_scalar() {
            app.dump(&format!("symjit_{}_simd.bin", self.salt), "simd");
        };

        Ok(app)
    }
}

/// Translates Symbolica IR (generated by export_instructions) into a Symjit Model
#[derive(Debug, Clone)]
pub struct IndirectTranslator {
    config: Config,
    ssa: Vec<Instruction>,
    consts: Vec<Complex<f64>>, // constants
    count_params: usize,
    count_outs: usize,
    count_statics: usize,
    temps: HashMap<Slot, Slot>,    // Temp/Out Slot => Static Slot
    counts: HashMap<usize, usize>, // Static idx => number of usage on the RHS
    //cache: HashMap<usize, Node>,   // cache of Static variables (Static idx => Node)
    cache: Vec<Option<Node>>,
    outs: HashMap<usize, Slot>, // cache of Outs (Out idx => Static Slot)
    reals: HashSet<Loc>,        // list of real Loc
    join_rhs: HashSet<Slot>, // the set of Static slots used in the RHS of a Join operation (cannot move)
    num_params: usize,
    last_label: usize,
    builder: Builder,
    slot_size: usize,
    arena_mode: bool,
}

impl Composer for IndirectTranslator {
    fn append_constant(&mut self, z: Complex<f64>) -> Result<usize> {
        self.consts.push(z);
        if self.config.is_complex() {
            Ok((self.consts.len() - 1) / 2)
        } else {
            Ok(self.consts.len() - 1)
        }
    }

    fn append_add(&mut self, lhs: &Slot, args: &[Slot], num_reals: usize) -> Result<()> {
        let args = self.consume_list(args)?;
        let lhs = self.produce(lhs)?;
        self.ssa.push(Instruction::Add(lhs, args, num_reals));
        Ok(())
    }

    fn append_mul(&mut self, lhs: &Slot, args: &[Slot], num_reals: usize) -> Result<()> {
        let args = self.consume_list(args)?;
        let lhs = self.produce(lhs)?;
        self.ssa.push(Instruction::Mul(lhs, args, num_reals));
        Ok(())
    }

    fn append_pow(&mut self, lhs: &Slot, arg: &Slot, p: i64, is_real: bool) -> Result<()> {
        let arg = self.consume(arg)?;
        let lhs = self.produce(lhs)?;
        self.ssa.push(Instruction::Pow(lhs, arg, p, is_real));
        Ok(())
    }

    fn append_powf(&mut self, lhs: &Slot, arg: &Slot, p: &Slot, is_real: bool) -> Result<()> {
        let arg = self.consume(arg)?;
        let p = self.consume(p)?;
        let lhs = self.produce(lhs)?;
        self.ssa.push(Instruction::Powf(lhs, arg, p, is_real));
        Ok(())
    }

    fn append_assign(&mut self, lhs: &Slot, rhs: &Slot) -> Result<()> {
        let rhs = self.consume(rhs)?;
        let lhs = self.produce(lhs)?;
        self.ssa.push(Instruction::Assign(lhs, rhs));
        Ok(())
    }

    fn append_label(&mut self, id: usize) -> Result<()> {
        self.ssa.push(Instruction::Label(id));
        Ok(())
    }

    fn append_if_else(&mut self, cond: &Slot, id: usize) -> Result<()> {
        let cond = self.consume(cond)?;
        self.ssa.push(Instruction::IfElse(cond, id));
        Ok(())
    }

    fn append_goto(&mut self, id: usize) -> Result<()> {
        self.last_label = self.last_label.max(id);
        self.ssa.push(Instruction::Goto(id));
        Ok(())
    }

    fn append_external_fun(&mut self, lhs: &Slot, op: &str, args: &[Slot]) -> Result<()> {
        let args = self.consume_list(args)?;
        let lhs = self.produce(lhs)?;
        self.ssa
            .push(Instruction::ExternalFun(lhs, op.to_string(), args));
        Ok(())
    }

    fn append_fun_v1(
        &mut self,
        lhs: &Slot,
        fun: &BuiltinSymbol,
        arg: &Slot,
        is_real: bool,
    ) -> Result<()> {
        let op = match fun.0 {
            2 => "symbolica_exp",
            3 => "symbolica_ln",
            4 => "symbolica_sin",
            5 => "symbolica_cos",
            6 => "symbolica_sqrt",
            7 => "symbolica_conjugate",
            8 => "symbolica_abs",
            _ => return Err(anyhow!("Builtin function {} is not defined.", fun.0)),
        };

        self.append_fun(lhs, op, &[*arg], is_real)
    }

    fn append_fun(&mut self, lhs: &Slot, fun: &str, args: &[Slot], is_real: bool) -> Result<()> {
        let args = self.consume_list(args)?;
        let lhs = self.produce(lhs)?;
        self.ssa
            .push(Instruction::Fun(lhs, fun.to_string(), args, is_real));
        Ok(())
    }

    fn append_join(
        &mut self,
        lhs: &Slot,
        cond: &Slot,
        true_val: &Slot,
        false_val: &Slot,
    ) -> Result<()> {
        let cond = self.consume(cond)?;
        let true_val = self.consume(true_val)?;
        let false_val = self.consume(false_val)?;
        let lhs = self.produce(lhs)?;
        self.ssa
            .push(Instruction::Join(lhs, cond, true_val, false_val));
        self.join_rhs.insert(true_val);
        self.join_rhs.insert(false_val);
        Ok(())
    }

    fn set_num_params(&mut self, num_params: usize) {
        self.num_params = num_params
    }

    fn compile(&mut self) -> Result<Application> {
        let (prog, reals) = self.translate()?;
        let mut app = Application::new(prog, reals)?;
        app.prepare_simd();
        Ok(app)
    }
}

impl IndirectTranslator {
    pub fn new(config: Config, salt: &str) -> IndirectTranslator {
        let slot_size = if config.is_complex() { 2 } else { 1 };
        let arena_mode = config.direct_arena();

        let mut builder = Builder::new(config.clone());
        builder.set_salt(salt.into());

        IndirectTranslator {
            config,
            ssa: Vec::new(),
            consts: Vec::new(),
            count_params: 0,
            count_outs: 0,
            count_statics: 0,
            temps: HashMap::new(),
            counts: HashMap::new(),
            // cache: HashMap::new(),
            cache: Vec::new(),
            outs: HashMap::new(),
            reals: HashSet::new(),
            join_rhs: HashSet::new(),
            num_params: 0,
            last_label: 0,
            builder,
            slot_size,
            arena_mode,
        }
    }

    fn create_static(&mut self) -> Result<Slot> {
        let s = Slot::Static(self.count_statics);
        self.counts.insert(self.count_statics, 0);
        self.count_statics += 1;
        Ok(s)
    }

    /// Produces a new Static variable if needed.
    /// slot should be an LHS.
    fn produce(&mut self, slot: &Slot) -> Result<Slot> {
        match slot {
            Slot::Temp(_) => {
                let s = self.create_static()?;
                self.temps.insert(*slot, s);
                Ok(s)
            }
            Slot::Out(idx) => {
                let s = self.create_static()?;
                self.temps.insert(*slot, s);

                self.count_outs = self.count_outs.max(*idx + 1);
                self.outs.insert(*idx, s);

                Ok(s)
            }
            _ => Err(anyhow!("unacceptable lhs.")),
        }
    }

    /// Consumes a slot.
    /// slot should be an RHS.
    fn consume(&mut self, slot: &Slot) -> Result<Slot> {
        match slot {
            Slot::Temp(_) | Slot::Out(_) => {
                if let Some(Slot::Static(s)) = self.temps.get(slot) {
                    *self.counts.get_mut(s).unwrap() += 1;
                    Ok(Slot::Static(*s))
                } else {
                    Err(anyhow!("Not a static reg."))
                }
            }
            Slot::Param(idx) => Ok(Slot::Param(*idx)),
            Slot::Const(idx) => Ok(Slot::Const(*idx)),
            Slot::Static(_) | Slot::Arg(_) => Err(anyhow!("Undefined Static/Arg.")),
        }
    }

    fn consume_list(&mut self, slots: &[Slot]) -> Result<Vec<Slot>> {
        slots.iter().map(|s| self.consume(s)).collect()
    }

    /* helper Node functions */

    fn unary_node(&mut self, op: &str, arg: Node) -> Result<Node> {
        self.builder.add_unary(Operation::new_checked(op), arg)
    }

    fn binary_node(&mut self, op: &str, left: Node, right: Node) -> Result<Node> {
        self.builder
            .add_binary(Operation::new_checked(op), left, right)
    }

    fn const_node(&mut self, val: f64) -> Node {
        self.builder.create_const(val).unwrap()
    }

    fn var_node(&mut self, name: &str) -> Node {
        self.builder.create_var(name).unwrap()
    }

    fn binop(&mut self, op: Operation, left: Node, right: Node) -> Result<Node> {
        assert!(op.is_plus() || op.is_times());
        self.builder.create_binary(op, left, right)
    }

    /// The second pass. It translates the SSA-form into a Symjit model.
    pub fn translate(&mut self) -> Result<(Program, HashSet<Loc>)> {
        let ssa = std::mem::take(&mut self.ssa);
        self.cache = vec![None; self.count_statics];

        for line in ssa.iter() {
            match line {
                Instruction::Add(lhs, args, n) => {
                    self.translate_nary(Operation::Plus, lhs, args, *n)?
                }
                Instruction::Mul(lhs, args, n) => {
                    self.translate_nary(Operation::Times, lhs, args, *n)?
                }
                Instruction::Pow(lhs, arg, p, is_real) => {
                    let p = self.const_node(*p as f64);
                    self.translate_pow(lhs, arg, p, *is_real)?
                }
                Instruction::Powf(lhs, arg, p, is_real) => {
                    let p = self.expr(p, false);
                    self.translate_pow(lhs, arg, p, *is_real)?
                }
                Instruction::Assign(lhs, rhs) => self.translate_assign(lhs, rhs)?,
                Instruction::Fun(lhs, fun, args, is_real) => {
                    self.translate_fun(lhs, fun, args, *is_real)?
                }
                Instruction::Join(lhs, cond, true_val, false_val) => {
                    self.translate_join(lhs, cond, true_val, false_val)?
                }
                Instruction::Label(id) => self.translate_label(*id)?,
                Instruction::IfElse(cond, id) => self.translate_ifelse(cond, *id)?,
                Instruction::Goto(id) => self.translate_goto(*id)?,
                Instruction::ExternalFun(lhs, op, args) => {
                    self.translate_external_fun(lhs, op, args, false)?
                }
            }
        }

        // Important! Outs are cached and should be written to final outputs.
        let base = if self.arena_mode {
            self.count_params
        } else {
            0
        };

        for k in 0..self.count_outs {
            let name = &format!("Mem{}", k);
            let loc = Loc::Mem((self.slot_size * (base + k)) as u32);
            self.builder.symbol_table().add_mem_loc(name, loc);
            let out = self.var_node(name);

            match self.outs.get(&k) {
                Some(Slot::Static(s)) => {
                    let eq = self.var_node(&format!("__Static{}", s));
                    self.builder.add_assign(out, eq).unwrap();
                }
                _ => {
                    return Err(anyhow!("output var {} not found.", k));
                }
            }
        }

        let np = self.slot_size * self.count_params.max(self.num_params);
        let count_states = if self.arena_mode { np } else { 0 };
        let count_params = if self.arena_mode { 0 } else { np };

        let prog = Program {
            builder: std::mem::take(&mut self.builder),
            count_states,
            count_params,
            count_obs: self.slot_size * self.count_outs,
            count_diffs: 0,
            count_loops: 0,
        };

        Ok((prog, self.reals.clone()))
    }

    // The counterpart of consume for the second-pass
    fn expr(&mut self, slot: &Slot, is_real: bool) -> Node {
        match slot {
            Slot::Param(idx) => {
                let name = &format!("Param{}", idx);
                let loc = if self.arena_mode {
                    let loc = Loc::Mem((self.slot_size * *idx) as u32);
                    self.builder.symbol_table().add_mem_loc(name, loc);
                    loc
                } else {
                    let loc = Loc::Param((self.slot_size * *idx) as u32);
                    self.builder.symbol_table().add_param_loc(name, loc);
                    loc
                };

                if is_real {
                    self.reals.insert(loc);
                }

                self.count_params = self.count_params.max(*idx + 1);
                self.var_node(name)
            }
            Slot::Out(idx) => {
                let name = &format!("Out{}", idx);
                self.builder.block().create_tmp_named(name)
            }
            Slot::Temp(idx) => {
                let name = &format!("__Temp{}", idx);
                self.builder.block().create_tmp_named(name)
            }
            Slot::Const(idx) => {
                let val = self.consts[*idx];
                if val.im != 0.0 {
                    let re = self.const_node(self.consts[*idx].re);
                    let im = self.const_node(self.consts[*idx].im);
                    self.binary_node("complex", re, im).unwrap()
                } else {
                    self.const_node(self.consts[*idx].re)
                }
            }
            Slot::Static(idx) => {
                let name = format!("__Static{}", idx);
                match &mut self.cache[*idx] {
                    Some(_) => std::mem::take(&mut self.cache[*idx]).unwrap(),
                    None => self.builder.block().create_tmp_named(&name),
                }
                /*
                self.cache
                    .remove(idx)
                    .unwrap_or(self.builder.block().create_tmp_named(&name))
                */
            }
            Slot::Arg(idx) => {
                let name = &format!("__Arg{}", idx);
                self.var_node(name)
            }
        }
    }

    // The counterpart of produce for the second-pass
    fn assign(&mut self, lhs: &Slot, rhs: Node) -> Result<()> {
        if let Slot::Static(idx) = lhs {
            // Important! If a static variable is used only once, it
            // is pushed into the cache to be incorporated into the
            // destination expression tree, unless it is on the right
            // hand side of a Join operation, which is a Φ-function.
            if self.counts.get(idx).is_some_and(|c| *c == 1) && !self.join_rhs.contains(lhs) {
                //self.cache.insert(*idx, rhs);
                self.cache[*idx] = Some(rhs);
                return Ok(());
            }
        }

        let lhs = self.expr(lhs, false);
        self.builder.add_assign(lhs, rhs).map(|_| ())
    }

    fn binary_tree(&mut self, op: Operation, args: &[Node]) -> Node {
        if args.len() == 1 {
            args[0].clone()
        } else {
            let k = args.len() / 2;
            let left = self.binary_tree(op.clone(), &args[..k]);
            let right = self.binary_tree(op.clone(), &args[k..]);
            self.binop(op, left, right).unwrap()
        }
    }

    fn translate_nary(&mut self, op: Operation, lhs: &Slot, args: &[Slot], n: usize) -> Result<()> {
        if args.len() == 2 && op.is_times() && args[0] == args[1] {
            let c = self.const_node(2.0);
            return self.translate_pow(lhs, &args[0], c, n != 0);
        } else if args.len() == 3 && op.is_times() && args[0] == args[1] && args[1] == args[2] {
            let c = self.const_node(3.0);
            return self.translate_pow(lhs, &args[0], c, n != 0);
        }

        let args: Vec<Node> = args
            .iter()
            .enumerate()
            .map(|(i, x)| self.expr(x, i < n))
            .collect();

        if n == 0 || n >= args.len() {
            let rhs = self.binary_tree(op, &args);
            self.assign(lhs, rhs)
        } else {
            let l = self.binary_tree(op.clone(), &args[..n]);
            let r = self.binary_tree(op.clone(), &args[n..]);
            let rhs = self.binop(op, l, r)?;
            self.assign(lhs, rhs)
        }
    }

    fn translate_pow(&mut self, lhs: &Slot, arg: &Slot, power: Node, is_real: bool) -> Result<()> {
        let arg = self.expr(arg, is_real);
        let rhs = self.binary_node("power", arg, power)?;
        self.assign(lhs, rhs)
    }

    fn translate_assign(&mut self, lhs: &Slot, rhs: &Slot) -> Result<()> {
        let rhs = self.expr(rhs, false);
        self.assign(lhs, rhs)
    }

    fn translate_fun(&mut self, lhs: &Slot, fun: &str, args: &[Slot], is_real: bool) -> Result<()> {
        self.translate_external_fun(lhs, &self.config.symbolica_fun(fun, is_real), args, is_real)
    }

    fn translate_external_fun(
        &mut self,
        lhs: &Slot,
        op: &str,
        args: &[Slot],
        is_real: bool,
    ) -> Result<()> {
        let n = args.len();
        assert!(n <= SLICE_CAP);

        if let Slot::Param(idx) = lhs {
            if is_real {
                self.reals.insert(Loc::Param(*idx as u32));
            }
        }

        let mut v: Vec<Node> = Vec::new();
        for a in args.iter() {
            let p = self.expr(a, is_real);
            if is_real {
                let arg = self.unary_node("real", p)?;
                v.push(arg);
            } else {
                v.push(p);
            }
        }
        let mut args = v;

        if VirtualTable::from_str(op).is_ok() || op.starts_with("composer_") {
            if n == 1 {
                let rhs = self.unary_node(op, args.remove(0))?;
                self.assign(lhs, rhs)?;
            } else if n == 2 {
                let rhs = self.binary_node(op, args.remove(0), args.remove(0))?;
                self.assign(lhs, rhs)?;
            } else {
                return Err(anyhow!("wrong number of arguments to {:?}", op));
            }
        } else if self.config.is_intrinsic_unary(&Operation::new(op)) && n == 1 {
            let rhs = self.unary_node(op, args.remove(0))?;
            self.assign(lhs, rhs)?;
        } else if self.config.is_intrinsic_binary(&Operation::new(op)) && n == 2 {
            let rhs = self.binary_node(op, args.remove(0), args.remove(0))?;
            self.assign(lhs, rhs)?;
        } else {
            let temps: Vec<Slot> = (0..n).map(|_| self.create_static().unwrap()).collect();
            let slice: Vec<Slot> = (0..n).map(Slot::Arg).collect();

            for i in 0..n {
                self.assign(&temps[i], args[i].clone())?;
            }

            for i in 0..n {
                if let Slot::Static(idx) = temps[i] {
                    let n = self
                        .builder
                        .create_var(&format!("__Static{}", idx))
                        .unwrap();
                    self.assign(&slice[i], n)?;
                }
            }

            let op = format!("${}", op);
            let l = self.const_node(0.0);
            let r = self.const_node(n as f64);
            let n = self.binary_node(op.as_str(), l, r)?;
            self.assign(lhs, n)?;
        }

        Ok(())
    }

    fn translate_label(&mut self, id: usize) -> Result<()> {
        let label = format!("L.{}", id);
        self.builder.block().add_label(&label);
        Ok(())
    }

    fn translate_ifelse(&mut self, cond: &Slot, id: usize) -> Result<()> {
        let label = format!("L.{}", id);
        let cond = self.expr(cond, false);
        let if_clause = self.unary_node("iszero", cond)?;
        self.builder.block().add_branch_if(if_clause, &label, true);
        Ok(())
    }

    fn translate_goto(&mut self, id: usize) -> Result<()> {
        if !self.config.simd_branch() || !self.config.symbolica() {
            let label = format!("L.{}", id);
            self.builder.block().add_branch(&label);
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
        // Join is essentially a Φ-function.
        let t = self.expr(true_val, false);
        let f = self.expr(false_val, false);
        let cond = self.expr(cond, false);
        let if_clause = self.unary_node("iszero", cond)?;
        let rhs = self.builder.create_ifelse(if_clause, f, t).unwrap();
        self.assign(lhs, rhs)?;
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

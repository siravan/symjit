#![allow(uncommon_codepoints)]

//! Symjit (<https://github.com/siravan/symjit>) is a lightweight just-in-time (JIT)
//! optimizer compiler for mathematical expressions written in Rust. It was originally
//! designed to compile SymPy (Python’s symbolic algebra package) expressions
//! into machine code and to serve as a bridge between SymPy and numerical routines
//! provided by NumPy and SciPy libraries.
//!
//! Symjit crate is the core compiler coupled to a Rust interface to expose the
//! JIT functionality to the Rust ecosystem and allow Rust applications to
//! generate code dynamically. Considering its origin, symjit is geared toward
//! compiling mathematical expressions instead of being a general-purpose JIT
//! compiler. Therefore, the only supported types for variables are `f64`,
//! `__m256d` (packed f64x4), and implicitly, `bool` and `i32`.
//!
//! Symjit emits AMD64 (x86-64), ARM64 (aarch64), and 64-bit RISC-V (riscv64) machine
//! codes on Linux, Windows, and macOS platforms. SIMD is supported on x86-64
//! CPUs with AVX instruction sets.
//!
//! # Workflow
//!
//! 1. Create terminals (variables and constants) and compose expressions using `Expr` methods:
//!    * Constructors: `var`, `from`, `unary`, `binary`, ...
//!    * Standard algebraic operations: `add`, `mul`, ...
//!    * Standard operators `+`, `-`, `*`, `/`, `%`, `&`, `|`, `^`, `!`.
//!    * Unary functions such as `sin`, `exp`, and other standard mathematical functions.
//!    * Binary functions such as `pow`, `min`, ...
//!    * IfElse operation `ifelse(cond, true_val, false_val)`.
//!    * Heaviside function: `heaviside(x)`, which returns 1 if `x >= 0`; otherwise 0.
//!    * Comparison methods `eq`, `ne`, `lt`, `le`, `gt`, and `ge`.
//!    * Looping constructs `sum` and `prod`.
//! 2. Create a new `Compiler` object (say, `comp`) using one of its constructors: `new()`
//!    or `with_compile_type(ty: CompilerType)`.
//! 3. Fine-tune the optimization passes using `opt_level`, `simd`, `fastmath`,
//!    and `cse` methods (optional).
//! 4. Define user-defined functions by calling `comp.def_unary` and `comp.def_binary`
//!    (optional).
//! 5. Compile by calling `comp.compile` or `comp.compile_params`. The result is of
//!    type `Application` (say, `app`).
//! 6. Execute the compiled code using one of the `app`'s `call` functions:
//!    * `call(&[f64])`: scalar call.
//!    * `call_params(&[f64], &[f64])`: scalar call with parameters.
//!    * `call_simd(&[__m256d])`: simd call.
//!    * `call_simd_params(&[__m256d], &[f64])`: simd call with parameters.
//! 7. Optionally, generate a standalone fast function to execute.
//!
//! Note that you can use the helper functions `var(&str) -> Expr`, `int(i32) -> Expr`,
//! `double(f64) -> Expr`, and `boolean(bool) -> f64` to reduce clutter.
//!
//! # Examples
//!
//! ```rust
//! use anyhow::Result;
//! use symjit::{Compiler, Expr};
//!
//! pub fn test_scalar() -> Result<()> {
//!     let x = Expr::var("x");
//!     let y = Expr::var("y");
//!     let u = &x + &y;
//!     let v = &x * &y;
//!
//!     let mut comp = Compiler::new();
//!     let mut app = comp.compile(&[x, y], &[u, v])?;
//!     let res = app.call(&[3.0, 5.0]);
//!     println!("{:?}", &res);   // prints [8.0, 15.0]
//!
//!     Ok(())
//! }
//! ```
//!
//! `test_scalar` is similar to the following basic example in Python/SymPy:
//!
//! ```python
//! from symjit import compile_func
//! from sympy import symbols
//!
//! x, y = symbols('x y')
//! f = compile_func([x, y], [x+y, x*y])
//! print(f(3.0, 5.0))  # prints [8.0, 15.0]
//! ```
//!
//! A more elaborate example, showcasing having a parameter, changing the
//! optimization level, and using SIMD:
//!
//! ```rust
//! use anyhow::Result;
//! use symjit::{var, Compiler, Expr};
//!
//! pub fn test_simd() -> Result<()> {
//!     use std::arch::x86_64::_mm256_loadu_pd;
//!
//!     let x = var("x");   // note var instead of Expr::var
//!     let p = var("p");   // the parameter
//!
//!     let u = &x.square() * &p;    // x^2 * p
//!     let mut comp = Compiler::new();
//!     comp.opt_level(2);  // optional (opt_level 0 to 2; default 1)
//!     let mut app = comp.compile_params(&[x], &[u], &[p])?;
//!
//!     let a = &[1.0, 2.0, 3.0, 4.0];
//!     let a = unsafe { _mm256_loadu_pd(a.as_ptr()) };
//!     let res = app.call_simd_params(&[a], &[5.0])?;
//!     println!("{:?}", &res);   // prints [__m256d(5.0, 20.0, 45.0, 80.0)]
//!     Ok(())
//! }
//! ```
//!
//! # Conditional Expression and Loops
//!
//! Many mathematical formulas need conditional expressions (`ifelse`) and loops.
//! Following SymPy, Symjit uses reduction loops such as `sum` and `prod`. The following
//! example returns the exponential functions:
//!
//! ```rust
//! use symjit::{int, var, Compiler};
//!
//! fn test_exp() -> Result<()> {
//!     let x = var("x");
//!     let i = var("i");   // loop variable
//!     let j = var("j");   // loop variable
//!
//!     // u = x^j / factorial(j) for j in j in 0..=50
//!     let u = x
//!         .pow(&j)
//!         .div(&i.prod(&i, &int(1), &j))
//!         .sum(&j, &int(0), &int(50));
//!
//!     let mut app = Compiler::new().compile(&[x], &[u])?;
//!     println!("{:?}", app(&[2.0])[0]); // returns exp(2.0) = 7.38905...
//!     Ok(())
//! }
//! ```
//!
//! An example showing how to calculate pi using the Leibniz formula:
//!
//! ```rust
//! use symjit::{int, var, Compiler};
//!
//! fn test_pi() -> Result<()> {
//!     let n = var("n");
//!     let i = var("i");   // loop variable
//!     let j = var("j");   // loop variable
//!
//!     // numer = if j % 2 == 0 { 4 } else { -4 }
//!     let numer = j.rem(&int(2)).eq(&int(0)).ifelse(&int(4), &int(-4));
//!     // denom = j * 2 + 1
//!     let denom = j.mul(&int(2)).add(&int(1));
//!     // v = numer / denom for j in 0..=n
//!     let v = (&numer / &denom).sum(&j, &int(0), &int(&n));
//!
//!     let mut app = Compiler::new().compile(&[x], &[v])?;
//!     println!("{:?}", app(&[100000000])[0]); // returns pi
//!     Ok(())
//! }
//! ```
//!
//! Note that here we are using explicit functions (`add`, `mul`, ...) instead of
//! the overloaded operators for clarity.
//!
//! # Fast Functions
//!
//! `Application`'s call functions need to copy the input slice into the function
//! memory area and then copy the output to a `Vec`. This process is acceptable
//! for large and complex functions but incurs a penalty for small ones.
//! Therefore, for a certain subset of applications, Symjit can compile to a
//! *fast function* and return a function pointer. Examples:
//!
//! ```rust
//! use anyhow::Result;
//! use symjit::{int, var, Compiler, FastFunc};
//!
//! fn test_fast() -> Result<()> {
//!     let x = var("x");
//!     let y = var("y");
//!     let z = var("z");
//!     let u = &x * &(&y - &z).pow(&int(2));    // x * (y - z)^2
//!
//!     let mut comp = Compiler::new();
//!     let mut app = comp.compile(&[x, y, z], &[u])?;
//!     let f = app.fast_func()?;
//!
//!     if let FastFunc::F3(f, _) = f {
//!         // f is of type extern "C" fn(f64, f64, f64) -> f64
//!         let res = f(3.0, 5.0, 9.0);
//!         println!("fast\t{:?}", &res);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! The conditions for a fast function are:
//!
//! * A fast function can have 1 to 8 arguments.
//! * No SIMD and no parameters.
//! * It returns only a single value.
//!
//! If these conditions are met, you can generate a fast function by calling
//! `app.fast_func()`, which returns a `Result<FastFunc>`. `FastFunc` is an
//! enum with eight variants `F1`, `F2`, ..., `F8`, corresponding to functions
//! with 1 to 8 arguments.
//!
//! # User-Defined Functions
//!
//! Symjit functions can call into user-defined Rust functions. Currently,
//! only the following function signatures are accepted:
//!
//! ```rust
//! pub type UnaryFunc = extern "C" fn(f64) -> f64;
//! pub type BinaryFunc = extern "C" fn(f64, f64) -> f64;
//! ```
//!
//! For example:
//!
//! ```rust
//! extern "C" fn f(x: f64) -> f64 {
//!     x.exp()
//! }
//!
//! extern "C" fn g(x: f64, y: f64) -> f64 {
//!     x.ln() * y
//! }
//!
//! fn test_external() -> Result<()> {
//!     let x = Expr::var("x");
//!     let u = Expr::unary("f_", &x);
//!     let v = &x * &Expr::binary("g_", &u, &x);
//!
//!     // v(x) = x * (ln(exp(x)) * x) = x ^ 3
//!
//!     let mut comp = Compiler::new();
//!     comp.def_unary("f_", f);
//!     comp.def_binary("g_", g);
//!     let mut app = comp.compile(&[x], &[v])?;
//!     println!("{:?}", app.call(&[5.0])[0]);
//!
//!     Ok(())
//! }
//! ```
//!
//! # Dynamic Expressions
//!
//! All the examples up to this point use static expressions. Of course, it
//! would have been easier just to use Rust expressions for these examples!
//! The main utility of Symjit for Rust is for dynamic code generation. Here,
//! we provide a simple example to calculate pi using Viete's formula
//! (<https://en.wikipedia.org/wiki/Vi%C3%A8te%27s_formula>):
//!
//! ```rust
//! fn test_pi_viete(silent: bool) -> Result<()> {
//!     let x = var("x");
//!     let mut u = int(1);
//!
//!     for i in 0..50 {
//!         let mut t = x.clone();
//!
//!         for _ in 0..i {
//!             t = &x + &(&x * &t.sqrt());
//!         }
//!
//!         u = &u * &t.sqrt();
//!     }
//!
//!     // u has 1275 = 50 * 51 / 2 sqrt operations
//!     let mut app = Compiler::new().compile(&[x], &[&int(2) / &u])?;
//!     println!("pi = \t{:?}", app.call(&[0.5])[0]);
//!     Ok(())
//! }
//! ```
//!
//! # C-Interface
//!
//! In addition to `Compiler`, this crate provides a C-style interface
//! used by the Python (<https://github.com/siravan/symjit>) and Julia
//! (<https://github.com/siravan/Symjit.jl>) packages. This interface
//! is composed of crate functions like `compile`, `execute`, and
//! `ptr_states`,..., and is not needed by the Rust interface but can be
//! used to link symjit to other programming languages.
//!

use anyhow::anyhow;
use std::ffi::{c_char, CStr, CString};

mod block;
mod code;
pub mod compiler;
mod defuns;
pub mod expr;
mod machine;
mod matrix;
mod memory;
mod model;
mod runnable;
mod utils;

mod allocator;
mod assembler;
mod builder;
mod generator;
mod mir;
mod node;
mod statement;
mod symbol;

mod amd;
mod arm;

#[allow(non_upper_case_globals)]
mod riscv64;

use defuns::Defuns;
use matrix::Matrix;
use model::{CellModel, Program};

pub use compiler::{Compiler, FastFunc};
pub use expr::{double, int, var, Expr};
pub use runnable::{Application, CompilerType};

pub const COUNT_SCRATCH: u8 = 14;

pub const USE_SIMD: u32 = 0x01;
pub const USE_THREADS: u32 = 0x02;
pub const CSE: u32 = 0x04;
pub const FASTMATH: u32 = 0x08;
pub const SANITIZE: u32 = 0x10;
pub const OPT_LEVEL_MASK: u32 = 0x0f00;
pub const OPT_LEVEL_SHIFT: usize = 8;

#[derive(Debug, Clone, Copy)]
pub enum CompilerStatus {
    Ok,
    Incomplete,
    InvalidUtf8,
    ParseError,
    InvalidCompiler,
    CompilationError,
}

pub struct CompilerResult {
    app: Option<Application>,
    status: CompilerStatus,
}

/// Compiles a model.
///
/// * `model` is a json string encoding the model.
/// * `ty` is the requested arch (amd, arm, native, or bytecode).
/// * `opt`: compilation options.
/// * `df`: user-defined functions.
///
/// # Safety
///     * both model and ty are pointers to null-terminated strings.
///     * The output is a raw pointer to a CompilerResults.
///
#[no_mangle]
pub unsafe extern "C" fn compile(
    model: *const c_char,
    ty: *const c_char,
    opt: u32,
    df: *const Defuns,
) -> *const CompilerResult {
    let mut res = CompilerResult {
        app: None,
        status: CompilerStatus::Incomplete,
    };

    let model = unsafe {
        match CStr::from_ptr(model).to_str() {
            Ok(model) => model,
            Err(msg) => {
                println!("{}", msg);
                res.status = CompilerStatus::InvalidUtf8;
                return Box::into_raw(Box::new(res)) as *const _;
            }
        }
    };

    let ty = unsafe {
        match CStr::from_ptr(ty).to_str() {
            Ok(ty) => ty,
            Err(msg) => {
                println!("{}", msg);
                res.status = CompilerStatus::InvalidUtf8;
                return Box::into_raw(Box::new(res)) as *const _;
            }
        }
    };

    let ml = match CellModel::load(model) {
        Ok(ml) => ml,
        Err(msg) => {
            println!("{}", msg);
            res.status = CompilerStatus::ParseError;
            return Box::into_raw(Box::new(res)) as *const _;
        }
    };

    let prog = match Program::new(&ml, opt & CSE != 0) {
        Ok(prog) => prog,
        Err(msg) => {
            println!("{}", msg);
            res.status = CompilerStatus::CompilationError;
            return Box::into_raw(Box::new(res)) as *const _;
        }
    };

    let df: &Defuns = unsafe { &*df };

    let app = match ty {
        "bytecode" => Application::new(prog, CompilerType::ByteCode, opt, df),
        "arm" => Application::new(prog, CompilerType::Arm, opt, df),
        "riscv" => Application::new(prog, CompilerType::RiscV, opt, df),
        "amd" => Application::new(prog, CompilerType::Amd, opt, df),
        "amd-avx" => Application::new(prog, CompilerType::AmdAVX, opt, df),
        "amd-sse" => Application::new(prog, CompilerType::AmdSSE, opt, df),
        "native" => Application::new(prog, CompilerType::Native, opt, df),
        "debug" => Application::new(prog, CompilerType::Debug, opt, df),
        _ => Err(anyhow!("invalid ty")),
    };

    match app {
        Ok(app) => {
            res.app = Some(app);
            res.status = CompilerStatus::Ok;
        }
        Err(msg) => {
            println!("{}", msg);
            res.status = CompilerStatus::InvalidCompiler;
        }
    }

    Box::into_raw(Box::new(res)) as *const _
}

/// Checks the status of a `CompilerResult`.
///
/// Returns a null-terminated string representing the status message.
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult.
///
#[no_mangle]
pub unsafe extern "C" fn check_status(q: *const CompilerResult) -> *const c_char {
    let q: &CompilerResult = unsafe { &*q };
    let msg = match q.status {
        CompilerStatus::Ok => c"Success",
        CompilerStatus::CompilationError => c"Compilation error",
        CompilerStatus::Incomplete => c"Incomplete (internal error)",
        CompilerStatus::InvalidUtf8 => c"The input string is not valid UTF8",
        CompilerStatus::ParseError => c"Parse error",
        CompilerStatus::InvalidCompiler => c"Compiler type not found",
    };
    msg.as_ptr() as *const _
}

/// Returns the number of states (dependent variables).
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult.
///
#[no_mangle]
pub unsafe extern "C" fn count_states(q: *const CompilerResult) -> usize {
    let q: &CompilerResult = unsafe { &*q };
    if let Some(app) = &q.app {
        app.count_states
    } else {
        0
    }
}

/// Returns the number of parameters.
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult.
///
#[no_mangle]
pub unsafe extern "C" fn count_params(q: *const CompilerResult) -> usize {
    let q: &CompilerResult = unsafe { &*q };
    if let Some(app) = &q.app {
        app.count_params
    } else {
        0
    }
}

/// Returns the number of observables (output).
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult.
///
#[no_mangle]
pub unsafe extern "C" fn count_obs(q: *const CompilerResult) -> usize {
    let q: &CompilerResult = unsafe { &*q };
    if let Some(app) = &q.app {
        app.count_obs
    } else {
        0
    }
}

/// Returns the number of differential equations.
///
/// Generally, it should be the same as the number of states.
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult.
///
#[no_mangle]
pub unsafe extern "C" fn count_diffs(q: *const CompilerResult) -> usize {
    let q: &CompilerResult = unsafe { &*q };
    if let Some(app) = &q.app {
        app.count_diffs
    } else {
        0
    }
}

/// Deprecated. Previously used for interfacing to DifferentialEquation.jl. It is
/// replaced with <https://github.com/siravan/SymJit.jl>.
///
/// # Safety
///
/// Deprecated. No effects.
#[no_mangle]
pub unsafe extern "C" fn run(
    _q: *mut CompilerResult,
    _du: *mut f64,
    _u: *const f64,
    _ns: usize,
    _p: *const f64,
    _np: usize,
    _t: f64,
) -> bool {
    // let q: &mut CompilerResult = unsafe { &mut *q };

    // if let Some(app) = &mut q.app {
    //     if app.count_states != ns || app.count_params != np {
    //         return false;
    //     }

    //     let du: &mut [f64] = unsafe { std::slice::from_raw_parts_mut(du, ns) };
    //     let u: &[f64] = unsafe { std::slice::from_raw_parts(u, ns) };
    //     let p: &[f64] = unsafe { std::slice::from_raw_parts(p, np) };
    //     app.call(du, u, p, t);
    //     true
    // } else {
    //     false
    // }
    false
}

/// Executes the compiled function.
///
/// The calling routine should fill the states and parameters before
/// calling `execute`. The result populates obs or diffs (as defined in
/// model passed to `compile`).
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult.
///
#[no_mangle]
pub unsafe extern "C" fn execute(q: *mut CompilerResult, t: f64) -> bool {
    let q: &mut CompilerResult = unsafe { &mut *q };

    if let Some(app) = &mut q.app {
        app.exec(t);
        true
    } else {
        false
    }
}

/// Executes the compiled function `n` times (vectorized).
///
/// The calling function provides `buf`, which is a k x n matrix of doubles,
/// where k is equal to the `maximum(count_states, count_obs)`. The calling
/// funciton fills the first `count_states` rows of buf. The result is returned
/// in the first count_obs rows of buf.
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult.
///
///     In addition, buf should points to a valid matrix of correct size.
///
#[no_mangle]
pub unsafe extern "C" fn execute_vectorized(
    q: *mut CompilerResult,
    buf: *mut f64,
    n: usize,
) -> bool {
    let q: &mut CompilerResult = unsafe { &mut *q };

    if let Some(app) = &mut q.app {
        let h = usize::max(app.count_states, app.count_obs);
        let buf: &mut [f64] = unsafe { std::slice::from_raw_parts_mut(buf, h * n) };
        let states = Matrix::from_buf(buf, h, n);
        let mut obs = Matrix::from_buf(buf, h, n);
        app.exec_vectorized(&states, &mut obs);
        true
    } else {
        false
    }
}

/// Returns a pointer to the state variables (`count_states` doubles).
///
/// The function calling `execute` should write the state variables in this area.
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult.
///
#[no_mangle]
pub unsafe extern "C" fn ptr_states(q: *mut CompilerResult) -> *mut f64 {
    let q: &mut CompilerResult = unsafe { &mut *q };
    if let Some(app) = &mut q.app {
        &mut app.compiled.mem_mut()[app.first_state] as *mut f64
    } else {
        std::ptr::null_mut()
    }
}

/// Returns a pointer to the parameters (`count_params` doubles).
///
/// The function calling `execute` should write the parameters in this area.
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult.
///
#[no_mangle]
pub unsafe extern "C" fn ptr_params(q: *mut CompilerResult) -> *mut f64 {
    let q: &mut CompilerResult = unsafe { &mut *q };
    if let Some(app) = &mut q.app {
        //&mut app.compiled.mem_mut()[app.first_param] as *mut f64
        &mut app.params[app.first_param] as *mut f64
    } else {
        std::ptr::null_mut()
    }
}

/// Returns a pointer to the observables (`count_obs` doubles).
///
/// The function calling `execute` reads the observables from this area.
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult.
///
#[no_mangle]
pub unsafe extern "C" fn ptr_obs(q: *mut CompilerResult) -> *const f64 {
    let q: &CompilerResult = unsafe { &*q };
    if let Some(app) = &q.app {
        &app.compiled.mem()[app.first_obs] as *const f64
    } else {
        std::ptr::null()
    }
}

/// Returns a pointer to the differentials (`count_diffs` doubles).
///
/// The function calling `execute` reads the differentials from this area.
///
/// Note: whether the output is returned as observables or differentials is
/// defined in the model.
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult.
///
#[no_mangle]
pub unsafe extern "C" fn ptr_diffs(q: *mut CompilerResult) -> *const f64 {
    let q: &CompilerResult = unsafe { &*q };
    if let Some(app) = &q.app {
        &app.compiled.mem()[app.first_diff] as *const f64
    } else {
        std::ptr::null()
    }
}

/// Dumps the compiled binary code to a file (`name`).
///
/// This function is useful for debugging but is not necessary for
/// normal operations.
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult.
///
#[no_mangle]
pub unsafe extern "C" fn dump(
    q: *mut CompilerResult,
    name: *const c_char,
    what: *const c_char,
) -> bool {
    let q: &mut CompilerResult = unsafe { &mut *q };
    if let Some(app) = &mut q.app {
        let name = unsafe { CStr::from_ptr(name).to_str().unwrap() };
        let what = unsafe { CStr::from_ptr(what).to_str().unwrap() };
        app.dump(name, what)
    } else {
        false
    }
}

/// Deallocates the CompilerResult pointed by `q`.
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult and that after
///     calling this function, q is invalid and should not
///     be used anymore.
///
#[no_mangle]
pub unsafe extern "C" fn finalize(q: *mut CompilerResult) {
    if !q.is_null() {
        let _ = unsafe { Box::from_raw(q) };
    }
}

/// Returns a null-terminated string representing the version.
///
/// Used for debugging.
///
/// # Safety
///     the return value is a null-terminated string that should not
///     be freed.
///
#[no_mangle]
pub unsafe extern "C" fn info() -> *const c_char {
    // let msg = c"symjit 1.3.3";
    let msg = CString::new(env!("CARGO_PKG_VERSION")).unwrap();
    msg.into_raw() as *const _
}

/// Returns a pointer to the fast function if one can be compiled.
///
/// # Safety
///     1. If the model cannot be compiled to a fast function, NULL is returned.
///     2. A fast function code memory is leaked and is not deallocated.
///
#[no_mangle]
pub unsafe extern "C" fn fast_func(q: *mut CompilerResult) -> *const usize {
    let q: &mut CompilerResult = unsafe { &mut *q };
    if let Some(app) = &mut q.app {
        match app.get_fast() {
            Some(f) => f as *const usize,
            None => std::ptr::null(),
        }
    } else {
        std::ptr::null()
    }
}

/// Interface for Sympy's LowLevelCallable.
///
/// # Safety
///     1. If the model cannot be compiled to a fast function, NULL is returned.
///     2. The resulting function lives as long as q does and should not be stored
///         separately.
///
#[no_mangle]
pub unsafe extern "C" fn callable_quad(n: usize, xx: *const f64, q: *mut CompilerResult) -> f64 {
    let q: &mut CompilerResult = unsafe { &mut *q };
    let xx: &[f64] = unsafe { std::slice::from_raw_parts(xx, n) };

    if let Some(app) = &mut q.app {
        app.exec_callable(xx)
    } else {
        f64::NAN
    }
}

/// Interface for Sympy's LowLevelCallable.
///
/// # Safety
///     1. If the model cannot be compiled to a fast function, NULL is returned.
///     2. The resulting function lives as long as q does and should not be stored
///         separately.
///
#[no_mangle]
pub unsafe extern "C" fn callable_quad_fast(n: usize, xx: *const f64, f: *const usize) -> f64 {
    let xx: &[f64] = unsafe { std::slice::from_raw_parts(xx, n) };

    match n {
        0 => {
            let f: fn() -> f64 = unsafe { std::mem::transmute(f) };
            f()
        }
        1 => {
            let f: fn(f64) -> f64 = unsafe { std::mem::transmute(f) };
            f(xx[0])
        }
        2 => {
            let f: fn(f64, f64) -> f64 = unsafe { std::mem::transmute(f) };
            f(xx[0], xx[1])
        }
        3 => {
            let f: fn(f64, f64, f64) -> f64 = unsafe { std::mem::transmute(f) };
            f(xx[0], xx[1], xx[2])
        }
        4 => {
            let f: fn(f64, f64, f64, f64) -> f64 = unsafe { std::mem::transmute(f) };
            f(xx[0], xx[1], xx[2], xx[3])
        }
        5 => {
            let f: fn(f64, f64, f64, f64, f64) -> f64 = unsafe { std::mem::transmute(f) };
            f(xx[0], xx[1], xx[2], xx[3], xx[4])
        }
        6 => {
            let f: fn(f64, f64, f64, f64, f64, f64) -> f64 = unsafe { std::mem::transmute(f) };
            f(xx[0], xx[1], xx[2], xx[3], xx[4], xx[5])
        }
        7 => {
            let f: fn(f64, f64, f64, f64, f64, f64, f64) -> f64 = unsafe { std::mem::transmute(f) };
            f(xx[0], xx[1], xx[2], xx[3], xx[4], xx[5], xx[6])
        }
        _ => {
            panic!("too many parameters for a fast func");
        }
    }
}

/// Interface for Sympy's LowLevelCallable (image filtering).
///
/// # Safety
///     1. If the model cannot be compiled to a fast function, NULL is returned.
///     2. The resulting function lives as long as q does and should not be stored
///         separately.
///
#[no_mangle]
pub unsafe extern "C" fn callable_filter(
    buffer: *const f64,
    filter_size: usize,
    return_value: *mut f64,
    q: *mut CompilerResult,
) -> i64 {
    let q: &mut CompilerResult = unsafe { &mut *q };
    let xx: &[f64] = unsafe { std::slice::from_raw_parts(buffer, filter_size) };

    if let Some(app) = &mut q.app {
        let p: &mut f64 = unsafe { &mut *return_value };
        *p = app.exec_callable(xx);
        1
    } else {
        0
    }
}

/************************************************/

/// Creates an empty Matrix (a 2d array).
///
/// # Safety
///     It returns a pointer to the allocated Matrix, which needs to be
///     deallocated eventually.
///
#[no_mangle]
pub unsafe extern "C" fn create_matrix() -> *const Matrix {
    let mat = Matrix::new();
    Box::into_raw(Box::new(mat)) as *const Matrix
}

/// Finalizes (deallocates) the Matrix.
///
/// # Safety
///     1, mat should point to a valid Matrix object created by create_matrix.
///     2. After finalize_matrix is called, mat is invalid.
///
#[no_mangle]
pub unsafe extern "C" fn finalize_matrix(mat: *mut Matrix) {
    if !mat.is_null() {
        let _ = unsafe { Box::from_raw(mat) };
    }
}

/// Adds a row to the Matrix.
///
/// # Safety
///     1, mat should point to a valid Matrix object created by create_matrix.
///     2. v should point to a valid array of doubles of length at least n.
///     3. v should remains valid for the lifespan of mat.
///
#[no_mangle]
pub unsafe extern "C" fn add_row(mat: *mut Matrix, v: *mut f64, n: usize) {
    let mat: &mut Matrix = unsafe { &mut *mat };
    mat.add_row(v, n);
}

/// Executes (runs) the matrix model encoded by `q`.
///
/// # Safety
///     1, q should point to a valid CompilerResult object.
///     2. states should point to a valid Matrix of at least count_states rows.
///     3. obs should point to a valid Matrix of at least count_obs rows.
///
#[no_mangle]
pub unsafe extern "C" fn execute_matrix(
    q: *mut CompilerResult,
    states: *const Matrix,
    obs: *mut Matrix,
) -> bool {
    let q: &mut CompilerResult = unsafe { &mut *q };
    let states: &Matrix = unsafe { &*states };
    let obs: &mut Matrix = unsafe { &mut *obs };

    if let Some(app) = &mut q.app {
        app.exec_vectorized(states, obs);
        true
    } else {
        false
    }
}

/************************************************/

/// Creates an empty `Defun` (a list of user-defined functions).
///
/// `Defuns` are used to pass user-defined functions (either Python
/// functions or symjit-compiled functions).
///
/// # Safety
///     It returns a pointer to the allocated Defun, which needs to be
///     deallocated eventually.
///
#[no_mangle]
pub unsafe extern "C" fn create_defuns() -> *const Defuns {
    let df = Defuns::new();
    Box::into_raw(Box::new(df)) as *const Defuns
}

/// Finalizes (deallocates) a `Defun`.
///
/// # Safety
///     1, df should point to a valid Defun object created by create_defuns.
///     2. After finalize_defun is called, df is invalid.
///
#[no_mangle]
pub unsafe extern "C" fn finalize_defuns(df: *mut Defuns) {
    if !df.is_null() {
        let _ = unsafe { Box::from_raw(df) };
    }
}

/// Adds a new function to a `Defun`.
///
/// # Safety
///     1, df should point to a valid Defun object created by create_defun.
///     2. name should be a valid utf8 string.
///     3. p should point to a valid C-styple function pointer that accepts
///         num_args double arguments.
///
#[no_mangle]
pub unsafe extern "C" fn add_func(
    df: *mut Defuns,
    name: *const c_char,
    p: *const usize,
    num_args: usize,
) {
    let df: &mut Defuns = unsafe { &mut *df };
    let name = unsafe { CStr::from_ptr(name).to_str().unwrap() };
    df.add_func(name, p, num_args);
}

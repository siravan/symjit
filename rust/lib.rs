#![allow(uncommon_codepoints)]

//! Symjit (<https://github.com/siravan/symjit>) is a lightweight just-in-time (JIT)
//! optimizer compiler for mathematical expressions. It was originally designed to
//! directly translate SymPy (Python’s symbolic algebra package) expressions into
//! machine code.
//!
//! Symjit crate is the core compiler coupled to Rust interface classes to expose
//! the JIT functionality to the Rust ecosystem and allows Rust applications to
//! generate code dynamically. Considering its origin, symjit is geared toward
//! compiling mathematical expressions instead of being a general-purpose JIT
//! compiler.
//!
//! Symjit emits AMD64 (x86-64), ARM64 (aarch64), and 64-bit RISC-V (riscv64) machine
//! codes on Linux, Windows, and Darwin (MacOS) platforms.
//!
//! # Workflow
//!
//! 1. Create variables and expressions using `Expr` methods.
//! 2. Create a new `Compiler` object (say, `comp`) using one of its constructors: `new`
//!     and `with_compile_type`, where `ty` is of type `CompilerType`.
//! 3. Fine-tune the optimization passes using methods `opt_level`, `simd`, `fastmath`,
//!     and `cse` (optional).
//! 4. Generate JIT code (say, `app`) by calling `comp.compile`. The result is of type `Application`.
//! 5. Change parameters by writing directly to `app.params` (optional).
//! 6. Call the compiled code using one of the `app`'s `call` functions:
//!     * `call(&[f64])`: scalar call.
//!     * `call_params(&[f64], &[f64])`: scalar call with parameters.
//!     * `call_simd(&[__m256d])`: simd call.
//!     * `call_simd_params(&[__m256d], &[f64])`: simd call with parameters.
//!
//! Currently, SIMD is only supported on x86-64 CPUs with AVX instruction sets.
//!
//! # Examples
//!
//! A basic Python/Sympy example:
//!
//! ```python//!
//! from symjit import compile_func
//! from sympy import symbols
//!
//! x, y = symbols('x y')
//! f = compile_func([x, y], [x+y, x*y])
//! print(f(3.0, 5.0))  # prints [8.0, 15.0]
//! ```
//!
//! Same code in Rust:
//!
//! ```rust
//! use anyhow::Result;
//! use symjit::{Compiler, Expr};
//!
//! pub fn test_scalar() -> Result<()> {
//!     let x = Expr::var("x");
//!     let y = Expr::var("y");
//!     let p = &x + &y;
//!     let q = &x * &y;
//!
//!     let mut comp = Compiler::new();
//!     comp.opt_level(2);  // optional (opt_level 0 to 2; default 1)
//!     let mut app = comp.compile(&[x, y], &[p, q])?;
//!     let v = app.call(&[3.0, 5.0]);
//!     println!("{:?}", &v);
//!
//!     Ok(())
//! }
//! ```
//!
//! A combined SIMD and parameter use example:
//!
//! pub fn test_simd() -> Result<()> {
//!     use std::arch::x86_64::_mm256_loadu_pd;
//!
//!     let x = Expr::var("x");
//!     let p = Expr::var("p"); // parameter
//!
//!     let expr = &x.square() * &p;    // x^2 * p
//!     let mut comp = Compiler::new();
//!     let mut app = comp.compile_params(&[x], &[expr], &[p])?;
//!
//!     let v = vec![1.0, 2.0, 3.0, 4.0];
//!     let p = unsafe { vec![_mm256_loadu_pd(v.as_ptr())] };
//!     let q = app.call_simd_params(&p, &[5.0])?;
//!     println!("{:?}", &q);   // prints [__m256d(5.0, 20.0, 45.0, 80.0)]
//!     Ok(())
//! }

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

pub use compiler::Compiler;
pub use expr::Expr;
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
    func: Option<Application>,
    status: CompilerStatus,
}

/// Compiles a model (a json string encoding the func model).
/// ty is the requested arch (amd, arm, native, or bytecode).
///
/// # Safety
///     both model and ty are pointers to null-terminated strings
///     the output is a raw pointer to a CompilerResults.
///
#[no_mangle]
pub unsafe extern "C" fn compile(
    model: *const c_char,
    ty: *const c_char,
    opt: u32,
    df: *const Defuns,
) -> *const CompilerResult {
    let mut res = CompilerResult {
        func: None,
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

    let func = match ty {
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

    match func {
        Ok(func) => {
            res.func = Some(func);
            res.status = CompilerStatus::Ok;
        }
        Err(msg) => {
            println!("{}", msg);
            res.status = CompilerStatus::InvalidCompiler;
        }
    }

    Box::into_raw(Box::new(res)) as *const _
}

/// Checks the status of a CompilerResult.
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
    if let Some(func) = &q.func {
        func.count_states
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
    if let Some(func) = &q.func {
        func.count_params
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
    if let Some(func) = &q.func {
        func.count_obs
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
    if let Some(func) = &q.func {
        func.count_diffs
    } else {
        0
    }
}

/// Fills du with the results of differentials after executing one step of the model
/// This function is mainly for DifferentialEquation.jl compatibility and not for python/sympy
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult
///
#[no_mangle]
pub unsafe extern "C" fn run(
    q: *mut CompilerResult,
    du: *mut f64,
    u: *const f64,
    ns: usize,
    p: *const f64,
    np: usize,
    t: f64,
) -> bool {
    // let q: &mut CompilerResult = unsafe { &mut *q };

    // if let Some(func) = &mut q.func {
    //     if func.count_states != ns || func.count_params != np {
    //         return false;
    //     }

    //     let du: &mut [f64] = unsafe { std::slice::from_raw_parts_mut(du, ns) };
    //     let u: &[f64] = unsafe { std::slice::from_raw_parts(u, ns) };
    //     let p: &[f64] = unsafe { std::slice::from_raw_parts(p, np) };
    //     func.call(du, u, p, t);
    //     true
    // } else {
    //     false
    // }
    false
}

/// Executes the compiled function.
///
/// The calling routine should fill the states and parameters before
/// calling execute. The result populates obs or diffs (as defined in
/// model passed to compile).
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult.
///
#[no_mangle]
pub unsafe extern "C" fn execute(q: *mut CompilerResult, t: f64) -> bool {
    let q: &mut CompilerResult = unsafe { &mut *q };

    if let Some(func) = &mut q.func {
        func.exec(t);
        true
    } else {
        false
    }
}

/// Executes the compiled function n times (vectorized).
///
/// The calling function provides buf, which is a k x n matrix of doubles
/// k is equal to maximum(count_states, count_obs). The calling funciton
/// fills the first count_states rows of buf. The result is returned in
/// the first count_obs rows of buf.
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

    if let Some(func) = &mut q.func {
        let h = usize::max(func.count_states, func.count_obs);
        let buf: &mut [f64] = unsafe { std::slice::from_raw_parts_mut(buf, h * n) };
        let states = Matrix::from_buf(buf, h, n);
        let mut obs = Matrix::from_buf(buf, h, n);
        func.exec_vectorized(&states, &mut obs);
        true
    } else {
        false
    }
}

/// Returns a pointer to the state variables (count_states doubles).
///
/// The function calling execute should write the state variables in this area.
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult.
///
#[no_mangle]
pub unsafe extern "C" fn ptr_states(q: *mut CompilerResult) -> *mut f64 {
    let q: &mut CompilerResult = unsafe { &mut *q };
    if let Some(func) = &mut q.func {
        &mut func.compiled.mem_mut()[func.first_state] as *mut f64
    } else {
        std::ptr::null_mut()
    }
}

/// Returns a pointer to the parameters (count_params doubles).
///
/// The function calling execute should write the parameters in this area.
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult.
///
#[no_mangle]
pub unsafe extern "C" fn ptr_params(q: *mut CompilerResult) -> *mut f64 {
    let q: &mut CompilerResult = unsafe { &mut *q };
    if let Some(func) = &mut q.func {
        //&mut func.compiled.mem_mut()[func.first_param] as *mut f64
        &mut func.params[func.first_param] as *mut f64
    } else {
        std::ptr::null_mut()
    }
}

/// Returns a pointer to the observables (count_obs doubles).
///
/// The function calling execute reads the observables from this area.
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult.
///
#[no_mangle]
pub unsafe extern "C" fn ptr_obs(q: *mut CompilerResult) -> *const f64 {
    let q: &CompilerResult = unsafe { &*q };
    if let Some(func) = &q.func {
        &func.compiled.mem()[func.first_obs] as *const f64
    } else {
        std::ptr::null()
    }
}

/// Returns a pointer to the differentials (count_diffs doubles).
///
/// The function calling execute reads the differentials from this area.
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
    if let Some(func) = &q.func {
        &func.compiled.mem()[func.first_diff] as *const f64
    } else {
        std::ptr::null()
    }
}

/// Dumps the compiled binary code to a file (name).
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
    if let Some(func) = &mut q.func {
        let name = unsafe { CStr::from_ptr(name).to_str().unwrap() };
        let what = unsafe { CStr::from_ptr(what).to_str().unwrap() };
        func.dump(name, what)
    } else {
        false
    }
}

/// Deallocates the CompilerResult pointed by q.
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
///     2. The resulting function lives as long as q does and should not be stored
///         separately.
///
#[no_mangle]
pub unsafe extern "C" fn fast_func(q: *mut CompilerResult) -> *const usize {
    let q: &mut CompilerResult = unsafe { &mut *q };
    if let Some(func) = &mut q.func {
        match func.get_fast() {
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

    if let Some(func) = &mut q.func {
        func.exec_callable(xx)
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

    if let Some(func) = &mut q.func {
        let p: &mut f64 = unsafe { &mut *return_value };
        *p = func.exec_callable(xx);
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

/// Finalized (deallocates) the Matrix.
///
/// # Safety
///     1, mat should point to a valid Matrix object created by create_matrix
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
///     1, mat should point to a valid Matrix object created by create_matrix
///     2. v should point to a valid array of doubles of length at least n
///     3. v should remains valid for the lifespan of mat
///
#[no_mangle]
pub unsafe extern "C" fn add_row(mat: *mut Matrix, v: *mut f64, n: usize) {
    let mat: &mut Matrix = unsafe { &mut *mat };
    mat.add_row(v, n);
}

/// Executes (runs) the model encoded by q.
///
/// # Safety
///     1, q should point to a valid CompilerResult object
///     2. states should point to a valid Matrix of at least count_states rows
///     3. obs should point to a valid Matrix of at least count_obs rows
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

    if let Some(func) = &mut q.func {
        func.exec_vectorized(states, obs);
        true
    } else {
        false
    }
}

/************************************************/

/// Creates an empty Defun (a list of Defined-Functions).
///
/// Defuns are used to pass functions (either Python functions or
/// other symjit-compiled functions).
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

/// Finalized (deallocates) a Defun.
///
/// # Safety
///     1, mat should point to a valid Defun object created by create_matrix
///     2. After finalize_defun is called, df is invalid.
///
#[no_mangle]
pub unsafe extern "C" fn finalize_defuns(df: *mut Defuns) {
    if !df.is_null() {
        let _ = unsafe { Box::from_raw(df) };
    }
}

/// Adds a new function to a Defun.
///
/// # Safety
///     1, df should point to a valid Defun object created by create_defun
///     2. name should be a valid utf8 string
///     3. p should point to a valid C-styple function pointer that accepts
///         num_args double arguments
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

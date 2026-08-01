#![allow(uncommon_codepoints)]

use std::collections::HashSet;
use std::ffi::{c_char, CStr, CString};
use std::fmt::Debug;
use std::str::FromStr;

mod allocator;
mod amd;
mod applet;
mod arm;
mod assembler;
mod block;
mod builder;
mod code;
mod compactor;
mod compiler;
mod complexify;
mod composer;
mod config;
mod defuns;
mod expr;
mod generator;
mod instruction;
mod machine;
mod matrix;
mod memory;
mod mir;
mod model;
mod node;
mod operation;
mod parser;
mod runnable;
mod serializer;
mod statement;
mod symbol;
mod types;
mod utils;

#[allow(non_upper_case_globals)]
mod riscv64;

pub use compiler::Compiler;
pub use config::Config;
pub use defuns::Defuns;
pub use matrix::Matrix;
pub use model::{CellModel, Program};
pub use runnable::{Application, CompilerType};
pub use utils::{Compiled, Storage};

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
    msg: CString,
}

fn error_message<E: Debug>(msg: &str, err: E) -> CString {
    let s = format!("{:?}: {:?}", msg, err);
    CString::from_str(&s).unwrap()
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
        msg: CString::from_str("Success").unwrap(),
    };

    let model = unsafe {
        match CStr::from_ptr(model).to_str() {
            Ok(model) => model,
            Err(msg) => {
                res.status = CompilerStatus::InvalidUtf8;
                res.msg = error_message("Invalid encoding", msg);
                return Box::into_raw(Box::new(res)) as *const _;
            }
        }
    };

    let ty = unsafe {
        match CStr::from_ptr(ty).to_str() {
            Ok(ty) => ty,
            Err(msg) => {
                res.status = CompilerStatus::InvalidUtf8;
                res.msg = error_message("Invalid compiler type", msg);
                return Box::into_raw(Box::new(res)) as *const _;
            }
        }
    };

    let ml = match CellModel::load(model) {
        Ok(ml) => ml,
        Err(msg) => {
            res.status = CompilerStatus::ParseError;
            res.msg = error_message("Cannot parse JSON", msg);
            return Box::into_raw(Box::new(res)) as *const _;
        }
    };

    if let Ok(mut config) = Config::from_name(ty, opt) {
        let df: Defuns = unsafe {
            if df.is_null() {
                Defuns::new()
            } else {
                (&*df).clone()
            }
        };

        config.set_defuns(df);

        let prog = match Program::new(&ml, config) {
            Ok(prog) => prog,
            Err(msg) => {
                res.status = CompilerStatus::CompilationError;
                res.msg = error_message("Compilation error (prog)", msg);
                return Box::into_raw(Box::new(res)) as *const _;
            }
        };

        let app = Application::new(prog, HashSet::new());

        match app {
            Ok(app) => {
                res.status = CompilerStatus::Ok;
                res.app = Some(app);
            }
            Err(msg) => {
                res.status = CompilerStatus::CompilationError;
                res.msg = error_message("Compilation error (app)", &msg);
            }
        }
    } else {
        res.status = CompilerStatus::InvalidCompiler;
        res.msg = error_message("Config error", opt);
    }

    Box::into_raw(Box::new(res)) as *const _
}

/// Translates a Symbolica model.
///
/// * `json` is a json string encoding the output of `export_instructions`.
/// * `ty` is the requested arch (amd, arm, native, or bytecode).
/// * `opt`: compilation options.
/// * `df`: user-defined functions (currently ignored).
///
/// # Safety
///     * both model and ty are pointers to null-terminated strings.
///     * The output is a raw pointer to a CompilerResults.
///
#[no_mangle]
pub unsafe extern "C" fn translate(
    json: *const c_char,
    ty: *const c_char,
    opt: u32,
    df: *mut Defuns,
    num_params: usize,
) -> *const CompilerResult {
    let mut res = CompilerResult {
        app: None,
        status: CompilerStatus::Incomplete,
        msg: CString::from_str("Success").unwrap(),
    };

    let json = unsafe {
        match CStr::from_ptr(json).to_str() {
            Ok(json) => json,
            Err(msg) => {
                res.status = CompilerStatus::InvalidUtf8;
                res.msg = error_message("Invalid encoding", msg);
                return Box::into_raw(Box::new(res)) as *const _;
            }
        }
    };

    let ty = unsafe {
        match CStr::from_ptr(ty).to_str() {
            Ok(ty) => ty,
            Err(msg) => {
                res.status = CompilerStatus::InvalidUtf8;
                res.msg = error_message("Invalid compiler type", msg);
                return Box::into_raw(Box::new(res)) as *const _;
            }
        }
    };

    if let Ok(mut config) = Config::from_name(ty, opt) {
        let df: Defuns = unsafe {
            if df.is_null() {
                Defuns::new()
            } else {
                (&*df).clone()
            }
        };

        config.set_defuns(df);
        let mut comp = Compiler::with_config(config);
        let app = comp.translate(json.to_string(), num_params);

        match app {
            Ok(app) => {
                res.app = Some(app);
                res.status = CompilerStatus::Ok;
            }
            Err(msg) => {
                res.status = CompilerStatus::InvalidCompiler;
                res.msg = error_message("Compilation error", msg);
            }
        }
    } else {
        res.status = CompilerStatus::InvalidCompiler;
        res.msg = error_message("Config error", opt);
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
    q.msg.as_ptr() as *const _
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
pub unsafe extern "C" fn save(q: *const CompilerResult, file: *const c_char) -> bool {
    let q: &CompilerResult = unsafe { &*q };
    let file = unsafe {
        match CStr::from_ptr(file).to_str() {
            Ok(file) => file,
            Err(_) => return false,
        }
    };

    if let Some(app) = &q.app {
        if let Ok(mut fs) = std::fs::File::create(file) {
            app.save(&mut fs).is_ok()
        } else {
            false
        }
    } else {
        false
    }
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
pub unsafe extern "C" fn load(file: *const c_char, df: *mut Defuns) -> *const CompilerResult {
    let mut res = CompilerResult {
        app: None,
        status: CompilerStatus::Incomplete,
        msg: CString::from_str("Success").unwrap(),
    };

    let df: Defuns = unsafe {
        if df.is_null() {
            Defuns::new()
        } else {
            (&*df).clone()
        }
    };

    let file = unsafe {
        match CStr::from_ptr(file).to_str() {
            Ok(file) => file,
            Err(_) => return Box::into_raw(Box::new(res)) as *const _,
        }
    };

    let fs = std::fs::File::open(file);

    match fs {
        Ok(mut fs) => match Application::load(&mut fs, &Config::from_defuns(df).unwrap()) {
            Ok(app) => {
                res.app = Some(app);
                res.status = CompilerStatus::Ok;
            }
            Err(err) => {
                res.status = CompilerStatus::ParseError;
                res.msg = error_message("File parse error", &err);
            }
        },
        Err(err) => {
            res.msg = error_message("File I/O error", &err);
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
pub unsafe extern "C" fn get_config(q: *const CompilerResult) -> usize {
    let q: &CompilerResult = unsafe { &*q };

    match &q.app {
        Some(app) => {
            let config = app.prog.config();

            let ty: usize = match config.ty {
                CompilerType::Native => 0,
                CompilerType::Amd => 1,
                CompilerType::AmdAVX => 2,
                CompilerType::AmdSSE => 3,
                CompilerType::Arm => 4,
                CompilerType::RiscV => 5,
                CompilerType::ByteCode => 6,
                CompilerType::Debug => 7,
            };

            (config.opt as usize) | (ty << 32)
        }
        None => 0,
    }
}

/// Returns the number of state variables.
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
pub unsafe extern "C" fn execute(q: *mut CompilerResult) -> bool {
    let q: &mut CompilerResult = unsafe { &mut *q };

    if let Some(app) = &mut q.app {
        app.exec();
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
        let p: &mut [f64] = unsafe { std::slice::from_raw_parts_mut(buf, h * n) };
        let mut states = Matrix::from_buf(p, h, n);
        let p: &mut [f64] = unsafe { std::slice::from_raw_parts_mut(buf, h * n) };
        let mut obs = Matrix::from_buf(p, h, n);
        app.exec_vectorized(&mut states, &mut obs);
        true
    } else {
        false
    }
}

/// Evaluates the compiled function. This is for Symbolica compatibility.
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult.
///
#[no_mangle]
pub unsafe extern "C" fn evaluate(
    q: *mut CompilerResult,
    args: *const f64,
    nargs: usize,
    outs: *mut f64,
    nouts: usize,
) -> bool {
    let q: &mut CompilerResult = unsafe { &mut *q };

    if let Some(app) = &mut q.app {
        let args: &[f64] = unsafe { std::slice::from_raw_parts(args, nargs) };
        let outs: &mut [f64] = unsafe { std::slice::from_raw_parts_mut(outs, nouts) };
        app.evaluate(args, outs);
        true
    } else {
        false
    }
}

/// Evaluates the compiled function. This is for Symbolica compatibility.
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult.
///
#[no_mangle]
pub unsafe extern "C" fn evaluate_matrix(
    q: *mut CompilerResult,
    args: *const f64,
    nargs: usize,
    outs: *mut f64,
    nouts: usize,
) -> bool {
    let q: &mut CompilerResult = unsafe { &mut *q };

    if let Some(app) = &mut q.app {
        if app.count_params == 0 {
            return false;
        }

        let args: &[f64] = unsafe { std::slice::from_raw_parts(args, nargs) };
        let outs: &mut [f64] = unsafe { std::slice::from_raw_parts_mut(outs, nouts) };
        let n = nargs / app.count_params;
        app.evaluate_matrix(args, outs, n);
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
        if let Some(f) = &mut app.compiled {
            &mut f.mem_mut()[app.first_state] as *mut f64
        } else {
            &mut app.bytecode.mem_mut()[app.first_state] as *mut f64
        }
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
        if let Some(f) = &app.compiled {
            &f.mem()[app.first_obs] as *const f64
        } else {
            &app.bytecode.mem()[app.first_obs] as *const f64
        }
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
        if let Some(f) = &app.compiled {
            &f.mem()[app.first_diff] as *const f64
        } else {
            &app.bytecode.mem()[app.first_diff] as *const f64
        }
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
pub unsafe extern "C" fn create_matrix<'a>() -> *const Matrix<'a> {
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
    states: *mut Matrix,
    obs: *mut Matrix,
) -> bool {
    let q: &mut CompilerResult = unsafe { &mut *q };
    let states: &mut Matrix = unsafe { &mut *states };
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
pub unsafe extern "C" fn finalize_defuns(_df: *mut Defuns) {
    // if !df.is_null() {
    //     let _ = unsafe { Box::from_raw(df) };
    // }
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

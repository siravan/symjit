use anyhow::anyhow;
use std::ffi::{c_char, CStr, CString};

// mod analyzer;
mod code;
mod machine;
mod matrix;
mod memory;
mod model;
mod runnable;
mod utils;

mod assembler;
mod builder;
mod generator;
mod node;
mod statement;
mod symbol;

mod amd;
mod arm;

use matrix::Matrix;
use model::{CellModel, Program};
use runnable::{CompilerType, Runnable};

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
    func: Option<Runnable>,
    status: CompilerStatus,
}

/// Compiles a model (a json string encoding the func model)
/// ty is the requested arch (amd, arm, native, or bytecode)
///
/// # Safety
///     both model and ty are pointers to null-terminated strings
///     the output is a raw pointer to a CompilerResults
///
#[no_mangle]
pub unsafe extern "C" fn compile(
    model: *const c_char,
    ty: *const c_char,
    use_simd: bool,
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

    let prog = match Program::new(&ml) {
        Ok(prog) => prog,
        Err(msg) => {
            println!("{}", msg);
            res.status = CompilerStatus::CompilationError;
            return Box::into_raw(Box::new(res)) as *const _;
        }
    };

    let func = match ty {
        "bytecode" => Runnable::new(prog, CompilerType::ByteCode, use_simd),
        "arm" => Runnable::new(prog, CompilerType::Arm, use_simd),
        "amd" => Runnable::new(prog, CompilerType::Amd, use_simd),
        "amd-avx" => Runnable::new(prog, CompilerType::AmdAVX, use_simd),
        "amd-sse" => Runnable::new(prog, CompilerType::AmdSSE, use_simd),
        "native" => Runnable::new(prog, CompilerType::Native, use_simd),
        "debug" => Runnable::new(prog, CompilerType::Debug, use_simd),
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

/// Checks the status of a CompilerResult
/// returns a null-terminated string representing the status message
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult
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

/// Returns the number of states (dependent variables)
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult
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

/// Returns the number of parameters
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult
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

/// Returns the number of observables (output)
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult
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

/// Returns the number of differential equations
/// Generally, it should be the same as the number of states
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult
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
    let q: &mut CompilerResult = unsafe { &mut *q };

    if let Some(func) = &mut q.func {
        if func.count_states != ns || func.count_params != np {
            return false;
        }

        let du: &mut [f64] = unsafe { std::slice::from_raw_parts_mut(du, ns) };
        let u: &[f64] = unsafe { std::slice::from_raw_parts(u, ns) };
        let p: &[f64] = unsafe { std::slice::from_raw_parts(p, np) };
        func.call(du, u, p, t);
        true
    } else {
        false
    }
}

/// Executes the compiled function
/// The calling routine should fill the states and parameters before
/// calling execute
/// The result populates obs or diffs (as defined in model passed to compile)
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult
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

/// Executes the compiled function n times (vectorized)
/// The calling function provides buf, which is a k x n matrix of doubles
/// k is equal to maximum(count_states, count_obs)
/// The calling funciton fills the first count_states rows of buf
/// The result is returned in the first count_obs rows of buf
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult
///     In addition, buf should points to a valid matrix of correct size
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

/// Returns a pointer to the state variables (count_states doubles)
/// The function calling execute should write the state variables in this area
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult
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

/// Returns a pointer to the parameters (count_params doubles)
/// The function calling execute should write the parameters in this area
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult
///
#[no_mangle]
pub unsafe extern "C" fn ptr_params(q: *mut CompilerResult) -> *mut f64 {
    let q: &mut CompilerResult = unsafe { &mut *q };
    if let Some(func) = &mut q.func {
        &mut func.compiled.mem_mut()[func.first_param] as *mut f64
    } else {
        std::ptr::null_mut()
    }
}

/// Returns a pointer to the observables (count_obs doubles)
/// The function calling execute reads the observables from this area
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult
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

/// Returns a pointer to the differentials (count_diffs doubles)
/// The function calling execute reads the differentials from this area
/// note: whether the output is returned as observables or differentials is
/// defined in the model
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult
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

/// Dumps the compiled binary code to a file (name)
/// This function is useful for debugging but is not necessary for normal operations
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult
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

/// Deallocates the CompilerResult pointed by q
///
/// # Safety
///     it is the responsibility of the calling function to ensure
///     that q points to a valid CompilerResult and that after
///     calling this function, q is invalid and should not
///     be used anymore
///
#[no_mangle]
pub unsafe extern "C" fn finalize(q: *mut CompilerResult) {
    if !q.is_null() {
        let _ = unsafe { Box::from_raw(q) };
    }
}

/// Returns a null-terminated string representing the version
/// Used for debugging
///
/// # Safety
///     the return value is a null-terminated string that should not
///     be freed
///
#[no_mangle]
pub unsafe extern "C" fn info() -> *const c_char {
    // let msg = c"symjit 1.3.3";
    let msg = CString::new(env!("CARGO_PKG_VERSION")).unwrap();
    msg.into_raw() as *const _
}

/// Returns a pointer to the fast function if one can be compiled
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

/************************************************/

/// Creates an empty Matrix (a 2d array)
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

/// Finalized (deallocates) a Matrix
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

/// Adds a row to a Matrix
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

/// Executes (runs) the model encoded by q
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

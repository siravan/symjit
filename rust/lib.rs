#![allow(uncommon_codepoints)]

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
mod strings;
mod symbol;
mod topology;
mod types;
mod utils;

// lockstep evaluation of batches with yield points (a prototype)
#[cfg(feature = "async")]
mod lockstep;

#[allow(non_upper_case_globals)]
mod riscv64;
// the WebAssembly backend (`ty = "wasm"`)
#[cfg(feature = "wasm")]
mod wasm;
// object files (`Application::write_obj`)
#[cfg(feature = "obj")]
mod linker;

// a typed builder for Composer kernels
#[cfg(feature = "composer")]
pub mod kernel;

#[cfg(feature = "math")]
pub mod libm;

// the C interface used by the Python package (compiling SymPy models)
#[cfg(feature = "sympy")]
pub mod sympy;

pub use applet::Applet;
pub use compiler::{Compiler, FastFunc, Translator};
pub use composer::Composer;
pub use config::Config;
pub use defuns::Defuns;
pub use expr::Expr;
pub use instruction::{BuiltinSymbol, Instruction, Slot, SymbolicaModel};
pub use matrix::Matrix;
pub use model::{CellModel, Program};
pub use num_complex::{Complex, ComplexFloat};
pub use runnable::{Application, CompilerType};
pub use serializer::MirWriter;
pub use types::{ElemType, Element};
pub use utils::{Compiled, CompiledPlaneFunc, PlaneDescriptor, Storage};

pub fn var(name: &str) -> Expr {
    Expr::var(name)
}

pub fn double(val: f64) -> Expr {
    Expr::from(val)
}

pub fn int(val: i32) -> Expr {
    Expr::from(val)
}

fn bool_to_f64(b: bool) -> f64 {
    const T: f64 = f64::from_bits(!0);
    const F: f64 = f64::from_bits(0);
    if b {
        T
    } else {
        F
    }
}

pub fn boolean(val: bool) -> Expr {
    Expr::from(bool_to_f64(val))
}

// the C functions and their types stay at the crate root, as before the `sympy` feature
#[cfg(feature = "sympy")]
pub use sympy::*;

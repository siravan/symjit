#![allow(uncommon_codepoints)]

// the intermediate representation and code generators (crate symjit-core), under the module
// names this crate's modules use (`super::mir::Mir`, ...)
#[allow(unused_imports)]
pub(crate) use symjit_core::{
    allocator, amd, arm, assembler, block, builder, code, compactor, compilation, complexify,
    composer, config, defuns, expr, generator, instruction, mir, model, node, operation, parser,
    riscv64, serializer, statement, strings, symbol, topology, translator, types, utils,
};

// the tests of symjit-core's modules that run kernels
#[cfg(test)]
mod core_tests;

// `compile` of the front ends (`Translator`, `KernelBuilder`, ...): applications
mod compile;

mod applet;
mod bytecode;
mod compiler;
mod machine;
mod matrix;
mod memory;
mod runnable;

// lockstep evaluation of batches with yield points (a prototype)
#[cfg(feature = "async")]
mod lockstep;

// a typed builder for Composer kernels
#[cfg(feature = "composer")]
pub mod kernel;

#[cfg(feature = "math")]
pub use symjit_core::libm;

// the C interface used by the Python package (compiling SymPy models)
#[cfg(feature = "sympy")]
pub mod sympy;

/// The traits that give the front ends of symjit-core their `compile` (an application):
/// `use symjit::prelude::*;`.
pub mod prelude {
    pub use crate::compile::CompileExt;
    #[cfg(feature = "composer")]
    pub use crate::kernel::KernelCompile;
}

pub use applet::Applet;
pub use compile::CompileExt;
pub use compilation::{Compilation, Model};
pub use compiler::{Compiler, FastFunc};
pub use translator::Translator;
pub use composer::Composer;
pub use config::{CompilerType, Config};
pub use defuns::Defuns;
pub use expr::Expr;
pub use instruction::{BuiltinSymbol, Instruction, Slot, SymbolicaModel};
pub use matrix::Matrix;
pub use model::{CellModel, Program};
pub use num_complex::{Complex, ComplexFloat};
pub use runnable::Application;
pub use serializer::MirWriter;
pub use types::{ElemType, Element};
pub use machine::Compiled;
pub use utils::{CompiledPlaneFunc, PlaneDescriptor, Storage};

pub use symjit_core::{boolean, double, int, var};

// the C functions and their types stay at the crate root, as before the `sympy` feature
#[cfg(feature = "sympy")]
pub use sympy::*;

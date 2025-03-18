use std::simd::f64x4;

use crate::model::Program;

pub trait Compiled<T> {
    fn exec(&mut self);
    fn mem(&self) -> &[T];
    fn mem_mut(&mut self) -> &mut [T];
    fn dump(&self, name: &str);
}

pub trait Compiler<C: Compiled<f64>> {
    fn compile(&mut self, prog: &Program) -> C;
}

pub trait CompilerSimd<C: Compiled<f64x4>> {
    fn compile(&mut self, prog: &Program) -> C;
}


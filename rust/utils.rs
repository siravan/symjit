/*
*   This file contains stubs for Vector.
*   The full Vector impl is in vector.rs and is used to
*   generate the binary output but is not needed for lib.
*/

use crate::model::Program;

pub trait Callable {
    fn call(&mut self, du: &mut [f64], u: &[f64], p: &[f64], t: f64);
    fn exec(&mut self, t: f64);
    fn dump(&self, name: &str);
}

/********************************************/

pub trait Compiled {
    fn exec(&mut self);
    fn mem(&self) -> &[f64];
    fn mem_mut(&mut self) -> &mut [f64];
    fn dump(&self, name: &str);
}

pub trait Compiler<T: Compiled> {
    fn compile(&mut self, prog: &Program) -> T;
}

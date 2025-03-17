use crate::model::Program;

pub trait Compiled {
    fn exec(&mut self);
    fn mem(&self) -> &[f64];
    fn mem_mut(&mut self) -> &mut [f64];
    fn dump(&self, name: &str);
}

pub trait Compiler<T: Compiled> {
    fn compile(&mut self, prog: &Program) -> T;
}


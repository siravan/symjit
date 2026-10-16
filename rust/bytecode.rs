//! The bytecode interpreter (`Mir::exec_instruction`) as a compiled kernel: `ty = "bytecode"`,
//! and the kernels of applications whose machine code is not run in-process.

use std::fs;
use std::io::Write;
use std::rc::Rc;

use super::machine::{Compiled, MachineCode};
use super::mir::Mir;
use super::utils::CompiledFunc;

#[derive(Clone)]
pub struct CompiledMir {
    pub mir: Rc<Mir>,
    pub mem: Vec<f64>,
    pub stack: Vec<f64>,
    pub regs: Vec<f64>,
}

impl CompiledMir {
    pub fn new(mir: Mir, mem: Vec<f64>, stack: Vec<f64>) -> CompiledMir {
        let regs = vec![0.0; 16];

        CompiledMir {
            mir: Rc::new(mir),
            mem,
            stack,
            regs,
        }
    }
}

impl Compiled<f64> for CompiledMir {
    fn exec(&mut self, params: &[f64]) {
        self.mir
            .exec_instruction(&mut self.mem, &mut self.stack, &mut self.regs, params);
    }

    fn evaluate(&mut self, args: &[f64], outs: &mut [f64]) {
        self.mir
            .exec_instruction(&mut self.mem, &mut self.stack, &mut self.regs, args);
        outs.copy_from_slice(&self.mem[0..outs.len()]);
    }

    fn evaluate_single(&mut self, args: &[f64]) -> f64 {
        self.mir
            .exec_instruction(&mut self.mem, &mut self.stack, &mut self.regs, args);
        self.mem[0]
    }

    fn mem(&self) -> &[f64] {
        &self.mem[..]
    }

    fn mem_mut(&mut self) -> &mut [f64] {
        &mut self.mem[..]
    }

    fn dump(&self, name: &str) {
        let mut fs = fs::File::create(name).unwrap();
        let _ = writeln!(fs, "#! bytecode");
        let _ = write!(fs, "{:?}", self.mir);
    }

    fn dumps(&self) -> Vec<u8> {
        let s = format!("{:?}", self.mir);
        s.into_bytes()
    }

    fn func(&self) -> CompiledFunc<f64> {
        unreachable!()
    }

    fn support_indirect(&self) -> bool {
        false
    }

    fn count_lanes(&self) -> usize {
        1
    }

    fn as_machine(&self) -> Option<&MachineCode<f64>> {
        None
    }
}

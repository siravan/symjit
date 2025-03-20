use super::code::*;
use super::model::Program;
use super::register::Word;
use super::utils::*;

#[derive(Debug)]
pub enum Fast<T> {
    Unary {
        x: u32,
        dst: u32,
        f: BinaryFunc<T>,
    },
    Binary {
        x: u32,
        y: u32,
        dst: u32,
        f: BinaryFunc<T>,
    },
    IfElse {
        x1: u32,
        x2: u32,
        cond: u32,
        dst: u32,
    },
}

#[derive(Debug)]
pub struct Interpreter {}

impl Interpreter {
    pub fn new() -> Interpreter {
        Self {}
    }
}

impl Compiler<ByteCode<f64>> for Interpreter {
    fn compile(&mut self, prog: &Program) -> ByteCode<f64> {
        let vt = VirtualTable::<f64>::from_names(&prog.ft);
        let mut code: Vec<Fast<f64>> = Vec::new();
        let mut mem = prog.frame.mem();
        let m = mem.len();
        let h = |x: &Word| -> u32 { (if x.is_temp() { m + x.0 } else { x.0 }) as u32 };

        for c in prog.code.iter() {
            match c {
                Instruction::Unary { p, x, dst, .. } => {
                    code.push(Fast::Unary {
                        f: vt[p.0],
                        x: h(x),
                        dst: h(dst),
                    });
                }
                Instruction::Binary { p, x, y, dst, .. } => {
                    code.push(Fast::Binary {
                        f: vt[p.0],
                        x: h(x),
                        y: h(y),
                        dst: h(dst),
                    });
                }
                Instruction::IfElse { x1, x2, cond, dst } => {
                    code.push(Fast::IfElse {
                        x1: h(x1),
                        x2: h(x2),
                        cond: h(cond),
                        dst: h(dst),
                    });
                }
                _ => {}
            }
        }

        mem.extend_from_slice(&vec![0.0; prog.frame.stack_size()]);

        ByteCode::new(code, mem)
    }
}

pub struct ByteCode<T> {
    code: Vec<Fast<T>>,
    _mem: Vec<T>,
}

impl<T> ByteCode<T> {
    fn new(code: Vec<Fast<T>>, _mem: Vec<T>) -> ByteCode<T> {
        ByteCode { code, _mem }
    }
}

impl Compiled<f64> for ByteCode<f64> {
    fn exec(&mut self) {
        for c in self.code.iter() {
            match c {
                Fast::Unary { f, x, dst, .. } => {
                    self._mem[*dst as usize] = f(self._mem[*x as usize], 0.0);
                }
                Fast::Binary { f, x, y, dst, .. } => {
                    self._mem[*dst as usize] = f(self._mem[*x as usize], self._mem[*y as usize]);
                }
                Fast::IfElse { x1, x2, cond, dst } => {
                    self._mem[*dst as usize] = if self._mem[*cond as usize] > 0.0 {
                        self._mem[*x1 as usize]
                    } else {
                        self._mem[*x2 as usize]
                    }
                }
            }
        }
    }

    #[inline]
    fn mem(&self) -> &[f64] {
        &self._mem[..]
    }

    #[inline]
    fn mem_mut(&mut self) -> &mut [f64] {
        &mut self._mem[..]
    }

    fn dump(&self, _name: &str) {}
}

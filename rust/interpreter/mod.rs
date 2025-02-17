use super::code::*;
use super::model::Program;
use super::register::Word;
use super::utils::*;

#[derive(Debug)]
pub enum Fast {
    Unary {
        x: u32,
        dst: u32,
        f: BinaryFunc,
    },
    Binary {
        x: u32,
        y: u32,
        dst: u32,
        f: BinaryFunc,
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

impl Compiler<ByteCode> for Interpreter {
    fn compile(&mut self, prog: &Program) -> ByteCode {
        let vt = prog.virtual_table();
        let mut code: Vec<Fast> = Vec::new();
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

        for _ in 0..prog.frame.stack_size() {
            mem.push(0.0);
        }

        ByteCode::new(code, mem)
    }
}

pub struct ByteCode {
    code: Vec<Fast>,
    _mem: Vec<f64>,
}

impl ByteCode {
    fn new(code: Vec<Fast>, _mem: Vec<f64>) -> ByteCode {
        ByteCode { code, _mem }
    }
}

impl Compiled for ByteCode {
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

    fn dump(&self, name: &str) {}
}

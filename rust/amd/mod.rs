#[macro_use]
mod macros;

use std::collections::{HashMap, HashSet};

use super::analyzer::{Analyzer, Stack};
use super::code::*;
use super::machine::MachineCode;
use super::model::Program;
use super::register::{Frame, Word};
use super::utils::*;

#[derive(Debug)]
pub struct AmdCompiler {
    machine_code: Vec<u8>,
    stack: Stack,
    allocs: HashMap<Word, u8>,
}

#[cfg(target_family = "unix")]
const COUNT_TEMP_XMM: u8 = 13; // XMM3-XMM15
#[cfg(target_family = "windows")]
const COUNT_TEMP_XMM: u8 = 3; // XMM3-XMM5 (Window's ABI)

impl AmdCompiler {
    pub fn new() -> AmdCompiler {
        Self {
            machine_code: Vec::new(),
            stack: Stack::new(),
            allocs: HashMap::new(),
        }
    }

    pub fn emit(&mut self, v: Vec<u8>) {
        self.machine_code.extend_from_slice(&v[..]);
    }

    fn op_code(&mut self, op: &str, p: Proc, ry: u8) {
        match op {
            "mov" => {}
            "plus" => self.emit(amd! {addsd xmm(0), xmm(ry)}),
            "minus" => self.emit(amd! {subsd xmm(0), xmm(ry)}),
            "times" => self.emit(amd! {mulsd xmm(0), xmm(ry)}),
            "divide" => self.emit(amd! {divsd xmm(0), xmm(ry)}),
            "gt" => self.emit(amd! {cmpnlesd xmm(0), xmm(ry)}),
            "geq" => self.emit(amd! {cmpnltsd xmm(0), xmm(ry)}),
            "lt" => self.emit(amd! {cmpltsd xmm(0), xmm(ry)}),
            "leq" => self.emit(amd! {cmplesd xmm(0), xmm(ry)}),
            "eq" => self.emit(amd! {cmpeqsd xmm(0), xmm(ry)}),
            "neq" => self.emit(amd! {cmpneqsd xmm(0), xmm(ry)}),
            "and" => self.emit(amd! {andpd xmm(0), xmm(ry)}),
            "or" => self.emit(amd! {orpd xmm(0), xmm(ry)}),
            "xor" => self.emit(amd! {xorpd xmm(0), xmm(ry)}),
            "neg" => {
                self.emit(amd! {movsd xmm(1), qword ptr [rbp+8*Frame::MINUS_ZERO.0]});
                self.emit(amd! {xorpd xmm(0), xmm(1)});
            }
            "abs" => {
                self.emit(amd! {movapd xmm(1), xmm(0)});
                self.emit(amd! {movsd xmm(0), qword ptr [rbp+8*Frame::MINUS_ZERO.0]});
                self.emit(amd! {andnpd xmm(0), xmm(1)});
            }
            "root" => {
                self.emit(amd! {sqrtsd xmm(0), xmm(0)});
            }
            "square" => {
                self.emit(amd! {mulsd xmm(0), xmm(0)});
            }
            "cube" => {
                self.emit(amd! {movapd xmm(1), xmm(0)});
                self.emit(amd! {mulsd xmm(0), xmm(0)});
                self.emit(amd! {mulsd xmm(0), xmm(1)});
            }
            "recip" => {
                self.emit(amd! {movapd xmm(1), xmm(0)});
                self.emit(amd! {movsd xmm(0), qword ptr [rbp+8*Frame::ONE.0]});
                self.emit(amd! {divsd xmm(0), xmm(1)});
            }
            "power" | "rem" => {
                if ry != 1 {
                    self.emit(amd! {movapd xmm(1), xmm(ry)});
                }
                self.emit(amd! {mov rax, qword ptr [rbx+8*p.0]});
                self.emit(amd! {call rax});
            }
            _ => {
                self.emit(amd! {mov rax, qword ptr [rbx+8*p.0]});
                self.emit(amd! {call rax});
            }
        }
    }

    // xmm(2) == true ? xmm(0) : xmm(1)
    fn ifelse(&mut self) {
        self.emit(amd! {andpd xmm(0), xmm(2)});
        self.emit(amd! {andnpd xmm(2), xmm(1)});
        self.emit(amd! {orpd xmm(0), xmm(2)});
    }

    fn load(&mut self, x: u8, r: Word, rename: bool) -> u8 {
        if let Some(s) = self.allocs.get(&r) {
            let s = *s;

            if s < COUNT_TEMP_XMM {
                if rename {
                    return s + 3;
                } else {
                    self.emit(amd! {movapd xmm(x), xmm(s+3)});
                    return x;
                }
            }
        }

        if r == Frame::ZERO {
            self.emit(amd! {xorpd xmm(x), xmm(x)});
        } else if r.is_temp() {
            let k = self.stack.pop(&r);
            #[cfg(target_family = "unix")]
            self.emit(amd! {movsd xmm(x), qword ptr [rsp+8*k]});
            #[cfg(target_family = "windows")]
            self.emit(amd! {movsd xmm(x), qword ptr [rsp+8*(k+4)]});
        } else {
            self.emit(amd! {movsd xmm(x), qword ptr [rbp+8*r.0]});
        };

        x
    }

    fn save(&mut self, x: u8, r: Word) {
        if let Some(s) = self.allocs.get(&r) {
            let s = *s;

            if s < COUNT_TEMP_XMM {
                self.emit(amd! {movapd xmm(s+3), xmm(x)});
                return;
            }
        }

        if r.is_temp() {
            let k = self.stack.push(&r);
            #[cfg(target_family = "unix")]
            self.emit(amd! {movsd qword ptr [rsp+8*k], xmm(x)});
            #[cfg(target_family = "windows")]
            self.emit(amd! {movsd qword ptr [rsp+8*(k+4)], xmm(x)});
        } else {
            self.emit(amd! {movsd qword ptr [rbp+8*r.0], xmm(x)});
        }
    }

    // *nix and windows have different ABI
    // MacOs follows the same ABI rules as linux...

    #[cfg(target_family = "unix")]
    fn prologue(&mut self, n: usize) {
        self.emit(amd! {push rbp});
        self.emit(amd! {push rbx});
        self.emit(amd! {mov rbp, rdi});
        self.emit(amd! {mov rbx, rdx});

        if n > 0 {
            self.emit(amd! {sub rsp, n});
        }
    }

    #[cfg(target_family = "windows")]
    fn prologue(&mut self, n: usize) {
        self.emit(amd! {mov qword ptr [rsp+0x08], rbp});
        self.emit(amd! {mov qword ptr [rsp+0x10], rbx});
        self.emit(amd! {mov rbp, rcx});
        self.emit(amd! {mov rbx, r8});
        self.emit(amd! {sub rsp, n+32});
    }

    #[cfg(target_family = "unix")]
    fn epilogue(&mut self, n: usize) {
        if n > 0 {
            self.emit(amd! {add rsp, n});
        }
        self.emit(amd! {pop rbx});
        self.emit(amd! {pop rbp});
        self.emit(amd! {ret});
    }

    #[cfg(target_family = "windows")]
    fn epilogue(&mut self, n: usize) {
        self.emit(amd! {add rsp, n+32});
        self.emit(amd! {mov rbx, qword ptr [rsp+0x10]});
        self.emit(amd! {mov rbp, qword ptr [rsp+0x08]});
        self.emit(amd! {ret});
    }

    fn codegen(&mut self, prog: &Program, saveable: &HashSet<Word>) {
        let mut r = Frame::ZERO;

        for c in prog.code.iter() {
            match c {
                Instruction::Unary { p, x, dst, op } => {
                    if r != *x {
                        self.load(0, *x, false);
                    };
                    self.op_code(op, *p, 0);
                    r = *dst;
                }
                Instruction::Binary { p, x, y, dst, op } => {
                    // commutative operators
                    let (x, y) = if (op == "plus" || op == "times") && *y == r {
                        (y, x)
                    } else {
                        (x, y)
                    };

                    let ry = if *y == r {
                        self.emit(amd! {movapd xmm(1), xmm(0)});
                        1
                    } else {
                        self.load(1, *y, true)
                    };

                    if *x != r {
                        self.load(0, *x, false);
                    }

                    self.op_code(op, *p, ry);
                    r = *dst;
                }
                Instruction::IfElse { x1, x2, cond, dst } => {
                    if *cond == r {
                        self.emit(amd! {movapd xmm(2), xmm(0)});
                    } else {
                        self.load(2, *cond, false);
                    }

                    if *x2 == r {
                        self.emit(amd! {movapd xmm(1), xmm(0)});
                    } else {
                        self.load(1, *x2, false);
                    }

                    if *x1 != r {
                        self.load(0, *x1, false);
                    }

                    self.ifelse();
                    r = *dst;
                }
                _ => {
                    continue;
                }
            }

            if prog.frame.should_save(&r) || saveable.contains(&r) {
                self.save(0, r);
                r = Frame::ZERO;
            }
        }
    }
}

impl Compiler<MachineCode<f64>> for AmdCompiler {
    fn compile(&mut self, prog: &Program) -> MachineCode<f64> {
        let analyzer = Analyzer::new(prog);
        let saveable = analyzer.find_saveable();

        self.allocs = analyzer.alloc_regs();

        self.codegen(prog, &saveable);
        self.machine_code.clear();

        let cap = self.stack.capacity();
        let pad = (cap + 1) & 1; // padding to make sure that rsp is
                                 // aligned at 16 (required by Windows)
        let n = 8 * (cap + pad);

        self.prologue(n);
        self.codegen(prog, &saveable);
        self.epilogue(n);

        MachineCode::new(
            "x86_64",
            self.machine_code.clone(),
            VirtualTable::<f64>::from_names(&prog.ft),
            prog.frame.mem(),
        )
    }
}

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
pub struct AmdCompilerSimd {
    machine_code: Vec<u8>,
    stack: Stack,
    allocs: HashMap<Word, u8>,
}

#[cfg(target_family = "unix")]
const COUNT_TEMP_YMM: u8 = 13; // YMM3-YMM15
#[cfg(target_family = "windows")]
const COUNT_TEMP_YMM: u8 = 3; // YMM3-YMM5 (Windows ABI)

impl AmdCompilerSimd {
    pub fn new() -> AmdCompilerSimd {
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
            "plus" => self.emit(amd! {vaddpd ymm(0), ymm(0), ymm(ry)}),
            "minus" => self.emit(amd! {vsubpd ymm(0), ymm(0), ymm(ry)}),
            "times" => self.emit(amd! {vmulpd ymm(0), ymm(0), ymm(ry)}),
            "divide" => self.emit(amd! {vdivpd ymm(0), ymm(0), ymm(ry)}),
            "gt" => self.emit(amd! {vcmpnlepd ymm(0), ymm(0), ymm(ry)}),
            "geq" => self.emit(amd! {vcmpnltpd ymm(0), ymm(0), ymm(ry)}),
            "lt" => self.emit(amd! {vcmpltpd ymm(0), ymm(0), ymm(ry)}),
            "leq" => self.emit(amd! {vcmplepd ymm(0), ymm(0), ymm(ry)}),
            "eq" => self.emit(amd! {vcmpeqpd ymm(0), ymm(0), ymm(ry)}),
            "neq" => self.emit(amd! {vcmpneqpd ymm(0), ymm(0), ymm(ry)}),
            "and" => self.emit(amd! {vandpd ymm(0), ymm(0), ymm(ry)}),
            "or" => self.emit(amd! {vorpd ymm(0), ymm(0), ymm(ry)}),
            "xor" => self.emit(amd! {vxorpd ymm(0), ymm(0), ymm(ry)}),
            "neg" => {
                self.emit(amd! {vbroadcastsd ymm(1), qword ptr [rbp+32*Frame::MINUS_ZERO.0]});
                self.emit(amd! {vxorpd ymm(0), ymm(0), ymm(1)});
            }
            "abs" => {
                self.emit(amd! {vbroadcastsd ymm(1), qword ptr [rbp+32*Frame::MINUS_ZERO.0]});
                self.emit(amd! {vandnpd ymm(0), ymm(1), ymm(0)});
            }
            "root" => {
                self.emit(amd! {vsqrtpd ymm(0), ymm(0)});
            }
            "square" => {
                self.emit(amd! {vmulpd ymm(0), ymm(0), ymm(0)});
            }
            "cube" => {
                self.emit(amd! {vmulpd ymm(1), ymm(0), ymm(0)});
                self.emit(amd! {vmulpd ymm(0), ymm(0), ymm(1)});
            }
            "recip" => {
                self.emit(amd! {vbroadcastsd ymm(1), qword ptr [rbp+32*Frame::ONE.0]});
                self.emit(amd! {vdivpd ymm(0), ymm(1), ymm(0)});
            }
            "power" | "rem" => {
                self.call_binary(p.0, ry);
            }
            _ => {
                self.call_unary(p.0);
            }
        }
    }

    fn call_unary(&mut self, fp: usize) {
        self.emit(amd! {mov r12, qword ptr [rbx+8*fp]});

        // reserves 64 bytes in the stack
        // 32 bytes for shadow store (mandatory in Windows)
        // 32 bytes to save ymm0
        self.emit(amd! {sub rsp, 2*32});
        self.emit(amd! {vmovupd [rsp+32], ymm(0)});
        self.emit(amd! {vzeroupper}); // vzeroupper is here because the routine called by r12
                                      // may have legacy SSE instructions
        for i in 0..4 {
            self.emit(amd! {vmovsd xmm(0), [rsp+32+8*i]});
            self.emit(amd! {call r12});
            self.emit(amd! {vmovsd [rsp+32+8*i], xmm(0)});
        }

        self.emit(amd! {vmovupd ymm(0), [rsp+32]});
        self.emit(amd! {add rsp, 2*32});
    }

    fn call_binary(&mut self, fp: usize, ry: u8) {
        self.emit(amd! {mov r12, qword ptr [rbx+8*fp]});

        // reserves 96 bytes in the stack
        // 32 bytes for shadow store (mandatory in Windows)
        // 32 bytes to save ymm0
        // 32 bytes to save ymm1
        self.emit(amd! {sub rsp, 3*32});
        self.emit(amd! {vmovupd [rsp+32], ymm(0)});
        self.emit(amd! {vmovupd [rsp+64], ymm(ry)});
        self.emit(amd! {vzeroupper}); // vzeroupper is here because the routine called by r12
                                      // may have legacy SSE instructions
        for i in 0..4 {
            self.emit(amd! {vmovsd xmm(0), [rsp+32+8*i]});
            self.emit(amd! {vmovsd xmm(1), [rsp+64+8*i]});
            self.emit(amd! {call r12});
            self.emit(amd! {vmovsd [rsp+32+8*i], xmm(0)});
        }

        self.emit(amd! {vmovupd ymm(0), [rsp+32]});
        self.emit(amd! {add rsp, 3*32});
    }

    // ymm(2) == true ? ymm(0) : ymm(1)
    fn ifelse(&mut self) {
        self.emit(amd! {vandpd ymm(0), ymm(2), ymm(0)});
        self.emit(amd! {vandnpd ymm(1), ymm(2), ymm(1)});
        self.emit(amd! {vorpd ymm(0), ymm(0), ymm(1)});
    }

    fn load(&mut self, x: u8, r: Word, rename: bool) -> u8 {
        if let Some(s) = self.allocs.get(&r) {
            let s = *s;

            if s < COUNT_TEMP_YMM {
                if rename {
                    return s + 3;
                } else {
                    self.emit(amd! {vmovapd ymm(x), ymm(s+3)});
                    return x;
                }
            }
        }

        if r == Frame::ZERO {
            self.emit(amd! {vxorpd ymm(x), ymm(x), ymm(x)});
        } else if r.is_temp() {
            let k = self.stack.pop(&r);
            self.emit(amd! {vmovupd ymm(x), [rsp+32*k]});
        } else {
            self.emit(amd! {vmovupd ymm(x), [rbp+32*r.0]});
        };

        x
    }

    fn save(&mut self, x: u8, r: Word) {
        if let Some(s) = self.allocs.get(&r) {
            let s = *s;

            if s < COUNT_TEMP_YMM {
                self.emit(amd! {vmovapd ymm(s+3), ymm(x)});
                return;
            }
        }

        if r.is_temp() {
            let k = self.stack.push(&r);
            self.emit(amd! {vmovupd [rsp+32*k], ymm(x)});
        } else {
            self.emit(amd! {vmovupd [rbp+32*r.0], ymm(x)});
        }
    }

    // *nix and windows have different ABI
    // MacOs follows the same ABI rules as linux...

    #[cfg(target_family = "unix")]
    fn prologue(&mut self, n: usize) {
        self.emit(amd! {push rbp});
        self.emit(amd! {push rbx});
        self.emit(amd! {push r12});
        self.emit(amd! {mov rbp, rdi});
        self.emit(amd! {mov rbx, rdx});
        if n > 0 {
            self.emit(amd! {sub rsp, n});
        }
    }

    #[cfg(target_family = "windows")]
    fn prologue(&mut self, n: usize) {
        // Windows expects rsp to be a multiple of 16
        // the return address decreases rsp by 8
        // therefore, if we have only an even number of pushes,
        // we would need to decrease rsp by an extra 8
        // here, three pushes align the stack correctly
        self.emit(amd! {push rbp});
        self.emit(amd! {push rbx});
        self.emit(amd! {push r12});
        // note that Windows ABI is different than *nix
        self.emit(amd! {mov rbp, rcx});
        self.emit(amd! {mov rbx, r8});
        if n > 0 {
            self.emit(amd! {sub rsp, n});
        }
    }

    #[cfg(target_family = "unix")]
    fn epilogue(&mut self, n: usize) {
        self.emit(amd! {vzeroupper});
        if n > 0 {
            self.emit(amd! {add rsp, n});
        }
        self.emit(amd! {pop r12});
        self.emit(amd! {pop rbx});
        self.emit(amd! {pop rbp});
        self.emit(amd! {ret});
    }

    #[cfg(target_family = "windows")]
    fn epilogue(&mut self, n: usize) {
        self.emit(amd! {vzeroupper});
        if n > 0 {
            self.emit(amd! {add rsp, n});
        }
        self.emit(amd! {pop r12});
        self.emit(amd! {pop rbx});
        self.emit(amd! {pop rbp});
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
                        self.emit(amd! {vmovapd ymm(1), ymm(0)});
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
                        self.emit(amd! {vmovapd ymm(2), ymm(0)});
                    } else {
                        self.load(2, *cond, false);
                    }

                    if *x2 == r {
                        self.emit(amd! {vmovapd ymm(1), ymm(0)});
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

impl Compiler<MachineCode<f64x4>> for AmdCompilerSimd {
    fn compile(&mut self, prog: &Program) -> MachineCode<f64x4> {
        let analyzer = Analyzer::new(prog);
        let saveable = analyzer.find_saveable();

        self.allocs = analyzer.alloc_regs();

        self.codegen(prog, &saveable);
        self.machine_code.clear();

        let cap = self.stack.capacity();
        let n = 32 * cap;

        self.prologue(n);
        self.codegen(prog, &saveable);
        self.epilogue(n);

        MachineCode::new(
            "x86_64",
            self.machine_code.clone(),
            VirtualTable::<f64>::from_names(&prog.ft),
            prog.frame.mem_simd(),
        )
    }
}

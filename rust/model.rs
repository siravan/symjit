use serde::Deserialize;
use std::error::Error;

use crate::code::*;
use crate::register::*;

// lowers Expr and its constituents into a three-address_code format
pub trait Lower {
    fn lower(&self, prog: &mut Program) -> Word;
}

// collects instructions and registers
#[derive(Debug)]
pub struct Program {
    pub code: Vec<Instruction>, // the list of instructions
    pub frame: Frame,           // memory (states, registers, constants, ...)
    pub ft: Vec<String>,        // function table (used to generate a virtual table)
}

impl Program {
    pub fn new(ml: &CellModel) -> Program {
        let mut frame = Frame::new();

        /*
            this section lays the memory format
            the order of different sections is important!

            the layout is:

            +------------------------+
            | predefined constants   |
            +------------------------+
            | independent variable   |
            +------------------------+
            | state variables        |
            +------------------------+
            | parameters             |
            +------------------------+
            | observables (output)   |
            +------------------------+
            | differentials (output) |
            +------------------------+
            | constants and temps    |
            +------------------------+
        */

        frame.alloc(WordType::Var(ml.iv.name.clone()));

        for v in &ml.states {
            frame.alloc(WordType::State(v.name.clone(), v.val));
        }

        for v in &ml.params {
            frame.alloc(WordType::Param(v.name.clone(), v.val));
        }

        for eq in &ml.obs {
            if let Some(name) = eq.lhs.var() {
                frame.alloc(WordType::Obs(name));
            } else {
                panic!("lhs var not found");
            }
        }

        for eq in &ml.odes {
            if let Some(name) = eq.lhs.diff_var() {
                frame.alloc(WordType::Diff(name));
            } else {
                panic!("lhs diff var not found");
            }
        }

        let mut prog = Program {
            code: Vec::new(),
            frame,
            ft: Vec::new(),
        };

        ml.lower(&mut prog);
        prog.code.push(Instruction::Nop);

        prog
    }

    // pushes a non-op into code
    // useful for debugging
    pub fn push(&mut self, s: Instruction) {
        self.code.push(s)
    }

    pub fn pop(&mut self) {
        let _ = self.code.pop();
    }

    fn proc(&mut self, op: &str) -> Proc {
        let p = match self.ft.iter().position(|s| s == op) {
            Some(p) => p,
            None => {
                self.ft.push(op.to_string());
                self.ft.len() - 1
            }
        };
        Proc(p)
    }

    pub fn push_eq(&mut self, dst: Word) {
        self.code.push(Instruction::Eq { dst })
    }

    // pushes an Op into code and adjusts the virtual table accordingly
    pub fn push_unary(&mut self, op: &str, x: Word, dst: Word) {
        let p = self.proc(op);

        self.code.push(Instruction::Unary {
            op: op.to_string(),
            x,
            dst,
            p,
        })
    }

    pub fn push_binary(&mut self, op_: &str, x_: Word, y_: Word, dst_: Word) {
        // optimization by fusing x + (-y) to x - y
        if op_ == "plus" && !self.code.is_empty() {
            let c = self.code.pop().unwrap();
            if let Instruction::Unary { op, x, dst, .. } = c.clone() {
                if op == "neg" && dst == y_ {
                    let p = self.proc("minus");
                    self.code.push(Instruction::Binary {
                        op: "minus".to_string(),
                        x: x_,
                        y: x,
                        dst: dst_,
                        p,
                    });
                    return;
                }
            };
            self.code.push(c);
        }

        let p = self.proc(op_);

        self.code.push(Instruction::Binary {
            op: op_.to_string(),
            x: x_,
            y: y_,
            dst: dst_,
            p,
        })
    }

    pub fn push_ifelse(&mut self, x1: Word, x2: Word, cond: Word, dst: Word) {
        self.code.push(Instruction::IfElse { x1, x2, cond, dst })
    }

    // allocates a constant register
    pub fn alloc_const(&mut self, val: f64) -> Word {
        self.frame.alloc(WordType::Const(val))
    }

    // allocates a temporary register
    pub fn alloc_temp(&mut self) -> Word {
        self.frame.alloc(WordType::Temp)
    }

    // allocates an obeservable register
    pub fn alloc_obs(&mut self, name: &str) -> Word {
        self.frame.alloc(WordType::Obs(name.to_string()))
    }

    pub fn free(&mut self, r: Word) {
        self.frame.free(r);
    }

    pub fn reg(&self, name: &str) -> Word {
        self.frame.find(name).expect("cannot find reg by name")
    }

    pub fn reg_diff(&self, name: &str) -> Word {
        self.frame.find_diff(name).expect("cannot find reg by name")
    }

    pub fn virtual_table(&self) -> Vec<fn(f64, f64) -> f64> {
        let vt: Vec<fn(f64, f64) -> f64> = self.ft.iter().map(|s| Code::from_str(s)).collect();
        vt
    }
}

// A defined (state or param) variable
#[derive(Debug, Clone, Deserialize)]
pub struct Variable {
    pub name: String,
    pub val: f64,
}

impl Lower for Variable {
    fn lower(&self, prog: &mut Program) -> Word {
        prog.reg(&self.name)
    }
}

// Expr tree
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum Expr {
    Tree { op: String, args: Vec<Expr> },
    Const { val: f64 },
    Var { name: String },
}

impl Expr {
    // extracts the differentiated variable from the lhs of a diff eq
    pub fn diff_var(&self) -> Option<String> {
        if let Expr::Tree { args, op } = self {
            if op != "Differential" {
                return None;
            }
            if let Expr::Var { name } = &args[0] {
                return Some(name.clone());
            }
        };
        None
    }

    // extracts the regular variable from the lhs of an observable eq
    pub fn var(&self) -> Option<String> {
        if let Expr::Var { name } = self {
            return Some(name.clone());
        };
        None
    }

    fn lower_unary(&self, prog: &mut Program, op: &str, args: &Vec<Expr>) -> Word {
        let x = args[0].lower(prog);
        let dst = prog.alloc_temp();
        prog.push_unary(op, x, dst);
        prog.free(x);
        dst
    }

    fn lower_binary(&self, prog: &mut Program, op: &str, args: &Vec<Expr>) -> Word {
        if op == "times" {
            return self.lower_times(prog, args);
        }

        let x = args[0].lower(prog);
        let y = args[1].lower(prog);
        let dst = prog.alloc_temp();

        prog.push_binary(op, x, y, dst);
        prog.free(y);
        prog.free(x);

        dst
    }

    fn lower_times(&self, prog: &mut Program, args: &Vec<Expr>) -> Word {
        let x = args[0].lower(prog);
        let dst = prog.alloc_temp();

        if x == Frame::MINUS_ONE {
            prog.pop();
            let y = args[1].lower(prog);
            prog.push_unary("neg", y, dst);
            prog.free(y);
        } else {
            let y = args[1].lower(prog);
            if y == Frame::MINUS_ONE {
                prog.pop();
                prog.push_unary("neg", x, dst);
            } else {
                prog.push_binary("times", x, y, dst);
            };
            prog.free(y);
        }

        prog.free(x);

        dst
    }

    fn lower_ternary(&self, prog: &mut Program, op: &str, args: &Vec<Expr>) -> Word {
        if op != "ifelse" {
            return self.lower_poly(prog, op, args);
        }

        let x1 = args[1].lower(prog);
        let x2 = args[2].lower(prog);
        let cond = args[0].lower(prog);
        let dst = prog.alloc_temp();

        prog.push_ifelse(x1, x2, cond, dst);

        prog.free(cond);
        prog.free(x2);
        prog.free(x1);

        dst
    }

    fn lower_poly(&self, prog: &mut Program, op: &str, args: &Vec<Expr>) -> Word {
        if !(op == "plus" || op == "times") {
            panic!("missing op: {}", op);
        }

        let mut x = args[0].lower(prog);
        for i in 1..args.len() {
            let y = args[i].lower(prog);
            let dst = prog.alloc_temp();
            prog.push_binary(op, x, y, dst);
            prog.free(x);
            x = dst;
        }

        x
    }
}

impl Lower for Expr {
    fn lower(&self, prog: &mut Program) -> Word {
        match self {
            Expr::Const { val } => {
                let dst = if *val == 0.0 {
                    Frame::ZERO
                } else if *val == 1.0 {
                    Frame::ONE
                } else if *val == -1.0 {
                    Frame::MINUS_ONE
                } else {
                    prog.alloc_const(*val)
                };
                prog.push(Instruction::Num { val: *val, dst });
                dst
            }
            Expr::Var { name } => {
                // Technically, this is not necessary but having Instruction::Var in the code
                // is helpful for debugging
                let dst = prog.reg(name);
                prog.push(Instruction::Var {
                    name: name.clone(),
                    reg: dst,
                });
                dst
            }
            Expr::Tree { op, args } => match args.len() {
                1 => self.lower_unary(prog, &op, &args),
                2 => self.lower_binary(prog, &op, &args),
                3 => self.lower_ternary(prog, &op, &args),
                _ => self.lower_poly(prog, &op, &args),
            },
        }
    }
}

// abstracts equation lhs ~ rhs
#[derive(Debug, Clone, Deserialize)]
pub struct Equation {
    pub lhs: Expr,
    pub rhs: Expr,
}

impl Lower for Equation {
    fn lower(&self, prog: &mut Program) -> Word {
        let dst = if let Some(var) = self.lhs.diff_var() {
            prog.reg_diff(&var)
        } else if let Some(var) = self.lhs.var() {
            prog.reg(&var)
        } else {
            panic!("undefined diff variable");
        };

        prog.push_eq(dst);

        let src = self.rhs.lower(prog);

        prog.push_unary("mov", src, dst);
        Frame::ZERO
    }
}

// loads from a JSON CellModel file
#[derive(Debug, Clone, Deserialize)]
pub struct CellModel {
    pub iv: Variable,
    pub params: Vec<Variable>,
    pub states: Vec<Variable>,
    #[allow(dead_code)]
    pub algs: Vec<Equation>,
    pub odes: Vec<Equation>,
    pub obs: Vec<Equation>,
}

impl CellModel {
    pub fn load(text: &str) -> Result<CellModel, Box<dyn Error>> {
        Ok(serde_json::from_str(text)?)
    }
}

impl Lower for CellModel {
    fn lower(&self, prog: &mut Program) -> Word {
        for eq in &self.obs {
            eq.lower(prog);
        }

        for eq in &self.odes {
            eq.lower(prog);
        }

        Frame::ZERO
    }
}
